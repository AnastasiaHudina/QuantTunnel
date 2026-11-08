# scripts/train_barrier_above.py
import os

os.environ["DDE_BACKEND"] = "pytorch"

import sys
import time
from pathlib import Path

import numpy as np
import deepxde as dde

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from model.pinn_barrier_above import (
    build_model,
    compile_model,
    transmission_pinn,
    X0, B, EPS,
    W_BARRIER, BARRIER_WIDTH,
    L, T,
)
from physics.analytics import D_barrier_above_W

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


class TqdmCallback(dde.callbacks.Callback):
    def __init__(self, total, desc="train"):
        super().__init__()
        self.total = total
        self.desc = desc
        self.pbar = None

    def on_train_begin(self):
        if HAS_TQDM:
            self.pbar = tqdm(
                total=self.total, desc=self.desc, unit="it",
                leave=True, dynamic_ncols=True,
            )

    def on_epoch_end(self):
        if self.pbar is not None:
            self.pbar.update(1)

    def on_train_end(self):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


def set_lbfgs(maxiter, maxls=None):
    """
    Обёртка над set_LBFGS_options.

    maxls -- максимум шагов line search на одну итерацию L-BFGS.
    Без этого ограничения на плоской функции line search уходит в
    разнос: каждая "итерация" делает сотни function evals, обучение
    растягивается на дни. maxls=15-20 -- разумный компромисс между
    качеством и скоростью.

    Старые версии DeepXDE могут не принимать maxls -- фолбэк на
    вариант без него.
    """
    if maxls is not None:
        try:
            dde.optimizers.config.set_LBFGS_options(
                maxiter=maxiter, maxls=maxls
            )
            return
        except TypeError:
            pass
    dde.optimizers.config.set_LBFGS_options(maxiter=maxiter)


def train_with_bar(model, total, desc, display_every, optimizer_type,
                   ckpt_cb=None):
    cb_list = [TqdmCallback(total, desc)]
    if ckpt_cb is not None:
        cb_list.append(ckpt_cb)
    t0 = time.time()

    if optimizer_type == "adam":
        model.train(
            iterations=total,
            callbacks=cb_list,
            display_every=display_every,
        )
    else:  # L-BFGS
        model.train(
            callbacks=cb_list,
            display_every=display_every,
        )

    elapsed = time.time() - t0
    rate = total / elapsed if elapsed > 0 else float("inf")
    print(f"  прошло {elapsed:.1f} с ({rate:.1f} итераций/с)")


def main():
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("ТЕСТ 3: НАДБАРЬЕРНОЕ ПРОХОЖДЕНИЕ (ε > W)")
    print("=" * 70)
    print(f"x0={X0}, b={B}, ε={EPS}, W={W_BARRIER}, d={BARRIER_WIDTH}, "
          f"L={L}, T={T}")
    if not HAS_TQDM:
        print("(!) tqdm не установлен, прогресс-бар отключён. "
              "Установить: pip install tqdm")
    print()

    # ==================================================================
    # ФАЗЫ 1-2: модель БЕЗ norm_penalty.
    #
    # Зачем отдельная модель: при norm_penalty=False в data.TimePDE
    # НЕ добавляется PointSetOperatorBC с 6400 точками нормировочной
    # сетки. Эти точки всё равно вычисляются на каждом forward, даже
    # если вес нормы 0, -- просто как часть списка потерь. Убирая их
    # из фаз 1-2, получаем ~2x ускорение.
    #
    # num_domain=2000 (вместо 4000) -- ещё ~1.5x. На точность для
    # этого теста не влияет заметно: внутри барьера всё равно ~25
    # точек коллокации, этого с избытком.
    # ==================================================================
    print("Собираем модель без norm_penalty (фазы 1-2)...")
    model_plain = build_model(
        num_domain=2000,
        num_boundary=400,
        num_initial=400,
    )

    # ---------- ФАЗА 1: Adam ----------
    print("\n=== ФАЗА 1: Adam (5000 итераций) ===")
    compile_model(model_plain, "adam", lr=1e-3, norm_penalty=False)
    train_with_bar(model_plain, total=5000, desc="Фаза 1 (Adam)",
                   display_every=500, optimizer_type="adam")

    # ---------- ФАЗА 2: L-BFGS с ограниченным line search ----------
    print("\n=== ФАЗА 2: L-BFGS (500 итераций, maxls=15) ===")
    set_lbfgs(maxiter=500, maxls=15)
    compile_model(model_plain, "L-BFGS", norm_penalty=False)
    train_with_bar(model_plain, total=500, desc="Фаза 2 (L-BFGS)",
                   display_every=50, optimizer_type="lbfgs")

    # ==================================================================
    # ФАЗА 3: модель С norm_penalty.
    #
    # Собираем заново, веса переносим из model_plain через
    # load_state_dict. Архитектура сети одна и та же (FNN [2]+[50]*5+[2]),
    # поэтому ключи state_dict совпадают.
    #
    # norm_penalty_nx=400, norm_penalty_slices=4 -- вместо 800/8.
    # На таком разрешении штраф за норму уже эффективно работает,
    # а стоит в 4 раза дешевле.
    # ==================================================================
    print("\nСобираем модель С norm_penalty (фаза 3)...")
    model = build_model(
        num_domain=2000,
        num_boundary=400,
        num_initial=400,
        norm_penalty=True,
        norm_penalty_slices=4,
        norm_penalty_nx=400,
    )
    model.net.load_state_dict(model_plain.net.state_dict())
    print("Веса перенесены из model_plain в model_norm.")

    print("\n=== ФАЗА 3: L-BFGS со штрафом за норму (150 итераций) ===")
    set_lbfgs(maxiter=150, maxls=15)
    compile_model(model, "L-BFGS",
                  norm_penalty=True, norm_penalty_weight=100.0)
    train_with_bar(model, total=150, desc="Фаза 3 (L-BFGS+norm)",
                   display_every=25, optimizer_type="lbfgs")

    save_path = str(checkpoints_dir / "pinn_barrier_above")
    model.save(save_path)
    print(f"\nМодель сохранена: {save_path}-*.pt")

    # ==================================================================
    # ОЦЕНКА
    # ==================================================================
    print()
    print("=" * 70)
    print("РЕЗУЛЬТАТЫ: ТЕСТ 3")
    print("=" * 70)

    D_analytic = D_barrier_above_W(EPS, W_BARRIER, BARRIER_WIDTH)
    print(f"\nАналитический D = {D_analytic:.6f}\n")

    x_test = np.linspace(-L, L, 3000)
    times = np.arange(0.0, T + 0.5, 1.0)

    header = "| t | норма ∫\\|Ψ\\|²dx | D_PINN | D_analytic | \\|ΔD\\| |"
    sep = "|---|---|---|---|---|"

    md_lines = [
        "# Тест 3: надбарьерное прохождение (ε > W)\n",
        f"Параметры: x0={X0}, b={B}, ε={EPS}, W={W_BARRIER}, "
        f"d={BARRIER_WIDTH}, L={L}, T={T}\n",
        f"Аналитический D = {D_analytic:.6f}\n",
        "",
        header,
        sep,
    ]
    print(header)
    print(sep)

    for t_val in times:
        X = np.column_stack([x_test, np.full_like(x_test, t_val)])
        y_pred = model.predict(X)
        prob = y_pred[:, 0] ** 2 + y_pred[:, 1] ** 2
        norm = _trapz(prob, x_test)

        D_pinn = transmission_pinn(model, t=t_val,
                                   x_start=BARRIER_WIDTH, x_end=L)
        d_err = abs(D_pinn - D_analytic)

        row = (f"| {t_val:.0f} | {norm:.4f} | {D_pinn:.4f} | "
               f"{D_analytic:.4f} | {d_err:.4f} |")
        print(row)
        md_lines.append(row)

    X_T = np.column_stack([x_test, np.full_like(x_test, T)])
    y_T = model.predict(X_T)
    prob_T = y_T[:, 0] ** 2 + y_T[:, 1] ** 2
    norm_T = _trapz(prob_T, x_test)
    D_T = transmission_pinn(model, t=T, x_start=BARRIER_WIDTH, x_end=L)

    norm_dev = abs(norm_T - 1.0)
    d_T_err = abs(D_T - D_analytic)
    ok_norm = norm_dev < 1e-3
    ok_D = d_T_err < 0.05
    passed = ok_norm and ok_D

    print()
    print(f"При t = T = {T}:")
    print(f"  |norm - 1|            = {norm_dev:.4f}   (критерий: < 1e-3)")
    print(f"  |D_pinn - D_analytic| = {d_T_err:.4f}   (критерий: < 0.05)")
    print()
    print("РЕЗУЛЬТАТ: " + ("ПРОЙДЕН" if passed else "НЕ ПРОЙДЕН"))

    md_lines += [
        "",
        f"**При t = T = {T}:**",
        f"- `|norm - 1| = {norm_dev:.4f}` (критерий: < 1e-3) — "
        f"{'OK' if ok_norm else 'FAIL'}",
        f"- `|D_pinn - D_analytic| = {d_T_err:.4f}` (критерий: < 0.05) — "
        f"{'OK' if ok_D else 'FAIL'}",
        "",
        f"**РЕЗУЛЬТАТ: {'ПРОЙДЕН' if passed else 'НЕ ПРОЙДЕН'}**",
    ]

    md_path = checkpoints_dir / "barrier_above_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"\nОтчёт сохранён в {md_path}")


if __name__ == "__main__":
    main()