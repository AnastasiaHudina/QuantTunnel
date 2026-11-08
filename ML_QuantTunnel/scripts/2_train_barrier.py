# scripts/train_barrier.py
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

from model.pinn_barrier import (
    build_model,
    compile_model,
    transmission_pinn,
    X0, B, EPS,
    W_BARRIER, BARRIER_WIDTH,
    L, T,
)
from physics.analytics import D_barrier_below_W

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


class TqdmCallback(dde.callbacks.Callback):
    """
    Прогресс-бар tqdm, обновляемый на каждой итерации обучения.

    DeepXDE на каждой итерации вызывает on_epoch_end у всех callbacks --
    это и есть точка, где мы двигаем бар вперёд. on_train_begin/on_train_end
    открывают и закрывают бар вокруг вызова model.train().
    """

    def __init__(self, total, desc="train"):
        super().__init__()
        self.total = total
        self.desc = desc
        self.pbar = None

    def on_train_begin(self):
        if HAS_TQDM:
            self.pbar = tqdm(
                total=self.total,
                desc=self.desc,
                unit="it",
                leave=True,
                dynamic_ncols=True,
            )

    def on_epoch_end(self):
        if self.pbar is not None:
            self.pbar.update(1)

    def on_train_end(self):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


def train_with_bar(model, total, desc, display_every, optimizer_type):
    """
    Обёртка над model.train() с прогресс-баром и таймингом.

    Параметр optimizer_type:
        "adam"  -- вызываем model.train(iterations=total, ...)
        "lbfgs" -- model.train() без iterations; лимит задаётся заранее
                   через dde.optimizers.config.set_LBFGS_options(maxiter=...).
    """
    cb = TqdmCallback(total, desc)
    t0 = time.time()

    if optimizer_type == "adam":
        model.train(
            iterations=total,
            callbacks=[cb],
            display_every=display_every,
        )
    else:  # L-BFGS
        model.train(
            callbacks=[cb],
            display_every=display_every,
        )

    elapsed = time.time() - t0
    rate = total / elapsed if elapsed > 0 else float("inf")
    print(f"  прошло {elapsed:.1f} с ({rate:.1f} итераций/с)")


def main():
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("ТЕСТ 2: ТУННЕЛИРОВАНИЕ ЧЕРЕЗ БАРЬЕР (ε < W)")
    print("=" * 70)
    print(f"x0={X0}, b={B}, ε={EPS}, W={W_BARRIER}, d={BARRIER_WIDTH}, "
          f"L={L}, T={T}")
    if not HAS_TQDM:
        print("(!) tqdm не установлен, прогресс-бар отключён. "
              "Установить: pip install tqdm")
    print()

    model = build_model(
        norm_penalty=True,
        norm_penalty_slices=8,
        norm_penalty_nx=800,
    )

    # ---------- ФАЗА 1: Adam, норма выключена ----------
    print("=== ФАЗА 1: Adam без штрафа за норму (5000 итераций) ===")
    compile_model(model, "adam", lr=1e-3,
                  norm_penalty=True, norm_penalty_weight=0.0)
    train_with_bar(model, total=5000, desc="Фаза 1 (Adam)",
                   display_every=500, optimizer_type="adam")

    # ---------- ФАЗА 2: L-BFGS, норма выключена ----------
    print("\n=== ФАЗА 2: L-BFGS без штрафа за норму (5000 итераций) ===")
    dde.optimizers.config.set_LBFGS_options(maxiter=5000)
    compile_model(model, "L-BFGS",
                  norm_penalty=True, norm_penalty_weight=0.0)
    train_with_bar(model, total=5000, desc="Фаза 2 (L-BFGS)",
                   display_every=500, optimizer_type="lbfgs")

    # ---------- ФАЗА 3: L-BFGS, норма включена ----------
    print("\n=== ФАЗА 3: L-BFGS со штрафом за норму (500 итераций) ===")
    dde.optimizers.config.set_LBFGS_options(maxiter=500)
    compile_model(model, "L-BFGS",
                  norm_penalty=True, norm_penalty_weight=100.0)
    train_with_bar(model, total=500, desc="Фаза 3 (L-BFGS)",
                   display_every=50, optimizer_type="lbfgs")

    save_path = str(checkpoints_dir / "pinn_barrier")
    model.save(save_path)
    print(f"\nМодель сохранена: {save_path}-*.pt")

    # ==================================================================
    # ОЦЕНКА: таблица как в eval_free_ideal.py
    # ==================================================================
    print()
    print("=" * 70)
    print("РЕЗУЛЬТАТЫ: ТЕСТ 2")
    print("=" * 70)

    D_analytic = D_barrier_below_W(EPS, W_BARRIER, BARRIER_WIDTH)
    print(f"\nАналитический D = {D_analytic:.6f}\n")

    x_test = np.linspace(-L, L, 3000)
    times = np.arange(0.0, T + 0.5, 1.0)

    header = "| t | норма ∫\\|Ψ\\|²dx | D_PINN | D_analytic | \\|ΔD\\| |"
    sep = "|---|---|---|---|---|"

    md_lines = [
        "# Тест 2: туннелирование через барьер\n",
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

    # ---------- Итог при t = T ----------
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

    md_path = checkpoints_dir / "barrier_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"\nОтчёт сохранён в {md_path}")


if __name__ == "__main__":
    main()