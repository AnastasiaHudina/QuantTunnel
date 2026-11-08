# scripts/train_well.py
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

from model.pinn_well import (
    build_model,
    compile_model,
    transmission_pinn,
    X0, B, EPS,
    U0_WELL, WELL_HALFWIDTH,
    L, T,
)
from physics.constants import h_planck, m
from physics.analytics import D_well

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


def D_packet_numeric(EPS, b, U0, a, n_pts=4000, n_sigma=4.0):
    """
    Усреднённый по спектру гауссова пакета коэффициент прохождения:

        D_packet = ∫ |a(p)|² D_well(p²/2m, U0, a) dp / ∫ |a(p)|² dp

    где |a(p)|² -- гауссово распределение по импульсу с центром
    p0 = sqrt(2m·EPS) и шириной sigma_p = ħ / (sqrt(2)·b).

    Почему это нужно: формула D_well даёт D = 1 только для
    МОНОХРОМАТИЧЕСКОЙ волны точно на резонансе. Пакет конечной ширины
    b имеет разброс по энергии, и часть спектра попадает мимо
    резонанса. D_packet -- реалистичная цель для PINN на пакете,
    в отличие от идеализированной D_well = 1.
    """
    p0 = np.sqrt(2.0 * m * EPS)
    sigma_p = h_planck / (np.sqrt(2.0) * b)
    p_arr = np.linspace(
        p0 - n_sigma * sigma_p,
        p0 + n_sigma * sigma_p,
        n_pts,
    )
    weight = np.exp(-0.5 * ((p_arr - p0) / sigma_p) ** 2)
    eps_arr = p_arr ** 2 / (2.0 * m)
    D_arr = D_well(eps_arr, U0, a)
    num = _trapz(weight * D_arr, p_arr)
    den = _trapz(weight, p_arr)
    return num / den


class TqdmCallback(dde.callbacks.Callback):
    """Прогресс-бар tqdm, обновляемый на каждой итерации."""

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
    cb = TqdmCallback(total, desc)
    t0 = time.time()

    if optimizer_type == "adam":
        model.train(
            iterations=total,
            callbacks=[cb],
            display_every=display_every,
        )
    else:
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
    print("ТЕСТ 4: РЕЗОНАНСНОЕ ПРОХОЖДЕНИЕ ЧЕРЕЗ ЯМУ")
    print("=" * 70)
    print(f"x0={X0}, b={B}, ε={EPS:.6f} (резонанс n=2), "
          f"U0={U0_WELL}, a={WELL_HALFWIDTH}, L={L}, T={T}")
    if not HAS_TQDM:
        print("(!) tqdm не установлен, прогресс-бар отключён. "
              "Установить: pip install tqdm")
    print()

    model = build_model(
        norm_penalty=True,
        norm_penalty_slices=8,
        norm_penalty_nx=1000,
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

    save_path = str(checkpoints_dir / "pinn_well")
    model.save(save_path)
    print(f"\nМодель сохранена: {save_path}-*.pt")

    # ==================================================================
    # ОЦЕНКА
    # ==================================================================
    print()
    print("=" * 70)
    print("РЕЗУЛЬТАТЫ: ТЕСТ 4")
    print("=" * 70)

    # Три ориентира:
    #   D_well(EPS)         -- точечный предел (1 на резонансе), справочно;
    #   D_packet_numeric    -- реалистичная цель для пакета ширины b;
    #   D_PINN              -- из модели.
    D_well_point = D_well(EPS, U0_WELL, WELL_HALFWIDTH)
    D_packet = D_packet_numeric(EPS, B, U0_WELL, WELL_HALFWIDTH)

    print(f"\nD_well(ε={EPS:.4f}) = {D_well_point:.6f}  (точечный предел, справочно)")
    print(f"D_packet (пакет b={B})  = {D_packet:.6f}  (реалистичная цель)")
    print()

    x_test = np.linspace(-L, L, 4000)
    times = np.arange(0.0, T + 0.5, 1.0)

    header = ("| t | норма ∫\\|Ψ\\|²dx | D_PINN | D_packet | \\|ΔD\\| | "
              "D_well (точечный) |")
    sep = "|---|---|---|---|---|---|"

    md_lines = [
        "# Тест 4: резонансное прохождение через яму\n",
        f"Параметры: x0={X0}, b={B}, ε={EPS:.6f}, U0={U0_WELL}, "
        f"a={WELL_HALFWIDTH}, L={L}, T={T}\n",
        "",
        f"Резонансная энергия ε = {EPS:.6f} (n=2, чётное состояние).",
        "",
        f"- `D_well(ε)` = {D_well_point:.6f}  "
        f"(точечный предел, справочно)",
        f"- `D_packet` = {D_packet:.6f}  "
        f"(пакет b={B}, реалистичная цель)",
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
                                   x_start=WELL_HALFWIDTH, x_end=L)
        d_err = abs(D_pinn - D_packet)

        row = (f"| {t_val:.0f} | {norm:.4f} | {D_pinn:.4f} | "
               f"{D_packet:.4f} | {d_err:.4f} | {D_well_point:.4f} |")
        print(row)
        md_lines.append(row)

    # ---------- Итог при t = T ----------
    X_T = np.column_stack([x_test, np.full_like(x_test, T)])
    y_T = model.predict(X_T)
    prob_T = y_T[:, 0] ** 2 + y_T[:, 1] ** 2
    norm_T = _trapz(prob_T, x_test)
    D_T = transmission_pinn(model, t=T, x_start=WELL_HALFWIDTH, x_end=L)

    norm_dev = abs(norm_T - 1.0)
    d_T_err = abs(D_T - D_packet)
    ok_norm = norm_dev < 1e-3
    ok_D = d_T_err < 0.10   # 10% -- см. комментарий про конечную ширину пакета
    passed = ok_norm and ok_D

    print()
    print(f"При t = T = {T}:")
    print(f"  |norm - 1|                  = {norm_dev:.4f}   "
          f"(критерий: < 1e-3)")
    print(f"  |D_PINN - D_packet|         = {d_T_err:.4f}   "
          f"(критерий: < 0.10)")
    print(f"  (справочно: |D_PINN - 1|    = {abs(D_T - 1.0):.4f})")
    print()
    print("РЕЗУЛЬТАТ: " + ("ПРОЙДЕН" if passed else "НЕ ПРОЙДЕН"))

    md_lines += [
        "",
        f"**При t = T = {T}:**",
        f"- `|norm - 1| = {norm_dev:.4f}` (критерий: < 1e-3) — "
        f"{'OK' if ok_norm else 'FAIL'}",
        f"- `|D_PINN - D_packet| = {d_T_err:.4f}` (критерий: < 0.10) — "
        f"{'OK' if ok_D else 'FAIL'}",
        f"- `|D_PINN - 1| = {abs(D_T - 1.0):.4f}` (справочно: пакет "
        f"конечной ширины не проходит 100% даже в идеале)",
        "",
        f"**РЕЗУЛЬТАТ: {'ПРОЙДЕН' if passed else 'НЕ ПРОЙДЕН'}**",
    ]

    md_path = checkpoints_dir / "well_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"\nОтчёт сохранён в {md_path}")


if __name__ == "__main__":
    main()