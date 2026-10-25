# models/eval_free_ideal.py
import os

os.environ["DDE_BACKEND"] = "pytorch"

import sys
import glob
import re
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.pinn_free_ideal import build_pinn_free_ideal
from physics.constants import h_planck, m
from physics.initial_condition import k0_from_energy


def main():
    L, T = 15.0, 5.0
    x0, b, eps = -5.0, 1.0, 0.5

    print("Сборка модели...")
    model = build_pinn_free_ideal(L, T, x0, b, eps)

    # Веса должны совпадать (7 штук)
    loss_weights = [1, 1, 10, 10, 10, 10, 1]
    model.compile("adam", lr=1e-3, loss_weights=loss_weights)

    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    files = glob.glob(str(checkpoints_dir / "pinn_free_ideal-*.pt"))
    if not files:
        raise FileNotFoundError("Нет чекпоинтов. Сначала запустите train_free_ideal.py")

    files.sort(key=lambda f: int(re.search(r'-(\d+)\.pt$', f).group(1)))
    checkpoint_path = files[-1]

    print(f"Загружаем чекпоинт: {Path(checkpoint_path).name}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.net.load_state_dict(checkpoint["model_state_dict"])
    print("Модель загружена!\n")

    x_test = np.linspace(-L, L, 500)
    k0 = k0_from_energy(eps)
    v_group = h_planck * k0 / m
    times = np.arange(0.0, 5.1, 1.0)

    md_lines = [
        "## Результаты проверок (Идеальная модель)\n",
        "Сравнение обученной PINN-модели с аналитическим решением:\n"
    ]
    table_header = "| t | норма ∫\\|Ψ\\|²dx | ⟨x⟩ PINN | ⟨x⟩ аналитика | ошибка ⟨x⟩ (в σ) | σ PINN | σ аналитика | ошибка σ |"
    table_sep = "|---|---|---|---|---|---|---|---|"
    md_lines.extend([table_header, table_sep])

    print(table_header)
    print(table_sep)

    plt.figure(figsize=(15, 10))
    plot_times = [0.0, 1.0, 3.0, 5.0]

    for t_val in times:
        X = np.column_stack((x_test, np.full_like(x_test, t_val)))
        y_pred = model.predict(X)
        prob_density = y_pred[:, 0] ** 2 + y_pred[:, 1] ** 2

        norm = np.trapezoid(prob_density, x_test)
        x_center_pinn = np.trapezoid(x_test * prob_density, x_test) / norm
        x_sq_mean = np.trapezoid(x_test ** 2 * prob_density, x_test) / norm
        sigma_pinn = np.sqrt(max(0, x_sq_mean - x_center_pinn ** 2))

        x_center_ana = x0 + v_group * t_val
        b_t = np.sqrt(b ** 2 + (h_planck * t_val / (m * b)) ** 2)
        sigma_ana = b_t / np.sqrt(2)

        err_x = abs(x_center_pinn - x_center_ana) / sigma_ana
        err_sigma = abs(sigma_pinn - sigma_ana) / sigma_ana

        row = f"| {t_val:.0f} | {norm:.3f} | {x_center_pinn:.3f} | {x_center_ana:.3f} | {err_x:.2f} | {sigma_pinn:.3f} | {sigma_ana:.3f} | {err_sigma * 100:.2f}% |"
        print(row)
        md_lines.append(row)

        if t_val in plot_times:
            ax = plt.subplot(2, 2, plot_times.index(t_val) + 1)
            ax.plot(x_test, prob_density, "b-", label="PINN")
            env = (1 / np.sqrt(np.pi * b_t ** 2)) * np.exp(-((x_test - x_center_ana) ** 2) / b_t ** 2)
            ax.plot(x_test, env, "r--", label="Analytic")
            ax.set_title(f"t = {t_val}")
            ax.set_xlim(-10, 10)
            ax.legend()

    plot_path = checkpoints_dir / "test1_results_ideal.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    md_path = checkpoints_dir / "eval_report_ideal.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nТаблица сохранена в {md_path}")
    print(f"Графики сохранены в {plot_path}")


if __name__ == "__main__":
    main()