# scripts/train_free_ideal.py
import os

os.environ["DDE_BACKEND"] = "pytorch"

import sys
from pathlib import Path
import torch
import deepxde as dde

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.pinn_free_ideal import build_pinn_free_ideal


def main():
    L, T = 15.0, 5.0
    x0, b, eps = -5.0, 1.0, 0.5
    checkpoints_dir = PROJECT_ROOT / "checkpoints"

    # ==========================================
    # ФАЗА 1: Доводим форму (ширину) без штрафа за норму
    # ==========================================
    print("=== ФАЗА 1: L-BFGS без штрафа за норму (15000 шагов) ===")
    model = build_pinn_free_ideal(L, T, x0, b, eps)

    # Вес нормы = 0 (отключаем её)
    loss_weights_phase1 = [1, 1, 10, 10, 10, 10, 0]
    model.compile("adam", lr=1e-3, loss_weights=loss_weights_phase1)

    # Загружаем веса от модели 64 нейрона
    weight_files = list(checkpoints_dir.glob("pinn_free_new-*.pt"))
    weight_files.sort(key=lambda f: int(f.stem.split('-')[1]))
    best_weights = weight_files[-1]
    print(f"Загружаем веса из {best_weights.name}...")
    checkpoint = torch.load(best_weights, map_location=torch.device('cpu'))
    model.net.load_state_dict(checkpoint["model_state_dict"])

    # Ставим 15000 шагов, чтобы ширина успела дойти до идеала
    dde.optimizers.config.set_LBFGS_options(maxiter=15000)
    model.compile("L-BFGS", loss_weights=loss_weights_phase1)
    model.train(display_every=1000)

    # ==========================================
    # ФАЗА 2: Включаем штраф за норму и делаем 100 шагов
    # ==========================================
    print("\n=== ФАЗА 2: L-BFGS со штрафом за норму (100 шагов) ===")
    loss_weights_phase2 = [1, 1, 10, 10, 10, 10, 100]
    dde.optimizers.config.set_LBFGS_options(maxiter=100)
    model.compile("L-BFGS", loss_weights=loss_weights_phase2)
    model.train(display_every=20)

    # Удаляем старые чекпоинты ideal, чтобы не путаться
    for f in checkpoints_dir.glob("pinn_free_ideal-*.pt"):
        f.unlink()

    save_path = str(checkpoints_dir / "pinn_free_ideal")
    model.save(save_path)
    print(f"\nМодель сохранена в {save_path}")


if __name__ == "__main__":
    main()