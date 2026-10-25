# scripts/train_free_new.py
import os

os.environ["DDE_BACKEND"] = "pytorch"

import sys
from pathlib import Path
import tqdm
import deepxde as dde

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.pinn_free_new import build_pinn_free_new


class TqdmCallback(dde.callbacks.Callback):
    def __init__(self, n_steps):
        super().__init__()
        self.n_steps = n_steps
        self.pbar = None

    def on_train_begin(self):
        self.pbar = tqdm.tqdm(total=self.n_steps, desc="Обучение", unit="it")

    def on_epoch_end(self):
        self.pbar.update(1)

    def on_train_end(self):
        self.pbar.close()


def main():
    L, T = 15.0, 5.0
    x0, b, eps = -5.0, 1.0, 0.5

    print("Сборка модели (5 слоёв по 64)...")
    model = build_pinn_free_new(L, T, x0, b, eps)

    loss_weights = [1, 1, 10, 10, 10, 10]

    print("Компиляция...")
    model.compile("adam", lr=1e-3, loss_weights=loss_weights)

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Уменьшили итерации для скорости
    print("\nЭтап 1: Adam (6000 итераций с ресемплингом)...")
    resampler = dde.callbacks.PDEPointResampler(period=1000)

    model.train(
        iterations=6000,
        display_every=1000,
        batch_size=1000,
        callbacks=[TqdmCallback(6000), resampler]
    )

    print("\nЭтап 2: L-BFGS (300 итераций)...")
    dde.optimizers.config.set_LBFGS_options(maxiter=300)
    model.compile("L-BFGS", loss_weights=loss_weights)
    model.train(callbacks=[TqdmCallback(300)])

    save_path = str(checkpoint_dir / "pinn_free_new")
    model.save(save_path)
    print(f"\nМодель сохранена в {save_path}")


if __name__ == "__main__":
    main()