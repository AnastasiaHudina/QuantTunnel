# scripts/check_norm_all.py
"""
Тест 5: сохранение нормы ∫|Ψ(x,t)|²dx = 1 во всех моделях.

Загружает чекпоинты всех обученных моделей (свободное движение,
туннелирование, надбарьерное, яма) и строит единую таблицу нормы
по времени. Тест считается пройденным для модели, если max_t |norm−1|
< 1e-3.

Это агрегатор: сами модели проверялись в своих train_*.py, здесь мы
только собираем результаты в одно место для отчёта.
"""
import os

os.environ["DDE_BACKEND"] = "pytorch"

import sys
import re
import glob
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from physics.constants import h_planck, m

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


# ---------------------------------------------------------------------------
# Реестр моделей: имя, модуль сборки, чекпоинт, L, T, x0, b, eps.
# Для каждой модели параметры совпадают с train_*.py.
# ---------------------------------------------------------------------------

def _latest_checkpoint(pattern):
    files = sorted(
        glob.glob(str(PROJECT_ROOT / "checkpoints" / pattern)),
        key=lambda f: int(re.search(r"-(\d+)\.pt$", f).group(1)),
    )
    return files[-1] if files else None


def _make_models():
    """
    Возвращает список кортежей:
        (name, build_fn, compile_fn, checkpoint_path, L, T)
    Импорты делаются внутри, чтобы не тянуть все модели, если
    какая-то ещё не обучена.
    """
    entries = []

    # --- Тест 1: свободное движение (pinn_free) ---
    try:
        from model.pinn_free import (
            build_model as b1, compile_model as c1,
            L as L1, T as T1,
        )
        ckpt = _latest_checkpoint("pinn_free-*.pt")
        if ckpt:
            entries.append(("Тест 1: свободное движение",
                            b1, c1, ckpt, L1, T1))
    except ImportError:
        pass

    # --- Тест 2: туннелирование ---
    try:
        from model.pinn_barrier import (
            build_model as b2, compile_model as c2,
            L as L2, T as T2,
        )
        ckpt = _latest_checkpoint("pinn_barrier-*.pt")
        if ckpt:
            entries.append(("Тест 2: туннелирование",
                            b2, c2, ckpt, L2, T2))
    except ImportError:
        pass

    # --- Тест 3: надбарьерное ---
    try:
        from model.pinn_barrier_above import (
            build_model as b3, compile_model as c3,
            L as L3, T as T3,
        )
        ckpt = _latest_checkpoint("pinn_barrier_above-*.pt")
        if ckpt:
            entries.append(("Тест 3: надбарьерное",
                            b3, c3, ckpt, L3, T3))
    except ImportError:
        pass

    # --- Тест 4: яма ---
    try:
        from model.pinn_well import (
            build_model as b4, compile_model as c4,
            L as L4, T as T4,
        )
        ckpt = _latest_checkpoint("pinn_well-*.pt")
        if ckpt:
            entries.append(("Тест 4: резонанс в яме",
                            b4, c4, ckpt, L4, T4))
    except ImportError:
        pass

    return entries


def _load_weights(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.net.load_state_dict(ckpt["model_state_dict"])


def norm_at(model, t, L, n_pts=4000):
    """Возвращает ∫|Ψ(x,t)|²dx на [-L, L]."""
    x = np.linspace(-L, L, n_pts)
    X = np.column_stack([x, np.full_like(x, t)])
    y = model.predict(X)
    prob = y[:, 0] ** 2 + y[:, 1] ** 2
    return _trapz(prob, x)


def main():
    entries = _make_models()
    if not entries:
        raise RuntimeError(
            "Не найдено ни одного чекпоинта. Сначала запустите "
            "train_free_particle.py / train_barrier.py / "
            "train_barrier_above.py / train_well.py"
        )

    # Срезов по t берём 11 -- достаточно, чтобы увидеть тренд.
    n_t = 11

    md_lines = [
        "# Тест 5: сохранение нормы во всех моделях\n",
        "Критерий: `max_t |∫|Ψ|²dx − 1| < 1e-3`.\n",
    ]

    results = []  # (name, max_dev, passed, ts, norms)

    for name, build_fn, compile_fn, ckpt, L, T in entries:
        print(f"\n=== {name} ===")
        print(f"  чекпоинт: {Path(ckpt).name}")

        # Собираем модель и грузим веса. Компилируем с norm_penalty=True,
        # чтобы список потерь совпал с тем, с каким модель обучалась
        # (у pinn_free.py по умолчанию norm_penalty=False -- попробуем
        # оба варианта).
        for norm_flag in (True, False):
            try:
                model = build_fn(norm_penalty=norm_flag)
                compile_fn(model, "adam", norm_penalty=norm_flag)
                break
            except TypeError:
                # модель не принимает norm_penalty (pinn_free.py) --
                # фолбэк на вариант по умолчанию
                model = build_fn()
                compile_fn(model, "adam")
                break

        _load_weights(model, ckpt)

        ts = np.linspace(0.0, T, n_t)
        norms = np.array([norm_at(model, t, L) for t in ts])
        dev = np.abs(norms - 1.0)
        max_dev = float(dev.max())
        passed = max_dev < 1e-3

        for t, nm in zip(ts, norms):
            print(f"  t={t:5.2f}   norm={nm:.6f}")

        print(f"  max|norm−1| = {max_dev:.4e}  "
              f"→ {'OK' if passed else 'FAIL'}")

        results.append((name, max_dev, passed, ts, norms))

    # ---------- Сводная таблица ----------
    print("\n" + "=" * 70)
    print("СВОДКА: ТЕСТ 5")
    print("=" * 70)

    header = "| Модель | max\\|norm−1\\| | Критерий < 1e-3 |"
    sep = "|---|---|---|"
    print(header)
    print(sep)
    md_lines += ["", header, sep]

    all_passed = True
    for name, max_dev, passed, _, _ in results:
        row = (f"| {name} | {max_dev:.4e} | "
               f"{'OK' if passed else 'FAIL'} |")
        print(row)
        md_lines.append(row)
        if not passed:
            all_passed = False

    # ---------- Таблица norm vs t ----------
    md_lines += ["", "## Норма по времени", ""]
    t_header = "| t | " + " | ".join(n for n, *_ in results) + " |"
    t_sep = "|---|" + "---|" * len(results)
    md_lines += [t_header, t_sep]

    ts_all = results[0][3]  # предполагаем, что у всех моделей одна сетка t
    # Если T разные -- выровняем по индексу
    for i, t in enumerate(ts_all):
        vals = " | ".join(
            f"{norms[i]:.4f}" for _, _, _, _, norms in results
        )
        md_lines.append(f"| {t:.2f} | {vals} |")

    print()
    print("РЕЗУЛЬТАТ: " +
          ("ПРОЙДЕН" if all_passed else "НЕ ПРОЙДЕН"))
    md_lines += [
        "",
        f"**РЕЗУЛЬТАТ: {'ПРОЙДЕН' if all_passed else 'НЕ ПРОЙДЕН'}**",
    ]

    md_path = PROJECT_ROOT / "checkpoints" / "test5_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"\nОтчёт сохранён в {md_path}")


if __name__ == "__main__":
    main()