# scripts/check_norm_all.py
"""
Тест 5: сохранение нормы ∫|Ψ(x,t)|²dx = 1 во всех обученных моделях.

Агрегатор: сами модели обучались и проверялись в train_*.py, здесь
только собираем результаты по норме в один отчёт.
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

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


def _latest_checkpoint(pattern):
    files = sorted(
        glob.glob(str(PROJECT_ROOT / "checkpoints" / pattern)),
        key=lambda f: int(re.search(r"-(\d+)\.pt$", f).group(1)),
    )
    return files[-1] if files else None


def _try_add(entries, name, module_name, ckpt_pattern):
    """Пытается добавить модель в реестр. Возвращает True, если добавил."""
    try:
        mod = __import__(module_name, fromlist=["build_model"])
    except ImportError as e:
        print(f"[skip] {name}: модуль {module_name} не найден ({e})")
        return False

    ckpt = _latest_checkpoint(ckpt_pattern)
    if ckpt is None:
        print(f"[skip] {name}: нет чекпоинта {ckpt_pattern}")
        return False

    entries.append((
        name,
        mod.build_model,
        mod.compile_model,
        ckpt,
        mod.L,
        mod.T,
    ))
    return True


def _make_models():
    entries = []
    _try_add(entries, "Тест 1: свободное движение",
             "model.pinn_free", "pinn_free-*.pt")
    _try_add(entries, "Тест 2: туннелирование",
             "model.pinn_barrier", "pinn_barrier-*.pt")
    _try_add(entries, "Тест 3: надбарьерное",
             "model.pinn_barrier_above", "pinn_barrier_above-*.pt")
    _try_add(entries, "Тест 4: резонанс в яме",
             "model.pinn_well", "pinn_well-*.pt")
    return entries


def _load_weights(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.net.load_state_dict(ckpt["model_state_dict"])


def norm_at(model, t, L, n_pts=4000):
    x = np.linspace(-L, L, n_pts)
    X = np.column_stack([x, np.full_like(x, t)])
    y = model.predict(X)
    prob = y[:, 0] ** 2 + y[:, 1] ** 2
    return _trapz(prob, x)


def _build_and_compile(build_fn, compile_fn):
    """
    Собирает модель и компилирует adam. Пробуем norm_penalty=True,
    фолбэк -- без него (на случай старых сигнатур).
    """
    try:
        model = build_fn(norm_penalty=True)
        compile_fn(model, "adam", norm_penalty=True)
        return model
    except TypeError:
        model = build_fn()
        compile_fn(model, "adam")
        return model


def main():
    entries = _make_models()
    if not entries:
        raise RuntimeError(
            "Не найдено ни одного чекпоинта. Сначала запустите "
            "соответствующие train_*.py."
        )

    n_t = 11

    md_lines = [
        "# Тест 5: сохранение нормы во всех моделях\n",
        "Критерий: `max_t |∫|Ψ|²dx − 1| < 1e-3`.\n",
        f"Проверено моделей: {len(entries)}.\n",
    ]

    results = []  # (name, max_dev, passed, ts, norms)

    for name, build_fn, compile_fn, ckpt, L, T in entries:
        print(f"\n=== {name} ===")
        print(f"  чекпоинт: {Path(ckpt).name}, L={L}, T={T}")

        model = _build_and_compile(build_fn, compile_fn)
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
    md_lines += ["", "## Сводка", "", header, sep]

    all_passed = True
    for name, max_dev, passed, _, _ in results:
        row = (f"| {name} | {max_dev:.4e} | "
               f"{'OK' if passed else 'FAIL'} |")
        print(row)
        md_lines.append(row)
        if not passed:
            all_passed = False

    # ---------- Норма по времени: отдельная таблица для каждой модели ----------
    # ВАЖНО: у разных тестов разные T (5, 8, 8, 10), поэтому общая
    # таблица с одним столбцом t была бы некорректна: t-сетки у моделей
    # не совпадают. Показываем каждую модель на её собственной сетке.
    md_lines += ["", "## Норма по времени (по моделям)", ""]
    for name, _, _, ts, norms in results:
        md_lines += [f"### {name}", "", "| t | norm |", "|---|---|"]
        for t, nm in zip(ts, norms):
            md_lines.append(f"| {t:.2f} | {nm:.6f} |")
        md_lines.append("")

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