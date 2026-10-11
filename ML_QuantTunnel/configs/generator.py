"""
Генератор физических конфигураций и разбиение на train/val/test.

Что такое «физическая конфигурация»
-----------------------------------

Физическая конфигурация — это один набор параметров задачи, полностью
задающий сценарий моделирования. Для барьера это:

    (x0, b, eps, W, d),

где:
    x0  — начальное положение центра волнового пакета (x0 < 0),
    b   — ширина пакета,
    eps — средняя энергия частицы,
    W   — высота барьера,
    d   — ширина барьера.

Для ямы — аналогично:

    (x0, b, eps, U0, a),

где U0 — глубина ямы, a — её полуширина.

Здесь x0 и b относятся к начальному состоянию (гауссов пакет),
а W, d (или U0, a) и eps — к потенциалу и к энергии частицы.


Зачем генерировать много конфигураций
-------------------------------------

PINN для одной фиксированной задачи (один набор параметров) даёт
одно решение. Чтобы модель «умела» работать для разных W, d, eps, b,
её обучают сразу на множестве конфигураций: на вход подаются не
только (x, t), но и сами параметры задачи. Тогда после обучения
мы получаем модель вида

    (x, t, W, d, eps, b) → (u, v),

которая работает при любых значениях параметров из обучающего
диапазона.

Именно поэтому конфигурации генерируются заранее, а не являются
частью процесса обучения: они задают сетку сценариев, на которых
модель будет тренироваться.


Три режима для барьера
----------------------

Для барьера физически важны три режима в зависимости от отношения

    r = ε / W:

    1. r < 0.9   — туннелирование (ε заметно меньше W);
    2. 0.9 ≤ r ≤ 1.1 — окрестность вершины барьера (ε ≈ W);
    3. r > 1.1   — надбарьерное прохождение (ε заметно больше W).

Нельзя генерировать ε просто равномерно: тогда почти все конфигурации
попадут либо в 1, либо в 3, а режим 2 (интересный с точки зрения
физики) останется непокрытым. Поэтому здесь используется
стратифицированная генерация: сначала выбирается режим, затем ε
генерируется внутри нужного интервала.


Разбиение train / val / test
----------------------------

Разбиение выполняется по конфигурациям, а не по отдельным точкам
(x, t). Это важно: точки коллокации генерируются динамически во время
обучения, а конфигурации фиксированы. Если бы мы перемешивали точки,
одна и та же физическая задача попала бы и в train, и в test — это
привело бы к утечке данных.

Пропорции: 70% / 15% / 15%.


Ссылки
------
[1] Лыков С.Н. Квантовая механика: учеб. пособие.
    СПб.: Изд-во Политехн. ун-та, 2008. — 236 с.
[2] Лыков С.Н., Гасумянц В.Э., Рыков С.А. Квантовая механика. Задачи
    с решениями: учеб. пособие.
    СПб.: Изд-во Политехн. ун-та, 2008. — 315 с.
"""

import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Диапазоны параметров.
# Эти значения — рабочая гипотеза. Их можно менять, но нужно следить,
# чтобы во всех диапазонах аналитические формулы оставались
# применимыми (например, eps ≠ W ровно, чтобы не делить на ноль).
# ---------------------------------------------------------------------------

BARRIER_RANGES = dict(
    W=(0.5, 2.0),
    d=(0.5, 2.0),
    b=(0.5, 1.5),
    x0=-5.0,          # фиксировано: центр пакета слева от барьера
)

WELL_RANGES = dict(
    U0=(0.5, 2.0),
    a=(0.5, 2.0),
    eps=(0.1, 2.5),
    b=(0.5, 1.5),
    x0=-5.0,
)

# Пропорции трёх режимов для барьера. В сумме должно быть 1.
BARRIER_MODE_FRACTIONS = dict(
    tunnel=0.4,       # ε < 0.9 W
    near=0.2,         # 0.9 W ≤ ε ≤ 1.1 W
    above=0.4,        # ε > 1.1 W
)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def sample_barrier_configs(n, rng, ranges=BARRIER_RANGES,
                           mode_fractions=BARRIER_MODE_FRACTIONS):
    """
    Стратифицированная генерация конфигураций барьера.

    Параметры
    ---------
    n : int
        Число конфигураций.
    rng : np.random.Generator
        Генератор случайных чисел (см. ниже пояснение).
    ranges : dict
        Диапазоны параметров W, d, b, а также фиксированное значение x0.
    mode_fractions : dict
        Доли трёх режимов (tunnel / near / above).

    Возвращает
    ----------
    configs : dict
        Словарь с массивами:
            W, d, b, x0, eps, mode
        Все массивы длины n.
    """
    n_tunnel = int(round(n * mode_fractions["tunnel"]))
    n_near = int(round(n * mode_fractions["near"]))
    n_above = n - n_tunnel - n_near     # остаток — чтобы сумма была ровно n

    # --- 1. Базовые параметры потенциала: W, d, b — равномерно ---
    W = rng.uniform(*ranges["W"], size=n)
    d = rng.uniform(*ranges["d"], size=n)
    b = rng.uniform(*ranges["b"], size=n)

    # --- 2. Энергия: стратифицированно по отношению r = eps / W ---
    eps = np.empty(n)

    # Туннелирование: r ∈ [0.05, 0.9]
    r_tunnel = rng.uniform(0.05, 0.9, size=n_tunnel)
    eps[:n_tunnel] = r_tunnel * W[:n_tunnel]

    # Окрестность вершины: r ∈ [0.9, 1.1]
    r_near = rng.uniform(0.9, 1.1, size=n_near)
    eps[n_tunnel:n_tunnel + n_near] = r_near * W[n_tunnel:n_tunnel + n_near]

    # Надбарьерное: r ∈ [1.1, 3.0]
    r_above = rng.uniform(1.1, 3.0, size=n_above)
    eps[n_tunnel + n_near:] = r_above * W[n_tunnel + n_near:]

    # --- 3. Метка режима (для проверок и разбиения по режимам) ---
    r = eps / W
    mode = np.where(r < 0.9, "tunnel",
                    np.where(r <= 1.1, "near", "above"))

    # --- 4. Начальное положение центра пакета: фиксировано ---
    x0 = np.full(n, ranges["x0"])

    return dict(W=W, d=d, b=b, eps=eps, x0=x0, mode=mode)


def sample_well_configs(n, rng, ranges=WELL_RANGES):
    """
    Генерация конфигураций ямы (без стратификации: у ямы нет
    качественно разных режимов по ε, кроме резонансов).

    Параметры
    ---------
    n : int
        Число конфигураций.
    rng : np.random.Generator
        Генератор случайных чисел.
    ranges : dict
        Диапазоны параметров U0, a, eps, b и фиксированное x0.

    Возвращает
    ----------
    configs : dict
        Словарь с массивами U0, a, eps, b, x0.
    """
    U0 = rng.uniform(*ranges["U0"], size=n)
    a = rng.uniform(*ranges["a"], size=n)
    eps = rng.uniform(*ranges["eps"], size=n)
    b = rng.uniform(*ranges["b"], size=n)
    x0 = np.full(n, ranges["x0"])

    return dict(U0=U0, a=a, eps=eps, b=b, x0=x0)


def split_configs(configs, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC,
                  seed=0):
    """
    Разбиение конфигураций на train / val / test.

    Разбиение выполняется по физическим конфигурациям (индексам),
    а не по отдельным точкам (x, t).

    Параметры
    ---------
    configs : dict
        Словарь с массивами параметров (длины n).
    train_frac, val_frac : float
        Доли train и val. Доля test = 1 - train_frac - val_frac.
    seed : int
        Seed для воспроизводимости. Разные seed — разные разбиения.

    Возвращает
    ----------
    train, val, test : dict
        Три словаря той же структуры, что configs, с подмножествами
        конфигураций.
    """
    n = len(next(iter(configs.values())))
    perm = np.random.default_rng(seed).permutation(n)

    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]

    def _take(idx):
        return {k: v[idx] for k, v in configs.items()}

    return _take(idx_train), _take(idx_val), _take(idx_test)


def describe(configs, name="configs"):
    """
    Печатает краткую сводку по конфигурациям: количество, диапазоны
    параметров, распределение по режимам (если есть поле mode).
    """
    n = len(next(iter(configs.values())))
    print(f"{name}: n = {n}")

    for key in ("W", "d", "eps", "b", "U0", "a"):
        if key in configs:
            arr = configs[key]
            print(f"  {key:4s}: min = {arr.min():.4f}, "
                  f"max = {arr.max():.4f}, mean = {arr.mean():.4f}")

    if "mode" in configs:
        modes, counts = np.unique(configs["mode"], return_counts=True)
        print(f"  modes: {dict(zip(modes.tolist(), counts.tolist()))}")


def save_configs(configs, path):
    """
    Сохраняет конфигурации в JSON-файл.

    JSON удобен тем, что его можно открыть глазами и посмотреть,
    какие именно конфигурации получились.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        key: (val.tolist() if isinstance(val, np.ndarray) else val)
        for key, val in configs.items()
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def load_configs(path):
    """
    Загружает конфигурации из JSON, возвращает dict с numpy-массивами.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    for key, val in data.items():
        if isinstance(val, list):
            # Числовые поля → float, строковые (mode) → str
            try:
                result[key] = np.asarray(val, dtype=float)
            except (ValueError, TypeError):
                result[key] = np.asarray(val)
        else:
            result[key] = np.asarray([val]) if np.isscalar(val) else val
    return result


# ---------------------------------------------------------------------------
# Простой самопроверочный блок.
# Запускается только при прямом вызове файла:
#     python configs/generator.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    # --- 1. Барьер ---
    n_barrier = 2000
    barrier = sample_barrier_configs(n_barrier, rng)
    describe(barrier, name="barrier (all)")

    train, val, test = split_configs(barrier, seed=0)
    print()
    describe(train, name="barrier train")
    describe(val, name="barrier val")
    describe(test, name="barrier test")

    # --- 2. Яма ---
    print()
    n_well = 1000
    well = sample_well_configs(n_well, rng)
    describe(well, name="well (all)")

    # --- 3. Проверка сохранения / загрузки ---
    print()
    save_configs(barrier, "data/processed/configs_barrier.json")
    loaded = load_configs("data/processed/configs_barrier.json")
    print("Файл сохранён и загружен. Проверка на совпадение:")
    for key in ("W", "d", "eps", "b"):
        same = np.allclose(barrier[key], loaded[key])
        print(f"  {key}: совпадает = {same}")