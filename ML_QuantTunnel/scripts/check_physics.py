"""
Единый скрипт проверки физического модуля проекта.

Назначение
----------

Этот скрипт последовательно прогоняет все физические проверки,
которые должны быть выполнены до начала обучения PINN. Если хотя бы
одна проверка не проходит — обучение запускать нельзя.

Проверки:

    1. Нормировка начального гауссова пакета:  ∫ |ψ(x,0)|² dx = 1.
    2. Форма потенциалов: барьер и яма имеют правильные значения
       в правильных областях.
    3. Аналитические формулы D для барьера и ямы:
         - туннелирование:  0 < D < 1;
         - монотонность D по ε в режиме ε < W;
         - надбарьерное:    0 < D < 1;
         - резонанс в яме:  D ≈ 1.
    4. Генератор физических конфигураций:
         - разбиение train / val / test;
         - покрытие трёх режимов для барьера;
         - сохранение и загрузка JSON.
    5. Граничные условия:
         - функция ψ(±L) = 0 корректно определена;
         - генератор точек на границах работает;
         - для заданных L и T волновой пакет не достигает границ
           (иначе BC ψ(±L) = 0 физически неоправданы).

Запуск
------

    python scripts/check_physics.py

Все проверки выполняются в одной сессии, результат печатается
в консоль. Если всё в порядке — в конце печатается "ВСЕ ПРОВЕРКИ
ПРОЙДЕНЫ".

Ссылки
------
[1] Лыков С.Н. Квантовая механика: учеб. пособие.
    СПб.: Изд-во Политехн. ун-та, 2008. — 236 с.
[2] Лыков С.Н., Гасумянц В.Э., Рыков С.А. Квантовая механика. Задачи
    с решениями: учеб. пособие.
    СПб.: Изд-во Политехн. ун-та, 2008. — 315 с.
"""

# ---------------------------------------------------------------------------
# Импорт: чтобы скрипт видел пакеты physics/ и configs/, добавим корень
# проекта в sys.path. Это позволяет запускать скрипт как
#     python scripts/check_physics.py
# из любой рабочей директории.
# ---------------------------------------------------------------------------

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Импорт функций из модулей проекта.
# ---------------------------------------------------------------------------

import numpy as np

from physics.constants import h_planck, m
from physics.potentials import V_barrier, V_well
from physics.initial_condition import (
    psi_initial,
    k0_from_energy,
    norm_constant,
    check_norm,
)
from physics.analytics import (
    D_barrier_below_W,
    D_barrier_above_W,
    D_well,
    resonance_energies,
)
from physics.boundary_conditions import (
    boundary_values,
    generate_boundary_points,
    check_L_sufficient,
)
from configs.generator import (
    sample_barrier_configs,
    sample_well_configs,
    split_configs,
    describe,
    save_configs,
    load_configs,
    BARRIER_RANGES,
    WELL_RANGES,
)


# ===========================================================================
# Параметры вычислительной области, используемые в проверках BC.
#
# L = 35 выбрано так, чтобы все конфигурации из диапазонов generator.py
# (включая быстрые пакеты при большом ε и малом b) укладывались в область
# при T = 5. Проверка check_L_sufficient подтверждает, что пакет за время
# моделирования не достигает границ x = ±L.
# ===========================================================================

L_DEFAULT = 35.0     # полуширина пространственной области
T_DEFAULT = 5.0      # полное время моделирования


# ===========================================================================
# Вспомогательные функции для красивого вывода
# ===========================================================================

def header(title):
    """Печатает заголовок блока проверок."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def check(condition, message):
    """
    Печатает результат одной проверки.

    Если condition=True, печатает "OK", иначе "FAIL" и ставит
    глобальный флаг неудачи. Все неудачи суммируются в конце.
    """
    global _failed
    if condition:
        print(f"  [OK]   {message}")
    else:
        print(f"  [FAIL] {message}")
        _failed = True


_failed = False


# ===========================================================================
# Проверка 1. Нормировка начального пакета
# ===========================================================================

def test_initial_condition():
    header("Проверка 1. Нормировка начального гауссова пакета")

    # Достаточно широкая и плотная сетка.
    x = np.linspace(-30.0, 30.0, 200_000)

    cases = [
        dict(x0=-5.0, b=1.0, eps=0.5),
        dict(x0=-8.0, b=1.5, eps=1.0),
        dict(x0=-6.0, b=0.7, eps=2.0),
    ]

    for case in cases:
        norm = check_norm(x, **case)
        msg = (f"∫|ψ|² dx = {norm:.10f} "
               f"(x0={case['x0']}, b={case['b']}, eps={case['eps']})")
        check(abs(norm - 1.0) < 1e-6, msg)

    # Проверка связи k0 с энергией.
    for eps in (0.25, 1.0, 4.0):
        k0 = k0_from_energy(eps)
        expected = np.sqrt(2.0 * m * eps) / h_planck
        check(abs(k0 - expected) < 1e-12,
              f"k0(ε={eps}) = {k0:.6f} (ожидается {expected:.6f})")


# ===========================================================================
# Проверка 2. Форма потенциалов
# ===========================================================================

def test_potentials():
    header("Проверка 2. Форма потенциалов")

    # --- Барьер ---
    x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    V = V_barrier(x, W=1.0, d=1.0)
    expected = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    check(np.allclose(V, expected),
          f"V_barrier на контрольных точках: {V} (ожидается {expected})")

    # --- Яма ---
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    V = V_well(x, U0=1.0, a=1.0)
    expected = np.array([0.0, -1.0, -1.0, -1.0, 0.0])
    check(np.allclose(V, expected),
          f"V_well на контрольных точках: {V} (ожидается {expected})")


# ===========================================================================
# Проверка 3. Аналитические формулы для D
# ===========================================================================

def test_analytics_barrier():
    header("Проверка 3a. Аналитика: барьер")

    # --- Туннелирование ---
    cases = [
        (0.2, 1.0, 1.0),
        (0.5, 1.0, 1.0),
        (0.8, 1.0, 1.5),
    ]
    for eps, W, d in cases:
        D = D_barrier_below_W(eps, W, d)
        check(0.0 < D < 1.0,
              f"туннелирование: ε={eps}, W={W}, d={d} → D={D:.6f}")

    # --- Монотонность D(ε) при ε < W ---
    # Физически правильное свойство: при фиксированном W коэффициент
    # прохождения растёт с ростом энергии. Ранее в тесте было неверное
    # ожидание D → 1 при ε → W⁻: правильный предел равен
    #     1 / (1 + k_W^2 d^2 / 4),
    # где k_W = √(2mW)/ℏ. При W = 1, d = 1 это ≈ 0.667.
    D_low  = D_barrier_below_W(0.3, 1.0, 1.0)
    D_mid  = D_barrier_below_W(0.6, 1.0, 1.0)
    D_high = D_barrier_below_W(0.9, 1.0, 1.0)
    check(D_low < D_mid < D_high,
          f"монотонность при ε < W: D(0.3)={D_low:.4f}, "
          f"D(0.6)={D_mid:.4f}, D(0.9)={D_high:.4f}")

    # --- Надбарьерное прохождение ---
    cases = [
        (1.5, 1.0, 1.0),
        (2.0, 1.0, 1.5),
        (3.0, 1.0, 0.5),
    ]
    for eps, W, d in cases:
        D = D_barrier_above_W(eps, W, d)
        check(0.0 < D < 1.0,
              f"надбарьерное: ε={eps}, W={W}, d={d} → D={D:.6f}")


def test_analytics_well():
    header("Проверка 3b. Аналитика: яма")

    # --- Обычный случай (вдали от резонанса) ---
    D = D_well(eps=0.7, U0=1.0, a=1.0)
    check(0.0 < D < 1.0, f"яма (обычный случай): D={D:.6f}")

    # --- Резонанс: D должно быть ровно 1 ---
    U0, a = 1.0, 1.0
    resonances = resonance_energies(U0, a, n_max=3)
    check(len(resonances) > 0,
          f"найдены резонансные энергии: {resonances}")

    for eps_res in resonances:
        D = D_well(eps_res, U0, a)
        check(abs(D - 1.0) < 1e-10,
              f"резонанс ε={eps_res:.6f} → D={D:.12f}")


# ===========================================================================
# Проверка 4. Генератор конфигураций
# ===========================================================================

def test_configs():
    header("Проверка 4. Генератор физических конфигураций")

    rng = np.random.default_rng(seed=42)

    # --- Барьер ---
    n = 2000
    barrier = sample_barrier_configs(n, rng)
    check(len(barrier["W"]) == n,
          f"барьер: сгенерировано {len(barrier['W'])} конфигураций")

    # Распределение режимов.
    modes, counts = np.unique(barrier["mode"], return_counts=True)
    mode_dict = dict(zip(modes.tolist(), counts.tolist()))
    print(f"  распределение режимов: {mode_dict}")

    check(mode_dict.get("tunnel", 0) > 0, "есть режим 'tunnel'")
    check(mode_dict.get("near", 0) > 0, "есть режим 'near'")
    check(mode_dict.get("above", 0) > 0, "есть режим 'above'")

    # --- Разбиение train / val / test ---
    train, val, test = split_configs(barrier, seed=0)
    n_train = len(train["W"])
    n_val = len(val["W"])
    n_test = len(test["W"])

    print(f"  train / val / test: {n_train} / {n_val} / {n_test}")
    check(n_train + n_val + n_test == n,
          "сумма частей равна общему числу конфигураций")
    check(abs(n_train / n - 0.70) < 0.01, "доля train ≈ 70%")
    check(abs(n_val / n - 0.15) < 0.01, "доля val ≈ 15%")
    check(abs(n_test / n - 0.15) < 0.01, "доля test ≈ 15%")

    # Проверим, что все три части содержат все режимы.
    for name, subset in [("train", train), ("val", val), ("test", test)]:
        modes_sub = set(np.unique(subset["mode"]).tolist())
        for needed in ("tunnel", "near", "above"):
            check(needed in modes_sub,
                  f"  {name} содержит режим '{needed}'")

    # --- Сохранение / загрузка ---
    out_path = PROJECT_ROOT / "data" / "processed" / "configs_barrier.json"
    save_configs(barrier, out_path)
    check(out_path.exists(), f"файл сохранён: {out_path.name}")

    loaded = load_configs(out_path)
    for key in ("W", "d", "eps", "b"):
        same = np.allclose(barrier[key], loaded[key])
        check(same, f"  поле '{key}' сохранилось без потерь")

    # --- Яма ---
    well = sample_well_configs(500, rng)
    check(len(well["U0"]) == 500, "яма: сгенерировано 500 конфигураций")
    for key in ("U0", "a", "eps", "b"):
        arr = well[key]
        check(arr.min() > 0.0,
              f"  параметр '{key}' положителен: min = {arr.min():.4f}")

    # --- Печать сводки ---
    print()
    describe(barrier, name="barrier (all)")
    print()
    describe(well, name="well (all)")

    return barrier, well


# ===========================================================================
# Проверка 5. Граничные условия
# ===========================================================================

def test_boundary_conditions():
    header("Проверка 5. Граничные условия")

    # --- 5.1. Значения BC ---
    x = np.array([-L_DEFAULT, 0.0, L_DEFAULT])
    u_bc, v_bc = boundary_values(x)
    check(np.allclose(u_bc, 0.0),
          f"u(-L, 0, L) = {u_bc} (ожидается [0, 0, 0])")
    check(np.allclose(v_bc, 0.0),
          f"v(-L, 0, L) = {v_bc} (ожидается [0, 0, 0])")

    # --- 5.2. Генерация точек на границах ---
    points, sides = generate_boundary_points(
        L_DEFAULT, T_DEFAULT, n_left=5, n_right=5,
    )
    check(points.shape == (10, 2),
          f"форма массива точек: {points.shape} (ожидается (10, 2))")
    check(np.all(points[:5, 0] == -L_DEFAULT),
          "первые 5 точек лежат на левой границе x = -L")
    check(np.all(points[5:, 0] == +L_DEFAULT),
          "последние 5 точек лежат на правой границе x = +L")
    check(np.all(sides[:5] == -1) and np.all(sides[5:] == 1),
          "метки сторон корректны (-1 для левой, +1 для правой)")


def test_boundary_L_sufficient(barrier_configs, well_configs):
    header("Проверка 5b. Достаточность L для всех конфигураций")

    print(f"  Используются L = {L_DEFAULT}, T = {T_DEFAULT}")

    # --- Барьер ---
    n_bad_barrier = 0
    worst_margin_barrier = np.inf
    for i in range(len(barrier_configs["W"])):
        ok, info = check_L_sufficient(
            L=L_DEFAULT,
            x0=barrier_configs["x0"][i],
            b=barrier_configs["b"][i],
            eps=barrier_configs["eps"][i],
            T=T_DEFAULT,
        )
        worst_margin_barrier = min(worst_margin_barrier, info["margin"])
        if not ok:
            n_bad_barrier += 1

    check(n_bad_barrier == 0,
          f"барьер: конфигураций с недостаточным L = {n_bad_barrier} "
          f"(худший запас margin = {worst_margin_barrier:.2f})")

    # --- Яма ---
    n_bad_well = 0
    worst_margin_well = np.inf
    for i in range(len(well_configs["U0"])):
        ok, info = check_L_sufficient(
            L=L_DEFAULT,
            x0=well_configs["x0"][i],
            b=well_configs["b"][i],
            eps=well_configs["eps"][i],
            T=T_DEFAULT,
        )
        worst_margin_well = min(worst_margin_well, info["margin"])
        if not ok:
            n_bad_well += 1

    check(n_bad_well == 0,
          f"яма: конфигураций с недостаточным L = {n_bad_well} "
          f"(худший запас margin = {worst_margin_well:.2f})")


# ===========================================================================
# Основная точка входа
# ===========================================================================

def main():
    print("ПРОВЕРКА ФИЗИЧЕСКОГО МОДУЛЯ ПРОЕКТА")
    print("Константы: ℏ = 1, m = 1 (атомные единицы)")

    test_initial_condition()
    test_potentials()
    test_analytics_barrier()
    test_analytics_well()

    # test_configs возвращает конфигурации, чтобы использовать их
    # в проверках BC.
    barrier, well = test_configs()

    test_boundary_conditions()
    test_boundary_L_sufficient(barrier, well)

    header("ИТОГ")
    if _failed:
        print("ЕСТЬ ЗАМЕЧАНИЯ. Смотрите строки [FAIL] выше.")
        sys.exit(1)
    else:
        print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ.")


if __name__ == "__main__":
    main()