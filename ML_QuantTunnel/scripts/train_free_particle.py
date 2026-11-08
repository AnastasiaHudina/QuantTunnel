"""
Обучение и проверка PINN на тесте "свободное движение" (Тест 1 из плана
валидации): модель без барьера и без ямы, V ≡ 0.

Что проверяется после обучения (как в плане валидации):

    1. Сохранение нормы:       ∫|Ψ(x,t)|² dx ≈ 1       для всех t.
    2. Движение центра пакета: <x>(t) ≈ x0 + (ħk0/m) t.
    3. Расплывание пакета:     σ(t) ≈ sqrt(b² + (ħt/(mb))²).

Все три сравниваются с результатом PINN на сетке по x при нескольких t.

Запуск
------
    DDE_BACKEND=pytorch python scripts/train_free_particle.py [--iterations N]

Дообучение с чекпоинта (вместо обучения с нуля) + доводка L-BFGS:

    DDE_BACKEND=pytorch python scripts/train_free_particle.py \
        --resume checkpoints/pinn_free-8000.pt --iterations 0 --lbfgs_maxiter 1500

Дообучение с явным штрафом за сохранение нормы поверх уже обученных весов
(веса берутся без состояния оптимизатора -- оно всё равно не актуально для
новой функции потерь):

    DDE_BACKEND=pytorch python scripts/train_free_particle.py \
        --init_weights checkpoints/pinn_free-8850.pt --iterations 0 \
        --lbfgs_maxiter 3000 --norm_penalty --norm_penalty_weight 100

Источник аналитических формул: [2], стр. 117-119, задача 4.2, формулы (4), (5).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# np.trapezoid появился в NumPy 2.0 (переименование np.trapz). Поддерживаем
# обе версии NumPy.
_trapz = getattr(np, "trapezoid", np.trapz)

import deepxde as dde

from physics.constants import h_planck, m
from physics.initial_condition import k0_from_energy
from model.pinn_free import build_model, compile_model, load_net_weights, X0, B, EPS, L, T


def header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def analytic_center(t):
    k0 = k0_from_energy(EPS)
    v = h_planck * k0 / m
    return X0 + v * t


def analytic_width(t):
    """
    Стандартное отклонение |Psi(x,t)|^2 (а НЕ "ширина" b(t) в том смысле, в
    котором b параметризует огибающую exp(-(x-x0)^2/b(t)^2) в задачнике).

    |Psi|^2 ~ exp(-(x-<x>)^2 / b(t)^2) -- это гауссиана с дисперсией
    b(t)^2 / 2, то есть std = b(t) / sqrt(2). Если сравнивать с std,
    посчитанным как момент распределения (как ниже, по предсказаниям
    модели), нужно явно делить на sqrt(2) -- иначе получится фиктивная
    ошибка ~29% даже у идеальной сети (найдено при первом прогоне: сеть
    предсказывала std=0.709 при b=1.0, что совпадает с 1/sqrt(2)=0.7071,
    а не с "ожидаемым" 1.0).
    """
    b_t = np.sqrt(B ** 2 + (h_planck * t / (m * B)) ** 2)
    return b_t / np.sqrt(2.0)


def evaluate(model, t_values, nx=4000):
    """
    На каждом t строит плотный срез по x, получает (u,v) из модели,
    численно интегрирует |Ψ|² и сравнивает норму/центр/ширину с аналитикой.
    """
    x = np.linspace(-L, L, nx)
    rows = []
    for t in t_values:
        X = np.column_stack([x, np.full_like(x, t)])
        y = model.predict(X)
        u, v = y[:, 0], y[:, 1]
        prob = u ** 2 + v ** 2

        norm = _trapz(prob, x)
        center_pinn = _trapz(x * prob, x) / norm if norm > 0 else np.nan
        var_pinn = _trapz((x - center_pinn) ** 2 * prob, x) / norm if norm > 0 else np.nan
        width_pinn = np.sqrt(max(var_pinn, 0.0))

        center_an = analytic_center(t)
        width_an = analytic_width(t)

        # Ошибка центра относительно координаты x0 бессмысленна (центр
        # проходит через x=0 примерно на середине симуляции — знаменатель
        # обнуляется без всякой физической причины). Нормируем на
        # характерный масштаб задачи -- аналитическую ширину пакета в этот
        # момент времени: ошибка "в сигмах" физически осмысленна при любом t.
        center_err_abs = abs(center_pinn - center_an)
        center_err = center_err_abs / width_an

        rows.append(dict(
            t=t, norm=norm,
            center_pinn=center_pinn, center_an=center_an,
            center_err_abs=center_err_abs, center_err=center_err,
            width_pinn=width_pinn, width_an=width_an,
            width_err=abs(width_pinn - width_an) / width_an,
        ))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5000,
                         help="Итераций Adam. 0 -- пропустить фазу Adam "
                              "(используется вместе с --resume + --lbfgs_maxiter).")
    parser.add_argument("--num_domain", type=int, default=4000)
    parser.add_argument("--num_boundary", type=int, default=400)
    parser.add_argument("--num_initial", type=int, default=400)
    parser.add_argument("--resume", type=str, default=None,
                         help="Путь к чекпоинту (.pt), с которого продолжить "
                              "обучение вместо случайной инициализации.")
    parser.add_argument("--resume_optimizer", choices=["adam", "L-BFGS"], default="adam",
                         help="Каким оптимизатором был СОХРАНЁН чекпоинт в "
                              "--resume. DeepXDE сохраняет состояние "
                              "оптимизатора вместе с весами, и model.restore() "
                              "падает с KeyError, если на момент restore "
                              "скомпилирован оптимизатор другого типа — "
                              "поэтому перед restore нужно скомпилировать "
                              "именно тот оптимизатор, с которым был сохранён "
                              "чекпоинт (adam для обычного обучения с нуля, "
                              "L-BFGS -- если чекпоинт сам получен доводкой "
                              "L-BFGS, как pinn_free-3000.pt после первого "
                              "раунда доводки).")
    parser.add_argument("--lbfgs_maxiter", type=int, default=0,
                         help="Если > 0, после Adam запускается (или "
                              "продолжается) доводка L-BFGS с этим "
                              "ограничением на число итераций (L-BFGS обычно "
                              "даёт заметно более точную сходимость, чем "
                              "чистый Adam, но дороже по времени на итерацию).")
    parser.add_argument("--init_weights", type=str, default=None,
                         help="Путь к чекпоинту (.pt), откуда взять ТОЛЬКО "
                              "веса сети (без состояния оптимизатора) -- "
                              "аналог --resume, но без ограничения на тип "
                              "оптимизатора и без переноса его состояния "
                              "(полезно, когда меняется сама функция "
                              "потерь, например, включается "
                              "--norm_penalty, и состояние старого "
                              "оптимизатора всё равно не актуально).")
    parser.add_argument("--norm_penalty", action="store_true",
                         help="Добавить мягкое ограничение на сохранение "
                              "нормы ∫|Ψ|²dx=1 на нескольких срезах t "
                              "(см. model.pinn_free.build_norm_penalty_bc).")
    parser.add_argument("--norm_penalty_weight", type=float, default=100.0)
    parser.add_argument("--norm_penalty_slices", type=int, default=8)
    parser.add_argument("--norm_penalty_nx", type=int, default=400)
    args = parser.parse_args()

    header("ОБУЧЕНИЕ PINN: СВОБОДНОЕ ДВИЖЕНИЕ (V=0)")
    print(f"Конфигурация: x0={X0}, b={B}, eps={EPS}, L={L}, T={T}")
    print(f"Точки коллокации: domain={args.num_domain}, "
          f"boundary={args.num_boundary}, initial={args.num_initial}")
    if args.norm_penalty:
        print(f"Штраф за норму: включён (weight={args.norm_penalty_weight}, "
              f"slices={args.norm_penalty_slices}, nx={args.norm_penalty_nx})")

    model = build_model(
        num_domain=args.num_domain,
        num_boundary=args.num_boundary,
        num_initial=args.num_initial,
        norm_penalty=args.norm_penalty,
        norm_penalty_slices=args.norm_penalty_slices,
        norm_penalty_nx=args.norm_penalty_nx,
    )

    # current_optimizer отслеживает, какой оптимизатор РЕАЛЬНО сейчас
    # скомпилирован в model, чтобы не перекомпилировать (и тем самым не
    # сбрасывать восстановленное состояние оптимизатора) без необходимости.
    current_optimizer = None

    def compile_for(optimizer, lr=1e-3, maxiter=None):
        nonlocal current_optimizer
        if optimizer == "L-BFGS" and maxiter is not None:
            dde.optimizers.config.set_LBFGS_options(maxiter=maxiter)
        compile_model(model, optimizer=optimizer, lr=lr,
                      norm_penalty=args.norm_penalty,
                      norm_penalty_weight=args.norm_penalty_weight)
        current_optimizer = optimizer

    if args.init_weights:
        # Веса без состояния оптимизатора: компилируем adam со свежим
        # (нулевым) состоянием и просто подставляем веса сети поверх --
        # restore-совместимость оптимизатора тут не нужна и не используется.
        compile_for("adam", lr=1e-3)
        print(f"Загружаем веса сети (без оптимизатора) из: {args.init_weights}")
        load_net_weights(model, args.init_weights)
    elif args.resume:
        # Компилируем ТОТ оптимизатор, которым чекпоинт был сохранён --
        # иначе model.restore() падает с KeyError на optimizer_state_dict.
        compile_for(args.resume_optimizer, lr=1e-3,
                    maxiter=args.lbfgs_maxiter if args.resume_optimizer == "L-BFGS" else None)
        print(f"Восстанавливаем веса из чекпоинта: {args.resume} "
              f"(оптимизатор: {args.resume_optimizer})")
        model.restore(args.resume, verbose=1)
    else:
        compile_for("adam", lr=1e-3)

    losshistory = None

    if args.iterations > 0:
        if current_optimizer != "adam":
            compile_for("adam", lr=1e-3)
        print(f"Итераций Adam: {args.iterations}")
        losshistory, _ = model.train(
            iterations=args.iterations, display_every=max(args.iterations // 10, 1)
        )
    else:
        print("Фаза Adam пропущена (--iterations 0).")

    if args.lbfgs_maxiter > 0:
        header(f"ДОВОДКА L-BFGS (maxiter={args.lbfgs_maxiter})")
        # Перекомпилируем L-BFGS, только если он ещё не скомпилирован с
        # нужным maxiter (т.е. не был только что восстановлен как раз для
        # этого) -- иначе теряем restore-нутое состояние оптимизатора зря.
        if current_optimizer != "L-BFGS" or args.iterations > 0:
            compile_for("L-BFGS", maxiter=args.lbfgs_maxiter)
        losshistory, _ = model.train(display_every=max(args.lbfgs_maxiter // 10, 1))

    header("ПРОВЕРКА: ТЕСТ 1 (СВОБОДНОЕ ДВИЖЕНИЕ)")
    t_values = np.linspace(0.0, T, 6)
    rows = evaluate(model, t_values)

    print(f"{'t':>5} | {'norm':>10} | {'<x>_PINN':>10} | {'<x>_analyt':>10} | "
          f"{'err_x[sigma]':>12} | {'sigma_PINN':>10} | {'sigma_an':>9} | {'err_sig':>7}")
    print("-" * 95)
    for r in rows:
        print(f"{r['t']:5.2f} | {r['norm']:10.4f} | {r['center_pinn']:10.3f} | "
              f"{r['center_an']:10.3f} | {r['center_err']:12.3f} | "
              f"{r['width_pinn']:10.3f} | {r['width_an']:9.3f} | "
              f"{r['width_err']*100:6.2f}%")

    norm_dev = max(abs(r["norm"] - 1.0) for r in rows)
    center_err_sigma = max(r["center_err"] for r in rows)
    width_err = max(r["width_err"] for r in rows)

    header("ИТОГ")
    print(f"max |norm - 1|            = {norm_dev:.4f}   (критерий плана: < 1e-3)")
    print(f"max ошибка <x> (в сигмах) = {center_err_sigma:.3f}   (ориентир: < 0.1 сигмы)")
    print(f"max относит. ошибка sigma  = {width_err*100:.2f}%  (критерий плана: < 2%)")

    passed = norm_dev < 1e-3 and center_err_sigma < 0.1 and width_err < 0.02
    print()
    print("РЕЗУЛЬТАТ: " + ("ПРОЙДЕН" if passed else "ПОКА НЕ ПРОЙДЕН (нужно больше итераций / тюнинг)"))

    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    save_path = model.save(str(ckpt_dir / "pinn_free"))
    print(f"\nВеса модели сохранены: {save_path}")

    return model, losshistory, rows


if __name__ == "__main__":
    main()
