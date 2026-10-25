"""
PINN для теста "свободное движение" (V = 0) — Тест 1 из плана валидации.

Зачем именно этот случай первый
--------------------------------

Это самая простая версия уравнения Шрёдингера: без барьера и без ямы.
У неё есть точное аналитическое решение (эволюция гауссова пакета), с
которым можно сравнить результат сети. Если PINN не может воспроизвести
даже свободное движение, нет смысла переходить к барьеру или яме — сначала
нужно убедиться, что базовая связка (геометрия, PDE, IC, BC, архитектура
сети) вообще работает.

Используются готовые физические функции из physics/: потенциала здесь
просто нет (V ≡ 0), поэтому PDE для (u, v) упрощается — слагаемые с V(x)
пропадают, остаётся только линейная система с лапласианом.

PDE (из README корня проекта, при V = 0):

    ∂u/∂t = -(ħ/2m) ∂²v/∂x²
    ∂v/∂t =  (ħ/2m) ∂²u/∂x²

Источник: [1], стр. 109, формулы (2.162-2.164), при V(x) ≡ 0.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import deepxde as dde

from physics.constants import h_planck, m
from physics.initial_condition import u_initial, v_initial
from physics.boundary_conditions import boundary_values, check_L_sufficient


# ---------------------------------------------------------------------------
# Представительная конфигурация для этого теста.
#
# Взята из physics/initial_condition.py (один из случаев, на которых уже
# проверена нормировка): x0=-5, b=1.0, eps=0.5.
#
# L, T подобраны ПОД ЭТУ КОНКРЕТНУЮ конфигурацию через check_L_sufficient,
# а не взяты как L=35 "с запасом на все случаи" (как в
# scripts/check_physics.py). Свободному пакету такой большой запас не нужен,
# а меньшая область обучается заметно быстрее — больше нет барьера/ямы,
# из-за которых нужно было закладываться на самые быстрые пакеты сразу по
# всему диапазону eps.
# ---------------------------------------------------------------------------

X0 = -5.0
B = 1.0
EPS = 0.5

L = 15.0
T = 5.0


def build_norm_penalty_bc(n_slices, nx_slice, offset):
    """
    Мягкое ограничение на сохранение нормы: ∫|Ψ(x,t)|²dx = 1 на нескольких
    срезах t ∈ [0,T].

    Зачем это нужно: после обучения (Adam + несколько раундов L-BFGS) PDE-
    невязка уже минимизирована почти до предела -- L-BFGS сам останавливался
    по своему критерию сходимости (маленький градиент), а норма всё ещё
    отклонялась от 1 на ~0.4% к t=5 вместо требуемых <0.1%. Это значит, что
    дальше чистое обучение той же поточечной PDE+IC+BC-невязки не поможет:
    она уже мала везде локально, но её интеграл по x всё ещё заметно не
    нулевой. Явный штраф напрямую требует от сети сохранять норму, вместо
    того чтобы рассчитывать на это как на побочный эффект PDE-лосса.

    Реализовано через dde.icbc.PointSetOperatorBC: на n_slices срезах по t
    берём равномерную сетку из nx_slice точек по x, интегрируем |u²+v²| по
    трапециям (вручную, через тензорные операции PyTorch, чтобы сохранить
    градиент) и штрафуем квадратичное отклонение интеграла от 1.

    offset: позиция НАЧАЛА точек этого BC внутри общего конкатенированного
    батча (inputs/outputs), который DeepXDE передаёт в func целиком, а НЕ
    только точки этого конкретного BC -- несмотря на то, что название
    параметров "inputs, outputs" наводит на мысль, что это только "свои"
    точки. DeepXDE собирает полный батч в порядке: все bc/ic из списка
    icbcs (в их собственном порядке) друг за другом, а точки самого PDE
    (domain) -- ПОСЛЕДНИМИ. offset = сумма длин всех bc/ic, идущих в
    icbcs ПЕРЕД этим (см. build_model: offset = 2*num_initial +
    2*num_boundary, т.к. norm_penalty_bc добавляется в icbcs последним,
    после ic_u, ic_v, bc_u, bc_v). Если порядок icbcs в build_model
    изменится, offset нужно пересчитать соответственно.

    ВАЖНО: функция использует операции PyTorch напрямую (torch.Tensor),
    поэтому работает только с DDE_BACKEND=pytorch (как и весь остальной
    проект, см. requirements.txt).
    """
    t_values = np.linspace(0.0, T, n_slices)
    x_grid = np.linspace(-L, L, nx_slice)
    dx = x_grid[1] - x_grid[0]

    points = np.concatenate(
        [np.column_stack([x_grid, np.full(nx_slice, t)]) for t in t_values],
        axis=0,
    )
    values = np.zeros((points.shape[0], 1))
    n_own = points.shape[0]

    def func(inputs, outputs, X):
        own = outputs[offset:offset + n_own]
        u = own[:, 0]
        v = own[:, 1]
        prob = (u ** 2 + v ** 2).reshape(n_slices, nx_slice)
        # Трапеции на равномерной сетке: dx*(sum - 0.5*(первый+последний)).
        integral = dx * (prob.sum(dim=1) - 0.5 * (prob[:, 0] + prob[:, -1]))
        residual = integral - 1.0  # форма (n_slices,), цель -- 0
        own_result = residual.repeat_interleave(nx_slice).reshape(-1, 1)

        # error() внутри PointSetOperatorBC сам вырежет [offset:offset+n_own]
        # из того, что вернёт func -- поэтому возвращаем результат той же
        # длины, что и полный outputs, с нулями везде, кроме "своего" блока.
        full = torch.zeros((outputs.shape[0], 1), dtype=own_result.dtype,
                            device=own_result.device)
        full[offset:offset + n_own] = own_result
        return full

    return dde.icbc.PointSetOperatorBC(points, values, func)


def build_model(num_domain=4000, num_boundary=400, num_initial=400,
                 norm_penalty=False, norm_penalty_slices=8, norm_penalty_nx=400):
    """Собирает dde.Model для теста свободного движения.

    norm_penalty: если True, добавляет мягкое ограничение на сохранение
    нормы (см. build_norm_penalty_bc) как дополнительный член в icbcs.
    """

    ok, info = check_L_sufficient(L=L, x0=X0, b=B, eps=EPS, T=T)
    assert ok, (
        f"L={L} недостаточно для x0={X0}, b={B}, eps={EPS}, T={T}: {info}"
    )

    geom = dde.geometry.Interval(-L, L)
    timedomain = dde.geometry.TimeDomain(0.0, T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    coeff = h_planck / (2.0 * m)  # = 0.5 при hbar=1, m=1 (physics/constants.py)

    def pde(x, y):
        du_t = dde.grad.jacobian(y, x, i=0, j=1)
        dv_t = dde.grad.jacobian(y, x, i=1, j=1)
        du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
        eq_u = du_t + coeff * dv_xx
        eq_v = dv_t - coeff * du_xx
        return [eq_u, eq_v]

    def ic_u_func(x):
        return u_initial(x[:, 0:1], X0, B, EPS)

    def ic_v_func(x):
        return v_initial(x[:, 0:1], X0, B, EPS)

    def bc_func(x):
        u_bc, _ = boundary_values(x[:, 0:1])
        return u_bc.reshape(-1, 1)

    on_initial = lambda x, on_init: on_init
    on_boundary = lambda x, on_bnd: on_bnd

    ic_u = dde.icbc.IC(geomtime, ic_u_func, on_initial, component=0)
    ic_v = dde.icbc.IC(geomtime, ic_v_func, on_initial, component=1)
    bc_u = dde.icbc.DirichletBC(geomtime, bc_func, on_boundary, component=0)
    bc_v = dde.icbc.DirichletBC(geomtime, bc_func, on_boundary, component=1)

    icbcs = [ic_u, ic_v, bc_u, bc_v]
    if norm_penalty:
        # norm_penalty_bc добавляется ПОСЛЕДНИМ в icbcs, поэтому его точки
        # в общем батче идут после точек ic_u, ic_v, bc_u, bc_v -- ровно
        # 2*num_initial + 2*num_boundary точек перед ним (см. docstring
        # build_norm_penalty_bc).
        offset = 2 * num_initial + 2 * num_boundary
        icbcs.append(build_norm_penalty_bc(norm_penalty_slices, norm_penalty_nx, offset))

    data = dde.data.TimePDE(
        geomtime,
        pde,
        icbcs,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
    )

    # Архитектура из плана проекта: вход (x,t), 5 скрытых слоёв по 50, tanh.
    net = dde.nn.FNN([2] + [50] * 5 + [2], "tanh", "Glorot uniform")

    model = dde.Model(data, net)
    return model


def compile_model(model, optimizer="adam", lr=1e-3, norm_penalty=False, norm_penalty_weight=100.0):
    # Порядок потерь в TimePDE: [pde_u, pde_v, ic_u, ic_v, bc_u, bc_v, (norm)].
    # λ_IC = λ_BC = 10, как в карточке проекта (усиливаем IC/BC).
    #
    # Явно передаём loss_weights при КАЖДОМ compile(), в том числе для
    # L-BFGS: если loss_weights не передать, DeepXDE молча сбрасывает веса
    # в None (равные веса), а не сохраняет те, что были заданы при
    # предыдущем compile(). Это обнаружено при разборе того, почему три
    # прошлых раунда доводки L-BFGS в scripts/train_free_particle.py
    # фактически обучались с равными весами, а не с λ_IC=λ_BC=10.
    loss_weights = [1, 1, 10, 10, 10, 10]
    if norm_penalty:
        loss_weights.append(norm_penalty_weight)
    model.compile(optimizer, lr=lr, loss_weights=loss_weights)
    return model


def load_net_weights(model, path):
    """
    Загружает ТОЛЬКО веса сети из чекпоинта model.save(), минуя состояние
    оптимизатора.

    dde.Model.restore() восстанавливает веса и состояние оптимизатора
    вместе и падает с KeyError, если текущий скомпилированный оптимизатор
    не совпадает по типу с тем, которым чекпоинт был сохранён (см.
    --resume_optimizer в scripts/train_free_particle.py). Если нужно просто
    "подхватить" уже обученные веса для ДРУГОЙ конфигурации обучения
    (например, добавив norm_penalty -- новый член функции потерь, после
    которого состояние предыдущего оптимизатора всё равно не актуально),
    проще и надёжнее загрузить только веса сети и обучать с чистого
    состояния оптимизатора.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.net.load_state_dict(checkpoint["model_state_dict"])
