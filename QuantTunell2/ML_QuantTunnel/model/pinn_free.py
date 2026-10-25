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


def build_model(num_domain=4000, num_boundary=400, num_initial=400):
    """Собирает dde.Model для теста свободного движения."""

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

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [ic_u, ic_v, bc_u, bc_v],
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
    )

    # Архитектура из плана проекта: вход (x,t), 5 скрытых слоёв по 50, tanh.
    net = dde.nn.FNN([2] + [50] * 5 + [2], "tanh", "Glorot uniform")

    model = dde.Model(data, net)
    return model


def compile_model(model, lr=1e-3):
    # Порядок потерь в TimePDE: [pde_u, pde_v, ic_u, ic_v, bc_u, bc_v].
    # λ_IC = λ_BC = 10, как в карточке проекта (усиливаем IC/BC).
    loss_weights = [1, 1, 10, 10, 10, 10]
    model.compile("adam", lr=lr, loss_weights=loss_weights)
    return model
