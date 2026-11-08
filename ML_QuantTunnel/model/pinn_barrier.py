"""
PINN для теста 2: туннелирование через прямоугольный барьер (ε < W).

Что делает этот модуль
----------------------
Собирает dde.Model для одномерного TDSE с потенциалом

    V(x) = W,  0 <= x <= d,
    V(x) = 0,  иначе.

В отличие от model/pinn_free.py, здесь V(x) не равно нулю, поэтому в PDE
появляются слагаемые (V/ħ)v и -(V/ħ)u.

Конфигурация фиксирована под Тест 2 из docs/validation_plan.md:
ε < W, то есть режим туннелирования. Конкретные W, d, ε, L, T можно
менять, но тогда нужно заново проверить check_L_sufficient и левую
границу (отражённая волна не должна дойти до x = -L).

Источник уравнений:
[1] Лыков С.Н. Квантовая механика: учеб. пособие.
    СПб.: Изд-во Политехн. ун-та, 2008. — 236 с.
[2] Лыков С.Н., Гасумянц В.Э., Рыков С.А. Квантовая механика. Задачи
    с решениями: учеб. пособие.
    СПб.: Изд-во Политехн. ун-та, 2008. — 315 с.
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
# Фиксированная конфигурация Теста 2.
#
# ε = 0.5 < W = 1.0 — туннелирование.
# L = 40, T = 8 подобраны так, чтобы:
#   - прошедшая волна не дошла до x = +L;
#   - отражённая волна не дошла до x = -L.
# Проверка правой границы — через check_L_sufficient;
# проверка левой границы — отдельным assert в build_model.
# ---------------------------------------------------------------------------

X0 = -5.0
B = 1.0
EPS = 0.5

W_BARRIER = 1.0
BARRIER_WIDTH = 1.0

L = 40.0
T = 8.0


def V_barrier_torch(x, W, d):
    """
    Torch-версия потенциала прямоугольного барьера.

    physics.potentials.V_barrier использует numpy, а внутри dde.grad
    с бэкендом PyTorch приходят torch.Tensor. Поэтому здесь отдельная
    torch-совместимая реализация.

    Параметры
    ---------
    x : torch.Tensor формы (N, 1)
        Пространственная координата.
    W : float
        Высота барьера.
    d : float
        Ширина барьера.

    Возвращает
    ----------
    V : torch.Tensor формы (N, 1)
        Значение потенциала: W при 0 <= x <= d, иначе 0.
    """
    inside = (x >= 0.0) & (x <= d)
    return torch.where(
        inside,
        torch.full_like(x, float(W)),
        torch.zeros_like(x),
    )


def build_norm_penalty_bc(n_slices, nx_slice, offset, L, T):
    """
    Мягкое ограничение на сохранение нормы: ∫|Ψ(x,t)|²dx = 1
    на нескольких срезах t ∈ [0, T].

    Это локальная копия функции из model/pinn_free.py, но с явными
    параметрами L и T, потому что у барьера они другие.

    offset — позиция начала точек этого BC в общем конкатенированном
    батче DeepXDE. В build_model icbcs имеют порядок:
        [ic_u, ic_v, bc_u, bc_v, norm_penalty_bc],
    поэтому offset = 2*num_initial + 2*num_boundary.
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

        # Трапеции на равномерной сетке.
        integral = dx * (prob.sum(dim=1) - 0.5 * (prob[:, 0] + prob[:, -1]))
        residual = integral - 1.0
        own_result = residual.repeat_interleave(nx_slice).reshape(-1, 1)

        full = torch.zeros(
            (outputs.shape[0], 1),
            dtype=own_result.dtype,
            device=own_result.device,
        )
        full[offset:offset + n_own] = own_result
        return full

    return dde.icbc.PointSetOperatorBC(points, values, func)


def build_model(
    num_domain=4000,
    num_boundary=400,
    num_initial=400,
    norm_penalty=False,
    norm_penalty_slices=8,
    norm_penalty_nx=400,
):
    """
    Собирает dde.Model для Теста 2 (туннелирование через барьер).
    """
    assert EPS < W_BARRIER, (
        f"Для Теста 2 нужно ε < W, а сейчас EPS={EPS}, W={W_BARRIER}"
    )

    # Правая граница: прошедшая волна не должна дойти до x = +L.
    ok, info = check_L_sufficient(L=L, x0=X0, b=B, eps=EPS, T=T)
    assert ok, (
        f"L={L} недостаточно для правой границы: {info}"
    )

    # Левая граница: отражённая волна не должна дойти до x = -L.
    x_left_reach = X0 - info["v_max"] * T
    assert x_left_reach > -L, (
        f"L={L} недостаточно для левой границы: "
        f"x_left_reach={x_left_reach:.3f} <= -L={-L}"
    )

    geom = dde.geometry.Interval(-L, L)
    timedomain = dde.geometry.TimeDomain(0.0, T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    coeff = h_planck / (2.0 * m)  # = 0.5 при ħ = 1, m = 1

    def pde(x, y):
        du_t = dde.grad.jacobian(y, x, i=0, j=1)
        dv_t = dde.grad.jacobian(y, x, i=1, j=1)
        du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)

        u = y[:, 0:1]
        v = y[:, 1:2]

        # V(x) в точках коллокации (torch-тензор, не numpy).
        V = V_barrier_torch(x[:, 0:1], W_BARRIER, BARRIER_WIDTH)

        # Система для Ψ = u + iv:
        #   ∂u/∂t = -(ħ/2m) ∂²v/∂x² + (V/ħ) v
        #   ∂v/∂t =  (ħ/2m) ∂²u/∂x² - (V/ħ) u
        eq_u = du_t + coeff * dv_xx - (V / h_planck) * v
        eq_v = dv_t - coeff * du_xx + (V / h_planck) * u
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
        offset = 2 * num_initial + 2 * num_boundary
        icbcs.append(
            build_norm_penalty_bc(
                norm_penalty_slices,
                norm_penalty_nx,
                offset,
                L,
                T,
            )
        )

    data = dde.data.TimePDE(
        geomtime,
        pde,
        icbcs,
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
    )

    net = dde.nn.FNN([2] + [50] * 5 + [2], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    return model


def compile_model(
    model,
    optimizer="adam",
    lr=1e-3,
    norm_penalty=False,
    norm_penalty_weight=100.0,
):
    """
    Компилирует модель с явными loss_weights.

    Порядок потерь:
        [pde_u, pde_v, ic_u, ic_v, bc_u, bc_v, (norm)]
    """
    loss_weights = [1, 1, 10, 10, 10, 10]
    if norm_penalty:
        loss_weights.append(norm_penalty_weight)
    model.compile(optimizer, lr=lr, loss_weights=loss_weights)
    return model


def load_net_weights(model, path):
    """
    Загружает только веса сети из чекпоинта model.save(),
    без состояния оптимизатора.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.net.load_state_dict(checkpoint["model_state_dict"])


# ---------------------------------------------------------------------------
# Вспомогательная функция для Теста 2: D_PINN.
# ---------------------------------------------------------------------------

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


def transmission_pinn(model, t=T, x_start=BARRIER_WIDTH, x_end=L, nx=6000):
    """
    Вычисляет D_PINN = ∫_{x_start}^{x_end} |Ψ(x,t)|² dx.

    Для барьера x_start = d (правее барьера), x_end = L.
    """
    x = np.linspace(x_start, x_end, nx)
    X = np.column_stack([x, np.full_like(x, t)])
    y = model.predict(X)
    prob = y[:, 0] ** 2 + y[:, 1] ** 2
    return _trapz(prob, x)