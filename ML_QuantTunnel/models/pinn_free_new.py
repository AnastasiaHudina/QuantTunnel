# models/pinn_free_new.py
import deepxde as dde
import numpy as np

from physics.constants import h_planck, m
from physics.initial_condition import u_initial, v_initial
from physics.boundary_conditions import boundary_values


def build_pinn_free_new(L, T, x0, b, eps):
    geom = dde.geometry.Interval(-L, L)
    timedomain = dde.geometry.TimeDomain(0.0, T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    coeff = h_planck / (2.0 * m)

    def pde(x, y):
        du_t = dde.grad.jacobian(y, x, i=0, j=1)
        dv_t = dde.grad.jacobian(y, x, i=1, j=1)
        du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)

        eq_u = du_t + coeff * dv_xx
        eq_v = dv_t - coeff * du_xx
        return [eq_u, eq_v]

    def ic_u_func(x):
        return u_initial(x[:, 0:1], x0, b, eps)

    def ic_v_func(x):
        return v_initial(x[:, 0:1], x0, b, eps)

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
        geomtime, pde, [ic_u, ic_v, bc_u, bc_v],
        num_domain=4000, num_boundary=400, num_initial=400,
    )

    # Оптимизация: 5 слоев по 64 нейрона (быстрее, чем 128, но лучше, чем 50)
    net = dde.nn.FNN([2] + [64] * 5 + [2], "tanh", "Glorot uniform")

    model = dde.Model(data, net)
    return model