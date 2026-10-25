# models/pinn_free_ideal.py
import deepxde as dde
import numpy as np
import torch

from physics.constants import h_planck, m
from physics.initial_condition import u_initial, v_initial
from physics.boundary_conditions import boundary_values

# Безопасный trapz для разных версий torch
_trapz = getattr(torch, "trapezoid", torch.trapz)


class IdealTimePDE(dde.data.TimePDE):
    def __init__(self, *args, L=15.0, T=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        # Сетка для расчёта нормы
        self.x_norm = torch.linspace(-L, L, 400).view(-1, 1)
        self.t_norm = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
        self.dx = (2 * L) / 399

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        # Стандартные лоссы (PDE, IC, BC)
        std_losses = super().losses(targets, outputs, loss_fn, inputs, model, aux)

        # Штраф за нарушение нормы
        norm_loss = 0.0
        for t_val in self.t_norm:
            t_exp = t_val.expand_as(self.x_norm)
            X_eval = torch.hstack([self.x_norm, t_exp])
            y_eval = model.net(X_eval)
            prob = y_eval[:, 0:1] ** 2 + y_eval[:, 1:2] ** 2
            integral = _trapz(prob.squeeze(), dx=self.dx)
            norm_loss += (integral - 1.0) ** 2

        std_losses.append(norm_loss / len(self.t_norm))
        return std_losses


def build_pinn_free_ideal(L, T, x0, b, eps):
    geom = dde.geometry.Interval(-L, L)
    timedomain = dde.geometry.TimeDomain(0.0, T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    coeff = h_planck / (2.0 * m)

    def pde(x, y):
        du_t = dde.grad.jacobian(y, x, i=0, j=1)
        dv_t = dde.grad.jacobian(y, x, i=1, j=1)
        du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
        return [du_t + coeff * dv_xx, dv_t - coeff * du_xx]

    def ic_u_func(x):
        return u_initial(x[:, 0:1], x0, b, eps)

    def ic_v_func(x):
        return v_initial(x[:, 0:1], x0, b, eps)

    def bc_func(x):
        u_bc, _ = boundary_values(x[:, 0:1])
        return u_bc.reshape(-1, 1)

    on_initial = lambda x, on_init: on_init
    on_boundary = lambda x, on_boundary_flag: on_boundary_flag

    ic_u = dde.icbc.IC(geomtime, ic_u_func, on_initial, component=0)
    ic_v = dde.icbc.IC(geomtime, ic_v_func, on_initial, component=1)
    bc_u = dde.icbc.DirichletBC(geomtime, bc_func, on_boundary, component=0)
    bc_v = dde.icbc.DirichletBC(geomtime, bc_func, on_boundary, component=1)

    data = IdealTimePDE(
        geomtime, pde, [ic_u, ic_v, bc_u, bc_v],
        num_domain=4000, num_boundary=400, num_initial=400,
        L=L, T=T
    )

    net = dde.nn.FNN([2] + [64] * 5 + [2], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    return model