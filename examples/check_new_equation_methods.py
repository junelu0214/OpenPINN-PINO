"""Define one new equation and print method compatibility without training."""

import math

import openpinn as op
import torch


DOMAIN = op.Interval(-1.0, 1.0)


def equation(x, u):
    """Residual of ``-u_xx = pi^2 sin(pi*x)``."""

    u_xx = op.hessian(u, x, i=0, j=0)
    source = math.pi**2 * torch.sin(math.pi * x[:, :1])
    return -u_xx - source


PROBLEM = op.PDEProblem(
    DOMAIN,
    equation,
    [op.DirichletBC(0.0, name="boundary")],
    name="my_new_equation",
    metadata={"kind": "custom_static", "residual_names": ("pde",)},
)

REPORT = op.check_method_compatibility(PROBLEM)
print(REPORT.to_text())
