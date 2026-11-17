"""One-file template for a user-defined control equation."""

from __future__ import annotations

import math

import openpinn as op
import torch
from torch import Tensor


def build_problem() -> op.PDEProblem:
    """Return ``-u_xx = pi^2 sin(pi*x)`` with zero endpoint values."""

    domain = op.Interval(-1.0, 1.0)

    def exact(x: Tensor) -> Tensor:
        return torch.sin(math.pi * x[:, :1])

    def equation(x: Tensor, u: Tensor) -> Tensor:
        # 用户主要修改这里：返回控制方程左边减右边的残差。
        return -op.hessian(u, x, i=0, j=0) - math.pi**2 * torch.sin(
            math.pi * x[:, :1]
        )

    return op.PDEProblem(
        domain,
        equation,
        [op.DirichletBC(0.0, name="boundary")],
        exact_solution=exact,
        name="my_poisson",
        metadata={"kind": "custom_static", "equation": "-u_xx = pi^2 sin(pi*x)"},
    )
