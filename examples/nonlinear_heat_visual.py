"""Visual-designer example: a nonlinear one-dimensional reaction-diffusion PDE."""

from pathlib import Path

import openpinn as op
import torch


# Beginner-editable settings.
STEPS = 5
DIFFUSIVITY = 0.1

DOMAIN = op.SpaceTimeDomain(op.Interval(-1.0, 1.0), 0.0, 1.0)


def equation(x, u):
    """Return u_t - diffusivity * u_xx + u**3."""

    u_t = op.grad(u, x)[:, 1:2]
    u_xx = op.hessian(u, x, i=0, j=0)
    return u_t - DIFFUSIVITY * u_xx + u**3


PROBLEM = op.PDEProblem(
    DOMAIN,
    equation,
    [
        op.DirichletBC(0.0, name="boundary"),
        op.InitialCondition(
            lambda points: torch.sin(torch.pi * points[:, 0:1]),
            name="initial",
        ),
    ],
    name="nonlinear_heat_1d",
    metadata={
        "kind": "custom_dynamic",
        "residual_names": ("pde",),
    },
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

result = op.solve(
    PROBLEM,
    method="baseline",
    variant="plain_pinn",
    model={"hidden_layers": [32, 32, 32], "activation": "tanh"},
    steps=STEPS,
    training={"log_every": 1},
    output_root=PROJECT_ROOT / "runs",
)

print("训练完成")
print("结果目录：", result.output_dir)
