"""Runnable polygon example matching the web configurator output."""

from pathlib import Path

import openpinn as op
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLYGON_VERTICES = [
    [-2, -1.5],
    [2, -1.5],
    [2, -0.25],
    [0.5, -0.25],
    [0.5, 1.5],
    [-2, 1.5],
]
POLYGON_HOLES: list[list[list[float]]] = []
SPACE = op.Polygon(POLYGON_VERTICES, holes=POLYGON_HOLES)


def residual(points: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Residual for ``-laplacian(u) = 1`` on an L-shaped domain."""

    return (
        -op.hessian(values, points, i=0, j=0)
        - op.hessian(values, points, i=1, j=1)
        - torch.ones_like(values)
    )


PROBLEM = op.PDEProblem(
    SPACE,
    residual,
    [op.DirichletBC(0.0, name="boundary_polygon")],
    name="l_shape_poisson",
    metadata={
        "kind": "custom_static",
        "geometry": "polygon",
        "residual_names": ("pde",),
    },
)

result = op.solve(
    PROBLEM,
    methods=[op.method("baseline", "plain_pinn")],
    steps=5,
    seed=0,
    model={"hidden_layers": [24, 24]},
    sampling={"interior": 128, "boundary": 64},
    training={"log_every": 1},
    output_root=PROJECT_ROOT / "runs/polygon_example",
)

print("训练完成")
print("结果目录：", result.output_dir)
