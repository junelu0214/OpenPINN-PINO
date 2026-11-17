from __future__ import annotations

import pytest
import torch

import openpinn as op


@pytest.mark.parametrize(
    "space",
    [
        op.Disk(center=(0.0, 0.0), radius=1.0),
        op.Ellipse(
            center=(0.25, -0.4),
            semi_axis_a=1.6,
            semi_axis_b=0.7,
            rotation_deg=32.0,
        ),
        op.RoundedRectangle(lower=(-1.5, -1.0), upper=(1.5, 1.0), corner_radius=0.35),
        op.ClosedSpline(
            ((-1.5, -0.6), (-0.8, -1.0), (0.8, -1.0), (1.5, -0.6),
             (1.5, 0.6), (0.8, 1.0), (-0.8, 1.0), (-1.5, 0.6)),
            tension=0.25,
        ),
    ],
)
def test_curved_geometry_mixed_boundary_short_run(tmp_path, space) -> None:
    def equation(points: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return -op.hessian(values, points, i=0, j=0) - op.hessian(
            values, points, i=1, j=1
        )

    problem = op.PDEProblem(
        space,
        equation,
        [op.NeumannBC(0.0, normal=space.boundary_normal, name="boundary_curve")],
        name="curved_geometry_smoke",
        metadata={"kind": "custom_static", "geometry": type(space).__name__},
    )
    result = op.solve(
        problem,
        methods=[op.method("baseline", "plain_pinn")],
        steps=1,
        seed=0,
        model={"hidden_layers": [8, 8]},
        sampling={"interior": 16, "boundary": 16},
        output_root=tmp_path,
        plot=False,
    )

    assert "boundary_curve_rmse" in result.metrics
