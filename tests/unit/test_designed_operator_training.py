from __future__ import annotations

import math
from pathlib import Path

import torch

import openpinn as op


def _problem() -> op.PDEProblem:
    return op.PDEProblem(
        op.Interval(-1, 1),
        lambda x, u: op.hessian(u, x, i=0, j=0) + u,
        [
            op.DirichletBC(0.0, boundary=lambda points: torch.isclose(points[:, 0], points.new_tensor(-1.0))),
            op.DirichletBC(0.0, boundary=lambda points: torch.isclose(points[:, 0], points.new_tensor(1.0))),
        ],
        name="designed_helmholtz",
    )


def test_designed_deeponet_uses_observation_data_term() -> None:
    coordinates = torch.linspace(-1, 1, 5).reshape(-1, 1)
    observation = op.OperatorObservation(
        coordinates=coordinates,
        values=torch.sin(math.pi * coordinates),
        source=Path("observation.csv"),
    )
    config = op.build_operator_config(
        "deeponet",
        training_mode="data_only",
        n_dims=1,
        n_fields=1,
        n_sensors=4,
        latent_dim=4,
        branch_hidden=[4],
        trunk_hidden=[4],
        batch_size=1,
        n_query=4,
        n_modes=1,
        steps=1,
    )
    result = op.train_operator_problem(config, _problem(), observation=observation)
    assert result.operator == "deeponet"
    assert math.isfinite(result.metrics["final_data_loss"])


def test_designed_fno_physics_uses_grid_derivatives_without_observation() -> None:
    config = op.build_operator_config(
        "fno",
        training_mode="physics_only",
        n_dims=1,
        in_channels=1,
        out_channels=1,
        width=2,
        n_layers=1,
        n_modes=2,
        project_width=2,
        grid_size=4,
        batch_size=1,
        n_harmonics=1,
        derivative="finite_difference",
        steps=1,
    )
    result = op.train_operator_problem(config, _problem())
    assert result.operator == "fno"
    assert math.isfinite(result.metrics["final_physics_loss"])
    coordinates = torch.linspace(-1, 1, 4).reshape(-1, 1)
    reference_only = op.OperatorObservation(
        coordinates=coordinates,
        values=torch.full_like(coordinates, 99.0),
        source=Path("reference.csv"),
    )
    with_reference = op.train_operator_problem(
        config, _problem(), observation=reference_only
    )
    assert with_reference.final_loss == result.final_loss
