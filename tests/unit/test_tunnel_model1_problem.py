from __future__ import annotations

import torch

import openpinn as op


def test_twin_tunnel_geometry_samples_valid_soil_and_all_boundaries() -> None:
    space = op.RectangleWithCircularHoles(
        (0.0, 0.0),
        (1.0, 1.0),
        ((0.25, 0.5, 0.1), (0.75, 0.5, 0.1)),
    )
    interior = space.sample(512, seed=17, dtype=torch.float64)
    boundary = space.boundary(512, seed=23, dtype=torch.float64)

    assert interior.shape == (512, 2)
    assert boundary.shape == (512, 2)
    assert bool(space.contains(interior).all())
    assert bool(space.on_boundary(boundary, atol=1e-10).all())
    for center_x, center_y, radius in space.holes:
        distances = torch.linalg.vector_norm(
            interior - interior.new_tensor((center_x, center_y)), dim=1
        )
        assert bool((distances > radius).all())


def test_model1_residual_uses_x_y_and_time_derivatives() -> None:
    problem = op.make_tunnel_consolidation_model1_problem()
    coordinates = torch.tensor(
        [[0.10, 0.15, 0.20], [0.45, 0.75, 0.80]],
        dtype=torch.float64,
        requires_grad=True,
    )
    prediction = (
        coordinates[:, 0:1].square()
        + coordinates[:, 1:2].square()
        + coordinates[:, 2:3]
    )
    residual = problem.evaluate_residual(coordinates, prediction)["pde"]
    expected = 1.0 - 2.0 * 0.0075 - 2.0 * 0.015

    assert problem.domain.dim == 3
    assert torch.allclose(residual, torch.full_like(residual, expected))
    assert problem.exact_solution is None


def test_model1_short_name_builds_editable_problem_parameters() -> None:
    config, method = op.build_config(
        "tunnel_model1",
        method="baseline",
        steps=3,
        problem_options={"parameters": {"initial_pressure_kpa": 30.0}},
    )

    assert config["problem"]["name"] == "tunnel_consolidation_model1"
    assert config["problem"]["parameters"]["initial_pressure_kpa"] == 30.0
    assert config["sampling"] == {
        "name": "uniform",
        "interior": 512,
        "boundary": 256,
        "initial": 256,
    }
    assert config["evaluation"]["points"] == {"x": 61, "y": 61, "time": 3}
    assert method.name == "baseline"
