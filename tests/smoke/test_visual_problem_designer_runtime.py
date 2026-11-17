from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import openpinn as op


@pytest.mark.parametrize(
    ("space_dim", "dynamic"),
    ((1, False), (2, False), (1, True), (2, True)),
)
def test_visual_designer_problem_matrix_runs(
    tmp_path, space_dim: int, dynamic: bool
) -> None:
    space = (
        op.Interval(-1.0, 1.0)
        if space_dim == 1
        else op.RectangleWithCircularHoles(
            lower=(-1.0, -1.0), upper=(1.0, 1.0), holes=()
        )
    )
    domain = op.SpaceTimeDomain(space, 0.0, 1.0) if dynamic else space

    def equation(x, u):
        laplacian = op.hessian(u, x, i=0, j=0)
        if space_dim == 2:
            laplacian = laplacian + op.hessian(u, x, i=1, j=1)
        if dynamic:
            return op.grad(u, x)[:, space_dim : space_dim + 1] - laplacian
        x_hat = torch.sin(torch.pi * x[:, 0:1])
        if space_dim == 1:
            return -laplacian - torch.pi**2 * x_hat
        y_hat = torch.sin(torch.pi * x[:, 1:2])
        return -laplacian - 2.0 * torch.pi**2 * x_hat * y_hat

    conditions = [op.DirichletBC(0.0, name="boundary")]
    if dynamic:
        conditions.append(
            op.InitialCondition(
                lambda points: torch.sin(torch.pi * points[:, 0:1])
                * (
                    torch.sin(torch.pi * points[:, 1:2])
                    if space_dim == 2
                    else 1.0
                ),
                name="initial",
            )
        )
    problem = op.PDEProblem(
        domain,
        equation,
        conditions,
        name=f"visual_{'dynamic' if dynamic else 'static'}_{space_dim}d",
        metadata={
            "kind": "custom_dynamic" if dynamic else "custom_static",
            "residual_names": ("pde",),
        },
    )

    result = op.solve(
        problem,
        method="baseline",
        variant="plain_pinn",
        model={"hidden_layers": [8, 8], "activation": "tanh"},
        sampling={"interior": 8, "boundary": 4, "initial": 4},
        overrides={"evaluation": {"points": 16}},
        steps=1,
        training={"log_every": 1},
        output_root=tmp_path,
        plot=False,
    )

    assert result.metrics["problem"] == problem.name
    assert all(np.isfinite(values).all() for values in result.predictions.values())
    assert result.metrics["reference_solution_available"] is False


@pytest.mark.parametrize(
    "template",
    (
        "helmholtz_1d",
        "nonlinear_poisson_2d",
        "burgers_1d",
        "advection_diffusion_2d",
        "free_dynamic_2d",
    ),
)
def test_extended_configurator_templates_run(tmp_path, template: str) -> None:
    dimension = 2 if template.endswith("_2d") else 1
    dynamic = template in {
        "burgers_1d",
        "advection_diffusion_2d",
        "free_dynamic_2d",
    }
    space = (
        op.Interval(-1.0, 1.0)
        if dimension == 1
        else op.RectangleWithCircularHoles(
            lower=(-1.0, -1.0), upper=(1.0, 1.0), holes=()
        )
    )
    domain = op.SpaceTimeDomain(space, 0.0, 1.0) if dynamic else space

    def example_field(points):
        x_hat = (points[:, 0:1] + 1.0) / 2.0
        value = torch.sin(math.pi * x_hat)
        if dimension == 2:
            y_hat = (points[:, 1:2] + 1.0) / 2.0
            value = value * torch.sin(math.pi * y_hat)
        return value

    def equation(x, u):
        u_xx = op.hessian(u, x, i=0, j=0)
        laplacian = u_xx
        if dimension == 2:
            laplacian = laplacian + op.hessian(u, x, i=1, j=1)
        if template == "helmholtz_1d":
            eigenvalue = (math.pi / 2.0) ** 2
            source = (eigenvalue + 4.0) * example_field(x)
            return -laplacian + 4.0 * u - source
        if template == "nonlinear_poisson_2d":
            eigenvalue = 2.0 * (math.pi / 2.0) ** 2
            source = eigenvalue * example_field(x) + example_field(x) ** 3
            return -laplacian + u**3 - source
        gradient = op.grad(u, x)
        u_t = gradient[:, dimension : dimension + 1]
        if template == "burgers_1d":
            return u_t + u * gradient[:, 0:1] - 0.01 * u_xx
        if template == "advection_diffusion_2d":
            return (
                u_t
                + gradient[:, 0:1]
                + 0.5 * gradient[:, 1:2]
                - 0.1 * laplacian
            )
        return u_t - 0.1 * u_xx - 0.1 * op.hessian(u, x, i=1, j=1) + 0.2 * u**3

    conditions = [op.DirichletBC(0.0, name="boundary")]
    if dynamic:
        conditions.append(op.InitialCondition(example_field, name="initial"))
    problem = op.PDEProblem(
        domain,
        equation,
        conditions,
        exact_solution=example_field if not dynamic else None,
        name=f"configurator_{template}",
        metadata={
            "kind": "custom_dynamic" if dynamic else "custom_static",
            "residual_names": ("pde",),
        },
    )

    result = op.solve(
        problem,
        method="baseline",
        variant="plain_pinn",
        model={"hidden_layers": [8, 8], "activation": "tanh"},
        sampling={"interior": 8, "boundary": 4, "initial": 4},
        overrides={"evaluation": {"points": 16}},
        steps=1,
        training={"log_every": 1},
        output_root=tmp_path,
        plot=False,
    )

    assert result.metrics["problem"] == problem.name
    assert all(np.isfinite(values).all() for values in result.predictions.values())
    assert result.metrics["reference_solution_available"] is (not dynamic)
