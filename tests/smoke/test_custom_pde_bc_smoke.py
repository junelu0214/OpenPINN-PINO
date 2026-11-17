"""Smoke-level runs for the custom PDE and custom BC examples."""

from __future__ import annotations

from pathlib import Path

import torch

import openpinn as op

from examples.custom_bc import EndpointFluxBC, build_problem as build_bc_problem
from examples.custom_pde import register_problem as register_pde


def test_custom_pde_registers_and_solves(tmp_path: Path) -> None:
    registry = register_pde()
    problem = registry.create("problem", "helmholtz_1d")
    assert isinstance(problem, op.PDEProblem)
    result = op.solve(
        problem,
        methods=[op.method("baseline", "plain_pinn")],
        steps=1,
        seed=0,
        model={"hidden_layers": [8, 8]},
        sampling={"interior": 8, "boundary": 2},
        output_root=tmp_path / "custom_pde",
        plot=False,
    )
    assert torch.isfinite(torch.tensor(result.metrics["final_total_loss"]))


def test_custom_bc_error_matches_exact_flux() -> None:
    problem = build_bc_problem()
    flux = next(
        condition
        for condition in problem.conditions
        if isinstance(condition, EndpointFluxBC)
    )
    points = problem.domain.sample_boundary(8).requires_grad_(True)
    error = flux.error(lambda coords: problem.exact(coords), points)
    assert torch.isfinite(error).all()
    torch.testing.assert_close(error, torch.zeros_like(error), atol=1e-5, rtol=1e-5)


def test_custom_bc_registers_and_solves(tmp_path: Path) -> None:
    from examples.custom_bc import register_problem

    registry = register_problem()
    problem = registry.create("problem", "poisson_custom_flux")
    result = op.solve(
        problem,
        methods=[op.method("baseline", "plain_pinn")],
        steps=1,
        seed=1,
        model={"hidden_layers": [8, 8]},
        sampling={"interior": 8, "boundary": 4},
        output_root=tmp_path / "custom_bc",
        plot=False,
    )
    assert torch.isfinite(torch.tensor(result.metrics["final_total_loss"]))
