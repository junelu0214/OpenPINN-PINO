"""Minimal custom PDE: 1D Helmholtz with a manufactured solution.

The residual lives on ``PDEProblem``, not inside a network ``forward``.
Optionally register the problem so YAML ``problem.name`` can resolve it.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor

import openpinn as op


def build_problem() -> op.PDEProblem:
    """Return ``-u_xx - u = (pi^2 - 1) sin(pi x)`` with zero endpoints."""

    domain = op.Interval(-1.0, 1.0)

    def exact(x: Tensor) -> Tensor:
        return torch.sin(math.pi * x[:, :1])

    def residual(x: Tensor, u: Tensor) -> Tensor:
        source = (math.pi**2 - 1.0) * torch.sin(math.pi * x[:, :1])
        return -op.hessian(u, x, i=0, j=0) - u - source

    return op.PDEProblem(
        domain,
        residual,
        [op.DirichletBC(0.0, name="boundary")],
        exact_solution=exact,
        name="helmholtz_1d",
        metadata={
            "kind": "custom_static",
            "equation": "-u_xx - u = (pi^2 - 1) sin(pi x)",
            "residual_names": ("pde",),
        },
    )


def register_problem(registry: op.ComponentRegistry | None = None) -> op.ComponentRegistry:
    """Register the Helmholtz problem on an isolated component registry."""

    selected = (
        op.default_component_registry() if registry is None else registry
    )
    if not selected.contains("problem", "helmholtz_1d"):
        selected.register("problem", "helmholtz_1d", build_problem)
    return selected


def main(output_dir: str | Path | None = None) -> op.ExperimentResult:
    registry = register_problem()
    problem = registry.create("problem", "helmholtz_1d")
    return op.solve(
        problem,
        methods=[op.method("baseline", "plain_pinn")],
        steps=5,
        seed=0,
        model={"hidden_layers": [16, 16]},
        sampling={"interior": 32, "boundary": 4},
        training={"log_every": 1},
        output_root=output_dir or Path(__file__).resolve().parents[1] / "runs/custom_pde",
        plot=False,
    )


if __name__ == "__main__":
    result = main()
    print("custom PDE finished")
    print("output:", result.output_dir)
    print("final total loss:", result.metrics.get("final_total_loss"))
