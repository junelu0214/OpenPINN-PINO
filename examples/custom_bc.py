"""Minimal custom boundary condition: 1D endpoint flux.

``EndpointFluxBC`` subclasses ``Condition`` and prescribes ``du/dx`` at
selected points.  The PDE residual stays on ``PDEProblem``; the network
``forward`` still returns the field value.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor, nn

import openpinn as op
from openpinn.conditions import Condition, evaluate_value


class EndpointFluxBC(Condition):
    """Prescribe ``du/dx`` at selected 1D boundary points."""

    def __init__(
        self,
        value,
        boundary=None,
        *,
        name: str = "flux",
        weight: float = 1.0,
    ) -> None:
        super().__init__(value=value, name=name, weight=weight, selector=boundary)
        self.sample_kind = "boundary"

    def error(self, model: nn.Module, points: Tensor) -> Tensor:
        selected = self.select(points)
        if not selected.requires_grad:
            selected = selected.detach().requires_grad_(True)
        prediction = model(selected)
        flux = op.grad(prediction, selected)[:, :1]
        target = evaluate_value(self.value, selected, like=flux)
        return flux - target


def build_problem() -> op.PDEProblem:
    """Poisson manufactured solution with Dirichlet left and flux right."""

    domain = op.Interval(-1.0, 1.0)

    def exact(x: Tensor) -> Tensor:
        return torch.sin(math.pi * x[:, :1])

    def residual(x: Tensor, u: Tensor) -> Tensor:
        return -op.hessian(u, x, i=0, j=0) - math.pi**2 * torch.sin(
            math.pi * x[:, :1]
        )

    def left(points: Tensor) -> Tensor:
        return points[:, 0] < 0.0

    def right(points: Tensor) -> Tensor:
        return points[:, 0] >= 0.0

    def flux_value(points: Tensor) -> Tensor:
        return math.pi * torch.cos(math.pi * points[:, :1])

    return op.PDEProblem(
        domain,
        residual,
        [
            op.DirichletBC(0.0, boundary=left, name="left"),
            EndpointFluxBC(flux_value, boundary=right, name="right_flux"),
        ],
        exact_solution=exact,
        name="poisson_custom_flux",
        metadata={
            "kind": "custom_static",
            "residual_names": ("pde",),
        },
    )


def register_problem(registry: op.ComponentRegistry | None = None) -> op.ComponentRegistry:
    """Register the mixed-BC problem on an isolated component registry."""

    selected = (
        op.default_component_registry() if registry is None else registry
    )
    if not selected.contains("problem", "poisson_custom_flux"):
        selected.register("problem", "poisson_custom_flux", build_problem)
    return selected


def main(output_dir: str | Path | None = None) -> op.ExperimentResult:
    registry = register_problem()
    problem = registry.create("problem", "poisson_custom_flux")
    return op.solve(
        problem,
        methods=[op.method("baseline", "plain_pinn")],
        steps=5,
        seed=1,
        model={"hidden_layers": [16, 16]},
        sampling={"interior": 32, "boundary": 8},
        training={"log_every": 1},
        output_root=output_dir or Path(__file__).resolve().parents[1] / "runs/custom_bc",
        plot=False,
    )


if __name__ == "__main__":
    result = main()
    print("custom BC finished")
    print("output:", result.output_dir)
    print("final total loss:", result.metrics.get("final_total_loss"))
