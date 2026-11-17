"""Strong-form residual and condition losses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from .autodiff import grad, use_derivative_backend
from .models import DomainDecompositionPINN
from .problems import PDEProblem
from .sampling import SampleBatch


@runtime_checkable
class FormulationProtocol(Protocol):
    """Structural contract consumed by :class:`openpinn.Trainer`.

    A formulation owns the mathematical problem and converts a sampled batch
    into named scalar losses.  Implementations do not need to inherit from
    :class:`StrongFormulation`; satisfying this protocol is sufficient.
    """

    @property
    def problem(self) -> PDEProblem:
        """Problem whose residuals and conditions define the objective."""

        ...

    def loss_terms(
        self, model: nn.Module, batch: SampleBatch
    ) -> Mapping[str, Tensor]:
        """Return named scalar loss tensors for one sampled batch."""

        ...


class StrongFormulation:
    """Pointwise strong-form PINN objective.

    This baseline computes mean-square PDE residuals at interior collocation
    points and mean-square pointwise errors for every condition.  Weak, mixed,
    energy and discrete-time forms are deliberately left as future classes.
    """

    def __init__(self, problem: PDEProblem, derivative_backend: object | None = None) -> None:
        self.problem = problem
        self.derivative_backend = derivative_backend

    def residuals(self, model: nn.Module, points: Tensor) -> dict[str, Tensor]:
        """Evaluate named strong-form residual tensors."""

        with use_derivative_backend(self.derivative_backend):
            prediction = model(points)
            return self.problem.evaluate_residual(points, prediction)

    def pointwise_loss_terms(
        self, model: nn.Module, batch: SampleBatch
    ) -> dict[str, Tensor]:
        """Evaluate one loss value per collocation/condition point."""

        if batch.interior.shape[0] == 0:
            raise ValueError("Strong-form training requires at least one interior point.")
        terms: dict[str, Tensor] = {}
        for name, values in self.residuals(model, batch.interior).items():
            squared = values.square().reshape(values.shape[0], -1).mean(dim=1)
            if batch.interior_weights is not None:
                if batch.interior_weights.shape != (squared.shape[0], 1):
                    raise ValueError(
                        "interior_weights must have shape (n_interior, 1)."
                    )
                squared = squared * batch.interior_weights[:, 0]
            terms[name] = squared
        for condition in self.problem.conditions:
            points = getattr(batch, condition.sample_kind)
            if points.shape[0] == 0:
                raise ValueError(
                    f"Condition {condition.name!r} requires {condition.sample_kind} samples."
                )
            name = condition.name
            if name in terms:
                suffix = 2
                while f"{name}_{suffix}" in terms:
                    suffix += 1
                name = f"{name}_{suffix}"
            errors = condition.error(model, points)
            terms[name] = condition.weight * errors.square().reshape(
                errors.shape[0], -1
            ).mean(dim=1)
        return terms

    def loss_terms(self, model: nn.Module, batch: SampleBatch) -> dict[str, Tensor]:
        """Evaluate named scalar losses for a sampled batch."""

        return {
            name: torch.mean(values)
            for name, values in self.pointwise_loss_terms(model, batch).items()
        }


class InterfaceStrongFormulation(StrongFormulation):
    """Strong form plus two-subdomain solution and physical-flux matching."""

    def __init__(self, problem: PDEProblem, derivative_backend: object | None = None) -> None:
        super().__init__(problem, derivative_backend)
        required = {
            "interface",
            "left_diffusivity",
            "right_diffusivity",
        }
        missing = sorted(required.difference(problem.metadata))
        if missing:
            raise ValueError(
                "Interface formulation requires heterogeneous problem metadata: "
                + ", ".join(missing)
            )

    def loss_terms(self, model: nn.Module, batch: SampleBatch) -> dict[str, Tensor]:
        """Add value and flux continuity at the material interface."""

        if not isinstance(model, DomainDecompositionPINN):
            raise TypeError(
                "InterfaceStrongFormulation requires DomainDecompositionPINN."
            )
        terms = super().loss_terms(model, batch)
        interface = float(self.problem.metadata["interface"])
        if model.interface != interface:
            raise ValueError("Model and problem interface locations must match.")
        point = batch.interior.new_tensor([[interface]], requires_grad=True)
        left, right = model.branch_values(point)
        with use_derivative_backend(self.derivative_backend):
            left_slope = grad(left, point)[:, :1]
            right_slope = grad(right, point)[:, :1]
        left_flux = float(self.problem.metadata["left_diffusivity"]) * left_slope
        right_flux = float(self.problem.metadata["right_diffusivity"]) * right_slope
        terms["interface_solution"] = torch.mean((left - right).square())
        terms["interface_flux"] = torch.mean((left_flux - right_flux).square())
        return terms
