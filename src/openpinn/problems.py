"""PDE problem definitions."""

from __future__ import annotations

import operator
from typing import Any, Callable, Mapping, Sequence, Union

from torch import Tensor

from .conditions import Condition
from .geometry import (
    ClosedSpline,
    Disk,
    Ellipse,
    Interval,
    Polygon,
    RectangleWithCircularHoles,
    RoundedRectangle,
    SpaceTimeDomain,
)

Residual = Callable[[Tensor, Tensor], Union[Tensor, Mapping[str, Tensor]]]
ExactSolution = Callable[[Tensor], Tensor]


class PDEProblem:
    """Bundle a domain, PDE residual and boundary/initial conditions.

    Parameters
    ----------
    domain:
        Geometry on which the equation is solved.
    residual:
        Callable ``residual(coordinates, prediction)``.  It may return one
        tensor or a mapping of named residual tensors for multi-equation PDEs.
    conditions:
        Dirichlet and/or initial constraints.
    exact_solution:
        Optional callable used only for evaluation, never for training.
    """

    def __init__(
        self,
        domain: Interval | Disk | Ellipse | RoundedRectangle | Polygon | ClosedSpline | RectangleWithCircularHoles | SpaceTimeDomain,
        residual: Residual | None = None,
        conditions: Sequence[Condition] | None = None,
        *,
        exact_solution: ExactSolution | None = None,
        name: str = "pde",
        metadata: Mapping[str, Any] | None = None,
        field_names: Sequence[str] | None = None,
        output_dim: int | None = None,
        pde: Residual | None = None,
        bcs: Sequence[Condition] | None = None,
    ) -> None:
        if residual is not None and pde is not None:
            raise ValueError("Pass residual or pde, not both.")
        residual_fn = residual if residual is not None else pde
        if residual_fn is None or not callable(residual_fn):
            raise TypeError("PDEProblem requires a callable residual.")
        if conditions is not None and bcs is not None:
            raise ValueError("Pass conditions or bcs, not both.")
        selected_conditions = conditions if conditions is not None else bcs
        self.domain = domain
        self.residual = residual_fn
        self.conditions = tuple(selected_conditions or ())
        self.exact_solution = exact_solution
        self.name = str(name)
        self.metadata = dict(metadata or {})
        if output_dim is not None and isinstance(output_dim, bool):
            raise TypeError("output_dim must be a positive integer, not bool.")
        try:
            declared_output = 1 if output_dim is None else operator.index(output_dim)
        except TypeError as exc:
            raise TypeError("output_dim must be a positive integer.") from exc
        if declared_output <= 0:
            raise ValueError("output_dim must be positive.")
        if field_names is None:
            names = (
                ("u",)
                if declared_output == 1
                else tuple(f"u_{index}" for index in range(declared_output))
            )
        else:
            names = tuple(field_names)
            if not names or any(
                not isinstance(field, str) or not field.strip() for field in names
            ):
                raise ValueError("field_names must contain non-empty strings.")
            names = tuple(field.strip() for field in names)
            if len(set(names)) != len(names):
                raise ValueError("field_names must be unique.")
            if output_dim is None:
                declared_output = len(names)
        if len(names) != declared_output:
            raise ValueError(
                f"field_names has length {len(names)}, but output_dim={declared_output}."
            )
        self.field_names = names
        self.output_dim = declared_output

    @property
    def pde(self) -> Residual:
        """Alias retained for familiar PINN terminology."""

        return self.residual

    @property
    def bcs(self) -> tuple[Condition, ...]:
        """Alias for all boundary and initial conditions."""

        return self.conditions

    def evaluate_residual(
        self, coordinates: Tensor, prediction: Tensor
    ) -> dict[str, Tensor]:
        """Evaluate and normalize the PDE residual to a named mapping."""

        if coordinates.ndim != 2:
            raise ValueError("coordinates must have shape (N, domain_dim).")
        if coordinates.shape[1] != self.domain.dim:
            raise ValueError(
                f"coordinates have width {coordinates.shape[1]}, expected "
                f"domain dimension {self.domain.dim}."
            )
        if prediction.ndim != 2:
            raise ValueError("prediction must have shape (N, output_dim).")
        if prediction.shape[0] != coordinates.shape[0]:
            raise ValueError(
                "prediction and coordinates must have the same batch dimension."
            )
        if prediction.shape[1] != self.output_dim:
            raise ValueError(
                f"prediction has width {prediction.shape[1]}, expected "
                f"problem.output_dim={self.output_dim}."
            )

        values = self.residual(coordinates, prediction)
        if isinstance(values, Tensor):
            values = {"pde": values}
        if not isinstance(values, Mapping) or not values:
            raise TypeError("Residual must return a Tensor or a non-empty mapping of Tensors.")
        normalized: dict[str, Tensor] = {}
        for key, value in values.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"Residual {key!r} is not a torch.Tensor.")
            name = str(key)
            if name in normalized:
                raise ValueError(f"Residual name {name!r} is duplicated after normalization.")
            if value.ndim != 2 or value.shape[1] != 1:
                raise ValueError(
                    f"Residual {name!r} must have shape (N, 1), got "
                    f"{tuple(value.shape)}."
                )
            if value.shape[0] != coordinates.shape[0]:
                raise ValueError(
                    f"Residual {name!r} has batch size {value.shape[0]}, expected "
                    f"{coordinates.shape[0]}."
                )
            normalized[name] = value
        return normalized

    def exact(self, coordinates: Tensor) -> Tensor:
        """Evaluate the optional reference solution."""

        if self.exact_solution is None:
            raise RuntimeError(f"Problem {self.name!r} has no exact solution callable.")
        values = self.exact_solution(coordinates)
        if not isinstance(values, Tensor):
            raise TypeError("Exact solution must return a torch.Tensor.")
        expected = (coordinates.shape[0], self.output_dim)
        if tuple(values.shape) != expected:
            raise ValueError(
                f"Exact solution must have shape {expected}, got {tuple(values.shape)}."
            )
        return values
