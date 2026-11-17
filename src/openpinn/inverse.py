"""Inverse analysis for a Darcy instance and a consolidation problem family.

The PINN route jointly optimizes one pressure field and one bounded permeability.
The operator route learns ``(cv, U0) -> u(z, t)`` with DeepONet, then freezes
that map and optimizes the consolidation coefficient of a new family member.
All quantities are expected to be consistently nondimensionalized by the caller.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from .models import MLP
from .operators.deeponet import DeepONet, DeepONetConfig


def darcy_pressure(x: Tensor, permeability: Tensor | float, source: Tensor | float = 1.0) -> Tensor:
    """Exact pressure for ``-d(k dp/dx)/dx=q`` on ``[0,1]``, ``p(0)=p(1)=0``."""

    return torch.as_tensor(source, dtype=x.dtype, device=x.device) * x * (1.0 - x) / (
        2.0 * torch.as_tensor(permeability, dtype=x.dtype, device=x.device)
    )


def _bounded_raw(initial: float, lower: float, upper: float, *, name: str = "parameter") -> Tensor:
    if not (0.0 < lower < upper):
        raise ValueError(f"{name} bounds must satisfy 0 < lower < upper.")
    if not lower < initial < upper:
        raise ValueError(f"initial {name} must lie strictly inside its bounds.")
    fraction = (initial - lower) / (upper - lower)
    return torch.tensor(math.log(fraction / (1.0 - fraction)), dtype=torch.float32)


def _bounded_value(raw: Tensor, lower: float, upper: float) -> Tensor:
    return lower + (upper - lower) * torch.sigmoid(raw)


def _observations(coordinates: Tensor | Sequence[float], values: Tensor | Sequence[float]) -> tuple[Tensor, Tensor]:
    x = torch.as_tensor(coordinates, dtype=torch.float32).reshape(-1, 1)
    p = torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1)
    if x.shape != p.shape or x.shape[0] < 2:
        raise ValueError("inverse observations require at least two matching x and pressure values.")
    if not torch.isfinite(x).all() or not torch.isfinite(p).all():
        raise ValueError("inverse observations must be finite.")
    if torch.any((x < 0.0) | (x > 1.0)):
        raise ValueError("Darcy inverse coordinates must lie in [0, 1].")
    return x, p


def load_inverse_observations_csv(
    path: str | Path, *, coordinate_column: str = "x", value_column: str = "p"
) -> tuple[Tensor, Tensor]:
    """Load named ``x``/pressure columns for either inverse route."""

    source = Path(path)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or coordinate_column not in reader.fieldnames or value_column not in reader.fieldnames:
            raise ValueError(
                f"Observation CSV must contain {coordinate_column!r} and {value_column!r} columns."
            )
        rows = list(reader)
    try:
        coordinates = [float(row[coordinate_column]) for row in rows]
        values = [float(row[value_column]) for row in rows]
    except (TypeError, ValueError) as exc:
        raise ValueError("Observation CSV contains a nonnumeric inverse-analysis value.") from exc
    return _observations(coordinates, values)


class DarcyInversePINN(nn.Module):
    """Pressure MLP with one bounded, trainable permeability."""

    def __init__(
        self,
        *,
        initial_permeability: float,
        permeability_bounds: tuple[float, float],
        hidden_layers: Sequence[int] = (32, 32),
    ) -> None:
        super().__init__()
        self.lower, self.upper = map(float, permeability_bounds)
        self.raw_permeability = nn.Parameter(
            _bounded_raw(float(initial_permeability), self.lower, self.upper, name="permeability")
        )
        self.network = MLP(1, 1, hidden_layers=hidden_layers, activation="tanh")

    @property
    def permeability(self) -> Tensor:
        return _bounded_value(self.raw_permeability, self.lower, self.upper)

    def forward(self, coordinates: Tensor) -> Tensor:
        return coordinates * (1.0 - coordinates) * self.network(coordinates)


@dataclass
class DarcyPINNInverseResult:
    model: DarcyInversePINN
    estimated_permeability: float
    history: list[dict[str, float]]

    def predict(self, coordinates: Tensor | Sequence[float]) -> Tensor:
        x = torch.as_tensor(coordinates, dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            return self.model(x)


def inverse_darcy_pinn(
    coordinates: Tensor | Sequence[float],
    values: Tensor | Sequence[float],
    *,
    source: float = 1.0,
    initial_permeability: float = 1.0,
    permeability_bounds: tuple[float, float] = (0.2, 5.0),
    hidden_layers: Sequence[int] = (32, 32),
    collocation_points: int = 64,
    steps: int = 1200,
    learning_rate: float = 2e-3,
    data_weight: float = 500.0,
    seed: int = 0,
) -> DarcyPINNInverseResult:
    """Jointly identify one Darcy pressure field and its permeability."""

    observed_x, observed_p = _observations(coordinates, values)
    if collocation_points < 4 or steps < 1 or learning_rate <= 0 or data_weight <= 0:
        raise ValueError("collocation_points>=4, steps>=1, learning_rate>0 and data_weight>0 are required.")
    torch.manual_seed(int(seed))
    model = DarcyInversePINN(
        initial_permeability=initial_permeability,
        permeability_bounds=permeability_bounds,
        hidden_layers=hidden_layers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    history: list[dict[str, float]] = []
    for step in range(int(steps)):
        x = torch.linspace(0.0, 1.0, int(collocation_points)).reshape(-1, 1).requires_grad_(True)
        pressure = model(x)
        first = torch.autograd.grad(pressure, x, torch.ones_like(pressure), create_graph=True)[0]
        flux = model.permeability * first
        second = torch.autograd.grad(flux, x, torch.ones_like(flux), create_graph=True)[0]
        residual = -second - float(source)
        pde_loss = residual.square().mean()
        data_loss = (model(observed_x) - observed_p).square().mean()
        total = pde_loss + float(data_weight) * data_loss
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        if step == 0 or step == steps - 1 or (step + 1) % max(1, steps // 20) == 0:
            history.append({
                "step": float(step + 1), "total_loss": float(total.detach()),
                "pde_loss": float(pde_loss.detach()), "data_loss": float(data_loss.detach()),
                "permeability": float(model.permeability.detach()),
            })
    return DarcyPINNInverseResult(model, float(model.permeability.detach()), history)


def consolidation_pressure(
    coordinates: Tensor, consolidation_coefficient: Tensor | float, amplitude: Tensor | float = 1.0
) -> Tensor:
    """First-mode Terzaghi solution ``u=U0 sin(pi z) exp(-pi² cv t)``."""

    z = coordinates[..., 0:1]
    time = coordinates[..., 1:2]
    coefficient = torch.as_tensor(
        consolidation_coefficient, dtype=coordinates.dtype, device=coordinates.device
    )
    initial = torch.as_tensor(amplitude, dtype=coordinates.dtype, device=coordinates.device)
    return initial * torch.sin(math.pi * z) * torch.exp(-(math.pi**2) * coefficient * time)


def _spacetime_observations(
    coordinates: Tensor | Sequence[Sequence[float]], values: Tensor | Sequence[float]
) -> tuple[Tensor, Tensor]:
    points = torch.as_tensor(coordinates, dtype=torch.float32)
    pressure = torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] != pressure.shape[0] or points.shape[0] < 3:
        raise ValueError("consolidation observations require at least three matching (z,t,u) rows.")
    if not torch.isfinite(points).all() or not torch.isfinite(pressure).all():
        raise ValueError("consolidation observations must be finite.")
    if torch.any((points < 0.0) | (points > 1.0)):
        raise ValueError("normalized consolidation coordinates z and t must lie in [0, 1].")
    return points, pressure


def load_consolidation_observations_csv(
    path: str | Path, *, depth_column: str = "z", time_column: str = "t", value_column: str = "u"
) -> tuple[Tensor, Tensor]:
    """Load normalized ``z,t,u`` observations for operator inversion."""

    source = Path(path)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = (depth_column, time_column, value_column)
        if not reader.fieldnames or any(name not in reader.fieldnames for name in required):
            raise ValueError(f"Observation CSV must contain columns {required!r}.")
        rows = list(reader)
    try:
        points = [[float(row[depth_column]), float(row[time_column])] for row in rows]
        values = [float(row[value_column]) for row in rows]
    except (TypeError, ValueError) as exc:
        raise ValueError("Consolidation CSV contains a nonnumeric value.") from exc
    return _spacetime_observations(points, values)


class ConsolidationFamilyOperator(nn.Module):
    """DeepONet for the transient consolidation family ``(cv,U0) -> u(z,t)``."""

    def __init__(
        self,
        *,
        coefficient_bounds: tuple[float, float] = (0.05, 0.8),
        amplitude_bounds: tuple[float, float] = (0.5, 2.0),
        latent_dim: int = 24,
        hidden_layers: Sequence[int] = (32, 32),
    ) -> None:
        super().__init__()
        self.coefficient_bounds = tuple(map(float, coefficient_bounds))
        self.amplitude_bounds = tuple(map(float, amplitude_bounds))
        if not (0 < self.coefficient_bounds[0] < self.coefficient_bounds[1]):
            raise ValueError("invalid consolidation-coefficient family bounds.")
        if not (0 < self.amplitude_bounds[0] < self.amplitude_bounds[1]):
            raise ValueError("invalid initial-pressure family bounds.")
        self.operator = DeepONet(DeepONetConfig(
            n_sensors=2, n_dims=2, n_fields=1, latent_dim=int(latent_dim),
            branch_hidden=tuple(hidden_layers), trunk_hidden=tuple(hidden_layers), activation="tanh",
        ))

    def condition(self, coefficient: Tensor, amplitude: Tensor) -> Tensor:
        c0, c1 = self.coefficient_bounds
        a0, a1 = self.amplitude_bounds
        scaled_c = 2.0 * (torch.log(coefficient) - math.log(c0)) / (math.log(c1) - math.log(c0)) - 1.0
        scaled_a = 2.0 * (amplitude - a0) / (a1 - a0) - 1.0
        return torch.stack((scaled_c, scaled_a), dim=-1)

    def evaluate_family(self, coefficient: Tensor, amplitude: Tensor, coordinates: Tensor) -> Tensor:
        raw = self.operator.evaluate(self.condition(coefficient, amplitude), coordinates)
        drainage_shape = torch.sin(math.pi * coordinates[:, 0]).reshape(1, -1, 1)
        return drainage_shape * raw

    def predict(self, coordinates: Tensor, *, coefficient: Tensor | float, amplitude: Tensor | float) -> Tensor:
        cv = torch.as_tensor(coefficient, dtype=coordinates.dtype, device=coordinates.device).reshape(1)
        initial = torch.as_tensor(amplitude, dtype=coordinates.dtype, device=coordinates.device).reshape(1)
        return self.evaluate_family(cv, initial, coordinates)[0]


@dataclass
class ConsolidationOperatorFamilyResult:
    model: ConsolidationFamilyOperator
    history: list[dict[str, float]]


@dataclass
class ConsolidationOperatorInverseResult:
    estimated_coefficient: float
    history: list[dict[str, float]]
    predictions: Tensor


def train_consolidation_family_operator(
    *,
    coefficient_bounds: tuple[float, float] = (0.05, 0.8),
    amplitude_bounds: tuple[float, float] = (0.5, 2.0),
    family_batch_size: int = 48,
    query_points: int = 64,
    steps: int = 800,
    learning_rate: float = 2e-3,
    seed: int = 0,
) -> ConsolidationOperatorFamilyResult:
    """Train DeepONet over a continuously sampled Terzaghi problem family."""

    if family_batch_size < 2 or query_points < 4 or steps < 1 or learning_rate <= 0:
        raise ValueError("family_batch_size>=2, query_points>=4, steps>=1 and learning_rate>0 are required.")
    torch.manual_seed(int(seed))
    model = ConsolidationFamilyOperator(
        coefficient_bounds=coefficient_bounds, amplitude_bounds=amplitude_bounds
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    c0, c1 = map(float, coefficient_bounds)
    a0, a1 = map(float, amplitude_bounds)
    history: list[dict[str, float]] = []
    for step in range(int(steps)):
        coordinates = torch.rand(int(query_points), 2)
        log_c = math.log(c0) + torch.rand(int(family_batch_size)) * (math.log(c1) - math.log(c0))
        coefficient = torch.exp(log_c)
        amplitude = a0 + torch.rand(int(family_batch_size)) * (a1 - a0)
        prediction = model.evaluate_family(coefficient, amplitude, coordinates)
        target = consolidation_pressure(
            coordinates.reshape(1, -1, 2),
            coefficient.reshape(-1, 1, 1), amplitude.reshape(-1, 1, 1),
        )
        loss = (prediction - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or step == steps - 1 or (step + 1) % max(1, steps // 20) == 0:
            history.append({"step": float(step + 1), "family_data_loss": float(loss.detach())})
    return ConsolidationOperatorFamilyResult(model, history)


def inverse_consolidation_operator(
    family: ConsolidationOperatorFamilyResult | ConsolidationFamilyOperator,
    coordinates: Tensor | Sequence[Sequence[float]],
    values: Tensor | Sequence[float],
    *,
    amplitude: float = 1.0,
    initial_coefficient: float = 0.2,
    steps: int = 300,
    learning_rate: float = 3e-2,
) -> ConsolidationOperatorInverseResult:
    """Freeze the family operator and infer one member's consolidation coefficient."""

    model = family.model if isinstance(family, ConsolidationOperatorFamilyResult) else family
    observed_points, observed_pressure = _spacetime_observations(coordinates, values)
    lower, upper = model.coefficient_bounds
    raw = nn.Parameter(_bounded_raw(initial_coefficient, lower, upper, name="consolidation coefficient"))
    previous = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam([raw], lr=float(learning_rate))
    history: list[dict[str, float]] = []
    try:
        for step in range(int(steps)):
            coefficient = _bounded_value(raw, lower, upper)
            prediction = model.predict(observed_points, coefficient=coefficient, amplitude=float(amplitude))
            loss = (prediction - observed_pressure).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 0 or step == steps - 1 or (step + 1) % max(1, steps // 20) == 0:
                history.append({"step": float(step + 1), "observation_loss": float(loss.detach()), "coefficient": float(coefficient.detach())})
    finally:
        for parameter, requires_grad in zip(model.parameters(), previous, strict=True):
            parameter.requires_grad_(requires_grad)
    estimated = float(_bounded_value(raw, lower, upper).detach())
    with torch.no_grad():
        predictions = model.predict(observed_points, coefficient=estimated, amplitude=float(amplitude)).detach()
    return ConsolidationOperatorInverseResult(estimated, history, predictions)
