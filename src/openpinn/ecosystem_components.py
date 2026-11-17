"""Executable reference components for the five-axis method catalogue.

Each class below runs through the public component protocol.  The formulas are
the ones implemented in this file; they are small enough to audit and are not
presented as equivalent to a full paper algorithm.  The repository does not
contain a paper section that defines this catalogue.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from .autodiff import AutodiffDerivative
from .geometry import Interval, RectangleWithCircularHoles, SpaceTimeDomain
from .losses import LossComposer
from .optimizers import OptimizerAdapter
from .operators.grid_derivative import FiniteDifferenceGridDerivative
from .problems import PDEProblem
from .registry import ComponentRegistry
from .sampling import SampleBatch, UniformSampler


def _options(config: Mapping[str, Any] | None) -> dict[str, Any]:
    values = dict(config or {})
    values.pop("name", None)
    values.pop("kind", None)
    nested = values.pop("options", None)
    if nested is not None:
        values.update(dict(nested))
    return values


def _require_known(values: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise ValueError(
            f"{label} received unsupported option(s): " + ", ".join(unknown)
        )


class GraphKernelPINN(nn.Module):
    """Coordinate graph neural operator used as a pointwise PINN backbone."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        width: int = 48,
        layers: int = 3,
        length_scale: float = 1.0,
        multipole: bool = False,
    ) -> None:
        super().__init__()
        if min(input_dim, output_dim, width, layers) <= 0 or length_scale <= 0:
            raise ValueError("GraphKernelPINN dimensions and length_scale must be positive.")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.length_scale = float(length_scale)
        self.multipole = bool(multipole)
        self.lift = nn.Linear(input_dim, width)
        self.local = nn.ModuleList(nn.Linear(width, width) for _ in range(layers))
        self.message = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(layers))
        self.project = nn.Linear(width, output_dim)

    def forward(self, coordinates: Tensor) -> Tensor:
        if coordinates.ndim != 2 or coordinates.shape[1] != self.input_dim:
            raise ValueError("GraphKernelPINN expects (n_points, input_dim) coordinates.")
        features = torch.tanh(self.lift(coordinates))
        difference = coordinates[:, None, :] - coordinates[None, :, :]
        distance2 = difference.square().sum(dim=-1)
        kernel = torch.exp(-distance2 / (2.0 * self.length_scale**2))
        kernel = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1e-12)
        for local, message in zip(self.local, self.message, strict=True):
            aggregate = kernel @ message(features)
            if self.multipole:
                # A coarse global level supplies the long-range multipole term.
                aggregate = aggregate + message(features.mean(dim=0, keepdim=True))
            features = torch.tanh(local(features) + aggregate)
        return self.project(features)


class LowRankKernelPINN(nn.Module):
    """Low-rank separable kernel coordinate network."""

    def __init__(self, input_dim: int, output_dim: int, *, rank: int = 32) -> None:
        super().__init__()
        if min(input_dim, output_dim, rank) <= 0:
            raise ValueError("LowRankKernelPINN dimensions must be positive.")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.left = nn.Linear(input_dim, rank)
        self.right = nn.Linear(input_dim, rank)
        self.project = nn.Linear(rank, output_dim)

    def forward(self, coordinates: Tensor) -> Tensor:
        basis = torch.sin(self.left(coordinates)) * torch.cos(self.right(coordinates))
        return self.project(basis)


class PhysicsInformedKAN(nn.Module):
    """KAN-style network with trainable univariate Gaussian basis functions."""

    def __init__(
        self, input_dim: int, output_dim: int, *, grid_size: int = 12, width: int = 32
    ) -> None:
        super().__init__()
        if min(input_dim, output_dim, grid_size, width) <= 0:
            raise ValueError("PhysicsInformedKAN dimensions must be positive.")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        centers = torch.linspace(-1.0, 1.0, grid_size).repeat(input_dim, 1)
        self.centers = nn.Parameter(centers)
        self.log_widths = nn.Parameter(torch.zeros(input_dim, grid_size))
        self.mix = nn.Linear(input_dim * grid_size, width)
        self.output = nn.Linear(width, output_dim)

    def forward(self, coordinates: Tensor) -> Tensor:
        delta = coordinates.unsqueeze(-1) - self.centers.unsqueeze(0)
        widths = self.log_widths.clamp(-8.0, 8.0).exp()
        basis = torch.exp(-delta.square() * widths.unsqueeze(0))
        return self.output(torch.tanh(self.mix(basis.flatten(1))))


class DesignedPointSampler(UniformSampler):
    """Uniform boundary sampling with a deterministic interior point design."""

    def __init__(self, *args: Any, design: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if design not in {"grid", "lhs", "qmc"}:
            raise ValueError("design must be grid, lhs, or qmc.")
        self.design = design
        self._draw_index = 0

    def _unit(self, count: int, dimension: int, dtype: torch.dtype) -> Tensor:
        if count == 0:
            return torch.empty((0, dimension), dtype=dtype)
        if self.design == "qmc":
            # Unscrambled Sobol is deterministic and does not import SciPy.
            # ``seed`` only affects scrambled sequences, so it is not passed here.
            engine = torch.quasirandom.SobolEngine(dimension, scramble=False)
            drawn = engine.draw(self._draw_index + count)
            return drawn[self._draw_index :].to(dtype=dtype)
        if self.design == "lhs":
            columns = []
            for _ in range(dimension):
                permutation = torch.randperm(count, generator=self._generator)
                jitter = torch.rand(count, generator=self._generator, dtype=dtype)
                columns.append((permutation.to(dtype) + jitter) / count)
            return torch.stack(columns, dim=1)
        if dimension == 1:
            return ((torch.arange(count, dtype=dtype) + 0.5) / count).reshape(-1, 1)
        side = max(2, math.ceil(count ** (1.0 / dimension)))
        axes = [((torch.arange(side, dtype=dtype) + 0.5) / side)] * dimension
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, dimension)[:count]

    def _map_interior(self, domain: Any, count: int, dtype: torch.dtype) -> Tensor:
        if isinstance(domain, Interval):
            unit = self._unit(count, 1, dtype)
            return domain.lower + (domain.upper - domain.lower) * unit
        if isinstance(domain, SpaceTimeDomain):
            space = domain.space
            if isinstance(space, Interval):
                unit = self._unit(count, 2, dtype)
                x = space.lower + (space.upper - space.lower) * unit[:, :1]
                t = domain.time.lower + (domain.time.upper - domain.time.lower) * unit[:, 1:2]
                return torch.cat((x, t), dim=1)
        if isinstance(domain, RectangleWithCircularHoles) and not domain.holes:
            unit = self._unit(count, 2, dtype)
            lower = unit.new_tensor(domain.lower)
            upper = unit.new_tensor(domain.upper)
            return lower + (upper - lower) * unit
        # General curved/holed domains keep the design contract for conditions
        # while using the geometry's rejection sampler for valid interiors.
        return domain.sample_interior(count, generator=self._generator, dtype=dtype)

    def sample(self, problem=None, *, device=None, dtype=None, requires_grad=True):
        batch = super().sample(problem, device=device, dtype=dtype, requires_grad=requires_grad)
        selected = problem if problem is not None else self.problem
        domain = selected.domain if isinstance(selected, PDEProblem) else selected
        target_dtype = self.dtype if dtype is None else dtype
        interior = self._map_interior(domain, self.n_interior, target_dtype).to(
            device=self.device if device is None else device
        )
        self._draw_index += self.n_interior
        return SampleBatch(
            interior.detach().requires_grad_(requires_grad), batch.boundary, batch.initial
        )

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update({"design": self.design, "draw_index": self._draw_index})
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state["design"] != self.design:
            raise ValueError("Point design differs from checkpoint.")
        super().load_state_dict(state)
        self._draw_index = int(state.get("draw_index", 0))


class IndicatorAdaptiveSampler(UniformSampler):
    """Shared engine for RAR, ensemble-uncertainty, and DAS sampling."""

    def __init__(
        self,
        problem: PDEProblem,
        *args: Any,
        strategy: str,
        candidate_pool: int = 512,
        enrich: int = 16,
        ensemble_size: int = 4,
        density_steps: int = 12,
        **kwargs: Any,
    ) -> None:
        super().__init__(problem, *args, **kwargs)
        if strategy not in {"rar", "uncertainty", "das"}:
            raise ValueError("Unknown indicator-adaptive strategy.")
        if candidate_pool < self.n_interior or enrich <= 0:
            raise ValueError("candidate_pool must cover the batch and enrich must be positive.")
        self.strategy = strategy
        self.candidate_pool = int(candidate_pool)
        self.enrich = int(enrich)
        self.ensemble_size = int(ensemble_size)
        self.density_steps = int(density_steps)
        self._selected: Tensor | None = None
        self._updates = 0

    def _residual_scores(self, model: nn.Module, points: Tensor) -> Tensor:
        prediction = model(points)
        residuals = self.problem.evaluate_residual(points, prediction)
        values = torch.cat([item.reshape(item.shape[0], -1) for item in residuals.values()], dim=1)
        return values.square().sum(dim=1).sqrt()

    def _uncertainty_scores(self, model: nn.Module, points: Tensor) -> Tensor:
        from torch.func import functional_call

        parameters = {
            name: value.detach() for name, value in model.named_parameters()
        }
        state = {
            name: value.detach() for name, value in model.named_buffers()
        }
        predictions = []
        for index in range(self.ensemble_size):
            generator = torch.Generator(device="cpu").manual_seed(
                self.seed + 7919 * (self._updates + 1) + index
            )
            perturbed = dict(state)
            perturbed.update(
                {
                    name: value
                    + 1e-3
                    * torch.randn(
                        value.shape, generator=generator, dtype=value.dtype
                    ).to(value.device)
                    for name, value in parameters.items()
                }
            )
            predictions.append(functional_call(model, perturbed, (points,)).detach())
        return torch.stack(predictions).var(dim=0, unbiased=False).mean(dim=1).sqrt()

    def update(self, model: nn.Module, *, device=None, dtype=None) -> None:
        target_dtype = self.dtype if dtype is None else dtype
        target_device = self.device if device is None else device
        candidates = self.problem.domain.sample_interior(
            self.candidate_pool,
            generator=self._generator,
            dtype=target_dtype,
            device=target_device,
        ).detach().requires_grad_(True)
        scores = (
            self._uncertainty_scores(model, candidates)
            if self.strategy == "uncertainty"
            else self._residual_scores(model, candidates)
        ).detach()
        if self.strategy == "das":
            density = nn.Sequential(
                nn.Linear(candidates.shape[1], 24), nn.Tanh(), nn.Linear(24, 1)
            ).to(device=target_device, dtype=target_dtype)
            optimizer = torch.optim.Adam(density.parameters(), lr=2e-2)
            target = scores / scores.mean().clamp_min(1e-12)
            for _ in range(self.density_steps):
                optimizer.zero_grad(set_to_none=True)
                predicted = torch.nn.functional.softplus(density(candidates.detach()).squeeze(1))
                loss = (predicted - target).square().mean()
                loss.backward()
                optimizer.step()
            scores = torch.nn.functional.softplus(density(candidates.detach()).squeeze(1)).detach()
        if self.strategy == "rar":
            count = min(
                self.n_interior if self._selected is None else self.enrich,
                int(candidates.shape[0]),
            )
            new = candidates.detach()[torch.topk(scores, count).indices]
            if self._selected is None:
                chosen = new[: self.n_interior]
            else:
                retained_count = max(0, self.n_interior - int(new.shape[0]))
                retained = self._selected.to(device=new.device, dtype=new.dtype)
                if retained.shape[0] > retained_count:
                    retained = retained[-retained_count:]
                chosen = torch.cat((retained, new), dim=0)
            self._selected = chosen.detach().cpu()
        else:
            probability = (scores + 1e-12) / (scores.sum() + 1e-12 * scores.numel())
            indices = torch.multinomial(
                probability.cpu(),
                self.n_interior,
                replacement=False,
                generator=self._generator,
            )
            self._selected = candidates.detach().cpu()[indices]
        self._updates += 1

    def sample(self, problem=None, *, device=None, dtype=None, requires_grad=True):
        batch = super().sample(problem, device=device, dtype=dtype, requires_grad=requires_grad)
        if self._selected is None:
            self._selected = batch.interior.detach().cpu()
            return batch
        interior = self._selected.to(
            device=self.device if device is None else device,
            dtype=self.dtype if dtype is None else dtype,
        ).detach().requires_grad_(requires_grad)
        return SampleBatch(interior, batch.boundary, batch.initial)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update({"strategy": self.strategy, "selected": self._selected, "updates": self._updates})
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state["strategy"] != self.strategy:
            raise ValueError("Adaptive strategy differs from checkpoint.")
        super().load_state_dict(state)
        self._selected = state.get("selected")
        self._updates = int(state.get("updates", 0))


class GradientStatisticLoss(LossComposer):
    """Balance named losses with a momentum average of one gradient statistic.

    ``gradient`` uses the mean absolute parameter gradient.  ``ntk`` uses the
    squared Euclidean norm of that same loss gradient.  ``inverse_dirichlet``
    uses the variance of the parameter gradient.  None of these statistics is
    an NTK eigenvalue or a learned precision parameter.
    """

    def __init__(self, weights=None, *, rule: str, momentum: float = 0.9, reduction="sum"):
        super().__init__(weights, reduction=reduction)
        if rule not in {"gradient", "ntk", "inverse_dirichlet"}:
            raise ValueError("Unknown gradient statistic rule.")
        self.rule = rule
        self.momentum = float(momentum)
        self._parameters: tuple[nn.Parameter, ...] = ()
        self._dynamic: dict[str, float] = {}

    def bind_model(self, model: nn.Module) -> None:
        self._parameters = tuple(item for item in model.parameters() if item.requires_grad)

    def _statistic(self, term: Tensor) -> Tensor:
        gradients = torch.autograd.grad(
            term, self._parameters, retain_graph=True, create_graph=False, allow_unused=True
        )
        flat_items = [
            item.detach().reshape(-1) for item in gradients if item is not None
        ]
        if not flat_items:
            return term.detach().new_tensor(1.0)
        flat = torch.cat(flat_items)
        if not torch.isfinite(flat).all():
            raise FloatingPointError("Gradient-statistic loss received a non-finite gradient.")
        if self.rule == "gradient":
            return flat.abs().mean()
        if self.rule == "ntk":
            return flat.square().sum()
        return flat.var(unbiased=False).clamp_min(1e-12)

    def weighted_terms(self, terms: Mapping[str, Tensor]) -> dict[str, Tensor]:
        if not self._parameters:
            return super().weighted_terms(terms)
        if not torch.is_grad_enabled():
            return {
                name: value * self.weights.get(name, 1.0) * self._dynamic.get(name, 1.0)
                for name, value in terms.items()
            }
        statistics = {name: self._statistic(value) for name, value in terms.items()}
        target = torch.stack(tuple(statistics.values())).mean()
        for name, statistic in statistics.items():
            ratio = (target / statistic.clamp_min(1e-12)).detach()
            if not torch.isfinite(ratio):
                raise FloatingPointError(
                    f"Gradient-statistic weight for {name!r} is not finite."
                )
            proposed = float(ratio.clamp(1e-6, 1e6).cpu())
            previous = self._dynamic.get(name, proposed)
            self._dynamic[name] = self.momentum * previous + (1.0 - self.momentum) * proposed
        return {
            name: value * self.weights.get(name, 1.0) * self._dynamic[name]
            for name, value in terms.items()
        }

    def training_state_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "dynamic_weights": dict(self._dynamic)}

    def load_training_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("rule") != self.rule:
            raise ValueError("Gradient-statistic loss rule differs from checkpoint.")
        self._dynamic = {
            str(name): float(value)
            for name, value in dict(state.get("dynamic_weights", {})).items()
        }


class CausalLossComposer(LossComposer):
    """Weight pointwise residuals by earlier residuals in time.

    The trainer must call :meth:`bind_batch` with the current space-time
    batch.  PDE and boundary terms are sorted by the last coordinate, which
    is time on a :class:`~openpinn.geometry.SpaceTimeDomain`.  The initial
    condition is one time slice, so it keeps uniform weights.  A static
    domain is rejected: ordering its arbitrary collocation list would not be
    a temporal causal weight.
    """

    requires_pointwise_terms = True

    def __init__(self, weights=None, *, epsilon: float = 1.0, reduction="sum"):
        super().__init__(weights, reduction=reduction)
        if epsilon <= 0:
            raise ValueError("causal epsilon must be positive.")
        self.epsilon = float(epsilon)
        self._batch: SampleBatch | None = None
        self._problem: PDEProblem | None = None

    def bind_batch(self, batch: SampleBatch, problem: PDEProblem) -> None:
        self._batch = batch
        self._problem = problem

    def _points_for(self, name: str, count: int) -> Tensor | None:
        if self._batch is None:
            raise RuntimeError("CausalLossComposer requires a bound training batch.")
        key = str(name)
        if key == "initial" or key.startswith("initial_"):
            if self._batch.initial.shape[0] != count:
                raise ValueError(
                    f"Causal term {key!r} has {count} values but the initial "
                    f"point set has {self._batch.initial.shape[0]}."
                )
            return None
        if key == "boundary" or key.startswith("boundary_"):
            points = self._batch.boundary
        else:
            points = self._batch.interior
        if points.shape[0] != count:
            raise ValueError(
                f"Causal term {key!r} has {count} values but the matching "
                f"point set has {points.shape[0]}."
            )
        return points

    def compose_pointwise(self, terms: Mapping[str, Tensor]) -> Tensor:
        if self._batch is None or self._problem is None:
            raise RuntimeError("CausalLossComposer requires a bound training batch.")
        if not isinstance(self._problem.domain, SpaceTimeDomain):
            raise ValueError(
                "Causal loss weighting requires a space-time problem. "
                "A static domain has no time coordinate to order."
            )
        if not terms:
            raise ValueError("At least one loss term is required.")
        totals = []
        for name, values in terms.items():
            flat = values.reshape(-1)
            points = self._points_for(str(name), int(flat.numel()))
            if points is None:
                ordered = flat
                causal = torch.ones_like(flat)
            else:
                order = torch.argsort(points[:, -1].detach())
                ordered = flat[order]
                prefix = torch.cumsum(ordered.detach(), dim=0) - ordered.detach()
                causal = torch.exp(-self.epsilon * prefix / max(1, ordered.numel()))
            totals.append(self.weights.get(str(name), 1.0) * (causal * ordered).mean())
        total = torch.stack(totals).sum()
        return total / len(totals) if self.reduction == "mean" else total


class DiagonalCurvatureAdapter(OptimizerAdapter):
    """Damped per-parameter normalization by squared gradients.

    With ``natural=False`` the curvature is the current squared gradient, so
    the step is ``-lr * g / (|g| + damping)``.  With ``natural=True`` the
    squared gradient is tracked by an exponential moving average and the step
    is ``-lr * g / (sqrt(v) + damping)``.  The curvature buffer is part of
    :meth:`state_dict`.
    """

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        *,
        lr: float = 1e-2,
        damping: float = 1e-4,
        beta: float = 0.95,
        natural: bool = False,
    ) -> None:
        params = list(parameters)
        super().__init__(torch.optim.SGD(params, lr=lr))
        self.damping = float(damping)
        self.beta = float(beta)
        self.natural = bool(natural)
        self._curvature = [torch.zeros_like(item) for item in params]

    def step(self, loss_closure):
        loss = self._backward_closure(loss_closure)
        lr = float(self.optimizer.param_groups[0]["lr"])
        with torch.no_grad():
            for parameter, curvature in zip(self.optimizer.param_groups[0]["params"], self._curvature, strict=True):
                if parameter.grad is None:
                    continue
                squared = parameter.grad.square()
                if self.natural:
                    curvature.mul_(self.beta).add_(squared, alpha=1.0 - self.beta)
                else:
                    curvature.copy_(squared)
                parameter.addcdiv_(parameter.grad, curvature.sqrt().add_(self.damping), value=-lr)
        return loss.detach()

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update({
            "curvature": [item.detach().clone() for item in self._curvature],
            "damping": self.damping,
            "beta": self.beta,
            "natural": self.natural,
        })
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if bool(state.get("natural")) != self.natural:
            raise ValueError("Curvature optimizer kind differs from checkpoint.")
        super().load_state_dict(state)
        restored = list(state.get("curvature", ()))
        if len(restored) != len(self._curvature):
            raise ValueError("Curvature optimizer parameter count differs from checkpoint.")
        for target, value in zip(self._curvature, restored, strict=True):
            if not isinstance(value, Tensor) or value.shape != target.shape:
                raise ValueError("Curvature optimizer state does not match this parameter.")
            target.copy_(value.detach().to(device=target.device, dtype=target.dtype))


class FiniteVolumeGridDerivative(FiniteDifferenceGridDerivative):
    """Periodic face-average derivative on a structured grid.

    The first derivative ``(face[i+1/2] - face[i-1/2]) / Δx`` with arithmetic
    face values is algebraically the same central difference as
    :class:`FiniteDifferenceGridDerivative`.  Endpoints wrap.  The second
    derivative reuses that central stencil.  ``solve()`` strong-form residuals
    do not call this object.
    """

    def partial(self, field: Tensor, axis: int, order: int = 1) -> Tensor:
        if order == 2:
            return super().partial(field, axis, order)
        dimension = 2 + axis
        spacing = self.spacing[axis] if isinstance(self.spacing, tuple) else self.spacing
        right_face = 0.5 * (field + torch.roll(field, -1, dims=dimension))
        left_face = 0.5 * (torch.roll(field, 1, dims=dimension) + field)
        return (right_face - left_face) / float(spacing)


class WeakVariationalDerivative(AutodiffDerivative):
    """Coordinate autodiff plus an explicit integration-by-parts sum.

    :meth:`grad` is ordinary coordinate autodiff.  :meth:`integrate_by_parts`
    evaluates ``-sum(flux * test_gradient * weights)`` for a caller that
    builds its own weak residual.  Selecting this name in the derivative
    registry does not change :func:`openpinn.solve` residuals.
    """

    def integrate_by_parts(self, flux: Tensor, test_gradient: Tensor, weights: Tensor) -> Tensor:
        return -(flux * test_gradient * weights).sum(dim=0)


class HybridADNumericalDerivative(AutodiffDerivative):
    """Coordinate autodiff and a separate periodic grid-difference path.

    :meth:`grad` delegates to coordinate autodiff.  :meth:`partial` delegates
    to :class:`FiniteDifferenceGridDerivative`.  The two paths are not added
    or otherwise coupled inside one residual.
    """

    def __init__(self, spacing: float | Sequence[float] = 1.0) -> None:
        self.grid = FiniteDifferenceGridDerivative(spacing)

    def partial(self, field: Tensor, axis: int, order: int = 1) -> Tensor:
        return self.grid.partial(field, axis, order)


def _hidden_widths(hidden: Any, depth: Any) -> list[int] | None:
    if hidden is None:
        return None
    if isinstance(hidden, int):
        repeats = 3 if depth is None else int(depth)
        if repeats <= 0:
            raise ValueError("model depth must be positive.")
        return [int(hidden)] * repeats
    return [int(item) for item in hidden]


def _model_factory(kind: str):
    def factory(*, config, problem, registry):
        from .experiment import _constraint, _feature, _feature_output_dimension
        from .models import PINNModel, freeze_all_but_last_linear_layers

        model_config = dict(config)
        values = _options(config)
        input_dim = int(values.pop("input_dim", problem.domain.dim))
        if input_dim != problem.domain.dim:
            raise ValueError(
                f"model.input_dim={input_dim} does not match problem dimension "
                f"{problem.domain.dim}."
            )
        output_dim = int(values.pop("output_dim", problem.output_dim))
        hidden = values.pop("hidden_layers", values.pop("layers", None))
        depth = values.pop("depth", None)
        for unused in (
            "activation",
            "activation_options",
            "initialization",
            "bias",
            "feature",
            "feature_transform",
            "output_transform",
            "output",
            "constraint",
            "multipole",
        ):
            values.pop(unused, None)
        trainable_last = values.pop("trainable_last_linear_layers", None)
        widths = _hidden_widths(hidden, depth)
        if widths and kind in {"gno", "multipole_gno"}:
            values.setdefault("width", widths[0])
            values.setdefault("layers", len(widths))
        elif widths and kind == "kan":
            values.setdefault("width", widths[0])
        feature = _feature(model_config, problem.domain.dim, registry)
        network_input = _feature_output_dimension(feature, problem.domain.dim)
        if kind in {"gno", "multipole_gno"}:
            _require_known(values, {"width", "layers", "length_scale"}, "Graph kernel model")
            network = GraphKernelPINN(
                network_input,
                output_dim,
                multipole=kind == "multipole_gno",
                **values,
            )
        elif kind == "low_rank_operator":
            _require_known(values, {"rank"}, "Low-rank coordinate model")
            network = LowRankKernelPINN(network_input, output_dim, **values)
        else:
            _require_known(values, {"grid_size", "width"}, "Gaussian basis model")
            network = PhysicsInformedKAN(network_input, output_dim, **values)
        model = PINNModel(
            network,
            feature_transform=feature,
            output_transform=_constraint(model_config, problem, registry),
        )
        if trainable_last is not None:
            freeze_all_but_last_linear_layers(model, int(trainable_last))
        return model

    return factory


def _sampler_factory(kind: str):
    def factory(*, config, problem, seed, device, dtype):
        values = _options(config)
        counts = {
            "n_interior": int(values.pop("interior", values.pop("n_interior", 128))),
            "n_boundary": int(values.pop("boundary", values.pop("n_boundary", 32))),
            "n_initial": int(values.pop("initial", values.pop("n_initial", 32))),
            "seed": int(values.pop("seed", seed)),
            "device": device,
            "dtype": dtype,
        }
        if kind in {"grid", "lhs", "qmc"}:
            _require_known(values, set(), f"{kind} sampler")
            return DesignedPointSampler(problem, design=kind, **counts)
        _require_known(
            values,
            {"candidate_pool", "enrich", "ensemble_size", "density_steps"},
            f"{kind} sampler",
        )
        return IndicatorAdaptiveSampler(problem, strategy=kind, **counts, **values)

    return factory


def _loss_factory(rule: str):
    def factory(*, config, weights=None, reduction="sum"):
        values = _options(config)
        if rule == "causal":
            _require_known(values, {"epsilon"}, "causal loss")
            return CausalLossComposer(weights, reduction=reduction, **values)
        _require_known(values, {"momentum"}, f"{rule} loss")
        return GradientStatisticLoss(weights, rule=rule, reduction=reduction, **values)

    return factory


def _optimizer_factory(natural: bool):
    def factory(*, config, parameters):
        values = _options(config)
        if "lr" in values and "learning_rate" in values:
            raise ValueError("Pass optimizer.lr or optimizer.learning_rate, not both.")
        lr = float(values.pop("lr", values.pop("learning_rate", 1e-2)))
        _require_known(values, {"damping", "beta"}, "curvature optimizer")
        return DiagonalCurvatureAdapter(parameters, lr=lr, natural=natural, **values)

    return factory


def _spacing_from_config(config: Mapping[str, Any] | None) -> Any:
    values = _options(config)
    spacing = values.pop("spacing", 1.0)
    _require_known(values, set(), "grid derivative")
    return spacing


def register_method_ecosystem_builtins(registry: ComponentRegistry) -> None:
    """Register the executable five-axis reference components."""

    registry.register("model", "gno", _model_factory("gno"), aliases=("graph_operator",))
    registry.register("model", "multipole_gno", _model_factory("multipole_gno"), aliases=("mgno",))
    registry.register("model", "low_rank_operator", _model_factory("low_rank_operator"), aliases=("low_rank_kernel",))
    registry.register("model", "physics_informed_kan", _model_factory("kan"), aliases=("pikan", "kan"))
    for name in ("grid", "lhs", "qmc", "rar", "uncertainty", "das"):
        registry.register("sampler", name, _sampler_factory(name))
    registry.register("loss", "gradient_balanced", _loss_factory("gradient"), aliases=("learning_rate_annealing",))
    registry.register("loss", "ntk_balanced", _loss_factory("ntk"), aliases=("ntk_weighting",))
    registry.register("loss", "inverse_dirichlet", _loss_factory("inverse_dirichlet"))
    registry.register("loss", "causal", _loss_factory("causal"), aliases=("causal_weighting",))
    registry.register("optimizer", "gauss_newton", _optimizer_factory(False), aliases=("gn",))
    registry.register("optimizer", "natural_gradient", _optimizer_factory(True), aliases=("ngd",))
    registry.register(
        "derivative",
        "finite_volume",
        lambda **kwargs: FiniteVolumeGridDerivative(_spacing_from_config(kwargs.get("config"))),
        aliases=("fv",),
    )
    registry.register(
        "derivative",
        "weak_variational",
        lambda **kwargs: WeakVariationalDerivative(),
        aliases=("vpinn", "weak"),
    )
    registry.register(
        "derivative",
        "hybrid_ad_numerical",
        lambda **kwargs: HybridADNumericalDerivative(_spacing_from_config(kwargs.get("config"))),
        aliases=("can_pinn", "hybrid"),
    )


__all__ = [
    "CausalLossComposer", "DiagonalCurvatureAdapter", "GraphKernelPINN",
    "HybridADNumericalDerivative", "IndicatorAdaptiveSampler", "LowRankKernelPINN",
    "PhysicsInformedKAN", "WeakVariationalDerivative", "register_method_ecosystem_builtins",
]
