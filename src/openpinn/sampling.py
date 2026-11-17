"""Reproducible uniform collocation sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from .geometry import Interval, RectangleWithCircularHoles, SpaceTimeDomain
from .problems import PDEProblem


@dataclass
class SampleBatch:
    """Collocation points grouped by their training role."""

    interior: Tensor
    boundary: Tensor
    initial: Tensor
    interior_weights: Tensor | None = None

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "SampleBatch":
        """Return a batch moved to a device/dtype with fresh leaf tensors."""

        def move(points: Tensor) -> Tensor:
            moved = points.detach().to(device=device, dtype=dtype)
            return moved.requires_grad_(points.requires_grad)

        weights = (
            self.interior_weights.detach().to(device=device, dtype=dtype)
            if self.interior_weights is not None
            else None
        )
        return SampleBatch(
            move(self.interior), move(self.boundary), move(self.initial), weights
        )

    def as_dict(self) -> dict[str, Tensor]:
        """Expose the three groups as a mapping."""

        values = {
            "interior": self.interior,
            "boundary": self.boundary,
            "initial": self.initial,
        }
        if self.interior_weights is not None:
            values["interior_weights"] = self.interior_weights
        return values


@runtime_checkable
class SamplerProtocol(Protocol):
    """Structural contract consumed by :class:`openpinn.Trainer`.

    Stateful samplers expose checkpoint state in addition to producing a
    :class:`SampleBatch`.  Adaptive and curriculum samplers can implement this
    contract without inheriting from :class:`UniformSampler`.
    """

    def sample(
        self,
        problem: PDEProblem | Interval | RectangleWithCircularHoles | SpaceTimeDomain | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        requires_grad: bool = True,
    ) -> SampleBatch:
        """Draw one batch compatible with the selected problem."""

        ...

    def state_dict(self) -> Mapping[str, Any]:
        """Return state required for deterministic checkpoint/resume."""

        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`."""

        ...


class UniformSampler:
    """Uniformly sample interior, spatial-boundary and initial points.

    The sampler is deliberately static and simple.  Residual-adaptive sampling
    is an advanced method and should be implemented as a separately tested
    extension rather than being implied by this class.
    """

    def __init__(
        self,
        problem: PDEProblem | Interval | RectangleWithCircularHoles | SpaceTimeDomain | None = None,
        n_interior: int = 128,
        n_boundary: int = 32,
        n_initial: int = 32,
        *,
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        interior: int | None = None,
        boundary: int | None = None,
        initial: int | None = None,
    ) -> None:
        self.problem = problem
        self.n_interior = int(n_interior if interior is None else interior)
        self.n_boundary = int(n_boundary if boundary is None else boundary)
        self.n_initial = int(n_initial if initial is None else initial)
        if min(self.n_interior, self.n_boundary, self.n_initial) < 0:
            raise ValueError("Sample counts must be non-negative.")
        self.seed = int(seed)
        self.device = device
        self.dtype = dtype
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)

    def reset(self, seed: int | None = None) -> None:
        """Reset the pseudo-random sequence."""

        if seed is not None:
            self.seed = int(seed)
        self._generator.manual_seed(self.seed)

    def state_dict(self) -> dict[str, Any]:
        """Return reproducibility-critical sampler state."""

        return {
            "seed": self.seed,
            "n_interior": self.n_interior,
            "n_boundary": self.n_boundary,
            "n_initial": self.n_initial,
            "generator_state": self._generator.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a compatible sampler's pseudo-random stream."""

        expected = (self.n_interior, self.n_boundary, self.n_initial)
        restored = (
            int(state["n_interior"]),
            int(state["n_boundary"]),
            int(state["n_initial"]),
        )
        if restored != expected:
            raise ValueError(
                f"Sampler counts differ from checkpoint: expected {expected}, got {restored}."
            )
        self.seed = int(state["seed"])
        self._generator.set_state(state["generator_state"])

    def sample(
        self,
        problem: PDEProblem | Interval | RectangleWithCircularHoles | SpaceTimeDomain | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        requires_grad: bool = True,
    ) -> SampleBatch:
        """Draw one fresh batch."""

        selected = problem if problem is not None else self.problem
        if selected is None:
            raise ValueError("A PDEProblem or domain is required for sampling.")
        domain = selected.domain if isinstance(selected, PDEProblem) else selected
        target_device = self.device if device is None else device
        target_dtype = self.dtype if dtype is None else dtype
        interior_points = domain.sample_interior(
            self.n_interior,
            generator=self._generator,
            dtype=target_dtype,
            device=target_device,
        )
        boundary_points = domain.sample_boundary(
            self.n_boundary,
            generator=self._generator,
            dtype=target_dtype,
            device=target_device,
        )
        if isinstance(domain, SpaceTimeDomain):
            initial_points = domain.sample_initial(
                self.n_initial,
                generator=self._generator,
                dtype=target_dtype,
                device=target_device,
            )
        else:
            initial_points = torch.empty(
                (0, domain.dim), dtype=target_dtype, device=target_device
            )

        return SampleBatch(
            interior_points.detach().requires_grad_(requires_grad),
            boundary_points.detach().requires_grad_(requires_grad),
            initial_points.detach().requires_grad_(requires_grad),
        )

    resample = sample


class ResidualAdaptiveSampler:
    """Resample interior points from a residual-weighted candidate pool.

    This independent minimal RAD-style sampler follows the residual-based
    distribution idea in Wu et al. (2023), DOI
    ``10.1016/j.cma.2022.115671``. Boundary and initial points remain uniform,
    and the number of interior points never changes.
    """

    def __init__(
        self,
        problem: PDEProblem,
        n_interior: int = 128,
        n_boundary: int = 32,
        n_initial: int = 32,
        *,
        candidate_pool: int = 2048,
        power: float = 1.0,
        uniform_mix: float = 0.1,
        epsilon: float = 1e-12,
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not isinstance(problem, PDEProblem):
            raise TypeError("ResidualAdaptiveSampler requires a PDEProblem.")
        if n_interior <= 0:
            raise ValueError("ResidualAdaptiveSampler requires positive interior count.")
        if candidate_pool < n_interior:
            raise ValueError("candidate_pool must be at least n_interior.")
        if power <= 0.0 or epsilon <= 0.0:
            raise ValueError("power and epsilon must be positive.")
        if not 0.0 <= uniform_mix <= 1.0:
            raise ValueError("uniform_mix must lie in [0, 1].")
        self.problem = problem
        self.n_interior = int(n_interior)
        self.n_boundary = int(n_boundary)
        self.n_initial = int(n_initial)
        self.candidate_pool = int(candidate_pool)
        self.power = float(power)
        self.uniform_mix = float(uniform_mix)
        self.epsilon = float(epsilon)
        self.seed = int(seed)
        self.device = device
        self.dtype = dtype
        self._uniform = UniformSampler(
            problem,
            n_interior=self.n_interior,
            n_boundary=self.n_boundary,
            n_initial=self.n_initial,
            seed=self.seed,
            device=device,
            dtype=dtype,
        )
        self._candidate_generator = torch.Generator(device="cpu")
        self._candidate_generator.manual_seed(self.seed + 104729)
        self._selected_interior: Tensor | None = None
        self._update_count = 0
        self._last_diagnostics: dict[str, float | int] = {
            "update_count": 0,
            "candidate_pool": self.candidate_pool,
        }

    def update(
        self,
        model: nn.Module,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Score a fresh candidate pool and cache the next interior batch."""

        target_device = self.device if device is None else device
        target_dtype = self.dtype if dtype is None else dtype
        candidates = self.problem.domain.sample_interior(
            self.candidate_pool,
            generator=self._candidate_generator,
            dtype=target_dtype,
            device=target_device,
        ).detach().requires_grad_(True)
        was_training = model.training
        model.eval()
        prediction = model(candidates)
        residuals = self.problem.evaluate_residual(candidates, prediction)
        columns = torch.cat(tuple(residuals.values()), dim=1)
        scores = torch.sqrt(torch.sum(columns.square(), dim=1) + self.epsilon).detach()
        model.train(was_training)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Residual-adaptive candidate scores contain NaN or Inf.")
        powered = torch.pow(scores.cpu() + self.epsilon, self.power)
        residual_probability = powered / powered.sum()
        uniform_probability = torch.full_like(
            residual_probability, 1.0 / self.candidate_pool
        )
        probability = (
            (1.0 - self.uniform_mix) * residual_probability
            + self.uniform_mix * uniform_probability
        )
        selected_indices = torch.multinomial(
            probability,
            self.n_interior,
            replacement=False,
            generator=self._candidate_generator,
        )
        self._selected_interior = candidates.detach().cpu()[selected_indices]
        selected_scores = scores.cpu()[selected_indices]
        self._update_count += 1
        self._last_diagnostics = {
            "update_count": self._update_count,
            "candidate_pool": self.candidate_pool,
            "candidate_mean_residual": float(scores.mean().cpu()),
            "candidate_max_residual": float(scores.max().cpu()),
            "selected_mean_residual": float(selected_scores.mean()),
            "selected_max_residual": float(selected_scores.max()),
        }

    def diagnostics(self) -> dict[str, float | int]:
        """Return the latest finite sampling diagnostics."""

        return dict(self._last_diagnostics)

    def sample(
        self,
        problem: PDEProblem | Interval | RectangleWithCircularHoles | SpaceTimeDomain | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        requires_grad: bool = True,
    ) -> SampleBatch:
        """Return cached adaptive interior points and uniform conditions."""

        selected_problem = problem if problem is not None else self.problem
        if selected_problem is not self.problem:
            raise ValueError("ResidualAdaptiveSampler cannot switch problems after construction.")
        batch = self._uniform.sample(
            self.problem,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        if self._selected_interior is None:
            return batch
        target_device = self.device if device is None else device
        target_dtype = self.dtype if dtype is None else dtype
        interior = self._selected_interior.to(
            device=target_device, dtype=target_dtype
        ).detach().requires_grad_(requires_grad)
        return SampleBatch(interior, batch.boundary, batch.initial)

    resample = sample

    def state_dict(self) -> dict[str, Any]:
        """Return both uniform and candidate random streams."""

        return {
            "seed": self.seed,
            "n_interior": self.n_interior,
            "n_boundary": self.n_boundary,
            "n_initial": self.n_initial,
            "candidate_pool": self.candidate_pool,
            "power": self.power,
            "uniform_mix": self.uniform_mix,
            "epsilon": self.epsilon,
            "uniform": self._uniform.state_dict(),
            "candidate_generator_state": self._candidate_generator.get_state(),
            "selected_interior": self._selected_interior,
            "update_count": self._update_count,
            "last_diagnostics": dict(self._last_diagnostics),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a configuration-compatible adaptive sampler."""

        expected = (
            self.n_interior,
            self.n_boundary,
            self.n_initial,
            self.candidate_pool,
            self.power,
            self.uniform_mix,
            self.epsilon,
        )
        restored = (
            int(state["n_interior"]),
            int(state["n_boundary"]),
            int(state["n_initial"]),
            int(state["candidate_pool"]),
            float(state["power"]),
            float(state["uniform_mix"]),
            float(state["epsilon"]),
        )
        if restored != expected:
            raise ValueError(
                f"Adaptive sampler settings differ from checkpoint: expected {expected}, got {restored}."
            )
        self.seed = int(state["seed"])
        self._uniform.load_state_dict(state["uniform"])
        self._candidate_generator.set_state(state["candidate_generator_state"])
        selected = state.get("selected_interior")
        self._selected_interior = selected.detach().cpu() if selected is not None else None
        self._update_count = int(state.get("update_count", 0))
        self._last_diagnostics = dict(state.get("last_diagnostics", {}))


class AdaptiveMovementSampler:
    """Keep part of the collocation set fixed and move the remaining points.

    This independently written implementation follows the RAM/WAM probability
    definitions in Hou et al. (2023), DOI ``10.1007/s11071-023-08654-w``.
    ``indicator='residual'`` selects RAM; ``indicator='solution_gradient'``
    selects WAM.  The total number of training points stays constant.
    """

    _INDICATORS = {"residual", "solution_gradient"}

    def __init__(
        self,
        problem: PDEProblem,
        n_interior: int = 128,
        n_boundary: int = 32,
        n_initial: int = 32,
        *,
        indicator: str,
        candidate_pool: int = 2048,
        movable_fraction: float = 0.5,
        power: float = 1.0,
        epsilon: float = 1e-12,
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not isinstance(problem, PDEProblem):
            raise TypeError("AdaptiveMovementSampler requires a PDEProblem.")
        if indicator not in self._INDICATORS:
            raise ValueError(
                "indicator must be 'residual' or 'solution_gradient'."
            )
        if n_interior <= 1:
            raise ValueError("Adaptive movement requires at least two interior points.")
        if candidate_pool < n_interior:
            raise ValueError("candidate_pool must be at least n_interior.")
        if not 0.0 < movable_fraction <= 1.0:
            raise ValueError("movable_fraction must lie in (0, 1].")
        if power <= 0.0 or epsilon <= 0.0:
            raise ValueError("power and epsilon must be positive.")
        self.problem = problem
        self.n_interior = int(n_interior)
        self.n_boundary = int(n_boundary)
        self.n_initial = int(n_initial)
        self.indicator = indicator
        self.candidate_pool = int(candidate_pool)
        self.movable_fraction = float(movable_fraction)
        self.movable_count = max(1, int(round(self.n_interior * movable_fraction)))
        self.fixed_count = self.n_interior - self.movable_count
        self.power = float(power)
        self.epsilon = float(epsilon)
        self.seed = int(seed)
        self.device = device
        self.dtype = dtype
        self._uniform = UniformSampler(
            problem,
            n_interior=self.n_interior,
            n_boundary=self.n_boundary,
            n_initial=self.n_initial,
            seed=self.seed,
            device=device,
            dtype=dtype,
        )
        self._candidate_generator = torch.Generator(device="cpu")
        self._candidate_generator.manual_seed(self.seed + 130363)
        self._selected_interior: Tensor | None = None
        self._update_count = 0
        self._last_diagnostics: dict[str, float | int | str] = {
            "update_count": 0,
            "indicator": self.indicator,
            "candidate_pool": self.candidate_pool,
            "fixed_count": self.fixed_count,
            "movable_count": self.movable_count,
        }

    def _scores(self, model: nn.Module, candidates: Tensor) -> Tensor:
        prediction = model(candidates)
        if self.indicator == "residual":
            residuals = self.problem.evaluate_residual(candidates, prediction)
            columns = torch.cat(
                [value.reshape(value.shape[0], -1) for value in residuals.values()],
                dim=1,
            )
            return torch.sqrt(torch.sum(columns.square(), dim=1) + self.epsilon)

        gradient_energy = torch.zeros(
            candidates.shape[0], device=candidates.device, dtype=candidates.dtype
        )
        for output_index in range(prediction.shape[1]):
            coordinate_gradient = torch.autograd.grad(
                prediction[:, output_index].sum(),
                candidates,
                retain_graph=output_index + 1 < prediction.shape[1],
                create_graph=False,
            )[0]
            gradient_energy = gradient_energy + torch.sum(
                coordinate_gradient.square(), dim=1
            )
        return torch.sqrt(1.0 + gradient_energy + self.epsilon)

    def update(
        self,
        model: nn.Module,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Move the configured subset according to the current indicator."""

        if self._selected_interior is None:
            raise RuntimeError("Call sample() once before updating movable points.")
        target_device = self.device if device is None else device
        target_dtype = self.dtype if dtype is None else dtype
        candidates = self.problem.domain.sample_interior(
            self.candidate_pool,
            generator=self._candidate_generator,
            dtype=target_dtype,
            device=target_device,
        ).detach().requires_grad_(True)
        was_training = model.training
        model.eval()
        scores = self._scores(model, candidates).detach()
        model.train(was_training)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Adaptive-movement scores contain NaN or Inf.")
        powered = torch.pow(scores.cpu() + self.epsilon, self.power)
        probability = powered / powered.sum()
        selected_indices = torch.multinomial(
            probability,
            self.movable_count,
            replacement=False,
            generator=self._candidate_generator,
        )
        fixed = self._selected_interior[: self.fixed_count]
        moved = candidates.detach().cpu()[selected_indices]
        self._selected_interior = torch.cat((fixed, moved), dim=0)
        selected_scores = scores.cpu()[selected_indices]
        self._update_count += 1
        self._last_diagnostics = {
            "update_count": self._update_count,
            "indicator": self.indicator,
            "candidate_pool": self.candidate_pool,
            "fixed_count": self.fixed_count,
            "movable_count": self.movable_count,
            "candidate_mean_indicator": float(scores.mean().cpu()),
            "candidate_max_indicator": float(scores.max().cpu()),
            "moved_mean_indicator": float(selected_scores.mean()),
            "moved_max_indicator": float(selected_scores.max()),
        }

    def diagnostics(self) -> dict[str, float | int | str]:
        """Return the latest movement summary."""

        return dict(self._last_diagnostics)

    def sample(
        self,
        problem: PDEProblem | Interval | RectangleWithCircularHoles | SpaceTimeDomain | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        requires_grad: bool = True,
    ) -> SampleBatch:
        """Return the current fixed/movable interior set and fresh condition points."""

        selected_problem = problem if problem is not None else self.problem
        if selected_problem is not self.problem:
            raise ValueError("AdaptiveMovementSampler cannot switch problems after construction.")
        batch = self._uniform.sample(
            self.problem,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        if self._selected_interior is None:
            self._selected_interior = batch.interior.detach().cpu()
            return batch
        target_device = self.device if device is None else device
        target_dtype = self.dtype if dtype is None else dtype
        interior = self._selected_interior.to(
            device=target_device, dtype=target_dtype
        ).detach().requires_grad_(requires_grad)
        return SampleBatch(interior, batch.boundary, batch.initial)

    resample = sample

    def state_dict(self) -> dict[str, Any]:
        """Return deterministic sampler and movement state."""

        return {
            "seed": self.seed,
            "n_interior": self.n_interior,
            "n_boundary": self.n_boundary,
            "n_initial": self.n_initial,
            "indicator": self.indicator,
            "candidate_pool": self.candidate_pool,
            "movable_fraction": self.movable_fraction,
            "power": self.power,
            "epsilon": self.epsilon,
            "uniform": self._uniform.state_dict(),
            "candidate_generator_state": self._candidate_generator.get_state(),
            "selected_interior": self._selected_interior,
            "update_count": self._update_count,
            "last_diagnostics": dict(self._last_diagnostics),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a configuration-compatible movement sampler."""

        expected = (
            self.n_interior,
            self.n_boundary,
            self.n_initial,
            self.indicator,
            self.candidate_pool,
            self.movable_fraction,
            self.power,
            self.epsilon,
        )
        restored = (
            int(state["n_interior"]),
            int(state["n_boundary"]),
            int(state["n_initial"]),
            str(state["indicator"]),
            int(state["candidate_pool"]),
            float(state["movable_fraction"]),
            float(state["power"]),
            float(state["epsilon"]),
        )
        if restored != expected:
            raise ValueError(
                "Adaptive movement settings differ from checkpoint: "
                f"expected {expected}, got {restored}."
            )
        self.seed = int(state["seed"])
        self._uniform.load_state_dict(state["uniform"])
        self._candidate_generator.set_state(state["candidate_generator_state"])
        selected = state.get("selected_interior")
        self._selected_interior = selected.detach().cpu() if selected is not None else None
        self._update_count = int(state.get("update_count", 0))
        self._last_diagnostics = dict(state.get("last_diagnostics", {}))


class ImportanceSamplingSampler:
    """Sample PDE points by loss and apply inverse-probability correction.

    The finite candidate-pool estimator follows Nabian et al. (2021), DOI
    ``10.1111/mice.12685``: points are sampled with probability proportional
    to their current PDE loss, and each selected contribution receives
    ``1 / (N q_i)`` so the mini-batch mean remains an unbiased estimator of
    the candidate-pool mean.  Boundary and initial losses remain uniform.
    """

    def __init__(
        self,
        problem: PDEProblem,
        n_interior: int = 128,
        n_boundary: int = 32,
        n_initial: int = 32,
        *,
        candidate_pool: int = 2048,
        epsilon: float = 1e-12,
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not isinstance(problem, PDEProblem):
            raise TypeError("ImportanceSamplingSampler requires a PDEProblem.")
        if n_interior <= 0:
            raise ValueError("Importance sampling requires positive interior count.")
        if candidate_pool < n_interior:
            raise ValueError("candidate_pool must be at least n_interior.")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        self.problem = problem
        self.n_interior = int(n_interior)
        self.n_boundary = int(n_boundary)
        self.n_initial = int(n_initial)
        self.candidate_pool = int(candidate_pool)
        self.epsilon = float(epsilon)
        self.seed = int(seed)
        self.device = device
        self.dtype = dtype
        self._uniform = UniformSampler(
            problem,
            n_interior=self.n_interior,
            n_boundary=self.n_boundary,
            n_initial=self.n_initial,
            seed=self.seed,
            device=device,
            dtype=dtype,
        )
        self._candidate_generator = torch.Generator(device="cpu")
        self._candidate_generator.manual_seed(self.seed + 169087)
        self._selected_interior: Tensor | None = None
        self._selected_weights: Tensor | None = None
        self._update_count = 0
        self._last_diagnostics: dict[str, float | int] = {
            "update_count": 0,
            "candidate_pool": self.candidate_pool,
        }

    def update(
        self,
        model: nn.Module,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Draw a corrected importance mini-batch from a fresh candidate pool."""

        target_device = self.device if device is None else device
        target_dtype = self.dtype if dtype is None else dtype
        candidates = self.problem.domain.sample_interior(
            self.candidate_pool,
            generator=self._candidate_generator,
            dtype=target_dtype,
            device=target_device,
        ).detach().requires_grad_(True)
        was_training = model.training
        model.eval()
        prediction = model(candidates)
        residuals = self.problem.evaluate_residual(candidates, prediction)
        columns = torch.cat(
            [value.reshape(value.shape[0], -1) for value in residuals.values()], dim=1
        )
        per_point_loss = torch.sum(columns.square(), dim=1).detach()
        model.train(was_training)
        if not torch.isfinite(per_point_loss).all():
            raise FloatingPointError("Importance-sampling losses contain NaN or Inf.")
        probability = per_point_loss.cpu() + self.epsilon
        probability = probability / probability.sum()
        selected_indices = torch.multinomial(
            probability,
            self.n_interior,
            replacement=True,
            generator=self._candidate_generator,
        )
        selected_probability = probability[selected_indices]
        correction = 1.0 / (self.candidate_pool * selected_probability)
        self._selected_interior = candidates.detach().cpu()[selected_indices]
        self._selected_weights = correction.reshape(-1, 1)
        self._update_count += 1
        self._last_diagnostics = {
            "update_count": self._update_count,
            "candidate_pool": self.candidate_pool,
            "candidate_mean_loss": float(per_point_loss.mean().cpu()),
            "candidate_max_loss": float(per_point_loss.max().cpu()),
            "min_probability": float(probability.min()),
            "max_probability": float(probability.max()),
            "mean_inverse_probability_weight": float(correction.mean()),
        }

    def diagnostics(self) -> dict[str, float | int]:
        """Return the latest sampling/correction summary."""

        return dict(self._last_diagnostics)

    def sample(
        self,
        problem: PDEProblem | Interval | RectangleWithCircularHoles | SpaceTimeDomain | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        requires_grad: bool = True,
    ) -> SampleBatch:
        """Return selected points and their inverse-probability weights."""

        selected_problem = problem if problem is not None else self.problem
        if selected_problem is not self.problem:
            raise ValueError("ImportanceSamplingSampler cannot switch problems after construction.")
        batch = self._uniform.sample(
            self.problem,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        if self._selected_interior is None:
            return batch
        target_device = self.device if device is None else device
        target_dtype = self.dtype if dtype is None else dtype
        interior = self._selected_interior.to(
            device=target_device, dtype=target_dtype
        ).detach().requires_grad_(requires_grad)
        weights = self._selected_weights.to(  # type: ignore[union-attr]
            device=target_device, dtype=target_dtype
        )
        return SampleBatch(interior, batch.boundary, batch.initial, weights)

    resample = sample

    def state_dict(self) -> dict[str, Any]:
        """Return deterministic selection and correction state."""

        return {
            "seed": self.seed,
            "n_interior": self.n_interior,
            "n_boundary": self.n_boundary,
            "n_initial": self.n_initial,
            "candidate_pool": self.candidate_pool,
            "epsilon": self.epsilon,
            "uniform": self._uniform.state_dict(),
            "candidate_generator_state": self._candidate_generator.get_state(),
            "selected_interior": self._selected_interior,
            "selected_weights": self._selected_weights,
            "update_count": self._update_count,
            "last_diagnostics": dict(self._last_diagnostics),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a configuration-compatible importance sampler."""

        expected = (
            self.n_interior,
            self.n_boundary,
            self.n_initial,
            self.candidate_pool,
            self.epsilon,
        )
        restored = (
            int(state["n_interior"]),
            int(state["n_boundary"]),
            int(state["n_initial"]),
            int(state["candidate_pool"]),
            float(state["epsilon"]),
        )
        if restored != expected:
            raise ValueError(
                "Importance sampler settings differ from checkpoint: "
                f"expected {expected}, got {restored}."
            )
        self.seed = int(state["seed"])
        self._uniform.load_state_dict(state["uniform"])
        self._candidate_generator.set_state(state["candidate_generator_state"])
        selected = state.get("selected_interior")
        weights = state.get("selected_weights")
        self._selected_interior = selected.detach().cpu() if selected is not None else None
        self._selected_weights = weights.detach().cpu() if weights is not None else None
        self._update_count = int(state.get("update_count", 0))
        self._last_diagnostics = dict(state.get("last_diagnostics", {}))


class TimeWindowSampler:
    """Uniformly sample an expanding sequence of time windows.

    This generic curriculum is intentionally distinct from the patented,
    non-commercial causal-loss algorithm in the CausalPINNs reference code.
    It only changes the upper time bound exposed to the existing sampler.
    """

    def __init__(
        self,
        problem: PDEProblem,
        n_interior: int = 128,
        n_boundary: int = 32,
        n_initial: int = 32,
        *,
        window_ends: tuple[float, ...] | list[float],
        seed: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not isinstance(problem, PDEProblem) or not isinstance(
            problem.domain, SpaceTimeDomain
        ):
            raise TypeError("TimeWindowSampler requires a time-dependent PDEProblem.")
        ends = tuple(float(value) for value in window_ends)
        if not ends:
            raise ValueError("window_ends cannot be empty.")
        lower, upper = problem.domain.time.bounds
        if any(not lower < value <= upper for value in ends):
            raise ValueError("Every time-window end must lie in (t_min, t_max].")
        if any(right <= left for left, right in zip(ends, ends[1:])):
            raise ValueError("window_ends must be strictly increasing.")
        if not torch.isclose(torch.tensor(ends[-1]), torch.tensor(upper)):
            raise ValueError("The final time window must end at the problem's t_max.")
        self.problem = problem
        self.window_ends = ends
        self.current_window_index = 0
        self.update_count = 0
        self.n_interior = int(n_interior)
        self.n_boundary = int(n_boundary)
        self.n_initial = int(n_initial)
        self.seed = int(seed)
        self.device = device
        self.dtype = dtype
        self._uniform = UniformSampler(
            problem,
            n_interior=self.n_interior,
            n_boundary=self.n_boundary,
            n_initial=self.n_initial,
            seed=self.seed,
            device=device,
            dtype=dtype,
        )

    @property
    def current_window_end(self) -> float:
        """Return the inclusive upper time bound currently exposed."""

        return self.window_ends[self.current_window_index]

    def update(
        self,
        model: nn.Module,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Open the next configured window; model/device arguments are unused."""

        del model, device, dtype
        if self.current_window_index < len(self.window_ends) - 1:
            self.current_window_index += 1
        self.update_count += 1

    def sample(
        self,
        problem: PDEProblem | Interval | RectangleWithCircularHoles | SpaceTimeDomain | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        requires_grad: bool = True,
    ) -> SampleBatch:
        """Sample uniformly over space and the currently open time prefix."""

        selected_problem = problem if problem is not None else self.problem
        if selected_problem is not self.problem:
            raise ValueError("TimeWindowSampler cannot switch problems after construction.")
        domain = self.problem.domain
        window = SpaceTimeDomain(
            domain.space,
            type(domain.time)(domain.time.lower, self.current_window_end),
        )
        return self._uniform.sample(
            window,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )

    resample = sample

    def diagnostics(self) -> dict[str, Any]:
        """Return the schedule and current phase."""

        return {
            "update_count": self.update_count,
            "current_window_index": self.current_window_index,
            "current_window_end": self.current_window_end,
            "window_ends": list(self.window_ends),
        }

    def state_dict(self) -> dict[str, Any]:
        """Return schedule phase and uniform random stream."""

        return {
            "seed": self.seed,
            "n_interior": self.n_interior,
            "n_boundary": self.n_boundary,
            "n_initial": self.n_initial,
            "window_ends": self.window_ends,
            "current_window_index": self.current_window_index,
            "update_count": self.update_count,
            "uniform": self._uniform.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a configuration-compatible time-window sampler."""

        expected = (
            self.n_interior,
            self.n_boundary,
            self.n_initial,
            self.window_ends,
        )
        restored = (
            int(state["n_interior"]),
            int(state["n_boundary"]),
            int(state["n_initial"]),
            tuple(float(value) for value in state["window_ends"]),
        )
        if restored != expected:
            raise ValueError(
                f"Time-window settings differ from checkpoint: expected {expected}, got {restored}."
            )
        self.seed = int(state["seed"])
        self.current_window_index = int(state["current_window_index"])
        self.update_count = int(state.get("update_count", 0))
        self._uniform.load_state_dict(state["uniform"])
