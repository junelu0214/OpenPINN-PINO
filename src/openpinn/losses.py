"""Named PINN loss composition."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


class LossComposer:
    """Combine named scalar losses with fixed explicit weights."""

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        *,
        reduction: str = "sum",
    ) -> None:
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'.")
        self.weights = {str(key): float(value) for key, value in (weights or {}).items()}
        if any(weight < 0.0 for weight in self.weights.values()):
            raise ValueError("Loss weights must be non-negative.")
        self.reduction = reduction

    def weighted_terms(self, terms: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Apply configured weights without reducing across terms."""

        if not terms:
            raise ValueError("At least one loss term is required.")
        term_names = {str(name) for name in terms}
        unknown_weights = sorted(set(self.weights).difference(term_names))
        if unknown_weights:
            raise ValueError(
                "Loss weight(s) do not match any produced loss term: "
                + ", ".join(unknown_weights)
            )
        weighted: dict[str, Tensor] = {}
        for name, value in terms.items():
            if not isinstance(value, Tensor) or value.numel() != 1:
                raise ValueError(f"Loss term {name!r} must be a scalar Tensor.")
            weighted[str(name)] = value * self.weights.get(str(name), 1.0)
        return weighted

    def compose(self, terms: Mapping[str, Tensor]) -> Tensor:
        """Return the weighted scalar total."""

        values = tuple(self.weighted_terms(terms).values())
        total = torch.stack(values).sum()
        return total / len(values) if self.reduction == "mean" else total

    __call__ = compose


class MagnitudePowerLossComposer(LossComposer):
    """Compress multi-magnitude loss terms with fixed fractional powers.

    This implements only the loss transformation from MMPINN (Wang et al.,
    2024, DOI ``10.1016/j.jcp.2024.113112``): ``w_i L_i**(1/r_i)``.
    It is a fixed loss-balancing transform, not an adaptive-weight algorithm;
    MMPINN's optional multi-level network components are intentionally outside
    this minimal one-component variant.
    """

    def __init__(
        self,
        weights: Mapping[str, float],
        *,
        roots: Mapping[str, float] | None = None,
        default_root: float = 1.0,
        epsilon: float = 1e-12,
        reduction: str = "sum",
    ) -> None:
        super().__init__(weights, reduction=reduction)
        if default_root <= 0.0 or epsilon <= 0.0:
            raise ValueError("default_root and epsilon must be positive.")
        configured = {str(name): float(value) for name, value in (roots or {}).items()}
        unknown = sorted(set(configured).difference(self.weights))
        if unknown:
            raise ValueError(
                "Magnitude-power roots do not match configured loss terms: "
                + ", ".join(unknown)
            )
        if any(not math.isfinite(value) or value <= 0.0 for value in configured.values()):
            raise ValueError("Every magnitude-power root must be positive and finite.")
        self.roots = configured
        self.default_root = float(default_root)
        self.epsilon = float(epsilon)

    def weighted_terms(self, terms: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Apply fixed coefficient and fractional-power compression."""

        if not terms:
            raise ValueError("At least one loss term is required.")
        unknown_weights = sorted(set(self.weights).difference(str(name) for name in terms))
        if unknown_weights:
            raise ValueError(
                "Loss weight(s) do not match any produced loss term: "
                + ", ".join(unknown_weights)
            )
        weighted: dict[str, Tensor] = {}
        for name, value in terms.items():
            if not isinstance(value, Tensor) or value.numel() != 1:
                raise ValueError(f"Loss term {name!r} must be a scalar Tensor.")
            root = self.roots.get(str(name), self.default_root)
            transformed = value.clamp_min(self.epsilon).pow(1.0 / root)
            weighted[str(name)] = transformed * self.weights.get(str(name), 1.0)
        return weighted


class AdaptiveLossComposer(nn.Module):
    """Learn positive term-level weights by ascent while the PINN descends.

    This is a deliberately small, independently written term-level variant of
    the saddle-point idea in McClenny and Braga-Neto (2023), DOI
    ``10.1016/j.jcp.2022.111722``.  The paper's full SA-PINN uses one mask per
    training point; this class instead adds one trainable scalar per named
    loss term so it fits OpenPINN's existing scalar loss contract.
    """

    def __init__(
        self,
        weights: Mapping[str, float],
        *,
        reduction: str = "sum",
        lr: float = 1e-3,
        min_weight: float = 1e-6,
        max_weight: float = 100.0,
    ) -> None:
        super().__init__()
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'.")
        if not weights:
            raise ValueError("AdaptiveLossComposer requires named initial weights.")
        if lr <= 0.0:
            raise ValueError("Adaptive weight learning rate must be positive.")
        if min_weight < 0.0 or max_weight <= min_weight:
            raise ValueError("Adaptive weight bounds must satisfy 0 <= min < max.")
        self.term_names = tuple(str(name) for name in weights)
        if len(set(self.term_names)) != len(self.term_names):
            raise ValueError("Adaptive loss term names must be unique.")
        initial = [float(weights[name]) for name in weights]
        if any(not math.isfinite(value) for value in initial):
            raise ValueError("Adaptive initial weights must be finite.")
        if any(value <= min_weight or value > max_weight for value in initial):
            raise ValueError("Adaptive initial weights must lie in (min_weight, max_weight].")
        raw_values = [math.log(math.expm1(value - min_weight)) for value in initial]
        self.raw_weights = nn.Parameter(torch.tensor(raw_values, dtype=torch.float32))
        self.reduction = reduction
        self.lr = float(lr)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self._optimizer = torch.optim.SGD([self.raw_weights], lr=self.lr)

    def _validate_terms(self, terms: Mapping[str, Tensor]) -> None:
        if not terms:
            raise ValueError("At least one loss term is required.")
        actual = tuple(str(name) for name in terms)
        if set(actual) != set(self.term_names):
            missing = sorted(set(self.term_names).difference(actual))
            unknown = sorted(set(actual).difference(self.term_names))
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if unknown:
                details.append("unknown: " + ", ".join(unknown))
            raise ValueError("Adaptive loss terms do not match configuration (" + "; ".join(details) + ").")
        for name, value in terms.items():
            if not isinstance(value, Tensor) or value.numel() != 1:
                raise ValueError(f"Loss term {name!r} must be a scalar Tensor.")

    def current_weights(self, *, detach: bool = True) -> dict[str, Tensor]:
        """Return the positive weights keyed by configured loss term."""

        values = torch.nn.functional.softplus(self.raw_weights) + self.min_weight
        if detach:
            values = values.detach()
        return {
            name: values[index]
            for index, name in enumerate(self.term_names)
        }

    def weighted_terms(self, terms: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Weight terms for the network descent with masks held fixed."""

        self._validate_terms(terms)
        weights = self.current_weights(detach=True)
        return {str(name): value * weights[str(name)] for name, value in terms.items()}

    def compose(self, terms: Mapping[str, Tensor]) -> Tensor:
        """Return the network objective without differentiating the masks."""

        values = tuple(self.weighted_terms(terms).values())
        total = torch.stack(values).sum()
        return total / len(values) if self.reduction == "mean" else total

    __call__ = compose

    def adapt(self, terms: Mapping[str, Tensor]) -> dict[str, float]:
        """Take one gradient-ascent step on the weights using detached losses."""

        self._validate_terms(terms)
        self._optimizer.zero_grad(set_to_none=True)
        weights = self.current_weights(detach=False)
        values = [weights[str(name)] * value.detach() for name, value in terms.items()]
        objective = torch.stack(values).sum()
        if self.reduction == "mean":
            objective = objective / len(values)
        (-objective).backward()
        self._optimizer.step()
        raw_max = math.log(math.expm1(self.max_weight - self.min_weight))
        with torch.no_grad():
            self.raw_weights.clamp_(max=raw_max)
        return {
            name: float(value.cpu())
            for name, value in self.current_weights(detach=True).items()
        }

    def training_state_dict(self) -> dict[str, Any]:
        """Return parameters plus the internal ascent optimizer state."""

        return {
            "module": self.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }

    def load_training_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore parameters and the internal ascent optimizer."""

        self.load_state_dict(dict(state["module"]))
        self._optimizer.load_state_dict(dict(state["optimizer"]))


class PointwiseAdaptiveLossComposer(nn.Module):
    """Full pointwise self-adaptive PINN masks with ascent updates.

    One positive mask is created for every value returned by
    ``StrongFormulation.pointwise_loss_terms``.  Network parameters descend
    the masked objective while masks ascend it, matching the per-collocation
    saddle-point structure of McClenny and Braga-Neto (2023), DOI
    ``10.1016/j.jcp.2022.111722``.  The implementation is independent and uses
    a monotone softplus mask to keep every coefficient positive.
    """

    requires_pointwise_terms = True

    def __init__(
        self,
        weights: Mapping[str, float],
        *,
        reduction: str = "sum",
        lr: float = 1e-2,
        min_weight: float = 1e-6,
        max_weight: float = 100.0,
    ) -> None:
        super().__init__()
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'.")
        if not weights:
            raise ValueError("PointwiseAdaptiveLossComposer requires named weights.")
        if lr <= 0.0:
            raise ValueError("Pointwise adaptive learning rate must be positive.")
        if min_weight < 0.0 or max_weight <= min_weight:
            raise ValueError("Pointwise weight bounds must satisfy 0 <= min < max.")
        self.term_names = tuple(str(name) for name in weights)
        self.initial_weights = {str(name): float(value) for name, value in weights.items()}
        if any(
            not math.isfinite(value) or value <= min_weight or value > max_weight
            for value in self.initial_weights.values()
        ):
            raise ValueError("Pointwise initial weights must lie within configured bounds.")
        self.raw_weights = nn.ParameterDict()
        self.reduction = reduction
        self.lr = float(lr)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self._optimizer: torch.optim.Optimizer | None = None

    def _validate(self, pointwise_terms: Mapping[str, Tensor]) -> None:
        if set(str(name) for name in pointwise_terms) != set(self.term_names):
            raise ValueError("Pointwise loss terms do not match configured term names.")
        for name, values in pointwise_terms.items():
            if not isinstance(values, Tensor) or values.ndim != 1 or values.numel() == 0:
                raise ValueError(
                    f"Pointwise loss term {name!r} must be a non-empty 1-D Tensor."
                )

    def _ensure_initialized(self, pointwise_terms: Mapping[str, Tensor]) -> None:
        self._validate(pointwise_terms)
        if self.raw_weights:
            for name, values in pointwise_terms.items():
                if self.raw_weights[str(name)].numel() != values.numel():
                    raise ValueError(
                        f"Point count for {name!r} changed from "
                        f"{self.raw_weights[str(name)].numel()} to {values.numel()}; "
                        "pointwise SA-PINN requires fixed sample counts."
                    )
            return
        for name in self.term_names:
            values = pointwise_terms[name]
            initial = self.initial_weights[name]
            raw = math.log(math.expm1(initial - self.min_weight))
            self.raw_weights[name] = nn.Parameter(
                values.new_full((values.numel(),), raw)
            )
        self._optimizer = torch.optim.Adam(self.raw_weights.parameters(), lr=self.lr)

    def point_weights(
        self, *, detach: bool = True
    ) -> dict[str, Tensor]:
        """Return every positive point mask, keyed by loss term."""

        values: dict[str, Tensor] = {}
        for name, raw in self.raw_weights.items():
            weight = torch.nn.functional.softplus(raw) + self.min_weight
            values[name] = weight.detach() if detach else weight
        return values

    def current_weights(self, *, detach: bool = True) -> dict[str, Tensor]:
        """Return mean point mask per term for compact metrics."""

        return {
            name: values.mean()
            for name, values in self.point_weights(detach=detach).items()
        }

    def compose_pointwise(self, pointwise_terms: Mapping[str, Tensor]) -> Tensor:
        """Return the network objective with point masks held fixed."""

        self._ensure_initialized(pointwise_terms)
        weights = self.point_weights(detach=True)
        values = [
            torch.mean(pointwise_terms[name] * weights[name])
            for name in self.term_names
        ]
        total = torch.stack(values).sum()
        return total / len(values) if self.reduction == "mean" else total

    def weighted_terms(self, terms: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Summarize scalar terms with mean masks for history reporting."""

        weights = self.current_weights(detach=True)
        return {name: terms[name] * weights[name] for name in self.term_names}

    def adapt(self, pointwise_terms: Mapping[str, Tensor]) -> dict[str, float]:
        """Take one mask gradient-ascent step using detached point losses."""

        self._ensure_initialized(pointwise_terms)
        assert self._optimizer is not None
        self._optimizer.zero_grad(set_to_none=True)
        weights = self.point_weights(detach=False)
        values = [
            torch.mean(pointwise_terms[name].detach() * weights[name])
            for name in self.term_names
        ]
        objective = torch.stack(values).sum()
        if self.reduction == "mean":
            objective = objective / len(values)
        (-objective).backward()
        self._optimizer.step()
        raw_max = math.log(math.expm1(self.max_weight - self.min_weight))
        with torch.no_grad():
            for raw in self.raw_weights.values():
                raw.clamp_(max=raw_max)
        return {
            name: float(value.cpu())
            for name, value in self.current_weights(detach=True).items()
        }

    def training_state_dict(self) -> dict[str, Any]:
        return {
            "module": self.state_dict(),
            "shapes": {name: parameter.numel() for name, parameter in self.raw_weights.items()},
            "optimizer": self._optimizer.state_dict() if self._optimizer is not None else None,
        }

    def load_training_state_dict(self, state: Mapping[str, Any]) -> None:
        shapes = {str(name): int(size) for name, size in dict(state.get("shapes", {})).items()}
        module_state = dict(state["module"])
        if shapes and not self.raw_weights:
            reference = next(iter(module_state.values()))
            for name in self.term_names:
                if name not in shapes:
                    raise ValueError(f"Checkpoint is missing pointwise masks for {name!r}.")
                self.raw_weights[name] = nn.Parameter(
                    reference.new_empty((shapes[name],))
                )
            self._optimizer = torch.optim.Adam(self.raw_weights.parameters(), lr=self.lr)
        self.load_state_dict(module_state)
        optimizer_state = state.get("optimizer")
        if optimizer_state is not None:
            if self._optimizer is None:
                raise ValueError("Checkpoint contains pointwise optimizer state without masks.")
            self._optimizer.load_state_dict(dict(optimizer_state))


class UncertaintyLossComposer(nn.Module):
    """Balance scalar PINN terms with learned homoscedastic uncertainties.

    The effective coefficient is ``exp(-s_i)`` and ``s_i = log(sigma_i^2)``.
    The network minimizes the weighted losses while the uncertainty variables
    minimize ``sum(exp(-s_i) * L_i + s_i)`` using detached loss values.  This
    is independently implemented from Kendall et al. (2018), arXiv:1705.07115,
    and the AMAW-PINN formulation of Hou et al. (2023).
    """

    def __init__(
        self,
        weights: Mapping[str, float],
        *,
        reduction: str = "sum",
        lr: float = 1e-3,
        min_log_variance: float = -10.0,
        max_log_variance: float = 10.0,
    ) -> None:
        super().__init__()
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'.")
        if not weights:
            raise ValueError("UncertaintyLossComposer requires named initial weights.")
        if lr <= 0.0:
            raise ValueError("Uncertainty learning rate must be positive.")
        if min_log_variance >= max_log_variance:
            raise ValueError("min_log_variance must be smaller than max_log_variance.")
        self.term_names = tuple(str(name) for name in weights)
        initial_weights = [float(weights[name]) for name in weights]
        if any(not math.isfinite(value) or value <= 0.0 for value in initial_weights):
            raise ValueError("Uncertainty initial weights must be positive and finite.")
        self.log_variances = nn.Parameter(
            torch.tensor([-math.log(value) for value in initial_weights], dtype=torch.float32)
        )
        self.reduction = reduction
        self.lr = float(lr)
        self.min_log_variance = float(min_log_variance)
        self.max_log_variance = float(max_log_variance)
        self._optimizer = torch.optim.Adam([self.log_variances], lr=self.lr)

    def _validate_terms(self, terms: Mapping[str, Tensor]) -> None:
        actual = tuple(str(name) for name in terms)
        if set(actual) != set(self.term_names):
            raise ValueError("Uncertainty loss terms do not match configured term names.")
        for name, value in terms.items():
            if not isinstance(value, Tensor) or value.numel() != 1:
                raise ValueError(f"Loss term {name!r} must be a scalar Tensor.")

    def current_weights(self, *, detach: bool = True) -> dict[str, Tensor]:
        """Return ``exp(-s_i)`` for logging and network descent."""

        values = torch.exp(-self.log_variances)
        if detach:
            values = values.detach()
        return {
            name: values[index]
            for index, name in enumerate(self.term_names)
        }

    def weighted_terms(self, terms: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Apply current uncertainty coefficients with them held fixed."""

        self._validate_terms(terms)
        weights = self.current_weights(detach=True)
        return {str(name): value * weights[str(name)] for name, value in terms.items()}

    def compose(self, terms: Mapping[str, Tensor]) -> Tensor:
        values = tuple(self.weighted_terms(terms).values())
        total = torch.stack(values).sum()
        return total / len(values) if self.reduction == "mean" else total

    __call__ = compose

    def adapt(self, terms: Mapping[str, Tensor]) -> dict[str, float]:
        """Update log variances by descending the uncertainty objective."""

        self._validate_terms(terms)
        self._optimizer.zero_grad(set_to_none=True)
        coefficients = self.current_weights(detach=False)
        values = [
            coefficients[str(name)] * value.detach() + self.log_variances[index]
            for index, (name, value) in enumerate(terms.items())
        ]
        objective = torch.stack(values).sum()
        if self.reduction == "mean":
            objective = objective / len(values)
        objective.backward()
        self._optimizer.step()
        with torch.no_grad():
            self.log_variances.clamp_(self.min_log_variance, self.max_log_variance)
        return {
            name: float(value.cpu())
            for name, value in self.current_weights(detach=True).items()
        }

    def training_state_dict(self) -> dict[str, Any]:
        return {"module": self.state_dict(), "optimizer": self._optimizer.state_dict()}

    def load_training_state_dict(self, state: Mapping[str, Any]) -> None:
        self.load_state_dict(dict(state["module"]))
        self._optimizer.load_state_dict(dict(state["optimizer"]))


class ReLoBRaLoLossComposer(nn.Module):
    """Relative loss balancing with random lookback (ReLoBRaLo).

    This is an independent implementation of the update equations in Bischof
    and Kraus, *Multi-Objective Loss Balancing for Physics-Informed Deep
    Learning*.  The balancing weights are detached state, not model
    parameters, and their random stream is checkpointed for reproducibility.
    """

    def __init__(
        self,
        weights: Mapping[str, float],
        *,
        reduction: str = "sum",
        alpha: float = 0.999,
        temperature: float = 0.1,
        expected_lookback: float = 0.999,
        epsilon: float = 1e-12,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'.")
        if not weights:
            raise ValueError("ReLoBRaLoLossComposer requires named initial weights.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1].")
        if temperature <= 0.0 or epsilon <= 0.0:
            raise ValueError("temperature and epsilon must be positive.")
        if not 0.0 <= expected_lookback <= 1.0:
            raise ValueError("expected_lookback must lie in [0, 1].")
        self.term_names = tuple(str(name) for name in weights)
        initial_weights = torch.tensor(
            [float(weights[name]) for name in weights], dtype=torch.float32
        )
        if not torch.isfinite(initial_weights).all() or torch.any(initial_weights <= 0):
            raise ValueError("ReLoBRaLo initial weights must be positive and finite.")
        initial_weights = len(self.term_names) * initial_weights / initial_weights.sum()
        self.register_buffer("balancing_weights", initial_weights)
        self.register_buffer("initial_losses", torch.empty(0, dtype=torch.float32))
        self.register_buffer("previous_losses", torch.empty(0, dtype=torch.float32))
        self.reduction = reduction
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.expected_lookback = float(expected_lookback)
        self.epsilon = float(epsilon)
        self.seed = int(seed)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)
        self.update_count = 0

    def _vector(self, terms: Mapping[str, Tensor]) -> Tensor:
        if set(str(name) for name in terms) != set(self.term_names):
            raise ValueError("ReLoBRaLo loss terms do not match configured term names.")
        values = []
        for name in self.term_names:
            value = terms[name]
            if not isinstance(value, Tensor) or value.numel() != 1:
                raise ValueError(f"Loss term {name!r} must be a scalar Tensor.")
            values.append(value.detach())
        vector = torch.stack(values).to(
            device=self.balancing_weights.device,
            dtype=self.balancing_weights.dtype,
        )
        if not torch.isfinite(vector).all():
            raise FloatingPointError("ReLoBRaLo received a non-finite loss.")
        return vector

    def _balanced(self, current: Tensor, reference: Tensor) -> Tensor:
        ratios = current / (self.temperature * reference.clamp_min(self.epsilon))
        return len(self.term_names) * torch.softmax(ratios, dim=0)

    def current_weights(self, *, detach: bool = True) -> dict[str, Tensor]:
        values = self.balancing_weights.detach() if detach else self.balancing_weights
        return {
            name: values[index]
            for index, name in enumerate(self.term_names)
        }

    def weighted_terms(self, terms: Mapping[str, Tensor]) -> dict[str, Tensor]:
        self._vector(terms)
        return {
            name: terms[name]
            * self.balancing_weights[index].detach().to(
                device=terms[name].device, dtype=terms[name].dtype
            )
            for index, name in enumerate(self.term_names)
        }

    def compose(self, terms: Mapping[str, Tensor]) -> Tensor:
        values = tuple(self.weighted_terms(terms).values())
        total = torch.stack(values).sum()
        return total / len(values) if self.reduction == "mean" else total

    __call__ = compose

    def adapt(self, terms: Mapping[str, Tensor]) -> dict[str, float]:
        current = self._vector(terms)
        if self.initial_losses.numel() == 0:
            self.initial_losses = current.clone()
            self.previous_losses = current.clone()
        else:
            balanced_initial = self._balanced(current, self.initial_losses)
            balanced_previous = self._balanced(current, self.previous_losses)
            rho = float(
                torch.rand((), generator=self._generator).item()
                < self.expected_lookback
            )
            historical = (
                rho * self.balancing_weights + (1.0 - rho) * balanced_initial
            )
            self.balancing_weights = (
                self.alpha * historical + (1.0 - self.alpha) * balanced_previous
            )
            self.previous_losses = current.clone()
        self.update_count += 1
        return {
            name: float(value.cpu())
            for name, value in self.current_weights(detach=True).items()
        }

    def training_state_dict(self) -> dict[str, Any]:
        return {
            "module": self.state_dict(),
            "generator_state": self._generator.get_state(),
            "update_count": self.update_count,
        }

    def load_training_state_dict(self, state: Mapping[str, Any]) -> None:
        module_state = dict(state["module"])
        self.initial_losses = torch.empty_like(module_state["initial_losses"])
        self.previous_losses = torch.empty_like(module_state["previous_losses"])
        self.load_state_dict(module_state)
        self._generator.set_state(state["generator_state"])
        self.update_count = int(state.get("update_count", 0))
