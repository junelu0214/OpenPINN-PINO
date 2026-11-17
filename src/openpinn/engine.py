"""Budgeted, attachable entry point for strong-form PINN training."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
import traceback
from typing import Any, Literal

import torch
from torch import nn

from .artifacts import ArtifactWriter
from .conditions import Condition
from .expressions import CompiledProblemExpressions
from .formulations import FormulationProtocol
from .losses import (
    AdaptiveLossComposer,
    LossComposer,
    MagnitudePowerLossComposer,
    PointwiseAdaptiveLossComposer,
    ReLoBRaLoLossComposer,
    UncertaintyLossComposer,
)
from .optimizers import OptimizerAdapter
from .problems import PDEProblem
from .sampling import SamplerProtocol
from .training import Trainer


EngineStatus = Literal[
    "budget_exhausted", "interrupted", "failed", "not_converged", "completed"
]
StopCondition = Callable[[Mapping[str, float]], bool]
StepCallback = Callable[[Mapping[str, float]], None]
CancelFlag = Callable[[], bool] | object
LossComposerLike = (
    LossComposer
    | AdaptiveLossComposer
    | MagnitudePowerLossComposer
    | PointwiseAdaptiveLossComposer
    | UncertaintyLossComposer
    | ReLoBRaLoLossComposer
)


@dataclass(frozen=True)
class EngineRunResult:
    """Persisted outcome of one explicitly budgeted engine run."""

    status: EngineStatus
    steps_completed: int
    elapsed_seconds: float
    device: str
    max_steps: int
    max_wall_seconds: float
    stop_reason: str
    history_path: str
    checkpoint_path: str | None
    failure: Mapping[str, str] | None = None
    schema_version: int = 1
    kind: str = "PINNEngineRun"

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready run record."""

        return asdict(self)


class _EngineSignal(RuntimeError):
    pass


class _StopConditionMet(_EngineSignal):
    pass


class _WallBudgetExhausted(_EngineSignal):
    pass


class _CancelRequested(_EngineSignal):
    pass


class _NonFiniteLoss(_EngineSignal):
    pass


def _cancelled(flag: CancelFlag | None) -> bool:
    if flag is None:
        return False
    if callable(flag):
        return bool(flag())
    is_set = getattr(flag, "is_set", None)
    if callable(is_set):
        return bool(is_set())
    raise TypeError("cancel_requested must be callable or provide is_set()")


def _failure_info(error: BaseException) -> dict[str, str]:
    stack = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    return {
        "type": type(error).__name__.lstrip("_"),
        "message": str(error),
        "traceback": stack[-4000:],
    }


def problem_from_compiled_expressions(
    compiled: CompiledProblemExpressions,
    domain: Any,
    conditions: Sequence[Condition] = (),
    *,
    name: str = "compiled_expressions",
) -> PDEProblem:
    """Attach Stage B equation callables to a standard :class:`PDEProblem`.

    Boundary and initial conditions remain ordinary OpenPINN ``Condition``
    objects supplied by the caller.  This hook contains no physics-family
    selection and does not establish product backend support.
    """

    if not isinstance(compiled, CompiledProblemExpressions):
        raise TypeError("compiled must be a CompiledProblemExpressions instance")
    if not hasattr(domain, "dim"):
        raise TypeError("domain must provide a coordinate dimension")
    if int(domain.dim) != len(compiled.coordinate_names):
        raise ValueError(
            "domain dimension does not match the compiled coordinate binding"
        )

    def residual(
        coordinates: torch.Tensor, prediction: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {
            residual_name: evaluate(coordinates, prediction)
            for residual_name, evaluate in compiled.equations.items()
        }

    return PDEProblem(
        domain,
        residual,
        conditions,
        name=name,
        field_names=compiled.field_names,
        output_dim=len(compiled.field_names),
        metadata={
            "source": "compiled_expressions",
            "coordinate_names": list(compiled.coordinate_names),
        },
    )


def run_pinn(
    formulation: PDEProblem | FormulationProtocol,
    *,
    model: nn.Module,
    sampler: SamplerProtocol,
    output_dir: str | Path,
    max_steps: int,
    max_wall_seconds: float,
    optimizer: str | OptimizerAdapter = "adam",
    lr: float | None = 1e-3,
    loss_composer: LossComposerLike | None = None,
    loss_weights: Mapping[str, float] | None = None,
    optimizer_kwargs: Mapping[str, Any] | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    resample_every: int = 1,
    checkpoint_every: int = 1,
    stop_condition: StopCondition | None = None,
    cancel_requested: CancelFlag | None = None,
    callback: StepCallback | None = None,
) -> EngineRunResult:
    """Run the existing ``Trainer`` loop under explicit budgets.

    ``completed`` means only that the caller's stop condition fired before a
    budget.  With no stop condition, using all requested steps is recorded as
    ``budget_exhausted`` even when every loss is finite.
    """

    if isinstance(max_steps, bool) or int(max_steps) != max_steps or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    if (
        isinstance(max_wall_seconds, bool)
        or not isinstance(max_wall_seconds, (int, float))
        or not math.isfinite(float(max_wall_seconds))
        or max_wall_seconds <= 0
    ):
        raise ValueError("max_wall_seconds must be positive and finite")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")

    step_budget = int(max_steps)
    wall_budget = float(max_wall_seconds)
    writer = ArtifactWriter(output_dir)
    history_path = writer.write_history([])
    checkpoint_path: Path | None = None
    trainer: Trainer | None = None
    started = time.perf_counter()
    status: EngineStatus = "failed"
    stop_reason = "initialization_failed"
    failure: Mapping[str, str] | None = None

    try:
        trainer = Trainer(
            model,
            formulation,
            sampler,
            optimizer=optimizer,
            lr=lr,
            loss_composer=loss_composer,
            loss_weights=loss_weights,
            optimizer_kwargs=optimizer_kwargs,
            device=device,
            dtype=dtype,
        )

        if _cancelled(cancel_requested):
            raise _CancelRequested("cooperative cancellation requested")
        if time.perf_counter() - started >= wall_budget:
            raise _WallBudgetExhausted(
                "wall-clock budget exhausted before the first step"
            )

        def on_step(record: Mapping[str, float]) -> None:
            nonlocal checkpoint_path, history_path
            history_path = writer.write_history(trainer.history.as_list())
            if trainer.steps_completed % checkpoint_every == 0:
                checkpoint_path = writer.write_checkpoint(trainer.state_dict())
            total = float(record["total"])
            if not math.isfinite(total):
                raise _NonFiniteLoss(
                    f"non-finite total loss at step {trainer.steps_completed}"
                )
            if callback is not None:
                callback(dict(record))
            if _cancelled(cancel_requested):
                raise _CancelRequested("cooperative cancellation requested")
            if time.perf_counter() - started >= wall_budget:
                raise _WallBudgetExhausted("wall-clock budget exhausted")
            if stop_condition is not None and bool(stop_condition(dict(record))):
                raise _StopConditionMet("caller-supplied stop condition met")

        trainer.fit(
            steps=step_budget,
            resample_every=resample_every,
            callback=on_step,
        )
        if stop_condition is None:
            status = "budget_exhausted"
            stop_reason = "max_steps"
        else:
            status = "not_converged"
            stop_reason = "stop_condition_not_met_before_max_steps"
    except _StopConditionMet:
        status = "completed"
        stop_reason = "stop_condition_met"
    except _WallBudgetExhausted:
        status = "budget_exhausted"
        stop_reason = "max_wall_seconds"
    except (_CancelRequested, KeyboardInterrupt):
        status = "interrupted"
        stop_reason = "cancel_requested"
    except _NonFiniteLoss as error:
        status = "failed"
        stop_reason = "non_finite_loss"
        failure = _failure_info(error)
    except Exception as error:
        status = "failed"
        stop_reason = "exception"
        failure = _failure_info(error)

    elapsed = float(time.perf_counter() - started)
    steps_completed = trainer.steps_completed if trainer is not None else 0
    if trainer is not None:
        trainer.elapsed_seconds = elapsed
        history_path = writer.write_history(trainer.history.as_list())
        checkpoint_path = writer.write_checkpoint(trainer.state_dict())

    result = EngineRunResult(
        status=status,
        steps_completed=steps_completed,
        elapsed_seconds=elapsed,
        device=str(torch.device(device)),
        max_steps=step_budget,
        max_wall_seconds=wall_budget,
        stop_reason=stop_reason,
        history_path=history_path.name,
        checkpoint_path=checkpoint_path.name if checkpoint_path is not None else None,
        failure=failure,
    )
    writer.write_json("engine_run.json", result.as_dict())
    writer.write_manifest(metadata={"kind": result.kind, "status": result.status})
    return result
