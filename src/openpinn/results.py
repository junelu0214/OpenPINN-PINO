"""Backend-neutral result view for PINN and neural-operator runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class UnifiedRunResult:
    """Stable cross-backend result surface with access to the raw result."""

    backend: str
    model: str
    status: str
    steps: int
    history: tuple[Mapping[str, float], ...]
    metrics: Mapping[str, Any]
    output_dir: Path | None = None
    artifacts: Mapping[str, str] = field(default_factory=dict)
    raw: Any = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "status": self.status,
            "steps": self.steps,
            "history": [dict(record) for record in self.history],
            "metrics": dict(self.metrics),
            "output_dir": None if self.output_dir is None else str(self.output_dir),
            "artifacts": dict(self.artifacts),
        }


def as_unified_result(result: Any) -> UnifiedRunResult:
    """Adapt an ``ExperimentResult`` or ``OperatorTrainResult``."""

    if all(
        hasattr(result, name)
        for name in ("output_dir", "predictions", "compute_ledger")
    ):
        metrics = dict(result.metrics)
        history = tuple(dict(record) for record in result.history)
        return UnifiedRunResult(
            backend="pinn",
            model=type(result.model).__name__,
            status="completed",
            steps=int(metrics.get("steps", len(history))),
            history=history,
            metrics=metrics,
            output_dir=Path(result.output_dir),
            artifacts={"run_directory": str(result.output_dir)},
            raw=result,
        )

    if all(
        hasattr(result, name)
        for name in ("operator", "initial_loss", "final_loss")
    ):
        values = tuple(float(value) for value in result.history)
        history = tuple(
            {"step": float(index), "total": value}
            for index, value in enumerate(values, start=1)
        )
        metrics = {
            "initial_loss": float(result.initial_loss),
            "final_loss": float(result.final_loss),
            **dict(result.metrics),
        }
        return UnifiedRunResult(
            backend="operator",
            model=str(result.operator),
            status=(
                "completed"
                if math.isfinite(float(result.final_loss))
                else "failed_non_finite"
            ),
            steps=int(result.steps),
            history=history,
            metrics=metrics,
            raw=result,
        )

    raise TypeError(
        "result must be an OpenPINN ExperimentResult or OperatorTrainResult"
    )


__all__ = ["UnifiedRunResult", "as_unified_result"]
