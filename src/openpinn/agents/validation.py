"""Public Specification Validation Agent facade.

The implementation delegates to the deterministic validator engine.  It adds
stable product naming and an optional stateful helper that can prove whether a
previously validated ProblemSpec/MethodSpec pair is still current.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..spec import MethodSpec, ProblemSpec, load_method_spec, load_problem_spec
from .validator import ValidationError, ValidationReport, validate


_FINAL_METHOD_FIELDS = {
    "method", "path", "sampling", "loss", "optimizer", "network",
    "differentiator", "training",
}


def _fingerprint(problem: ProblemSpec, method: MethodSpec) -> str:
    payload = {
        "problem": problem.model_dump(mode="json"),
        "method": method.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_specs(
    problem: ProblemSpec | str | dict[str, Any],
    method: MethodSpec | str | dict[str, Any],
    *,
    code: str | None = None,
) -> ValidationReport:
    """Validate the final ProblemSpec and MethodSpec combination.

    This is the preferred public entry point for the paper §4.3
    Specification Validation Agent.  The legacy :func:`validate` entry point
    remains fully supported and uses the same engine.
    """

    report = validate(problem, method, code=code)
    if report.method_spec is not None:
        supplied = report.method_spec.model_fields_set
        for name in sorted(_FINAL_METHOD_FIELDS.difference(supplied)):
            report.errors.append(ValidationError(
                code="final_spec.missing_method_field",
                path=f"method.{name}",
                message=f"Final MethodSpec must explicitly provide {name!r}",
            ))
    report.ok = not report.errors
    return report


class ValidationAgent:
    """Track validation of one immutable final-spec snapshot.

    Editing either specification makes :meth:`is_current` return ``False``;
    callers must validate the changed pair again before export or codegen.
    """

    def __init__(self) -> None:
        self.report: ValidationReport | None = None
        self._validated_fingerprint: str | None = None

    def validate(
        self,
        problem: ProblemSpec | str | dict[str, Any],
        method: MethodSpec | str | dict[str, Any],
        *,
        code: str | None = None,
    ) -> ValidationReport:
        report = validate_specs(problem, method, code=code)
        self.report = report
        self._validated_fingerprint = (
            _fingerprint(report.problem_spec, report.method_spec)
            if report.ok and report.problem_spec is not None and report.method_spec is not None
            else None
        )
        return report

    def is_current(
        self,
        problem: ProblemSpec | str | dict[str, Any],
        method: MethodSpec | str | dict[str, Any],
    ) -> bool:
        if self._validated_fingerprint is None:
            return False
        try:
            current = _fingerprint(load_problem_spec(problem), load_method_spec(method))
        except Exception:
            return False
        return current == self._validated_fingerprint

    def require_current(
        self,
        problem: ProblemSpec | str | dict[str, Any],
        method: MethodSpec | str | dict[str, Any],
    ) -> ValidationReport:
        """Return the successful report or block a stale/failed export."""

        if self.report is None or not self.report.ok or not self.is_current(problem, method):
            raise ValueError(
                "Specification Validation Agent has not accepted the current "
                "ProblemSpec/MethodSpec pair; validate the final pair again."
            )
        return self.report


__all__ = ["ValidationAgent", "validate_specs"]
