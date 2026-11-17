"""Deterministic engine for the Specification Validation Agent.

Composes M2 pydantic schema validation with additional agent-level
checks (codegen readiness, boundary/field consistency, optional static
code compliance) and returns a machine-readable ``ValidationReport``.

This module does **not** rewrite or fork the M2 schema; it delegates
to ``openpinn.spec.load_problem_spec`` / ``load_method_spec`` and
layers extra domain checks on top.
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from ..spec import (
    ProblemSpec,
    MethodSpec,
    load_problem_spec,
    load_method_spec,
)


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    error = "error"
    warning = "warning"


@dataclass(frozen=True)
class ValidationError:
    """One validation issue with a stable code, field path, and message."""

    code: str
    path: str
    message: str
    severity: Severity = Severity.error

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "severity": self.severity.value,
        }


@dataclass
class ValidationReport:
    """Aggregated validation result returned by :func:`validate`."""

    ok: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    problem_spec: ProblemSpec | None = None
    method_spec: MethodSpec | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [e.as_dict() for e in self.errors],
            "warnings": [w.as_dict() for w in self.warnings],
        }


# ---------------------------------------------------------------------------
# Codegen-readiness whitelist (what M3 currently supports)
# ---------------------------------------------------------------------------

_SUPPORTED_PHYSICS = {"linear_elasticity"}
_SUPPORTED_DIMS = {2}
_SUPPORTED_GEOMETRY_OUTER = {"rectangle"}
_SUPPORTED_CONSTITUTIVE = {"isotropic_linear"}
_SUPPORTED_FIELD_KINDS = {"displacement"}


def _registry_name(value: object) -> str:
    """Normalize a MethodSpec component name that may be a str or a str Enum.

    M4 stores ``differentiator.mode`` as a registry string. Do not assume
    ``.value`` exists: a plain ``str`` has no such attribute, while a
    leftover ``DifferentiatorMode`` enum still exposes one.
    """

    raw = getattr(value, "value", value)
    return str(raw)


def _check_codegen_readiness(
    problem: ProblemSpec,
    method: MethodSpec,
    errors: list[ValidationError],
    warnings: list[ValidationError],
) -> None:
    """Check whether the validated specs are supported by M3 codegen."""
    if problem.dim not in _SUPPORTED_DIMS:
        errors.append(ValidationError(
            code="codegen.unsupported_dim",
            path="problem.dim",
            message=f"M3 codegen supports dims {sorted(_SUPPORTED_DIMS)}, "
                    f"got {problem.dim}",
        ))

    if problem.physics.value not in _SUPPORTED_PHYSICS:
        errors.append(ValidationError(
            code="codegen.unsupported_physics",
            path="problem.physics",
            message=f"M3 codegen supports physics {sorted(_SUPPORTED_PHYSICS)}, "
                    f"got {problem.physics.value!r}",
        ))

    if problem.geometry.outer.type not in _SUPPORTED_GEOMETRY_OUTER:
        errors.append(ValidationError(
            code="codegen.unsupported_geometry",
            path="problem.geometry.outer.type",
            message=f"M3 codegen supports outer geometry types "
                    f"{sorted(_SUPPORTED_GEOMETRY_OUTER)}, "
                    f"got {problem.geometry.outer.type!r}",
        ))

    if problem.constitutive.model.value not in _SUPPORTED_CONSTITUTIVE:
        errors.append(ValidationError(
            code="codegen.unsupported_constitutive",
            path="problem.constitutive.model",
            message=f"M3 codegen supports constitutive models "
                    f"{sorted(_SUPPORTED_CONSTITUTIVE)}, "
                    f"got {problem.constitutive.model.value!r}",
        ))

    if problem.field.kind.value not in _SUPPORTED_FIELD_KINDS:
        errors.append(ValidationError(
            code="codegen.unsupported_field_kind",
            path="problem.field.kind",
            message=f"M3 codegen supports field kinds "
                    f"{sorted(_SUPPORTED_FIELD_KINDS)}, "
                    f"got {problem.field.kind.value!r}",
        ))

    if method.path == "operator":
        errors.append(ValidationError(
            code="codegen.operator_path",
            path="method.path",
            message=(
                "M3 plate-with-hole codegen uses the PINN solve path. "
                "Operator methods are trained with train_operator, not solve. "
                "Switch path to 'pinn' or generate an operator script instead."
            ),
        ))

    if _registry_name(method.differentiator.mode) != "autodiff" and method.path == "pinn":
        errors.append(ValidationError(
            code="codegen.unsupported_differentiator",
            path="method.differentiator.mode",
            message="M3 linear-elasticity PINN codegen requires autodiff; "
                    f"got {_registry_name(method.differentiator.mode)!r}",
        ))

    if problem.constitutive.assumption.value != "plane_stress":
        errors.append(ValidationError(
            code="codegen.unsupported_structural_assumption",
            path="problem.constitutive.assumption",
            message=(
                "M3 linear-elasticity codegen currently supports only the explicit "
                "plane_stress assumption; plane_strain remains editing-only."
            ),
        ))
    else:
        warnings.append(ValidationError(
            code="codegen.plane_stress_only",
            path="problem.constitutive.assumption",
            message=(
                "ProblemSpec explicitly selects assumption='plane_stress'; M3 uses "
                "the plane-stress matrix E/(1-nu^2) for this run."
            ),
            severity=Severity.warning,
        ))


# ---------------------------------------------------------------------------
# Boundary / field consistency (beyond M2)
# ---------------------------------------------------------------------------

def _check_boundary_consistency(
    problem: ProblemSpec,
    errors: list[ValidationError],
    warnings: list[ValidationError],
) -> None:
    """Additional boundary/field cross-checks that go beyond M2 schema."""
    regions_seen: dict[str, int] = {}
    for bc in problem.boundaries:
        regions_seen[bc.region] = regions_seen.get(bc.region, 0) + 1

    required_regions = {"left", "right", "top", "bottom"}
    required_regions.update(
        f"hole_{index}" for index, _hole in enumerate(problem.geometry.holes)
    )
    for region in sorted(required_regions.difference(regions_seen)):
        errors.append(ValidationError(
            code="boundary.missing_region",
            path=f"problem.boundaries.{region}",
            message=f"No boundary condition is defined for required region {region!r}",
        ))

    if not any(bc.type.value == "dirichlet" for bc in problem.boundaries):
        errors.append(ValidationError(
            code="boundary.missing_dirichlet",
            path="problem.boundaries",
            message="At least one Dirichlet boundary is required to anchor displacement",
        ))

    bc_ids = [bc.id for bc in problem.boundaries]
    if len(bc_ids) != len(set(bc_ids)):
        seen: set[str] = set()
        for bid in bc_ids:
            if bid in seen:
                errors.append(ValidationError(
                    code="boundary.duplicate_id",
                    path=f"problem.boundaries[id={bid}]",
                    message=f"Duplicate boundary condition id {bid!r}",
                ))
            seen.add(bid)

    for bc in problem.boundaries:
        if bc.type.value == "neumann" and isinstance(bc.value, list):
            if all(v == 0.0 for v in bc.value):
                comp = problem.field.components
                if set(bc.dofs) == set(comp):
                    warnings.append(ValidationError(
                        code="boundary.traction_free_explicit",
                        path=f"problem.boundaries[id={bc.id}]",
                        message=f"Boundary {bc.id!r} is a zero-traction "
                                f"(traction-free) condition on all dofs; "
                                f"this is valid but equivalent to natural BC",
                        severity=Severity.warning,
                    ))


# ---------------------------------------------------------------------------
# Optional static code compliance
# ---------------------------------------------------------------------------

_IMPORT_WHITELIST = frozenset({
    "torch", "numpy", "math", "openpinn",
    "typing", "pathlib", "os", "sys", "collections",
    "functools", "itertools", "dataclasses", "enum",
})

_FORBIDDEN_FORWARD_PATTERNS = [
    re.compile(r"\bgrad\s*\(", re.MULTILINE),
    re.compile(r"\bhessian\s*\(", re.MULTILINE),
    re.compile(r"\btorch\.autograd\.grad\s*\(", re.MULTILINE),
]


def _check_code_compliance(
    code: str,
    errors: list[ValidationError],
    warnings: list[ValidationError],
) -> None:
    """Static checks on generated code: import whitelist, no PDE in forward."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        errors.append(ValidationError(
            code="code.syntax_error",
            path="code",
            message=f"Generated code has a syntax error: {exc}",
        ))
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in _IMPORT_WHITELIST:
                    errors.append(ValidationError(
                        code="code.forbidden_import",
                        path=f"code:line {node.lineno}",
                        message=f"Import of {alias.name!r} is not in the "
                                f"allowed whitelist: {sorted(_IMPORT_WHITELIST)}",
                    ))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top not in _IMPORT_WHITELIST:
                    errors.append(ValidationError(
                        code="code.forbidden_import",
                        path=f"code:line {node.lineno}",
                        message=f"Import from {node.module!r} is not in the "
                                f"allowed whitelist: {sorted(_IMPORT_WHITELIST)}",
                    ))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "forward":
                body_start = node.body[0].lineno if node.body else node.lineno
                body_end = node.body[-1].end_lineno if node.body else node.lineno
                lines = code.splitlines()
                forward_body = "\n".join(
                    lines[body_start - 1 : body_end]
                )
                for pat in _FORBIDDEN_FORWARD_PATTERNS:
                    match = pat.search(forward_body)
                    if match:
                        errors.append(ValidationError(
                            code="code.pde_in_forward",
                            path=f"code:forward",
                            message="Generated code must not compute PDE "
                                    "residuals (grad/hessian/autograd.grad) "
                                    "inside forward(); the residual belongs "
                                    "in a separate function",
                        ))
                        break


# ---------------------------------------------------------------------------
# M2 schema error extraction
# ---------------------------------------------------------------------------

def _extract_pydantic_errors(
    exc: Exception,
    prefix: str,
) -> list[ValidationError]:
    """Convert a pydantic ValidationError into structured issues."""
    from pydantic import ValidationError as PydanticValidationError

    result: list[ValidationError] = []
    if isinstance(exc, PydanticValidationError):
        for err in exc.errors():
            loc_parts = [str(p) for p in err.get("loc", [])]
            field_path = f"{prefix}.{'.'.join(loc_parts)}" if loc_parts else prefix
            result.append(ValidationError(
                code=f"schema.{err.get('type', 'validation_error')}",
                path=field_path,
                message=err.get("msg", str(exc)),
            ))
    else:
        result.append(ValidationError(
            code="schema.load_error",
            path=prefix,
            message=str(exc),
        ))
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate(
    problem: ProblemSpec | str | dict[str, Any],
    method: MethodSpec | str | dict[str, Any] | None = None,
    *,
    code: str | None = None,
) -> ValidationReport:
    """Validate specs and optionally check codegen readiness / code compliance.

    Parameters
    ----------
    problem:
        A ``ProblemSpec`` instance, YAML file path, or raw dict.
    method:
        A ``MethodSpec`` instance, YAML file path, raw dict, or ``None``
        to skip method validation.
    code:
        Optional generated-code string to run static compliance checks on.

    Returns
    -------
    ValidationReport
        ``ok=True`` when no errors (warnings are allowed).
    """
    errors: list[ValidationError] = []
    warnings: list[ValidationError] = []
    prob_spec: ProblemSpec | None = None
    meth_spec: MethodSpec | None = None

    if isinstance(problem, ProblemSpec):
        prob_spec = problem
    else:
        try:
            prob_spec = load_problem_spec(problem)
        except Exception as exc:
            errors.extend(_extract_pydantic_errors(exc, "problem"))

    if method is not None:
        if isinstance(method, MethodSpec):
            meth_spec = method
        else:
            try:
                meth_spec = load_method_spec(method)
            except Exception as exc:
                errors.extend(_extract_pydantic_errors(exc, "method"))

    if prob_spec is not None:
        _check_boundary_consistency(prob_spec, errors, warnings)

    if prob_spec is not None and meth_spec is not None:
        _check_codegen_readiness(prob_spec, meth_spec, errors, warnings)

    if code is not None:
        _check_code_compliance(code, errors, warnings)

    return ValidationReport(
        ok=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        problem_spec=prob_spec,
        method_spec=meth_spec,
    )
