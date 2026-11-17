"""Natural-language understanding for problem specifications.

This module is the first stage of the agent chain. It extracts problem-side
facts and decides missing fields, follow-up questions, and
``ready_for_validation`` with the physics-family gate in ``readiness``. It
does not call the Validation Agent, select a solver, generate code, or call training APIs.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from .llm import (
    LLMConnectionError,
    LLMExtraction,
    LLMPayloadError,
    extract_with_local_llm,
)
from .readiness import assess_family, questions_for, reconcile_llm_report
from .validator import Severity, ValidationError, ValidationReport


@dataclass
class UnderstandReport:
    """A partial ProblemSpec-shaped draft and the clarification it still needs.

    ``ready_for_validation`` is true when the draft is self-consistent for its
    declared physics family. It does not mean the Validation Agent accepted the final pair.
    ``validation`` stays ``None`` on this path; configurator and CodeBuilder
    still call ``openpinn.agents.validate`` separately.
    """

    draft: dict[str, Any]
    missing: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    ready_for_validation: bool = False
    validation: ValidationReport | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "draft": copy.deepcopy(self.draft),
            "missing": list(self.missing),
            "questions": list(self.questions),
            "warnings": [warning.as_dict() for warning in self.warnings],
            "ready_for_validation": self.ready_for_validation,
            "validation": (
                self.validation.as_dict() if self.validation is not None else None
            ),
        }


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _first_pair(patterns: tuple[str, ...], text: str) -> tuple[float, float] | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1)), float(match.group(2))
    return None


def _first_number(patterns: tuple[str, ...], text: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _has(text: str, *phrases: str) -> bool:
    lowered = re.sub(r"\s+", " ", text.lower())
    return any(re.sub(r"\s+", " ", phrase.lower()) in lowered for phrase in phrases)


def _extract_geometry(text: str) -> dict[str, Any] | None:
    x_bounds = _first_pair(
        (
            rf"x\s*(?:=|范围(?:为)?|从)?\s*({_NUMBER})"
            rf"\s*(?:到|至|to|~|～|,|，)\s*({_NUMBER})",
            rf"({_NUMBER})\s*(?:<=|≤)\s*x\s*(?:<=|≤)\s*({_NUMBER})",
        ),
        text,
    )
    y_bounds = _first_pair(
        (
            rf"y\s*(?:=|范围(?:为)?|从)?\s*({_NUMBER})"
            rf"\s*(?:到|至|to|~|～|,|，)\s*({_NUMBER})",
            rf"({_NUMBER})\s*(?:<=|≤)\s*y\s*(?:<=|≤)\s*({_NUMBER})",
        ),
        text,
    )
    if not (x_bounds and y_bounds and _has(text, "矩形", "rectangle")):
        return None

    geometry: dict[str, Any] = {
        "outer": {
            "type": "rectangle",
            "x_min": x_bounds[0],
            "x_max": x_bounds[1],
            "y_min": y_bounds[0],
            "y_max": y_bounds[1],
        },
        "holes": [],
    }
    center = _first_pair(
        (
            rf"(?:圆孔|孔|hole).*?(?:中心|center)\s*(?:为|at)?"
            rf"\s*[（(]\s*({_NUMBER})\s*[,，]\s*({_NUMBER})\s*[)）]",
        ),
        text,
    )
    radius = _first_number(
        (rf"(?:圆孔|孔|hole).*?(?:半径|radius)\s*(?:为|=|is)?\s*({_NUMBER})",),
        text,
    )
    if center is not None and radius is not None:
        geometry["holes"].append(
            {"type": "circle", "center": list(center), "radius": radius}
        )
    return geometry


def _extract_boundaries(text: str, has_hole: bool) -> list[dict[str, Any]]:
    boundaries: list[dict[str, Any]] = []

    if _has(
        text,
        "左边固定",
        "左边固支",
        "左边界固定",
        "left edge fixed",
        "left edge is fixed",
        "fixed left",
    ):
        boundaries.append({
            "id": "fix_left",
            "region": "left",
            "type": "dirichlet",
            "dofs": ["ux", "uy"],
            "value": [0.0, 0.0],
        })

    traction = _first_number(
        (
            rf"(?:右边界|右边|right(?:\s+edge)?).*?(?:x\s*方向)?"
            rf"(?:牵引|traction|载荷|load).*?(?:为|=|is)?\s*({_NUMBER})",
        ),
        text,
    )
    if traction is not None:
        boundaries.extend((
            {
                "id": "traction_right",
                "region": "right",
                "type": "neumann",
                "dofs": ["ux"],
                "value": traction,
            },
            {
                "id": "shear_right",
                "region": "right",
                "type": "neumann",
                "dofs": ["uy"],
                "value": 0.0,
            },
        ))

    free_all = _has(
        text,
        "上下边和孔边自由",
        "上、下边和孔边自由",
        "上边、下边和孔边自由",
        "top, bottom, and hole boundaries are traction-free",
        "top, bottom and hole boundaries are traction-free",
    )
    free_top_bottom = free_all or _has(
        text,
        "上下边自由",
        "上、下边自由",
        "top and bottom are traction-free",
        "top and bottom edges are traction-free",
    )
    if free_top_bottom:
        for region in ("top", "bottom"):
            boundaries.append({
                "id": f"free_{region}",
                "region": region,
                "type": "neumann",
                "dofs": ["ux", "uy"],
                "value": [0.0, 0.0],
            })
    free_hole = free_all or _has(
        text,
        "孔边自由",
        "圆孔边界自由",
        "hole boundary is traction-free",
    )
    if has_hole and free_hole:
        boundaries.append({
            "id": "free_hole",
            "region": "hole_0",
            "type": "neumann",
            "dofs": ["ux", "uy"],
            "value": [0.0, 0.0],
        })
    return boundaries


def _rule_extract(text: str) -> dict[str, Any]:
    draft: dict[str, Any] = {}
    if _has(text, "二维", "2d", "two-dimensional"):
        draft["dim"] = 2
    if _has(text, "线弹性", "linear elasticity", "linear elastic"):
        draft["physics"] = "linear_elasticity"

    geometry = _extract_geometry(text)
    if geometry is not None:
        draft["geometry"] = geometry

    if _has(text, "位移", "displacement", "ux", "uy"):
        draft["field"] = {"kind": "displacement", "components": ["ux", "uy"]}

    youngs_modulus = _first_number(
        (rf"(?:杨氏模量|young['’]?s modulus|\bE)\s*(?:为|=|is)?\s*({_NUMBER})",),
        text,
    )
    poisson_ratio = _first_number(
        (rf"(?:泊松比|poisson['’]?s ratio|\bnu|ν)\s*(?:为|=|is)?\s*({_NUMBER})",),
        text,
    )
    constitutive: dict[str, Any] = {}
    if draft.get("physics") == "linear_elasticity":
        constitutive["model"] = "isotropic_linear"
        constitutive["stress_voigt"] = True
    if youngs_modulus is not None:
        constitutive["E"] = youngs_modulus
    if poisson_ratio is not None:
        constitutive["nu"] = poisson_ratio
    if constitutive:
        draft["constitutive"] = constitutive

    has_hole = bool(geometry and geometry["holes"])
    boundaries = _extract_boundaries(text, has_hole)
    if boundaries:
        draft["boundaries"] = boundaries
    return draft


def _deep_merge(target: dict[str, Any], additions: Mapping[str, Any]) -> None:
    for key, value in additions.items():
        if "." in key:
            parts = key.split(".")
            cursor = target
            for part in parts[:-1]:
                child = cursor.setdefault(part, {})
                if not isinstance(child, dict):
                    child = {}
                    cursor[part] = child
                cursor = child
            cursor[parts[-1]] = copy.deepcopy(value)
        elif isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _operator_warning(path: str | None) -> list[ValidationError]:
    if path != "operator":
        return []
    return [ValidationError(
        code="understanding.operator_planning_deferred",
        path="path",
        message=(
            "The problem draft can be validated independently, but operator "
            "method planning is outside Problem Understanding."
        ),
        severity=Severity.warning,
    )]


def _rules_report(
    text: str,
    *,
    path: str | None,
    answers: dict[str, Any] | None,
) -> UnderstandReport:
    draft = _rule_extract(text)
    if answers:
        _deep_merge(draft, answers)

    assessment = assess_family(draft)
    return UnderstandReport(
        draft=draft,
        missing=list(assessment.missing),
        questions=questions_for(assessment.missing),
        warnings=_operator_warning(path),
        ready_for_validation=assessment.ready,
        validation=None,
    )


def _llm_report(
    extraction: LLMExtraction,
    *,
    path: str | None,
    answers: dict[str, Any] | None,
) -> UnderstandReport:
    """Build a report from the physics-family gate, not the Validation Agent."""
    draft = copy.deepcopy(extraction.draft)
    if answers:
        _deep_merge(draft, answers)

    missing, questions, ready = reconcile_llm_report(
        draft, list(extraction.missing), list(extraction.questions)
    )
    warnings = [
        ValidationError(
            code=item["code"],
            path=item["path"],
            message=item["message"],
            severity=Severity.warning,
        )
        for item in extraction.warnings
    ]
    warnings.extend(_operator_warning(path))
    return UnderstandReport(
        draft=draft,
        missing=missing,
        questions=questions,
        warnings=warnings,
        ready_for_validation=ready,
        validation=None,
    )


def _fallback_report(
    text: str,
    *,
    path: str | None,
    answers: dict[str, Any] | None,
    reason: str,
    invalid_payload: bool,
) -> UnderstandReport:
    report = _rules_report(text, path=path, answers=answers)
    report.warnings.insert(0, ValidationError(
        code="understanding.llm_fallback",
        path="backend",
        message=f"LLM endpoint unavailable or invalid; used rules fallback: {reason}",
        severity=Severity.warning,
    ))
    if invalid_payload:
        # An invalid LLM response must never be mistaken for an LLM-backed
        # readiness decision, even if the conservative rules draft is valid.
        report.ready_for_validation = False
    return report


def understand_problem(
    text: str,
    *,
    path: Literal["pinn", "operator"] | None = None,
    answers: dict[str, Any] | None = None,
    backend: Literal["rules", "llm", "auto"] | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout: float | None = None,
    api_key: str | None = None,
) -> UnderstandReport:
    """Extract a partial problem draft from Chinese or English prose.

    ``backend="rules"`` is deterministic and requires no API key.
    ``backend="llm"`` and ``"auto"`` call an OpenAI-compatible endpoint
    (local Ollama by default) and fall back visibly to rules on connection or
    payload errors. ``api_key`` adds Bearer authentication for a user-owned
    compatible endpoint. ``answers`` may contain nested problem fragments or
    dotted keys; explicit answers override extracted values. Readiness is the
    physics-family gate:
    a draft is ready when it is self-consistent for the declared family.
    The model does not set that flag, and the Validation Agent is not consulted here.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty natural-language description")
    if path not in (None, "pinn", "operator"):
        raise ValueError("path must be 'pinn', 'operator', or None")
    if answers is not None and not isinstance(answers, dict):
        raise TypeError("answers must be a dict or None")
    resolved_backend = backend or os.getenv("OPENPINN_UNDERSTAND_BACKEND", "rules")
    if resolved_backend not in ("rules", "llm", "auto"):
        raise ValueError("backend must be 'rules', 'llm', or 'auto'")
    if resolved_backend == "rules":
        return _rules_report(text, path=path, answers=answers)

    try:
        extraction = extract_with_local_llm(
            text,
            path=path,
            answers=answers,
            base_url=base_url,
            model=model,
            timeout=timeout,
            api_key=api_key,
        )
    except LLMPayloadError as exc:
        return _fallback_report(
            text,
            path=path,
            answers=answers,
            reason=str(exc),
            invalid_payload=True,
        )
    except LLMConnectionError as exc:
        return _fallback_report(
            text,
            path=path,
            answers=answers,
            reason=str(exc),
            invalid_payload=False,
        )
    return _llm_report(extraction, path=path, answers=answers)


__all__ = ["UnderstandReport", "understand_problem"]
