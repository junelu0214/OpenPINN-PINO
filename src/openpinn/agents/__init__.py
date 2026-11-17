"""OpenPINN structured Understanding, Validation, and Planning APIs.

This optional package validates specs and checks codegen readiness
*before* training or code generation proceeds.  It composes M2
``validate_problem_spec`` / ``validate_method_spec`` with additional
agent-level checks (registry matching, boundary consistency, optional
static code compliance) and returns a structured ``ValidationReport``.

Planning recommends a MethodSpec draft but never replaces the Validation Agent. Core and
training modules must NOT import this package.
"""

from .validator import (
    ValidationError as ValidationIssue,
    ValidationReport,
    validate,
)
from .validation import ValidationAgent, validate_specs
from .understanding import UnderstandReport, understand_problem
from .planning import PlanReport, plan_method

__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "UnderstandReport",
    "understand_problem",
    "PlanReport",
    "plan_method",
    "validate",
    "validate_specs",
    "ValidationAgent",
]
