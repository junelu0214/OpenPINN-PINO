"""M34 Specification Validation Agent facade and final-combination gate."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from openpinn.agents import ValidationAgent, validate, validate_specs
from openpinn.codegen import from_specs


SPEC_DIR = Path(__file__).resolve().parents[2] / "examples" / "specs"


def final_specs() -> tuple[dict, dict]:
    problem = yaml.safe_load((SPEC_DIR / "plate_with_hole_problem.yaml").read_text())
    method = yaml.safe_load((SPEC_DIR / "pinn_method.yaml").read_text())
    method["training"]["steps"] = 2
    method["sampling"]["interior"] = 16
    method["sampling"]["boundary"] = 8
    method["network"]["hidden_layers"] = [8, 8]
    return problem, method


def test_new_facade_and_legacy_validate_are_equivalent() -> None:
    problem, method = final_specs()
    legacy = validate(problem, method)
    named = validate_specs(problem, method)

    assert legacy.ok is True
    assert named.as_dict() == legacy.as_dict()
    assert named.problem_spec == legacy.problem_spec
    assert named.method_spec == legacy.method_spec
    assert named.problem_spec.field.components == ["ux", "uy"]
    assert named.problem_spec.geometry.holes[0].center == (1.0, 1.0)
    assert named.problem_spec.geometry.holes[0].radius == 0.25
    assert {bc.id for bc in named.problem_spec.boundaries} >= {
        "fix_left", "traction_right", "free_hole",
    }
    assert any(item.code == "codegen.plane_stress_only" for item in named.warnings)


def test_plane_strain_is_explicitly_rejected_by_current_codegen() -> None:
    problem, method = final_specs()
    problem["constitutive"]["assumption"] = "plane_strain"

    report = validate_specs(problem, method)

    assert report.ok is False
    assert any(
        item.code == "codegen.unsupported_structural_assumption"
        and item.path == "problem.constitutive.assumption"
        for item in report.errors
    )


def test_validation_state_invalidates_on_problem_or_method_edit() -> None:
    problem, method = final_specs()
    agent = ValidationAgent()

    assert agent.validate(problem, method).ok is True
    assert agent.is_current(problem, method) is True
    agent.require_current(problem, method)

    changed_problem = copy.deepcopy(problem)
    changed_problem["constitutive"]["E"] = 70e9
    assert agent.is_current(changed_problem, method) is False

    changed_method = copy.deepcopy(method)
    changed_method["training"]["steps"] += 1
    assert agent.is_current(problem, changed_method) is False


def test_failed_and_incomplete_pairs_never_become_current() -> None:
    problem, method = final_specs()
    del problem["constitutive"]
    del method["optimizer"]
    agent = ValidationAgent()
    report = agent.validate(problem, method)

    assert report.ok is False
    assert agent.is_current(problem, method) is False
    assert any(item.path.startswith("problem.constitutive") for item in report.errors)
    assert any(item.path.startswith("method.optimizer") for item in report.errors)


def test_missing_boundary_unknown_method_and_incompatible_diff_fail() -> None:
    problem, method = final_specs()
    problem["boundaries"] = [
        bc for bc in problem["boundaries"] if bc["region"] != "hole_0"
    ]
    missing = validate_specs(problem, method)
    assert missing.ok is False
    assert any("Missing boundary" in item.message for item in missing.errors)

    problem, method = final_specs()
    method["method"] = "not_registered"
    unknown = validate_specs(problem, method)
    assert unknown.ok is False

    problem, method = final_specs()
    method["differentiator"] = {"mode": "grid"}
    incompatible = validate_specs(problem, method)
    assert incompatible.ok is False
    assert any(item.code == "codegen.unsupported_differentiator" for item in incompatible.errors)


def test_default_from_specs_validates_same_final_pair(monkeypatch) -> None:
    problem, method = final_specs()
    seen: list[tuple[object, object]] = []

    from openpinn.agents import validation

    original = validation.validate_specs

    def recording_validate(problem_spec, method_spec, *, code=None):
        seen.append((problem_spec, method_spec))
        return original(problem_spec, method_spec, code=code)

    monkeypatch.setattr(validation, "validate_specs", recording_validate)
    generated_problem, kwargs = from_specs(problem, method)

    assert generated_problem.output_dim == 2
    assert generated_problem.domain.holes == ((1.0, 1.0, 0.25),)
    assert {condition.name for condition in generated_problem.conditions} >= {
        "dirichlet_left", "traction_right", "traction_hole_0",
    }
    assert kwargs["method"] == method["method"]
    assert len(seen) == 1
    assert seen[0][0].model_dump(mode="json")["geometry"] == problem["geometry"]
    assert seen[0][1].model_dump(mode="json")["training"] == method["training"]
