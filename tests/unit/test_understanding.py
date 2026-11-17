"""Tests for the deterministic Problem Understanding thin slice."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from openpinn.agents import UnderstandReport, understand_problem, validate
from openpinn.agents.readiness import assess_family, reconcile_llm_report


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures" / "understanding" / "cases.json"
)
FIXTURES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
COMPLETE_ZH = FIXTURES["complete_zh"]
MISSING_ZH = FIXTURES["missing_zh"]
COMPLETE_EN = FIXTURES["complete_en"]


def test_complete_description_is_ready_without_calling_agent_a() -> None:
    report = understand_problem(COMPLETE_ZH, path="pinn")

    assert isinstance(report, UnderstandReport)
    assert report.ready_for_validation is True
    assert report.missing == []
    assert report.questions == []
    assert report.validation is None
    assert validate(report.draft).ok is True
    assert report.draft["geometry"]["holes"][0]["radius"] == 0.25


def test_missing_description_returns_draft_and_questions() -> None:
    report = understand_problem(MISSING_ZH)

    assert report.ready_for_validation is False
    assert report.validation is None
    assert report.draft["dim"] == 2
    assert report.draft["physics"] == "linear_elasticity"
    assert report.draft["geometry"]["outer"]["x_max"] == 2.0
    assert "constitutive.E" in report.missing
    assert "constitutive.nu" in report.missing
    assert "boundaries" in report.missing
    assert report.questions


def test_complete_english_description_uses_the_same_gate() -> None:
    report = understand_problem(COMPLETE_EN)

    assert report.ready_for_validation is True
    assert report.validation is None
    assert report.missing == []
    assert report.draft["constitutive"]["E"] == 210e9


def test_answers_merge_into_the_draft_before_validation() -> None:
    first = understand_problem(MISSING_ZH)
    completed = understand_problem(
        MISSING_ZH,
        answers={
            "constitutive.E": 70e9,
            "constitutive.nu": 0.33,
            "boundaries": [
                {"id": "fix_left", "region": "left", "type": "dirichlet",
                 "dofs": ["ux", "uy"], "value": [0.0, 0.0]},
                {"id": "free_right", "region": "right", "type": "neumann",
                 "dofs": ["ux", "uy"], "value": [0.0, 0.0]},
                {"id": "free_top", "region": "top", "type": "neumann",
                 "dofs": ["ux", "uy"], "value": [0.0, 0.0]},
                {"id": "free_bottom", "region": "bottom", "type": "neumann",
                 "dofs": ["ux", "uy"], "value": [0.0, 0.0]},
            ],
        },
    )

    assert first.ready_for_validation is False
    assert completed.ready_for_validation is True
    assert completed.validation is None
    assert completed.missing == []


def test_invalid_material_is_not_ready_for_its_family() -> None:
    report = understand_problem(
        COMPLETE_ZH,
        answers={"constitutive.E": -1.0},
    )

    assert report.ready_for_validation is False
    assert report.validation is None
    assert "constitutive.E" in report.missing
    assert report.questions
    assert all("ux" not in question for question in report.questions)


def test_report_as_dict_is_serializable_shape() -> None:
    data = understand_problem(COMPLETE_ZH).as_dict()

    assert data["ready_for_validation"] is True
    assert data["validation"] is None
    assert isinstance(data["warnings"], list)


def test_unknown_physics_asks_what_physics_not_elasticity() -> None:
    report = understand_problem("请根据描述帮我解一个方程。")

    assert report.ready_for_validation is False
    assert report.validation is None
    assert report.missing == ["physics"]
    assert report.questions == ["要描述哪一种物理方程或本构问题？"]
    blob = " ".join(report.missing + report.questions)
    for token in ("constitutive.E", "constitutive.nu", "ux", "uy", "杨氏", "泊松比"):
        assert token not in blob


def test_declared_non_triad_family_uses_generic_family_readiness() -> None:
    draft = {
        "dim": 1,
        "physics": "wave_equation",
        "geometry": {"type": "interval", "x_min": 0.0, "x_max": 1.0},
        "field": {"kind": "scalar", "components": ["u"]},
        "pde": {"equation": "dt(u)-dx(dx(u))=0", "time_dependent": True},
        "boundaries": [
            {"region": "left", "type": "dirichlet", "value": 0.0},
            {"region": "right", "type": "dirichlet", "value": 0.0},
        ],
        "initial_conditions": {"displacement": "sin(pi*x)", "velocity": 0.0},
    }

    assessment = assess_family(draft)
    assert assessment.family == "wave_equation"
    assert assessment.ready is True
    assert assessment.missing == []
    missing, questions, ready = reconcile_llm_report(
        draft,
        ["constitutive.E", "constitutive.nu", "ux", "uy"],
        ["What are E, nu, ux, and uy?"],
    )
    assert (missing, questions, ready) == ([], [], True)


def test_declared_non_triad_family_has_generic_gaps_without_elasticity_pollution() -> None:
    assessment = assess_family({"physics": "navier_stokes"})

    assert assessment.family == "navier_stokes"
    assert assessment.ready is False
    assert set(assessment.missing) == {"dim", "geometry", "field", "pde", "boundaries"}
    blob = " ".join(assessment.missing)
    assert "constitutive" not in blob
    assert "ux" not in blob


def test_understanding_source_does_not_call_agent_a_validate() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "src" / "openpinn" / "agents" / "understanding.py"
    ).read_text(encoding="utf-8")
    assert "validate(" not in source


def test_understanding_does_not_import_training_or_codegen() -> None:
    source_path = (
        Path(__file__).resolve().parents[2]
        / "src" / "openpinn" / "agents" / "understanding.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden = {"training", "codegen", "solve", "operators"}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not forbidden.intersection(node.module.split("."))
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not forbidden.intersection(alias.name.split("."))


def test_operator_path_does_not_plan_a_method() -> None:
    report = understand_problem(COMPLETE_ZH, path="operator")

    assert report.ready_for_validation is True
    assert "method" not in report.draft
    assert any(
        warning.code == "understanding.operator_planning_deferred"
        for warning in report.warnings
    )
