"""Python/JavaScript parity checks for the static Understanding trial."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from openpinn.agents import understand_problem


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "understanding" / "cases.json"


def test_browser_engine_matches_python_reports_for_shared_fixtures() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for Understanding browser parity.")

    script = r"""
const fs = require("fs");
const trial = require("./docs/_static/understanding_trial.js");
const cases = JSON.parse(fs.readFileSync("tests/fixtures/understanding/cases.json", "utf8"));
const reports = {};
for (const [name, text] of Object.entries(cases)) {
  reports[name] = trial.understandProblem(text, {language: name.endsWith("_en") ? "en" : "zh"});
}
process.stdout.write(JSON.stringify(reports));
"""
    completed = subprocess.run(
        [node, "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    browser_reports = json.loads(completed.stdout)
    fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    for name, text in fixtures.items():
        python_report = understand_problem(text)
        browser_report = browser_reports[name]
        assert browser_report["draft"] == python_report.draft
        assert browser_report["missing"] == python_report.missing
        assert (
            browser_report["ready_for_validation"]
            == python_report.ready_for_validation
        )
        assert [item["code"] for item in browser_report["warnings"]] == [
            item.code for item in python_report.warnings
        ]


def test_browser_family_gate_matches_python_for_heat_and_poisson() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for Understanding browser parity.")

    poisson = {
        "dim": 2,
        "physics": "poisson",
        "geometry": {
            "type": "quadrilateral",
            "bounds": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        },
        "field": {"kind": "scalar", "components": ["u"]},
        "pde": {"source": {"expression": "1", "value": 1.0}},
        "boundaries": [
            {"region": side, "type": "dirichlet", "value": 0.0}
            for side in ("left", "right", "top", "bottom")
        ],
    }
    heat = {
        "dim": 1,
        "physics": "heat_equation",
        "geometry": {"type": "interval", "x_min": 0.0, "x_max": 1.0},
        "field": {"kind": "temperature", "components": ["u"]},
        "pde": {"type": "diffusion", "diffusivity": 0.01},
        "boundaries": [
            {"location": "x=0", "type": "dirichlet", "value": None},
            {"location": "x=1", "type": "neumann", "value": None},
        ],
        "initial_condition": {"expression": "sin(pi*x/L)"},
    }
    script = r"""
const trial = require("./docs/_static/understanding_trial.js");
const payload = JSON.parse(process.argv[1]);
const requests = [];
function mockFetch(_url, init) {
  requests.push(JSON.parse(init.body));
  return Promise.resolve({
    ok: true,
    json: async () => ({choices: [{message: {content: payload.response}}]})
  });
}
(async () => {
  const rulesHeat = trial.understandProblem(payload.heatText, {language: "zh"});
  const rulesPoisson = trial.understandProblem("泊松方程", {language: "zh", answers: payload.poisson});
  const cloudHeat = await trial.understandWithCloud(payload.heatText, {
    language: "zh",
    apiKey: "test-api-key",
    baseUrl: "https://example.invalid/v1",
    model: "mock",
    fetch: mockFetch
  });
  const cloudPrompt = requests[0].messages[1].content;
  process.stdout.write(JSON.stringify({rulesHeat, rulesPoisson, cloudHeat, cloudPrompt}));
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
"""
    heat_response = json.dumps({
        "draft": heat,
        "missing": ["boundaries[0].value", "boundaries[1].value", "L", "constitutive.E", "ux"],
        "questions": [
            "x=0 的 Dirichlet 值是多少？",
            "x=1 的 Neumann 通量是多少？",
            "L 的数值是多少？",
            "杨氏模量 E 和 ux 是多少？",
        ],
        "warnings": [],
    }, ensure_ascii=False)
    completed = subprocess.run(
        [node, "-e", script, json.dumps({
            "heatText": (
                "一维热传导方程，x=0处为Dirichlet 边界条件，x=1处为Neumann "
                "边界条件，初始条件为sin(pai*x/L)，热传导系数为0.01。"
            ),
            "poisson": poisson,
            "response": heat_response,
        }, ensure_ascii=False)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    browser = json.loads(completed.stdout)
    heat_text = (
        "一维热传导方程，x=0处为Dirichlet 边界条件，x=1处为Neumann "
        "边界条件，初始条件为sin(pai*x/L)，热传导系数为0.01。"
    )
    python_rules_heat = understand_problem(heat_text)
    python_rules_poisson = understand_problem("泊松方程", answers=poisson)
    assert browser["rulesHeat"]["draft"] == python_rules_heat.draft
    assert browser["rulesHeat"]["missing"] == python_rules_heat.missing
    assert browser["rulesHeat"]["questions"] == python_rules_heat.questions
    assert browser["rulesHeat"]["ready_for_validation"] == python_rules_heat.ready_for_validation
    assert browser["rulesPoisson"]["ready_for_validation"] is True
    assert browser["rulesPoisson"]["missing"] == python_rules_poisson.missing == []
    assert browser["rulesPoisson"]["ready_for_validation"] == python_rules_poisson.ready_for_validation

    from openpinn.agents.llm import LLMExtraction
    from openpinn.agents.understanding import _llm_report

    python_cloud = _llm_report(
        LLMExtraction(
            draft=heat,
            missing=["boundaries[0].value", "boundaries[1].value", "L", "constitutive.E", "ux"],
            questions=[
                "x=0 的 Dirichlet 值是多少？",
                "x=1 的 Neumann 通量是多少？",
                "L 的数值是多少？",
                "杨氏模量 E 和 ux 是多少？",
            ],
            warnings=[],
        ),
        path=None,
        answers=None,
    )
    assert browser["cloudHeat"]["draft"] == python_cloud.draft
    assert browser["cloudHeat"]["missing"] == python_cloud.missing
    assert browser["cloudHeat"]["questions"] == python_cloud.questions
    assert browser["cloudHeat"]["ready_for_validation"] is False
    assert browser["cloudHeat"]["ready_for_validation"] == python_cloud.ready_for_validation
    assert "constitutive.E" not in browser["cloudHeat"]["missing"]
    assert "ux" not in " ".join(browser["cloudHeat"]["questions"])
    prompt = browser["cloudPrompt"]
    assert "understanding.inferred" in prompt
    assert "sin(pai*x/L)" in prompt
    assert "f(x, y)=1" in prompt
    assert "training plan" in prompt
    assert "Agent A" not in prompt


def test_trial_readiness_path_does_not_call_validate_draft() -> None:
    source = (ROOT / "docs" / "_static" / "understanding_trial.js").read_text(
        encoding="utf-8"
    )
    rules = source.split("function understandProblem", 1)[1].split("function llmEndpoint", 1)[0]
    cloud = source.split("function llmReport", 1)[1].split("function llmPrompt", 1)[0]
    assert "validateDraft" not in rules
    assert "validateDraft" not in cloud


def test_trial_source_offers_no_codegen_or_training_action() -> None:
    source = (ROOT / "docs" / "_static" / "understanding_trial.js").read_text(
        encoding="utf-8"
    )
    forbidden_call = re.compile(
        r"\b(?:solve|train_operator|train_operator_problem)\s*\("
    )

    assert forbidden_call.search(source) is None
    assert "browserFallback" not in source
    assert "understanding.llm_fallback" not in source
    assert 'localStorage.setItem("openpinn-understanding-api-key"' in source
    assert 'localStorage.removeItem("openpinn-understanding-api-key"' in source
    assert 'CONFIGURATOR_HANDOFF_KEY = "openpinn-understanding-configurator-v1"' in source
    assert "sessionStorage.setItem(CONFIGURATOR_HANDOFF_KEY" in source
    assert "document.cookie" not in source
