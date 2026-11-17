"""Mock-HTTP tests for local OpenAI-compatible Problem Understanding."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import pytest

from openpinn.agents import understand_problem, validate


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = json.loads(
    (ROOT / "tests/fixtures/understanding/cases.json").read_text(encoding="utf-8")
)
HEAT_TEXT = (
    "一维热传导方程，x=0处为Dirichlet 边界条件，x=1处为Neumann "
    "边界条件，初始条件为sin(pai*x/L)，热传导系数为0.01。"
)
POISSON_TEXT = (
    "二维泊松方程，方程定义域为[0, 1]*[0, 1]的四边形，边界条件均为0的"
    "Dirichlet边界条件，方程的源项f(x, y)=1。"
)

HEAT_EXTRACTION = {
    "draft": {
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
    },
    "missing": ["boundaries[0].value", "boundaries[1].value", "L"],
    "questions": [
        "x=0 的 Dirichlet 值是多少？",
        "x=1 的 Neumann 通量是多少？",
        "L 的数值是多少？",
    ],
    "warnings": [],
}

POISSON_EXTRACTION = {
    "draft": {
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
    },
    "missing": [],
    "questions": [],
    "warnings": [],
}


class RecordedRequests(list[dict]):
    def __init__(self) -> None:
        super().__init__()
        self.authorizations: list[str | None] = []


@contextmanager
def mock_chat_server(content: str) -> Iterator[tuple[str, RecordedRequests]]:
    requests = RecordedRequests()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - HTTP handler API
            length = int(self.headers.get("Content-Length", "0"))
            requests.append(json.loads(self.rfile.read(length)))
            requests.authorizations.append(self.headers.get("Authorization"))
            body = json.dumps({
                "choices": [{"message": {"role": "assistant", "content": content}}]
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: object) -> None:
            del args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


ELASTICITY_ONLY = ("constitutive.E", "constitutive.nu", "ux", "uy", "杨氏", "泊松比")


def _assert_no_elasticity_pollution(report) -> None:
    blob = " ".join(report.missing + report.questions)
    for token in ELASTICITY_ONLY:
        assert token not in blob
    assert "agent a" not in blob.lower()
    assert report.validation is None


@pytest.mark.parametrize(
    ("text", "extraction", "expected_physics", "expect_ready"),
    [
        (HEAT_TEXT, HEAT_EXTRACTION, "heat_equation", False),
        (POISSON_TEXT, POISSON_EXTRACTION, "poisson", True),
    ],
)
def test_hard_cases_extract_problem_facts_via_mock_llm(
    text: str,
    extraction: dict,
    expected_physics: str,
    expect_ready: bool,
) -> None:
    with mock_chat_server(json.dumps(extraction, ensure_ascii=False)) as (url, calls):
        report = understand_problem(
            text,
            backend="llm",
            base_url=url,
            model="qwen2.5:1.5b-instruct",
            api_key="test-api-key",
        )

    assert report.draft["physics"] == expected_physics
    assert report.draft["dim"] == extraction["draft"]["dim"]
    assert report.ready_for_validation is expect_ready
    _assert_no_elasticity_pollution(report)
    assert len(calls) == 1
    assert calls[0]["model"] == "qwen2.5:1.5b-instruct"
    assert calls[0]["response_format"] == {"type": "json_object"}
    prompt = calls[0]["messages"][1]["content"]
    assert "training plan" in prompt
    assert "understanding.inferred" in prompt
    assert "sin(pai*x/L)" in prompt
    assert "f(x, y)=1" in prompt
    assert "Agent A" not in prompt
    assert calls.authorizations == ["Bearer test-api-key"]
    assert "test-api-key" not in json.dumps(calls[0])


def test_heat_case_keeps_unknown_values_as_questions() -> None:
    with mock_chat_server(json.dumps(HEAT_EXTRACTION, ensure_ascii=False)) as (url, _):
        report = understand_problem(HEAT_TEXT, backend="llm", base_url=url)

    assert report.draft["pde"]["diffusivity"] == pytest.approx(0.01)
    assert report.draft["initial_condition"]["expression"] == "sin(pi*x/L)"
    assert [bc["type"] for bc in report.draft["boundaries"]] == [
        "dirichlet",
        "neumann",
    ]
    assert all(bc["value"] is None for bc in report.draft["boundaries"])
    assert len(report.questions) == 3


def test_cloud_api_key_can_come_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENPINN_UNDERSTAND_API_KEY", "test-env-key")
    with mock_chat_server(json.dumps(HEAT_EXTRACTION, ensure_ascii=False)) as (url, calls):
        report = understand_problem(HEAT_TEXT, backend="llm", base_url=url)

    assert report.draft["physics"] == "heat_equation"
    assert calls.authorizations == ["Bearer test-env-key"]
    assert "test-env-key" not in json.dumps(calls[0])


def test_poisson_case_extracts_square_zero_boundary_and_source() -> None:
    with mock_chat_server(json.dumps(POISSON_EXTRACTION, ensure_ascii=False)) as (url, _):
        report = understand_problem(POISSON_TEXT, backend="auto", base_url=url)

    assert report.draft["geometry"]["bounds"] == {
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
    }
    assert report.draft["pde"]["source"]["value"] == 1.0
    assert {bc["value"] for bc in report.draft["boundaries"]} == {0.0}


def test_supported_elasticity_llm_draft_is_ready_without_agent_a() -> None:
    rules = understand_problem(FIXTURES["complete_zh"])
    extraction = {
        "draft": rules.draft,
        "missing": [],
        "questions": [],
        "warnings": [],
    }
    with mock_chat_server(json.dumps(extraction, ensure_ascii=False)) as (url, _):
        report = understand_problem(
            FIXTURES["complete_zh"], backend="llm", base_url=url
        )

    assert report.ready_for_validation is True
    assert report.validation is None
    assert report.missing == []
    assert validate(report.draft).ok is True


def test_bad_llm_json_warns_falls_back_and_never_marks_ready() -> None:
    with mock_chat_server("not JSON") as (url, _):
        report = understand_problem(
            FIXTURES["complete_zh"], backend="llm", base_url=url
        )

    assert report.draft["physics"] == "linear_elasticity"
    assert report.ready_for_validation is False
    assert any(warning.code == "understanding.llm_fallback" for warning in report.warnings)


def test_unreachable_auto_endpoint_warns_and_uses_rules() -> None:
    report = understand_problem(
        FIXTURES["missing_zh"],
        backend="auto",
        base_url="http://127.0.0.1:1/v1",
        timeout=0.1,
    )

    assert report.draft["physics"] == "linear_elasticity"
    assert report.questions
    assert any(warning.code == "understanding.llm_fallback" for warning in report.warnings)


COMPLETE_HEAT = {
    "draft": {
        "dim": 1,
        "physics": "heat_equation",
        "geometry": {"type": "interval", "x_min": 0.0, "x_max": 1.0},
        "field": {"kind": "temperature", "components": ["u"]},
        "pde": {"type": "diffusion", "diffusivity": 0.01},
        "boundaries": [
            {"location": "x=0", "type": "dirichlet", "value": 0.0},
            {"location": "x=1", "type": "neumann", "value": 0.0},
        ],
        "initial_condition": {"expression": "sin(pi*x)"},
    },
    "missing": ["constitutive.E", "constitutive.nu", "ux", "uy"],
    "questions": ["材料的杨氏模量 E 是多少？", "未知量是 ux 和 uy 吗？"],
    "warnings": [{
        "code": "understanding.inferred",
        "path": "geometry",
        "message": "Interval [0, 1] was inferred from the stated endpoints x=0 and x=1.",
    }],
}


def test_complete_heat_is_ready_without_agent_a_and_drops_elasticity_gaps() -> None:
    with mock_chat_server(json.dumps(COMPLETE_HEAT, ensure_ascii=False)) as (url, _):
        report = understand_problem(HEAT_TEXT, backend="llm", base_url=url)

    assert report.ready_for_validation is True
    assert report.missing == []
    assert report.questions == []
    assert any(warning.code == "understanding.inferred" for warning in report.warnings)
    _assert_no_elasticity_pollution(report)
    assert validate(report.draft).ok is False


def test_heat_pollution_does_not_replace_real_boundary_questions() -> None:
    polluted = json.loads(json.dumps(HEAT_EXTRACTION))
    polluted["missing"] = list(HEAT_EXTRACTION["missing"]) + ["constitutive.E", "ux"]
    polluted["questions"] = list(HEAT_EXTRACTION["questions"]) + ["杨氏模量 E 和 ux 是多少？"]
    with mock_chat_server(json.dumps(polluted, ensure_ascii=False)) as (url, _):
        report = understand_problem(HEAT_TEXT, backend="llm", base_url=url)

    assert report.ready_for_validation is False
    assert report.questions == HEAT_EXTRACTION["questions"]
    assert report.missing == HEAT_EXTRACTION["missing"]
    _assert_no_elasticity_pollution(report)


def test_incomplete_poisson_boundary_still_asks_a_question() -> None:
    incomplete = json.loads(json.dumps(POISSON_EXTRACTION))
    incomplete["draft"]["boundaries"][3]["value"] = None
    with mock_chat_server(json.dumps(incomplete, ensure_ascii=False)) as (url, _):
        report = understand_problem(POISSON_TEXT, backend="llm", base_url=url)

    assert report.ready_for_validation is False
    assert report.questions
    assert any("boundaries" in item and "value" in item for item in report.missing)
    _assert_no_elasticity_pollution(report)


def test_unknown_physics_llm_draft_does_not_default_to_elasticity() -> None:
    extraction = {
        "draft": {"dim": 2},
        "missing": ["constitutive.E", "constitutive.nu", "ux", "uy"],
        "questions": ["材料的杨氏模量 E 是多少？", "位移分量 ux、uy 是什么？"],
        "warnings": [],
    }
    with mock_chat_server(json.dumps(extraction, ensure_ascii=False)) as (url, _):
        report = understand_problem("请解这个方程。", backend="llm", base_url=url)

    assert report.ready_for_validation is False
    assert report.missing == ["physics"]
    assert report.questions == ["要描述哪一种物理方程或本构问题？"]
    _assert_no_elasticity_pollution(report)


def test_llm_schema_still_rejects_ready_flag_and_bad_json() -> None:
    invalid = {
        "draft": {"physics": "poisson"},
        "missing": [],
        "questions": [],
        "warnings": [],
        "ready_for_validation": True,
    }
    with mock_chat_server(json.dumps(invalid, ensure_ascii=False)) as (url, _):
        report = understand_problem(POISSON_TEXT, backend="llm", base_url=url)

    assert report.ready_for_validation is False
    assert any(warning.code == "understanding.llm_fallback" for warning in report.warnings)


def test_llm_rejects_out_of_scope_script_payload() -> None:
    invalid = {
        "draft": {"training_script": "print('not allowed')"},
        "missing": [],
        "questions": [],
        "warnings": [],
    }
    with mock_chat_server(json.dumps(invalid)) as (url, _):
        report = understand_problem(HEAT_TEXT, backend="llm", base_url=url)

    assert report.ready_for_validation is False
    assert any(warning.code == "understanding.llm_fallback" for warning in report.warnings)
