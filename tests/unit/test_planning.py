"""LLM-primary Solver Planning thin-slice tests."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import yaml

from openpinn.agents import PlanReport, plan_method, validate
from openpinn.spec import load_method_spec


ROOT = Path(__file__).resolve().parents[2]
ELASTICITY = ROOT / "examples/specs/plate_with_hole_problem.yaml"

PINN_METHOD = {
    "method": "baseline",
    "variant": "plain_pinn",
    "path": "pinn",
    "sampling": {"name": "uniform", "interior": 512, "boundary": 128},
    "loss": {
        "formulation": "strong",
        "weighting": "fixed",
        "weights": {"pde": 1.0, "boundary": 10.0},
    },
    "optimizer": {"name": "adam", "lr": 0.001},
    "network": {
        "name": "mlp",
        "hidden_layers": [64, 64, 64, 64],
        "activation": "tanh",
    },
    "differentiator": {"mode": "autodiff"},
    "training": {"steps": 5000},
}


class RecordedRequests(list[dict]):
    def __init__(self) -> None:
        super().__init__()
        self.authorizations: list[str | None] = []


@contextmanager
def mock_chat_server(content: str) -> Iterator[tuple[str, RecordedRequests]]:
    calls = RecordedRequests()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            calls.append(json.loads(self.rfile.read(length)))
            calls.authorizations.append(self.headers.get("Authorization"))
            body = json.dumps({
                "choices": [{"message": {"role": "assistant", "content": content}}]
            }).encode()
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
        yield f"http://{host}:{port}/v1", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def response(method: dict = PINN_METHOD, reason: str = "采用注册表中的基线 PINN。") -> str:
    return json.dumps({
        "method_draft": method,
        "rationale": [reason],
        "warnings": [],
    }, ensure_ascii=False)


def clear_keys(monkeypatch) -> None:
    monkeypatch.delenv("OPENPINN_PLAN_API_KEY", raising=False)
    monkeypatch.delenv("OPENPINN_UNDERSTAND_API_KEY", raising=False)


def test_elasticity_mock_llm_returns_methodspec_and_passes_agent_a() -> None:
    problem = yaml.safe_load(ELASTICITY.read_text(encoding="utf-8"))
    with mock_chat_server(response(reason="二维线弹性使用 MLP 与自动微分。")) as (url, calls):
        report = plan_method(
            problem, backend="llm", base_url=url, model="mock-planner",
            api_key="test-api-key",
        )

    assert isinstance(report, PlanReport)
    assert report.ready_for_validation is True
    assert report.rationale == ["二维线弹性使用 MLP 与自动微分。"]
    assert load_method_spec(report.method_draft).method == "baseline"
    assert validate(problem, report.method_draft).ok is True
    assert len(calls) == 1
    assert calls.authorizations == ["Bearer test-api-key"]
    request = calls[0]
    assert request["response_format"] == {"type": "json_object"}
    assert "Never output executable code" in request["messages"][0]["content"]
    assert "Specification Validation Agent" in request["messages"][1]["content"]
    assert "test-api-key" not in json.dumps(request)


def test_heat_mock_llm_returns_complete_methodspec() -> None:
    heat = {
        "dim": 1,
        "physics": "heat_equation",
        "pde": {"type": "diffusion", "diffusivity": 0.01},
        "field": {"kind": "temperature", "components": ["u"]},
    }
    heat_method = json.loads(json.dumps(PINN_METHOD))
    heat_method["sampling"]["initial"] = 128
    with mock_chat_server(response(heat_method, "瞬态热方程增加初始条件采样。")) as (url, _):
        report = plan_method(
            heat, path="pinn", backend="llm", base_url=url,
            model="mock-planner", api_key="test-api-key",
        )

    spec = load_method_spec(report.method_draft)
    assert report.ready_for_validation is True
    assert report.rationale
    assert spec.sampling.initial == 128
    assert spec.differentiator.mode == "autodiff"


def test_auto_without_key_uses_visible_rules_fallback(monkeypatch) -> None:
    clear_keys(monkeypatch)
    report = plan_method({"physics": "poisson", "dim": 2}, backend="auto")

    assert report.ready_for_validation is True
    assert load_method_spec(report.method_draft).method == "baseline"
    assert any(
        isinstance(item, dict) and item.get("code") == "planning.rules_fallback"
        for item in report.warnings
    )


def test_auto_bad_json_falls_back_but_explicit_llm_never_claims_ready() -> None:
    problem = {"physics": "poisson", "dim": 2}
    with mock_chat_server("not-json") as (url, _):
        auto = plan_method(
            problem, backend="auto", base_url=url, api_key="test-api-key"
        )
    with mock_chat_server("not-json") as (url, _):
        explicit = plan_method(
            problem, backend="llm", base_url=url, api_key="test-api-key"
        )

    assert auto.ready_for_validation is True
    assert any(item.get("code") == "planning.rules_fallback" for item in auto.warnings)
    assert explicit.ready_for_validation is False
    assert explicit.method_draft == {}
    assert any(item.get("code") == "planning.llm_failed" for item in explicit.warnings)

    executable = response({**PINN_METHOD, "script": "print(1)"})
    with mock_chat_server(executable) as (url, _):
        rejected = plan_method(
            problem, backend="llm", base_url=url, api_key="test-api-key"
        )
    assert rejected.ready_for_validation is False
    assert rejected.method_draft == {}


def test_unknown_physics_and_unknown_hint_are_not_silent() -> None:
    unknown = plan_method({"physics": "quantum_foam", "dim": 7}, backend="rules")
    invalid = plan_method(
        {"physics": "poisson", "dim": 2}, backend="rules",
        hints={"method": "invented_solver", "secret_knob": 1},
    )

    assert unknown.ready_for_validation is False
    assert any(item.get("code") == "planning.unknown_physics" for item in unknown.warnings)
    assert invalid.ready_for_validation is False
    codes = {item.get("code") for item in invalid.warnings if isinstance(item, dict)}
    assert {"planning.hint_ignored", "planning.invalid_method_draft"} <= codes


def test_plan_report_as_dict_is_a_copy() -> None:
    report = PlanReport({"method": "baseline"}, ["原因"], ["警告"], True)
    exported = report.as_dict()
    exported["method_draft"]["method"] = "changed"
    assert report.method_draft["method"] == "baseline"
