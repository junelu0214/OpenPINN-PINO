"""OpenAI-compatible client for Problem Understanding.

The client is intentionally small and uses only the Python standard library.
It asks for JSON problem facts, validates the response envelope and output
shape, and never generates code or solver plans.
"""

from __future__ import annotations

import json
import math
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping


DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen2.5:1.5b-instruct"
DEFAULT_TIMEOUT = 15.0
_MAX_RESPONSE_BYTES = 1_000_000
_FORBIDDEN_DRAFT_KEYS = {
    "code",
    "method",
    "network",
    "optimizer",
    "script",
    "training",
    "training_script",
}


class LLMUnderstandingError(RuntimeError):
    """Base error raised by the optional local-LLM backend."""


class LLMConnectionError(LLMUnderstandingError):
    """The configured endpoint could not be reached or returned an HTTP error."""


class LLMPayloadError(LLMUnderstandingError):
    """The endpoint returned malformed or schema-invalid JSON."""


@dataclass(frozen=True)
class LLMExtraction:
    """Validated problem facts returned by an LLM endpoint."""

    draft: dict[str, Any]
    missing: list[str]
    questions: list[str]
    warnings: list[dict[str, str]]


def _chat_endpoint(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    if not value:
        raise ValueError("LLM base_url must not be empty")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return value + "/chat/completions"
    return value + "/v1/chat/completions"


_HEAT_FEWSHOT_INPUT = (
    "一维热传导方程，x=0处为Dirichlet 边界条件，x=1处为Neumann "
    "边界条件，初始条件为sin(pai*x/L)，热传导系数为0.01。"
)
_POISSON_FEWSHOT_INPUT = (
    "二维泊松方程，方程定义域为[0, 1]*[0, 1]的四边形，边界条件均为0的"
    "Dirichlet边界条件，方程的源项f(x, y)=1。"
)
_HEAT_FEWSHOT_OUTPUT = {
    "draft": {
        "dim": 1,
        "physics": "heat_equation",
        "geometry": {"type": "interval", "x_min": 0, "x_max": 1},
        "field": {"kind": "temperature", "components": ["u"]},
        "pde": {
            "type": "diffusion",
            "diffusivity": 0.01,
            "expression": "dt(u)-alpha*dx(dx(u))=0",
        },
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
    "warnings": [
        {
            "code": "understanding.inferred",
            "path": "geometry",
            "message": "Interval [0, 1] was inferred from the stated endpoints x=0 and x=1.",
        },
        {
            "code": "understanding.inferred",
            "path": "initial_condition.expression",
            "message": "Normalized pai to pi; L remains unknown.",
        },
    ],
}
_POISSON_FEWSHOT_OUTPUT = {
    "draft": {
        "dim": 2,
        "physics": "poisson",
        "geometry": {
            "type": "quadrilateral",
            "bounds": {"x": [0, 1], "y": [0, 1]},
        },
        "field": {"kind": "scalar", "components": ["u"]},
        "pde": {
            "expression": "-dx(dx(u))-dy(dy(u))=f",
            "source": {"expression": "1", "value": 1},
        },
        "boundaries": [
            {"region": side, "type": "dirichlet", "value": 0}
            for side in ("left", "right", "top", "bottom")
        ],
    },
    "missing": [],
    "questions": [],
    "warnings": [],
}


def _problem_prompt(
    text: str,
    path: str | None,
    answers: Mapping[str, Any] | None,
) -> str:
    context = {
        "description": text,
        "path_context": path,
        "confirmed_answers": answers or {},
    }
    return (
        "Extract problem-side PDE facts only. Do not recommend a solver, model, "
        "sampler, loss, optimizer, training plan, or Python code. You may "
        "reasonably extend obvious facts. Normalize spelling such as pai to pi. "
        "If endpoints such as x=0 and x=1 are stated and no other domain is "
        "given, the interval [0, 1] is a reasonable inference. Mark every "
        "inferred fact in warnings as an object with code "
        "\"understanding.inferred\", a path, and a message. Do not invent "
        "numeric values the user did not state, including Dirichlet values, "
        "Neumann fluxes, symbolic lengths, and material constants; put those "
        "true gaps in missing and questions. Return exactly one JSON object "
        "with keys draft, missing, questions, warnings. draft must be a JSON "
        "object of problem facts for the declared physics family; missing and "
        "questions must be arrays of strings and may mention only fields that "
        "family needs. If the physics family is unknown, ask which physics it "
        "is and do not assume linear elasticity or ask for E, nu, ux, or uy. "
        "Whenever a single-field PDE can be expressed, include pde.expression "
        "or top-level expression_source using ASCII editor syntax: dx(u), "
        "dy(u), dt(u), nested calls for second derivatives, explicit *, and "
        "ASCII parameter names. Never emit Delta, nabla, unicode subscripts, "
        "or implicit multiplication in that expression. "
        "warnings must be an array of strings or objects with code, path, and "
        "message strings. Do not include ready_for_validation: the application "
        "decides readiness from physics-family consistency, not from an "
        "elasticity schema.\n"
        "Example 1 input: " + _HEAT_FEWSHOT_INPUT + "\n"
        "Example 1 output:\n"
        + json.dumps(_HEAT_FEWSHOT_OUTPUT, ensure_ascii=False) + "\n"
        "Example 2 input: " + _POISSON_FEWSHOT_INPUT + "\n"
        "Example 2 output:\n"
        + json.dumps(_POISSON_FEWSHOT_OUTPUT, ensure_ascii=False) + "\n"
        "Input:\n" + json.dumps(context, ensure_ascii=False)
    )


def _ensure_json_value(value: Any, *, depth: int = 0) -> None:
    if depth > 12:
        raise LLMPayloadError("LLM draft nesting is too deep")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LLMPayloadError("LLM output numbers must be finite")
        return
    if isinstance(value, list):
        for item in value:
            _ensure_json_value(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise LLMPayloadError("LLM draft keys must be strings")
            if key.lower() in _FORBIDDEN_DRAFT_KEYS:
                raise LLMPayloadError(
                    f"LLM draft contains out-of-scope key {key!r}"
                )
            _ensure_json_value(item, depth=depth + 1)
        return
    raise LLMPayloadError(
        f"LLM output contains non-JSON value {type(value).__name__}"
    )


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise LLMPayloadError(f"LLM field {name!r} must be an array of strings")
    return list(value)


def _warning_list(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise LLMPayloadError("LLM field 'warnings' must be an array")
    result: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            result.append({
                "code": "understanding.llm_warning",
                "path": "draft",
                "message": item,
            })
            continue
        if not isinstance(item, dict):
            raise LLMPayloadError(
                f"LLM warning at index {index} must be a string or object"
            )
        allowed = {"code", "path", "message"}
        if set(item) != allowed or not all(
            isinstance(item[key], str) and item[key] for key in allowed
        ):
            raise LLMPayloadError(
                "LLM warning objects require non-empty code, path, and message"
            )
        result.append({key: item[key] for key in ("code", "path", "message")})
    return result


def _parse_extraction(content: str) -> LLMExtraction:
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise LLMPayloadError("LLM message content is not valid JSON") from exc
    if not isinstance(data, dict):
        raise LLMPayloadError("LLM output must be a JSON object")
    expected = {"draft", "missing", "questions", "warnings"}
    if set(data) != expected:
        raise LLMPayloadError(
            "LLM output must contain exactly draft, missing, questions, warnings"
        )
    if not isinstance(data["draft"], dict):
        raise LLMPayloadError("LLM field 'draft' must be an object")
    _ensure_json_value(data["draft"])
    return LLMExtraction(
        draft=data["draft"],
        missing=_string_list(data["missing"], "missing"),
        questions=_string_list(data["questions"], "questions"),
        warnings=_warning_list(data["warnings"]),
    )


def extract_with_local_llm(
    text: str,
    *,
    path: str | None = None,
    answers: Mapping[str, Any] | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout: float | None = None,
    api_key: str | None = None,
) -> LLMExtraction:
    """Call an OpenAI-compatible chat endpoint and validate its JSON.

    ``api_key`` is optional for local endpoints.  When provided (directly or
    through ``OPENPINN_UNDERSTAND_API_KEY``), it is sent only in the HTTP
    ``Authorization`` header and is never included in prompts or errors.
    """
    resolved_url = (
        base_url
        or os.getenv("OPENPINN_UNDERSTAND_BASE_URL")
        or os.getenv("OPENPINN_OLLAMA_BASE_URL", DEFAULT_BASE_URL)
    )
    resolved_model = (
        model
        or os.getenv("OPENPINN_UNDERSTAND_MODEL")
        or os.getenv("OPENPINN_OLLAMA_MODEL", DEFAULT_MODEL)
    )
    resolved_api_key = api_key or os.getenv("OPENPINN_UNDERSTAND_API_KEY")
    resolved_timeout = (
        timeout
        if timeout is not None
        else float(
            os.getenv(
                "OPENPINN_UNDERSTAND_TIMEOUT",
                os.getenv("OPENPINN_OLLAMA_TIMEOUT", str(DEFAULT_TIMEOUT)),
            )
        )
    )
    if not resolved_model.strip():
        raise ValueError("LLM model must not be empty")
    if resolved_timeout <= 0:
        raise ValueError("LLM timeout must be positive")
    if resolved_api_key is not None:
        resolved_api_key = resolved_api_key.strip()
        if not resolved_api_key:
            resolved_api_key = None
        elif "\r" in resolved_api_key or "\n" in resolved_api_key:
            raise ValueError("LLM api_key must not contain line breaks")

    payload = {
        "model": resolved_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You extract PDE problem facts as one JSON object. "
                    "You may normalize obvious notation and infer a domain only "
                    "from stated endpoints, and you must mark every inferred "
                    "fact in warnings with code understanding.inferred. Never "
                    "inject linear-elasticity fields into another declared "
                    "physics family. Include an editor-legal ASCII "
                    "pde.expression for every representable single-field PDE. "
                    "If physics is unknown, ask for its type. "
                    "Output JSON only."
                ),
            },
            {"role": "user", "content": _problem_prompt(text, path, answers)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if resolved_api_key is not None:
        headers["Authorization"] = f"Bearer {resolved_api_key}"
    request = urllib.request.Request(
        _chat_endpoint(resolved_url),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=resolved_timeout) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise LLMConnectionError(
            f"LLM endpoint returned HTTP {exc.code}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
        raise LLMConnectionError("LLM endpoint is unreachable") from exc
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise LLMPayloadError("LLM response is too large")
    try:
        envelope = json.loads(raw.decode("utf-8"))
        content = envelope["choices"][0]["message"]["content"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise LLMPayloadError(
            "LLM response does not match the OpenAI chat envelope"
        ) from exc
    if not isinstance(content, str):
        raise LLMPayloadError("LLM message content must be a JSON string")
    return _parse_extraction(content)


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "LLMConnectionError",
    "LLMExtraction",
    "LLMPayloadError",
    "LLMUnderstandingError",
    "extract_with_local_llm",
]
