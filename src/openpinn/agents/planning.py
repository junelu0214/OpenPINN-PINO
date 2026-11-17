"""LLM-primary solver planning for structured OpenPINN problems.

Planning recommends a validated ``MethodSpec`` draft.  It never generates code,
starts training, or replaces the Specification Validation Agent gate.
"""

from __future__ import annotations

import copy
import json
import math
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from ..spec import ProblemSpec, load_method_spec
from .llm import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    LLMConnectionError,
    LLMPayloadError,
    _chat_endpoint,
)


_MAX_RESPONSE_BYTES = 1_000_000
_KNOWN_FAMILIES = {
    "linear_elasticity", "heat", "heat_equation", "diffusion", "poisson"
}
_ALLOWED_HINTS = {
    "method", "variant", "path", "sampling", "loss", "optimizer",
    "network", "differentiator", "training",
}
_FORBIDDEN_PLAN_KEYS = {"code", "python", "script", "training_script"}


@dataclass
class PlanReport:
    """A MethodSpec-aligned draft and the reasons behind the recommendation."""

    method_draft: dict[str, Any]
    rationale: list[str] = field(default_factory=list)
    warnings: list[Any] = field(default_factory=list)
    ready_for_validation: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "method_draft": copy.deepcopy(self.method_draft),
            "rationale": list(self.rationale),
            "warnings": copy.deepcopy(self.warnings),
            "ready_for_validation": self.ready_for_validation,
        }


def _warning(code: str, message: str, path: str = "method_draft") -> dict[str, str]:
    return {"code": code, "path": path, "message": message}


def _json_value(value: Any, *, depth: int = 0) -> None:
    if depth > 12:
        raise LLMPayloadError("Planning output nesting is too deep")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LLMPayloadError("Planning output numbers must be finite")
        return
    if isinstance(value, list):
        for item in value:
            _json_value(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise LLMPayloadError("Planning output keys must be strings")
            _json_value(item, depth=depth + 1)
        return
    raise LLMPayloadError(
        f"Planning output contains non-JSON value {type(value).__name__}"
    )


def _warning_list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise LLMPayloadError("Planning warnings must be an array")
    result: list[Any] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
        elif isinstance(item, dict):
            if set(item) != {"code", "path", "message"} or not all(
                isinstance(item[key], str) and item[key].strip()
                for key in ("code", "path", "message")
            ):
                raise LLMPayloadError(
                    "Planning warning objects require code, path, and message"
                )
            result.append({key: item[key] for key in ("code", "path", "message")})
        else:
            raise LLMPayloadError(
                "Planning warnings must contain non-empty strings or warning objects"
            )
    return result


def _parse_plan(content: str) -> tuple[dict[str, Any], list[str], list[Any]]:
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise LLMPayloadError("Planning message content is not valid JSON") from exc
    if not isinstance(data, dict) or set(data) != {
        "method_draft", "rationale", "warnings"
    }:
        raise LLMPayloadError(
            "Planning output must contain exactly method_draft, rationale, warnings"
        )
    if not isinstance(data["method_draft"], dict):
        raise LLMPayloadError("Planning method_draft must be an object")
    def reject_executable_keys(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in _FORBIDDEN_PLAN_KEYS:
                    raise LLMPayloadError(
                        f"Planning method_draft contains forbidden key {key!r}"
                    )
                reject_executable_keys(item)
        elif isinstance(value, list):
            for item in value:
                reject_executable_keys(item)
    reject_executable_keys(data["method_draft"])
    rationale = data["rationale"]
    if not isinstance(rationale, list) or not rationale or not all(
        isinstance(item, str) and item.strip() for item in rationale
    ):
        raise LLMPayloadError("Planning rationale must be a non-empty string array")
    _json_value(data["method_draft"])
    return copy.deepcopy(data["method_draft"]), [item.strip() for item in rationale], _warning_list(data["warnings"])


def _problem_dict(problem: ProblemSpec | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(problem, ProblemSpec):
        return problem.model_dump(mode="json")
    if not isinstance(problem, Mapping):
        raise TypeError("problem must be a ProblemSpec or mapping")
    result = copy.deepcopy(dict(problem))
    _json_value(result)
    return result


def _physics_family(problem: Mapping[str, Any]) -> str | None:
    value = problem.get("physics")
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    pde = problem.get("pde")
    if isinstance(pde, Mapping):
        value = pde.get("type")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def _resolve_path(
    problem: Mapping[str, Any],
    path: Literal["pinn", "operator"] | None,
    hints: Mapping[str, Any] | None,
) -> tuple[Literal["pinn", "operator"], list[Any]]:
    warnings: list[Any] = []
    candidate = path
    if candidate is None and hints and hints.get("path") in ("pinn", "operator"):
        candidate = hints["path"]
    if candidate is None and problem.get("path") in ("pinn", "operator"):
        candidate = problem["path"]
    if candidate is None:
        candidate = "pinn"
        warnings.append(_warning(
            "planning.path_defaulted",
            "问题未声明求解路径，规划草稿保守采用 PINN；请在校验前核对。",
            "path",
        ))
    if candidate not in ("pinn", "operator"):
        raise ValueError("path must be 'pinn', 'operator', or None")
    return candidate, warnings


def _default_method(path: str, family: str | None) -> tuple[dict[str, Any], list[str]]:
    transient = family in {"heat", "heat_equation", "diffusion"}
    if path == "operator":
        return ({
            "method": "deeponet", "path": "operator",
            "sampling": {"name": "uniform", "interior": 512, "boundary": 128},
            "loss": {"formulation": "strong", "weighting": "fixed"},
            "optimizer": {"name": "adam", "lr": 0.0005},
            "network": {"name": "mlp", "hidden_layers": [128, 128], "activation": "tanh"},
            "differentiator": {"mode": "autodiff"},
            "training": {"steps": 5000},
        }, ["算子路径采用 DeepONet 草稿，并用查询坐标自动微分表达物理约束。"])
    sampling: dict[str, Any] = {"name": "uniform", "interior": 512, "boundary": 128}
    if transient:
        sampling["initial"] = 128
    family_reason = {
        "linear_elasticity": "二维线弹性采用 MLP PINN 与坐标自动微分计算位移导数。",
        "poisson": "Poisson 椭圆问题采用强形式 PINN 与均匀域内、边界采样。",
        "heat": "瞬态热传导采用强形式 PINN，并为初始条件保留采样预算。",
        "heat_equation": "瞬态热传导采用强形式 PINN，并为初始条件保留采样预算。",
        "diffusion": "扩散问题采用强形式 PINN，并为初始条件保留采样预算。",
    }.get(family, "未知物理族仅生成保守 PINN 草稿，必须人工核对后再交给 Validation Agent。")
    return ({
        "method": "baseline", "variant": "plain_pinn", "path": "pinn",
        "sampling": sampling,
        "loss": {"formulation": "strong", "weighting": "fixed", "weights": {"pde": 1.0, "boundary": 10.0}},
        "optimizer": {"name": "adam", "lr": 0.001},
        "network": {"name": "mlp", "hidden_layers": [64, 64, 64, 64], "activation": "tanh"},
        "differentiator": {"mode": "autodiff"},
        "training": {"steps": 5000},
    }, [family_reason])


def _merge_dict(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _apply_hints(draft: dict[str, Any], hints: Mapping[str, Any] | None) -> tuple[dict[str, Any], list[Any]]:
    if hints is None:
        return draft, []
    if not isinstance(hints, Mapping):
        raise TypeError("hints must be a mapping or None")
    unknown = sorted(set(hints) - _ALLOWED_HINTS)
    warnings = [
        _warning("planning.hint_ignored", f"忽略不支持的规划提示 {name!r}。", f"hints.{name}")
        for name in unknown
    ]
    allowed = {key: value for key, value in hints.items() if key in _ALLOWED_HINTS}
    return _merge_dict(draft, allowed), warnings


def _finalize(
    draft: dict[str, Any], rationale: list[str], warnings: list[Any],
    family: str | None, hints: Mapping[str, Any] | None,
) -> PlanReport:
    draft, hint_warnings = _apply_hints(draft, hints)
    warnings.extend(hint_warnings)
    ready = True
    try:
        spec = load_method_spec(draft)
        draft = spec.model_dump(mode="json")
    except Exception as exc:
        ready = False
        warnings.append(_warning(
            "planning.invalid_method_draft",
            "方法草稿未通过 MethodSpec/注册表校验：" + str(exc),
        ))
    if family not in _KNOWN_FAMILIES:
        ready = False
        warnings.append(_warning(
            "planning.unknown_physics",
            "物理族未知或尚未覆盖；保守草稿仅供查看，不得声称可生码。",
            "problem.physics",
        ))
    return PlanReport(draft, rationale, warnings, ready)


_ELASTICITY_EXAMPLE = {
    "method_draft": {
        "method": "baseline", "variant": "plain_pinn", "path": "pinn",
        "sampling": {"name": "uniform", "interior": 512, "boundary": 128},
        "loss": {"formulation": "strong", "weighting": "fixed", "weights": {"pde": 1.0, "boundary": 10.0}},
        "optimizer": {"name": "adam", "lr": 0.001},
        "network": {"name": "mlp", "hidden_layers": [64, 64, 64, 64], "activation": "tanh"},
        "differentiator": {"mode": "autodiff"}, "training": {"steps": 5000},
    },
    "rationale": ["二维线弹性采用 MLP PINN 与坐标自动微分计算位移导数。"],
    "warnings": [],
}


def _planning_prompt(problem: Mapping[str, Any], path: str, hints: Mapping[str, Any] | None) -> str:
    return (
        "Plan a solver configuration for the structured PDE problem. Return exactly one JSON object "
        "with keys method_draft, rationale, warnings. method_draft must match OpenPINN MethodSpec: "
        "method, optional variant, path, sampling, loss, optimizer, network, differentiator, training. "
        "Use registered names only. For PINN use baseline/plain_pinn, uniform, strong, fixed, adam, "
        "mlp, and autodiff unless the problem gives a concrete reason to choose another registered "
        "component. rationale must be a non-empty array of short Chinese strings. warnings may contain "
        "strings or objects with code, path, message. Do not return Python, executable code, scripts, "
        "training results, ready flags, or problem extraction. Planning does not authorize codegen; "
        "the caller must run validate_specs(problem, method_draft) with the Specification Validation Agent.\nExample output:\n" +
        json.dumps(_ELASTICITY_EXAMPLE, ensure_ascii=False) + "\nInput:\n" +
        json.dumps({"problem": problem, "path": path, "hints": hints or {}}, ensure_ascii=False)
    )


def _request_llm(
    problem: Mapping[str, Any], path: str, hints: Mapping[str, Any] | None,
    *, base_url: str | None, model: str | None, api_key: str | None,
    timeout: float | None,
) -> tuple[dict[str, Any], list[str], list[Any]]:
    resolved_url = base_url or os.getenv("OPENPINN_PLAN_BASE_URL") or os.getenv("OPENPINN_UNDERSTAND_BASE_URL") or os.getenv("OPENPINN_OLLAMA_BASE_URL", DEFAULT_BASE_URL)
    resolved_model = model or os.getenv("OPENPINN_PLAN_MODEL") or os.getenv("OPENPINN_UNDERSTAND_MODEL") or os.getenv("OPENPINN_OLLAMA_MODEL", DEFAULT_MODEL)
    resolved_key = api_key or os.getenv("OPENPINN_PLAN_API_KEY") or os.getenv("OPENPINN_UNDERSTAND_API_KEY")
    resolved_timeout = timeout if timeout is not None else float(os.getenv("OPENPINN_PLAN_TIMEOUT") or os.getenv("OPENPINN_UNDERSTAND_TIMEOUT") or os.getenv("OPENPINN_OLLAMA_TIMEOUT", str(DEFAULT_TIMEOUT)))
    if not resolved_model.strip():
        raise ValueError("Planning model must not be empty")
    if resolved_timeout <= 0:
        raise ValueError("Planning timeout must be positive")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if resolved_key:
        if "\r" in resolved_key or "\n" in resolved_key:
            raise ValueError("Planning api_key must not contain line breaks")
        headers["Authorization"] = "Bearer " + resolved_key.strip()
    payload = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": "You recommend a registered OpenPINN MethodSpec as strict JSON only. Never output executable code or claim validation succeeded."},
            {"role": "user", "content": _planning_prompt(problem, path, hints)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        _chat_endpoint(resolved_url),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=resolved_timeout) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise LLMConnectionError(f"Planning endpoint returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
        raise LLMConnectionError("Planning endpoint is unreachable") from exc
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise LLMPayloadError("Planning response is too large")
    try:
        envelope = json.loads(raw.decode("utf-8"))
        content = envelope["choices"][0]["message"]["content"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise LLMPayloadError("Planning response does not match the OpenAI chat envelope") from exc
    if not isinstance(content, str):
        raise LLMPayloadError("Planning message content must be a JSON string")
    return _parse_plan(content)


def plan_method(
    problem: ProblemSpec | dict[str, Any],
    *,
    path: Literal["pinn", "operator"] | None = None,
    hints: dict[str, Any] | None = None,
    backend: Literal["llm", "rules", "auto"] = "auto",
    base_url: str | None = None,
    model: str | None = None,
    timeout: float | None = None,
    api_key: str | None = None,
) -> PlanReport:
    """Recommend a MethodSpec draft without generating code or training.

    ``auto`` uses the LLM when a planning or Understanding API key is present;
    otherwise it returns the deterministic rules fallback with an explicit
    warning.  Codegen callers must separately require ``validate(...).ok``.
    """
    if backend not in ("llm", "rules", "auto"):
        raise ValueError("backend must be 'llm', 'rules', or 'auto'")
    problem_data = _problem_dict(problem)
    family = _physics_family(problem_data)
    resolved_path, path_warnings = _resolve_path(problem_data, path, hints)
    if backend == "rules":
        draft, rationale = _default_method(resolved_path, family)
        return _finalize(draft, rationale, path_warnings, family, hints)

    has_key = bool(api_key or os.getenv("OPENPINN_PLAN_API_KEY") or os.getenv("OPENPINN_UNDERSTAND_API_KEY"))
    if backend == "auto" and not has_key:
        draft, rationale = _default_method(resolved_path, family)
        path_warnings.append(_warning(
            "planning.rules_fallback",
            "未配置 Planning/Understanding API Key，已使用确定性 rules fallback。",
            "backend",
        ))
        return _finalize(draft, rationale, path_warnings, family, hints)
    try:
        draft, rationale, llm_warnings = _request_llm(
            problem_data, resolved_path, hints, base_url=base_url, model=model,
            api_key=api_key, timeout=timeout,
        )
    except (LLMConnectionError, LLMPayloadError, ValueError) as exc:
        if backend == "auto":
            draft, rationale = _default_method(resolved_path, family)
            path_warnings.append(_warning(
                "planning.rules_fallback",
                "LLM 规划失败，已使用确定性 rules fallback：" + str(exc),
                "backend",
            ))
            return _finalize(draft, rationale, path_warnings, family, hints)
        return PlanReport({}, [], path_warnings + [_warning(
            "planning.llm_failed", "LLM 规划失败：" + str(exc), "backend"
        )], False)
    return _finalize(draft, rationale, path_warnings + llm_warnings, family, hints)


__all__ = ["PlanReport", "plan_method"]
