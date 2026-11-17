"""Physics-family readiness gate for Problem Understanding.

Understanding asks follow-up questions and sets ``ready_for_validation`` from
the draft's declared physics family. This module does not call the Validation Agent
``validate`` and must not invent elasticity-only gaps (E, nu, ux, uy) for
heat, Poisson, another declared family, or an unknown family.

The Specification Validation Agent remains the later final-spec gate used by configurator YAML and
CodeBuilder. A family-complete heat or Poisson draft can be ready here and
still fail that later gate.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping


_ELASTICITY_TOP_LEVEL = (
    "dim",
    "physics",
    "geometry",
    "field",
    "constitutive",
    "boundaries",
)

_QUESTIONS = {
    "dim": "问题是二维的吗？请说明空间维数。",
    "physics": "要描述哪一种物理方程或本构问题？",
    "geometry": "请给出求解区域的范围。",
    "geometry.outer": "请给出矩形区域的 x/y 范围。",
    "geometry.holes": "请检查孔洞的中心、半径及其是否严格位于外边界内。",
    "field": "未知场是什么？例如位移分量 ux、uy。",
    "constitutive.E": "材料的杨氏模量 E 是多少？请同时注明单位。",
    "constitutive.nu": "材料的泊松比 nu 是多少？",
    "boundaries": "请说明每条外边界上的 Dirichlet/Neumann 条件。",
    "boundaries.left": "左边界条件是什么？",
    "boundaries.right": "右边界条件是什么？",
    "boundaries.top": "上边界条件是什么？",
    "boundaries.bottom": "下边界条件是什么？",
    "boundaries.hole_0": "圆孔边界条件是什么？",
    "pde.diffusivity": "热传导系数或扩散系数是多少？",
    "pde.source": "方程的源项 f 是什么？",
    "pde": "请给出该物理问题的控制方程或 PDE 描述。",
    "initial_condition": "初始条件是什么？",
    "L": "L 的数值是多少？",
}

_KNOWN_SYMBOLS = {
    "x", "y", "z", "t", "pi", "sin", "cos", "tan", "exp", "log", "ln",
    "sqrt", "abs", "pow", "sinh", "cosh", "tanh",
}

_ELASTICITY_NAMES = {"linear_elasticity", "elasticity", "linear_elastic"}
_HEAT_NAMES = {
    "heat",
    "heat_equation",
    "heat_conduction",
    "heat_transfer",
    "diffusion",
    "thermal",
}
_POISSON_NAMES = {"poisson", "poisson_equation"}
_UNKNOWN_NAMES = {"unknown", "unknown_physics", "unspecified", "other", "其他", "未知"}

# Elasticity-only facts and the legacy validator boilerplate. Applied only when the
# declared family is not linear elasticity, except the schema/boilerplate
# clause which is never a physics-family gap.
_POLLUTION = re.compile(
    r"("
    r"constitutive(?:\.(?:e|nu))?\b"
    r"|young(?:['’]s)?(?:\s+modulus)?"
    r"|poisson(?:['’]s)?\s+ratio"
    r"|杨氏模量"
    r"|泊松比"
    r"|isotropic[_ -]?linear"
    r"|stress[_ -]?voigt"
    r"|\bnu\b"
    r"|ν"
    r"|\bux\b"
    r"|\buy\b"
    r"|displacement"
    r"|位移"
    r")",
    re.IGNORECASE,
)
_SCHEMA_NOISE = re.compile(
    r"("
    r"^problem\."
    r"|agent a"
    r"|尚不能验证"
    r"|only linear_elasticity"
    r"|only dim\s*=\s*2"
    r"|extra fields are not permitted"
    r")",
    re.IGNORECASE,
)
_BOUNDARY_VALUE = re.compile(r"boundaries\[(\d+)\]\.value")
_BOUNDARY_TYPE = re.compile(r"boundaries\[(\d+)\]\.type")


@dataclass(frozen=True)
class FamilyAssessment:
    """Gaps that keep a draft from being ready for its physics family."""

    family: str
    missing: list[str]
    ready: bool


def questions_for(missing: list[str]) -> list[str]:
    """Chinese follow-up questions for family gaps."""
    result: list[str] = []
    for item in missing:
        if item in _QUESTIONS:
            result.append(_QUESTIONS[item])
        elif _BOUNDARY_VALUE.fullmatch(item):
            result.append("请给出该边界条件的数值。")
        elif _BOUNDARY_TYPE.fullmatch(item):
            result.append("请说明该边界是 Dirichlet、Neumann 还是 Robin。")
        else:
            result.append(f"请补充 {item}。")
    return result


def assess_family(draft: Mapping[str, Any] | None) -> FamilyAssessment:
    """Return family gaps. ``ready`` is true only when ``missing`` is empty."""
    if not isinstance(draft, Mapping):
        return FamilyAssessment("unknown", ["physics"], False)
    family = _family_name(draft.get("physics"))
    if family == "linear_elasticity":
        missing = _elasticity_gaps(draft)
    elif family == "heat":
        missing = _heat_gaps(draft)
    elif family == "poisson":
        missing = _poisson_gaps(draft)
    elif family != "unknown":
        missing = _generic_gaps(draft, family)
    else:
        missing = ["physics"]
    unique = list(dict.fromkeys(missing))
    return FamilyAssessment(family, unique, not unique)


def reconcile_llm_report(
    draft: Mapping[str, Any],
    llm_missing: list[str],
    llm_questions: list[str],
) -> tuple[list[str], list[str], bool]:
    """Merge model suggestions with the family gate.

    The draft is the source of readiness. Elasticity-only suggestions are
    dropped for every other family. When the draft is already consistent,
    model-supplied missing fields and questions are not kept.
    """
    assessment = assess_family(draft)
    if assessment.ready:
        return [], [], True
    if assessment.family == "unknown":
        return ["physics"], questions_for(["physics"]), False
    missing = list(assessment.missing)
    for item in _filter_llm_items(llm_missing, assessment.family):
        if item not in missing:
            missing.append(item)
    questions_in = _filter_llm_items(llm_questions, assessment.family)
    if questions_in:
        questions = list(dict.fromkeys(questions_in))
    else:
        questions = questions_for(assessment.missing)
    return missing, questions, False


def _family_name(physics: object) -> str:
    if not isinstance(physics, str) or not physics.strip():
        return "unknown"
    key = re.sub(r"[\s-]+", "_", physics.strip().lower())
    if key in _ELASTICITY_NAMES:
        return "linear_elasticity"
    if key in _HEAT_NAMES:
        return "heat"
    if key in _POISSON_NAMES:
        return "poisson"
    if key in _UNKNOWN_NAMES:
        return "unknown"
    return key


def _filter_llm_items(items: list[str], family: str) -> list[str]:
    kept: list[str] = []
    for item in items:
        if _SCHEMA_NOISE.search(item):
            continue
        if family != "linear_elasticity" and _POLLUTION.search(item):
            continue
        kept.append(item)
    return kept


def _finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _ordered(low: object, high: object) -> bool:
    return _finite(low) and _finite(high) and high > low


def _defined_names(draft: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for key in ("parameters", "constants", "coefficients"):
        value = draft.get(key)
        if isinstance(value, Mapping):
            for name, item in value.items():
                if isinstance(name, str) and (_finite(item) or isinstance(item, str)):
                    names.add(name)
    for key, value in draft.items():
        if isinstance(key, str) and _finite(value):
            names.add(key)
    field = draft.get("field")
    if isinstance(field, Mapping) and isinstance(field.get("components"), list):
        names.update(
            component for component in field["components"] if isinstance(component, str)
        )
    return names


def _free_symbols(expression: str, defined: set[str]) -> list[str]:
    found: list[str] = []
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|π", expression):
        if token == "π" or token.lower() in _KNOWN_SYMBOLS or token in defined:
            continue
        if token not in found:
            found.append(token)
    return found


def _concrete(value: object, defined: set[str]) -> bool:
    if _finite(value):
        return True
    if isinstance(value, (list, tuple)) and value and all(_finite(item) for item in value):
        return True
    if isinstance(value, str) and value.strip() and not _free_symbols(value, defined):
        return True
    return False


def _geometry_ok(geometry: object) -> bool:
    if not isinstance(geometry, Mapping):
        return False
    if _ordered(geometry.get("x_min"), geometry.get("x_max")):
        if "y_min" in geometry or "y_max" in geometry:
            return _ordered(geometry.get("y_min"), geometry.get("y_max"))
        return True
    outer = geometry.get("outer")
    if isinstance(outer, Mapping) and _ordered(outer.get("x_min"), outer.get("x_max")):
        if "y_min" in outer or "y_max" in outer:
            return _ordered(outer.get("y_min"), outer.get("y_max"))
        return True
    bounds = geometry.get("bounds")
    if isinstance(bounds, Mapping) and bounds:
        return all(
            isinstance(pair, (list, tuple))
            and len(pair) == 2
            and _ordered(pair[0], pair[1])
            for pair in bounds.values()
        )
    domain = geometry.get("domain")
    return (
        isinstance(domain, (list, tuple))
        and len(domain) == 2
        and _ordered(domain[0], domain[1])
    )


def _lookup_diffusivity(draft: Mapping[str, Any]) -> object:
    pde = draft.get("pde") if isinstance(draft.get("pde"), Mapping) else {}
    constitutive = (
        draft.get("constitutive") if isinstance(draft.get("constitutive"), Mapping) else {}
    )
    for container, key in (
        (pde, "diffusivity"),
        (pde, "alpha"),
        (pde, "coefficient"),
        (draft, "diffusivity"),
        (draft, "alpha"),
        (draft, "thermal_diffusivity"),
        (constitutive, "alpha"),
        (constitutive, "diffusivity"),
    ):
        if isinstance(container, Mapping) and key in container:
            return container[key]
    return None


def _diffusivity_ok(draft: Mapping[str, Any]) -> bool:
    value = _lookup_diffusivity(draft)
    return _finite(value) and value >= 0


def _source_ok(draft: Mapping[str, Any]) -> bool:
    pde = draft.get("pde") if isinstance(draft.get("pde"), Mapping) else {}
    if "source" in pde:
        source = pde["source"]
    elif "f" in pde:
        source = pde["f"]
    elif "source" in draft:
        source = draft["source"]
    else:
        return False
    defined = _defined_names(draft)
    if isinstance(source, Mapping):
        if "value" in source and _finite(source.get("value")):
            return True
        expression = source.get("expression")
        return isinstance(expression, str) and not _free_symbols(expression, defined)
    if _finite(source):
        return True
    return isinstance(source, str) and bool(source.strip()) and not _free_symbols(source, defined)


def _listed_boundary_gaps(boundaries: object, defined: set[str]) -> list[str]:
    if not isinstance(boundaries, list) or not boundaries:
        return ["boundaries"]
    gaps: list[str] = []
    for index, bc in enumerate(boundaries):
        if not isinstance(bc, Mapping):
            gaps.append(f"boundaries[{index}]")
            continue
        if bc.get("type") not in {"dirichlet", "neumann", "robin"}:
            gaps.append(f"boundaries[{index}].type")
        if not _concrete(bc.get("value"), defined):
            gaps.append(f"boundaries[{index}].value")
    return gaps


def _side_gaps(boundaries: object) -> list[str]:
    if not isinstance(boundaries, list):
        return []
    regions: set[str] = set()
    for bc in boundaries:
        if isinstance(bc, Mapping) and isinstance(bc.get("region"), str):
            regions.add(bc["region"].lower())
    if regions & {"all", "boundary", "domain_boundary", "outer"}:
        return []
    return [
        f"boundaries.{name}"
        for name in ("left", "right", "top", "bottom")
        if name not in regions
    ]


def _is_steady(draft: Mapping[str, Any]) -> bool:
    if draft.get("steady") is True:
        return True
    pde = draft.get("pde")
    return isinstance(pde, Mapping) and pde.get("steady") is True


def _initial_condition_gaps(draft: Mapping[str, Any]) -> list[str]:
    if _is_steady(draft):
        return []
    initial = draft.get("initial_condition")
    if initial is None:
        return ["initial_condition"]
    defined = _defined_names(draft)
    if isinstance(initial, str):
        return _free_symbols(initial, defined)
    if isinstance(initial, Mapping):
        expression = initial.get("expression")
        if isinstance(expression, str):
            return _free_symbols(expression, defined)
        if "value" in initial and _finite(initial.get("value")):
            return []
    return ["initial_condition"]


def _elasticity_gaps(draft: Mapping[str, Any]) -> list[str]:
    missing = [key for key in _ELASTICITY_TOP_LEVEL if key not in draft]
    geometry = draft.get("geometry")
    if isinstance(geometry, Mapping) and "outer" not in geometry:
        missing.append("geometry.outer")
    constitutive = draft.get("constitutive")
    if isinstance(constitutive, Mapping):
        for key in ("E", "nu"):
            if key not in constitutive:
                missing.append(f"constitutive.{key}")
        if "E" in constitutive and not (
            _finite(constitutive.get("E")) and constitutive["E"] > 0
        ):
            missing.append("constitutive.E")
        if "nu" in constitutive and not (
            _finite(constitutive.get("nu")) and -1 <= constitutive["nu"] < 0.5
        ):
            missing.append("constitutive.nu")
    if isinstance(geometry, Mapping):
        outer = geometry.get("outer")
        if isinstance(outer, Mapping) and {"x_min", "x_max", "y_min", "y_max"} & set(outer):
            if not (
                _ordered(outer.get("x_min"), outer.get("x_max"))
                and _ordered(outer.get("y_min"), outer.get("y_max"))
            ):
                missing.append("geometry.outer")
        holes = geometry.get("holes")
        if isinstance(holes, list) and isinstance(outer, Mapping):
            for hole in holes:
                center = hole.get("center") if isinstance(hole, Mapping) else None
                radius = hole.get("radius") if isinstance(hole, Mapping) else None
                if not (
                    isinstance(hole, Mapping)
                    and hole.get("type") == "circle"
                    and isinstance(center, (list, tuple))
                    and len(center) == 2
                    and all(_finite(item) for item in center)
                    and _finite(radius)
                    and radius > 0
                    and _finite(outer.get("x_min"))
                    and _finite(outer.get("x_max"))
                    and _finite(outer.get("y_min"))
                    and _finite(outer.get("y_max"))
                    and outer["x_min"] < center[0] - radius
                    and center[0] + radius < outer["x_max"]
                    and outer["y_min"] < center[1] - radius
                    and center[1] + radius < outer["y_max"]
                ):
                    missing.append("geometry.holes")
                    break
    boundaries = draft.get("boundaries")
    if isinstance(boundaries, list):
        regions = {
            bc.get("region") for bc in boundaries if isinstance(bc, Mapping)
        }
        required_regions = {"left", "right", "top", "bottom"}
        holes = geometry.get("holes", []) if isinstance(geometry, Mapping) else []
        required_regions.update(f"hole_{index}" for index in range(len(holes)))
        missing.extend(
            f"boundaries.{region}" for region in sorted(required_regions - regions)
        )
        defined = _defined_names(draft)
        for index, bc in enumerate(boundaries):
            if not isinstance(bc, Mapping):
                missing.append(f"boundaries[{index}]")
                continue
            if not _concrete(bc.get("value"), defined):
                region = bc.get("region")
                missing.append(
                    f"boundaries.{region}" if isinstance(region, str)
                    else f"boundaries[{index}].value"
                )
    return missing


def _heat_gaps(draft: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    if draft.get("dim") not in (1, 2, 3):
        missing.append("dim")
    if not _geometry_ok(draft.get("geometry")):
        missing.append("geometry")
    if not _diffusivity_ok(draft):
        missing.append("pde.diffusivity")
    defined = _defined_names(draft)
    missing.extend(_listed_boundary_gaps(draft.get("boundaries"), defined))
    missing.extend(_initial_condition_gaps(draft))
    return missing


def _poisson_gaps(draft: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    dim = draft.get("dim")
    if dim not in (1, 2, 3):
        missing.append("dim")
    geometry_ok = _geometry_ok(draft.get("geometry"))
    if not geometry_ok:
        missing.append("geometry")
    if not _source_ok(draft):
        missing.append("pde.source")
    defined = _defined_names(draft)
    boundaries = draft.get("boundaries")
    missing.extend(_listed_boundary_gaps(boundaries, defined))
    if dim == 2 and geometry_ok:
        missing.extend(_side_gaps(boundaries))
    return missing


def _field_ok(field: object) -> bool:
    if isinstance(field, str):
        return bool(field.strip())
    if not isinstance(field, Mapping):
        return False
    components = field.get("components")
    return (
        isinstance(field.get("name"), str) and bool(field["name"].strip())
    ) or (
        isinstance(components, list)
        and bool(components)
        and all(isinstance(item, str) and bool(item.strip()) for item in components)
    ) or (isinstance(field.get("kind"), str) and bool(field["kind"].strip()))


def _pde_ok(draft: Mapping[str, Any]) -> bool:
    for key in ("expression_source", "equation", "governing_equation"):
        value = draft.get(key)
        if isinstance(value, str) and value.strip():
            return True
    pde = draft.get("pde")
    if isinstance(pde, str):
        return bool(pde.strip())
    return isinstance(pde, Mapping) and bool(pde)


def _generic_dynamic(draft: Mapping[str, Any], family: str) -> bool:
    if draft.get("kind") == "dynamic" or draft.get("time_dependent") is True:
        return True
    pde = draft.get("pde")
    if isinstance(pde, Mapping) and (
        pde.get("kind") == "dynamic" or pde.get("time_dependent") is True
    ):
        return True
    return any(token in family for token in ("wave", "transient", "evolution", "dynamic"))


def _generic_initial_ok(value: object, defined: set[str]) -> bool:
    if _finite(value):
        return True
    if isinstance(value, str):
        return bool(value.strip()) and not _free_symbols(value, defined)
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_generic_initial_ok(item, defined) for item in value)
    if isinstance(value, Mapping):
        if not value:
            return False
        if "expression" in value:
            return _generic_initial_ok(value["expression"], defined)
        if "value" in value:
            return _generic_initial_ok(value["value"], defined)
        return all(_generic_initial_ok(item, defined) for item in value.values())
    return False


def _generic_gaps(draft: Mapping[str, Any], family: str) -> list[str]:
    """Common problem-side facts for a declared, otherwise arbitrary family."""
    missing: list[str] = []
    if draft.get("dim") not in (1, 2, 3):
        missing.append("dim")
    if not _geometry_ok(draft.get("geometry")):
        missing.append("geometry")
    if not _field_ok(draft.get("field")):
        missing.append("field")
    if not _pde_ok(draft):
        missing.append("pde")
    missing.extend(_listed_boundary_gaps(draft.get("boundaries"), _defined_names(draft)))
    if _generic_dynamic(draft, family):
        initial = draft.get("initial_condition", draft.get("initial_conditions"))
        if not _generic_initial_ok(initial, _defined_names(draft)):
            missing.append("initial_condition")
    return missing


__all__ = [
    "FamilyAssessment",
    "assess_family",
    "questions_for",
    "reconcile_llm_report",
]
