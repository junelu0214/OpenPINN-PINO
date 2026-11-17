"""Validated single-field physical expressions, evaluated without code execution.

Geometry and boundary values retain their input units. Inside a residual,
coordinates, the field and parameters are converted to SI. ``dx``/``dy``/``dt``
differentiate with respect to the SI coordinate (chain rule included). Returned
residuals are in SI; this is dimensional validation, not automatic loss scaling.
Multi-field poromechanics and flux boundary conditions are not implemented here.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import torch
from torch import Tensor

from .autodiff import _structured_derivatives_active, grad

_CALLS = {"dx", "dy", "dt", "sin", "cos", "exp", "sqrt", "log"}
_BINARY = {"+", "-", "*", "/", "**"}
_DIMENSIONLESS = (0.0, 0.0, 0.0, 0.0)
_LENGTH = (0.0, 1.0, 0.0, 0.0)
_TIME = (0.0, 0.0, 1.0, 0.0)
_SYMBOL = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,31}\Z")
_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,95}\Z")
_MAX_SPEC_BYTES = 65536
_MAX_CATALOG_BYTES = 1048576
_MAX_NODES = 512
_MAX_DEPTH = 32
_RESERVED = _CALLS | {"x", "y", "t", "pi"}


def _json_copy(value: Any, *, byte_limit: int) -> Any:
    """Accept JSON values only and bound work before recursively decoding ASTs."""
    budget = [0]

    def check(item: Any, depth: int) -> None:
        budget[0] += 1
        if depth > 80 or budget[0] > 100000:
            raise ValueError("Physics JSON exceeds nesting or element limits.")
        if item is None or type(item) is bool:
            return
        if type(item) in (int, float):
            _finite(item, "JSON number")
        elif type(item) is str:
            if len(item) > 8192:
                raise ValueError("Physics text exceeds 8192 characters.")
        elif type(item) is list:
            for child in item:
                check(child, depth + 1)
        elif type(item) is dict:
            for key, child in item.items():
                if type(key) is not str or len(key) > 128:
                    raise ValueError("Physics JSON keys must be short strings.")
                check(child, depth + 1)
        else:
            raise ValueError("Physics input must contain only plain JSON values.")

    check(value, 0)
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode("utf-8")) > byte_limit:
        raise ValueError(f"Physics JSON exceeds {byte_limit} bytes.")
    return json.loads(encoded)


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number.")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite number.") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    return result


def _keys(value: Any, required: set[str], optional: set[str], label: str) -> dict:
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object.")
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing or unknown:
        raise ValueError(f"{label}: missing {sorted(missing)}, unsupported {sorted(unknown)}.")
    return value


def _symbol(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SYMBOL.fullmatch(value) or "__" in value:
        raise ValueError(f"{label} must be a simple ASCII identifier (maximum 32 characters).")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{label} must be a stable ASCII identifier.")
    return value


@lru_cache(maxsize=1)
def _builtin_catalog() -> dict:
    return json.loads(Path(__file__).with_name("physics_catalog.json").read_text(encoding="utf-8"))


def _unit(name: Any) -> dict:
    if not isinstance(name, str) or name not in _builtin_catalog()["units"]:
        raise ValueError(f"Unknown physical unit: {name!r}; use the explicit unit table.")
    return _builtin_catalog()["units"][name]


def _dimension(name: str) -> tuple[float, ...]:
    return tuple(_unit(name)["dimensions"])


def _dimension_equal(left: tuple, right: tuple) -> bool:
    return all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(left, right))


def _inspect_expression(spec: dict, dimensions: dict) -> tuple[float, ...]:
    count = [0]
    used: set[str] = set()

    def visit(node: Any, depth: int, order: int = 0, time_order: int = 0):
        count[0] += 1
        if count[0] > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError("Physics expression exceeds 512 nodes or depth 32.")
        if type(node) is not dict:
            raise ValueError("Each expression node must be an object.")
        kind = node.get("type")
        if kind == "number":
            _keys(node, {"type", "value"}, set(), "number node")
            value = _finite(node["value"], "Expression literal")
            return _DIMENSIONLESS, value == 0, value
        if kind == "symbol":
            _keys(node, {"type", "name"}, set(), "symbol node")
            name = _symbol(node["name"], "Expression symbol")
            if name not in dimensions:
                raise ValueError(f"Unknown or unavailable symbol: {name}.")
            used.add(name)
            return dimensions[name], False, math.pi if name == "pi" else None
        if kind == "unary":
            _keys(node, {"type", "op", "arg"}, set(), "unary node")
            if node["op"] != "-":
                raise ValueError("Only unary minus is supported.")
            dim, zero, constant = visit(node["arg"], depth + 1, order, time_order)
            return dim, zero, -constant if constant is not None else None
        if kind == "binary":
            _keys(node, {"type", "op", "left", "right"}, set(), "binary node")
            op = node["op"]
            if not isinstance(op, str) or op not in _BINARY:
                raise ValueError("Unsupported binary operator.")
            left, left_zero, left_constant = visit(node["left"], depth + 1, order, time_order)
            right, right_zero, right_constant = visit(node["right"], depth + 1, order, time_order)
            if op in {"+", "-"}:
                if not (left_zero or right_zero or _dimension_equal(left, right)):
                    raise ValueError(f"Incompatible dimensions in {op}: {left} and {right}.")
                dim = right if left_zero else left
            elif op == "*":
                dim = tuple(a + b for a, b in zip(left, right))
            elif op == "/":
                if right_zero:
                    raise ValueError("Division by literal zero is not allowed.")
                dim = tuple(a - b for a, b in zip(left, right))
            else:
                if right_constant is None or not _dimension_equal(right, _DIMENSIONLESS):
                    raise ValueError("Power exponent must be a dimensionless numeric constant.")
                if abs(right_constant) > 16:
                    raise ValueError("Power exponent magnitude must not exceed 16.")
                if left_zero and right_constant <= 0:
                    raise ValueError("Zero base requires a positive exponent.")
                dim = tuple(a * right_constant for a in left)
            constant = None
            if left_constant is not None and right_constant is not None:
                try:
                    if op == "+": constant = left_constant + right_constant
                    elif op == "-": constant = left_constant - right_constant
                    elif op == "*": constant = left_constant * right_constant
                    elif op == "/": constant = left_constant / right_constant
                    else: constant = left_constant ** right_constant
                    constant = _finite(constant, "Constant expression")
                except (ArithmeticError, ValueError) as error:
                    raise ValueError("Invalid constant expression.") from error
            return dim, constant == 0 if constant is not None else False, constant
        if kind == "call":
            _keys(node, {"type", "name", "args"}, set(), "call node")
            name = node["name"]
            if not isinstance(name, str) or name not in _CALLS:
                raise ValueError("Only dx/dy/dt/sin/cos/exp/sqrt/log calls are allowed.")
            if type(node["args"]) is not list or len(node["args"]) != 1:
                raise ValueError("Physics functions require exactly one argument.")
            derivative = name in {"dx", "dy", "dt"}
            if name == "dy" and spec["dimension"] != 2:
                raise ValueError("dy is unavailable in one spatial dimension.")
            if name == "dt" and spec["kind"] != "dynamic":
                raise ValueError("dt is unavailable for a static problem.")
            next_order = order + int(derivative)
            next_time = time_order + int(name == "dt")
            if next_order > 2:
                raise ValueError("Derivative order must not exceed two.")
            if next_time > 1:
                raise ValueError("Second time derivatives require unsupported initial velocity data.")
            dim, zero, constant = visit(node["args"][0], depth + 1, next_order, next_time)
            if derivative:
                coordinate = _TIME if name == "dt" else _LENGTH
                return tuple(a - b for a, b in zip(dim, coordinate)), constant is not None, 0.0 if constant is not None else None
            if name == "sqrt":
                dim = tuple(a / 2 for a in dim)
            elif not _dimension_equal(dim, _DIMENSIONLESS):
                raise ValueError(f"{name} requires a dimensionless argument.")
            if constant is not None:
                try:
                    constant = _finite(getattr(math, name)(constant), "Constant function")
                except (ArithmeticError, ValueError) as error:
                    raise ValueError(f"Invalid domain for {name}.") from error
            return dim, constant == 0 if constant is not None else False, constant
        raise ValueError(f"Unknown expression node type: {kind!r}.")

    result, _, _ = visit(spec["expression"], 0)
    if spec["field"] not in used:
        raise ValueError("A physical residual must reference its scalar field.")
    return result


def validate_physics_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a pure-JSON scalar expression and return an isolated copy.

    Coordinates must use length/time units, with equal x/y units. Bare numbers
    are dimensionless; use a named parameter for a dimensional constant.
    ``expression_source`` is optional display text, never executed.
    """
    data = _json_copy(spec, byte_limit=_MAX_SPEC_BYTES)
    _keys(data, {"schema_version", "dimension", "kind", "field", "expression", "parameters", "coordinate_units", "field_unit"}, {"expression_source"}, "physics spec")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Physics spec requires schema_version: 1.")
    if type(data["dimension"]) is not int or data["dimension"] not in (1, 2):
        raise ValueError("Only one or two spatial dimensions are supported.")
    if data["kind"] not in ("static", "dynamic"):
        raise ValueError("Physics kind must be static or dynamic.")
    field = _symbol(data["field"], "Scalar field")
    if field in _RESERVED:
        raise ValueError("Field name collides with a coordinate or function.")
    if "expression_source" in data and not isinstance(data["expression_source"], str):
        raise ValueError("expression_source must be text.")
    coordinates = _keys(data["coordinate_units"], {"x"}, {"y", "t"}, "coordinate_units")
    if data["dimension"] == 2 and "y" not in coordinates:
        raise ValueError("Two-dimensional problems require a y unit.")
    if data["kind"] == "dynamic" and "t" not in coordinates:
        raise ValueError("Dynamic problems require a t unit.")
    for name, unit in coordinates.items():
        if not _dimension_equal(_dimension(unit), _TIME if name == "t" else _LENGTH):
            raise ValueError(f"Coordinate {name} has the wrong unit dimension.")
    if data["dimension"] == 2 and coordinates["x"] != coordinates["y"]:
        raise ValueError("x/y coordinate units must match to preserve boundary normals.")
    dimensions = {field: _dimension(data["field_unit"]), "x": _LENGTH, "pi": _DIMENSIONLESS}
    if data["dimension"] == 2: dimensions["y"] = _LENGTH
    if data["kind"] == "dynamic": dimensions["t"] = _TIME
    parameters = data["parameters"]
    if type(parameters) is not list or len(parameters) > 128:
        raise ValueError("parameters must be a list of at most 128 constants.")
    ids: set[str] = set()
    for parameter in parameters:
        _keys(parameter, {"id", "symbol", "value", "unit"}, {"label", "description", "category", "source", "minimum", "maximum"}, "parameter")
        identifier = _identifier(parameter["id"], "Parameter id")
        symbol = _symbol(parameter["symbol"], "Parameter symbol")
        if identifier in ids or symbol in dimensions or symbol in _RESERVED:
            raise ValueError("Duplicate parameter id/symbol or reserved symbol.")
        ids.add(identifier)
        value = _finite(parameter["value"], f"Parameter {symbol}")
        _finite(value * _unit(parameter["unit"])["factor"], f"SI parameter {symbol}")
        for key in ("minimum", "maximum"):
            if key in parameter:
                bound = _finite(parameter[key], key)
                if (key == "minimum" and value < bound) or (key == "maximum" and value > bound):
                    raise ValueError(f"Parameter {symbol} violates {key}.")
        dimensions[symbol] = _dimension(parameter["unit"])
    _inspect_expression(data, dimensions)
    return data


def make_physics_residual(spec: Mapping[str, Any]):
    """Return ``equation(raw_coordinates, raw_field)`` from a validated AST.

    The callable accepts one scalar output and requires differentiable input
    coordinates. Zero derivatives retain a zero-valued model graph so constant
    and linear manufactured fields can be backpropagated safely.
    """
    data = validate_physics_spec(spec)
    coordinate_names = ["x"] + (["y"] if data["dimension"] == 2 else []) + (["t"] if data["kind"] == "dynamic" else [])
    factors = {name: _unit(data["coordinate_units"][name])["factor"] for name in coordinate_names}
    field_factor = _unit(data["field_unit"])["factor"]
    parameters = {p["symbol"]: p["value"] * _unit(p["unit"])["factor"] for p in data["parameters"]}

    def equation(x: Tensor, u: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != len(coordinate_names):
            raise ValueError("Physics coordinates do not match the declared dimension/time layout.")
        if u.ndim != 2 or u.shape != (x.shape[0], 1):
            raise ValueError("Physics expressions support a single scalar output (N, 1).")
        if not x.requires_grad and not _structured_derivatives_active(u, x):
            raise ValueError("Physics coordinates must require gradients.")
        if x.device != u.device or x.dtype != u.dtype:
            raise ValueError("Physics coordinates and field must share device/dtype.")
        zero = x[:, :1] * 0.0 + u * 0.0
        values = {name: x[:, i:i + 1] * factors[name] for i, name in enumerate(coordinate_names)}
        values[data["field"]] = u * field_factor
        values["pi"] = x.new_tensor(math.pi)
        values.update({name: x.new_tensor(value) for name, value in parameters.items()})

        def evaluate(node: dict) -> Tensor:
            kind = node["type"]
            if kind == "number": return x.new_tensor(node["value"])
            if kind == "symbol": return values[node["name"]]
            if kind == "unary": return -evaluate(node["arg"])
            if kind == "binary":
                left, right = evaluate(node["left"]), evaluate(node["right"])
                op = node["op"]
                if op == "+": return left + right
                if op == "-": return left - right
                if op == "*": return left * right
                if op == "/": return left / right
                return left ** right
            name = node["name"]
            argument = evaluate(node["args"][0])
            if name in {"dx", "dy", "dt"}:
                axis = coordinate_names.index(name[1:])
                if not argument.requires_grad:
                    return zero
                derivative = grad(argument, x)
                return derivative[:, axis:axis + 1] / factors[name[1:]] + zero
            return {"sin": torch.sin, "cos": torch.cos, "exp": torch.exp, "sqrt": torch.sqrt, "log": torch.log}[name](argument)

        residual = evaluate(data["expression"]) + zero
        if not torch.isfinite(residual).all():
            raise ValueError("Physical residual is non-finite; check values and function domains.")
        return residual

    equation.physics_spec = copy.deepcopy(data)
    return equation


def validate_physics_catalog(catalog: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a full catalog or an extension pack; reject redefined units.

    Parameter definitions have no material defaults (``value`` must be null).
    Material values and runnable example templates are separate records.
    External packs may omit units; they always use the bundled unit whitelist.
    """
    data = _json_copy(catalog, byte_limit=_MAX_CATALOG_BYTES)
    _keys(data, {"schema_version", "categories", "parameters", "materials", "templates"}, {"units", "description"}, "physics catalog")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Physics catalog requires schema_version: 1.")
    canonical = _builtin_catalog()["units"]
    if "units" in data and data["units"] != canonical:
        raise ValueError("Extension catalogs must not redefine the core unit table.")
    data["units"] = copy.deepcopy(canonical)
    for key, maximum in (("categories", 64), ("parameters", 512), ("materials", 128), ("templates", 128)):
        if type(data[key]) is not list or len(data[key]) > maximum:
            raise ValueError(f"Catalog {key} must be a list with at most {maximum} records.")
    categories: set[str] = set()
    for category in data["categories"]:
        _keys(category, {"id", "label"}, {"description"}, "category")
        identifier = _identifier(category["id"], "Category id")
        if identifier in categories:
            raise ValueError("Duplicate category id.")
        categories.add(identifier)
    definitions: dict[str, dict] = {}
    for parameter in data["parameters"]:
        _keys(parameter, {"id", "category", "label", "symbol", "unit", "description", "value"}, {"source", "minimum", "maximum"}, "parameter definition")
        identifier = _identifier(parameter["id"], "Parameter id")
        if identifier in definitions or parameter["category"] not in categories:
            raise ValueError("Duplicate parameter id or unknown category.")
        if parameter["value"] is not None:
            raise ValueError("Parameter definitions must use value: null, not universal material values.")
        symbol = _symbol(parameter["symbol"], "Parameter symbol")
        if symbol in _RESERVED:
            raise ValueError("Parameter definition uses a reserved symbol.")
        _unit(parameter["unit"])
        for key in ("minimum", "maximum"):
            if key in parameter: _finite(parameter[key], key)
        definitions[identifier] = parameter
    seen: set[str] = set()
    for material in data["materials"]:
        _keys(material, {"id", "label", "parameters", "source"}, {"description", "conditions"}, "material")
        identifier = _identifier(material["id"], "Material id")
        if identifier in seen: raise ValueError("Duplicate material id.")
        seen.add(identifier)
        if type(material["parameters"]) is not list or len(material["parameters"]) > 128:
            raise ValueError("Material parameters must be a bounded list.")
        used: set[str] = set()
        for entry in material["parameters"]:
            _keys(entry, {"id", "value", "unit"}, {"source"}, "material parameter")
            if entry["id"] not in definitions or entry["id"] in used:
                raise ValueError("Material has an unknown or duplicate parameter id.")
            used.add(entry["id"])
            if not _dimension_equal(_dimension(entry["unit"]), _dimension(definitions[entry["id"]]["unit"])):
                raise ValueError("Material parameter unit has incompatible dimensions.")
            _finite(_finite(entry["value"], "Material value") * _unit(entry["unit"])["factor"], "SI material value")
    seen.clear()
    for template in data["templates"]:
        _keys(template, {"id", "label", "description", "is_demo", "expression_text", "spec"}, {"initial_value", "boundary_value"}, "template")
        identifier = _identifier(template["id"], "Template id")
        if identifier in seen: raise ValueError("Duplicate template id.")
        seen.add(identifier)
        if template["is_demo"] is not True:
            raise ValueError("Runnable example templates must be explicitly marked is_demo: true.")
        if not isinstance(template["expression_text"], str):
            raise ValueError("Template expression_text must be display text.")
        template["spec"] = validate_physics_spec(template["spec"])
        for key in ("initial_value", "boundary_value"):
            if key in template: _finite(template[key], key)
        for parameter in template["spec"]["parameters"]:
            if parameter["id"] not in definitions:
                raise ValueError("Template references an unknown parameter definition.")
            if not _dimension_equal(_dimension(parameter["unit"]), _dimension(definitions[parameter["id"]]["unit"])):
                raise ValueError("Template parameter unit disagrees with its definition.")
    return data


def load_physics_catalog(path: str | Path | None = None) -> dict[str, Any]:
    """Load the packaged catalog or a local JSON extension, and validate it."""
    if path is None:
        data = copy.deepcopy(_builtin_catalog())
    else:
        source = Path(path)
        if source.stat().st_size > _MAX_CATALOG_BYTES:
            raise ValueError("Physics catalog exceeds 1 MiB.")
        data = json.loads(source.read_text(encoding="utf-8"))
    return validate_physics_catalog(data)


__all__ = ["load_physics_catalog", "validate_physics_catalog", "validate_physics_spec", "make_physics_residual"]
