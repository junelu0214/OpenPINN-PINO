"""Safe compilation of declared strong-form expressions to PyTorch callables.

This module deliberately implements a small expression language.  It binds
names from a Stage A ``GenericProblemDescription`` and delegates coordinate
derivatives to :mod:`openpinn.autodiff`; it never executes source text.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
import re
from typing import Any

import torch
from torch import Tensor

from .autodiff import grad


class ExpressionCompileError(ValueError):
    """Raised when an expression or problem snapshot fails closed."""


FieldSource = Tensor | Callable[[Tensor], Tensor]
Residual = Callable[[Tensor, FieldSource], Tensor]


@dataclass(frozen=True)
class CompiledProblemExpressions:
    """Named residual callables compiled from one mapping-complete snapshot."""

    equations: Mapping[str, Residual]
    boundaries: Mapping[str, Residual]
    initial_conditions: Mapping[str, Residual]
    coordinate_names: tuple[str, ...]
    field_names: tuple[str, ...]
    parameters: Mapping[str, float]


@dataclass(frozen=True)
class _Bindings:
    coordinates: Mapping[str, int]
    fields: Mapping[str, int]
    field_names: tuple[str, ...]
    parameters: Mapping[str, float]
    output_dim: int


_FUNCTIONS: Mapping[str, Callable[[Tensor], Tensor]] = {
    "sin": torch.sin,
    "cos": torch.cos,
    "exp": torch.exp,
}
_LOCATION = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*$"
)


def _as_list(snapshot: Mapping[str, Any], key: str) -> list[Any]:
    value = snapshot.get(key)
    if not isinstance(value, list):
        raise ExpressionCompileError(f"problem.{key} must be a list")
    return value


def _bindings(snapshot: Mapping[str, Any]) -> _Bindings:
    contract = snapshot.get("trainable_contract")
    if not isinstance(contract, Mapping) or contract.get("mapping_complete") is not True:
        raise ExpressionCompileError(
            "problem.trainable_contract.mapping_complete must be true before compilation"
        )

    coordinates: dict[str, int] = {}
    for index, coordinate in enumerate(_as_list(snapshot, "coordinates")):
        if not isinstance(coordinate, Mapping) or not isinstance(coordinate.get("name"), str):
            raise ExpressionCompileError("every coordinate must have a string name")
        name = coordinate["name"]
        if not name or name in coordinates:
            raise ExpressionCompileError("coordinate names must be non-empty and unique")
        coordinates[name] = index

    fields: dict[str, int] = {}
    field_names: list[str] = []
    output_index = 0
    for field in _as_list(snapshot, "fields"):
        if not isinstance(field, Mapping):
            raise ExpressionCompileError("every field must be an object")
        components = field.get("components")
        if not isinstance(components, list) or not components:
            raise ExpressionCompileError("every field must declare at least one component")
        component_indexes: list[int] = []
        for component in components:
            if not isinstance(component, str) or not component or component in fields:
                raise ExpressionCompileError("field component names must be non-empty and unique")
            fields[component] = output_index
            field_names.append(component)
            component_indexes.append(output_index)
            output_index += 1
        name = field.get("name")
        if isinstance(name, str) and name:
            if name in fields and fields[name] != component_indexes[0]:
                raise ExpressionCompileError(f"field name {name!r} is ambiguous")
            if len(component_indexes) == 1:
                fields[name] = component_indexes[0]

    if not fields or output_index == 0:
        raise ExpressionCompileError("at least one scalar field component is required")

    parameters: dict[str, float] = {}
    for parameter in _as_list(snapshot, "parameters"):
        if not isinstance(parameter, Mapping) or not isinstance(
            parameter.get("name"), str
        ):
            raise ExpressionCompileError("every parameter must have a string name")
        name = parameter["name"]
        value = parameter.get("value")
        if (
            not name
            or name in parameters
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ExpressionCompileError(
                "parameter names must be unique and values must be numeric"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ExpressionCompileError("parameter values must be finite")
        parameters[name] = numeric

    collisions = (
        (set(coordinates) & set(fields))
        | (set(coordinates) & set(parameters))
        | (set(fields) & set(parameters))
    )
    if collisions:
        raise ExpressionCompileError(f"declared names collide: {', '.join(sorted(collisions))}")
    reserved = (set(coordinates) | set(fields) | set(parameters)) & (
        set(_FUNCTIONS) | {"pi"}
    )
    if reserved:
        raise ExpressionCompileError(
            f"declared names use reserved expression names: {', '.join(sorted(reserved))}"
        )
    return _Bindings(coordinates, fields, tuple(field_names), parameters, output_index)


def _split_equation(source: str) -> str:
    if not isinstance(source, str) or not source.strip():
        raise ExpressionCompileError("expression must be a non-empty string")
    if source.count("=") > 1:
        raise ExpressionCompileError("expressions may contain at most one equality")
    if "=" in source:
        left, right = source.split("=", 1)
        if not left.strip() or not right.strip():
            raise ExpressionCompileError("both sides of an equality are required")
        return f"({left})-({right})"
    return source


def _parse(source: str) -> ast.expr:
    try:
        tree = ast.parse(_split_equation(source), mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise ExpressionCompileError("expression is outside the restricted grammar") from exc
    return tree.body


class _SafeExpression:
    def __init__(self, source: str, bindings: _Bindings) -> None:
        self.source = source
        self.bindings = bindings
        self.node = _parse(source)
        self._validate(self.node)

    def _validate(self, node: ast.AST) -> None:
        if isinstance(node, ast.Constant):
            if (
                isinstance(node.value, bool)
                or not isinstance(node.value, (int, float))
                or not math.isfinite(float(node.value))
            ):
                raise ExpressionCompileError("only numeric literals are allowed")
            return
        if isinstance(node, ast.Name):
            if (
                node.id == "pi"
                or node.id in self.bindings.coordinates
                or node.id in self.bindings.fields
                or node.id in self.bindings.parameters
            ):
                return
            raise ExpressionCompileError(f"unbound symbol: {node.id}")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            self._validate(node.operand)
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            self._validate(node.left)
            self._validate(node.right)
            return
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and not node.keywords
            and len(node.args) == 1
        ):
            name = node.func.id
            if name.startswith("_"):
                raise ExpressionCompileError(f"unsafe function name: {name}")
            if name in _FUNCTIONS:
                self._validate(node.args[0])
                return
            if name.startswith("d") and name[1:] in self.bindings.coordinates:
                self._validate(node.args[0])
                return
            if name.startswith("d"):
                raise ExpressionCompileError(f"derivative uses undeclared coordinate: {name}")
            raise ExpressionCompileError(f"unregistered function: {name}")
        raise ExpressionCompileError(f"unsafe or unsupported syntax: {type(node).__name__}")

    def __call__(self, coordinates: Tensor, fields: Tensor) -> Tensor:
        return self._evaluate(self.node, coordinates, fields)

    def _evaluate(self, node: ast.AST, coordinates: Tensor, fields: Tensor) -> Tensor:
        if isinstance(node, ast.Constant):
            return coordinates[:, :1] * 0.0 + float(node.value)
        if isinstance(node, ast.Name):
            if node.id == "pi":
                return coordinates[:, :1] * 0.0 + math.pi
            if node.id in self.bindings.coordinates:
                index = self.bindings.coordinates[node.id]
                return coordinates[:, index : index + 1]
            if node.id in self.bindings.fields:
                index = self.bindings.fields[node.id]
                return fields[:, index : index + 1]
            return coordinates[:, :1] * 0.0 + self.bindings.parameters[node.id]
        if isinstance(node, ast.UnaryOp):
            value = self._evaluate(node.operand, coordinates, fields)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = self._evaluate(node.left, coordinates, fields)
            right = self._evaluate(node.right, coordinates, fields)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        value = self._evaluate(node.args[0], coordinates, fields)
        name = node.func.id
        if name in _FUNCTIONS:
            return _FUNCTIONS[name](value)
        derivative = grad(value, coordinates)
        index = self.bindings.coordinates[name[1:]]
        return derivative[:, index : index + 1]


def _normalize_fields(
    coordinates: Tensor,
    source: FieldSource,
    output_dim: int,
    coordinate_dim: int,
) -> Tensor:
    if coordinates.ndim != 2:
        raise ValueError("coordinates must have shape (N, coordinate_dim)")
    if coordinates.shape[1] != coordinate_dim:
        raise ValueError(
            f"coordinates must have width {coordinate_dim}, got {coordinates.shape[1]}"
        )
    if not coordinates.requires_grad:
        raise ValueError("coordinates.requires_grad must be True for compiled expressions")
    fields = source(coordinates) if callable(source) else source
    if not isinstance(fields, Tensor):
        raise TypeError("field source must return a torch.Tensor")
    expected = (coordinates.shape[0], output_dim)
    if tuple(fields.shape) != expected:
        raise ValueError(f"fields must have shape {expected}, got {tuple(fields.shape)}")
    return fields


def _residual(
    expression: _SafeExpression, output_dim: int, coordinate_dim: int
) -> Residual:
    def evaluate(coordinates: Tensor, source: FieldSource) -> Tensor:
        fields = _normalize_fields(coordinates, source, output_dim, coordinate_dim)
        result = expression(coordinates, fields)
        if result.ndim != 2 or result.shape != (coordinates.shape[0], 1):
            raise ValueError("compiled residual must have shape (N, 1)")
        return result

    return evaluate


def _target_source(value: Any) -> str:
    if isinstance(value, bool):
        raise ExpressionCompileError("condition values must be numeric or restricted expressions")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return repr(float(value))
    if isinstance(value, str) and value.strip():
        return value
    raise ExpressionCompileError("condition values must be numeric or restricted expressions")


def compile_problem_expressions(snapshot: Mapping[str, Any]) -> CompiledProblemExpressions:
    """Compile PDE, boundary, and initial residuals from a Stage A snapshot.

    Compilation requires ``trainable_contract.mapping_complete`` to be exactly
    true.  Returned callables accept ``(coordinates, fields_or_model)`` and
    return tensors shaped ``(N, 1)``.  This API does not create a trainer or
    mark a generated backend runnable.
    """

    if not isinstance(snapshot, Mapping):
        raise ExpressionCompileError("problem snapshot must be a mapping")
    bindings = _bindings(snapshot)

    equations: dict[str, Residual] = {}
    for index, equation in enumerate(_as_list(snapshot, "equations")):
        if not isinstance(equation, Mapping):
            raise ExpressionCompileError("every equation must be an object")
        name = str(equation.get("id") or f"equation_{index}")
        if name in equations:
            raise ExpressionCompileError(f"duplicate equation id: {name}")
        source = equation.get("expression", equation.get("residual"))
        equations[name] = _residual(
            _SafeExpression(source, bindings),
            bindings.output_dim,
            len(bindings.coordinates),
        )

    boundaries: dict[str, Residual] = {}
    first_field = min(bindings.fields, key=bindings.fields.get)
    for index, boundary in enumerate(_as_list(snapshot, "boundaries")):
        if not isinstance(boundary, Mapping):
            raise ExpressionCompileError("every boundary must be an object")
        name = str(boundary.get("id") or f"boundary_{index}")
        if name in boundaries:
            raise ExpressionCompileError(f"duplicate boundary id: {name}")
        field = boundary.get("field", first_field)
        if not isinstance(field, str) or field not in bindings.fields:
            raise ExpressionCompileError(f"boundary {name!r} references an undeclared field")
        target_source = boundary.get(
            "value_expression", boundary.get("expression", boundary.get("value"))
        )
        target = _SafeExpression(_target_source(target_source), bindings)
        kind = str(boundary.get("type", "")).lower()
        left = field
        if kind in {"neumann", "robin"}:
            location = boundary.get("location")
            match = _LOCATION.fullmatch(location) if isinstance(location, str) else None
            if match is None or match.group(1) not in bindings.coordinates:
                raise ExpressionCompileError(
                    f"{kind.title()} boundary {name!r} needs a declared-coordinate location"
                )
            derivative = f"d{match.group(1)}({field})"
            if kind == "neumann":
                left = derivative
            else:
                alpha = _target_source(boundary.get("alpha"))
                beta = _target_source(boundary.get("beta"))
                left = f"({alpha})*({field})+({beta})*({derivative})"
        elif kind != "dirichlet":
            raise ExpressionCompileError(f"unsupported boundary type: {kind or '<missing>'}")
        expression = _SafeExpression(left, bindings)

        def boundary_residual(
            coordinates: Tensor,
            source: FieldSource,
            lhs: _SafeExpression = expression,
            rhs: _SafeExpression = target,
        ) -> Tensor:
            fields = _normalize_fields(
                coordinates,
                source,
                bindings.output_dim,
                len(bindings.coordinates),
            )
            return lhs(coordinates, fields) - rhs(coordinates, fields)

        boundaries[name] = boundary_residual

    initials: dict[str, Residual] = {}
    for index, initial in enumerate(_as_list(snapshot, "initial_conditions")):
        if not isinstance(initial, Mapping):
            raise ExpressionCompileError("every initial condition must be an object")
        name = str(initial.get("id") or f"initial_{index}")
        if name in initials:
            raise ExpressionCompileError(f"duplicate initial-condition id: {name}")
        field = initial.get("field", first_field)
        if not isinstance(field, str) or field not in bindings.fields:
            raise ExpressionCompileError(f"initial condition {name!r} references an undeclared field")
        target_source = initial.get("expression", initial.get("value"))
        residual_expression = _SafeExpression(
            f"{field}-({_target_source(target_source)})", bindings
        )
        initials[name] = _residual(
            residual_expression, bindings.output_dim, len(bindings.coordinates)
        )

    if not equations:
        raise ExpressionCompileError("at least one equation is required")
    return CompiledProblemExpressions(
        equations=equations,
        boundaries=boundaries,
        initial_conditions=initials,
        coordinate_names=tuple(bindings.coordinates),
        field_names=bindings.field_names,
        parameters=dict(bindings.parameters),
    )
