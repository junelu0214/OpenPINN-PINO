"""Static checks for second-development rules.

These helpers inspect Python source with :mod:`ast`.  They are not imported
by the training loop, codegen, or ``openpinn.agents``.  Call them from tests
or a project check script.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

_FORBIDDEN_FORWARD_CALLS = frozenset(
    {
        "evaluate_residual",
        "make_physics_residual",
    }
)
_FORBIDDEN_FORWARD_ATTRS = frozenset(
    {
        "evaluate_residual",
        "residual",
        "pde",
    }
)
_RESIDUAL_TARGET_NAMES = frozenset(
    {
        "residual",
        "pde",
        "pde_residual",
        "residuals",
    }
)
_RESIDUAL_HELPERS = frozenset({"grad", "hessian", "partial"})


def _call_name(node: ast.AST) -> str | None:
    func = getattr(node, "func", None)
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _target_names(target: ast.AST) -> Iterable[str]:
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, ast.Tuple):
        for item in target.elts:
            yield from _target_names(item)


def _uses_residual_helper(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and _call_name(child) in _RESIDUAL_HELPERS:
            return True
    return False


def find_pde_residuals_in_forward(source: str) -> tuple[str, ...]:
    """Return messages for PDE-residual construction inside ``forward``.

    The check is conservative: it flags residual APIs and assignments that
    look like a PDE residual.  It does not execute code.
    """

    tree = ast.parse(source)
    violations: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_function(node)

        def _visit_function(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            if node.name == "forward":
                self._check_forward(node)
            self.generic_visit(node)

        def _check_forward(
            self, node: ast.FunctionDef | ast.AsyncFunctionDef
        ) -> None:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    name = _call_name(child)
                    if name in _FORBIDDEN_FORWARD_CALLS:
                        violations.append(
                            f"line {child.lineno}: call {name}() inside forward()"
                        )
                    elif (
                        isinstance(child.func, ast.Attribute)
                        and child.func.attr in _FORBIDDEN_FORWARD_ATTRS
                    ):
                        violations.append(
                            f"line {child.lineno}: call .{child.func.attr}() "
                            "inside forward()"
                        )
                if isinstance(child, ast.Assign):
                    targets = {
                        name
                        for target in child.targets
                        for name in _target_names(target)
                    }
                    if targets & _RESIDUAL_TARGET_NAMES and _uses_residual_helper(
                        child.value
                    ):
                        violations.append(
                            f"line {child.lineno}: PDE residual assigned inside "
                            "forward()"
                        )
                if isinstance(child, ast.AnnAssign) and child.value is not None:
                    targets = set(_target_names(child.target))
                    if targets & _RESIDUAL_TARGET_NAMES and _uses_residual_helper(
                        child.value
                    ):
                        violations.append(
                            f"line {child.lineno}: PDE residual assigned inside "
                            "forward()"
                        )

    Visitor().visit(tree)
    return tuple(violations)


def assert_no_pde_residual_in_forward(source: str | Path) -> None:
    """Raise ``ValueError`` if ``forward`` builds a PDE residual.

    Network ``forward`` methods must return the physical field (or an output
    transform of it).  PDE residuals belong on :class:`~openpinn.PDEProblem`,
    never inside ``nn.Module.forward``.
    """

    text = Path(source).read_text(encoding="utf-8") if isinstance(source, Path) else source
    violations = find_pde_residuals_in_forward(text)
    if violations:
        details = "; ".join(violations)
        raise ValueError(
            "PDE residuals must not be written inside network.forward "
            f"({details}). Put the residual on PDEProblem instead."
        )
