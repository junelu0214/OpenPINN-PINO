"""PDE residuals must not be written inside network.forward."""

from __future__ import annotations

import inspect

import pytest

import openpinn as op
from openpinn.models import MLP, ModifiedMLP, PINNModel
from openpinn.operators.fno import FNO
from openpinn.operators.pino import PINO


_BAD_EVALUATE = """
class BadNet(torch.nn.Module):
    def forward(self, x):
        u = self.net(x)
        return self.problem.evaluate_residual(x, u)
"""

_BAD_RESIDUAL_ASSIGN = """
class BadNet(torch.nn.Module):
    def forward(self, x):
        u = self.net(x)
        residual = -hessian(u, x, i=0, j=0)
        return residual
"""

_OK_FORWARD = """
class GoodNet(torch.nn.Module):
    def forward(self, x):
        return self.net(x)

    def pde_residual(self, x, u):
        return -hessian(u, x, i=0, j=0)
"""


@pytest.mark.parametrize(
    "cls",
    (MLP, PINNModel, ModifiedMLP, FNO, PINO),
)
def test_builtin_forwards_do_not_build_pde_residuals(cls: type) -> None:
    op.assert_no_pde_residual_in_forward(inspect.getsource(cls))


def test_evaluate_residual_in_forward_is_rejected() -> None:
    messages = op.find_pde_residuals_in_forward(_BAD_EVALUATE)
    assert messages
    with pytest.raises(ValueError, match="PDE residual"):
        op.assert_no_pde_residual_in_forward(_BAD_EVALUATE)


def test_residual_assignment_with_hessian_in_forward_is_rejected() -> None:
    with pytest.raises(ValueError, match="PDE residual"):
        op.assert_no_pde_residual_in_forward(_BAD_RESIDUAL_ASSIGN)


def test_residual_helper_outside_forward_is_allowed() -> None:
    op.assert_no_pde_residual_in_forward(_OK_FORWARD)


def test_docs_state_the_forward_residual_rule() -> None:
    from pathlib import Path

    text = (Path(__file__).resolve().parents[2] / "docs" / "extension.md").read_text(
        encoding="utf-8"
    )
    assert "Do not write PDE residuals inside `network.forward`" in text
    assert "assert_no_pde_residual_in_forward" in text
