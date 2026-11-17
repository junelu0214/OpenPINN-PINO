from __future__ import annotations

import pytest
import torch

import openpinn as op


@pytest.mark.parametrize("name", ("adam", "adamw", "lbfgs"))
def test_supported_optimizers_complete_a_finite_closure_step(name: str) -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    kwargs = {"max_iter": 2, "history_size": 5} if name == "lbfgs" else {}
    optimizer = op.make_optimizer(name, [parameter], lr=0.1, **kwargs)

    def closure() -> torch.Tensor:
        return (parameter - 1.0).square().sum()

    loss = optimizer.step(closure)

    assert torch.isfinite(loss)
    assert optimizer.closure_calls >= 1
    assert torch.isfinite(parameter).all()


def test_staged_optimizer_switches_from_adam_to_lbfgs() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    optimizer = op.StagedOptimizerAdapter(
        [parameter],
        (
            {"name": "adam", "steps": 1, "lr": 0.1},
            {"name": "lbfgs", "steps": 1, "lr": 0.5, "max_iter": 2},
        ),
    )

    def closure() -> torch.Tensor:
        return (parameter - 1.0).square().sum()

    optimizer.step(closure)
    assert optimizer.last_stage_index == 0
    assert optimizer.current_stage_index == 1
    optimizer.step(closure)
    diagnostics = optimizer.diagnostics()

    assert optimizer.last_stage_index == 1
    assert diagnostics["completed_outer_steps"] == 2
    assert diagnostics["stages"][0]["closure_calls"] == 1
    assert diagnostics["stages"][1]["closure_calls"] >= 1
    assert torch.isfinite(parameter).all()
