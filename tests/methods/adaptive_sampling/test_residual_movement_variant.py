from __future__ import annotations

from pathlib import Path

import torch

import openpinn as op


def test_ram_preserves_fixed_points_and_total_count() -> None:
    problem = op.make_poisson_problem()
    model = op.PINNModel(
        op.IdentityFeature(1), op.MLP(1, 1, hidden_layers=(8,)), op.IdentityOutput()
    )
    sampler = op.AdaptiveMovementSampler(
        problem,
        n_interior=10,
        n_boundary=2,
        n_initial=0,
        indicator="residual",
        candidate_pool=20,
        movable_fraction=0.4,
        seed=7,
    )
    before = sampler.sample(problem).interior.detach().clone()
    sampler.update(model)
    after = sampler.sample(problem).interior.detach()
    assert after.shape == before.shape == (10, 1)
    assert torch.equal(after[:6], before[:6])
    assert sampler.diagnostics()["movable_count"] == 4


def test_ram_trains_through_two_level_selector(tmp_path: Path) -> None:
    result = op.solve(
        "poisson",
        method="adaptive_sampling",
        variant="residual_movement",
        steps=2,
        model={"hidden_layers": [8, 8]},
        sampling={"interior": 8, "boundary": 2},
        method_options={
            "candidate_pool": 16,
            "movable_fraction": 0.5,
            "resample_every": 1,
        },
        overrides={"evaluation": {"points": 9}},
        output_dir=tmp_path / "ram",
    )
    assert result.metrics["variant"] == "residual_movement"
    assert result.compute_ledger["sampler_diagnostics"]["update_count"] == 1
    assert result.compute_ledger["sampler_diagnostics"]["fixed_count"] == 4
    assert (result.output_dir / "result.svg").is_file()
