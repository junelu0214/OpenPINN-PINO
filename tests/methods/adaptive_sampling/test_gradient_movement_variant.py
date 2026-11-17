from __future__ import annotations

from pathlib import Path

import torch

import openpinn as op


class Quadratic(torch.nn.Module):
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return coordinates[:, :1].square()


def test_wam_uses_finite_solution_gradient_indicator() -> None:
    problem = op.make_poisson_problem()
    sampler = op.AdaptiveMovementSampler(
        problem,
        n_interior=10,
        n_boundary=2,
        n_initial=0,
        indicator="solution_gradient",
        candidate_pool=20,
        movable_fraction=0.5,
        seed=11,
    )
    before = sampler.sample(problem).interior.detach().clone()
    sampler.update(Quadratic())
    after = sampler.sample(problem).interior.detach()
    diagnostics = sampler.diagnostics()
    assert after.shape == (10, 1)
    assert torch.equal(after[:5], before[:5])
    assert diagnostics["indicator"] == "solution_gradient"
    assert float(diagnostics["candidate_max_indicator"]) >= 1.0


def test_wam_trains_through_two_level_selector(tmp_path: Path) -> None:
    result = op.solve(
        "poisson",
        method="adaptive_sampling",
        variant="gradient_movement",
        steps=2,
        model={"hidden_layers": [8, 8]},
        sampling={"interior": 8, "boundary": 2},
        method_options={
            "candidate_pool": 16,
            "movable_fraction": 0.5,
            "resample_every": 1,
        },
        overrides={"evaluation": {"points": 9}},
        output_dir=tmp_path / "wam",
    )
    assert result.metrics["variant"] == "gradient_movement"
    diagnostics = result.compute_ledger["sampler_diagnostics"]
    assert diagnostics["indicator"] == "solution_gradient"
    assert diagnostics["update_count"] == 1
