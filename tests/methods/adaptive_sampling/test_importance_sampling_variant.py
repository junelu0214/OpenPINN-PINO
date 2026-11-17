from __future__ import annotations

from pathlib import Path

import torch

import openpinn as op


def _small_sampler(seed: int) -> tuple[op.PDEProblem, torch.nn.Module, op.ImportanceSamplingSampler]:
    problem = op.make_poisson_problem()
    model = op.PINNModel(
        op.IdentityFeature(1), op.MLP(1, 1, hidden_layers=(8,)), op.IdentityOutput()
    )
    sampler = op.ImportanceSamplingSampler(
        problem,
        n_interior=8,
        n_boundary=2,
        n_initial=0,
        candidate_pool=16,
        seed=seed,
    )
    return problem, model, sampler


def test_importance_sampling_is_reproducible_and_returns_correction_weights() -> None:
    problem_a, model_a, sampler_a = _small_sampler(19)
    problem_b, model_b, sampler_b = _small_sampler(19)
    model_b.load_state_dict(model_a.state_dict())
    sampler_a.update(model_a)
    sampler_b.update(model_b)
    batch_a = sampler_a.sample(problem_a)
    batch_b = sampler_b.sample(problem_b)
    assert torch.equal(batch_a.interior, batch_b.interior)
    assert torch.equal(batch_a.interior_weights, batch_b.interior_weights)
    assert batch_a.interior_weights is not None
    assert batch_a.interior_weights.shape == (8, 1)
    assert torch.isfinite(batch_a.interior_weights).all()
    assert torch.all(batch_a.interior_weights > 0)
    losses = op.StrongFormulation(problem_a).loss_terms(model_a, batch_a)
    assert all(torch.isfinite(value) for value in losses.values())


def test_importance_sampling_trains_through_two_level_selector(tmp_path: Path) -> None:
    result = op.solve(
        "poisson",
        method="adaptive_sampling",
        variant="importance_sampling",
        steps=2,
        model={"hidden_layers": [8, 8]},
        sampling={"interior": 8, "boundary": 2},
        method_options={"candidate_pool": 16, "resample_every": 1},
        overrides={"evaluation": {"points": 9}},
        output_dir=tmp_path / "importance",
    )
    assert result.metrics["variant"] == "importance_sampling"
    diagnostics = result.compute_ledger["sampler_diagnostics"]
    assert diagnostics["update_count"] == 1
    assert diagnostics["mean_inverse_probability_weight"] > 0.0
