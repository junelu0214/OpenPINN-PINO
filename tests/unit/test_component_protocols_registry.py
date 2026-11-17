from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch

import openpinn as op


class DelegatingFormulation:
    """Structural formulation used to prove inheritance is not required."""

    def __init__(self, problem: op.PDEProblem) -> None:
        self.problem = problem
        self._strong = op.StrongFormulation(problem)

    def loss_terms(
        self, model: torch.nn.Module, batch: op.SampleBatch
    ) -> Mapping[str, torch.Tensor]:
        return self._strong.loss_terms(model, batch)


class DelegatingSampler:
    """Structural sampler with checkpoint state, but no UniformSampler base."""

    def __init__(self, problem: op.PDEProblem, *, seed: int) -> None:
        self._uniform = op.UniformSampler(
            problem, interior=8, boundary=2, initial=0, seed=seed
        )

    def sample(
        self,
        problem: op.PDEProblem | op.Interval | op.SpaceTimeDomain | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        requires_grad: bool = True,
    ) -> op.SampleBatch:
        return self._uniform.sample(
            problem, device=device, dtype=dtype, requires_grad=requires_grad
        )

    def state_dict(self) -> Mapping[str, Any]:
        return self._uniform.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._uniform.load_state_dict(state)


def test_trainer_accepts_structural_formulation_and_sampler_protocols() -> None:
    torch.manual_seed(101)
    problem = op.make_poisson_problem()
    formulation = DelegatingFormulation(problem)
    sampler = DelegatingSampler(problem, seed=103)
    model = op.PINNModel(op.MLP(1, 1, hidden_layers=(8,)))

    assert isinstance(formulation, op.FormulationProtocol)
    assert isinstance(sampler, op.SamplerProtocol)

    trainer = op.Trainer(model, formulation, sampler, optimizer="adam", lr=1e-3)
    history = trainer.fit(steps=1)

    assert len(history) == 1
    assert torch.isfinite(torch.tensor(history.last["total"]))
    assert trainer.formulation is formulation
    assert trainer.sampler is sampler


def test_component_registry_is_category_separated_and_type_checked() -> None:
    registry = op.ComponentRegistry()
    registry.register(
        "problem",
        "toy-poisson",
        lambda: op.make_poisson_problem(),
        aliases=("toy_poisson_1d",),
    )
    registry.register("feature", "bad-feature", lambda: object())

    problem = registry.create("problem", "TOY poisson 1d")

    assert isinstance(problem, op.PDEProblem)
    assert registry.canonical_name("problem", "toy_poisson_1d") == "toypoisson"
    assert registry.available("problem") == ("toypoisson",)
    with pytest.raises(TypeError, match="torch.nn.Module"):
        registry.create("feature", "bad-feature")
    with pytest.raises(ValueError, match="already registered"):
        registry.register("problem", "toy-poisson", lambda: problem)
    with pytest.raises(ValueError, match="category"):
        registry.available("everything")  # type: ignore[arg-type]


def test_default_registry_copies_are_isolated() -> None:
    first = op.default_component_registry()
    second = op.default_component_registry()
    first.register("feature", "project-feature", lambda: op.IdentityFeature(1))

    assert first.contains("feature", "project-feature")
    assert not second.contains("feature", "project-feature")
    assert second.contains("optimizer", "adam")
    assert second.contains("loss", "fixed")
    assert second.contains("derivative", "autodiff")
    assert not second.contains("loss", "project-feature")

