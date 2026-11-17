from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch

import openpinn as op


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "reference_runs"


def _poisson_config() -> dict[str, Any]:
    config = deepcopy(op.load_config(CONFIG_DIR / "poisson_smoke.yaml"))
    config["problem"]["name"] = "poisson"
    config["model"]["name"] = "mlp"
    config["model"]["feature"] = {"name": "identity"}
    config["model"]["output_transform"] = {"name": "identity"}
    config["sampling"]["name"] = "uniform"
    config["loss"]["formulation"] = "strong"
    config["optimizer"]["name"] = "adam"
    return config


def test_experiment_constructs_every_component_category_through_registry(
    tmp_path: Path,
) -> None:
    registry = op.default_component_registry()
    calls: list[str] = []
    selected = (
        ("problem", "poisson"),
        ("model", "mlp"),
        ("feature", "identity"),
        ("constraint", "identity"),
        ("formulation", "strong"),
        ("sampler", "uniform"),
        ("loss", "fixed"),
        ("optimizer", "adam"),
    )
    for kind, name in selected:
        original = registry.factory(kind, name)  # type: ignore[arg-type]

        def recording_factory(
            *,
            _factory: Callable[..., Any] = original,
            _kind: str = kind,
            **kwargs: Any,
        ) -> Any:
            calls.append(_kind)
            return _factory(**kwargs)

        registry.register(
            kind, name, recording_factory, replace=True  # type: ignore[arg-type]
        )

    result = op.Experiment.from_config(
        _poisson_config(), registry=registry
    ).run(output_dir=tmp_path / "registry_run")

    assert calls == [
        "problem",
        "model",
        "feature",
        "constraint",
        "formulation",
        "sampler",
        "loss",
        "optimizer",
    ]
    assert result.metrics["problem"] == "poisson_1d"
    assert torch.isfinite(torch.tensor(result.metrics["final_total_loss"]))


class QuadraticFeature(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.out_features = 2 * input_dim

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return torch.cat((coordinates, coordinates.square()), dim=1)


def test_custom_feature_can_be_registered_without_changing_experiment(
    tmp_path: Path,
) -> None:
    registry = op.default_component_registry()

    def quadratic_factory(
        *, config: dict[str, Any], input_dim: int
    ) -> QuadraticFeature:
        assert config == {"name": "quadratic"}
        return QuadraticFeature(input_dim)

    registry.register("feature", "quadratic", quadratic_factory)
    config = _poisson_config()
    config["model"]["feature"] = {"name": "quadratic"}

    result = op.Experiment.from_config(config, registry=registry).run(
        output_dir=tmp_path / "custom_feature"
    )

    assert result.model.feature_transform.__class__ is QuadraticFeature
    assert result.compute_ledger["parameter_count"] > 0


def test_output_dimension_mismatch_fails_before_model_factory(
    tmp_path: Path,
) -> None:
    registry = op.default_component_registry()
    original = registry.factory("model", "mlp")
    model_factory_calls = 0

    def counting_model_factory(**kwargs: Any) -> Any:
        nonlocal model_factory_calls
        model_factory_calls += 1
        return original(**kwargs)

    registry.register("model", "mlp", counting_model_factory, replace=True)
    config = _poisson_config()
    config["model"]["output_dim"] = 2

    with pytest.raises(ValueError, match="problem.output_dim"):
        op.Experiment.from_config(config, registry=registry).run(
            output_dir=tmp_path / "must_not_exist"
        )

    assert model_factory_calls == 0
    assert not (tmp_path / "must_not_exist").exists()

