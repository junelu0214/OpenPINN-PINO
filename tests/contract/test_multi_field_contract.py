from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

import openpinn as op


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "reference_runs"


def test_coupled_factory_is_public_and_registered() -> None:
    problem = op.make_coupled_manufactured_problem()
    registry = op.default_component_registry()

    assert problem.field_names == ("u", "v")
    assert problem.output_dim == 2
    assert registry.canonical_name(
        "problem", "coupled_elliptic_manufactured_1d"
    ) == "coupledellipticmanufactured"


def test_coupled_config_output_dimension_fails_before_artifact_creation(
    tmp_path: Path,
) -> None:
    config = deepcopy(op.load_config(CONFIG_DIR / "coupled_manufactured_smoke.yaml"))
    config["model"]["output_dim"] = 1
    output_dir = tmp_path / "dimension_mismatch"

    with pytest.raises(ValueError, match="problem.output_dim"):
        op.Experiment.from_config(config).run(output_dir=output_dir)

    assert not output_dir.exists()


@pytest.mark.parametrize(
    "residual",
    (
        lambda coordinates, prediction: {"bad": prediction},
        lambda coordinates, prediction: {"bad": prediction[:1, :1]},
    ),
)
def test_residual_shape_contract_rejects_multi_column_or_wrong_batch(
    residual: object,
) -> None:
    problem = op.PDEProblem(
        op.Interval(-1.0, 1.0),
        residual,  # type: ignore[arg-type]
        output_dim=2,
        field_names=("u", "v"),
    )
    coordinates = torch.zeros((3, 1), requires_grad=True)
    prediction = torch.zeros((3, 2), requires_grad=True)

    with pytest.raises(ValueError, match=r"shape \(N, 1\)|batch size"):
        problem.evaluate_residual(coordinates, prediction)


def test_unknown_coupled_loss_weight_fails_instead_of_being_ignored(
    tmp_path: Path,
) -> None:
    config = deepcopy(op.load_config(CONFIG_DIR / "coupled_manufactured_smoke.yaml"))
    config["loss"]["weights"]["equaton_u"] = 1.0

    with pytest.raises(ValueError, match="do not match"):
        op.Experiment.from_config(config).run(output_dir=tmp_path / "bad_weight")
