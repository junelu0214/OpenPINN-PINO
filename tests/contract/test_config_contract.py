from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import pytest

import openpinn as op


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "reference_runs"


@pytest.mark.parametrize(
    ("problem", "kind"),
    (("poisson", "static"), ("burgers", "dynamic")),
)
def test_baseline_configs_define_a_plain_reproducible_reference(
    problem: str, kind: str
) -> None:
    config = op.load_config(CONFIG_DIR / f"{problem}_baseline.yaml")

    assert config["schema_version"] == 1
    assert config["problem"]["kind"] == kind
    assert config["experiment"]["device"] == "cpu"
    assert isinstance(config["experiment"]["seed"], int)
    assert config["model"]["hidden_layers"] == [64, 64, 64, 64]
    assert config["model"]["activation"] == "tanh"
    assert config["model"]["feature"]["name"] == "identity"
    assert config["model"]["output_transform"]["name"] == "soft"
    assert config["sampling"]["name"] == "uniform"
    assert config["optimizer"]["name"] == "adam"
    assert all(weight == 1.0 for weight in config["loss"]["weights"].values())


@pytest.mark.parametrize("problem", ("poisson", "burgers"))
def test_smoke_configs_are_tiny_cpu_checks_not_accuracy_claims(problem: str) -> None:
    config = op.load_config(CONFIG_DIR / f"{problem}_smoke.yaml")

    assert config["experiment"]["device"] == "cpu"
    assert config["training"]["steps"] <= 5
    assert config["sampling"]["interior"] <= 32
    assert "not an accuracy benchmark" in config["experiment"]["description"].lower()


def test_config_loader_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        op.load_config(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        ("sampling", "name", "adaptive", "not implemented"),
        ("loss", "formulation", "weak", "not implemented"),
    ),
)
def test_unimplemented_methods_fail_instead_of_silently_running_baseline(
    section: str, field: str, value: str, message: str
) -> None:
    config = op.load_config(CONFIG_DIR / "poisson_smoke.yaml")
    config[section][field] = value

    with pytest.raises(ValueError, match=message):
        op.Experiment.from_config(config)


def test_misspelled_problem_name_does_not_fall_back_to_poisson() -> None:
    config = op.load_config(CONFIG_DIR / "poisson_smoke.yaml")
    config["problem"]["name"] = "poisson_typo"

    with pytest.raises(ValueError, match="not implemented"):
        op.Experiment.from_config(config)


def test_conflicting_output_transform_aliases_are_rejected() -> None:
    config = op.load_config(CONFIG_DIR / "poisson_smoke.yaml")
    config["model"]["constraint"] = {"name": "hard"}

    with pytest.raises(ValueError, match="only one"):
        op.Experiment.from_config(config)


def test_misspelled_fourier_option_is_not_ignored() -> None:
    config = op.load_config(CONFIG_DIR / "poisson_smoke.yaml")
    config["model"]["feature"] = {
        "name": "fourier",
        "num_frequencies": 4,
        "scael": 2.0,
    }

    with pytest.raises(ValueError, match="unsupported option"):
        op.Experiment.from_config(config).run(output_dir=Path("unused"))


def test_non_finite_configuration_is_rejected_before_a_success_status() -> None:
    config = op.load_config(CONFIG_DIR / "poisson_smoke.yaml")
    config.setdefault("problem", {}).setdefault("parameters", {})["amplitude"] = float("nan")

    with pytest.raises(FloatingPointError, match="non-finite"):
        op.Experiment.from_config(config)


def test_reusing_a_nonempty_output_directory_is_rejected(
    poisson_smoke_result: object,
) -> None:
    config = op.load_config(CONFIG_DIR / "poisson_smoke.yaml")
    output_dir = Path(getattr(poisson_smoke_result, "output_dir"))

    with pytest.raises(FileExistsError, match="not empty"):
        op.Experiment.from_config(deepcopy(config)).run(output_dir=output_dir)
