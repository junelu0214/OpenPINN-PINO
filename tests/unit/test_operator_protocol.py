"""Smoke tests for CoordinateOperator and model-native derivative hooks."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

import openpinn as op


class StubCoordinateOperator:
    """Local stand-in. Not a registered training model."""

    def predict(self, coordinates, condition=None):
        reduced = coordinates.sum(dim=-1, keepdim=True)
        if condition is None:
            return reduced
        return reduced + condition.sum(dim=-1, keepdim=True)


class StubGridOperator:
    def predict(self, inputs, *, grid):
        del grid
        return inputs


class StubGridDerivative:
    def partial(self, field, axis: int, order: int = 1):
        del axis, order
        return torch.zeros_like(field)


def _stub_spec() -> op.OperatorHookSpec:
    return op.OperatorHookSpec(
        name="stub_coordinate",
        kind="coordinate",
        derivative_mode="autodiff",
        title="Stub coordinate model",
        summary="Import-and-register smoke test. Not a trainable operator.",
        implemented=True,
        selectable=False,
        milestone="",
    )


def _stub_grid_spec() -> op.OperatorHookSpec:
    return op.OperatorHookSpec(
        name="stub_grid",
        kind="grid",
        derivative_mode="grid",
        title="Stub grid operator",
        summary="Reference-signature grid registration smoke.",
        implemented=True,
        selectable=False,
    )


def test_stub_coordinate_operator_imports_and_registers() -> None:
    model = StubCoordinateOperator()
    assert isinstance(model, op.CoordinateOperator)
    # Deprecated spelling remains the exact protocol alias for one minor.
    assert isinstance(model, op.CoordinateModel)
    assert op.CoordinateModel is op.CoordinateOperator

    coordinates = torch.zeros(4, 2)
    values = model.predict(coordinates)
    assert tuple(values.shape) == (4, 1)
    conditioned = model.predict(coordinates, condition=torch.ones(4, 3))
    assert tuple(conditioned.shape) == (4, 1)

    registry = op.OperatorRegistry()
    spec = _stub_spec()
    registry.register(spec)
    resolved = registry.resolve("stub-coordinate")
    assert resolved.name == "stub_coordinate"
    assert resolved.kind == "coordinate"
    assert resolved.derivative_mode == "autodiff"
    assert resolved.implemented is True
    assert resolved.selectable is False
    assert registry.contains("stub_coordinate")


def test_formal_predict_signatures_match_reference_contract() -> None:
    coordinate = inspect.signature(op.CoordinateOperator.predict).parameters
    assert list(coordinate) == ["self", "coordinates", "condition"]
    assert coordinate["condition"].default is None

    grid = inspect.signature(op.GridOperator.predict).parameters
    assert list(grid) == ["self", "inputs", "grid"]
    assert grid["grid"].kind is inspect.Parameter.KEYWORD_ONLY

    concrete = inspect.signature(op.FNO.predict).parameters
    assert concrete["grid"].kind is inspect.Parameter.KEYWORD_ONLY


def test_stub_grid_operator_satisfies_grid_protocols() -> None:
    operator = StubGridOperator()
    derivative = StubGridDerivative()
    assert isinstance(operator, op.GridOperator)
    assert isinstance(derivative, op.GridDerivative)

    field = torch.ones(2, 1, 4, 4)
    grid = op.UniformGrid.from_shape((4, 4), spacing=0.25, periodic=True)
    assert grid.shape == (4, 4)
    assert grid.n_dims == 2
    assert grid.n_points == 16
    assert grid.spacing == (0.25, 0.25)
    assert grid.periodic == (True, True)
    assert tuple(operator.predict(field, grid=grid).shape) == (2, 1, 4, 4)
    assert tuple(derivative.partial(field, axis=0, order=1).shape) == (2, 1, 4, 4)

    registry = op.OperatorRegistry()
    registry.register(_stub_grid_spec())
    assert registry.resolve("stub-grid").kind == "grid"


def test_default_hooks_are_reserved_and_not_trainable_methods() -> None:
    hooks = {spec.name: spec for spec in op.list_operator_hooks()}
    assert list(hooks) == ["deeponet", "fno", "pino"]

    assert hooks["deeponet"].kind == "coordinate"
    assert hooks["deeponet"].derivative_mode == "autodiff"
    assert hooks["deeponet"].milestone == "M1c"
    assert hooks["fno"].kind == "grid"
    assert hooks["fno"].derivative_mode == "grid"
    assert hooks["fno"].milestone == "M1d"
    assert hooks["pino"].kind == "grid"
    assert hooks["pino"].derivative_mode == "grid"
    assert hooks["pino"].milestone == "M1e"
    assert hooks["deeponet"].implemented is True
    assert hooks["deeponet"].selectable is True
    assert hooks["deeponet"].parameters["latent_dim"]["default"] == 16
    assert hooks["deeponet"].parameters["n_modes"]["ui_hidden"] is True
    assert hooks["deeponet"].parameters["n_modes"]["default"] == 3
    assert hooks["fno"].implemented is True
    assert hooks["fno"].selectable is True
    assert hooks["fno"].parameters["n_modes"]["default"] == 4
    assert "ui_hidden" not in hooks["fno"].parameters["n_modes"]
    assert "ui_hidden" not in hooks["fno"].parameters["n_harmonics"]
    assert "ui_hidden" not in hooks["pino"].parameters["n_modes"]
    assert hooks["pino"].implemented is True
    assert hooks["pino"].selectable is False
    assert hooks["pino"].parameters["mode"]["default"] == "hybrid"
    assert hooks["pino"].parameters["data_pretrain_steps"]["default"] == 0

    assert op.default_operator_registry().resolve("deep_operator_network").name == "deeponet"
    assert op.default_operator_registry().resolve("fourier_neural_operator").name == "fno"
    assert op.default_operator_registry().resolve("physics_informed_neural_operator").name == "pino"
    assert [spec.name for spec in op.list_selectable_operators()] == ["deeponet", "fno"]
    assert "deeponet" not in {spec.name for spec in op.list_methods()}
    assert "fno" not in {spec.name for spec in op.list_methods()}
    assert "pino" not in {spec.name for spec in op.list_methods()}
    with pytest.raises(ValueError, match="build_operator_config"):
        op.build_config("poisson", method="deeponet", steps=1)
    with pytest.raises(ValueError, match="build_operator_config"):
        op.build_config("poisson", method="fno", steps=1)
    with pytest.raises(ValueError, match="Unknown OpenPINN method"):
        op.build_config("poisson", method="pino", steps=1)


def test_default_registry_is_isolated_from_a_later_stub() -> None:
    registry = op.default_operator_registry()
    registry.register(_stub_spec())
    assert registry.contains("stub_coordinate")
    assert op.default_operator_registry().contains("stub_coordinate") is False


def test_hook_registration_rejects_bad_records() -> None:
    with pytest.raises(ValueError, match="derivative mode"):
        op.OperatorHookSpec(
            name="bad",
            kind="coordinate",
            derivative_mode="grid",
            title="Bad",
            summary="Mismatched derivative mode.",
        )
    with pytest.raises(ValueError, match="selectable"):
        op.OperatorHookSpec(
            name="early",
            kind="grid",
            derivative_mode="grid",
            title="Early",
            summary="Selectable before any implementation.",
            implemented=False,
            selectable=True,
        )

    registry = op.OperatorRegistry()
    registry.register(_stub_spec())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_stub_spec())
    with pytest.raises(ValueError, match="available hooks"):
        registry.resolve("missing")


def test_configurator_payload_discovers_hooks_without_replacing_families() -> None:
    payload = op.attach_operator_hooks({"families": [{"method": "baseline"}]})
    assert payload["families"] == [{"method": "baseline"}]
    names = [item["name"] for item in payload["operator_hooks"]]
    assert names == ["deeponet", "fno", "pino"]
    by_name = {item["name"]: item for item in payload["operator_hooks"]}
    assert by_name["deeponet"]["selectable"] is True
    assert by_name["deeponet"]["implemented"] is True
    assert "n_sensors" in by_name["deeponet"]["parameters"]
    assert by_name["fno"]["selectable"] is True
    assert by_name["fno"]["implemented"] is True
    assert "n_modes" in by_name["fno"]["parameters"]
    assert by_name["pino"]["selectable"] is False
    assert by_name["pino"]["implemented"] is True
    assert "mode" in by_name["pino"]["parameters"]
    assert "data_weight" in by_name["pino"]["parameters"]
    assert "physics_weight" in by_name["pino"]["parameters"]
    assert by_name["deeponet"]["parameters"]["training_mode"]["choices"] == [
        "data_only",
        "physics_only",
        "hybrid",
    ]
    assert by_name["fno"]["parameters"]["training_mode"]["default"] == "data_only"

    catalog = Path(__file__).resolve().parents[2] / "docs" / "_ext" / "openpinn_method_catalog.py"
    source = catalog.read_text(encoding="utf-8")
    assert "attach_operator_hooks" in source
    assert "operator_hooks" in source
    assert "build_operator_config" in source
    assert 'data-role="operator-hooks"' in source
