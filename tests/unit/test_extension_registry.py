"""M4: first-class model/sampler/loss/optimizer/derivative registries."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from pydantic import ValidationError

import openpinn as op
from openpinn.spec import load_method_spec


def _valid_method() -> dict:
    return {
        "method": "baseline",
        "variant": "plain_pinn",
        "path": "pinn",
        "sampling": {"name": "uniform", "interior": 16, "boundary": 4},
        "loss": {"formulation": "strong", "weighting": "fixed"},
        "optimizer": {"name": "adam", "lr": 1e-3},
        "network": {"name": "mlp", "hidden_layers": [8, 8], "activation": "tanh"},
        "differentiator": {"mode": "autodiff"},
        "training": {"steps": 2},
    }


def test_public_extension_kinds_are_first_class() -> None:
    kinds = op.component_kinds()
    for kind in op.PUBLIC_EXTENSION_KINDS:
        assert kind in kinds
    assert op.PUBLIC_EXTENSION_KINDS == (
        "model",
        "sampler",
        "loss",
        "optimizer",
        "derivative",
    )


def test_default_registry_lists_builtins_for_five_kinds() -> None:
    registry = op.default_component_registry()
    expected = {
        "model": (
            "domaindecomposition",
            "gno",
            "lowrankoperator",
            "mlp",
            "modifiedmlp",
            "multipolegno",
            "physicsinformedkan",
        ),
        "sampler": (
            "adaptivemovement",
            "das",
            "grid",
            "importancesampling",
            "lhs",
            "qmc",
            "rar",
            "residualadaptive",
            "timewindow",
            "uncertainty",
            "uniform",
        ),
        "loss": (
            "causal",
            "fixed",
            "gradientbalanced",
            "inversedirichlet",
            "magnitudepower",
            "ntkbalanced",
            "pointwiseselfadaptive",
            "relobralo",
            "selfadaptive",
            "uncertainty",
        ),
        "optimizer": (
            "adam",
            "adamlbfgs",
            "adamw",
            "gaussnewton",
            "lbfgs",
            "naturalgradient",
        ),
        "derivative": (
            "autodiff",
            "finitedifference",
            "finitevolume",
            "grid",
            "hybridadnumerical",
            "weakvariational",
        ),
    }
    for kind, names in expected.items():
        assert registry.available(kind) == names


@pytest.mark.parametrize(
    ("kind", "name"),
    (
        ("model", "mlp"),
        ("sampler", "uniform"),
        ("loss", "fixed"),
        ("optimizer", "adam"),
        ("derivative", "autodiff"),
    ),
)
def test_create_builds_registered_builtins(kind: str, name: str) -> None:
    registry = op.default_component_registry()
    problem = op.make_poisson_problem()
    kwargs: dict = {}
    if kind == "model":
        kwargs = {
            "config": {"name": "mlp", "hidden_layers": [4, 4]},
            "problem": problem,
            "registry": registry,
        }
    elif kind == "sampler":
        kwargs = {
            "config": {"interior": 4, "boundary": 2},
            "problem": problem,
            "seed": 0,
            "device": torch.device("cpu"),
            "dtype": torch.float32,
        }
    elif kind == "loss":
        kwargs = {
            "config": {"name": "fixed"},
            "weights": {"pde": 1.0},
            "reduction": "sum",
        }
    elif kind == "optimizer":
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        kwargs = {"config": {"lr": 1e-3}, "parameters": [parameter]}
    component = registry.create(kind, name, **kwargs)
    assert component is not None
    if kind == "loss":
        total = component.compose({"pde": torch.tensor(2.0)})
        assert torch.isfinite(total)
    if kind == "derivative":
        coordinates = torch.tensor([[0.25], [0.5]], requires_grad=True)
        values = coordinates.square()
        gradient = component.grad(values, coordinates)
        torch.testing.assert_close(gradient, 2.0 * coordinates)


def test_register_custom_loss_and_create_it() -> None:
    registry = op.default_component_registry()

    class TwiceLoss:
        def compose(self, terms: Mapping[str, torch.Tensor]) -> torch.Tensor:
            return 2.0 * next(iter(terms.values()))

    registry.register("loss", "twice", lambda **kwargs: TwiceLoss())
    assert "twice" in registry.available("loss")
    composer = registry.create("loss", "TWICE")
    total = composer.compose({"pde": torch.tensor(3.0)})
    torch.testing.assert_close(total, torch.tensor(6.0))


def test_register_custom_derivative_and_create_it() -> None:
    registry = op.default_component_registry()

    class ScaledAutodiff(op.AutodiffDerivative):
        def grad(self, outputs, inputs, component=None, **kwargs):
            return 2.0 * super().grad(outputs, inputs, component, **kwargs)

    registry.register("derivative", "scaled_autodiff", lambda **kwargs: ScaledAutodiff())
    backend = registry.create("derivative", "scaled_autodiff")
    coordinates = torch.tensor([[1.0], [2.0]], requires_grad=True)
    values = coordinates.clone()
    torch.testing.assert_close(
        backend.grad(values, coordinates), 2.0 * torch.ones_like(coordinates)
    )


@pytest.mark.parametrize("kind", op.PUBLIC_EXTENSION_KINDS)
def test_unknown_name_raises_value_error(kind: str) -> None:
    registry = op.default_component_registry()
    with pytest.raises(ValueError, match="is not registered"):
        registry.create(kind, "definitely_not_registered")
    with pytest.raises(ValueError, match="is not registered"):
        registry.canonical_name(kind, "definitely_not_registered")
    assert not registry.contains(kind, "definitely_not_registered")


def test_unknown_kind_raises_value_error() -> None:
    registry = op.ComponentRegistry()
    with pytest.raises(ValueError, match="Unknown component category"):
        registry.available("not_a_kind")  # type: ignore[arg-type]


def test_unknown_loss_factory_result_is_type_error() -> None:
    registry = op.ComponentRegistry()
    registry.register("loss", "bad", lambda: object())
    with pytest.raises(TypeError, match="compose"):
        registry.create("loss", "bad")


def test_grid_derivative_create_exposes_partial() -> None:
    registry = op.default_component_registry()
    backend = registry.create("derivative", "grid", config={"spacing": 1.0})
    field = torch.randn(2, 1, 8)
    out = backend.partial(field, axis=0, order=1)
    assert out.shape == field.shape


def test_methodspec_unknown_component_names_fail() -> None:
    cases = (
        ("sampling", "name", "magic_sampler", "sampler"),
        ("network", "name", "transformer", "model"),
        ("optimizer", "name", "rmsprop", "optimizer"),
        ("loss", "weighting", "magic_loss", "loss"),
        ("differentiator", "mode", "symbolic", "derivative"),
        ("loss", "formulation", "collocation_magic", "formulation"),
    )
    for section, field, value, label in cases:
        data = _valid_method()
        data[section][field] = value
        with pytest.raises(ValidationError, match=rf"(?i)unknown.*{label}"):
            load_method_spec(data)


def test_methodspec_accepts_registered_grid_derivative() -> None:
    data = _valid_method()
    data["method"] = "deeponet"
    data["path"] = "operator"
    data["variant"] = None
    data["differentiator"]["mode"] = "grid"
    spec = load_method_spec(data)
    assert spec.differentiator.mode == "grid"


def test_methodspec_rejects_unimplemented_sgd_instead_of_fallback() -> None:
    data = _valid_method()
    data["optimizer"]["name"] = "sgd"
    with pytest.raises(ValidationError, match="(?i)unknown.*optimizer"):
        load_method_spec(data)
