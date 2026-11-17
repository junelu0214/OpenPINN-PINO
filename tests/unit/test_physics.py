"""Mathematical, unit, security and short-run contracts of physical expressions."""
from copy import deepcopy
import json
import math

import pytest
import torch

import openpinn as op


def symbol(name):
    return {"type": "symbol", "name": name}


def number(value):
    return {"type": "number", "value": value}


def call(name, argument):
    return {"type": "call", "name": name, "args": [argument]}


def binary(operator, left, right):
    return {"type": "binary", "op": operator, "left": left, "right": right}


def spec(expression, *, dimension=1, kind="static", field_unit="Pa", parameters=None):
    return {
        "schema_version": 1, "dimension": dimension, "kind": kind,
        "field": "p", "field_unit": field_unit,
        "coordinate_units": {"x": "m", "y": "m", "t": "s"},
        "expression": expression, "parameters": parameters or [],
    }


def parameter(name, value, unit):
    return {"id": "test." + name, "symbol": name, "value": value, "unit": unit}


def test_catalog_is_valid_separated_and_copied():
    catalog = op.load_physics_catalog()
    parameter_ids = [item["id"] for item in catalog["parameters"]]
    assert len(parameter_ids) >= 30
    assert len(parameter_ids) == len(set(parameter_ids))
    assert {"consolidation.diffusivity", "consolidation.initial_pressure"} <= set(parameter_ids)
    assert {item["category"] for item in catalog["parameters"]} == {"common", "seepage", "consolidation", "thermal", "elasticity"}
    assert all(item["value"] is None for item in catalog["parameters"])
    assert catalog["materials"] == []
    assert all(item["is_demo"] for item in catalog["templates"])
    catalog["units"]["m"]["factor"] = 42
    assert op.load_physics_catalog()["units"]["m"]["factor"] == 1


def test_variable_coefficient_divergence_and_mixed_derivatives():
    # d/dx ((a + b*x) dp/dx), p=x^3, a=2 m²/s, b=3 m/s.
    expression = call("dx", binary("*", binary("+", symbol("a"), binary("*", symbol("b"), symbol("x"))), call("dx", symbol("p"))))
    equation = op.make_physics_residual(spec(expression, parameters=[parameter("a", 2, "m^2/s"), parameter("b", 3, "m/s")]))
    x = torch.tensor([[.1], [.7], [1.3]], dtype=torch.float64, requires_grad=True)
    p = x ** 3
    torch.testing.assert_close(equation(x, p), 12 * x + 27 * x ** 2)
    mixed = op.make_physics_residual(spec(call("dx", call("dy", symbol("p"))), dimension=2))
    xy = torch.tensor([[.2, .4], [.8, .9]], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(mixed(xy, xy[:, :1] ** 2 * xy[:, 1:2]), 2 * xy[:, :1])


def test_coordinate_field_and_parameter_units_chain_rule():
    expression = binary("-", call("dt", symbol("p")), binary("*", symbol("c"), call("dx", call("dx", symbol("p")))))
    data = spec(expression, kind="dynamic", parameters=[parameter("c", 1, "m^2/day")])
    data["coordinate_units"] = {"x": "cm", "t": "day"}
    data["field_unit"] = "kPa"
    equation = op.make_physics_residual(data)
    raw = torch.tensor([[10., 1.], [40., 2.]], dtype=torch.float64, requires_grad=True)
    pressure = raw[:, :1] ** 2 + 3 * raw[:, 1:2]
    expected = torch.full_like(pressure, 3000 / 86400 - 2000 / (.01 ** 2 * 86400))
    torch.testing.assert_close(equation(raw, pressure), expected)
    # Coordinate symbols themselves are SI, not merely derivative-scaled.
    coordinate = op.make_physics_residual(spec(binary("/", symbol("p"), symbol("x")), field_unit="m"))
    converted = spec(binary("/", symbol("p"), symbol("x")), field_unit="cm")
    converted["coordinate_units"]["x"] = "cm"
    torch.testing.assert_close(coordinate(raw[:, :1], raw[:, :1]), op.make_physics_residual(converted)(raw[:, :1], raw[:, :1]))


@pytest.mark.parametrize("expression", [call("dx", call("dx", symbol("p"))), call("dx", binary("*", number(0), symbol("p")))])
def test_constant_linear_zero_derivatives_keep_backward(expression):
    equation = op.make_physics_residual(spec(expression))
    for constant in (False, True):
        x = torch.linspace(0, 1, 4).reshape(-1, 1).requires_grad_()
        weight = torch.nn.Parameter(torch.tensor(3.))
        field = weight.expand_as(x) if constant else weight * x
        residual = equation(x, field)
        assert torch.equal(residual, torch.zeros_like(field))
        residual.square().mean().backward()
        assert weight.grad is not None and weight.grad.item() == 0


def test_functions_power_and_zero_dimension_rule():
    expression = binary("+", binary("**", symbol("p"), number(2)), number(0))
    equation = op.make_physics_residual(spec(expression))
    x = torch.tensor([[.2], [.7]], requires_grad=True)
    torch.testing.assert_close(equation(x, x), x ** 2)
    expression = call("sin", binary("*", symbol("pi"), symbol("p")))
    torch.testing.assert_close(op.make_physics_residual(spec(expression, field_unit="1"))(x, x), torch.sin(math.pi * x))


@pytest.mark.parametrize("expression,dimension,kind", [
    (call("dy", symbol("p")), 1, "static"),
    (call("dt", symbol("p")), 1, "static"),
    (call("dt", call("dt", symbol("p"))), 1, "dynamic"),
    (call("dx", call("dx", call("dx", symbol("p")))), 1, "static"),
    (binary("+", symbol("p"), symbol("x")), 1, "static"),
    (call("sin", symbol("p")), 1, "static"),
    (binary("**", symbol("p"), symbol("p")), 1, "static"),
    (binary("**", symbol("p"), number(17)), 1, "static"),
    (binary("/", symbol("p"), number(0)), 1, "static"),
    (symbol("unknown"), 1, "static"),
    (call("__import__", symbol("p")), 1, "static"),
    ({"type": "attribute", "name": "p.__class__"}, 1, "static"),
    ({"type": "number", "value": float("inf")}, 1, "static"),
])
def test_invalid_ast_or_dimensions_rejected(expression, dimension, kind):
    with pytest.raises(ValueError):
        op.make_physics_residual(spec(expression, dimension=dimension, kind=kind))


def test_mixed_space_time_derivative_is_allowed():
    equation = op.make_physics_residual(spec(call("dx", call("dt", symbol("p"))), kind="dynamic"))
    xt = torch.tensor([[2., 3.], [4., 5.]], requires_grad=True)
    torch.testing.assert_close(equation(xt, xt[:, :1] ** 2 * xt[:, 1:2]), 2 * xt[:, :1])


def test_resource_limits_parameter_collisions_and_unit_validation():
    for parameter_list in ([parameter("a", None, "Pa")], [parameter("x", 1, "m")], [parameter("a", 1, "Pa"), parameter("a", 2, "Pa")], [parameter("a", 1, "madeup")]):
        with pytest.raises(ValueError):
            op.validate_physics_spec(spec(symbol("p"), parameters=parameter_list))
    expression = symbol("p")
    for _ in range(34): expression = {"type": "unary", "op": "-", "arg": expression}
    with pytest.raises(ValueError, match="depth"):
        op.validate_physics_spec(spec(expression))
    data = spec(symbol("p"), dimension=2)
    data["coordinate_units"]["y"] = "mm"
    with pytest.raises(ValueError, match="units must match"):
        op.validate_physics_spec(data)
    data = spec(symbol("p"))
    data["expression_source"] = "p" * 8193
    with pytest.raises(ValueError): op.validate_physics_spec(data)
    data = spec(symbol("p"))
    data["extra"] = "exec"
    with pytest.raises(ValueError): op.validate_physics_spec(data)


def test_spec_copy_and_nonfinite_runtime_guard():
    data = spec(binary("/", symbol("p"), symbol("x")))
    equation = op.make_physics_residual(data)
    data["expression"] = number(42)
    x = torch.ones((2, 1), requires_grad=True)
    torch.testing.assert_close(equation(x, 2 * x), 2 * x)
    with pytest.raises(ValueError, match="non-finite"):
        equation(torch.zeros((2, 1), requires_grad=True), torch.ones((2, 1)))
    with pytest.raises(ValueError, match="single scalar"):
        equation(x, torch.ones((2, 2)))


def test_extension_catalog_rejects_unit_redefinition_defaults_and_bad_references(tmp_path):
    catalog = op.load_physics_catalog()
    for mutate in (
        lambda c: c["units"]["m"].update(factor=2),
        lambda c: c["parameters"][0].update(value=1000),
        lambda c: c["parameters"].append(deepcopy(c["parameters"][0])),
        lambda c: c["templates"][0]["spec"]["parameters"][0].update(id="missing"),
    ):
        invalid = deepcopy(catalog)
        mutate(invalid)
        with pytest.raises(ValueError): op.validate_physics_catalog(invalid)
    extension = {"schema_version": 1, "categories": [{"id": "local", "label": "本地"}], "parameters": [{"id": "local.length", "category": "local", "label": "长度", "symbol": "L", "unit": "m", "description": "实测长度", "value": None}], "materials": [{"id": "local.sample", "label": "样本", "source": "实验记录", "parameters": [{"id": "local.length", "value": 5, "unit": "cm"}]}], "templates": []}
    target = tmp_path / "catalog.json"
    target.write_text(json.dumps(extension), encoding="utf-8")
    assert op.load_physics_catalog(target)["materials"][0]["parameters"][0]["value"] == 5
    extension["materials"][0]["parameters"][0]["unit"] = "s"
    with pytest.raises(ValueError, match="incompatible"):
        op.validate_physics_catalog(extension)


@pytest.mark.parametrize("template_id", ["pressure_diffusion", "steady_darcy", "heat_diffusion"])
def test_catalog_templates_execute_real_short_training(tmp_path, template_id):
    template = next(t for t in op.load_physics_catalog()["templates"] if t["id"] == template_id)
    data = template["spec"]
    space = op.RectangleWithCircularHoles((0., 0.), (1., 1.))
    domain = op.SpaceTimeDomain(space, op.Interval(0., 1.)) if data["kind"] == "dynamic" else space
    conditions = [op.DirichletBC(0., name="boundary")]
    if data["kind"] == "dynamic": conditions.append(op.InitialCondition(0., name="initial"))
    problem = op.PDEProblem(domain, op.make_physics_residual(data), conditions, name=template_id, metadata={"kind": "custom_" + data["kind"]})
    result = op.solve(problem, steps=1, seed=0, device="cpu", model={"hidden_layers": [4, 4]}, sampling={"interior": 8, "boundary": 8, "initial": 8}, overrides={"evaluation": {"points": {"x": 3, "y": 3, "time": 1}, "boundary_points": 8, "initial_points": 8}}, output_dir=tmp_path / template_id, plot=False)
    assert (result.output_dir / "metrics.json").is_file()
    assert all(math.isfinite(value) for value in result.metrics.values() if isinstance(value, (int, float)))
