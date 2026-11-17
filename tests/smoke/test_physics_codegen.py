"""Execute unchanged DOM-generated definitions; verify units and actual training."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest
import torch
import openpinn as op


def test_dom_generated_physics_code(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node/jsdom are needed for the real DOM code-generation regression.")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([node, "tests/js/physics_codegen.test.cjs", "--json"], cwd=root, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    programs = json.loads(result.stdout)
    residuals = {}
    for item in programs:
        compile(item["code"], item["name"], "exec")
        namespace = {}
        exec(item["code"].split("\nSTEPS =", 1)[0], namespace)
        spec, equation, problem = namespace["PHYSICS_SPEC"], namespace["equation"], namespace["PROBLEM"]
        assert problem.metadata["physics"] == op.validate_physics_spec(spec)
        assert problem.field_names == (spec["field"],)
        width = spec["dimension"] + int(spec["kind"] == "dynamic")
        x = torch.full((4, width), 0.3, dtype=torch.float64, requires_grad=True)
        u = x[:, :1]**2
        if spec["dimension"] == 2: u = u + 3*x[:, 1:2]**2
        if spec["kind"] == "dynamic": u = u + 2*x[:, -1:]
        residuals[item["name"]] = equation(x, u).detach()
        assert torch.isfinite(residuals[item["name"]]).all()
        if item["name"] in {"pressure", "pressure_user_equation", "darcy", "heat", "material"}:
            training = op.solve(problem, steps=2, seed=0, device="cpu", model={"hidden_layers":[4,4]},
                                sampling={"interior":8,"boundary":16,"initial":8},
                                overrides={"evaluation":{"points":{"x":3,"y":3,"time":1},"boundary_points":16,"initial_points":8}},
                                output_dir=tmp_path/item["name"], plot=False)
            assert (training.output_dir/"metrics.json").exists()
            assert all(torch.isfinite(torch.tensor(v)) for v in training.metrics.values() if isinstance(v,(float,int)))
        if item["name"] == "pressure":
            torch.testing.assert_close(namespace["example_field"](x),torch.full_like(u,2.))
        if item["name"] == "pressure_changed":
            cx, cy = [p["value"] for p in spec["parameters"]]
            torch.testing.assert_close(equation(x,u),torch.full_like(u,2 - 2*cx - 6*cy))
        if item["name"] == "converted_units":
            cx, cy = [p["value"] for p in spec["parameters"]]
            expected = 2000/86400 - (2*cx+6*cy)*1000/(.01**2)
            torch.testing.assert_close(equation(x,u),torch.full_like(u,expected))
    assert not torch.equal(residuals["pressure"], residuals["pressure_changed"])
    torch.testing.assert_close(residuals["pressure"], residuals["pressure_user_equation"])
    torch.testing.assert_close(residuals["heat_source_changed"]-residuals["heat_source"],torch.full((4,1),-1.,dtype=torch.float64))
    torch.testing.assert_close(residuals["material"]-residuals["heat_source_changed"],torch.full((4,1),-4.,dtype=torch.float64))
