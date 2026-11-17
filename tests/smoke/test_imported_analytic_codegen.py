"""Run actual DOM-generated scripts, rather than a hand-written approximation."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import torch

import openpinn as op


def test_dom_generated_analytic_and_fallback_problems(tmp_path):
    node = shutil.which("node")
    if not node or subprocess.run([node, "-e", "require('jsdom')"], capture_output=True).returncode:
        pytest.skip("Requires Node and jsdom; set NODE_PATH to the installed dependency directory.")
    root = Path(__file__).resolve().parents[2]
    command = [node, str(root / "tests/js/imported_analytic_geometry.test.cjs"), "--json"]
    if os.environ.get("OPENPINN_TEST_DXF"):
        command.append(os.environ["OPENPINN_TEST_DXF"])
    completed = subprocess.run(command, cwd=root, check=True, text=True, capture_output=True, timeout=60)
    programs = json.loads(completed.stdout)
    for program in programs:
        code = program["code"]
        compile(code, program["name"], "exec")
        definitions = code.split("\nSTEPS =", 1)[0]
        assert definitions != code, "Expected the generated training settings delimiter."
        namespace = {}
        exec(definitions, namespace)
        space, problem = namespace["SPACE"], namespace["PROBLEM"]
        if program["name"] == "fallback":
            assert isinstance(space, op.Polygon)
        else:
            assert isinstance(space, op.RectangleWithCircularHoles)
            assert space.stratified_boundary
            assert problem.metadata["geometry"] == "rectangle_with_circular_holes"
        if program["name"] == "edited":
            assert space.holes == ((11, 19, 5), (30, 20, 4))
        points = space.boundary(512, seed=7)
        if program["name"] == "dynamic":
            points = torch.cat((points, torch.zeros_like(points[:, :1])), dim=1)
        boundary = [condition for condition in problem.conditions if condition.sample_kind == "boundary"]
        selections = torch.stack([condition.selector(points) for condition in boundary])
        assert selections.sum(dim=0).eq(1).all(), program["name"]
        assert selections.sum(dim=1).gt(0).all()
        if program["name"] in {"static", "edited", "dynamic"}:
            normals = namespace["polygon_boundary_normal"](points)
            assert torch.isfinite(normals).all()
            torch.testing.assert_close(torch.linalg.vector_norm(normals, dim=1), torch.ones(points.shape[0]))
            if points.shape[1] == 3:
                assert normals[:, 2].eq(0).all()
            for index, (cx, cy, _) in enumerate(space.holes):
                mask = selections[index + 4]
                inward = points.new_tensor((cx, cy)) - points[mask, :2]
                assert ((normals[mask, :2] * inward).sum(dim=1) > 0).all()
        result = op.solve(
            problem, steps=2, seed=0,
            model={"hidden_layers": [8, 8]},
            sampling={"interior": 16, "boundary": 128, "initial": 16},
            output_root=tmp_path / program["name"], plot=False,
        )
        assert (result.output_dir / "metrics.json").is_file()
        assert all(torch.isfinite(torch.tensor(value)) for value in result.metrics.values() if isinstance(value, (float, int)))
