"""Run a short PINN training from the same DXF example parsed by the browser module."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

import openpinn as op


@pytest.mark.parametrize(
    ("filename", "outer_count", "hole_count"),
    [
        ("rectangle_with_hole.dxf", 4, 1),
        ("comsol_2d_export_example.dxf", 4, 1),
        ("cad_l_shape_example.dxf", 6, 0),
    ],
)
def test_imported_dxf_polygon_short_run(
    tmp_path: Path, filename: str, outer_count: int, hole_count: int
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the browser DXF-module test.")
    root = Path(__file__).resolve().parents[2]
    parser = root / "docs/_static/dxf_geometry_import.js"
    sample = root / "examples/geometries" / filename
    script = (
        "const fs=require('node:fs');"
        "const parser=require(process.argv[1]);"
        "process.stdout.write(JSON.stringify(parser.parse(fs.readFileSync(process.argv[2], 'utf8'))));"
    )
    completed = subprocess.run(
        [node, "-e", script, str(parser), str(sample)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    outer, holes = json.loads(completed.stdout)
    assert len(outer) == outer_count
    assert len(holes) == hole_count
    space = op.Polygon(outer, holes=holes)

    def equation(points: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return -op.hessian(values, points, i=0, j=0) - op.hessian(
            values, points, i=1, j=1
        )

    problem = op.PDEProblem(
        space,
        equation,
        [op.DirichletBC(0.0, name="boundary")],
        name="dxf_import_smoke",
        metadata={"kind": "custom_static", "geometry": "polygon"},
    )
    result = op.solve(
        problem,
        methods=[op.method("baseline", "plain_pinn")],
        steps=1,
        seed=0,
        model={"hidden_layers": [8, 8]},
        sampling={"interior": 16, "boundary": 16},
        output_root=tmp_path,
        plot=False,
    )
    assert "boundary_rmse" in result.metrics


def test_stitched_line_arc_dxf_short_run(tmp_path: Path) -> None:
    """Exercise the new multi-entity DXF path all the way through one optimizer step."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the browser DXF-module test.")
    parser = Path(__file__).resolve().parents[2] / "docs/_static/dxf_geometry_import.js"
    dxf = "\n".join(map(str, [
        0, "SECTION", 2, "ENTITIES",
        0, "LINE", 10, -1, 20, 0, 11, 1, 21, 0,
        0, "ARC", 10, 0, 20, 0, 40, 1, 50, 0, 51, 180,
        0, "ENDSEC", 0, "EOF", "",
    ]))
    script = (
        "let input='';process.stdin.on('data',chunk=>input+=chunk);"
        "process.stdin.on('end',()=>{const p=require(process.argv[1]);"
        "process.stdout.write(JSON.stringify(p.parse(input)));});"
    )
    completed = subprocess.run(
        [node, "-e", script, str(parser)], input=dxf,
        check=True, capture_output=True, text=True,
    )
    outer, holes = json.loads(completed.stdout)
    assert len(outer) > 20 and not holes
    domain = op.Polygon(outer)

    def equation(points: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return -op.hessian(values, points, i=0, j=0) - op.hessian(values, points, i=1, j=1)

    problem = op.PDEProblem(
        domain, equation, [op.DirichletBC(0.0, name="boundary")],
        name="stitched_dxf_smoke", metadata={"kind": "custom_static", "geometry": "polygon"},
    )
    result = op.solve(
        problem, methods=[op.method("baseline", "plain_pinn")], steps=1, seed=0,
        model={"hidden_layers": [8, 8]}, sampling={"interior": 16, "boundary": 16},
        output_root=tmp_path, plot=False,
    )
    assert "boundary_rmse" in result.metrics
