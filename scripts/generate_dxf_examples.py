"""Regenerate the CAD-viewer-compatible DXF examples.

This optional maintenance script needs ``ezdxf``. The generated files are
checked into the project, so users do not need this package to import them.
"""

from __future__ import annotations

from pathlib import Path

import ezdxf


OUTPUT = Path(__file__).resolve().parents[1] / "examples" / "geometries"


def new_document() -> ezdxf.document.Drawing:
    document = ezdxf.new("R2000", setup=True)
    document.header["$INSUNITS"] = 6  # metres
    return document


def comsol_example() -> None:
    document = new_document()
    document.layers.new("COMSOL_OUTER", dxfattribs={"color": 5})
    document.layers.new("COMSOL_HOLE", dxfattribs={"color": 1})
    space = document.modelspace()
    space.add_lwpolyline(
        [(0, 0), (8, 0), (8, 5), (0, 5)],
        close=True,
        dxfattribs={"layer": "COMSOL_OUTER"},
    )
    space.add_lwpolyline(
        [
            (4.8, 2.5),
            (4.565685, 3.065685),
            (4, 3.3),
            (3.434315, 3.065685),
            (3.2, 2.5),
            (3.434315, 1.934315),
            (4, 1.7),
            (4.565685, 1.934315),
        ],
        close=True,
        dxfattribs={"layer": "COMSOL_HOLE"},
    )
    document.saveas(OUTPUT / "comsol_2d_export_example.dxf")


def cad_example() -> None:
    document = new_document()
    document.layers.new("CAD_L_SHAPE", dxfattribs={"color": 3})
    document.modelspace().add_polyline2d(
        [(0, 0), (5, 0), (5, 2), (2, 2), (2, 5), (0, 5)],
        close=True,
        dxfattribs={"layer": "CAD_L_SHAPE"},
    )
    document.saveas(OUTPUT / "cad_l_shape_example.dxf")


def rectangle_with_hole_example() -> None:
    document = new_document()
    document.layers.new("OUTER", dxfattribs={"color": 5})
    document.layers.new("HOLE", dxfattribs={"color": 1})
    space = document.modelspace()
    space.add_lwpolyline(
        [(0, 0), (4, 0), (4, 3), (0, 3)],
        close=True,
        dxfattribs={"layer": "OUTER"},
    )
    space.add_lwpolyline(
        [(1, 1), (2, 1), (2, 2), (1, 2)],
        close=True,
        dxfattribs={"layer": "HOLE"},
    )
    document.saveas(OUTPUT / "rectangle_with_hole.dxf")


if __name__ == "__main__":
    comsol_example()
    cad_example()
    rectangle_with_hole_example()
