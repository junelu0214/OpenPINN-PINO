from __future__ import annotations

import base64
import re
import struct

import numpy as np

from openpinn.artifacts import _grid_field_panel, _plot_panel, _svg_document


def test_reference_curve_is_drawn_after_prediction_but_listed_first() -> None:
    x = np.asarray([0.0, 0.5, 1.0])
    reference = np.asarray([0.0, 1.0, 0.0])
    prediction = reference + 1e-6
    svg = _plot_panel(
        title="Solution",
        x=x,
        curves=(
            ("Reference", reference, "#111827", "7 5"),
            ("Prediction", prediction, "#2563eb", None),
        ),
        panel=(70.0, 120.0, 520.0, 300.0),
    )

    prediction_curve = svg.index('stroke="#2563eb" stroke-width="2" stroke-linejoin')
    reference_curve = svg.index('stroke="#111827" stroke-width="2" stroke-linejoin')
    assert prediction_curve < reference_curve
    assert svg.index("Reference</text>") < svg.index("Prediction</text>")
    assert 'stroke-dasharray="7 5"' in svg


def test_plot_panel_prints_numeric_axis_limits() -> None:
    svg = _plot_panel(
        title="Absolute error",
        x=np.asarray([0.0, 1.0]),
        curves=(("Error", np.asarray([0.0, 4.34e-4]), "#dc2626", None),),
        panel=(70.0, 120.0, 520.0, 300.0),
        x_label="x",
        y_label="Absolute error",
    )

    assert "e-04" in svg
    assert ">x</text>" in svg
    assert ">Absolute error</text>" in svg


def test_grid_field_border_does_not_cover_coloured_cells() -> None:
    panel = _grid_field_panel(
        title="Pressure",
        x=np.asarray([0.0, 1.0, 0.0, 1.0]),
        y=np.asarray([0.0, 0.0, 1.0, 1.0]),
        values=np.asarray([0.0, 0.5, 0.75, 1.0]),
        panel=(60.0, 125.0, 300.0, 300.0),
        bounds=((0.0, 1.0), (0.0, 1.0)),
        holes=np.empty((0, 2)),
        hole_radius=0.0,
        value_range=(0.0, 1.0),
        palette=((239, 246, 255), (8, 81, 156)),
    )
    svg = _svg_document("test", panel, width=420, height=480)

    match = re.search(r'href="data:image/png;base64,([A-Za-z0-9+/=]+)"', svg)
    assert match is not None
    png = base64.b64decode(match.group(1))
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", png[16:24]) == (480, 480)
    assert 'image-rendering="auto"' in svg
    assert 'class="plot-border"' in svg
    assert 'class="plot" fill="none"' not in svg
    assert ".plot-border { fill: none;" in svg
