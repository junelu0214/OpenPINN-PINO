:orphan:

Tutorial: 2D plate with a hole
================================

This is both a standalone runnable tutorial and a structured linear-elasticity
editor reached through the configurator's custom governing equation. It is not
an embedded problem option or a separate solver path. The
Agent-assisted panel can route an Understanding problem draft plus a Planning
method draft into this problem without discarding either draft. The example and its M3 codegen, Kirsch
reference, and tests remain supported.

Quick start
-----------

From the repository root with the Python environment activated:

.. code-block:: bash

   python -m openpinn.agents.validator \
     examples/specs/plate_with_hole_problem.yaml \
     examples/specs/pinn_method.yaml
   python examples/plate_with_hole.py --smoke

The validator must print ``PASS``. The ten-step smoke is a wiring check, not
an accuracy result. Its JSON output should report
``status=completed_with_finite_outputs`` and ``kirsch_scf=3.0``.

What the tutorial builds
------------------------

``examples/specs/plate_with_hole_problem.yaml`` describes a 2 m by 2 m
linear-elastic plate with a central circular hole, a fixed left edge, and
uniaxial traction on the right. ``examples/specs/pinn_method.yaml`` selects
the baseline PINN. ``openpinn.codegen.from_specs`` validates those files and
returns a ``PDEProblem`` plus arguments for the existing ``openpinn.solve``
API.

The Kirsch comparison is exact for an infinite plate, while this tutorial
uses a finite plate. It is most meaningful near the hole; disagreement near
the outer boundaries is not by itself a PINN defect. A ten-step smoke also
cannot establish convergence.

Longer runs and focused tests
-----------------------------

.. code-block:: bash

   python examples/plate_with_hole.py
   python examples/plate_with_hole.py --steps 5000
   python -m pytest tests/unit/test_codegen.py tests/unit/test_kirsch.py -v
   python -m pytest tests/smoke/test_plate_with_hole_smoke.py -v

The detailed source-oriented tutorial remains in
``docs/plate_with_hole_demo.md``. The configurator-generated script uses the
same ``openpinn.codegen.from_specs`` and ``openpinn.solve`` path and writes a
self-contained Diagnostic ``RunArtifact`` after the bounded run.
