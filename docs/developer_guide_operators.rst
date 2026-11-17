Operator evaluation and model-native derivatives
================================================

This page is the product contract for operator-learning families. It states
how operator models are evaluated, why each uses its natural derivative, and
how DeepONet/FNO combine with a training mode. Coordinate/grid adaptation is
not a formal product path.
``solve`` remains the PINN entry point.

The parallel work repository ``junelu0214/OpenPINN`` is a protocol reference
only. The public evaluation signatures here align with its ``predict``
contracts, while product changes and implementations remain in this
repository; its package layout is not vendored.

The numerical PINN families that already ship (baseline, Fourier features,
adaptive sampling, and the rest) stay on :class:`openpinn.MethodRegistry` and
``docs/method_catalog.yaml``. Operator hooks are a separate registry so those
families and their generated training scripts stay as they are.

CoordinateOperator
------------------

A coordinate operator maps continuous query coordinates to field values:

* input coordinates have shape ``(n_points, n_dims)``;
* the formal call is ``predict(coordinates, condition=None)``;
* an optional ``condition`` tensor carries an encoding of an input function
  (``None``, ``(n_branch,)``, or ``(n_points, n_branch,)``);
* the output has shape ``(n_points, n_fields)``.

DeepONet implements this contract. Its ``condition`` argument is the sensor
sample of an input function: one shared vector ``(n_sensors,)`` or a
per-query matrix ``(n_points, n_sensors)``. The branch network inside the
module builds the latent code. ``condition=None`` uses a learned default
sensor vector. A plain coordinate network can still leave ``condition`` unset.

For compatibility, ``DeepONet.forward(coordinates, branch=...)`` delegates to
``predict(coordinates, condition=...)`` and remains supported for at least one
minor version. New integrations should use ``predict`` and ``condition``.
The old protocol spelling ``CoordinateModel`` is an exact deprecated alias of
:class:`openpinn.CoordinateOperator` for one minor version.

Classic PINN objects such as :class:`openpinn.PINNModel` already evaluate
fields at coordinates, but they are not operator hooks. Training them still
goes through :class:`openpinn.Experiment` and :func:`openpinn.grad`.
This milestone does not wrap those models.

GridOperator
------------

A grid operator maps a discretized input field on a structured grid to an
output field on that same grid:

* the formal call is ``predict(inputs, *, grid=...)`` and ``grid`` is
  keyword-only;
* input shape ``(batch, in_channels, *spatial)``, with at least one spatial
  axis;
* output shape ``(batch, out_channels, *spatial)`` with the spatial size
  unchanged.

FNO implements this contract with Fourier layers on 1D and 2D grids.
:class:`openpinn.UniformGrid` supplies
axis ``nodes``, ``spacing``, ``periodic``, ``shape``, ``n_points``, and
``n_dims`` metadata. Existing ``forward(grid_field)`` calls remain a
grid-free compatibility alias for at least one minor version; new code should
pass grid metadata to ``predict``.

Reference-style evaluation therefore plugs directly into the product API::

   import math
   import torch
   import openpinn as op

   grid = op.UniformGrid.from_shape(32, spacing=2.0 * math.pi / 32, periodic=True)
   output = op.FNO().predict(torch.zeros(1, 1, 32), grid=grid)

Model-native derivatives
------------------------

The selectable models use different natural derivatives. DeepONet physics is
formed directly at query coordinates; FNO physics is formed on its output
grid. The configurator does not present cross-resampling as another model
capability.

``derivative_mode()`` remains metadata on the built-in models and registry
hooks; it is not required for structural ``isinstance`` checks. A reference-
style object that implements only the appropriate ``predict`` signature
satisfies its runtime-checkable protocol.

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - 
     - CoordinateOperator
     - GridOperator
   * - ``derivative_mode()``
     - ``"autodiff"``
     - ``"grid"``
   * - What is differentiated
     - The output with respect to the coordinate tensor
     - The output grid along a spatial axis
   * - Mechanism
     - :func:`openpinn.grad` and :func:`openpinn.hessian`
       (``torch.autograd``)
     - :class:`openpinn.GridDerivative`
   * - Why
     - Each output row depends on its coordinate row, so coordinate autodiff
       is the PDE derivative.
     - The network input is a grid tensor. Coordinate autodiff does not see
       a spatial derivative of that field.

``GridDerivative.partial`` takes a field of shape
``(batch, channels, *spatial)``, a spatial ``axis`` (``0`` is the first
spatial axis, not batch or channel), and ``order`` (``1`` or ``2``). The
result has the same shape as the field. Two periodic stencils ship with FNO:

* :class:`openpinn.SpectralGridDerivative` multiplies FFT modes by
  ``(2π i ξ)^order``;
* :class:`openpinn.FiniteDifferenceGridDerivative` uses wrapped central
  differences.

:func:`openpinn.make_grid_derivative` selects one by name. FNO's and PINO's
``grid_derivative()`` helpers return the same objects. FNO physics modes minimize
a :class:`openpinn.GridPDEProblem` residual with those derivatives.
Concrete stencils also expose ``derivative(field, axis)`` and
``second_derivative(field, axis)`` as thin first/second-order adapters;
``partial`` remains supported.

Hook registry and the configurator
-----------------------------------

:class:`openpinn.OperatorRegistry` stores :class:`openpinn.OperatorHookSpec`
records. :func:`openpinn.list_operator_hooks` returns the built-in slots:

.. list-table::
   :header-rows: 1
   :widths: 18 16 16 14 36

   * - Name
     - Kind
     - Derivative
     - Milestone
     - Status in this repository
   * - ``deeponet``
     - coordinate
     - autodiff
     - M1c
     - Implemented branch/trunk model. Selectable through
       ``build_operator_config`` and ``train_operator``. Absent from
       ``list_methods``.
   * - ``fno``
     - grid
     - grid
     - M1d
     - Implemented Fourier-layer grid operator. Selectable through
       ``build_operator_config`` and ``train_operator``. Absent from
       ``list_methods``.
   * - ``pino``
     - grid
     - grid
     - M1e
     - Deprecated, non-selectable compatibility record. Choose model ``fno``
       with ``training_mode="physics_only"`` or ``"hybrid"``. The old name
       remains a warning shim for one minor version.

Registration rules:

* A coordinate hook must use derivative mode ``autodiff``. A grid hook must
  use derivative mode ``grid``.
* ``selectable=True`` is rejected unless ``implemented=True``.
* :func:`openpinn.default_operator_registry` builds a new registry on every
  call. A stub registered on one instance is not visible on the next call
  and is not written into the process-wide method list.
* These names stay absent from :func:`openpinn.list_methods`. ``solve`` and
  ``build_config`` remain the PINN catalog. Passing ``method="deeponet"``,
  ``method="fno"``, or ``method="pino"`` raises an error that points at
  :func:`openpinn.build_operator_config` and :func:`openpinn.train_operator`.

The documentation configurator discovers the same records beside the PINN
families, but its canonical model choices include selectable records only:
DeepONet and FNO. :func:`openpinn.attach_operator_hooks` adds an
``operator_hooks`` list to the configurator JSON payload. Each record includes
a ``parameters`` object and the built-in trainer limits. The separate
``training_mode`` control has exactly ``data_only``, ``physics_only``, and
``hybrid``. The page script that builds PINN method checkboxes still reads
``families`` only.

The method configurator can also write a local ``.py`` file for a selectable
operator. Its operator path reuses the PINN configurator's existing 1D/2D
geometry, PDE, boundary-condition, and visualization controls. The generated
file constructs that exact ``PDEProblem`` and keeps it as
``DESIGNED_PROBLEM`` beside calls to ``build_operator_config`` and
``train_operator_problem``. ``solve`` stays on the PINN path. The browser does not
train.

The public ``train_operator_problem`` entry accepts a coordinate
``PDEProblem`` plus optional local CSV observations. DeepONet evaluates the
residual with coordinate autodiff. FNO evaluates it on a complete uniform
Cartesian grid and scopes ``grad``/``hessian`` to ``GridDerivative``; it does
not resample through a coordinate/grid model adapter. ``data_only`` and
``hybrid`` require observations, while ``physics_only`` does not. The
configurator keeps that training CSV separate from the optional post-train
reference CSV: generated scripts emit ``OBSERVATION_*`` only for
``train_operator_problem`` and ``REFERENCE_*`` only for
``evaluate_operator_reference_csv``. An unknown
name, a non-selectable hook, an unknown option, an unsupported dimension, or
an incomplete FNO observation grid fails loudly. The UI shows canonical model names
only; API aliases such as ``deep_operator_network`` remain accepted.
``physics_only`` is left as ``physics_only``. A collapsed advanced section
lists derivative, sampler, and optimizer names from the component registries
and writes the selections to ``EXTENSIONS`` in the generated script.

The plate-with-hole codegen remains tested, but its user entry is the
:doc:`plate_with_hole_demo` tutorial rather than a configurator solution
path.

Adding those names to ``docs/method_catalog.yaml`` would put them through
the PINN slot checks, so the method catalog stays unchanged.

DeepONet
--------

:class:`openpinn.DeepONet` is an unstacked branch/trunk network. The branch
MLP reads sensor values of an input function. The trunk MLP reads query
coordinates. The output field is their latent dot product plus an optional
bias. Coordinate derivatives of that field use :func:`openpinn.grad`.

The published architecture is Lu, Jin, Pang, Zhang, and Karniadakis,
`Nature Machine Intelligence (2021) <https://doi.org/10.1038/s42256-021-00302-5>`_.
This module is an independent PyTorch implementation. Its public evaluation
signature is aligned with the work repository; that repository's tree is not
vendored here.

Configurator knobs live on the operator hook and on
:func:`openpinn.build_operator_config`: sensor count, query dimension, output
fields, latent width, branch and trunk hidden widths, activation, output
bias, and the smoke-trainer settings (learning rate, steps, batch size,
queries, and seed). ``training_mode`` is shared with FNO. Synthetic smoke
batches still use the schema default for ``n_modes``; that parameter is
``ui_hidden`` and the configurator does not render it.

Selection path::

   import torch
   import openpinn as op

   config = op.build_operator_config(
       "deeponet",
       n_sensors=8,
       latent_dim=8,
       branch_hidden=(8, 8),
       trunk_hidden=(8, 8),
   )
   model = op.build_deeponet(config)
   coordinates = torch.linspace(0.0, 1.0, 4).unsqueeze(1)
   values = model.predict(coordinates, condition=torch.zeros(model.n_sensors))
   result = op.train_operator(config, steps=5, device="cpu")

``train_operator`` fits the scalar 1D antiderivative on one fixed synthetic
batch when ``training_mode="data_only"``. Physics modes build ``-u_xx-f`` at
query points and differentiate the DeepONet prediction with
:func:`openpinn.hessian`; no structured grid or ``GridPDEProblem`` is involved.
They prove residual execution, not paper-grade accuracy. That smoke supports
``n_dims=1`` and ``n_fields=1``. Other shapes are constructed with
:func:`openpinn.build_deeponet` and trained by the caller.

FNO
---

:class:`openpinn.FNO` is a lifted Fourier-layer network on a structured grid.
A 1×1 convolution raises the input channels, each block adds a truncated
spectral convolution to a local channel mix, and a two-layer projection
writes the output channels. 1D and 2D grids are implemented. Coordinate
autodiff is not used; :meth:`openpinn.FNO.grid_derivative` returns a
:class:`openpinn.GridDerivative` for PDE-side residuals of the output grid.

The published architecture is Li, Kovachki, Azizzadenesheli, Liu,
Bhattacharya, Stuart, and Anandkumar,
`ICLR (2021) <https://arxiv.org/abs/2010.08895>`_. This module is an
independent PyTorch implementation. Its public evaluation signature is
aligned with the work repository; that repository's tree is not vendored here.

Configurator knobs live on the operator hook and on
:func:`openpinn.build_operator_config`: spatial rank, input and output
channels, lifted width, layer count, retained Fourier modes, projection
width, activation, the smoke-trainer settings (learning rate, steps, batch
size, grid size, harmonics, seed), and which grid derivative to attach
(``spectral`` or ``finite_difference``).

Selection path::

   import math
   import torch
   import openpinn as op

   config = op.build_operator_config(
       "fno",
       width=8,
       n_modes=4,
       n_layers=2,
       grid_size=32,
   )
   model = op.build_fno(config)
   result = op.train_operator(config, steps=5, device="cpu")
   field = torch.randn(1, 1, result.config["grid_size"])
   dx = 2.0 * math.pi / result.config["grid_size"]
   grid = op.UniformGrid.from_shape(result.config["grid_size"], spacing=dx)
   prediction = result.model.predict(field, grid=grid)
   residual = result.model.grid_derivative("spectral", spacing=dx).partial(
       prediction, axis=0, order=2
   )

``data_only`` fits the scalar 1D periodic derivative on one fixed synthetic
batch. ``physics_only`` and ``hybrid`` inject the FNO into the M10 grid-PDE
loop, whose default is periodic Poisson. That smoke supports ``n_dims=1``,
``in_channels=1``, and ``out_channels=1``. Other shapes are constructed with
:func:`openpinn.build_fno` and trained by the caller.
:func:`openpinn.list_selectable_operators` returns DeepONet and FNO.

Physics training and PINO compatibility
---------------------------------------

:class:`openpinn.PINO` is the default FNO-backed model for a physics-informed
grid-operator trainer. ``predict(inputs, *, grid=...)`` is the formal grid
map; ``forward`` is its compatibility path. ``pde_residual`` remains a
compatibility helper that forms the periodic Poisson operator ``-Δu - f``
with :class:`openpinn.GridDerivative`. Coordinate autodiff is not used.

The training loop is not tied to that model or equation. A
:class:`openpinn.GridPDEProblem` is a structural protocol with three hooks:

* ``residual(solution, inputs, *, grid, derivative)`` returns a pointwise
  grid residual;
* ``boundary_residual(solution, inputs, *, grid)`` returns an optional BC
  residual (or ``None``); and
* ``physics_objective(residual, *, grid, mode, derivative)`` reduces the PDE
  residual to a scalar. The trainer adds the mean-square BC term when one is
  returned.

:class:`openpinn.PeriodicPoisson1D` supplies those hooks for the built-in
tutorial. It uses raw residual MSE in hybrid mode and retains the spectral
inverse-Laplacian objective in physics-only mode. This operators-side API is
separate from :class:`openpinn.PDEProblem`: the coordinate-PINN ``solve()``
contract is unchanged.

The published method is Li, Zheng, Kovachki, Jin, Chen, Liu,
Azizzadenesheli, and Anandkumar,
`arXiv 2111.03794 <https://arxiv.org/abs/2111.03794>`_. This module is an
independent PyTorch implementation of the hybrid data-plus-physics idea. Its
public evaluation signature is aligned with the work repository; that
repository's tree is not vendored here.

Three training modes are first-class shared knobs, while their physics
residual follows the selected model:

* ``data_only`` (default) keeps each model's existing supervised smoke.
* ``hybrid`` combines labeled MSE with the model-native physics residual.
  ``data_pretrain_steps`` runs data-only Adam updates first, then the
  combined fine-tune. Simultaneous hybrid is ``data_pretrain_steps=0``.
* ``physics_only`` drops the labeled solution from the objective. DeepONet
  minimizes its coordinate-autodiff residual directly. The FNO spectral 1D
  periodic Poisson trainer applies the inverse Laplacian to the
  residual: each nonzero Fourier mode is weighted by ``1/k²`` and the zero
  mode is discarded. This objective equals centered solution error without
  using the labeled target. The trainer still reports labeled data metrics
  for evaluation and the raw residual MSE as a diagnostic. Hybrid remains
  the practical default for objectives outside this built-in Poisson smoke.

Configurator knobs reuse the selected model widths and add ``training_mode``,
``data_weight``, ``physics_weight``, and ``data_pretrain_steps``.

Selection path::

   import math
   import torch
   import openpinn as op

   config = op.build_operator_config(
       "fno",
       training_mode="hybrid",
       width=8,
       n_modes=4,
       n_layers=2,
       grid_size=32,
       data_pretrain_steps=4,
   )
   model = op.build_fno(config)
   result = op.train_operator(config, steps=8, device="cpu")
   generator = torch.Generator().manual_seed(0)
   grid_x, forcing, _target = op.poisson_operator_batch(
       1, result.config["grid_size"], generator=generator
   )
   grid = op.UniformGrid(grid_x, spacing=float(grid_x[1] - grid_x[0]))
   residual = op.PeriodicPoisson1D().residual(
       result.model.predict(forcing, grid=grid), forcing,
       grid=grid, derivative=result.config["derivative"],
   )
   physics = result.metrics["final_physics_loss"]

``train_operator("pino", mode=...)`` remains for one minor version as a
deprecated warning shim to ``fno`` plus ``training_mode``. PINO is not a
configurator model or preset. :class:`openpinn.PINO`,
:func:`openpinn.build_pino`, and :func:`openpinn.train_pino` remain advanced
implementation helpers.

Custom problem and backbone injection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Advanced callers use :func:`openpinn.train_pino` with the same resolved
configuration. The injected model must be a trainable ``torch.nn.Module``
that structurally satisfies :class:`openpinn.GridOperator`; the loop calls
only ``predict(inputs, *, grid=...)`` and does not inspect FNO internals. For
example, this non-Poisson identity residual and non-FNO backbone hang on the
same loop::

   import torch
   import openpinn as op

   class IdentityProblem:
       def residual(self, solution, inputs, *, grid, derivative="spectral"):
           return solution - inputs

       def boundary_residual(self, solution, inputs, *, grid):
           return None

       def physics_objective(self, residual, *, grid, mode, derivative):
           return residual.square().mean()

   class ScaledIdentity(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self.scale = torch.nn.Parameter(torch.tensor(0.0))

       def predict(self, inputs, *, grid):
           return self.scale * inputs

   config = op.build_operator_config(
       "fno", training_mode="physics_only", steps=8, grid_size=16, n_modes=4
   )
   result = op.train_pino(
       config,
       problem=IdentityProblem(),
       model=ScaledIdentity(),
       device="cpu",
   )

Omitting ``problem`` and ``model`` in :func:`openpinn.train_pino` selects
:class:`openpinn.PeriodicPoisson1D` and the FNO-backed
:class:`openpinn.PINO`. ``forward`` remains a compatibility alias for existing
FNO/PINO callers. This grid trainer is the FNO physics path; DeepONet uses its
separate coordinate-autodiff trainer and is not injected here.

Registering a stub
------------------

A unit test can prove the hook imports and accepts a stand-in. This is not a
training step and it does not add a public model class::

   import torch
   import openpinn as op

   class StubCoordinateOperator:
       def predict(self, coordinates, condition=None):
           return coordinates.sum(dim=-1, keepdim=True)

   model = StubCoordinateOperator()
   assert isinstance(model, op.CoordinateOperator)
   values = model.predict(torch.zeros(4, 2), condition=None)

   registry = op.OperatorRegistry()
   registry.register(op.OperatorHookSpec(
       name="stub_coordinate",
       kind="coordinate",
       derivative_mode="autodiff",
       title="Stub coordinate model",
       summary="Import-and-register smoke test. Not a trainable operator.",
       implemented=True,
       selectable=False,
   ))
   assert registry.resolve("stub_coordinate").kind == "coordinate"

``implemented=True`` on a local stub means "this registry entry has a
Python object in the test." It does not publish a method, and ``selectable``
stays false so the stub cannot be mistaken for a configurator option.

Verification
------------

Run the operator tests, then the same CPU suite used in continuous integration::

   python -m pytest -q tests/unit/test_operator_protocol.py tests/unit/test_deeponet.py tests/unit/test_fno.py tests/unit/test_pino.py
   python -m pytest -q

The second command is the CI pytest selection. DeepONet and FNO each train a
data-only smoke; DeepONet physics uses coordinate autodiff and FNO physics
uses the grid-PDE loop. FNO's spectral
physics-only smoke checks centered periodic Poisson recovery and a decrease
in the raw residual.
