常用 Python 接口
==============================

以下页面由代码中的说明自动生成。普通用户从 ``solve`` 或 ``solve_config``
开始；需要精确控制组件或新增方法时，再查看 ``Experiment``、网络、采样和训练类。

配置和运行
----------

.. autofunction:: openpinn.solve

.. autofunction:: openpinn.method

.. autoclass:: openpinn.MethodSelection
   :members:

.. autofunction:: openpinn.solve_config

.. autofunction:: openpinn.build_config

.. autofunction:: openpinn.list_methods

.. autofunction:: openpinn.list_method_variants

.. autofunction:: openpinn.check_method_compatibility

.. autoclass:: openpinn.ProblemCompatibilityReport
   :members: applicable, conditional, incompatible, to_text

.. autofunction:: openpinn.load_config

.. autoclass:: openpinn.Experiment
   :members: from_config, run

.. autoclass:: openpinn.ExperimentResult
   :members:

.. autofunction:: openpinn.evaluate_reference_csv

方法注册
--------

.. autoclass:: openpinn.MethodSpec
   :members:

.. autoclass:: openpinn.MethodCompatibility
   :members:

.. autoclass:: openpinn.MethodVariantSpec
   :members:

.. autoclass:: openpinn.MethodRegistry
   :members:

Component registry
------------------

First-class factories for ``model``, ``sampler``, ``loss``, ``optimizer``,
and ``derivative`` (plus the existing ``problem`` / ``feature`` /
``constraint`` / ``formulation`` kinds).  YAML names resolve against this
registry; unknown names raise ``ValueError``.  See ``docs/extension.md``.

.. autoclass:: openpinn.ComponentRegistry
   :members:

.. autofunction:: openpinn.default_component_registry

.. autofunction:: openpinn.component_kinds

Unified capability and result contracts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: openpinn.ModelCapability

.. autoclass:: openpinn.DerivativeCapability

.. autoclass:: openpinn.ExecutionPlan

.. autoclass:: openpinn.CapabilityRegistry

.. autofunction:: openpinn.default_capability_registry

.. autofunction:: openpinn.resolve_execution_plan

.. autoclass:: openpinn.UnifiedRunResult

.. autofunction:: openpinn.as_unified_result

.. autoclass:: openpinn.AutodiffDerivative
   :members:

.. autofunction:: openpinn.assert_no_pde_residual_in_forward

算子协议
--------

Model-native operator hooks. DeepONet and FNO combine with a shared
``training_mode`` through ``build_operator_config`` and ``train_operator``.
PINO remains an advanced compatibility helper, not a selectable model. ``solve`` remains
the PINN entry point. See :doc:`developer_guide_operators`.

.. autoclass:: openpinn.CoordinateOperator
   :members:

``CoordinateModel`` is a deprecated alias retained for one minor version.

.. autoclass:: openpinn.CoordinateModel
   :members:

.. autoclass:: openpinn.GridOperator
   :members:

.. autoclass:: openpinn.GridDerivative
   :members:

.. autoclass:: openpinn.UniformGrid
   :members:

.. autoclass:: openpinn.OperatorHookSpec
   :members:

.. autoclass:: openpinn.OperatorRegistry
   :members:

.. autofunction:: openpinn.default_operator_registry

.. autofunction:: openpinn.list_operator_hooks

.. autofunction:: openpinn.operator_hook_records

.. autofunction:: openpinn.attach_operator_hooks

.. autofunction:: openpinn.list_selectable_operators

.. autofunction:: openpinn.build_operator_config

.. autofunction:: openpinn.build_deeponet

.. autofunction:: openpinn.build_fno

.. autofunction:: openpinn.build_pino

.. autofunction:: openpinn.train_operator

.. autoclass:: openpinn.DeepONet
   :members:

.. autoclass:: openpinn.DeepONetConfig
   :members:

.. autoclass:: openpinn.FNO
   :members:

.. autoclass:: openpinn.FNOConfig
   :members:

.. autoclass:: openpinn.PINO
   :members:

.. autoclass:: openpinn.PINOConfig
   :members:

.. autoclass:: openpinn.SpectralConv
   :members:

.. autoclass:: openpinn.SpectralGridDerivative
   :members:

.. autoclass:: openpinn.FiniteDifferenceGridDerivative
   :members:

.. autofunction:: openpinn.make_grid_derivative

.. autofunction:: openpinn.poisson_operator_batch

.. autoclass:: openpinn.OperatorTrainResult
   :members:

网络与特征
----------

.. autoclass:: openpinn.MLP
   :members:

.. autoclass:: openpinn.PINNModel
   :members:

.. autofunction:: openpinn.freeze_all_but_last_linear_layers

.. autoclass:: openpinn.FourierFeatures
   :members:

采样与训练
----------

.. autoclass:: openpinn.UniformSampler
   :members:

.. autoclass:: openpinn.Trainer
   :members: fit, predict, save, load

共用强形式引擎
----------------

``run_pinn`` 在现有 ``Trainer`` 循环外提供显式步数与墙钟预算、检查点和书面状态。
它接受 ``PDEProblem`` 或 ``FormulationProtocol``，默认使用 CPU。该入口是供后续 adapter
挂接的库级引擎，不会使网页 CodeBuilder 骨架成为可运行后端。

.. autofunction:: openpinn.run_pinn

.. autoclass:: openpinn.EngineRunResult
   :members:

.. autofunction:: openpinn.problem_from_compiled_expressions

单标量强形式适配器
------------------

``match_training_backend`` 按坐标、范围、算子、参数和边初值条件匹配冻结的 Poisson/heat
问题卡，也可匹配合同范围内的一维/二维、可选时间、多残差单标量强形式问题。通用能力支持
Dirichlet、坐标导数型 Neumann 与 Robin；不支持多场、向量场、复杂几何或周期条件。
``run_registered_pinn`` 始终复用上面的共用引擎，并写同次运行的 ``run_artifact.json``。
只有冻结卡有参考误差；通用问题不会猜测参考解或精度阈值。

.. autofunction:: openpinn.match_training_backend

.. autoclass:: openpinn.TrainingCapability
   :members:

.. autofunction:: openpinn.build_registered_problem

.. autofunction:: openpinn.run_registered_pinn

已有问题
--------

.. autoclass:: openpinn.PDEProblem
   :members: evaluate_residual, exact

.. autoclass:: openpinn.Polygon
   :members: from_file, sample, boundary, contains, on_boundary, boundary_normal

.. autoclass:: openpinn.DirichletBC
   :members:

.. autoclass:: openpinn.NeumannBC
   :members:

.. autoclass:: openpinn.RobinBC
   :members:

.. autofunction:: openpinn.make_poisson_problem

.. autofunction:: openpinn.make_piecewise_poisson_problem

.. autofunction:: openpinn.make_burgers_problem

.. autofunction:: openpinn.make_coupled_manufactured_problem

.. autofunction:: openpinn.make_tunnel_consolidation_model1_problem

.. autoclass:: openpinn.RectangleWithCircularHoles
   :members:
