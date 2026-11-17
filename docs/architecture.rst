项目结构与调用流程
==================

最终流程
--------

``定义/选择问题 → 生成训练点 → 选择网络与方法 → 统一训练 → 评价、画图和保存``

十种方法是可替换的小组件，不是十套复制的训练程序。Poisson、Burgers 和以后新增的
控制方程都走同一个 ``Experiment`` 与 ``Trainer``。

论文对齐的统一能力层
--------------------

``ProblemSpec`` 保存物理问题，``MethodSpec`` 保存模型、采样、损失、优化和微分
五个方法轴。点式 PINN 与神经算子共享这套方法描述，但不强行共享同一种张量布局或
训练循环：执行计划根据模型能力选择 ``pinn`` 或 ``operator`` 后端。

.. code-block:: text

   ProblemSpec + MethodSpec
              |
              v
   CapabilityRegistry
     model: MLP / DeepONet / FNO
     differentiation: autodiff / spectral / finite_difference
              |
              v
   ExecutionPlan
     pointwise          -> PINN Experiment / Trainer
     coordinate operator -> DeepONet operator runner
     grid operator       -> FNO operator runner
              |
              v
   UnifiedRunResult -> metrics / history / artifacts / diagnostics

``default_capability_registry()`` 是两个现有构造注册表之上的共享元数据视图。
它不会把 FNO 送入点式 ``Trainer``，而是在训练前拒绝不兼容的模型、后端、训练模式和
微分组合。``MethodSpec`` 同时接受论文命名
``backend/model/optimization/differentiation`` 与旧版
``path/network/optimizer/differentiator``；``canonical_method()`` 始终输出论文对齐的五轴表示。

``ExperimentResult`` 与 ``OperatorTrainResult`` 继续保留各自的专用字段，
``as_unified_result()`` 为后处理和诊断模块提供共同的只读结果契约。

源码文件各负责什么
------------------

.. list-table::
   :header-rows: 1

   * - 文件
     - 单一职责
     - 主要方法
   * - ``geometry.py`` / ``problems.py`` / ``conditions.py``
     - 区域、控制方程、边界和初值
     - 新方程在用户文件中组装这些对象
   * - ``sampling.py``
     - 生成和更新训练点
     - 自适应采样
   * - ``models.py``
     - MLP、Modified MLP、多网络和激活
     - 改进 MLP、自适应激活、部分迁移
   * - ``features.py``
     - 坐标特征变换
     - Fourier
   * - ``constraints.py``
     - 输出变换和硬边界
     - 强制边界
   * - ``autodiff.py`` / ``formulations.py`` / ``losses.py``
     - 自动微分、残差形式、损失合并
     - 自适应权重、区域界面损失
   * - ``optimizers.py`` / ``training.py``
     - 优化器和统一训练循环
     - Adam+L-BFGS、时间窗、迁移
   * - ``metrics.py`` / ``artifacts.py``
     - 同一把评价尺和统一结果保存
     - 所有问题和方法共用
   * - ``method_registry.py``
     - 把方法家族和具体变体翻译为组件设置
     - 不包含算法和训练循环
   * - ``operator_protocol.py`` / ``operators/deeponet.py`` / ``operators/fno.py`` / ``operators/pino.py``
     - CoordinateOperator / GridOperator hooks, DeepONet, FNO, PINO, and native derivatives
     - DeepONet uses coordinate autodiff and FNO uses grid derivatives; both are selectable with shared training modes. PINO is a compatibility helper. ``solve`` stays PINN-only.
   * - ``methods/<方法家族>/``
     - 保存该家族下各具体变体的注册、默认补丁和参数转换
     - 新变体不复制公共训练流程
   * - ``experiment.py`` / ``api.py``
     - 组装底层组件与提供简短入口
     - ``Experiment`` / ``solve``

文件夹为什么分开
----------------

.. list-table::
   :header-rows: 1

   * - 文件夹
     - 内容
   * - ``src/openpinn/``
     - 真正执行计算的库代码；新增方法才需要改这里。
   * - ``configs/``
     - 问题、方法、训练和实验选择；日常使用通常只改这里。
   * - ``user_projects/``
     - 用户自己的控制方程，不混入 OpenPINN 核心源码。
   * - ``examples/``
     - 展示如何调用，不复制算法实现。
   * - ``tests/``
     - 单元、短运行、接口、兼容性和回归检查。
   * - ``runs/``
     - 每次临时训练的原始输出。
   * - ``results/``
     - 人工确认后用于汇报或交付的基线、方法结果和失败记录。
   * - ``docs/``
     - 教程、方法页、API 和本网站源文件。

新增方法的固定闭环
------------------

1. 在最合适的核心职责文件中实现公共数值组件，不复制训练程序；
2. 在 ``src/openpinn/methods/<方法家族>/`` 增加具体变体文件；
3. 在 ``method_registry.py`` 把变体加入对应家族；
4. 在 ``configs/methods/<方法家族>/`` 添加同名默认设置；
5. 增加小测试，并在指定基准上与普通 PINN 公平比较；
6. 保存误差、时间、图和失败说明，在 ``docs/methods/`` 增加说明。

方法大类与具体变体
------------------

``method="adaptive_sampling"`` 选择方法家族，
``variant="residual_adaptive"`` 选择该家族中的具体实现。省略 ``variant`` 时使用
当前默认实现，以兼容旧代码。每个变体有独立源码入口和配置，但继续复用统一方程、
训练、评价、画图和保存流程。
