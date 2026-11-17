Tutorial：按物理问题学习 OpenPINN
===================================

本页先说明“求解什么问题”，再标注所用模型、训练模式和方法组件。每个公开算例应包含
问题定义、代码说明、运行方式、输出展示和结果边界；模型名不再代替问题名。

状态说明
--------

``已复现``
   仓库包含可运行代码、已提交指标和结果图，文档构建不会重新训练。
``可运行示例``
   已有脚本或 CPU smoke，可检查调用链路，但没有正式精度结论。
``预印本结果``
   表格来自随项目提供的预印本参考运行，当前仓库尚未包含完整复现产物。

基础方程与边值问题
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 18 16 41

   * - 物理问题
     - 方法标签
     - 状态
     - 学习重点
   * - :doc:`一维 Poisson 边值问题 <tutorials/03_pinn_poisson>`
     - 点式场网络、自动微分
     - 可运行示例
     - 从 PDE、Dirichlet 条件到解析参考评价
   * - :doc:`二维 Poisson 场求解 <tutorials/05_pinn_poisson_2d>`
     - 点式场网络、软边界
     - 已复现
     - 二维配点、损失曲线和解析场对比
   * - :doc:`二维 Poisson 参数族的解算子学习 <tutorials/07_operator_fno_physics_only>`
     - FNO、物理损失
     - 已复现
     - 规则网格上的物理训练与独立参考评价
   * - 一维 Helmholtz 波动问题（``examples/custom_pde.py``）
     - 自定义 PDE
     - 可运行示例
     - 注册制造解与新的残差
   * - 非线性反应--扩散问题（``examples/nonlinear_heat_visual.py``）
     - 可视化问题设计
     - 可运行示例
     - 非线性项、初边值与生成代码
   * - 两场耦合制造解（``examples/coupled_manufactured.py``）
     - 多场扩展
     - 可运行示例
     - 两个未知场和耦合残差

动态、非线性与方法选择
----------------------

.. list-table::
   :header-rows: 1
   :widths: 27 18 16 39

   * - 物理问题
     - 方法标签
     - 状态
     - 学习重点
   * - :doc:`黏性 Burgers 时空场 <tutorials/04_pinn_burgers>`
     - 点式场网络
     - 已复现
     - 瞬态 PDE、初边值和 Cole--Hopf 参考
   * - :doc:`黏性 Burgers 的四类方法比较 <tutorials/burgers_method_comparison>`
     - 特征、采样、损失、优化
     - 预印本结果
     - 固定问题与预算，每次只改变一个方法轴
   * - Burgers 长时间演化（``examples/burgers_time_window.py``）
     - 顺序时间窗
     - 可运行示例
     - 完整时间域与增长时间窗比较
   * - Burgers 参数迁移（``examples/burgers_transfer.py``）
     - 迁移学习
     - 可运行示例
     - 从头训练、完整迁移和末层迁移

非均质、复杂几何与工程问题
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 27 18 16 39

   * - 物理问题
     - 方法标签
     - 状态
     - 学习重点
   * - 一维非均质扩散（``examples/heterogeneous_poisson.py``）
     - 单网络
     - 可运行示例
     - 分段材料和界面条件
   * - 一维非均质扩散的区域分解（``examples/heterogeneous_poisson_domain_decomposition.py``）
     - 子域网络
     - 可运行示例
     - 解连续、通量连续与单网络对照
   * - 多边形区域 Poisson（``examples/custom_polygon_poisson.py``）
     - 自定义几何
     - 可运行示例
     - 凹多边形、逐边条件和网页导出
   * - :doc:`带圆孔板平面应力 <tutorials/05_pinn_plate_with_hole>`
     - 线弹性、点式场网络
     - 可运行示例
     - ProblemSpec、MethodSpec 和短链路训练
   * - :doc:`带孔板的五 Agent 诊断闭环 <tutorials/plate_hole_agent_workflow>`
     - 五 Agent、FEM 参考
     - 预印本结果
     - SI 失败、无量纲化、诊断和 L-BFGS 续训
   * - :ref:`稳态 Darcy 渗流的渗透率识别 <darcy-parameter-identification>`
     - PINN、稀疏观测
     - 可运行示例
     - 压力场与有界渗透率的联合识别
   * - :ref:`Terzaghi 固结问题族的固结系数识别 <consolidation-parameter-identification>`
     - DeepONet、算子反演
     - 可运行示例
     - 复用已训练问题族算子，由观测识别固结系数
   * - 双隧道固结与孔压扩散（``examples/tunnel_model1.py``）
     - 时空多物理
     - 可运行示例
     - 孔压场、参考数据和工程单位

参数化问题与解算子学习
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 18 16 36

   * - 物理问题
     - 方法标签
     - 状态
     - 学习重点
   * - :doc:`参数化反导数映射 <tutorials/06_operator_deeponet_data_only>`
     - DeepONet、数据监督
     - 可运行示例
     - 输入函数到积分结果的函数空间映射
   * - :doc:`二维 Poisson 参数族的解算子学习 <tutorials/07_operator_fno_physics_only>`
     - FNO、物理监督
     - 已复现
     - 规则网格、物理残差和训后参考
   * - :doc:`周期 Poisson 参数族的损失与微分比较 <tutorials/periodic_poisson_operator>`
     - FNO、混合损失、谱微分
     - 预印本结果
     - 加性常数、均值自由误差和受控单轴比较
   * - :doc:`二维 Poisson 参数族的混合监督 <tutorials/08_operator_hybrid>`
     - FNO、数据与物理
     - 可运行示例
     - 观测项与物理项的角色
   * - :doc:`自定义坐标 PDE 的两种监督方案 <tutorials/09_extension_custom_pde_and_mode>`
     - DeepONet、物理/混合监督
     - 可运行示例
     - 同一问题规范下切换训练目标

五 Agent 与产物工作流
--------------------------

* :doc:`安装、首页工作台与第一份配置 <tutorials/01_intro_install_and_configurator>`
* :doc:`读取训练产物与指标 <tutorials/02_intro_read_artifacts>`
* :doc:`带孔板的五 Agent 诊断闭环 <tutorials/plate_hole_agent_workflow>`
* :doc:`工作台教程 <configurator_tutorial>`

.. toctree::
   :maxdepth: 1
   :hidden:

   tutorials/01_intro_install_and_configurator
   tutorials/02_intro_read_artifacts
   tutorials/03_pinn_poisson
   tutorials/04_pinn_burgers
   tutorials/05_pinn_poisson_2d
   tutorials/05_pinn_plate_with_hole
   tutorials/06_operator_deeponet_data_only
   tutorials/07_operator_fno_physics_only
   tutorials/08_operator_hybrid
   tutorials/09_extension_custom_pde_and_mode
   tutorials/burgers_method_comparison
   tutorials/periodic_poisson_operator
   tutorials/plate_hole_agent_workflow
   inverse_seepage_consolidation

所有 notebook 源文件位于 ``examples/tutorials/``。Sphinx 只展示已提交输出；正式刷新结果时，
必须同时保存配置、随机种子、硬件、指标和失败说明。
