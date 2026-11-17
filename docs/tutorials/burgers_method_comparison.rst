黏性 Burgers：四类方法轴的受控比较
====================================

.. admonition:: 结果状态：预印本结果

   本页数字来自随项目提供的 OpenPINN/PINO 预印本第 5.1 节及其固定参考运行。
   当前仓库已有 Burgers 脚本、smoke 和单案例演示，但尚未提交这五个正式配方的完整运行目录。

问题定义
--------

在 :math:`x\in[-1,1]`、:math:`t\in[0,1]` 上求解

.. math::

   u_t + u u_x - \nu u_{xx} = 0, \qquad \nu=0.02,

初值为 :math:`u(x,0)=-\sin(\pi x)`，两端使用齐次 Dirichlet 条件。评价使用独立的
Cole--Hopf 参考解和共享的 :math:`129\times51` 时空网格。

公平比较设计
------------

基线使用宽度 ``[64, 64, 64, 64]`` 的 tanh MLP、256 个内部点、64 个边界点、
64 个初值点、固定损失权重、3000 步 Adam 加 40 个 L-BFGS 外层步。四个变体每次只改变
模型表示、采样、损失或优化中的一个轴。

.. list-table::
   :header-rows: 1

   * - 配方
     - 唯一变化
     - 相对 L2
     - 相对基线
   * - Pin-B
     - 基线
     - :math:`1.16\times10^{-2}`
     - 1.00
   * - Pin-M
     - 16 个随机 Fourier 频率，尺度 0.25
     - :math:`2.82\times10^{-3}`
     - 0.244
   * - Pin-S
     - 残差自适应内部采样
     - :math:`2.96\times10^{-3}`
     - 0.255
   * - Pin-L
     - 逐点自适应损失权重
     - :math:`9.04\times10^{-3}`
     - 0.781
   * - Pin-O
     - 只保留 3000 步 Adam，移除 L-BFGS
     - :math:`4.47\times10^{-2}`
     - 3.86

怎样复现而不混淆结论
--------------------

仓库中的 ``examples/burgers.py``、``examples/burgers_adaptive_sampling.py`` 和
``examples/burgers_adaptive_weights.py`` 可用于检查调用方式；现有
:doc:`04_pinn_burgers` 展示另一个已提交的演示训练。正式复现本表时应新建一组锁定配置，
统一网络、配点预算、评价网格和随机种子，并把五个运行的配置与产物一起归档。

结论只适用于该 Burgers 设置。Fourier 特征和残差自适应在此处改善明显，不代表它们在
其他 PDE、预算或随机种子下必然更优；移除 L-BFGS 同时改变了优化阶段和总计算量。
