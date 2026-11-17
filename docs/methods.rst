十类改进方法详解
================

五轴说明页见 :doc:`method_ecosystem`，结构与 OpenPINN/PINO 预印本 4.4 节一致。
本页保留原有十类实验记录和配对短训结果，并链接问题导向的论文级受控比较。

OpenPINN 已提供网络结构、激活函数、Fourier 特征、硬边界约束、
自适应权重与采样、组合优化器、时间窗、迁移学习和区域分解等方法。
每个方法页都给出调用方式、适用案例、对照设置、参考实验和已知限制。

统一选择方式
------------

所有方法使用同一个入口。``method`` 选择方法大类，``variant`` 选择该大类中的
具体实现：

.. code-block:: python

   import openpinn as op

   plain = op.solve(
       "poisson", method="baseline", variant="plain_pinn", steps=5000
   )
   improved = op.solve(
       "poisson", method="fourier", variant="random_fourier", steps=5000
   )

也可以运行 ``python -m openpinn list-methods`` 查看家族，再用
``python -m openpinn list-methods --family adaptive_sampling`` 查看具体变体。
``configs/methods/`` 按方法家族建立文件夹并保存变体默认值，
``configs/problems/`` 保存问题设置；``configs/reference_runs/``
中的“问题+方法”配置只是已经完成的公平对比记录，不是日常用户必须复制的入口。

省略 ``variant`` 时仍使用当前默认实现，所以旧代码不需要修改。十类是功能家族，
不是把具体算法数量锁死为十个；自适应采样和损失平衡家族现在已经有多个平行变体。

全部家族、具体变体、中文名称、适用方向和论文原文统一列在
:doc:`method_catalog`。该页面由 ``docs/method_catalog.yaml`` 自动生成，
不再在多个文档中手工维护重复表格。

五轴注册表的公式边界
--------------------

配置器卡片上的“已内置”表示默认注册表能够构造该对象。下面几项的注册名容易被读成
另一篇文献里的完整算法，实际计算仍以这里的代码路径和公式为准；预印本中的方法家族名称
不改变实现边界。

* ``gauss_newton``（别名 ``gn``）由 ``solve()`` 的 ``optimizer.name`` 调用。一步是 ``parameter -= lr * grad / (|grad| + damping)``。
* ``natural_gradient``（别名 ``ngd``）由 ``solve()`` 的 ``optimizer.name`` 调用。``v`` 是梯度平方的指数滑动，一步是 ``parameter -= lr * grad / (sqrt(v) + damping)``。
* ``ntk_balanced``（别名 ``ntk_weighting``）由 ``solve()`` 的 ``loss.weighting`` 调用。统计量是该损失项对参数梯度的平方和。
* ``finite_volume``（别名 ``fv``）的一阶 ``partial`` 等于 ``(u[i+1] - u[i-1]) / (2 * spacing)``，端点按周期卷绕。入口是 ``registry.create("derivative", "finite_volume").partial(field, axis)``。强形式 ``solve()`` 在缺少坐标 ``grad``/``hessian`` 后拒绝训练。
* ``weak_variational``（别名 ``vpinn``、``weak``）可由 ``differentiator.mode`` 安装。其 ``grad``/``hessian`` 仍是坐标自动微分；``integrate_by_parts`` 单独计算 ``-sum(flux * test_gradient * weights)``，强形式残差不调用它。
* ``hybrid_ad_numerical``（别名 ``can_pinn``、``hybrid``）可由 ``differentiator.mode`` 安装。强形式使用的 ``grad`` 仍是坐标自动微分；``partial`` 是周期网格差分，两条路径不相加。

增加同一家族的第二种方法
--------------------------

例如第二种自适应采样不应再建立一套训练器，而应增加平行文件：

.. code-block:: text

   src/openpinn/methods/adaptive_sampling/
   ├── residual_adaptive.py
   └── 新方法简称.py

   configs/methods/adaptive_sampling/
   ├── residual_adaptive.yaml
   └── 新方法简称.yaml

同时增加 ``tests/methods/adaptive_sampling/``、对应方法说明和正式结果。没有论文、
算法定义或许可边界时，不把占位名字冒充成已经实现的方法。

阅读方法页时建议关注七项信息
------------------------------

1. **作用**：它希望解决训练困难、高频解、非均质、长时间计算还是其他问题；
2. **依据**：记录主要论文、公式和参考代码链接；
3. **使用方法**：说明需要修改哪个设置、运行哪条命令；
4. **适用案例**：Poisson、非均质 Poisson、Burgers 或后续波动方程；
5. **普通方法**：保留一份普通 PINN 设置作为对照；
6. **结果**：保存误差、残差、时间、硬件、图表和随机种子；
7. **问题**：如实记录没有提升、运行失败或不适用的情况。

比较方法的推荐流程
--------------------

1. 先运行普通 PINN 短案例；
2. 复制一份配置，只改变目标方法需要的设置；
3. 实现方法并增加小测试；
4. 运行改进方法短案例，确认输出都是有限值；
5. 再进行较长训练；
6. 把两组结果放在同一张表或同一张图中；
7. 运行全部测试，确认旧案例没有被破坏。

已接入的方法
--------------

.. toctree::
   :maxdepth: 1

   methods/baseline
   methods/fourier_features
   methods/modified_mlp
   methods/adaptive_activation
   methods/hard_boundary
   methods/adaptive_weights
   methods/adaptive_sampling
   methods/adam_lbfgs
   methods/time_window
   methods/transfer_learning
   methods/domain_decomposition

新增二级变体
--------------

同一家族可以继续增加任意数量的二级变体，不受“十类方法”数量限制。新增记录后，
:doc:`method_catalog` 会自动把相同 ``method`` 合并在一起，并把 ``variant`` 链接到
对应详情页；完整步骤也写在该页面底部。

十种方法的实际结果摘要
------------------------

下表汇总 CPU 上单一固定随机种的 **短训参考** 实验，用于展示方法的可运行性和记录负结果，
不是充分收敛的精度排名。Burgers 解误差来自独立的 Cole–Hopf 参考解，只在训练完成后评估，
没有进入训练损失。每行同时列出同一方法页参考实验中的配对对照（普通 PINN、固定权重、
仅 Adam、目标域从头训练等），并非十种方法共用同一基线。三随机种子稳定性结果见
:doc:`reproducibility`；本表展示各方法页所引用的固定种正式运行。

表中的相对 L2、PDE 残差和用时都来自这些配对短训；相对 L2 接近或高于 1，只说明该次短训
尚未达到实用精度，不代表方法经过更长训练仍然无效。需要查看训练更充分的演示路径时，
请阅读 :doc:`configurator_tutorial` 和 :doc:`code_tutorials`。本表中高频 Poisson 的 Fourier、
Modified MLP 与硬边界三行已按 5000 步短训加长结果刷新；其余七行仍为原短训预算下的参考数字。
整表仍是短训参考，不是充分收敛排名。

.. list-table::
   :header-rows: 1

   * - 方法
     - 案例
     - 基线相对 L2
     - 改进组相对 L2
     - 基线 PDE 残差
     - 改进组 PDE 残差
     - 基线用时
     - 改进组用时
     - 硬件
     - 参考实验结论
   * - Fourier 特征
     - 高频 Poisson
     - 0.02585
     - 0.002066
     - 1.61061
     - 0.15380
     - 48.098 s
     - 88.401 s
     - CPU
     - 明显改善，但参数和时间增加
   * - Modified MLP
     - 高频 Poisson
     - 0.07654
     - 0.52236
     - 1.48772
     - 1.12530
     - 45.409 s
     - 95.606 s
     - CPU
     - 改进组相对 L2/边界更差（负结果保留）
   * - 自适应激活
     - Burgers
     - 0.40998
     - 0.40835
     - 0.22334
     - 0.22629
     - 5.289 s
     - 7.148 s
     - CPU
     - L2/边界/初值略好，最大误差、PDE 和时间更差
   * - 硬边界约束
     - 高频 Poisson
     - 0.07654
     - 0.03423
     - 1.48772
     - 1.86982
     - 63.323 s
     - 69.700 s
     - CPU
     - 边界精确为零且相对 L2 改善，但 PDE 残差更高
   * - 自适应权重
     - Burgers
     - 0.40998
     - 0.38543
     - 0.22334
     - 0.25131
     - 4.015 s
     - 4.600 s
     - CPU
     - 解误差/初值更好，PDE/边界和时间更差
   * - 自适应采样
     - Burgers
     - 0.44410
     - 0.42235
     - 0.23456
     - 0.21065
     - 4.537 s
     - 4.833 s
     - CPU
     - L2/最大误差/PDE 更好，初值明显更差
   * - Adam + L-BFGS
     - 低频 Poisson
     - 8.115e-4
     - 3.744e-5
     - 3.378e-2
     - 1.066e-3
     - 1.041 s
     - 1.277 s
     - CPU
     - 三组中最准，但计算量未严格对齐
   * - 顺序时间窗
     - Burgers
     - 0.32292
     - 0.37165
     - 0.43321
     - 0.36544
     - 4.140 s
     - 4.319 s
     - CPU
     - 仅 PDE 残差更好，解误差和条件误差更差
   * - 迁移学习
     - Burgers
     - 0.45208
     - 0.38798
     - 0.24172
     - 0.23574
     - 2.525 s
     - 2.587 s
     - CPU
     - 全网络多数误差改善，含预训练总计 5.029 s；末层迁移更快但更差
   * - 两网络区域分解
     - 非均质 Poisson
     - 0.24177
     - 3.124e-4
     - 0.00220
     - 0.01127
     - 1.202 s
     - 2.538 s
     - CPU
     - 解/通量显著改善，参数和时间约翻倍
