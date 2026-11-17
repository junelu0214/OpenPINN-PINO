稳态渗流与瞬态固结参数识别
============================

本项目用两个不同的物理问题演示两类反分析，而不是让 PINN 与算子重复求解同一算例。
所有量均按一致尺度无量纲化；接口不会猜测工程单位换算。

.. _darcy-parameter-identification:

稳态 Darcy 渗流的渗透率识别
----------------------------

PINN 使用一维稳态 Darcy 方程：

.. math::

   -\frac{\mathrm d}{\mathrm dx}\left(k\frac{\mathrm dp}{\mathrm dx}\right)=q,
   \qquad p(0)=p(1)=0.

``inverse_darcy_pinn`` 由稀疏 ``(x,p)`` 观测联合训练压力场与正且有界的渗透率
``k``。压力网络使用硬边界 ``p=x(1-x)N(x)``，适合没有问题族预训练、只识别一个
渗流工况的情形。

.. _consolidation-parameter-identification:

Terzaghi 固结问题族的固结系数识别
---------------------------------

DeepONet 使用一维 Terzaghi 固结模型：

.. math::

   \frac{\partial u}{\partial t}-c_v\frac{\partial^2u}{\partial z^2}=0,
   \qquad u(0,t)=u(1,t)=0.

``train_consolidation_family_operator`` 学习连续问题族 ``(cv,U0) -> u(z,t)``。
随后 ``inverse_consolidation_operator`` 冻结算子权重，由新成员的 ``(z,t,u)`` 观测
只优化有界固结系数 ``cv``。训练好的算子可重复服务同一参数范围内的多个正、反分析任务。

网页操作
--------

* 在“流体力学”中选择“渗流反分析（PINN）”，设置 ``k`` 范围与 ``q``，使用
  ``examples/references/darcy_inverse_observations.csv`` 生成单实例 PINN 脚本。
* 在“多物理场”中选择“固结问题族反分析（DeepONet）”，设置 ``cv`` 范围与 ``U0``，
  Model 选择 DeepONet，并使用
  ``examples/references/consolidation_inverse_observations.csv`` 生成问题族训练及反演脚本。

也可以直接运行：

.. code-block:: console

   python examples/inverse_seepage_consolidation.py

当前限制
--------

* 渗流算例识别空间常数渗透率；固结算例识别空间常数 ``cv``。
* 固结问题族当前接入 DeepONet；选择 FNO 会明确提示尚未接入，不会静默替换模型。
* 损失下降不构成参数唯一性证明；应使用与反演观测分开的参考数据评估结果。
