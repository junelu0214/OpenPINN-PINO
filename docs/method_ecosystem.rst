方法生态：五个可组合的数值方法轴
================================

本参考与 OpenPINN/PINO 预印本 4.4 节一致，将方法分成模型、采样、损失、优化和微分
五个相互独立的轴。每个方法族页面均给出作用、真实调用边界、最小配置、受控比较方法、
限制和主要参考文献。网页端可在 :doc:`method_configurator` 中组合组件并下载完整脚本。

.. list-table::
   :header-rows: 1

   * - 方法轴
     - 回答的问题
     - 推荐入门比较
   * - 模型
     - 用什么函数表示场或算子？
     - 同一 Poisson/Burgers 设置下比较表示能力、参数量和用时
   * - 采样
     - 在哪里评价物理、边界和数据残差？
     - 固定总点数，比较 uniform 与自适应点分布
   * - 损失
     - 怎样组装和平衡不同训练目标？
     - 同时报出 PDE、边界、初值与参考误差
   * - 优化
     - 怎样更新参数并分配训练预算？
     - 比较 Adam 与 Adam→L-BFGS，并记录墙钟时间
   * - 微分
     - 怎样形成 PDE 中的导数？
     - 先验证解析导数，再比较周期 Poisson 的 FD 与谱微分

预印本中的 :doc:`Burgers 单轴比较 <tutorials/burgers_method_comparison>` 和
:doc:`周期 Poisson 算子比较 <tutorials/periodic_poisson_operator>` 提供问题导向的示例。
它们是特定设置下的证据，不是跨问题排行榜。

.. toctree::
   :maxdepth: 2

   method_ecosystem/model/index
   method_ecosystem/sampler/index
   method_ecosystem/loss/index
   method_ecosystem/optimizer/index
   method_ecosystem/derivative/index

.. toctree::
   :hidden:

   method_catalog
