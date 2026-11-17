OpenPINN Agent 建模与代码生成工作台
=====================================

OpenPINN 是面向物理信息神经网络与神经算子的开源建模平台。五类 Agent 共享
``ProblemSpec`` 与 ``MethodSpec``，从问题描述出发，协同完成问题理解、求解规划、
规范验证、代码构建和训练结果诊断；数值训练始终在用户本地运行。

.. raw:: html

   <section class="agent-home-hero" aria-labelledby="agent-home-title">
     <p class="agent-home-kicker">AGENT-ASSISTED SCIENTIFIC COMPUTING</p>
     <h2 id="agent-home-title">描述物理问题，得到可检查、可运行、可诊断的建模方案</h2>
     <p>五个 Agent 不是单独的问答入口，而是一条可回溯的科学计算工作流。你可以从自然语言开始，也可以带着已有 JSON/YAML 配置直接进入规范验证。</p>
     <p class="agent-home-actions"><a class="agent-home-primary" href="#agent-pipeline-title">开始 Agent 建模</a><a href="code_tutorials.html">浏览问题算例</a><a href="method_ecosystem.html">比较方法生态</a></p>
   </section>
   <ol class="agent-home-stages" aria-label="五 Agent 工作流">
     <li><strong>1 · 问题理解 Agent</strong><span>提取几何、物理、未知场、参数和初边值条件。</span></li>
     <li><strong>2 · 求解规划 Agent</strong><span>规划模型、采样、损失、优化与微分方式。</span></li>
     <li><strong>3 · 规范验证 Agent</strong><span>检查完整性、兼容性、冲突、单位与实现边界。</span></li>
     <li><strong>4 · 代码构建 Agent</strong><span>确定性组装并导出本地训练与测试代码。</span></li>
     <li><strong>5 · 诊断评估 Agent</strong><span>解析 RunArtifact，给出证据、建议与下一轮入口。</span></li>
   </ol>
   <p class="agent-home-runtime"><strong>本地训练</strong>位于代码构建与诊断评估之间；浏览器不执行 Python，也不上传训练数据。</p>

.. openpinn-method-configurator::

从算例或文档开始
------------------

* :doc:`问题导向的 Tutorial <code_tutorials>`：按物理问题查找代码、说明、结果和复现状态。
* :doc:`方法生态 <method_ecosystem>`：比较模型、采样、损失、优化和微分五个可组合轴。
* :doc:`工作台完整说明 <method_configurator>`：查看导入格式、几何能力、运行方式和限制。
* :doc:`安装与快速开始 <getting_started>`：准备本地环境并运行第一份下载脚本。

.. note::

   短步数 smoke 只检查调用链路，不是精度结论。正式比较应固定问题、网络、训练点、
   步数和随机种子，每次只改变一个方法组件，并保留负结果。

.. toctree::
   :maxdepth: 1
   :caption: 开始

   method_configurator
   getting_started

.. toctree::
   :maxdepth: 1
   :caption: Tutorial 算例

   code_tutorials
   configurator_tutorial

.. toctree::
   :maxdepth: 1
   :caption: 方法生态

   method_ecosystem
   methods

.. toctree::
   :maxdepth: 1
   :caption: 使用指南与参考

   user_guide
   api
   quality_reference

.. toctree::
   :maxdepth: 1
   :caption: 开发者

   developer_guide
   architecture

.. toctree::
   :hidden:

   understanding_trial
