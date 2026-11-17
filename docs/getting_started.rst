安装与快速开始
==============

这一页把环境准备和第一条本地训练链路合并在一起。完整的平台差异、可选依赖和故障说明仍保留在
:doc:`install`；更多可修改的入门代码见 :doc:`quickstart`。

1. 安装 OpenPINN
----------------------------

在仓库根目录创建并激活 Python 环境，然后安装项目、测试和文档依赖：

.. code-block:: console

   python -m venv .venv
   python -m pip install --upgrade pip
   python -m pip install -e ".[test,docs]"

Windows PowerShell 可用 ``.\.venv\Scripts\python.exe`` 代替已激活环境中的 ``python``。
安装后先验证导入：

.. code-block:: console

   python -c "import openpinn; print(openpinn.__version__)"

2. 打开 Agent 工作台
-------------------------------

打开 :doc:`method_configurator`。可以从自然语言描述开始，让五个 Agent 依次完成问题理解、
求解规划、规范验证、代码构建和训练后诊断；已有配置时也可直接编辑结构化问题和方法设置。

3. 生成并运行第一份脚本
--------------------------------

#. 选择一维 Poisson 或另一个已有模板，先使用较短训练预算检查链路。
#. 下载 ``.py``，保存到项目 ``examples/`` 目录，避免覆盖仓库已有同名示例。
#. 从仓库根目录运行实际文件名，例如：

   .. code-block:: console

      python examples/first_poisson_run.py

#. 查看终端打印的结果目录，确认 ``metrics.json``、训练历史和预测产物已经生成。
#. 将 ``run_artifact.json`` 或 ZIP 上传回工作台的诊断评估 Agent。

4. 下一步
---------

* 按物理问题选择完整代码与结果：:doc:`code_tutorials`。
* 学习几何、PDE、边界和方法设置：:doc:`configurator_tutorial`。
* 了解模型、采样、损失、优化和微分：:doc:`method_ecosystem`。
* 让 Codex、Claude Code 等通用智能体直接调用：:doc:`agent_harness`。

.. warning::

   浏览器只负责建模、验证、代码导出和产物诊断，不执行 Python 训练，也不会自动读取本地数据路径。

.. toctree::
   :hidden:

   install
   quickstart
   agent_harness
