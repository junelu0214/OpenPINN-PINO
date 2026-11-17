OpenPINN Workflow Harness：让通用智能体负责求解
=================================================

OpenPINN Workflow Harness 不维护“已支持问题”列表，也不使用 ProblemSpec、MethodSpec 或方法
注册表限制通用智能体。Codex、Claude Code、Cursor 或自建 Agent 负责理解问题、设计方法、
在 OpenPINN 项目中编写代码并诊断结果；Harness 只提供 OpenPINN 启发的工作流程、受限运行和
可追溯证据。Harness 本身没有 LLM。

它会执行 Agent 在任务目录中编写的 Python，因此应只在受信任的工作区使用。Harness 使用独立
子进程、禁止 Shell 入口并限制墙钟时间与输出路径，但它不是操作系统安全沙箱，子进程仍继承
当前 Python 环境和当前用户权限。

安装与命令
----------

.. code-block:: console

   python -m pip install -e .
   openpinn harness workflow
   openpinn harness init openpinn_tasks/my_problem
   openpinn harness check openpinn_tasks/my_problem
   openpinn harness run openpinn_tasks/my_problem
   openpinn harness inspect openpinn_tasks/my_problem

也可使用独立入口 ``openpinn-harness``。命令的标准输出是一个 JSON 对象，Agent 程序的输出
进入本次运行目录的 ``harness.log``。

单一 Agent 工作区
-----------------

``init`` 不选择问题类型或计算方法，只创建：

.. code-block:: text

   openpinn_tasks/my_problem/
   ├── task.yaml
   ├── problem.md
   ├── plan.md
   ├── solution/
   │   └── run.py
   ├── tests/
   └── references/

通用 Agent 在 ``problem.md`` 中整理未知场、方程、几何、初边值条件、单位、参数、目标输出与
待确认事项；在 ``plan.md`` 中说明表示模型、采样、损失、优化、微分、无量纲化、smoke、参考
证据和指标；然后自由实现 ``solution/run.py``。它可以使用现有 OpenPINN API、创建新的
``PDEProblem``、在 ``user_projects`` 注册扩展、使用 Operator API、普通 PyTorch 或任务所需的
其他本地代码。Harness 不判断这种实现是否属于内置能力。

运行协议
--------

Harness 不用 Shell，而是以当前 Python 解释器直接运行入口文件，并提供：

``OPENPINN_TASK_DIR``
   任务目录绝对路径。
``OPENPINN_OUTPUT_DIR``
   本次全新输出目录。
``OPENPINN_DEVICE``
   ``task.yaml`` 声明的设备。
``OPENPINN_MAX_WALL_SECONDS``
   Harness 强制的墙钟上限。
``OPENPINN_RUN_ID``
   本次运行 ID。
``OPENPINN_BUDGET_JSON``
   Agent 自定义的 smoke/训练预算 JSON。

入口应在输出目录写 ``run_result.json``：

.. code-block:: json

   {
     "schema_version": 1,
     "status": "completed",
     "summary": "bounded run completed",
     "metrics": {"relative_l2": 0.03},
     "claims": {
       "convergence_verified": false,
       "accuracy_verified": false
     },
     "artifacts": ["loss_history.json"]
   }

允许的 Agent 状态为 ``completed``、``budget_exhausted``、``not_converged``、``failed``、
``interrupted`` 和 ``unknown``。指标必须是有限 JSON 数值；产物路径必须留在输出目录。

检查与证据边界
--------------

``check`` 只阻断结构和执行错误，例如无入口、Python 语法错误、路径越界或非法预算。缺少问题
说明、计划或独立验证只产生 advisory，不会因为方程、几何、多场、算子或方法未知而阻断。

每次运行生成 ``harness_artifact.json``，把两类信息分开：

* ``agent_result`` / ``agent_claims``：通用 Agent 代码报告的科学结果和结论；
* ``harness_observations``：Harness 实际观察到的退出码、超时、耗时、有限值、输出大小和文件存在性。

Harness 记录但不证明 Agent 的物理结论。脚本成功、loss 下降或有限输出都不自动等于收敛或
精度验证；通用 Agent 应结合独立参考、制造解、守恒量或用户声明的阈值给出结论。

放入通用智能体
--------------

仓库提供 ``integrations/agent_harness/SKILL.md`` 和 ``AGENTS.template.md``：

* **Codex/支持 Skills 的工具**：复制或链接整个 ``integrations/agent_harness`` 到 skills
  目录，或者把模板并入项目 ``AGENTS.md``。
* **Claude Code**：把模板并入 ``CLAUDE.md``，并提供 ``SKILL.md`` 作为完整流程说明。
* **Cursor、Continue 或自建 Agent**：把模板设为项目规则，并允许执行上述五个 CLI 命令。

可运行的未注册问题示例位于 ``examples/harness/agent_workspace``。它由 Agent 代码自行实现
指数衰减 ODE 的 PINN，Harness 没有为该问题注册适配器。
