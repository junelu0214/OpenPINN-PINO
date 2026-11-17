问题理解：从自然语言到问题草稿
==============================

``openpinn.agents.understand_problem`` 是智能体主链的第一段：它把中文或英文的
自然语言问题描述整理成接近 ``ProblemSpec`` 的部分草稿，并列出仍缺少的字段和需要
向用户追问的问题。内置规则后端是确定性的，不依赖远程大模型或 API 密钥，因此也可
直接用于离线程序和 CI。
可在 :doc:`Understanding 网页试用 <understanding_trial>` 中先体验相同报告字段。
试用里的就绪表示当前物理族草稿已经自洽，不替代下游 Python Specification Validation Agent 的最终组合闸，
也不提供代码生成或下载。

主链顺序固定为：

.. code-block:: text

   Understanding → Planning → Specification Validation Agent → Deterministic CodeBuilder
   Repair（路线图）

Problem Understanding 本身不选择网络、采样器、损失或优化器。Planning 只推荐方法草稿，
不生成训练脚本，也不调用 ``solve``
或 ``train_operator``。Repair 仍是路线图能力。

基本用法
--------

.. code-block:: python

   from openpinn.agents import understand_problem

   report = understand_problem(
       """二维线弹性位移问题。矩形 x 从 0 到 2，y 从 0 到 2；
       材料 E=210e9 Pa、nu=0.3；左边固定，右边 x 向牵引 1e6，
       上下边自由。""",
       path="pinn",
   )

   for question in report.questions:
       print(question)

Solver Planning
---------------

``openpinn.agents.plan_method`` 接收完整 ``ProblemSpec``、Understanding 可消费的结构化
草稿或配置器问题侧字典，返回固定字段的 ``PlanReport``：``method_draft``、
``rationale``、``warnings`` 与 ``ready_for_validation``。主路径是 OpenAI-compatible
LLM；确定性规则表只用于无 Key、接口失败、坏 JSON 和 CI fallback。

.. code-block:: python

   from openpinn.agents import plan_method, validate_specs

   plan = plan_method(problem_draft, path="pinn", backend="auto")
   if plan.ready_for_validation:
       gate = validate_specs(complete_problem, plan.method_draft)
       if gate.ok:
           # 只有这里之后，调用方才可以声称能够进入确定性 CodeBuilder。
           pass

``ready_for_validation=True`` 只表示方法草稿能通过 ``MethodSpec`` 与注册表检查，不表示
问题规格或问题—方法组合已经通过 Validation Agent。声称可生码、复制或下载前，调用方必须
对最终组合另外要求 ``validate_specs(problem, method_draft).ok``。修改任一规格后，旧通过
状态立即失效并必须重检。Planning 不生码、不训练，也不调用 Repair。

``backend`` 支持 ``llm``、``rules`` 与默认的 ``auto``。``auto`` 在有 Key 时先请求 LLM；
端点不可达或输出形状不合法会使用 rules fallback，并加入 ``planning.rules_fallback``
警告。显式 ``llm`` 失败会返回 ``ready_for_validation=False``，不会伪装成功。未知物理族
只给保守草稿并标记未就绪；LLM 或 hints 给出的未注册方法、模型、采样器、损失、优化器和
微分器会被 ``MethodSpec`` 拒绝。

Planning 环境变量优先使用 ``OPENPINN_PLAN_BASE_URL``、``OPENPINN_PLAN_MODEL``、
``OPENPINN_PLAN_API_KEY``、``OPENPINN_PLAN_TIMEOUT``；未设置时依次复用对应的
``OPENPINN_UNDERSTAND_*`` 和本地 Ollama 设置。Key 只进入 Bearer 请求头，不得写入仓库、
测试、文档或截图。完整二维线弹性可与当前 ``ProblemSpec`` 一起做 Validation Agent 端到端校验；
热传导与 Poisson 规划草稿能通过 ``MethodSpec``，但在它们进入完整 ``ProblemSpec`` 前，
不能声称已通过问题—方法端到端门禁。

网页中的“推荐方法”复用 Understanding 页的用户 URL、模型与 Key，只显示方法草稿、理由
和警告。浏览器形状预检不能代替 Python Validation Agent，也不会直接把结果写入代码生成器。

``report.draft`` 是可部分填写的 ``ProblemSpec`` 形状字典；``missing`` 和
``questions`` 说明下一轮需要补充什么。可用嵌套字典或点分字段把答复传回：

.. code-block:: python

   report = understand_problem(
       original_text,
       answers={"constitutive.E": 70e9, "constitutive.nu": 0.33},
   )

物理族就绪闸门
--------------

``ready_for_validation`` 在问题侧草稿对**已声明的物理族**自洽、且关键缺口已经写清时
为 ``True``。热传导/扩散、泊松方程和线弹性各自只追问本族需要的量。物理族未知时只追问
是哪一种物理，不会默认成线弹性，也不会把杨氏模量、泊松比或 ``ux`` / ``uy`` 写进
``missing`` / ``questions``。

这个判断由 Understanding 自己的物理族闸门完成，**不调用** Validation Agent
``openpinn.agents.validate``。因此一份字段齐全的一维热传导或二维泊松草稿可以在
Understanding 中就绪，即使它还不是当前 ``ProblemSpec`` 能接受的线弹性对象。
``report.validation`` 在这条路径上保持 ``None``。

Specification Validation Agent（原 Agent A）是 Planning 后的最终问题—方法组合闸，
并且仍然只接受现有线弹性 ``ProblemSpec``。旧 ``validate`` 与
``openpinn.agents.validator`` 导入保持兼容；新代码推荐 ``validate_specs`` 或
``ValidationAgent.validate``。下游代码生成不得绕过它。Understanding 显示就绪，只说明可以
离开问题理解；它不表示线弹性校验已经通过，也不提供生成或下载。

Validation 是确定性规格合规检查，不依赖 LLM。它检查当前实现真正支持的 schema、注册项、
边界、问题—方法兼容性与可选代码静态规则；不保证边值问题适定、训练收敛或物理结论正确。
当前二维弹性 codegen 只实现平面应力；``ProblemSpec`` 尚无平面应变选择器，因此平面应变会
诚实标为不支持。有限板固支加牵引算例也不把 Kirsch 无限板解当作精度门禁。

合理的明显延伸（例如把 ``pai`` 规范成 ``pi``，或在只给出端点 ``x=0`` 与 ``x=1``
时把区间收成 ``[0, 1]``）可以写入草稿，但必须在 ``warnings`` 里用
``understanding.inferred`` 标明。用户没有给出的真实数值仍然要追问，不能编造。

当前规则后端仍只抽取它原来就会抽取的二维线弹性子集。热传导和泊松草稿主要由
LLM 提取；规则路径在物理族没有被写进草稿时只会询问物理族。Understanding 的 Operator
路径仍只整理问题侧字段；方法选择由后续 ``plan_method`` 完成。

.. _local-llm-understanding:

本地开源 LLM 后端
------------------

M27 可选用本机 OpenAI 兼容接口提取更一般的 PDE 问题事实。默认连接
``http://127.0.0.1:11434/v1``，推荐中文和公式表现较好的 Qwen2.5-Instruct
1.5B 或 3B 级模型；模型权重由 Ollama 管理，不进入 OpenPINN 仓库：

.. code-block:: bash

   ollama pull qwen2.5:1.5b-instruct
   ollama serve

Python 调用不受浏览器 CORS 限制：

.. code-block:: python

   from openpinn.agents import understand_problem

   report = understand_problem(
       description,
       backend="llm",       # "rules" | "llm" | "auto"
       base_url="http://127.0.0.1:11434/v1",
       model="qwen2.5:1.5b-instruct",
       timeout=15,
   )

``rules`` 是默认的确定性离线路径；``llm`` 调用本地接口；``auto`` 优先尝试本地
接口，连接失败时退回规则。也可用环境变量 ``OPENPINN_UNDERSTAND_BACKEND``、
``OPENPINN_OLLAMA_BASE_URL``、``OPENPINN_OLLAMA_MODEL`` 和
``OPENPINN_OLLAMA_TIMEOUT`` 设置默认值。接口不可达、超时或 JSON 不合法都会在
``warnings`` 中加入 ``understanding.llm_fallback``，不会静默伪造成功。不合法的
LLM JSON 绝不会标记就绪。

LLM 只返回问题侧 JSON；输出中的代码、训练脚本、网络、优化器和方法规划字段会被拒绝。
应用忽略模型对就绪状态的任何主张，并按上面的物理族闸门重算 ``missing``、
``questions`` 和 ``ready_for_validation``。提示词包含一维热传导与二维泊松两个示例，
允许合理延伸，并要求把推断写进 ``understanding.inferred`` 警告。JSON 形状不合法时，
Python 会带 ``understanding.llm_fallback`` 退回规则，且不会把这次失败标成就绪；
网页试用保持 M28 的 API Key 路径，失败时直接报错，不静默改走规则。

提示词还要求 ``pde.expression`` 使用配置器可解析的 ASCII 语法：导数写作 ``dx``、
``dy``、``dt``，乘法使用显式 ``*``，不输出 ``Δ``、``∇``、下标导数或隐式乘法。
浏览器映射器会为缺少公式的结构化热传导或泊松草稿合成同一语法的残差。

一维热方程或二维 Poisson 因此可以在 Understanding 中就绪。它们在 Specification Validation Agent 的
``ProblemSpec`` 扩展支持之前仍然不能进入 CodeBuilder。

.. _cloud-api-understanding:

用户自有云 API Key
-------------------

同一个 ``backend="llm"`` 也支持用户自己的 OpenAI-compatible 云或国内兼容接口。
传入 ``api_key`` 后，标准库客户端只把它放在 ``Authorization: Bearer`` 请求头，
不会写入 prompt、报告或错误消息：

.. code-block:: python

   import os
   from openpinn.agents import understand_problem

   report = understand_problem(
       description,
       backend="llm",
       base_url=os.environ["OPENPINN_UNDERSTAND_BASE_URL"],
       model=os.environ["OPENPINN_UNDERSTAND_MODEL"],
       api_key=os.environ["OPENPINN_UNDERSTAND_API_KEY"],
   )

除参数外，也可直接使用 ``OPENPINN_UNDERSTAND_BASE_URL``、
``OPENPINN_UNDERSTAND_MODEL``、``OPENPINN_UNDERSTAND_API_KEY`` 和
``OPENPINN_UNDERSTAND_TIMEOUT``。不要把 Key 写入 Python 文件、shell 历史、文档、
截图、测试夹具或 Git。不同兼容服务的 base URL、模型名、配额和数据政策不同，必须以
所选服务商文档为准。

网页直连与 Key 安全
~~~~~~~~~~~~~~~~~~~~

:doc:`Understanding 网页试用 <understanding_trial>` 只有用户 API Key 路径，不显示
规则或自动模式。URL、模型和 Key 只保存在当前浏览器的 ``localStorage``；请求由浏览器
直接发往用户填写的接口，不经过 OpenPINN 服务器。点击“清除 Key”会删除对应存储项并
清空输入。只应在自己信任的设备和浏览器配置文件中使用；共享设备使用后应立即清除。

服务商必须允许文档页面 origin 的 CORS 请求，并接受 OpenAI-compatible
``/chat/completions`` 与 Bearer 认证。浏览器扩展、本机恶意软件和同源脚本仍可能读取
``localStorage``，因此网页保存并不等同于操作系统密钥库。缺少 Key、HTTP 错误、超时、
CORS 或 JSON 校验失败都会显示明确错误、清除旧的就绪显示，且不会静默退回规则并冒充
LLM 成功。页面交付后由用户用自己的真实 Key 自行测试；自动测试只使用 mock Key 和
mock HTTP，不连接任何服务商。

报告中的“填入配置器”使用当前标签页的 ``sessionStorage`` 做一次性传递，只包含问题
草稿和映射结果，不包含 API Key。所有已声明物理族都可进入自定义 PDE/BC 编辑器；未知
物理族仍会追问物理类型。``ready_for_validation`` 只表示当前物理族的问题侧信息自洽，
不等于 Specification Validation Agent 或代码生成校验通过。
