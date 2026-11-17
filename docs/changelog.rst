变更记录
========

2026-11-17：开放式通用智能体 Workflow Harness
----------------------------------------------

* ``openpinn harness`` / ``openpinn-harness`` 不再以内置问题、ProblemSpec、MethodSpec、方法
  注册表或后端列表限制任务；通用智能体可在单一 ``agent_workspace`` 中自由编写 OpenPINN、
  Operator、PyTorch 或任务局部扩展代码。
* Harness 只管理 ``workflow → init → check → run → inspect`` 流程。结构/语法/路径/预算错误会
  阻断，缺少问题说明、计划或参考验证只给 advisory；Python 入口以无 Shell 的限时子进程执行。
* 新的 ``OpenPINNAgentRun`` 将 ``agent_claims`` 与 ``harness_observations`` 分轨记录，并保存
  任务/源码哈希、退出码、超时、有限值、输出量和文件存在性。新增未注册 ODE 的 Agent 代码示例。

2026-11-12：Hybrid 语义证据路径兼容
------------------------------------

* Hybrid Validation 的受限路径解析同时接受 ``problem.equations[0]`` 点号/方括号语法和
  ``/problem/equations/0`` JSON Pointer 语法；两者仍必须解析到所提交 Problem、Method 或
  deterministic summary 内的真实节点。
* 真实模型返回 JSON Pointer 形式的 ``assumptions_noted.path`` 时不再误报
  ``schema.semantic``。Case 1 回归覆盖 Hybrid 完成、CodeBuilder 可运行脚本、本地非 smoke
  训练与 Diagnostic 工件校验。

2026-11-12：Robin 目标表达式优先级修复
---------------------------------------

* 当 Understanding 同时返回完整 Robin 等式 ``expression`` 和右端
  ``value_expression`` 时，规范化、能力匹配与 Python 编译统一优先采用右端目标。完整等式继续
  保留在 ``source_draft``，不会再因其中的 ``=`` 被误当作受限目标表达式而阻断 CodeBuilder。
* 回归测试使用生产验收中的原始形态：``alpha*w+beta*dx(w)=value`` 与
  ``value_expression=-pi`` 同时存在，并覆盖可运行脚本和本地非 smoke 训练。

2026-11-11：通用标量表达式边界闭环修复
-----------------------------------------

* Understanding 就绪判定接受边界的数值或坐标表达式，点路径补答会保留 ``boundaries`` 数组；
  ``pde.residuals`` 会提升为多条规范方程。
* Generic Validation 将 ``value_expression`` 规范为可编译边界目标，并移除 LLM 误放到参数表、
  且未被 PDE 使用的字符串 ``value`` 占位项。其余非数值参数继续 fail closed。
* Python 表达式编译器直接接受 ``value_expression`` 别名。Robin ``-pi`` 以及二维 ``x``、``y``、
  ``y+2`` 目标已覆盖生成脚本、本地非 smoke 训练和 Diagnostic 工件校验。
* Diagnostic 的“复用 Validation 设置”现在复用 Validation 实际使用的设置，包括 Validation
  继续复用 Planning 的链式情况；Understanding 传输错误不再误报为 Planning 失败。

2026-11-11：通用单标量强形式 PINN 后端
-----------------------------------------

* 执行能力从两张冻结卡扩展到受限合同内的单标量强形式问题：一维或二维矩形空间、可选时间轴、
  一条或多条安全编译残差，以及 Dirichlet、坐标导数型 Neumann 和 Robin 条件。字段名不再
  固定为 ``u``；所有路径仍复用 ``run_pinn``，没有新增训练循环。
* Python 与网页注册表按坐标、边界位置、几何、条件类型和受限表达式 fail closed。CodeBuilder
  对匹配问题继续生成同一个 ``openpinn_train.py``；通用问题没有参考解时只记录 loss，不生成
  或猜测误差与精度阈值。
* 新增二维多残差加 Robin 的真实训练测试，以及类 Burgers 单标量时变问题的网页生码、本地
  非 smoke 训练和 Diagnostic 闭环。向量场、多场耦合、复杂几何、周期边界和任意 PDE 仍不支持。

2026-11-11：Stage G 网页到本地训练闭环验收
---------------------------------------------

* 为冻结的一维 Poisson 与非定常 heat 卡增加产品级闭环回归：CodeBuilder 导出真实
  ``openpinn_train.py``，以显式 CPU 短预算进行非 smoke 训练，再把同次 JSON 和 ZIP
  ``RunArtifact`` 交给 Diagnostic。实测误差被持久化，但不作为精度通过门槛。
* 网页现在直接显示 ``execution_capability`` 状态与 backend id。``supported`` 会明确提示可进入
  CodeBuilder 生成本地脚本且网页不训练；``unsupported`` / ``incomplete`` 继续保留不可运行骨架。
* Understanding 页面加入两张可匹配提示词、真实脚本参数、虚拟环境、JSON/ZIP 上传、状态解释和
  常见失败提示。线弹性仍是保留缺口；本阶段没有 Repair、任意 PDE 后端或官方精度阈值。
* 本地与 CI 验收不代表官方站点 QA。Stage G 合入并部署后仍需由 operator 在受邀页面另行完成
  浏览器验收；本次变更不部署，也不声明线上页面已经包含 Stage G。

2026-11-10：Stage D–F 本地训练与诊断闭环
-------------------------------------------

* 新增结构匹配的 ``poisson_1d`` 与 ``heat_1d_unsteady`` adapter。两者编译 Stage B
  表达式并调用同一个 Stage C ``run_pinn``；adapter 内没有第二套训练循环。冻结参考网格分别为
  101 点与 51×51 点，报告实测 ``max_abs`` 和 ``rel_l2``，但没有虚构精度阈值。
* 网页 Validation 增加独立执行能力轨。映射完整、确定性与语义门禁通过且结构命中注册表时，
  CodeBuilder 生成 ``openpinn_train.py``，支持显式步数、墙钟、设备与一真实步 ``--smoke``；
  不支持或不完整的结构继续生成 ``backend_runnable=false`` 骨架。
* 本地脚本把同次运行的 engine 状态、预算、设备、loss、参考误差、失败信息及 pair fingerprint
  写入 ``RunArtifact``。Diagnostic 接受完整状态集合，拒绝 stale、fixture/example、smoke-only
  和缺少同次运行来源的工件。``completed`` 仍只表示停止条件，不表示达到精度。
* 用户路径是“下载脚本 → 本地运行 → 上传 ``RunArtifact`` 到 Diagnostic”。本阶段不包含
  Stage G、线弹性训练、任意 PDE 后端或官方精度阈值。

2026-11-10：Stage C 共用 PINN 引擎
------------------------------------

* 新增 ``run_pinn`` 共用强形式入口，复用现有 ``Trainer``、采样、loss composer、优化器和
  checkpoint。每次运行必须显式给出步数与墙钟预算，默认设备为 CPU，并写入 history、
  checkpoint、manifest 和独立的 ``PINNEngineRun`` JSON。
* 状态限定为 ``budget_exhausted``、``interrupted``、``failed``、``not_converged`` 与
  ``completed``。只有调用方给出的停止条件在预算内触发时才写 ``completed``；有限 loss 或
  用完步数不会被描述为训练成功。取消、KeyboardInterrupt、非有限 loss 和异常均保留明确结果。
* Stage B 编译残差可通过通用 ``PDEProblem`` hook 接入同一引擎。本阶段未加入 Poisson/heat
  adapter、网页可运行后端、产品 RunArtifact 或 Diagnostic/Validation 执行门禁；CodeBuilder
  继续保持 ``backend_runnable=false`` 与 ``NotImplementedError``。

2026-11-10：Stage B 安全表达式与自动微分
-----------------------------------------

* 新增受限表达式编译器，把 Stage A ``mapping_complete`` 问题中的字段、坐标与参数绑定为
  PyTorch 张量运算，并复用 ``openpinn.autodiff.grad`` 计算声明坐标上的一阶与嵌套导数。
  首期白名单为四则运算、括号、数值、``pi``、``sin``、``cos``、``exp`` 和
  ``d<coordinate>(...)``；未绑定名称、未声明坐标、其他函数及不安全语法均 fail closed。
* 可编译具名 PDE、Dirichlet/Neumann 边界和初值残差，输出保持 ``(N, 1)``，供后续共享
  训练引擎接入。网页受限检查器同步使用同一首期白名单。
* 本阶段只验证表达式绑定与自动微分。解析残差接近零不代表完成训练或物理精度验证；
  CodeBuilder 仍保持 ``backend_runnable=false`` 和 ``NotImplementedError``。训练引擎及
  Poisson/heat adapter 留待后续阶段。

2026-11-10：Stage A 最小可训问题合同
--------------------------------------

* 在现有 ``GenericProblemDescription`` 中把空间/时间 bounds 与 units、未知场、参数、PDE、
  BC、IC 和 geometry 定义为后续训练路径需要的一等字段，并新增独立的
  ``trainable_contract.mapping_complete`` 信号。
* 修复网页实测的一维非定常热传导映射：``t∈[0,1] s`` 不再只落入 opaque；方程使用
  ``alpha`` 时，等值 unit-less ``diffusivity`` 会显式归一到 ``alpha=0.01 m^2/s`` 并记录原因。
  冲突、缺单位、缺 BC/IC 或未绑定符号均 fail closed 为 mapping incomplete。
* 本阶段只交付合同与确定性映射。CodeBuilder 仍输出 ``backend_runnable=false`` 骨架和
  ``NotImplementedError``；未加入训练循环、physics adapter 或可运行入口。

2026-11-10：M38 Diagnostic Evaluation Agent
---------------------------------------------

* CodeBuilder 通用骨架现在运行即写 ``openpinn_run/run_artifact.json``，记录 pair
  fingerprint、后端能力、运行状态、loss/error 槽和 failure stack 槽；当前非运行后端诚实
  标记为 ``not_runnable``。
* 网页新增第五段 Diagnostic，支持本地 JSON/ZIP RunArtifact，分别展示确定性 metrics/schema
  findings 与真实可配置 LLM 诊断，并输出 ``DiagnosticReport`` 和 ``human_summary``。
* stale、无效 schema 或已完成却缺 loss 的 artifact 不会获得 healthy 状态。API 失败保持
  ``incomplete``，Key 不进入报告；所有 ``repair_hints`` 都是非绑定建议，不改写代码或规格。

2026-11-09：M37 Code Building Agent
------------------------------------

* 在网页 U→P→V 链路后新增确定性 Code Building Agent。它把当前已验证的通用
  Problem+Method 生成 Python 装配骨架和 ``BuildReport``，不依赖旧线弹性 adapter，
  也不使用 LLM 生码或解释代码。
* 确定性 fail、缺少模板字段、旧 pair fingerprint 或语义 ``incomplete`` 会阻断且不输出
  源码；语义 ``concerns`` 允许生成，但必须保留在报告和骨架中。
* 产物始终声明 ``backend_runnable=false`` 并保留未实现的后端绑定入口。骨架成功不代表
  可训练，也不会启动训练；任意物理可运行后端留待后续明确实现和验证。

2026-11-09：M36 Hybrid Validation Agent
------------------------------------------

* 在 M35 L0–L6 确定性结构报告旁新增真实 OpenAI-compatible LLM 语义审查，
  状态限定为 ``no_concern_identified`` / ``concerns`` /
  ``insufficient_information``，证据必须来自当前输入 path/excerpt。API 失败为 incomplete，
  不会用规则冒充 LLM 成功。
* 报告绑定完整 Problem/Method 快照、pair fingerprint、request id 和语义版本。
  编辑任一规格立即标记 stale；配置器可把当前编辑快照发回 Hybrid Validation。
  LLM 不能修改规格、覆盖确定性 fail/M34 或解锁训练导出。
* 修复 heat 扩散率/``alpha`` 参数在 Generic 规范化中丢失的症状，并在导出锁定时
  改用“代码预览”文案，不再声称脚本已可复制/下载。Validation Key 仅存当前浏览器，
  不写入报告或非训练导出。

2026-11-09：M35 通用网页描述与分层 Validation
--------------------------------------------------------

* 新增 ``GenericProblemDescription`` ``schema_version=1`` 与 L0–L6
  ``LayeredValidationReport``，允许自由物理标签、多场、耦合和 opaque 数学进入网页检查。
* Planning 配置器交接现在联合消费 ``problem_draft`` 与 ``method_draft``，
  缺少问题草稿时 fail closed，不再静默使用默认问题。
* 通用规格/方法/报告可作为标注清楚的非训练 JSON 导出。网页无同源 Python
  目标闸，因此训练代码保持锁定；Executable adapter 只能无损成功或显式拒绝，
  成功后仍必须走 M34 Python Validation。受限表达式检查不使用 ``eval``。

2026-11-09：M34 Specification Validation Agent
-----------------------------------------------

* 将原 Agent A 产品化为论文 §4.3 的 Specification Validation Agent，新增
  ``validate_specs`` 与 ``ValidationAgent`` 稳定入口，同时保留旧 ``validate`` 和
  ``openpinn.agents.validator`` 导入兼容。
* ``from_specs`` 默认对最终 ``ProblemSpec`` + ``MethodSpec`` 组合执行确定性 Validation；
  缺字段、不完整边界、未知注册项和不兼容微分器会阻断 codegen。修改任一规格会使已通过
  快照失效，验证与生码使用同一规范化 payload。
* 网页明确区分浏览器合同预检和完整 Python Validation。Planning 导入在最终组合通过
  Python Validation 前继续锁定复制与下载；带孔板导出脚本在运行 codegen 前再次校验。

2026-11-04：M33 Solver Planning 网页闭环
----------------------------------------

* Understanding 就绪后可使用独立的 Planning 接口、模型和浏览器本地 Key 请求方法草稿；
  浏览器会归一化常见别名和缺省项，并把 HTTP、网络/CORS、JSON 与结构错误明确显示且脱敏。
* 方法草稿可交接到配置器的 Method 侧，填入可表示的方法、变体、路径和训练步数，并列出
  未映射字段。Planning 不会伪造 Agent A 通过；复制和下载在 Python Agent A 校验前保持禁用。
* 清除 Key 或 Planning 失败会保留已经就绪的 Understanding 草稿，并新增构建后 DOM 回归与
  真实 OpenAI-compatible API 链路检验。

2026-11-04：M32 LLM Solver Planning 薄切片
------------------------------------------

* 新增 ``openpinn.agents.plan_method`` 与固定 ``PlanReport``，以
  OpenAI-compatible LLM 为主路径推荐注册表合法的 ``MethodSpec`` 草稿和简短中文理由。
  ``auto`` 在无 Key、接口失败或 JSON 不合法时带明确警告退回确定性 rules；rules 只用于
  fallback 与 CI。代码生成前仍须现有 Agent A ``validate(problem, method)`` 通过。
* Understanding 网页试用新增只读“推荐方法”，复用浏览器中的用户 API 设置并严格检查
  返回形状，不生成脚本、不训练，也不直接解锁复制或下载。

2026-11-02：M29 理解按物理族就绪
--------------------------------

* Understanding 的缺失字段、追问和 ``ready_for_validation`` 改为按已声明的物理族
  判断。热传导、扩散和泊松草稿不再被线弹性专用的 Agent A / ``validateDraft``
  写入 E、ν、``ux``、``uy``。物理族未知时只询问是哪一种物理，不默认线弹性。
* 提示词允许合理延伸，并用 ``understanding.inferred`` 警告标明推断，同时加入一维
  热传导和二维泊松两个示例。JSON 形状校验保持不变。Agent A 仍用于配置器、YAML 和
  CodeBuilder；本里程碑不改训练 API，也不加入浏览器训练。

2026-11-02：M28 用户云 API Key 网页理解
---------------------------------------

* Understanding 网页试用改为用户自有 OpenAI-compatible API Key 单一路径；URL、
  模型与 Key 只保存在浏览器 ``localStorage``，可明确清除，并由浏览器直连服务商。
  网页移除规则/自动模式和静默规则回退，缺少 Key、CORS/网络、超时或 JSON 错误均明确
  失败并清除旧就绪状态。Python 规则 CI 路径保留，兼容客户端新增可选 Bearer 认证。

2026-11-02：M27 本地开源 LLM 问题理解
--------------------------------------

* ``understand_problem`` 新增 ``rules``、``llm`` 和 ``auto`` 后端，以标准库客户端
  连接本机 OpenAI 兼容 Ollama 接口，并在不合法 JSON、超时或不可达时带警告退回规则。
  网页试用同步增加本地接口地址与模型控件。LLM 不决定就绪，现有 Agent A 门禁不变；
  未加入模型权重、Planning、Repair、训练 API 改动或浏览器训练。

2026-11-02：M26 Understanding 网页试用
--------------------------------------

* “开始”导航新增中英文可发现的 Understanding 静态试用页，以无需 API 密钥的
  JavaScript 确定性规则显示 M25 的草稿、缺失字段、追问、警告和就绪状态。试用页不
  提供填入配置器、生成、下载或浏览器训练；任何后续代码生成仍必须通过 Python Agent A
  ``validate`` 门禁。

2026-11-02：M25 问题理解薄片
----------------------------

* 新增 ``openpinn.agents.understand_problem`` 与 ``UnderstandReport``，以无需
  API 密钥的确定性规则从中英文自然语言形成部分 ``ProblemSpec`` 草稿、缺失字段和
  追问。只有草稿实际通过现有 Agent A 问题侧门禁后才标记就绪；本里程碑不含 Planning、
  Repair、训练脚本生成或训练 API 改动。

2026-10-30：M24 高频 Poisson 三条短训数字刷新
---------------------------------------------

* Fourier / Modified MLP / 硬边界方法页与十种方法摘要对应行，按 CPU 单随机种
  5000 步配对短训加长结果刷新数字与对照图；摘要仍标为短训参考。Modified MLP
  保留负结果。未改 ``configs/reference_runs`` 产品默认步数或训练 API。

2026-10-30：M23 十种方法摘要短训说明
-------------------------------------

* 十种方法结果摘要现明确标为 CPU 单随机种的短训参考，而非充分收敛的精度排名；相对 L2
  接近或高于 1 表示该次短训尚未达到实用精度，加长训练数字待后续算例训练交付再更新。

2026-10-30：M22 十种方法摘要基线
---------------------------------

* 十种方法的结果摘要现按各方法页的单随机种参考实验并列展示基线与改进组的相对 L2、
  PDE 残差和用时，并明确将原“解误差”标为“相对 L2”。

2026-10-29：M19 教程演示结果
----------------------------

* 代码教程三条主线提交演示图和指标：Burgers 相对 L2 约 0.041（3000 步）、
  PINN 2D Poisson 约 0.0337（5000 步）、FNO × physics_only 约 0.039（10000 步，
  维护者接受的展示质量；自动阈值仍为未通过）。
* 带孔板不再作为展示主线。notebook 仍保留 CPU smoke；文档构建不重新训练。

2026-10-29：M20 教程侧栏与双语文档
-----------------------------------

* 教程侧栏只保留“配置器教程”和“代码教程”。代码教程主线为 Burgers、PINN 2D Poisson
  （``tutorials/05_pinn_poisson_2d``）和 FNO × physics_only；带孔板 notebook 仍保留，
  但不进入侧栏。旧教程页面和补充 notebooks 仍作为不进入侧栏的独立页面构建，原 URL 保持可用。
* 文档构建同时输出默认中文站点与 ``en/`` 英文站点，每页提供语言切换。配置器的交互
  控件仍以中文为主；Operator 路径的标题、说明、校验与状态提示现已完成中文化。剩余本地化
  工作缩小为参数 schema 标签与 ``hook.summary`` 等注册表元数据。

2026-10-29：M18 教程收敛
------------------------

* 教程入口统一为“配置器教程”与“代码教程”；前者覆盖 PINN/Operator 的问题设计、
  CSV 角色和下载脚本，后者突出 Burgers、带孔板和二维 Poisson 三条主线。
* 主 notebooks 增加物理设置、训练损失与预测图；其余 M17 notebooks 保留在更多示例。

2026-10-29：M17 代码教程基础
----------------------------

* 增加入门、PINN、算子与扩展四段已执行 notebooks，并在 Sphinx 教程组中渲染。
* 已提交的输出是文档站的来源；Sphinx CI 设置为 ``nbsphinx_execute = "never"``，
  只验证 notebook 结构和输出，不在构建时重新训练。

2026-10-29：M16 文档导航
------------------------

* 站点侧栏整理为“开始 / 教程 / 用户指南 / 参考 / 开发者”五组，配置器移到开始组，
  导航默认折叠。旧的功能概览与计算成本页保留为不进入侧栏的历史页面。

2026-10-28：M15 算子观测 CSV 与训后参考 CSV
--------------------------------------------

* 算子路径的训练观测 CSV 与训后参考 CSV 分成两套入口，启用开关、路径和列映射互不共享。
* ``data_only`` / ``hybrid`` 必须填写观测，否则不能复制或下载；``physics_only`` 隐藏观测入口。
* 训后参考在三种训练模式下都可选。生成脚本分别写出 ``OBSERVATION_*`` 与 ``REFERENCE_*``，
  不会把只填写的一侧静默用作另一侧。PINN 仍只做训后参考评估。

2026-10-28：M13 对外口一致性
----------------------------

* 门控 Cloudflare Pages 演示已使用固定网址；未认证访问显示 HTTP 401 邀请页。
  邀请口令只向运营者线下获取，不写入仓库、公开 Issue 或文档。通过口令进入、
  下载 ``.py`` 和退出登录仍由持有口令的运营者按仓库内
  ``docs/demo_site.md`` 人工验收一次。
* 当前算子入口统一为 DeepONet/FNO ``model`` × ``training_mode``；PINO 只保留
  已弃用的 FNO 路由兼容层和高级网格训练辅助接口，不是第三个配置器模型。
* 算子路径的控制方程设计保持 custom-only；Poisson、Burgers、隧道和圆孔板留在教程。

2026-10-28：M12 CoordinateOperator 与自然求导路径
--------------------------------------------------

* ``CoordinateOperator`` 成为坐标算子的标准协议名；``CoordinateModel`` 仅作为
  一个小版本内的已弃用别名。
* DeepONet 的 ``physics_only`` / ``hybrid`` 在查询坐标上用自动微分形成残差；
  FNO 继续使用网格导数和 ``GridPDEProblem``。
* 移除 DeepONet 到网格的跨适配器，不把插值/重采样包装宣传为产品能力。

2026-10-28：M11 算子模型 × 训练模式配置器
------------------------------------------

* 配置器算子入口拆分为 DeepONet/FNO 模型与 ``data_only``、``physics_only``、
  ``hybrid`` 训练模式；复制与下载保留用户选择，不静默换模型或模式。
* PINO 不再是配置器模型或 preset。旧 ``build_operator_config("pino")`` /
  ``train_operator("pino")`` 在一个小版本内发出弃用警告并路由到 FNO 加训练模式。
* 算子控制方程设计只接受自定义 2D PDE/边界；内置案例从教程入口发现。

2026-10-03：COMSOL/CAD 二维 DXF 导入
---------------------------------------

* “二维几何”把原图片/坐标文件导入标为“识别图片”，COMSOL/CAD 统一使用
  一个 DXF 导入选项，读取二维 ASCII DXF 闭合轮廓。
* 支持闭合多段线、圆、完整椭圆和多段线圆弧 bulge；导入后可预览、编辑、
  逐边配置条件并生成 ``op.Polygon`` 代码。开放/三维/分离区域给出明确错误。
* 添加带孔矩形 DXF 示例及浏览器解析测试。

2026-09-30：可旋转椭圆几何
--------------------------------

* 核心新增 ``op.Ellipse``，支持圆心、半轴 ``a/b`` 和逆时针旋转角。
* 配置器实时更新椭圆外框、长短轴辅助线、边界条件和可复制代码；``a=b`` 时为精确圆。
* 旋转后的内部采样、弧长边界采样和解析外法线均完成自动化验证。
* 移除多边形导入区中的“加载 L 形示例”按钮及对应事件代码。

2026-09-30：图片曲率识别与可编辑闭合样条
------------------------------------------------

* 核心新增 ``op.ClosedSpline``，使用闭合三次样条表示平滑图片轮廓，边界法向量
  由曲线导数计算。
* 自动模式检测连续曲率：圆弧/平滑段进入样条，直线尖角图形继续进入多边形。
* 控制点拖动和张力参数都会实时更新 SVG 预览及可复制 Python 代码。
* 真实圆角截图与五角星完成浏览器分流测试，网页样条代码完成 1 步训练。

2026-09-30：真圆与可调圆角几何
----------------------------------------

* 核心新增 ``op.Disk`` 和 ``op.RoundedRectangle``，圆弧使用解析几何而非折线近似。
* 配置器新增圆心、半径和圆角半径参数，预览和代码实时更新。
* PNG/JPG 导入可强制拟合纯圆和轴对齐圆角矩形；自动模式由曲率识别在闭合样条
  与尖角多边形之间选择。
* 完成采样、外法线、纯圆退化和 Neumann 短训练测试。

2026-09-30：多边形拖动缩放与尺寸标注
---------------------------------------------

* 顶点拖动坐标改为从 PointerEvent 的 CSS 像素显式换算到 SVG
  ``viewBox``，包含居中留白，页面放大后不再偏移。
* 多边形预览图显示与表格一致的顶点序号和 ``i→j`` 实时物理边长，
  拖动时同步更新。
* 图片关键拐角简化默认值调为 2 px，用户五角星保留
  5 个外尖角和 5 个内凹角。
* 多边形外轮廓和孔洞按顶点序号自动生成逐边条件，每条边可独立设置
  Dirichlet、Neumann 或 Robin；边界采样在点数充足时保证每条边至少有一个样本。

2026-09-30
----------

* 核心新增 ``op.Polygon``：支持凹多边形、多个多边形孔洞、内部/边界采样、
  边界判定与外法线，并可读取 CSV、JSON 和 GeoJSON Polygon。
* 新方程配置器增加“导入多边形文件”：检查自相交、孔洞越界和孔洞重叠，
  实时预览外轮廓与孔洞，并把几何坐标嵌入可直接运行的 Python 代码。
* 增加 PNG/JPG 高对比填充色块轮廓识别：自动判断深色/浅色前景，提取最大
  连通区和孔洞，并允许调整阈值与物理坐标范围。
* 轮廓识别可选 384/768/1024 像素精度和简化强度；新增识别后轮廓编辑器，
  支持旋转、平移、分轴缩放、预览拖点及精确坐标/边长/内角编辑。
* 补齐静态与时空多边形评估网格，验证 Dirichlet 和可变外法线
  Neumann 条件；195 项测试、Sphinx 严格构建与独立示例短跑通过。

2026-09-21
----------

* 新方程配置器把单一 Dirichlet 常数改为逐侧边界编辑器；一维左右、二维四边
  可分别选择 Dirichlet、Neumann 或 Robin，并生成真实可训练的条件代码。
* 核心新增 ``NeumannBC`` 与 ``RobinBC``，训练和训练后边界 RMSE 均支持法向导数。
* 一维预览按左右 Dirichlet 值移动端点并自动调整 ``u`` 轴；一维/二维仅在零点
  落入当前范围时绘制坐标原点，二维四边实时显示各自条件。

2026-09-19
----------

* 新方程配置器增加“常用模板 / 自由组合”；常用模板扩展为
  Poisson、Helmholtz、非线性 Poisson、热方程、反应--扩散、
  Allen--Cahn、对流--扩散和一维 Burgers。
* 自由组合支持逐行添加带符号系数、导数项、非线性项与源项，同步生成
  公式、预览图和 PyTorch 自动微分代码。
* 当时页面限定为单变量 PDE 与统一 Dirichlet 边界；逐侧 Neumann/Robin
  支持已在 2026-09-21 补充。

2026-09-18
----------

* 配置生成器的“新控制方程”改为同页集成窗口：可选一维/二维、
  静态/含时间方程，实时预览公式、边界和示例场，并与改进方法共同生成代码。
* 修复二维静态用户方程的训练后评估分支；一维/二维与静态/动态四组合均完成
  短跑，全部 176 项测试通过。

2026-09-11
----------

* 新增独立“新控制方程与适用方法检查”页：可编辑静态、动态或一维
  两材料界面模板，并实时分组显示可尝试、有条件和不适用的方法变体。
* 新增 ``op.check_method_compatibility(PROBLEM)`` 公开接口，复制的方程代码
  可在终端使用与网页一致的注册规则输出报告，而不开始训练。

2026-09-05
----------

* 结果图改为按问题类型生成：Poisson 显示一维解与绝对误差，Burgers 显示时间切片，
  双隧道显示物理坐标下的二维孔压场、隧道轮廓和共享色标。
* 双隧道新增 ``diagnostics.svg``，保存末时刻 PDE 残差场、物理孔压剖面和平均孔压
  消散趋势；``metrics.json`` 新增孔压范围、消散比和物理越界比例。
* 有独立参考解的问题统一增加 MAE 和 R2；双隧道没有 COMSOL 参考场，明确不报告
  虚假的解 L2、解 RMSE 或 R2。
* 一维结果图补充坐标数值，并将虚线参考解放在预测线之上，避免高精度结果遮住参考线。

2026-08-30
----------

* 新增双隧道二维模型 ``tunnel_model1``：二维各向异性固结问题。
* 修复双隧道 SVG 云图的外框填充覆盖色块问题，旧预测可直接重绘，不需要重新训练。
* 双隧道云图使用双三次显示插值消除方格感，同时保留原始数值网格、预测文件和指标不变。
* 时空区域由一维空间推广到通用空间维数，并增加矩形减圆孔几何。
* 增加二维时间切片 SVG、参数配置、最短脚本、测试和独立验证结果目录。
* 明确模型一缺少可读取参考解、部分默认参数为假设，当前不作论文精度复现声明。

0.2.0 - 2026-08-29
------------------

* 增加 ``op.solve(problem, method=...)``、紧凑 YAML 和统一命令行入口。
* 十种方法与普通 PINN 都能通过简称选择，并在训练前检查兼容性。
* 增加用户控制方程模板、自动结果图和失败记录。
* 将配置、临时运行与正式结果分别整理到独立目录。
* 方法入口增加家族/变体两级选择，十类方法建立独立目录并保留旧调用兼容。
* 自动测试扩展到 142 项；Sphinx 严格构建和 0.2.0 wheel 隔离导入通过。
* 自适应采样新增 RAM、WAM、重要性采样；损失平衡新增不确定性权重、
  ReLoBRaLo、幂损失和逐点 SA-PINN。
* 七个新增二级变体分别完成普通 PINN 300 步参考短对比，并保留负/混合结果。

2026-08-28
----------

* 新增公共 API Poisson 快速示例和对应 smoke 测试。
* 快速入门页加入统一 Python 调用流程。
* 调整用户文档措辞和安装版本信息。

0.1.2 - 2026-08-26
------------------

* 补齐 Poisson 5,000 步和 Burgers 10,000 步正式基线及参考对照图。
* 完成十种 PINN 方法的统一配置、小测试、正式对照、产物和方法页。
* 新增 Burgers Cole--Hopf 训练后评估与参考解可信性检查。
* 自适应采样的主要演示案例切换为 Burgers；2,048 个点是临时候选池，不是模型参数。
* 迁移学习同时比较全网络微调和只微调最后线性层。
* 十种方法重新执行三随机种子，30/30 次比较完成，失败 0。
* 十种方法、两条基线和三种子汇总图均已接入 Sphinx。

更完整的文件级记录见项目根目录 ``CHANGELOG.md``。

0.1.1
-----

首个可安装的 OpenPINN 版本。
