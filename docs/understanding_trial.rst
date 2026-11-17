Understanding 网页试用
======================

在这里输入中文或英文的自然语言问题描述，并用你自己的 OpenAI-compatible API Key
直接请求所配置的模型。页面显示 ``ProblemSpec`` 草稿、缺失字段、追问、警告和
``ready_for_validation``；请求不会经过 OpenPINN 服务器。

首期可执行示例提示词
--------------------

下面两段描述与当前结构注册表完全匹配。完成 Understanding、Planning 和 Hybrid Validation 后，
Validation 应显示 ``supported`` 及对应 ``backend_id``，CodeBuilder 应生成
``backend_runnable=true`` 的 ``openpinn_train.py``。

**一维 Poisson：**

.. code-block:: text

   一维 Poisson 方程，空间坐标 x∈[0,1] m，未知标量场 u，单位 1。
   强形式方程为 -dx(dx(u))=pi*pi*sin(pi*x)。
   x=0 和 x=1 均为 Dirichlet 边界，u=0。请保留 ASCII 表达式和显式单位。

**一维非定常热方程：**

.. code-block:: text

   一维非定常热方程，x∈[0,1] m，t∈[0,1] s，未知标量温度场 u，单位 K。
   alpha=0.01 m^2/s，强形式方程为 dt(u)-alpha*dx(dx(u))=0。
   x=0 为 Dirichlet 边界 u=0 K，x=1 为 Neumann 边界 dx(u)=0 K/m；
   t=0 的初值为 u=sin(pi*x/2) K。请保留 ASCII 表达式和显式单位。

旧热方程教学例中的 ``sin(pi*x/L)`` 和空边界值、以及二维 Poisson 示例都不匹配首期执行
注册表；补齐为上面的冻结结构前，它们只用于 Understanding 教学，不会得到可运行后端。

.. raw:: html

   <div class="understanding-trial" data-openpinn-understanding-trial>
     <label><span data-en="Problem description">问题描述</span>
       <textarea data-role="text" data-en-value="A two-dimensional plane-stress linear elasticity displacement problem on a rectangle, x = 0 to 2 and y = 0 to 2, with no body force. A hole has center (1, 1) and radius 0.25. Young's modulus is 210e9 Pa and Poisson's ratio is 0.3. The left edge is fixed, the right edge has x-direction traction 1e6, and the top, bottom, and hole boundaries are traction-free. Unknowns are ux and uy.">这是一个二维平面应力线弹性位移问题，无体力。矩形区域 x 从 0 到 2，y 从 0 到 2，有一个圆孔，中心 (1, 1)，半径 0.25。材料杨氏模量 E=210e9 Pa，泊松比 nu=0.3。左边固定，右边施加 x 方向牵引 1e6，上下边和孔边自由。未知量为 ux、uy。</textarea>
     </label>
     <div class="understanding-trial-controls">
       <label><span data-en="Optional solution path">可选求解路径</span>
         <select data-role="path">
           <option value="" data-en="Unspecified">未指定</option>
           <option value="pinn">PINN</option>
           <option value="operator">Operator</option>
         </select>
       </label>
       <button type="button" data-role="run" data-en="Understand">理解 / 试一试</button>
     </div>
     <div class="understanding-trial-api">
       <h2 data-en="API configuration">API 配置</h2>
       <div class="understanding-trial-llm-settings">
         <label><span data-en="OpenAI-compatible base URL">OpenAI 兼容接口地址</span>
           <input data-role="base-url" type="url" value="https://api.openai.com/v1" spellcheck="false">
         </label>
         <label><span data-en="Model name">模型名称</span>
           <input data-role="model" type="text" value="gpt-4o-mini" spellcheck="false">
         </label>
         <label><span data-en="API Key (stored only in this browser)">API Key（只保存在本浏览器）</span>
           <input data-role="api-key" type="password" value="" autocomplete="off" spellcheck="false">
         </label>
         <button type="button" class="understanding-trial-clear-key" data-role="clear-key" data-en="Clear Key">清除 Key</button>
       </div>
       <p data-en="The Key is stored only in this browser's localStorage and is sent directly to the configured endpoint. It is never sent to an OpenPINN server.">Key 只写入本浏览器的 localStorage，并由浏览器直接发送到所配置的接口；不会发送给 OpenPINN 服务器。</p>
     </div>
     <label><span data-en="Optional answers to previous questions (JSON object; nested or dotted keys)">对上轮问题的可选补充答案（JSON 对象；支持嵌套或点分字段）</span>
       <textarea data-role="answers" spellcheck="false" placeholder='例如 {"constitutive.E": 70000000000}' data-en-placeholder='For example {"constitutive.E": 70000000000}'></textarea>
     </label>
     <p class="understanding-trial-error" data-role="input-error" role="alert"></p>
     <div class="understanding-trial-results" data-role="results" hidden>
       <p class="understanding-trial-status"><strong><code>ready_for_validation</code>:</strong> <span data-role="ready" data-value="false"></span></p>
       <div class="understanding-trial-apply">
         <a class="understanding-trial-apply-button" data-role="apply" href="method_configurator.html#configurator-workbench" aria-disabled="true" data-en="Apply to configurator">填入配置器</a>
         <p data-role="apply-warning" role="status" aria-live="polite"></p>
       </div>
       <section class="understanding-trial-planning" data-role="planning">
         <h2 data-en="Solver Planning preview">Solver Planning 预览</h2>
         <div class="understanding-trial-llm-settings">
           <label><span data-en="Planning API base URL">Planning 接口地址</span>
             <input data-role="plan-base-url" type="url" value="https://api.openai.com/v1" spellcheck="false">
           </label>
           <label><span data-en="Planning model name">Planning 模型名称</span>
             <input data-role="plan-model" type="text" value="gpt-4o-mini" spellcheck="false">
           </label>
           <label><span data-en="Planning API Key (stored only in this browser)">Planning API Key（只保存在本浏览器）</span>
             <input data-role="plan-api-key" type="password" value="" autocomplete="off" spellcheck="false">
           </label>
           <button type="button" data-role="plan-clear-key" data-en="Clear planning Key">清除 Planning Key</button>
         </div>
         <button type="button" data-role="plan" disabled data-en="Recommend method">推荐方法</button>
         <p data-role="plan-status" role="status" aria-live="polite" data-en="Planning only recommends a MethodSpec draft. Python Specification Validation Agent must validate the final pair before code generation or download.">Planning 只推荐 MethodSpec 草稿；生码或下载前仍须由 Python Specification Validation Agent 校验最终组合。</p>
         <div data-role="plan-results" hidden>
           <p><strong><code>ready_for_validation</code>:</strong> <span data-role="plan-ready" data-value="false"></span></p>
           <div class="understanding-trial-grid">
             <section><h3 data-en="Method draft">方法草稿</h3><pre><code data-role="method-draft"></code></pre></section>
             <section><h3 data-en="Rationale">理由</h3><ul data-role="plan-rationale"></ul></section>
             <section><h3 data-en="Planning warnings">规划警告</h3><ul data-role="plan-warnings"></ul></section>
           </div>
           <a class="understanding-trial-apply-button" data-role="apply-method" href="method_configurator.html#configurator-method-title" aria-disabled="true" data-en="Apply method draft to configurator">将方法草稿填入配置器</a>
           <a class="understanding-trial-apply-button" data-role="apply-joint" href="method_configurator.html#agent-pipeline-title" aria-disabled="true" data-en="Import Understanding + Planning together">将 Understanding 问题与 Planning 方法一起导入配置器</a>
         </div>
       </section>
       <section class="understanding-trial-planning" data-role="layered-validation">
         <h2 data-en="Hybrid Validation Agent">Hybrid Validation Agent</h2>
         <p data-en="Deterministic structure checks, executable capability matching, and real LLM semantic review are separate. None proves physical correctness; supported capability may proceed to CodeBuilder, while the browser never trains.">确定性结构检查、可执行能力匹配与真实 LLM 语义审查分开显示；三者都不证明物理正确。能力为 supported 时可继续进入 CodeBuilder，网页本身不会训练。</p>
         <div class="understanding-trial-llm-settings">
           <label><span data-en="Validation API base URL">Validation 接口地址</span>
             <input data-role="validation-base-url" type="url" value="https://api.openai.com/v1" spellcheck="false">
           </label>
           <label><span data-en="Validation model">Validation 模型</span>
             <input data-role="validation-model" type="text" value="gpt-4o-mini" spellcheck="false">
           </label>
           <label><span data-en="Validation API Key (browser only)">Validation API Key（仅当前浏览器）</span>
             <input data-role="validation-api-key" type="password" value="" autocomplete="off" spellcheck="false">
           </label>
           <label><input data-role="validation-reuse-planning" type="checkbox"> <span data-en="Explicitly reuse current Planning API settings">显式复用当前 Planning API 设置</span></label>
           <button type="button" data-role="validation-clear-key" data-en="Clear Validation Key">清除 Validation Key</button>
         </div>
         <button type="button" data-role="validate-layered" disabled data-en="Run Hybrid Validation">运行 Hybrid Validation</button>
         <button type="button" data-role="cancel-validation" disabled data-en="Cancel semantic review">取消语义审查</button>
         <button type="button" data-role="download-generic" disabled data-en="Download non-training Hybrid report">下载非训练 Hybrid 报告</button>
         <p data-role="execution-capability" role="status" aria-live="polite" data-en="Execution capability: not checked.">执行能力：尚未检查。</p>
         <p data-role="validation-status" role="status" aria-live="polite" data-en="Run Hybrid Validation before entering CodeBuilder.">进入 CodeBuilder 前请运行 Hybrid Validation。</p>
         <div class="understanding-trial-grid">
           <section><h3 data-en="Deterministic structure report">确定性结构报告</h3><pre><code data-role="validation-report"></code></pre></section>
           <section><h3 data-en="LLM semantic report">LLM 语义报告</h3><pre><code data-role="semantic-report"></code></pre></section>
         </div>
       </section>
       <section class="understanding-trial-planning" data-role="codebuilder">
         <h2 data-en="Code Building Agent">Code Building Agent</h2>
         <p data-en="A complete current Hybrid Validation report enters deterministic capability matching. Frozen Poisson/heat cards and contract-bounded single-scalar strong-form problems can produce a budgeted local training script; unsupported structures retain an honest non-runnable skeleton. The browser never starts training.">当前且完整的 Hybrid Validation 报告会进入确定性能力匹配。冻结 Poisson/heat 卡以及合同范围内的单标量强形式问题可生成带预算的本地训练脚本；不支持的结构仍生成诚实的不可运行骨架。网页不会启动训练。</p>
         <button type="button" data-role="build-code" disabled data-en="Generate Python code">生成 Python 代码</button>
         <button type="button" data-role="copy-code" disabled data-en="Copy Python code">复制 Python 代码</button>
         <button type="button" data-role="download-codebuilder" disabled data-en="Download Python .py">下载 Python .py</button>
         <p data-role="build-status" role="status" aria-live="polite" data-en="Run Hybrid Validation for the current final pair before code generation.">生码前，请先对当前最终组合运行 Hybrid Validation。</p>
         <div class="understanding-trial-grid">
           <section><h3 data-en="BuildReport">BuildReport</h3><pre><code data-role="build-report"></code></pre></section>
           <section><h3 data-en="Generated Python">生成的 Python</h3><pre><code data-role="build-source"></code></pre></section>
         </div>
       </section>
       <section class="understanding-trial-planning" data-role="diagnostic">
         <h2 data-en="Diagnostic Evaluation Agent">Diagnostic Evaluation Agent</h2>
         <p data-en="Upload a RunArtifact JSON or ZIP after CodeBuilder. Deterministic checks and real LLM diagnosis remain separate; diagnosis never rewrites code, specs, or hyperparameters and does not perform Repair.">在 CodeBuilder 后上传 RunArtifact JSON 或 ZIP。确定性检查与真实 LLM 诊断分开显示；诊断不会改写代码、规格或超参数，也不执行 Repair。</p>
         <label><span data-en="RunArtifact JSON or ZIP">RunArtifact JSON 或 ZIP</span>
           <input data-role="diagnostic-file" type="file" accept=".json,.zip,application/json,application/zip" disabled>
         </label>
         <div class="understanding-trial-llm-settings">
           <label><span data-en="Diagnostic API base URL">Diagnostic 接口地址</span>
             <input data-role="diagnostic-base-url" type="url" value="https://api.openai.com/v1" spellcheck="false">
           </label>
           <label><span data-en="Diagnostic model">Diagnostic 模型</span>
             <input data-role="diagnostic-model" type="text" value="gpt-4o-mini" spellcheck="false">
           </label>
           <label><span data-en="Diagnostic API Key (browser only)">Diagnostic API Key（仅当前浏览器）</span>
             <input data-role="diagnostic-api-key" type="password" value="" autocomplete="off" spellcheck="false">
           </label>
           <label><input data-role="diagnostic-reuse-validation" type="checkbox"> <span data-en="Explicitly reuse current Validation API settings">显式复用当前 Validation API 设置</span></label>
           <button type="button" data-role="diagnostic-clear-key" data-en="Clear Diagnostic Key">清除 Diagnostic Key</button>
         </div>
         <button type="button" data-role="run-diagnostic" disabled data-en="Run diagnosis">运行诊断</button>
         <button type="button" data-role="download-diagnostic" disabled data-en="Download DiagnosticReport">下载 DiagnosticReport</button>
         <p data-role="diagnostic-file-status" role="status" aria-live="polite" data-en="Generate current code, run it locally, then choose its RunArtifact.">生成当前代码并在本地运行后，再选择其 RunArtifact。</p>
         <p data-role="diagnostic-status" role="status" aria-live="polite" data-en="Diagnosis is guidance, not proof or Repair.">诊断只是建议，不是证明，也不是 Repair。</p>
         <div class="understanding-trial-grid">
           <section><h3 data-en="Deterministic diagnosis">确定性诊断</h3><pre><code data-role="diagnostic-deterministic"></code></pre></section>
           <section><h3 data-en="LLM diagnosis">LLM 诊断</h3><pre><code data-role="diagnostic-semantic"></code></pre></section>
           <section><h3 data-en="DiagnosticReport">DiagnosticReport</h3><pre><code data-role="diagnostic-report"></code></pre></section>
           <section><h3 data-en="Human summary">人工可读摘要</h3><p data-role="diagnostic-summary"></p></section>
         </div>
       </section>
       <div class="understanding-trial-grid">
         <section><h2 data-en="Draft">草稿</h2><pre><code data-role="draft"></code></pre></section>
         <section><h2 data-en="Missing fields">缺失字段</h2><ul data-role="missing"></ul></section>
         <section><h2 data-en="Questions">追问</h2><ul data-role="questions"></ul></section>
         <section><h2 data-en="Warnings">警告</h2><ul data-role="warnings"></ul></section>
       </div>
     </div>
   </div>

.. important::

   推荐使用标准联合按钮，把 Understanding 问题与 Planning 方法作为一份
   ``JointAgentHandoff`` 导入配置生成器。线弹性圆孔板会进入带孔板编辑界面；未映射字段、
   缺少来源、冲突及旧 Validation 状态都作为非阻断提示保留在原始 JSON 中。配置器始终允许
   导出诚实的完整代码或带显式 ``NotImplementedError`` 的不完整代码；网页本身不训练。

页面按已声明的物理族追问。热传导、泊松和其他已声明物理族使用各自或通用的问题侧完整性
检查，不会混入线弹性专用的 E、ν、``ux``、``uy``；物理族不清楚时只会询问是哪一种
物理，不会默认线弹性。
提示词允许把明显事实做合理延伸，并要求用 ``understanding.inferred`` 警告标出推断；
真正缺失的数值仍然要追问。Understanding 报告本身不选择网络、采样、损失或优化器；
独立的“推荐方法”按钮才调用 Planning。页面不实现 Repair。

预填通过当前标签页的 ``sessionStorage`` 传递，键为
``openpinn-understanding-configurator-v1``，配置器读取后立即删除。映射规则如下；只写入草稿
明确提供且能无损映射的值。热传导和泊松方程可由结构化草稿补成编辑器合法公式；补入的
只是公式结构与标准单位，不补造物理数值：

方法草稿使用 ``openpinn-planning-configurator-v1`` 交接。方法、变体、求解路径和训练步数
会在配置器存在对应控件时填入；不能表示的网络、采样、损失、优化器或微分设置会列为未映射
警告。交接数据不含 Key，且编辑后仍保持 Validation Agent 门禁。浏览器只执行草稿形状或
同源合同预检，不会声称已经运行完整 Python Validation。

M35 的“网页分层 Validation”把当前 ``problem_draft`` 和 ``method_draft`` 作为一个
指纹化组合处理。通用问题使用 ``GenericProblemDescription`` ``schema_version=1``，
物理标签是自由字符串；多场、参数、坐标、方程、几何、边初值、耦合与无法解释的
``opaque_blocks`` 都保留。检查报告按 L0–L6 给出 ``pass`` / ``fail`` / ``unknown`` /
``unsupported`` / ``skipped``，不使用一个 ``ok`` 暗示可训练。表达式只经过受限词法、
符号作用域和导数坐标检查，绝不通过 ``eval`` 执行。

适定性、稳定性、收敛、可训练性与物理正确性层始终为 ``unknown``。方法或后端
``unsupported`` 不会把问题描述标成无效。页面可下载明确标记为“非训练产物”的通用规格、
方法草稿与分层报告。这里描述的是 M35 当时的 Validation 产物；当前注册结构在完整 Hybrid
Validation 后可继续交给 CodeBuilder 生成本地脚本，但 Validation 自身不生码，浏览器也不训练。
只有通用字段可无损映射时 adapter 才产生 Executable pair；否则带 path/reason 拒绝。即使映射
成功，仍必须通过 Python M34 ``validate_specs``，通用报告不能绕过它。

M36 把这一步升级为 Hybrid Validation Agent。确定性轨仍只检查缺失声明、结构、
字段形状、受限表达式、符号引用、指纹和 adapter；语义轨使用用户明确配置的
OpenAI-compatible API，只报告 ``no_concern_identified`` / ``concerns`` /
``insufficient_information``。这些状态是带输入证据的模型建议，不是数学、物理、
收敛或可训练证明。未注册方法仅是 capability ``unsupported``，不会把问题判为物理无效。

Validation 接口、模型和 Key 有独立控件；只有勾选“显式复用”才读取当前 Planning 设置。
Key 仅保存在当前浏览器 ``localStorage``，不进入 prompt 以外的报告、下载、错误或代码库。
401/403、超时、网络/CORS、取消、返回形状错误和乱序响应都会使语义审查显示
``incomplete`` 或 stale，不会静默退回规则并冒充成功。LLM 不能覆盖确定性 fail、M34 或训练锁。

M35 合并本身不等于线上验收。M36 按实际验收症状修复了 heat ``alpha``/扩散率声明丢失、
配置器编辑后旧指纹不可见地过期，以及导出仍锁定时声称“已可复制完整代码”的文案。
配置器会明确标记 stale，并把当前参数和训练步数快照发回本页重新审查。

M37 在 U→P→V 后加入 Code Building Agent。它只接受当前指纹对应、确定性检查无 fail、
且语义审查完整的最终 Problem+Method；缺字段、旧指纹或 ``incomplete`` 都会硬阻断。
``concerns`` 不阻断骨架生成，但会原样出现在 ``BuildReport`` 和骨架中供人工复核。
M37 当时的通用产品路径只生成确定性的 Python 装配模板，不依赖旧线弹性 adapter，也不调用
LLM 写代码或解释代码；当时成功报告始终为 ``backend_runnable=false``。Stage D–F 后，只有
两张冻结问题卡能生成可运行本地脚本，其余结构仍保持该不可运行骨架。

M38 增加第五段 Diagnostic Evaluation Agent。当时的 CodeBuilder 骨架运行时会自动写入
``openpinn_run/run_artifact.json``；其中包含当前 pair fingerprint、后端可运行标记、
``not_runnable`` / ``completed`` / ``failed`` 状态、loss 与误差槽，以及 failure stack 槽。
通用骨架写 ``backend_runnable=false`` 和 ``status=not_runnable``，不会伪造训练结果；该规则
今天仍适用于未注册结构。

Diagnostic 页面只在本地读取 JSON，或从 ZIP 中读取 ``run_artifact.json``。确定性轨检查
schema、指纹、状态、loss/error metrics 和失败记录；真实 LLM 轨使用独立接口、模型和 Key，
只有显式勾选才复用 Validation 设置。认证、超时、CORS 或返回结构错误均显示
``incomplete``，不会把确定性规则冒充 LLM 成功。``DiagnosticReport`` 包含双轨结果、
``human_summary``、disclaimer 和非绑定 ``repair_hints``；诊断不会修改 Problem、Method、
脚本或超参数，``repair_performed=false``，也不保证后续 Repair 成功。

Diagnostic Key 仅存当前浏览器 ``localStorage``，不写入 RunArtifact、报告、下载、错误或
代码库。诊断是训后观察，不证明正确性、适定性、收敛性或可训练性；Repair 和任意物理
可运行后端仍属于后续里程碑。

Stage A 在现有 ``GenericProblemDescription`` 上增加最小可训问题字段合同，而没有增加训练
后端。空间和时间坐标现在携带一等的 ``bounds`` 与 ``unit``；未知场包含 name、components
和 unit；参数要求唯一 name、value 与 unit；PDE、BC、IC 和 geometry 继续作为一等字段。
``trainable_contract.mapping_complete`` 是“可供后续训练路径消费”的独立映射信号，不等于
``ready_for_validation``、物理正确或当前可训练。Validation 报告同时暴露
``summary.mapping_complete_for_later_training``。

对于方程使用 ``alpha``、同时存在等值 unit-less ``diffusivity`` 的情况，映射器明确选择
带单位的 ``alpha`` 作为 canonical parameter，并在 ``normalizations`` 中记录原因；数值或
单位冲突则保留为 issue，使 mapping complete 为 false。时间范围若只存在于 opaque block、
缺少 BC/动态问题 IC、未绑定表达式符号或任何必需单位，同样不能通过 Stage A 合同。
Stage D–F 在上述合同上加入结构注册表：冻结的一维 Poisson Dirichlet 卡与一维非定常 heat
混合边界卡可生成 ``backend_runnable=true`` 的 ``openpinn_train.py``，并在本地复用共享
``run_pinn`` 引擎。其他结构仍生成带 ``NotImplementedError`` 的诚实骨架。脚本要求显式步数
和墙钟预算，``--smoke`` 只运行一真实优化步；训练与参考网格误差在本地写入同次运行的
``RunArtifact``，网页不执行 Python。Diagnostic 拒绝把 smoke、示例、fixture、旧指纹或缺少
同次运行来源的工件当作完整评估；当前没有冻结精度阈值，因此 ``completed`` 也不等于精度通过。

当前执行注册表还支持合同范围内的通用单标量强形式问题：空间是 ``x`` 区间或 ``x,y`` 矩形，
可选末尾时间坐标 ``t``，可包含多条受限表达式残差，并支持 Dirichlet、沿声明坐标求导的
Neumann 与 ``alpha*u + beta*du/dcoordinate = value`` Robin 条件。通用问题可以使用 ``v`` 等
其他单标量字段名。多场、向量场、周期条件、复杂几何、弱形式及不在安全表达式白名单中的
函数会明确返回 ``unsupported`` 或 ``incomplete``。没有声明参考解时，RunArtifact 的
``error_metrics`` 为空；系统不会猜测参考解或精度结论。

下载脚本、本地短训与上传诊断
------------------------------

在 Python 3.10+ 虚拟环境中从仓库根目录安装依赖。CPU 足够；若出现
``ModuleNotFoundError: openpinn`` 或缺少 ``torch``，请确认使用的是这个虚拟环境：

.. code-block:: bash

   python -m pip install -e ".[test]"
   python scripts/build_docs.py

第二条命令用于本地预览，打开 ``docs/_build/html/understanding_trial.html``；它不是官方部署。
在网页完成 Hybrid Validation 后，仅当 BuildReport 显示 ``backend_runnable=true`` 时下载
``openpinn_train.py``，然后运行：

.. code-block:: bash

   python openpinn_train.py --max-steps 20 --max-wall-seconds 30 --device cpu --output-dir openpinn_run
   # 只检查一真实优化步；不能作为完整评估上传
   python openpinn_train.py --smoke --max-wall-seconds 30 --device cpu

脚本本地默认值是方法中的步数（缺失时 1000）和墙钟秒数（缺失时 300）；验收短训显式使用
20 步、30 秒或更小预算。运行后上传 ``openpinn_run/run_artifact.json``，也可以把该文件或
目录打成 ZIP 后上传。不要把骨架、``--smoke`` 产物或测试 fixture 当成训练结果。

阅读结果时检查 ``status``、``steps_completed``、实测 ``max_abs`` / ``rel_l2`` 以及 Diagnostic
outcome。短预算通常得到 ``budget_exhausted``，这不表示成功或收敛；``completed`` 也不表示
精度已验证。步数或墙钟必须为正；页面快照修改后旧指纹会被拒绝；错误目录或 Python 环境会
导致导入失败；结构偏离上述冻结卡会得到 ``unsupported`` 或 ``incomplete``。

线弹性的无损 Executable adapter 映射保持不变，仍需 Python M34；本期没有注册线弹性
``run_pinn`` 后端。Stage G 不包含 Repair、任意 PDE 后端或官方精度阈值。

.. list-table:: Understanding 草稿到自定义配置器
   :header-rows: 1

   * - 草稿字段
     - 配置器字段
   * - ``dim`` / ``kind`` / 时间依赖线索
     - 自定义问题的空间维度与静态/动态上下文
   * - ``expression_source``、``equation`` 或 ``pde.expression``
     - 先规范为仅含 ``dx``、``dy``、``dt`` 和显式 ``*`` 的 ASCII 公式；常见
       ``Δ``/``∇²``、下标导数和 Unicode 运算符会被转换
   * - ``parameters`` / ``constants`` / ``coefficients``
     - 保留明确数值；缺单位时使用无量纲并提示。热传导 ``alpha`` 默认单位为
       ``m^2/s``，泊松源项 ``f`` 默认单位为 ``1/m^2``
   * - ``field`` / ``units``
     - 未知场、场单位、坐标与时间单位
   * - ``geometry`` 的区间、矩形外框或 ``bounds``
     - 自定义几何范围；孔洞保留提示，由用户在自定义几何中继续编辑
   * - 可无损表示的标量 ``boundaries``
     - 自定义 PDE/BC 的对应边界行
   * - ``initial_condition`` / ``initial_conditions``
     - 动态上下文与可表示的标量常数初值；多分量初值保留提示并由用户核对

未就绪草稿可选择“仍填已有字段”；配置器会显示醒目警告，且不会强制开启复制或下载。
若模型没有返回合法公式，映射器会从热传导的扩散率或泊松的源项合成公式；仍无法合成时，
配置器保留原公式和参数并明确显示“仅完成部分预填”，不会把空公式当作成功。
准备正式配置时请前往 :doc:`方法配置器 <method_configurator>`；完整架构与 Python API
见 :doc:`问题理解说明 <problem_understanding>`。

接口必须支持浏览器跨域请求和 OpenAI-compatible ``/chat/completions``。可填写通用
OpenAI-compatible 或国内兼容服务商提供的 base URL 与模型名；具体 URL、模型、配额和
CORS 策略以服务商文档为准。缺少 Key、接口不可达、超时、CORS 被拒绝或 JSON 不合法
时，页面会明确报错并保持未就绪，不会静默换用另一种结果。点击“清除 Key”会从当前
浏览器删除它。页面交付后，由用户使用自己的真实 Key 自行验收；CI 不使用真实 Key。
