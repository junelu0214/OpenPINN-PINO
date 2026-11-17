# Changelog

## M32

- Add the LLM-primary ``openpinn.agents.plan_method`` API and fixed
  ``PlanReport`` schema for recommending a registered, MethodSpec-aligned
  method draft with short Chinese rationale. ``auto`` falls back visibly to a
  deterministic rules table when credentials, connectivity, or valid JSON are
  unavailable.
- Add a read-only Solver Planning preview to the Understanding web trial. It
  uses the user's existing browser API settings, validates the returned method
  shape, and never unlocks code generation: callers must still pass the problem
  and method draft through the existing Agent A ``validate`` gate.

## M31

- Synthesize editor-valid ASCII residuals when Understanding returns structured
  heat or Poisson data without a usable expression. Heat drafts use ``alpha``
  and Poisson drafts use ``f`` with explicit default units; common Unicode
  Laplacian notation is normalized before handoff.
- Make configurator prefill transactional: an empty or invalid expression is
  reported as a partial apply and cannot erase the current formula or parameter
  table. The equation editor now presents only the field, parameters, and
  equation while retaining its existing validation and import/export APIs.

## M30

- Add a bilingual “Apply to configurator” handoff from Understanding to the
  existing custom PDE/BC editor. The one-time `sessionStorage` payload maps
  declared dimensions, geometry bounds, scalar boundaries, initial values,
  fields, units, expressions, and explicitly unit-bearing parameters without
  inventing missing numbers. Linear elasticity follows this same custom path;
  no Understanding prefill selects the plate-with-hole UI.
- Treat any declared physics family with a generic family-scoped readiness
  gate while unknown physics still asks for its type. Partial apply is visibly
  marked and cannot unlock copy/download until the user edits and the existing
  configurator validation passes. Understanding readiness remains separate
  from Agent A; Planning, Repair, training APIs, and browser training are
  unchanged.

## M29

- Decide Understanding `missing`, `questions`, and `ready_for_validation` from
  the declared physics family. Heat, diffusion, and Poisson drafts are no
  longer blocked by the linear-elasticity Agent A / `validateDraft` gate, and
  unknown physics asks which equation it is instead of assuming elasticity.
- Allow the Understanding prompt to record reasonable inferences in
  `understanding.inferred` warnings, and add few-shot examples for the 1D heat
  and 2D Poisson cases. JSON shape checks stay in place. Agent A remains the
  later gate for configurator YAML and CodeBuilder; this milestone does not
  change training APIs or browser training.

## M28

- Change the Understanding web trial to a user-owned OpenAI-compatible API Key
  path only. URL, model, and Key stay in browser `localStorage`, Clear Key
  removes the stored Key, and requests go directly to the configured provider.
- Remove Rules/Auto choices and silent rules fallback from the web UI. Missing
  keys, CORS/network failures, timeouts, and invalid JSON fail visibly and clear
  stale readiness. Python keeps its deterministic rules CI path, while the
  compatible client gains optional Bearer authentication.

## M27

- Extend `understand_problem` with `rules`, `llm`, and `auto` backends. The
  optional stdlib client targets a local OpenAI-compatible Ollama endpoint and
  validates its JSON before unchanged Agent A determines readiness.
- Add local-LLM controls to the bilingual Understanding web trial. Connection,
  CORS, timeout, and payload failures visibly fall back to deterministic rules;
  no model weights, training APIs, browser training, Planning, or Repair are
  added.

## M26

- Add a discoverable bilingual Understanding web trial under the documentation
  Start section. Its deterministic JavaScript engine renders the M25 draft,
  missing fields, questions, warnings, and readiness shape without an API key.
- Keep the static trial pre-codegen only: it exposes no fill, generation,
  download, or browser-training action. Any downstream codegen must still pass
  the existing Python Agent A `validate` gate.

## M25

- Add deterministic `openpinn.agents.understand_problem` and
  `UnderstandReport` APIs for turning Chinese- or English-language problem
  descriptions into partial `ProblemSpec`-shaped drafts with missing fields
  and clarification questions.
- Require every draft marked ready to pass the existing Agent A problem-side
  `validate` gate. This milestone does not add solver planning, repair,
  free-form training-script generation, or training API changes.

## M19

- Commit demo-quality figures and metrics for the three code-tutorial primaries:
  Burgers relative L2 about 0.041 at 3000 steps, PINN 2D Poisson about 0.0337
  at 5000 steps, and FNO × physics_only about 0.039 at 10000 steps. The FNO
  result is an accepted showcase even though it misses the stricter automated
  threshold.
- Keep plate-with-hole out of the showcase path. Notebooks still include a
  short CPU smoke, and Sphinx does not retrain them.

## M20

- Converge the Tutorials sidebar on the Configurator and Code tutorial hubs.
  Code tutorials show three primaries: 1D Burgers, PINN 2D Poisson
  (`05_pinn_poisson_2d`), and FNO × physics_only. The plate-with-hole notebook
  stays in the tree but leaves the sidebar. Legacy pages remain built as orphan
  sources so existing URLs keep working.
- Build default Chinese and `/en/` English Sphinx trees with an in-page language
  switcher. Operator-path headings, help, validation, and status feedback are
  now Chinese on the default site; the remaining localization backlog is
  limited to registry metadata such as parameter-schema labels and
  `hook.summary`.

## M18

- Converge tutorials around a PINN/Operator configurator walkthrough and a
  three-case Code tutorials path: 1D Burgers, 2D plate-with-hole, and 2D
  Poisson with FNO × physics-only.
- Keep the remaining executed M17 notebooks under More examples and retain
  committed-output, execute-never Sphinx builds.

## M17

- Add an executed Jupyter tutorial path covering Intro, PINN, Operators, and
  Extension under `examples/tutorials/`, rendered by nbsphinx in the existing
  Tutorials sidebar group.
- Treat committed outputs as documentation source and disable notebook execution
  during Sphinx builds; a lightweight test validates notebook structure and output.

## M16

- Restructure the Sphinx sidebar into Start, Tutorials, User guide, Reference,
  and Developer groups, with the configurator in Start and collapsed navigation.
- Add a concise operator tutorial entry. Keep the historical delivery-status and
  workload pages outside the sidebar; no training or library API changes.

## M15

- On the operator path, training observation CSV and post-train reference CSV
  are separate entries. Each has its own enable switch, path, and column/scale
  mapping. `data_only` and `hybrid` require an observation; `physics_only`
  hides it; the reference stays optional on every mode.
- Generated scripts emit `OBSERVATION_*` only for `train_operator_problem`
  and `REFERENCE_*` only for `evaluate_operator_reference_csv`. Filling one
  side does not reuse that file for the other side. The PINN path remains
  post-train reference evaluation only.

## M14

- Operator configurator downloads now train DeepONet or FNO on the designed
  1D/2D PDE and boundary conditions. Local CSV observations provide the
  supervised term for data-only/hybrid and optional post-training evaluation.
- The governing-equation dropdown is custom-only; built-in cases remain in
  Tutorials and examples. No training or file upload runs in the browser.

## 2026-10-28 · Public demo and operator narrative consistency

- Confirm the M7 public documentation contract for the live gated demo:
  unauthenticated requests receive an HTTP 401 invite page, the shared invite
  is obtained from the operator out of band, and no passphrase belongs in git
  or public issues. Authenticated entry and `.py` download remain a one-time
  human verification documented in `docs/demo_site.md`.
- Bring the Sphinx changelog and onboarding notes forward through M11–M12.
  Current documentation consistently presents DeepONet/FNO as
  `model × training_mode`, `CoordinateOperator` as the canonical protocol, and
  coordinate autodiff versus grid derivatives as the natural model paths.
- Keep PINO documented only as a deprecated selection shim and advanced grid
  helper. Operator governing-PDE design remains custom-only; built-in cases
  remain tutorials.

## 2026-10-28 · CoordinateOperator and native DeepONet autodiff

- Rename the canonical coordinate evaluation protocol to `CoordinateOperator`.
  `CoordinateModel` remains an exact deprecated alias for one minor version.
- Route DeepONet `physics_only` and `hybrid` through a pointwise 1D Poisson
  residual differentiated at query coordinates with autodiff. The returned
  model is the coordinate-native `DeepONet`; labeled targets remain diagnostics
  only in `physics_only`.
- Remove `GridOperatorFromCoordinateModel` and its public exports. Cross-grid
  interpolation is no longer presented as a formal product capability. FNO
  physics continues to use `GridPDEProblem` and grid derivatives unchanged.
  M11 DeepONet configs that still pass `derivative="spectral"` or
  `"finite_difference"` receive a deprecation warning; the ignored option is
  accepted for one minor version so downloaded scripts keep running.

## 2026-10-28 · Operator model × training-mode configurator

- Redesign the operator path around a canonical model (DeepONet or FNO) and
  the shared `data_only`, `physics_only`, and `hybrid` training modes.
- Remove PINO from configurator choices and presets. For one minor version,
  `build_operator_config("pino")` and `train_operator("pino")` emit a
  `DeprecationWarning` and route to FNO plus the requested mode.
- Route FNO physics modes through the M10 `GridPDEProblem` trainer. The
  coordinate-native DeepONet physics path that supersedes M11's temporary
  adaptation is recorded in the M12 entry above.
- Add a collapsed registry-driven derivative/sampler/optimizer section whose
  selections are preserved in generated scripts. Operator governing PDE
  design remains custom-only; PINN problem dropdowns are unchanged.

## 2026-10-26 · README quick path and handoff-note cleanup

- README quick path is install, plate-with-hole Kirsch smoke, `train_operator` for DeepONet/FNO/PINO, then the local Sphinx configurator. Clone URL is `junelu0214/OpenPINN-upstream`. This mirror does not publish a public docs site.
- Remove dated handoff notes that Sphinx and the tests do not use as live pages: `DELIVERY_STATUS.md`, `FINAL_VALIDATION.md`, `PROJECT_SUMMARY.md`, `WORKLOAD.md`, and `catalog/VALIDATION_2026-10-15.md`.
- Keep `LICENSE_STATUS.md`. The Apache-2.0 grant stays in `LICENSE`. `runs/` and `results/` stay README-only.

## 2026-10-26 · PINO physics-informed operator

- Add a physics-informed neural operator on the FNO backbone. It is
  selectable with `build_operator_config` and `train_operator`.
- The loss is an optional data term plus a GridDerivative Poisson residual.
  `hybrid` is the practical default and can data-pretrain then fine-tune.
  `physics_only` runs and reports residuals; it is not claimed to recover
  the solution on high-mode k² problems. `solve` and the PINN catalog are
  unchanged.

## 2026-10-26 · FNO grid operator

- Add a Fourier neural operator that implements the GridOperator contract and
  is selectable with `build_operator_config` and `train_operator`.
- Add periodic spectral and finite-difference GridDerivative stencils for
  PDE-side derivatives of FNO outputs. `solve` and the PINN method catalog
  are unchanged. PINO stays unimplemented.

## 2026-10-26 · DeepONet coordinate operator

- Add an unstacked branch/trunk DeepONet that implements the CoordinateModel
  contract and is selectable with `build_operator_config` and `train_operator`.
- Expose its widths, activation, and smoke-trainer knobs on the operator hook
  and on the configurator payload. `solve` and the PINN method catalog are
  unchanged. FNO and PINO stay unimplemented.

## 2026-10-26 · Operator protocol hooks

- Document the CoordinateModel and GridOperator contracts: coordinate queries
  use autodiff, and structured-grid operators use grid derivatives.
- Register DeepONet, FNO, and PINO as unimplemented, non-selectable hooks that
  the documentation configurator can discover. No operator networks are added,
  and the existing method catalog is unchanged.

## 2026-10-16 · GitHub 首次使用验收修正

- README 开发验证顺序改为先构建网页，再运行依赖构建产物的网页测试。
- 新增基于已渲染正文的本地小节搜索索引，覆盖 HTML 说明书，支持中文原词、复合词、英文 API 名称和多关键词；摘要不再通过 `fetch` 读取本地文件。
- 搜索结果高亮避开 SVG、数学节点、代码和配置器控件，避免从搜索结果进入流程图页面时的 `getBBox` 异常。
- 圆孔编辑提示区分“轴对齐矩形＋全部圆孔”的解析训练路径与其他轮廓的多边形近似，不再一概宣称折线化。
- 增加正文提取、增量索引、离线搜索、分页、输入安全和开发命令顺序回归；不改变 PDE、训练算法、仓库可见性或 Pages 配置。

## 2026-10-03 · COMSOL/CAD 二维 DXF 导入

- “二维几何”将现有多边形导入标注为“识别图片”，COMSOL 和 CAD 共用一个 DXF 导入选项。
- 读取二维 ASCII DXF 闭合多段线、圆、完整椭圆及 bulge 圆弧，沿用预览、轮廓编辑、逐边条件和 `op.Polygon` 代码生成。
- 对开放轮廓、三维实体和多个不连通区域明确报错；提供带孔矩形 DXF 示例及解析测试。

## 2026-09-30 · 可旋转椭圆几何

- 新增 `op.Ellipse` 解析几何，可修改圆心、半轴 `a/b` 和旋转角。
- 配置器同步更新椭圆预览、轴线、物理范围与 `op.Ellipse(...)` 代码。
- 支持解析外法线和 Neumann/Robin 边界；`a=b` 时与圆形一致。
- 移除多边形导入区中的“加载 L 形示例”按钮及对应事件代码。

## 2026-09-30：真圆与可调圆角几何

- 核心新增 `op.Disk` 和 `op.RoundedRectangle`，圆弧使用解析几何而非多边形折线近似。
- 配置器新增圆心、半径和圆角半径参数，预览与生成代码随修改实时更新。
- PNG/JPG 导入提供强制纯圆与强制轴对齐圆角矩形拟合，自动模式则由后续的
  曲率识别在闭合样条与尖角多边形之间选择。
- 圆角半径为 0 时是矩形；正方形且半径等于半边长时严格退化为纯圆。
- 补齐包含判定、均匀采样、外法线和静态 Neumann 短训练测试。

## 2026-09-30：图片曲率识别与可编辑闭合样条

- 核心新增 `op.ClosedSpline`：闭合三次样条支持内部/边界采样、包含判定及解析切线外法线。
- 图片自动识别改为曲率分流：连续平滑段进入样条，直线与尖角继续使用多边形。
- 样条控制点可拖动，张力 `0–1` 可实时修改；预览和生成代码同步更新。
- 真实圆角截图识别为 16 控制点样条并完成网页代码短跑；五角星仍识别为 10 顶点多边形。

## 2026-09-30：多边形拖动缩放与尺寸标注

- 顶点拖动改为从 PointerEvent 的 CSS 像素显式换算到 SVG `viewBox`，
  包含居中留白，页面放大后不再发生坐标偏移。
- 多边形预览图增加与表格一致的顶点序号和 `i→j` 实时物理边长；
  拖动时顶点、序号和相邻边长同步更新。
- 图片关键拐角简化默认值调为 2 px，用户五角星保留 5 个外尖角和
  5 个内凹角。
- 多边形边界改为按顶点序号自动生成逐边设置；外轮廓和孔洞的每条边可
  独立设置 Dirichlet、Neumann 或 Robin。边界采样在点数充足时保证覆盖每条边。

## 2026-09-30：凹多边形与带孔区域导入

- 核心新增 `op.Polygon`：支持凹多边形、多个多边形孔洞、内部/边界采样、
  边界判定和外法线，可读取 CSV、JSON 和 GeoJSON Polygon。
- 新方程配置器增加多边形文件导入、实时预览、几何验证和可运行代码生成；
  异常文件与过多顶点会被明确拒绝。
- 配置器新增 PNG/JPG 高对比填充色块轮廓识别：支持深色/浅色前景、孔洞、
  噪点过滤、阈值调节和物理坐标映射；原图不会离开本地浏览器。
- 新增识别分辨率/简化强度选择和识别后轮廓编辑器：可旋转、平移、X/Y 独立缩放、
  拖动控制点，并精确修改顶点坐标、边长、内角及顶点数。
- 195 项测试、18/18 方法变体短跑、Sphinx 严格构建、wheel 项目外安装和
  GeoJSON 多边形 1 步训练全部通过。

## 2026-09-21：逐边界条件与坐标原点

- 新方程配置器支持一维左右、二维四边分别设置 Dirichlet、Neumann 或 Robin。
- 新增可训练、可评估的 `op.NeumannBC` 和 `op.RobinBC`，法向量采用外法向约定。
- 一维预览按左右 Dirichlet 值移动端点并调整纵轴；一维/二维坐标原点仅在零点
  位于当前范围内时显示。周期边界仍未伪装成可运行选项。

## 2026-09-19：新方程自由组合器

- 配置生成器增加“常用模板 / 自由组合”双模式。常用模板扩展为
  Poisson、Helmholtz、非线性 Poisson、热方程、三次反应--扩散、
  Allen--Cahn、对流--扩散和一维 Burgers。
- 自由组合可逐行添加带符号系数、一/二阶导数、时间导数、非线性项和
  源项，实时生成 KaTeX 公式、一/二维预览与可运行 PyTorch 自动微分代码。
- 增加统一 Dirichlet 边界值和动态初值选择；后续已在 2026-09-21 扩展为逐侧
  Dirichlet/Neumann/Robin。

## 2026-09-18：可视化方程设计器预览版

- 配置生成器的“新控制方程”升级为同页配套窗口：可选一维/二维、静态/
  含时间问题，实时查看公式、区域、边界和示例场图例，再与所选改进方法一起
  生成一份完整 Python 代码。图例明确标记为建模预览，不冒充训练结果。
- 修复无时间维度的二维 ``RectangleWithCircularHoles`` 自定义问题在训练后评估阶段
  误访问 ``domain.space`` 的问题；一维/二维与静态/动态四种组合均纳入短跑回归。
- 新增独立的 Blockly 积木式控制方程页面，支持一维/二维、静态/动态问题，实时
  显示 KaTeX 公式、区域边界、结构检查与完整 OpenPINN Python 代码。
- Blockly 13.3.0（Apache-2.0）和 KaTeX 0.18.7（MIT）固定版本并随本地文档分发；
  来源、用途和许可证记录在 ``THIRD_PARTY_NOTICES.md``，页面无需联网。
- 该页面只生成与预检查代码，不替代真实 1--5 步短跑，也不改变现有方法配置器。
- 增加可一键加载的一维非线性反应--扩散示例
  ``u_t - 0.1 u_xx + u^3 = 0``，并提供 ``examples/nonlinear_heat_visual.py``；
  网页生成结构与独立示例均完成 5 步有限输出短跑。

## 2026-09-11：新方程适用方法检查页

- 新增独立的新控制方程编辑/检查页，不把临时方程注册进方法配置器。
- 页面支持静态、动态和一维两材料界面模板，并按“结构上可尝试 / 有条件 /
  不适用”列出全部已注册变体和原因。
- 新增 ``op.check_method_compatibility(PROBLEM)``；复制后运行只打印报告，
  不触发训练。完整回归测试更新为 170 项。
- 配置生成器移除只适用于双隧道的“数值评估网格”输入；双隧道的新运行默认使用
  ``61 x 61`` 数值网格，其他问题的通用页面不再显示无关选项。

## 2026-09-05：按问题生成结果图与补充评价指标

- Poisson 保留一维解/绝对误差，Burgers 保留时间切片，双隧道改为物理坐标下的
  二维孔压场、双圆隧道轮廓和共享 kPa 色标。
- 双隧道额外保存 ``diagnostics.svg``：末时刻 PDE 残差场、孔压剖面、平均孔压消散；
  ``metrics.json`` 补充物理孔压、消散和越界指标。
- 有参考解时统一保存相对 L2、RMSE、MAE、最大绝对误差和 R2。双隧道暂无
  COMSOL 参考场，所以明确不生成虚假的解 L2/RMSE/R2。
- 一维图补充坐标数值，虚线参考解改为最后绘制，防止被几乎重合的预测线遮住。

## 0.2.0 - 2026-09-04：配置生成器与组合方法入口

- 新增网页“配置生成器”，可按问题兼容性选择一个或多个方法、编辑当前方法的
  全部专用超参数，并复制完整 Python 运行代码。
- 新增 ``op.method(...)`` 与 ``op.solve(methods=[...])``；方法按固定功能位置
  组合，同位置冲突、重复选择和非法参数在训练前明确失败。
- 方法参数元数据成为网页、Python 校验和测试共用的单一来源，新增方法时无需
  另维护一份手写网页参数表。
- 逐一执行网页目录中的 18 个具体变体短跑，18/18 通过；另将网页实际复制的
  Fourier + ReLoBRaLo 双方法代码原样运行通过。
- 配置生成器增加“终端输出间隔”，生成代码自动传入 ``training.log_every``；
  201 步网页复制代码按预期在第 1、100、200、201 步输出。
- 控制方程下拉框增加“新控制方程（可编辑一维静态模板）”；生成代码直接包含
  区域、PDE 残差、边界条件、可选解析解与 ``PDEProblem`` 组装，网页复制代码
  已完成 3 步真实训练和全套产物保存。
- 完整回归测试在当时更新为 161 项；2026-09-05 结果可视化与 SVG 云图覆盖修复后为 166 项。
- 修复双隧道 ``result.svg``/``diagnostics.svg`` 的外框填充覆盖色块问题；旧预测文件可直接重绘，无需重新训练。
- 双隧道云图改为从已保存评估网格生成双三次插值显示；原始预测和指标保持不变，图中明确标注插值口径。
- 修复多方法组合时默认结果目录名可能超过文件系统限制的问题；超长名称现保留可读前缀、稳定哈希和时间戳。
- 修复新控制方程与自适应损失权重组合时缺少初始损失项名称的问题；现会依据 PDE 残差和边界/初值条件自动建立单位初始权重。
- checkpoint 恢复统一改用 PyTorch 受限反序列化 ``weights_only=True``，避免来源不明的文件携带任意 pickle 对象；保存、恢复和续训回归测试保持通过。
- 18 个具体方法页面统一增加“超参数怎么改”：自动展示可复制的 ``op.method(...)``、默认值、参数含义和调大/调小的影响；配置生成器同步显示相同解释。

## 0.2.0 - 2026-08-29

- 新建独立 v0.6 项目，保留 v0.5 不动；将临时 `runs/` 与正式 `results/` 分开。
- 新增 `op.solve(problem, method=...)`、紧凑 YAML 和统一 CLI。
- 十种方法均注册为简称，并在训练前执行问题兼容性检查。
- 新增用户控制方程模板与 `module:function` / `file.py:function` 工厂入口。
- 统一入口自动保存误差、时间、预测、checkpoint、结果图、损失图和失败说明。
- 配置拆分为 problems、methods、training、experiments 和 reference_runs。
- 方法入口升级为 ``method`` 家族加 ``variant`` 具体实现；十类方法建立独立目录并保留旧调用兼容。
- 新增/扩展测试至 124 项，并逐一短跑十类方法默认变体、迁移和自定义方程。
- 重做 Sphinx 首页、快速入门、项目结构、自定义方程、方法选择与 API 页面。
- 构建并隔离验证 `openpinn-0.2.0-py3-none-any.whl`。
- 自适应采样新增 RAM ``residual_movement``、WAM ``gradient_movement`` 与带
  逆概率修正的 ``importance_sampling`` 二级变体。
- 损失平衡新增 ``uncertainty_weighting``、``relobralo``、
  ``magnitude_power`` 和完整逐配点 ``pointwise_self_adaptive``；旧
  ``self_adaptive`` 仍作为项级默认变体兼容。
- 七个新增变体分别完成普通 PINN 300 步参考短对比，结果、时间、SVG 与失败/
  限制说明保存于 `results/method_expansions/20260829/`。
- Lagaris et al. (1998) 归入现有硬边界方法的历史依据，不重复添加同功能实现。
- 四个论文参考仓库均未检测到明确许可证；所有新增算法只依据论文公式独立实现。

## 2026-08-28：统一 Python 调用入口

- 新增 `examples/poisson_api_quickstart.py`，按“题目—网络—模型—采样—训练器—训练”顺序展示公开 API；
- 在快速入门网页和 README 中加入公共 Python 调用代码；
- 新增一个两步 smoke 测试，检查该调用链路和有限损失。

## 0.1.2 - 2026-08-26

- 完成 Poisson 5,000 步和 Burgers 10,000 步正式基线，补齐解误差、残差、时间、图、checkpoint 和限制说明。
- 完成十种 PINN 改进方法的统一配置接入、小测试、对照实验、产物保存和 Sphinx 方法页。
- 新增 Burgers Cole--Hopf 评估器与自动可信性检查；参考解仅用于训练后评估。
- 将残差自适应采样的主要演示案例切换为 Burgers，保留 2,048 个候选点为临时评分池，不将其误报为模型参数。
- 迁移学习新增 ``model.trainable_last_linear_layers``，在同一入口比较全网络和最后线性层微调。
- 重新执行十种方法的三随机种子协议：30/30 次方法比较完成，失败 0。
- 统一动态对照图为四个时间切片的参考解、普通 PINN、方法组、分时最大误差和损失曲线。
- 十种方法、两条基线和三种子汇总图均已接入 Sphinx 文档。

## 0.1.1

- 首个可安装的 OpenPINN 版本。
