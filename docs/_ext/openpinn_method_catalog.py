"""Sphinx directives backed by the OpenPINN method documentation catalog."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from docutils import nodes
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.errors import ExtensionError
from sphinx.util.docutils import SphinxDirective


CATALOG_PATH = Path(__file__).resolve().parents[1] / "method_catalog.yaml"
REQUIRED_VARIANT_FIELDS = {
    "variant",
    "title",
    "summary",
    "suitable_for",
    "paper_title",
    "paper_url",
    "code_url",
    "code_license",
    "doc",
}


@lru_cache(maxsize=1)
def load_catalog() -> tuple[dict[str, Any], ...]:
    """Load and validate the documentation catalog once per Sphinx process."""

    with CATALOG_PATH.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    families = payload.get("families", []) if isinstance(payload, dict) else []
    if not families:
        raise ExtensionError(f"Method catalog is empty: {CATALOG_PATH}")

    seen: set[tuple[str, str]] = set()
    normalized: list[dict[str, Any]] = []
    for family in families:
        method = str(family.get("method", "")).strip()
        title = str(family.get("title", "")).strip()
        variants = family.get("variants", [])
        if not method or not title or not variants:
            raise ExtensionError(f"Invalid method family in {CATALOG_PATH}: {family!r}")
        clean_variants: list[dict[str, str]] = []
        for variant in variants:
            missing = REQUIRED_VARIANT_FIELDS.difference(variant)
            if missing:
                raise ExtensionError(
                    f"Catalog entry {method!r} is missing: {', '.join(sorted(missing))}"
                )
            clean = {key: str(variant[key]).strip() for key in REQUIRED_VARIANT_FIELDS}
            key = (method, clean["variant"])
            if key in seen:
                raise ExtensionError(f"Duplicate method catalog entry: {key!r}")
            if not clean["paper_url"].startswith("https://"):
                raise ExtensionError(f"Paper URL must use https for {key!r}")
            if clean["code_url"] and not clean["code_url"].startswith("https://"):
                raise ExtensionError(f"Code URL must use https for {key!r}")
            seen.add(key)
            clean_variants.append(clean)
        normalized.append(
            {"method": method, "title": title, "variants": tuple(clean_variants)}
        )
    return tuple(normalized)


def _plate_fields_html() -> str:
    """Physical ProblemSpec fields for the plate-with-hole PINN problem."""

    return """
<div data-role="plate-fields">
  <div class="configurator-path-fields">
    <label>Physics
      <select data-role="plate-physics"></select>
    </label>
    <label>Young's modulus E
      <input data-role="plate-young" type="number" step="any">
    </label>
    <label>Poisson's ratio nu
      <input data-role="plate-nu" type="number" step="any">
    </label>
    <label>x min<input data-role="plate-x-min" type="number" step="any"></label>
    <label>x max<input data-role="plate-x-max" type="number" step="any"></label>
    <label>y min<input data-role="plate-y-min" type="number" step="any"></label>
    <label>y max<input data-role="plate-y-max" type="number" step="any"></label>
    <label>Hole center x<input data-role="plate-hole-x" type="number" step="any"></label>
    <label>Hole center y<input data-role="plate-hole-y" type="number" step="any"></label>
    <label>Hole radius<input data-role="plate-hole-radius" type="number" step="any"></label>
    <label>Right-edge traction
      <input data-role="plate-traction" type="number" step="any">
    </label>
  </div>
  <div hidden aria-hidden="true">
    <select data-role="plate-solver-path"><option value="pinn">pinn</option></select>
    <select data-role="plate-assumption"><option value="plane_stress">plane_stress</option><option value="plane_strain">plane_strain</option></select>
    <select data-role="plate-method"></select><select data-role="plate-operator-method"></select>
    <input data-role="plate-interior"><input data-role="plate-boundary">
    <input data-role="plate-hidden"><select data-role="plate-activation"><option value="tanh">tanh</option><option value="relu">relu</option><option value="gelu">gelu</option><option value="silu">silu</option></select>
    <input data-role="plate-lr"><input data-role="plate-steps"><input data-role="plate-seed"><input data-role="plate-output">
    <input data-role="plate-network-name"><input data-role="plate-sampler-name"><input data-role="plate-loss-weighting">
    <input data-role="plate-optimizer-name"><input data-role="plate-differentiator-mode">
  </div>
</div>
"""


def _operator_panel_html() -> str:
    return """
<section class="configurator-path-panel" data-role="operator-hooks" hidden>
  <h3>Model 模块 · 算子训练参数与结构</h3>
  <p>物理问题设置与 PINN 共用；模型由上方五模块中的“模型”卡片选择。页面会把完整的 <code>PDEProblem</code> 写入本地脚本，
  并调用 <code>build_operator_config</code> 与 <code>train_operator_problem</code>；
  <code>solve</code> 仍只用于 PINN 模型。浏览器不会执行训练。选项不合法时会显示错误，
  并禁用复制和下载；页面不会擅自替换模型、模式或问题。物理模式使用所选模型的自然求导方式：
  DeepONet 对坐标执行自动微分，FNO 使用网格 PDE 训练器。生成的自定义坐标 PDE 就是实际物理目标，
  不会被已注册的 smoke 问题替换。<code>data_only</code> 和 <code>hybrid</code> 需要本地观测
  CSV 路径，<code>physics_only</code> 无需观测也可运行。工具栏中的训练步数只用于 PINN 模型；
  Operator 的训练步数请在下方字段中设置。</p>
  <div class="configurator-path-fields">
    <label hidden aria-hidden="true">模型<select data-role="operator-model"></select></label>
    <label>训练模式
      <select data-role="operator-training-mode"></select>
    </label>
  </div>
  <p data-role="operator-summary"></p>
  <div class="configurator-path-fields" data-role="operator-parameters"></div>
  <p>内置案例保留在教程中，不放入这个自定义控制方程选择器：
  <a href="poisson.html">Poisson</a>、<a href="burgers.html">Burgers</a> 和
  <a href="tunnel_model1.html">隧道固结</a>。</p>
  <details class="configurator-usage-details" data-role="operator-extensions" hidden aria-hidden="true">
    <summary>高级扩展：导数（derivative）/ 采样器（sampler）/ 优化器（optimizer）</summary>
    <div class="configurator-path-fields" data-role="operator-extension-fields"></div>
  </details>
</section>
"""


def _plate_panel_html() -> str:
    return (
        """
<section class="configurator-path-panel" data-role="plate-panel" hidden>
  <h3>自定义控制方程：二维线弹性结构化设置</h3>
  <p>该编辑器由 Understanding 的线弹性草稿触发，仍属于同一个“自定义控制方程”，不是内置问题或独立求解路径。
  ProblemSpec 与边界数据可在这里核对，模型由下方五模块方法生态选择。
  The browser performs only a contract precheck. The script calls the Python
  Specification Validation Agent <code>validate_specs</code> and then M3
  <code>from_specs</code> before <code>solve</code>. Nothing is solved in the
  browser. Unsupported physics, an operator solver path, an unknown method,
  or an illegal hole produces an explicit warned script with <code>NotImplementedError</code>.
  No fallback spec or fabricated result is written.</p>
"""
        + _plate_fields_html()
        + "</section>"
    )


def _agent_pipeline_html() -> str:
    return """
<section id="configurator-agent-pipeline" class="configurator-agent-pipeline" data-role="agent-pipeline" aria-labelledby="agent-pipeline-title">
  <h2 id="agent-pipeline-title">五 Agent 协同工作流</h2>
  <p>问题理解、求解规划、规范验证、代码构建和诊断评估使用同一组结构化规范。
  本地训练位于代码构建与诊断评估之间，不在浏览器中执行。缺字段、冲突或旧验证未完成会列为问题，
  但不会被静默丢弃。API Key 只保存在当前浏览器输入框中。</p>
  <ol class="configurator-agent-stages" aria-label="五 Agent 阶段">
    <li><a href="#agent-understanding"><strong>1 · 问题理解 Agent</strong><span>自然语言 → ProblemSpec 草稿</span></a></li>
    <li><a href="#agent-planning"><strong>2 · 求解规划 Agent</strong><span>问题规范 → MethodSpec 草稿</span></a></li>
    <li><a href="#configurator-workbench"><strong>3 · 规范验证 Agent</strong><span>完整性、冲突与兼容性检查</span></a></li>
    <li><a href="#configurator-code-title"><strong>4 · 代码构建 Agent</strong><span>确定性导出本地 Python 代码</span></a></li>
    <li><a href="#configurator-diagnostic"><strong>5 · 诊断评估 Agent</strong><span>RunArtifact → 证据与调整建议</span></a></li>
  </ol>
  <label id="agent-understanding">交给问题理解 Agent 的自然语言描述
    <textarea data-agent-role="description" rows="5">这是一个二维平面应力线弹性位移问题。矩形区域 x、y 均为 0 到 2 m；圆孔中心 (1,1)，半径 0.25 m。E=210e9 Pa，nu=0.3。左边固定，右边施加 x 方向 1e6 Pa 牵引，上下边和孔边自由，未知量为 ux、uy。</textarea>
  </label>
  <div class="configurator-agent-grid">
    <label>问题理解接口<input data-agent-role="understanding-base-url" type="url" value="https://api.openai.com/v1"></label>
    <label>问题理解模型<input data-agent-role="understanding-model" value="gpt-4o-mini"></label>
    <label>问题理解 Key<input data-agent-role="understanding-key" type="password" autocomplete="off"></label>
    <label id="agent-planning">求解规划接口<input data-agent-role="planning-base-url" type="url" value="https://api.openai.com/v1"></label>
    <label>求解规划模型<input data-agent-role="planning-model" value="gpt-4o-mini"></label>
    <label>求解规划 Key<input data-agent-role="planning-key" type="password" autocomplete="off"></label>
  </div>
  <div class="configurator-agent-actions">
    <button type="button" data-agent-role="understand">运行问题理解 Agent</button>
    <button type="button" data-agent-role="plan">运行求解规划 Agent</button>
    <button type="button" data-agent-role="build-joint">交给规范验证 Agent</button>
  </div>
  <p data-agent-role="status" role="status" aria-live="polite">也可直接在下面粘贴或编辑两份结构化 JSON，再联合导入。</p>
  <div class="configurator-agent-missing" data-agent-role="understanding-summary" role="status" aria-live="polite" hidden></div>
  <div class="configurator-agent-raw-grid">
    <label>问题理解 Agent · ProblemSpec JSON<textarea data-agent-role="understanding-json" rows="12" spellcheck="false">{}</textarea></label>
    <label>求解规划 Agent · MethodSpec JSON<textarea data-agent-role="planning-json" rows="12" spellcheck="false">{}</textarea></label>
  </div>
  <details><summary>JointAgentHandoff 原始 JSON（保留未映射字段与冲突）</summary>
    <textarea data-agent-role="joint-json" rows="18" spellcheck="false"></textarea>
    <button type="button" data-agent-role="apply-joint-json">应用编辑后的联合 JSON</button>
  </details>
  <details id="configurator-diagnostic" class="configurator-agent-diagnostic" open><summary>5 · 训练后诊断评估 Agent</summary>
    <p>本地运行下载脚本后，直接上传自包含的 <code>run_artifact.json</code> 或 ZIP；无需当前页面指纹或 CodeBuilder 会话。</p>
    <div class="configurator-agent-grid">
      <label>RunArtifact<input data-agent-role="diagnostic-file" type="file" accept=".json,.zip,application/json,application/zip"></label>
      <label>诊断评估接口<input data-agent-role="diagnostic-base-url" type="url" value="https://api.openai.com/v1"></label>
      <label>诊断评估模型<input data-agent-role="diagnostic-model" value="gpt-4o-mini"></label>
      <label>诊断评估 Key<input data-agent-role="diagnostic-key" type="password" autocomplete="off"></label>
    </div>
    <div class="configurator-agent-actions">
      <button type="button" data-agent-role="diagnose" disabled>运行诊断评估 Agent</button>
      <button type="button" data-agent-role="download-diagnostic-html" disabled>下载可视化 HTML 报告</button>
      <button type="button" data-agent-role="download-diagnostic-json" disabled>下载原始 JSON</button>
    </div>
    <p data-agent-role="diagnostic-status" role="status" aria-live="polite">上传 RunArtifact 后可生成双轨诊断、损失曲线和下一轮方法建议。</p>
    <div class="configurator-diagnostic-report" data-agent-role="diagnostic-report-view" hidden></div>
    <details><summary>查看原始诊断 JSON</summary><pre><code data-agent-role="diagnostic-output"></code></pre></details>
  </details>
</section>
"""


def _csv_panel_html(
    *,
    css_class: str,
    prefix: str,
    toggle: str,
    summary: str,
    file_label: str,
    path_label: str,
    path_placeholder: str,
    status: str,
    hidden: bool = False,
) -> str:
    """One CSV entry. Observation and reference use different prefixes so state is not shared."""

    section_hidden = " hidden" if hidden else ""
    return f"""
<section class="{css_class}"{section_hidden}>
  <label class="configurator-reference-toggle"><input data-role="{prefix}-enable" type="checkbox"> {toggle}</label>
  <p>{summary}</p>
  <div class="configurator-reference-fields" data-role="{prefix}-fields" hidden>
    <label>{file_label}
      <input data-role="{prefix}-file" type="file" accept=".csv,.txt,text/csv,text/plain">
    </label>
    <label>{path_label}
      <input data-role="{prefix}-path" type="text" placeholder="{path_placeholder}" spellcheck="false">
    </label>
    <label>坐标列（按模型输入顺序，逗号分隔）
      <input data-role="{prefix}-coordinates" type="text" value="x" spellcheck="false">
    </label>
    <label>单个时间切片的时间值（留空表示文件内有时间列）
      <input data-role="{prefix}-fixed-time" type="number" step="any" placeholder="例如 20">
    </label>
    <label>场值列
      <input data-role="{prefix}-value" type="text" value="u" spellcheck="false">
    </label>
    <label>坐标换算系数（每列一个，逗号分隔）
      <input data-role="{prefix}-coordinate-scales" type="text" value="1" spellcheck="false">
    </label>
    <label>场值换算系数
      <input data-role="{prefix}-value-scale" type="number" value="1" step="any">
    </label>
    <label>模型预测值换算系数
      <input data-role="{prefix}-prediction-scale" type="number" value="1" step="any">
    </label>
    <label>模型输出分量序号（从 0 开始）
      <input data-role="{prefix}-field-index" type="number" value="0" min="0" step="1">
    </label>
    <p data-role="{prefix}-status">{status}</p>
  </div>
</section>
"""


def _observation_panel_html() -> str:
    return _csv_panel_html(
        css_class="configurator-observation-panel",
        prefix="observation",
        toggle="使用本地 CSV 作为训练观测",
        summary=(
            "仅算子路径的 data_only / hybrid 使用这块面板，作为 "
            "<code>train_operator_problem(..., observation=...)</code> 的监督项。"
            "physics_only 隐藏本面板。未启用或未填路径时不能复制或下载。"
            "训后参考在下一块面板单独填写，不会自动当作观测。"
            "文件只由下载后的本地 Python 读取，不会上传。"
        ),
        file_label="选择观测表检查列名",
        path_label="运行时观测表路径（相对项目根目录）",
        path_placeholder="examples/references/observations.csv",
        status="选择文件可预览列名；无列名数值表用 #1、#2… 选列。浏览器无法把文件自动复制进项目；运行前请将观测表放在上方路径。",
        hidden=True,
    )


def _reference_panel_html() -> str:
    return _csv_panel_html(
        css_class="configurator-reference-panel",
        prefix="reference",
        toggle="训练后对比独立参考场",
        summary=(
            "可选。PINN 在训练结束后调用 <code>evaluate_reference_csv</code>；"
            "算子路径调用 <code>evaluate_operator_reference_csv</code>。"
            "参考表不进入训练损失。未启用则跳过评估，不报错。"
            "算子路径的训练观测在上一块面板单独填写，这里的路径不会被当作观测。"
            "文件只由下载后的本地 Python 读取，不会上传。"
        ),
        file_label="选择参考表检查列名",
        path_label="运行时参考表路径（相对项目根目录）",
        path_placeholder="examples/references/my_field.csv",
        status="选择文件可预览列名；无列名数值表用 #1、#2… 选列。浏览器无法把文件自动复制进项目；运行前请将文件放在上方路径。",
    )


def _designer_local_paths_html() -> str:
    return """
<section class="problem-designer-local" data-role="local-paths">
  <h2>本地 Operator 脚本</h2>
  <p>选择已注册的模型和训练模式，复制或下载 <code>.py</code> 文件后在本地运行。
  本页面不会调用 <code>train_operator</code> 或 <code>solve</code>。
  校验失败时仍可复制和下载带显式 <code>NotImplementedError</code> 的问题说明脚本；
  页面不会替换为其他脚本或伪造训练结果。</p>
  <label>脚本
    <select data-role="local-script"></select>
  </label>
  <label>训练模式
    <select data-role="operator-training-mode"></select>
  </label>
  <div data-role="designer-operator" hidden>
    <p data-role="operator-summary"></p>
    <div class="configurator-path-fields" data-role="operator-parameters"></div>
    <details data-role="operator-extensions">
      <summary>高级扩展</summary>
      <div class="configurator-path-fields" data-role="operator-extension-fields"></div>
    </details>
  </div>
  <p data-role="local-message" role="status"></p>
  <div class="problem-designer-actions">
    <button type="button" data-role="local-copy" disabled>复制 .py</button>
    <button type="button" data-role="local-download" disabled>下载 .py</button>
  </div>
  <pre class="problem-designer-local-code"><code data-role="local-code"></code></pre>
</section>
"""


def catalog_entries() -> tuple[dict[str, str], ...]:
    """Return flattened entries for validation and tests."""

    return tuple(
        {"method": family["method"], "family_title": family["title"], **variant}
        for family in load_catalog()
        for variant in family["variants"]
    )


def validate_catalog_contract(app) -> None:
    """Fail the docs build when runtime variants and documentation diverge."""

    from openpinn.method_registry import default_method_registry

    registry = default_method_registry()
    expected = {
        (family.name, variant.name)
        for family in registry.specs()
        for variant in registry.variant_specs(family.name)
    }
    entries = catalog_entries()
    actual = {(entry["method"], entry["variant"]) for entry in entries}
    if expected != actual:
        missing = sorted(expected - actual)
        stale = sorted(actual - expected)
        raise ExtensionError(
            "Method catalog does not match the runtime registry; "
            f"missing={missing}, stale={stale}"
        )

    docs_dir = Path(app.confdir)
    for entry in entries:
        page = docs_dir / f"{entry['doc']}.rst"
        if not page.is_file():
            raise ExtensionError(
                f"Method detail page does not exist for "
                f"{entry['method']}/{entry['variant']}: {page}"
            )


def _doc_link(text: str, target: str, *, literal: bool = False) -> nodes.Node:
    link = addnodes.pending_xref(
        "",
        refdomain="std",
        reftype="doc",
        reftarget=target,
        refexplicit=True,
    )
    link += nodes.literal(text, text) if literal else nodes.Text(text)
    return link


def _external_link(text: str, url: str) -> nodes.reference:
    return nodes.reference(text, text, refuri=url, internal=False)


def _entry(*children: nodes.Node, classes: list[str] | None = None) -> nodes.entry:
    cell = nodes.entry(classes=classes or [])
    paragraph = nodes.paragraph()
    paragraph.extend(children)
    cell += paragraph
    return cell


def _header(text: str) -> nodes.entry:
    cell = nodes.entry()
    cell += nodes.paragraph(text=text)
    return cell


class MethodCatalogDirective(SphinxDirective):
    """Render every registered shorthand as one grouped lookup table."""

    has_content = False
    optional_arguments = 0
    option_spec = {"caption": directives.unchanged}

    def run(self) -> list[nodes.Node]:
        table = nodes.table(classes=["method-catalog-table"])
        caption = self.options.get("caption", "OpenPINN 方法简称查询表")
        table += nodes.title(text=caption)

        tgroup = nodes.tgroup(cols=5)
        table += tgroup
        for width in (18, 25, 22, 42, 16):
            tgroup += nodes.colspec(colwidth=width)

        thead = nodes.thead()
        header_row = nodes.row()
        for text in ("method", "variant（点击查看）", "中文名称", "大致适用方向", "原文"):
            header_row += _header(text)
        thead += header_row
        tgroup += thead

        tbody = nodes.tbody()
        for family in load_catalog():
            variants = family["variants"]
            for index, variant in enumerate(variants):
                row = nodes.row()
                if index == 0:
                    family_cell = _entry(
                        _doc_link(family["method"], variants[0]["doc"], literal=True),
                        classes=["method-family-cell"],
                    )
                    if len(variants) > 1:
                        family_cell["morerows"] = len(variants) - 1
                    row += family_cell
                row += _entry(
                    _doc_link(variant["variant"], variant["doc"], literal=True),
                    classes=["method-variant-cell"],
                )
                row += _entry(nodes.Text(variant["title"]))
                row += _entry(nodes.Text(variant["suitable_for"]))
                row += _entry(_external_link("论文原文", variant["paper_url"]))
                tbody += row
        tgroup += tbody
        return [table]


class MethodSummaryDirective(SphinxDirective):
    """Render the consistent top-of-page summary for one concrete variant."""

    required_arguments = 2
    has_content = False

    def run(self) -> list[nodes.Node]:
        from openpinn.method_options import method_parameter_schema

        method, variant = self.arguments
        try:
            item = next(
                entry
                for entry in catalog_entries()
                if entry["method"] == method and entry["variant"] == variant
            )
        except StopIteration as exc:
            raise self.error(f"Unknown method catalog entry: {method}/{variant}") from exc

        card = nodes.container(classes=["method-summary-card"])
        card += nodes.rubric(text="方法速览")

        fields = nodes.field_list()
        call = nodes.paragraph()
        call += nodes.literal(method, f'method="{method}"')
        call += nodes.Text("，")
        call += nodes.literal(variant, f'variant="{variant}"')
        fields += self._field("调用简称", call)
        fields += self._field("简单介绍", nodes.paragraph(text=item["summary"]))
        fields += self._field("适用方向", nodes.paragraph(text=item["suitable_for"]))

        paper = nodes.paragraph()
        paper += _external_link(item["paper_title"], item["paper_url"])
        fields += self._field("论文原文", paper)

        code = nodes.paragraph()
        if item["code_url"]:
            code += _external_link("参考代码", item["code_url"])
            code += nodes.Text("；")
        code += nodes.Text(item["code_license"])
        fields += self._field("代码与许可", code)

        card += fields
        card += nodes.rubric(text="超参数怎么改")

        parameters = method_parameter_schema(method, variant)
        call_lines = self._method_call(method, variant, parameters)
        code_block = nodes.literal_block(call_lines, call_lines)
        code_block["language"] = "python"
        card += code_block

        if parameters:
            table = nodes.table(classes=["method-parameter-table"])
            tgroup = nodes.tgroup(cols=3)
            table += tgroup
            for width in (24, 18, 58):
                tgroup += nodes.colspec(colwidth=width)

            thead = nodes.thead()
            header = nodes.row()
            for text in ("参数", "默认值", "含义与修改影响"):
                header += _header(text)
            thead += header
            tgroup += thead

            tbody = nodes.tbody()
            for name, schema in parameters.items():
                row = nodes.row()
                row += _entry(
                    nodes.Text(f"{schema['label']} "),
                    nodes.literal(name, name),
                )
                default = self._option_literal(schema)
                row += _entry(nodes.literal(default, default))
                row += _entry(
                    nodes.Text(f"{schema['description']} {schema['effect']}")
                )
                tbody += row
            tgroup += tbody
            card += table
        else:
            card += nodes.paragraph(
                text=(
                    "这个变体当前没有方法专用超参数。网络层数、训练点、训练步数和 "
                    "log_every 等通用参数仍可在 solve(...) 中修改。"
                )
            )

        guidance = nodes.paragraph()
        guidance += nodes.Text(
            "建议先保留默认值完成短跑，再一次只改一个参数并记录误差与时间。也可以在 "
        )
        guidance += _doc_link("配置生成器", "method_configurator")
        guidance += nodes.Text(" 中选择方法后直接编辑这些参数；调大或调小都不保证精度一定提高。")
        card += guidance
        return [card]

    @staticmethod
    def _option_literal(schema: dict[str, Any]) -> str:
        value = schema.get("default")
        if value is None and schema.get("required"):
            value = schema.get("placeholder") or "请填写"
        return repr(value)

    @classmethod
    def _method_call(
        cls,
        method: str,
        variant: str,
        parameters: dict[str, dict[str, Any]],
    ) -> str:
        if not parameters:
            return f'op.method("{method}", "{variant}")'
        lines = ["op.method(", f'    "{method}",', f'    "{variant}",']
        for name, schema in parameters.items():
            lines.append(f"    {name}={cls._option_literal(schema)},")
        lines.append(")")
        return "\n".join(lines)

    @staticmethod
    def _field(name: str, body_node: nodes.Node) -> nodes.field:
        field = nodes.field()
        field += nodes.field_name(text=name)
        body = nodes.field_body()
        body += body_node
        field += body
        return field


class ConfiguratorGuideDirective(SphinxDirective):
    """Show the short clickable user journey before the workbench."""

    def run(self):
        from configurator_flow import configurator_flow

        return [nodes.raw("", configurator_flow(), format="html")]


ECOSYSTEM_REFERENCES: dict[str, tuple[tuple[str, str], ...]] = {
    "pointwise_mlp": (("Physics-informed neural networks", "https://doi.org/10.1016/j.jcp.2018.10.045"),),
    "modified_mlp": (("Understanding and mitigating gradient flow pathologies in PINNs", "https://doi.org/10.1137/20M1318043"),),
    "spectral_embedding": (("On the eigenvector bias of Fourier feature networks", "https://doi.org/10.1016/j.cma.2021.113938"),),
    "hard_constraints": (("Exact imposition of boundary conditions with distance functions", "https://doi.org/10.1016/j.cma.2021.114333"),),
    "domain_decomposition": (("XPINNs: generalized space-time domain decomposition", "https://doi.org/10.4208/cicp.OA-2020-0164"),),
    "branch_trunk": (("Learning nonlinear operators via DeepONet", "https://doi.org/10.1038/s42256-021-00302-5"),),
    "graph_operators": (("Neural operator: graph kernel network for PDEs", "https://arxiv.org/abs/2003.03485"),),
    "spectral_operators": (("Fourier neural operator for parametric PDEs", "https://openreview.net/forum?id=c8P9NQVtmnO"),),
    "emerging_bases": (("Kolmogorov–Arnold-informed neural network", "https://doi.org/10.1016/j.cma.2024.117518"),),
    "non_adaptive": (("A comprehensive study of non-adaptive and residual-based adaptive sampling", "https://doi.org/10.1016/j.cma.2022.115671"),),
    "residual_adaptive": (("A comprehensive study of non-adaptive and residual-based adaptive sampling", "https://doi.org/10.1016/j.cma.2022.115671"),),
    "gradient_adaptive": (("Residual/gradient-based adaptive sampling for sharp solutions", "https://doi.org/10.1007/s10483-023-2994-7"),),
    "uncertainty_adaptive": (("Improved training of PINNs with model ensembles", "https://doi.org/10.1109/IJCNN54540.2023.10191923"),),
    "learned_adaptive": (("DAS-PINNs: a deep adaptive sampling method", "https://doi.org/10.1016/j.jcp.2022.111868"),),
    "temporal_curriculum": (("Respecting causality for training PINNs", "https://doi.org/10.1016/j.cma.2024.116813"),),
    "fixed_composite": (("Physics-informed neural networks", "https://doi.org/10.1016/j.jcp.2018.10.045"),),
    "gradient_balanced": (("Understanding and mitigating gradient flow pathologies in PINNs", "https://doi.org/10.1137/20M1318043"),),
    "ntk_balanced": (("When and why PINNs fail to train: an NTK perspective", "https://doi.org/10.1016/j.jcp.2021.110768"),),
    "uncertainty_balanced": (("Inverse Dirichlet weighting enables reliable PINN training", "https://doi.org/10.1088/2632-2153/ac3712"),),
    "pointwise_adaptive": (("Self-adaptive physics-informed neural networks", "https://doi.org/10.1016/j.jcp.2022.111722"),),
    "multi_objective": (("Multi-objective loss balancing for physics-informed deep learning", "https://doi.org/10.1016/j.cma.2025.117914"),),
    "causal_curriculum": (("Respecting causality for training PINNs", "https://doi.org/10.1016/j.cma.2024.116813"),),
    "first_order": (("Adam: a method for stochastic optimization", "https://arxiv.org/abs/1412.6980"),),
    "quasi_newton": (("On the limited memory BFGS method for large scale optimization", "https://doi.org/10.1007/BF01589116"),),
    "second_order": (("Gauss–Newton natural gradient descent for physics-informed CFD", "https://doi.org/10.1016/j.compfluid.2025.106955"),),
    "hybrid_multistage": (("Unveiling the optimization process of PINNs", "https://doi.org/10.1016/j.jcp.2024.113656"),),
    "coordinate_ad": (("Automatic differentiation in machine learning: a survey", "https://jmlr.org/papers/v18/17-468.html"),),
    "numerical": (("CAN-PINN: coupled automatic–numerical differentiation", "https://doi.org/10.1016/j.cma.2022.114909"),),
    "spectral_grid": (("Physics-informed neural operator", "https://doi.org/10.1145/3648506"),),
    "discrete_grid": (("CAN-PINN: coupled automatic–numerical differentiation", "https://doi.org/10.1016/j.cma.2022.114909"),),
    "weak_variational": (("hp-VPINNs: variational PINNs with domain decomposition", "https://doi.org/10.1016/j.cma.2020.113547"),),
    "hybrid_ad_numerical": (("CAN-PINN: coupled automatic–numerical differentiation", "https://doi.org/10.1016/j.cma.2022.114909"),),
}


ECOSYSTEM_COMPARISONS = {
    "model": "在同一 Poisson 或 Burgers 问题上固定配点、损失、优化、步数和随机种子，只替换模型或输入表示；同时报告误差、残差、参数量和用时。",
    "sampler": "固定网络与总配点预算，在 Burgers 问题上比较 uniform 与目标采样器；记录最终点分布，避免把增加点数误写成采样策略收益。",
    "loss": "固定 Burgers 的网络、采样和优化，只替换损失组装；同时报告 PDE、边界、初值和参考解误差，不能只比较 total loss。",
    "optimizer": "固定 Poisson 的初始权重、网络和配点，比较优化阶段；Adam 与 Adam→L-BFGS 的计算量不同，需同时记录步数和墙钟时间。",
    "derivative": "在具有解析导数的周期场上先比较导数误差，再在周期 Poisson 参数族上固定 FNO 与损失，只替换有限差分或谱微分；非周期问题不得直接外推。",
}


def _ecosystem_yaml_example(axis: str, family: dict[str, Any]) -> str:
    """Return a compact configuration sketch using the live canonical id."""

    implementation = family["implementations"][0]
    qualified_kind, separator, name = implementation.partition(":")
    if not separator:
        qualified_kind, name = axis, implementation
    if qualified_kind == "operator":
        return f"model:\n  name: {name}\ntraining_mode: physics_only"
    if qualified_kind == "feature":
        return f"model:\n  name: mlp\n  feature:\n    name: {name}"
    if qualified_kind == "constraint":
        return f"model:\n  name: mlp\n  output_transform:\n    name: {name}"
    section = {"sampler": "sampling", "derivative": "differentiator"}.get(axis, axis)
    key = {"loss": "weighting", "derivative": "mode"}.get(axis, "name")
    return f"{section}:\n  {key}: {name}"


class EcosystemMethodDirective(SphinxDirective):
    """Render one paper-aligned method-family reference page."""

    required_arguments = 2
    has_content = False

    def run(self) -> list[nodes.Node]:
        from openpinn.method_ecosystem import method_ecosystem_payload

        axis_name, family_slug = self.arguments
        axis = next(
            (item for item in method_ecosystem_payload()["axes"] if item["kind"] == axis_name),
            None,
        )
        family = next(
            (item for item in (axis or {}).get("families", []) if item["slug"] == family_slug),
            None,
        )
        if axis is None or family is None:
            raise self.error(f"Unknown method ecosystem family: {axis_name}/{family_slug}")

        result: list[nodes.Node] = [
            nodes.paragraph(text=axis["description"]),
            nodes.rubric(text="作用与适用场景"),
            nodes.paragraph(text=family["setting"]),
            nodes.rubric(text="方法形式"),
            nodes.paragraph(text=family["forms"]),
            nodes.rubric(text="OpenPINN 内置实现"),
        ]
        implementations = nodes.bullet_list()
        for implementation in family["implementations"]:
            implementations += nodes.list_item(
                "", nodes.paragraph(text=f"{implementation}（默认组件注册表）")
            )
        result.append(implementations)
        result.extend(
            [
                nodes.rubric(text="实际调用边界"),
                nodes.paragraph(text=family["boundary"]),
                nodes.rubric(text="网页端使用"),
                nodes.paragraph(
                    text=(
                        "配置器下拉里出现的 PINN 组件会写入 ECOSYSTEM_OVERRIDES，并由 op.solve 创建。"
                        "DeepONet 与 FNO 走算子训练。只有网格 partial、没有坐标 grad/hessian 的微分后端不能用于强形式 solve()。"
                    )
                ),
                nodes.rubric(text="最小配置示例"),
                nodes.literal_block(
                    _ecosystem_yaml_example(axis_name, family),
                    _ecosystem_yaml_example(axis_name, family),
                    language="yaml",
                ),
                nodes.rubric(text="简单比较示例"),
                nodes.paragraph(text=ECOSYSTEM_COMPARISONS[axis_name]),
                nodes.rubric(text="验证与限制"),
                nodes.paragraph(
                    text=(
                        "内置状态表示实现、注册和最小执行链路已接通，不等同于对所有方程都更优。"
                        "正式比较时应固定采样预算、网络规模、训练步数与随机种子，并记录不收敛或负结果。"
                    )
                ),
            ]
        )
        result.append(nodes.rubric(text="参考文献"))
        references = nodes.bullet_list()
        for title, url in ECOSYSTEM_REFERENCES[family_slug]:
            paragraph = nodes.paragraph()
            paragraph += nodes.reference(text=title, refuri=url)
            references += nodes.list_item("", paragraph)
        result.append(references)
        return result


class MethodConfiguratorDirective(SphinxDirective):
    """Render the interactive method and hyperparameter code generator."""

    has_content = False
    optional_arguments = 0

    def run(self) -> list[nodes.Node]:
        from openpinn.method_options import (
            METHOD_PARAMETER_SCHEMAS,
            METHOD_SLOTS,
        )
        from openpinn.method_registry import default_method_registry
        from openpinn.method_ecosystem import method_ecosystem_payload
        from openpinn.operator_protocol import attach_operator_hooks
        from openpinn.physics import load_physics_catalog
        from openpinn.scriptgen import browser_payload

        registry = default_method_registry()
        catalog_lookup = {
            (entry["method"], entry["variant"]): entry
            for entry in catalog_entries()
        }
        families: list[dict[str, Any]] = []
        for family in registry.specs():
            variants: list[dict[str, Any]] = []
            for variant in registry.variant_specs(family.name):
                key = (family.name, variant.name)
                catalog = catalog_lookup[key]
                variants.append(
                    {
                        "variant": variant.name,
                        "title": variant.title,
                        "summary": catalog["summary"],
                        "suitable_for": catalog["suitable_for"],
                        "compatible_with": list(variant.compatible_with),
                        "parameters": METHOD_PARAMETER_SCHEMAS[key],
                    }
                )
            families.append(
                {
                    "method": family.name,
                    "title": family.title,
                    "description": family.description,
                    "slots": list(METHOD_SLOTS[family.name]),
                    "variants": variants,
                }
            )

        script_payload = browser_payload()
        payload = attach_operator_hooks({
            "problems": [{
                "name": "custom_equation_template",
                "title": "自定义控制方程",
                "families": ["custom"],
                "template": "custom_visual",
            }],
            "families": families,
            "physics_catalog": load_physics_catalog(),
            "method_ecosystem": method_ecosystem_payload(),
        })
        payload["operator_hooks"] = script_payload["operator_hooks"]
        payload["operator_choices"] = script_payload["operator_choices"]
        payload["training_modes"] = script_payload["training_modes"]
        payload["operator_extensions"] = script_payload["operator_extensions"]
        payload["plate_contract"] = script_payload["plate_contract"]
        encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
        path_panels = _plate_panel_html()
        html = f"""
<div id="configurator-workbench" class="openpinn-configurator" data-openpinn-configurator>
  {_agent_pipeline_html()}
  <div class="configurator-section-heading">
    <h2>1. 物理问题设置</h2>
    <p>所有问题都在同一个自定义控制方程中配置；不再提供内置特定问题或单独的求解路径。</p>
  </div>
  <div class="configurator-toolbar">
    <div id="configurator-problem"><span>控制方程</span><strong>自定义控制方程</strong>
      <select data-role="problem" hidden aria-hidden="true"></select>
    </div>
    <label id="configurator-budget">训练步数
      <input data-role="steps" type="number" min="1" step="1" value="1000">
    </label>
    <label>终端输出间隔
      <input data-role="log-every" type="number" min="1" step="1" value="100">
    </label>
    <label>随机种子
      <input data-role="seed" type="number" min="0" step="1" value="0">
    </label>
    <label>结果文件夹
      <input data-role="output-root" type="text" value="runs">
    </label>
  </div>
  <p class="configurator-navigation-status" data-role="guide-navigation-status" role="status" aria-live="polite" hidden></p>
  {_observation_panel_html()}
  {_reference_panel_html()}
  <section class="configurator-custom-problem" data-role="custom-problem-panel"
           aria-labelledby="configurator-custom-problem-title">
    <div class="configurator-custom-header">
      <div>
        <h2 id="configurator-custom-problem-title">通用物理问题与实时图例</h2>
        <p>统一编辑几何、数据、物理场与方程；图例是建模预览，不是训练结果。</p>
      </div>
      <span>与下方方法选择共同生成一份代码</span>
    </div>
    <p class="configurator-understanding-prefill" data-role="understanding-prefill-status"
       role="status" aria-live="polite" hidden></p>
    {path_panels}
    <div id="configurator-geometry" class="configurator-custom-fields">
      <label>问题名称
        <input data-role="custom-name" type="text" value="my_new_equation" spellcheck="false">
      </label>
      <label hidden aria-hidden="true">问题类型
        <select data-role="custom-kind">
          <option value="static">静态方程</option>
          <option value="dynamic">含时间方程</option>
        </select>
      </label>
      <label hidden aria-hidden="true">空间维度
        <select data-role="custom-dimension">
          <option value="1">一维</option>
          <option value="2">二维</option>
        </select>
      </label>
      <label data-role="custom-geometry-field">几何类型（空间维度联动）
        <select data-role="custom-geometry-type">
          <option value="rectangle">矩形（直接填写范围）</option>
          <option value="circle">圆形（圆心 + 半径，保持真圆）</option>
          <option value="ellipse">椭圆（圆心 + 半轴 a/b + 旋转角）</option>
          <option value="rounded_rectangle">圆角矩形（范围 + 圆角半径）</option>
          <option value="spline">可编辑闭合曲线（控制点 + 张力）</option>
          <option value="polygon">导入多边形文件（识别图片）</option>
          <option value="dxf">导入 DXF 图纸（COMSOL/CAD）</option>
        </select>
      </label>
      <input data-role="custom-equation-mode" type="hidden" value="expression">
      <span data-role="custom-template-field" hidden aria-hidden="true">
        <select data-role="custom-template" tabindex="-1"></select>
      </span>
      <label data-role="custom-coordinate-field">x 下限<input data-role="custom-x-min" type="number" value="-1" step="0.1"></label>
      <label data-role="custom-coordinate-field">x 上限<input data-role="custom-x-max" type="number" value="1" step="0.1"></label>
      <label data-role="custom-y-field" data-coordinate-field>y 下限
        <input data-role="custom-y-min" type="number" value="-1" step="0.1">
      </label>
      <label data-role="custom-y-field" data-coordinate-field>y 上限
        <input data-role="custom-y-max" type="number" value="1" step="0.1">
      </label>
      <label data-role="custom-circle-field" hidden>圆心 x
        <input data-role="custom-circle-center-x" type="number" value="0" step="any">
      </label>
      <label data-role="custom-circle-field" hidden>圆心 y
        <input data-role="custom-circle-center-y" type="number" value="0" step="any">
      </label>
      <label data-role="custom-circle-field" hidden>圆半径 R
        <input data-role="custom-circle-radius" type="number" value="1" min="0" step="any">
      </label>
      <label data-role="custom-ellipse-field" hidden>椭圆中心 x
        <input data-role="custom-ellipse-center-x" type="number" value="0" step="any">
      </label>
      <label data-role="custom-ellipse-field" hidden>椭圆中心 y
        <input data-role="custom-ellipse-center-y" type="number" value="0" step="any">
      </label>
      <label data-role="custom-ellipse-field" hidden>半轴 a（局部 x 方向）
        <input data-role="custom-ellipse-axis-a" type="number" value="2" min="0" step="any">
      </label>
      <label data-role="custom-ellipse-field" hidden>半轴 b（局部 y 方向）
        <input data-role="custom-ellipse-axis-b" type="number" value="1" min="0" step="any">
      </label>
      <label data-role="custom-ellipse-field" hidden>旋转角 θ（度，逆时针）
        <input data-role="custom-ellipse-rotation" type="number" value="0" step="any">
      </label>
      <label data-role="custom-rounded-field" hidden>圆角半径 r
        <input data-role="custom-corner-radius" type="number" value="0.25" min="0" step="any">
      </label>
      <label data-role="custom-spline-field" hidden>曲线张力（0 最圆滑，1 最贴近控制折线）
        <input data-role="custom-spline-tension" type="number" value="0.2" min="0" max="1" step="0.05">
      </label>
      <label data-role="custom-time-field">t 下限
        <input data-role="custom-t-min" type="number" value="0" step="0.1">
      </label>
      <label data-role="custom-time-field">t 上限
        <input data-role="custom-t-max" type="number" value="1" step="0.1">
      </label>
      <label id="configurator-initial" data-role="custom-initial-field">动态问题初值
        <select data-role="custom-initial-preset">
          <option value="sine">正弦场（边界为 0）</option>
          <option value="gaussian">高斯场</option>
          <option value="zero">零初值</option>
          <option value="constant">常数初值（按未知量单位）</option>
        </select>
      </label>
      <label data-role="custom-initial-value-field" hidden>初始值（整个区域）
        <input data-role="custom-initial-value" type="number" value="0" step="any">
      </label>
      <label data-role="custom-parameter-diffusivity">扩散系数 D
        <input data-role="custom-diffusivity" type="number" value="0.1" min="0" step="any">
      </label>
      <label data-role="custom-parameter-reaction">反应系数 r
        <input data-role="custom-reaction" type="number" value="1.0" step="any">
      </label>
      <label data-role="custom-parameter-wave-number">波数 k
        <input data-role="custom-wave-number" type="number" value="2.0" min="0" step="any">
      </label>
      <label data-role="custom-parameter-advection-x">x 方向速度 vₓ
        <input data-role="custom-advection-x" type="number" value="1.0" step="any">
      </label>
      <label data-role="custom-parameter-advection-y">y 方向速度 vᵧ
        <input data-role="custom-advection-y" type="number" value="0.5" step="any">
      </label>
      <label data-role="custom-parameter-viscosity">黏性系数 ν
        <input data-role="custom-viscosity" type="number" value="0.01" min="0" step="any">
      </label>
    </div>
    <div class="configurator-polygon-import" data-role="custom-polygon-import" hidden>
      <div>
        <strong>导入二维多边形</strong>
        <p>支持 CSV、JSON、GeoJSON，以及 PNG/JPG 高对比填充色块；自动识别外轮廓和孔洞。文件只在浏览器本地读取，轮廓坐标会直接嵌入生成的 Python 代码。</p>
      </div>
      <label>几何文件
        <input data-role="custom-polygon-file" type="file" accept=".csv,.json,.geojson,.png,.jpg,.jpeg,text/csv,application/json,application/geo+json,image/png,image/jpeg">
      </label>
      <div class="configurator-image-options" data-role="custom-image-options" hidden>
        <strong>图片轮廓识别设置</strong>
        <label>前景色
          <select data-role="custom-image-foreground">
            <option value="auto">自动判断</option>
            <option value="dark">深色区域</option>
            <option value="light">浅色区域</option>
          </select>
        </label>
        <label>二值化方式
          <select data-role="custom-image-threshold-mode">
            <option value="auto">自动区分背景与填充区域</option>
            <option value="manual">手动设置阈值</option>
          </select>
        </label>
        <label data-role="custom-image-threshold-field" hidden>手动阈值
          <input data-role="custom-image-threshold" type="number" value="128" min="1" max="254" step="1">
        </label>
        <label>识别分辨率
          <select data-role="custom-image-resolution">
            <option value="384">快速（384 px）</option>
            <option value="768">高（768 px）</option>
            <option value="1024" selected>推荐（1024 px，保留细节）</option>
            <option value="1536">精细（1536 px，较慢）</option>
            <option value="2048">原图优先（最长边上限 2048 px，较慢）</option>
          </select>
        </label>
        <label>曲线几何识别
          <select data-role="custom-image-shape-mode">
            <option value="auto">自动：有连续曲率用样条，否则用拐角</option>
            <option value="spline">强制保留为可编辑曲线</option>
            <option value="polygon">始终保留为多边形</option>
            <option value="circle">强制拟合纯圆</option>
            <option value="rounded_rectangle">强制拟合圆角矩形</option>
          </select>
        </label>
        <label>关键拐角简化（像素）
          <input data-role="custom-image-simplify" type="number" value="2" min="0.25" max="5" step="0.25">
        </label>
        <label>x 下限<input data-role="custom-image-x-min" type="number" value="0" step="any"></label>
        <label>x 上限<input data-role="custom-image-x-max" type="number" value="1" step="any"></label>
        <label>y 下限<input data-role="custom-image-y-min" type="number" value="0" step="any"></label>
        <label>y 上限<input data-role="custom-image-y-max" type="number" value="1" step="any"></label>
        <button type="button" data-role="custom-image-reprocess">按当前设置重新识别</button>
        <p>自动分色会对比背景与主要填充色，可识别灰色或彩色区域、孔洞及描线；透明背景使用透明度识别。识别不准时可切换手动阈值。更高分辨率有利于保留小孔和细边，但会更慢，也不能恢复原图没有的细节。曲线自动模式会检查连续的小角度转向：检测到平滑段时生成可编辑闭合样条；尖角仍用多边形。默认 2 px 简化像素阶梯，带孔轮廓继续使用多边形。</p>
      </div>
      <details>
        <summary>文件格式示例</summary>
        <pre>CSV: ring,x,y（ring 使用 outer、hole_1…）
JSON: {{"outer": [[x,y], ...], "holes": [[[x,y], ...]]}}
GeoJSON: Polygon 或只包含一个 Polygon 的 Feature/FeatureCollection
PNG/JPG: 高对比填充色块，可设前景颜色、阈值与物理坐标范围</pre>
      </details>
    </div>
    <div class="configurator-polygon-import" data-role="custom-dxf-import" hidden>
      <div>
        <strong>导入二维 DXF 几何</strong>
        <p>选择二维 ASCII DXF 文件。支持闭合多段线、圆、椭圆，以及能无分叉拼成闭环的 LINE、ARC、开放多段线和带完整控制点/节点的 SPLINE；可识别一条外轮廓及其孔洞。四顶点轴对齐矩形＋全部已识别圆孔，会生成解析圆孔几何；其他轮廓保留多边形近似，识别后显示训练几何说明。请核对单位、形状和边界。原生 .mph/.dwg 文件不能直接上传。</p>
      </div>
      <label>DXF 文件
        <input data-role="custom-dxf-file" type="file" accept=".dxf,application/dxf,image/vnd.dxf">
      </label>
      <label>坐标换算系数（图纸单位→模型单位）
        <input data-role="custom-dxf-unit-scale" type="number" value="1" min="0" step="any">
      </label>
      <details>
        <summary>支持范围与导出提示</summary>
        <p>COMSOL Geometry → Export 可选 DXF；CAD 请导出 ASCII DXF。LINE/ARC/SPLINE 等必须首尾闭合且无分叉。DXF 混合路径中的圆弧、椭圆弧及样条会采样为短边；只有拟合点、没有完整控制点/节点的 SPLINE、块引用、3D 几何及多个互不相连区域仍会明确报错。若图纸是毫米而模型用米，换算系数填 0.001；更改系数会重新读取所选文件。</p>
      </details>
    </div>
    <p class="configurator-polygon-status" data-role="custom-polygon-status" hidden>尚未导入几何。</p>
    <p class="configurator-curved-note" data-role="custom-curved-note" hidden>
      这是曲线几何：闭合样条可拖动控制点并用张力调节弯曲程度；圆形始终保持真圆；椭圆用 a、b 和旋转角控制，a=b 时为圆；圆角矩形的 r=0 是普通矩形，r 越大转角越圆。
    </p>
    <div class="configurator-polygon-editor" data-role="custom-polygon-editor" hidden>
      <div class="configurator-polygon-editor-heading">
        <div>
          <strong>识别后轮廓编辑器</strong>
          <p>多边形顶点可拖动或在下表修改；识别为圆形的 DXF 孔洞请直接调整圆心与半径，不必逐点编辑圆弧采样点。</p>
        </div>
        <label class="configurator-ring-picker">选择要编辑的外轮廓 / 孔洞<select data-role="custom-polygon-ring"></select><span>选择圆形孔洞后，可修改圆心、半径或直径。</span></label>
      </div>
      <div class="configurator-polygon-transform" data-role="custom-dxf-circle-editor" hidden>
        <label>圆心 x<input data-role="custom-dxf-circle-x" type="number" step="any"></label>
        <label>圆心 y<input data-role="custom-dxf-circle-y" type="number" step="any"></label>
        <label>半径 r<input data-role="custom-dxf-circle-radius" type="number" min="0" step="any"></label>
        <label>直径 D<input data-role="custom-dxf-circle-diameter" type="number" min="0" step="any"></label>
        <p>圆孔统一设置一组边界条件。修改圆心或半径会更新预览和生成代码。四顶点轴对齐矩形外框＋全部已识别圆孔使用解析圆孔训练；其他轮廓保留多边形近似。实际路径见上方“训练几何”提示。</p>
      </div>
      <div class="configurator-polygon-transform">
        <label>旋转角度（°）<input data-role="custom-polygon-rotate" type="number" value="0" step="any"></label>
        <label>X 缩放<input data-role="custom-polygon-scale-x" type="number" value="1" step="any"></label>
        <label>Y 缩放<input data-role="custom-polygon-scale-y" type="number" value="1" step="any"></label>
        <label>X 平移<input data-role="custom-polygon-translate-x" type="number" value="0" step="any"></label>
        <label>Y 平移<input data-role="custom-polygon-translate-y" type="number" value="0" step="any"></label>
        <button type="button" data-role="custom-polygon-transform-apply">应用整体变换</button>
        <button type="button" data-role="custom-polygon-reset">恢复刚导入的轮廓</button>
      </div>
      <div class="configurator-polygon-table-wrap">
        <table class="configurator-polygon-table">
          <thead><tr><th>顶点序号</th><th>x</th><th>y</th><th>到下一顶点的边长</th><th>内角</th><th>操作</th></tr></thead>
          <tbody data-role="custom-polygon-vertices"></tbody>
        </table>
      </div>
    </div>
    <div class="configurator-boundary-builder">
      <p class="configurator-boundary-note">含时间方程还需核对<a href="#configurator-initial">上方初值设置</a>；静态问题无需初值。</p>
      <div class="configurator-boundary-heading">
        <strong>逐边界条件</strong>
        <p>一维分别设置左、右；矩形设置四边；平滑闭合曲线与识别出的 DXF 圆孔各使用一组连续边界条件；其余多边形按顶点序号逐边设置。法向导数均按计算域外法线方向。</p>
      </div>
      <div class="configurator-boundary-list">
        <div class="configurator-boundary-row" data-role="custom-boundary-side" data-side="left">
          <strong>左边界</strong>
          <label>类型<select data-role="boundary-type"><option value="dirichlet">Dirichlet：u=c</option><option value="neumann">Neumann：∂u/∂n=q</option><option value="robin">Robin：αu+β∂u/∂n=c</option></select></label>
          <label>目标值 c/q<input data-role="boundary-value" type="number" value="0" step="any"></label>
          <label data-role="robin-parameter">α<input data-role="boundary-alpha" type="number" value="1" step="any"></label>
          <label data-role="robin-parameter">β<input data-role="boundary-beta" type="number" value="1" step="any"></label>
        </div>
        <div class="configurator-boundary-row" data-role="custom-boundary-side" data-side="right">
          <strong>右边界</strong>
          <label>类型<select data-role="boundary-type"><option value="dirichlet">Dirichlet：u=c</option><option value="neumann">Neumann：∂u/∂n=q</option><option value="robin">Robin：αu+β∂u/∂n=c</option></select></label>
          <label>目标值 c/q<input data-role="boundary-value" type="number" value="0" step="any"></label>
          <label data-role="robin-parameter">α<input data-role="boundary-alpha" type="number" value="1" step="any"></label>
          <label data-role="robin-parameter">β<input data-role="boundary-beta" type="number" value="1" step="any"></label>
        </div>
        <div class="configurator-boundary-row" data-role="custom-boundary-side" data-side="bottom" data-dimension="2">
          <strong>下边界</strong>
          <label>类型<select data-role="boundary-type"><option value="dirichlet">Dirichlet：u=c</option><option value="neumann">Neumann：∂u/∂n=q</option><option value="robin">Robin：αu+β∂u/∂n=c</option></select></label>
          <label>目标值 c/q<input data-role="boundary-value" type="number" value="0" step="any"></label>
          <label data-role="robin-parameter">α<input data-role="boundary-alpha" type="number" value="1" step="any"></label>
          <label data-role="robin-parameter">β<input data-role="boundary-beta" type="number" value="1" step="any"></label>
        </div>
        <div class="configurator-boundary-row" data-role="custom-boundary-side" data-side="top" data-dimension="2">
          <strong>上边界</strong>
          <label>类型<select data-role="boundary-type"><option value="dirichlet">Dirichlet：u=c</option><option value="neumann">Neumann：∂u/∂n=q</option><option value="robin">Robin：αu+β∂u/∂n=c</option></select></label>
          <label>目标值 c/q<input data-role="boundary-value" type="number" value="0" step="any"></label>
          <label data-role="robin-parameter">α<input data-role="boundary-alpha" type="number" value="1" step="any"></label>
          <label data-role="robin-parameter">β<input data-role="boundary-beta" type="number" value="1" step="any"></label>
        </div>
        <div class="configurator-boundary-row" data-role="custom-boundary-side" data-side="curve" data-dimension="2" hidden>
          <strong>整条曲线边界</strong>
          <label>类型<select data-role="boundary-type"><option value="dirichlet">Dirichlet：u=c</option><option value="neumann">Neumann：∂u/∂n=q</option><option value="robin">Robin：αu+β∂u/∂n=c</option></select></label>
          <label>目标值 c/q<input data-role="boundary-value" type="number" value="0" step="any"></label>
          <label data-role="robin-parameter">α<input data-role="boundary-alpha" type="number" value="1" step="any"></label>
          <label data-role="robin-parameter">β<input data-role="boundary-beta" type="number" value="1" step="any"></label>
        </div>
        <div class="configurator-polygon-boundary-list" data-role="custom-polygon-boundaries" hidden></div>
      </div>
      <p class="configurator-boundary-note">圆形、椭圆、圆角矩形和闭合样条对整条连续曲线设置一组边界条件；多边形可逐边设置。周期边界需要成对约束两侧的函数值与导数，当前核心尚未实现。</p>
    </div>
    <div id="configurator-equation" class="configurator-unified-physics">
      <span id="configurator-boundary" class="configurator-anchor-marker" aria-hidden="true"></span>
      <div data-role="multiphysics-editor"></div>
    </div>
    <div data-role="physics-editor" hidden></div>
    <div class="configurator-term-builder" data-role="custom-term-builder" hidden>
      <div class="configurator-term-heading">
        <div>
          <strong>按项组合 PDE（兼容旧版）</strong>
          <p>每行是“带符号系数 × 方程项”，所有行之和等于 0。</p>
        </div>
        <button type="button" data-role="custom-add-term">+ 添加方程项</button>
      </div>
      <div class="configurator-term-list" data-role="custom-term-list"></div>
    </div>
    <p class="configurator-custom-limit">
      当前网页方程编辑面向单场一维/二维 PDE。需要命名参数、单位和自写公式，请选择“公式与物理参数”。矩形和导入多边形均可逐边设置
      Dirichlet、Neumann 或 Robin 条件，多边形孔洞也按各自顶点序号独立生成边界。初始速度、周期条件、多区域材料及
      MultiPolygon 仍需扩展。
      完整 Biot 位移—孔压耦合需在 Python 中扩展，并非单场孔压扩散模板。
      <a href="physics_library.html">物理参数库与公式编辑使用说明</a>。
    </p>
    <div class="configurator-custom-preview-layout">
      <div class="configurator-custom-formula-wrap">
        <strong>控制方程预览</strong>
        <div class="configurator-custom-formula" data-role="custom-formula"></div>
        <p data-role="custom-preview-note"></p>
      </div>
      <svg class="configurator-custom-preview" data-role="custom-preview" viewBox="0 0 720 360"
           role="img" aria-label="新控制方程一维或二维图例"></svg>
    </div>
  </section>
  <p data-role="planning-method-status" role="status" aria-live="polite" hidden></p>
  <a data-role="hybrid-revalidate-link" href="understanding_trial.html#hybrid-validation" hidden>
    将当前编辑发送到 Hybrid Validation 重新检查
  </a>
  <div class="configurator-layout">
    <section class="configurator-method-pane" data-role="method-pane" aria-labelledby="configurator-method-title">
      <h2 id="configurator-method-title">2. 五模块方法生态与参数</h2>
      <p>不再单独选择 PINN 或算子；模型卡中的 MLP/KAN/GNO 等生成 PINN，DeepONet/FNO 生成算子代码，其余模块随模型自动显示兼容选项。</p>
      <section class="ecosystem-selection-panel" aria-labelledby="ecosystem-selection-title">
        <h3 id="ecosystem-selection-title">选择组件并设置参数</h3>
        <p>PINN 选择会写入并实例化 <code>ECOSYSTEM_OVERRIDES</code>；Operator 的模型和训练参数直接生效，其他组件按现有生成器契约写入 <code>EXTENSIONS</code>，供算子训练器接入。</p>
        <div class="ecosystem-control-grid" data-role="ecosystem-controls"></div>
        {_operator_panel_html()}
        <div data-role="pinn-method-presets">
          <h3>Model 模块 · PINN 组合预设</h3>
          <p>可选择一个或多个已实现组合方法；同一功能位置发生冲突时会在生成代码前提示。</p>
          <p><a href="#configurator-budget">定位训练步数、输出间隔与随机种子</a></p>
          <div class="configurator-method-list" data-role="method-list"></div>
        </div>
        <section class="configurator-output-pane ecosystem-pinn-parameters" data-role="parameter-pane" aria-labelledby="configurator-output-title">
          <h3 id="configurator-output-title">Model 模块 · PINN 组合参数</h3>
          <div data-role="parameter-list"></div>
          <div class="configurator-message" data-role="message" role="status" aria-live="polite"></div>
        </section>
        <details>
          <summary>查看当前组件配置</summary>
          <pre><code data-role="ecosystem-overrides"></code></pre>
        </details>
      </section>
      <details class="method-ecosystem-catalog">
        <summary>浏览五模块全部内置家族与说明</summary>
        <div class="method-ecosystem" data-role="method-ecosystem"></div>
      </details>
      <details class="method-extension-builder">
        <summary>＋ 注册自定义方法组件</summary>
        <p>选择所属轴并填写 Python 模块与工厂函数，页面会生成显式注册代码；不会上传或执行本地模块。</p>
        <div class="method-extension-fields">
          <label>所属板块<select data-role="extension-kind"></select></label>
          <label>注册名称<input data-role="extension-name" type="text" placeholder="my_component" spellcheck="false"></label>
          <label>Python 模块<input data-role="extension-module" type="text" placeholder="my_project.methods" spellcheck="false"></label>
          <label>工厂函数<input data-role="extension-factory" type="text" placeholder="build_component" spellcheck="false"></label>
        </div>
        <pre><code data-role="extension-code">请填写上方字段。</code></pre>
        <button type="button" data-role="copy-extension">复制注册代码</button>
        <span data-role="extension-status" role="status" aria-live="polite"></span>
      </details>
    </section>
    <section class="configurator-code-pane" data-role="code-pane" aria-labelledby="configurator-code-title">
      <div class="configurator-code-header">
        <h2 id="configurator-code-title">3. 完整 Python 代码</h2>
        <div class="configurator-code-actions">
          <button type="button" data-role="expand-code" aria-haspopup="dialog">放大查看</button>
          <button type="button" data-role="download-code" disabled>下载 .py</button>
          <button type="button" data-role="copy" disabled>复制代码</button>
        </div>
      </div>
      <p class="configurator-code-hint">保留原始缩进，不自动换行；长行可横向滚动。下载后请放入项目 examples/ 目录运行。</p>
      <pre class="configurator-code" tabindex="0" aria-label="生成的 Python 代码，可横向和纵向滚动"><code data-role="code"></code></pre>
      <p data-role="code-action-status" role="status" aria-live="polite"></p>
    </section>
  </div>
  <dialog class="configurator-code-dialog" data-role="code-dialog" aria-labelledby="configurator-code-title"></dialog>
</div>
<script type="application/json" data-openpinn-configurator-data>{encoded}</script>
"""
        return [nodes.raw("", html, format="html")]


class ProblemDesignerDirective(SphinxDirective):
    """Render a standalone user-equation editor and compatibility report."""

    has_content = False
    optional_arguments = 0

    def run(self) -> list[nodes.Node]:
        from openpinn.method_registry import default_method_registry
        from openpinn.scriptgen import browser_payload

        registry = default_method_registry()
        methods: list[dict[str, Any]] = []
        for family in registry.specs():
            for variant in registry.variant_specs(family.name):
                methods.append(
                    {
                        "method": family.name,
                        "family_title": family.title,
                        "variant": variant.name,
                        "title": variant.title,
                        "compatible_with": list(variant.compatible_with),
                        "requires_checkpoint": variant.requires_checkpoint,
                    }
                )
        designer_payload = {"methods": methods}
        designer_payload.update(browser_payload())
        encoded = json.dumps(designer_payload, ensure_ascii=False).replace(
            "</", "<\\/"
        )
        local_paths = _designer_local_paths_html()
        html = f"""
<div class="openpinn-problem-designer" data-openpinn-problem-designer>
  <div class="problem-designer-toolbar">
    <label>问题名称（英文简称）
      <input data-role="problem-name" type="text" value="my_new_equation">
    </label>
    <label>问题结构
      <select data-role="problem-kind">
        <option value="static">静态控制方程</option>
        <option value="dynamic">含时间的动态控制方程</option>
        <option value="heterogeneous">一维两材料界面问题</option>
      </select>
    </label>
    <label class="problem-designer-check">
      <input data-role="has-checkpoint" type="checkbox">
      我已准备迁移学习 checkpoint
    </label>
  </div>
  <p class="problem-designer-notice">
    页面不会把这个临时方程加入配置器。方法结果由上方“问题结构”与
    OpenPINN 运行时注册条件共同生成；浏览器不会执行或理解任意 Python 方程。
  </p>
  <div class="problem-designer-layout">
    <section class="problem-designer-editor" aria-labelledby="problem-editor-title">
      <h2 id="problem-editor-title">1. 编辑新控制方程</h2>
      <p>重点修改代码中的区域、 <code>equation</code>、边界/初值与问题名称。</p>
      <textarea data-role="source" rows="38" spellcheck="false" aria-label="新控制方程 Python 代码"></textarea>
      <div class="problem-designer-actions">
        <button type="button" data-role="reset">按当前结构重置模板</button>
        <button type="button" data-role="copy">复制代码</button>
      </div>
      <div class="problem-designer-message" data-role="message" role="status" aria-live="polite"></div>
    </section>
    <section class="problem-designer-results" aria-labelledby="problem-results-title">
      <h2 id="problem-results-title">2. 适用方法检查结果</h2>
      <p data-role="summary"></p>
      <div data-role="method-results"></div>
    </section>
  </div>
  {local_paths}
</div>
<script type="application/json" data-openpinn-problem-designer-data>{encoded}</script>
"""
        return [nodes.raw("", html, format="html")]


class VisualProblemDesignerDirective(SphinxDirective):
    """Render the offline visual PDE builder preview."""

    has_content = False
    optional_arguments = 0

    def run(self) -> list[nodes.Node]:
        html = """
<div class="openpinn-visual-designer" data-openpinn-visual-designer>
  <header class="visual-designer-hero">
    <div>
      <span class="visual-designer-eyebrow">OpenPINN Problem Studio · 预览版</span>
      <h2>用积木定义控制方程</h2>
      <p>定义区域、拼接残差、查看公式和边界，再生成一份可运行的 OpenPINN Python 文件。</p>
    </div>
    <span class="visual-designer-offline">本地离线可用</span>
  </header>

  <section class="visual-designer-card visual-designer-settings" aria-labelledby="visual-settings-title">
    <div class="visual-designer-step-title">
      <span>1</span><div><h3 id="visual-settings-title">问题与网络</h3><p>先选择最接近的新问题结构。</p></div>
    </div>
    <div class="visual-designer-form-grid">
      <label>问题名称
        <input data-role="problem-name" type="text" value="my_visual_pde" spellcheck="false">
      </label>
      <label>问题类型
        <select data-role="problem-kind">
          <option value="static">静态方程</option>
          <option value="dynamic">含时间方程</option>
        </select>
      </label>
      <label>空间维度
        <select data-role="space-dim">
          <option value="1">一维区间</option>
          <option value="2">二维矩形</option>
        </select>
      </label>
      <label>x 下限<input data-role="x-min" type="number" value="-1" step="0.1"></label>
      <label>x 上限<input data-role="x-max" type="number" value="1" step="0.1"></label>
      <label data-role="y-field">y 下限<input data-role="y-min" type="number" value="-1" step="0.1"></label>
      <label data-role="y-field">y 上限<input data-role="y-max" type="number" value="1" step="0.1"></label>
      <label data-role="time-field">t 下限<input data-role="t-min" type="number" value="0" step="0.1"></label>
      <label data-role="time-field">t 上限<input data-role="t-max" type="number" value="1" step="0.1"></label>
      <label>边界值 u<input data-role="boundary-value" type="number" value="0" step="0.1"></label>
      <label data-role="initial-field">初值模板
        <select data-role="initial-preset">
          <option value="sine">正弦初值</option>
          <option value="zero">零初值</option>
        </select>
      </label>
      <label>隐藏层宽度
        <input data-role="hidden-layers" type="text" value="32, 32, 32" spellcheck="false">
      </label>
      <label>激活函数
        <select data-role="activation">
          <option value="tanh">tanh</option>
          <option value="gelu">GELU</option>
          <option value="silu">SiLU</option>
          <option value="relu">ReLU</option>
        </select>
      </label>
      <label>短跑步数<input data-role="steps" type="number" value="5" min="1" step="1"></label>
    </div>
  </section>

  <div class="visual-designer-main-grid">
    <section class="visual-designer-card visual-designer-workspace-card" aria-labelledby="visual-equation-title">
      <div class="visual-designer-step-title">
        <span>2</span><div><h3 id="visual-equation-title">拖拽方程积木</h3><p>最外层积木表示“残差 = 0”。</p></div>
      </div>
      <div class="visual-designer-workspace" data-role="workspace" aria-label="控制方程积木工作区"></div>
      <xml data-role="toolbox" style="display:none">
        <category name="方程" colour="#1f6d9b">
          <block type="pde_residual"></block>
        </category>
        <category name="导数" colour="#7b4ab5">
          <block type="pde_derivative"></block>
          <block type="pde_field"></block>
        </category>
        <category name="坐标与常数" colour="#287a55">
          <block type="pde_symbol"></block>
          <block type="pde_number"></block>
        </category>
        <category name="运算" colour="#b36a19">
          <block type="pde_binary"></block>
          <block type="pde_negative"></block>
          <block type="pde_function"></block>
        </category>
      </xml>
      <div class="visual-designer-actions">
        <button type="button" data-role="reset-equation">恢复示例方程</button>
        <button type="button" data-role="load-nonlinear-heat">加载新模型：非线性反应–扩散</button>
        <button type="button" class="is-secondary" data-role="center-blocks">居中积木</button>
      </div>
    </section>

    <section class="visual-designer-card visual-designer-preview-card" aria-labelledby="visual-preview-title">
      <div class="visual-designer-step-title">
        <span>3</span><div><h3 id="visual-preview-title">公式与区域预览</h3><p>绿色表示检查通过，橙色表示仍需修改。</p></div>
      </div>
      <div class="visual-equation-preview" data-role="formula" aria-live="polite"></div>
      <svg class="visual-domain-preview" data-role="domain-preview" viewBox="0 0 640 330" role="img" aria-label="计算区域与边界预览"></svg>
      <div class="visual-validation" data-role="validation" aria-live="polite"></div>
    </section>
  </div>

  <section class="visual-designer-card visual-designer-output" aria-labelledby="visual-output-title">
    <div class="visual-designer-step-title">
      <span>4</span><div><h3 id="visual-output-title">生成可运行代码</h3><p>默认只做 5 步链路检查；正式训练前再调整步数和方法。</p></div>
    </div>
    <div class="visual-designer-output-toolbar">
      <strong data-role="code-status">等待方程检查</strong>
      <div class="visual-designer-actions">
        <button type="button" data-role="copy-code">复制代码</button>
        <button type="button" class="is-secondary" data-role="download-code">下载 .py</button>
      </div>
    </div>
    <pre class="visual-designer-code"><code data-role="code"></code></pre>
    <div class="visual-designer-message" data-role="message" role="status" aria-live="polite"></div>
  </section>
</div>
"""
        return [nodes.raw("", html, format="html")]


def setup(app):
    app.add_directive("openpinn-method-catalog", MethodCatalogDirective)
    app.add_directive("openpinn-method-summary", MethodSummaryDirective)
    app.add_directive("openpinn-method-configurator", MethodConfiguratorDirective)
    app.add_directive("openpinn-configurator-guide", ConfiguratorGuideDirective)
    app.add_directive("openpinn-ecosystem-method", EcosystemMethodDirective)
    app.add_directive("openpinn-problem-designer", ProblemDesignerDirective)
    app.add_directive(
        "openpinn-visual-problem-designer", VisualProblemDesignerDirective
    )
    app.add_css_file("method_catalog.css")
    app.add_css_file("method_configurator.css")
    app.add_css_file("understanding_trial.css")
    app.add_css_file("physics_editor.css")
    app.add_css_file("multiphysics_editor.css")
    app.add_css_file("problem_designer.css")
    app.add_css_file("vendor/katex/katex.min.css")
    app.add_css_file("visual_problem_designer.css")
    app.add_js_file("joint_handoff.js", defer="defer")
    app.add_js_file("polygon_image_trace.js", defer="defer")
    app.add_js_file("dxf_geometry_import.js", defer="defer")
    app.add_js_file("physics_editor.js", defer="defer")
    app.add_js_file("multiphysics_editor.js", defer="defer")
    app.add_js_file("configurator_paths.js", defer="defer")
    app.add_js_file("generic_validation.js", defer="defer")
    app.add_js_file("hybrid_validation.js", defer="defer")
    app.add_js_file("codebuilder_agent.js", defer="defer")
    app.add_js_file("diagnostic_agent.js", defer="defer")
    app.add_js_file("agent_pipeline.js", defer="defer")
    app.add_js_file("method_configurator.js", defer="defer")
    app.add_js_file("configurator_navigation.js", defer="defer")
    app.add_js_file("understanding_trial.js", defer="defer")
    app.add_js_file("problem_designer.js", defer="defer")
    app.add_js_file("vendor/blockly/blockly_compressed.js", defer="defer")
    app.add_js_file("vendor/blockly/zh-hans.js", defer="defer")
    app.add_js_file("vendor/katex/katex.min.js", defer="defer")
    app.add_js_file("visual_problem_designer.js", defer="defer")
    app.connect("builder-inited", validate_catalog_contract)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
