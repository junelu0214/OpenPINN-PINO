"""Compact, offline-native navigation for the configurator's actual workflow."""
from html import escape


def configurator_flow():
    def box(key, x, y, width, title, note, href, terminal=False):
        kind = "terminal" if terminal else "process"
        label = title + "：" + note + "（定位到本页对应区域）"
        return (
            f'<a class="framework-node framework-{kind}" data-node="{key}" '
            f'href="{escape(href, quote=True)}" tabindex="0" '
            f'aria-label="{escape(label, quote=True)}">'
            f'<rect x="{x}" y="{y}" width="{width}" height="68" rx="10"/>'
            f'<text x="{x + width / 2}" y="{y + 28}" text-anchor="middle" class="framework-title">{escape(title)}</text>'
            f'<text x="{x + width / 2}" y="{y + 51}" text-anchor="middle" class="framework-note">{escape(note)}</text></a>'
        )

    def edge(start, end, path):
        return (f'<g class="framework-edge" data-from="{start}" data-to="{end}">'
                f'<path d="{path}" marker-end="url(#configurator-flow-arrow)"/></g>')

    nodes = [
        box("start", 20, 205, 170, "选择建模方式", "直接配置或 Agent 辅助", "#configurator-workbench", True),
        box("direct_problem", 235, 55, 205, "A1 通用问题编辑", "选择物理类与空间维度", "#configurator-problem"),
        box("direct_modules", 485, 55, 220, "A2 完成问题模块", "几何/数据/场/方程/边界", "#configurator-geometry"),
        box("agent_understanding", 235, 325, 205, "B1 Understanding", "自然语言 → 问题草稿", "#configurator-agent-pipeline"),
        box("agent_planning", 485, 325, 220, "B2 Planning", "问题草稿 → 方法草稿", "#configurator-agent-pipeline"),
        box("agent_handoff", 735, 325, 185, "B3 联合导入", "映射后人工核对与编辑", "#configurator-problem"),
        box("method", 735, 55, 185, "汇合：五模块生态", "选方法并设置全部参数", "#configurator-method-title"),
        box("code", 955, 55, 185, "生成完整代码", "校验 → 复制/下载 .py", "#configurator-code-title"),
        box("run", 955, 205, 185, "本地运行", "训练并写出 RunArtifact", "#configurator-run-help"),
        box("diagnostic", 955, 355, 185, "诊断与报告", "评估/图表/建议/下载", "#configurator-diagnostic", True),
    ]
    edges = [
        edge("start", "direct_problem", "M190,228 H210 V89 H235"),
        edge("start", "agent_understanding", "M190,252 H210 V359 H235"),
        edge("direct_problem", "direct_modules", "M440,89 H485"),
        edge("direct_modules", "method", "M705,89 H735"),
        edge("agent_understanding", "agent_planning", "M440,359 H485"),
        edge("agent_planning", "agent_handoff", "M705,359 H735"),
        edge("agent_handoff", "method", "M827.5,325 V123"),
        edge("method", "code", "M920,89 H955"),
        edge("code", "run", "M1047.5,123 V205"),
        edge("run", "diagnostic", "M1047.5,273 V355"),
    ]
    return (
        '<figure class="manual-framework configurator-guide" aria-label="配置生成器使用流程">'
        '<figcaption>两条建模路线在五模块方法生态汇合</figcaption>'
        '<p>直接路线使用通用物理问题编辑器逐模块建模；Agent-assisted 路线先完成 Understanding、Planning 和联合导入，再由用户核对同一套问题配置。两条路线从五模块方法生态开始共用生码、运行和诊断流程。</p>'
        '<div class="manual-framework-scroll" tabindex="0" role="region" aria-label="配置流程图，窄屏可横向滚动">'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 460" role="group" aria-labelledby="configurator-flow-title configurator-flow-desc">'
        '<title id="configurator-flow-title">配置生成器使用流程</title>'
        '<desc id="configurator-flow-desc">流程从直接配置和 Agent-assisted 两条路线分开。直接配置路线完成通用问题编辑器的物理类、几何、数据、场、方程和边界模块。Agent 路线依次完成 Understanding、Planning、联合导入和人工核对。两条路线在五模块方法生态汇合，随后生成代码、本地运行并上传 RunArtifact 形成诊断报告。</desc>'
        '<defs><marker id="configurator-flow-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
        '<path d="M0,0 L8,4 L0,8 Z" fill="#54768b"/></marker></defs>'
        '<rect class="framework-group" x="220" y="38" width="715" height="102" rx="12"/>'
        '<rect class="framework-group" x="220" y="308" width="715" height="102" rx="12"/>'
        '<text x="235" y="30" class="framework-group-label">路线 A · 直接使用配置生成器</text>'
        '<text x="235" y="300" class="framework-group-label">路线 B · Agent-assisted：生成草稿后仍需人工核对</text>'
        + ''.join(edges) + ''.join(nodes) + '</svg></div>'
        '<p>网页只预览和生成代码，不执行训练。修改配置后需重新下载或复制；'
        '<a href="#configurator-results-help">结果与误差怎样检查</a> · '
        '<a href="#configurator-troubleshooting-help">出错时怎样排查</a>。</p></figure>'
        '<p class="configurator-guide-actions"><a href="#configurator-workbench">开始配置 ↓</a></p>'
    )
