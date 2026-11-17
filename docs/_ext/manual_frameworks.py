"""Two hand-laid-out SVG framework diagrams, shared by website and offline HTML."""
from html import escape

from manual_sections import MANUAL_PAGES


def framework(name, offline=False):
    routes = {anchor: f'{name}_guide_{slug}.html' for slug, _, _, anchors in MANUAL_PAGES[name] for anchor in anchors}
    prefix = 'framework-' + name
    def href(anchor):
        return '#' + anchor if offline else routes[anchor] + '#' + anchor
    def box(key, x, y, width, title, note, anchor, kind='process', height=62):
        if kind == 'decision':
            shape = f'<path d="M{x+width/2},{y} L{x+width},{y+height/2} L{x+width/2},{y+height} L{x},{y+height/2} Z"/>'
        else:
            shape = f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{28 if kind == "terminal" else 8}"/>'
        label = title + "：" + note
        return (f'<a class="framework-node framework-{kind}" data-node="{key}" href="{escape(href(anchor), quote=True)}" tabindex="0" aria-label="{escape(label, quote=True)}">'
                + shape + f'<text x="{x+width/2}" y="{y+height/2-5}" text-anchor="middle" class="framework-title">{escape(title)}</text>'
                + f'<text x="{x+width/2}" y="{y+height/2+18}" text-anchor="middle" class="framework-note">{escape(note)}</text></a>')
    def arrow(start, end, d, label='', lx=0, ly=0, dashed=False):
        return (f'<g class="framework-edge" data-from="{start}" data-to="{end}"><path d="{d}" marker-end="url(#{prefix}-arrow)"'
                + (' stroke-dasharray="6 5"' if dashed else '') + '/>'
                + (f'<text x="{lx}" y="{ly}" class="framework-edge-label">{escape(label)}</text>' if label else '') + '</g>')
    if name == 'user':
        title = '使用全流程：从问题定义到训练与验证'
        description = '先准备环境，再选择内置问题或自定义几何、控制方程和边界；两条路线汇合后设置训练、生成脚本、本地运行并验证。不满足目标时返回修改。'
        height = 960
        nodes = [
            box('start', 355, 20, 250, '安装并打开配置器', '先用小预算示例确认环境', 'install', 'terminal'),
            box('builtin', 105, 130, 260, '选择内置问题', 'Poisson / Burgers / 双隧道', 'builtins'),
            box('custom', 605, 130, 260, '建立自定义问题', '下面三步共同定义计算模型', 'custom-equation'),
            box('geometry', 440, 260, 145, '① 几何', '创建 / 图片 / DXF', 'geometry'),
            box('pde', 602, 260, 145, '② 控制方程', '模板 / 公式与参数', 'custom-equation'),
            box('boundary', 764, 260, 145, '③ 边界与初值', '逐边 / 孔洞设置', 'boundary'),
            box('settings', 355, 390, 250, '选择方法与训练参数', '网络、采样、步数与随机种子', 'methods'),
            box('code', 355, 490, 250, '核对并保存 Python 代码', '复制 / 下载；修改后重新保存', 'configurator'),
            box('run', 355, 590, 250, '在本地运行训练脚本', '网页预览本身不执行训练', 'python-yaml'),
            box('check', 365, 690, 230, '结果符合目标？', '指标与参考误差', 'metrics', 'decision', 102),
            box('done', 695, 710, 220, '保存与交接结果', '脚本 + 配置 + 完整产物', 'checkpoint', 'terminal'),
            box('revise', 45, 710, 250, '排查并修改对应设置', '几何 / PDE / 条件 / 训练参数', 'troubleshooting'),
        ]
        edges = [
            arrow('start','builtin','M480,82 V105 H235 V130'),
            arrow('start','custom','M480,82 V105 H735 V130'),
            arrow('custom','geometry','M735,192 V216 H512 V260'),
            arrow('geometry','pde','M585,291 H602'),
            arrow('pde','boundary','M747,291 H764'),
            arrow('builtin','settings','M235,192 V360 H480 V390','核对内置模型参数',65,345),
            arrow('boundary','settings','M836,322 V360 H480 V390'),
            arrow('settings','code','M480,452 V490'),
            arrow('code','run','M480,552 V590','代码交接到 Python 环境',505,578),
            arrow('run','check','M480,652 V690'),
            arrow('check','done','M595,741 H695','是',635,727),
            arrow('check','revise','M365,741 H295','否 / 报错',292,693),
            arrow('revise','custom','M170,772 V850 H940 V161 H865','返回对应设置后，重新生成、运行与验证',345,837,True),
        ]
        background = '<rect class="framework-group" x="423" y="238" width="501" height="98" rx="12"/>'
    else:
        title = '贡献流程：发现需求 → 实现改动 → 验证 → 审阅与维护'
        description = '贡献者先复现问题或提出方法，再明确范围并建立开发基线；按算法、数学问题或维护任务选择扩展点，完成最小改动、测试和文档后提交 PR，响应审阅，由维护者决定合并与发布。'
        height = 1100
        nodes = [
            box('feedback', 330, 20, 300, '从使用中发现不足或提出想法', '缺陷 / 新算法 / 新能力 / 细节改进', 'contribution-plan', 'terminal'),
            box('plan', 330, 125, 300, '复现或提案：定义要解决的问题', '最小输入、范围、验收标准；大改动先讨论', 'contribution-plan'),
            box('baseline', 330, 230, 300, '准备环境，建立未修改的基线', '确认源码来源；保留可复现记录', 'dev-scope'),
            box('algorithm', 45, 350, 260, '新增算法 / 改进方法', '组件实现 → 配置接入 → 方法测试', 'method-recipe'),
            box('math', 350, 350, 260, '扩展方程 / 几何 / 边界', '数学契约 → 最小问题 → 按需接入', 'extension-problem'),
            box('fix', 655, 350, 260, '修复缺陷 / 优化网页与文档', '失败案例 → 定位源文件 → 回归', 'maintenance-recipe'),
            box('implement', 330, 480, 300, '在对应扩展点做最小改动', '保留旧接口；同步必要示例与说明', 'collaboration'),
            box('tests', 330, 590, 300, '验证行为与兼容性', '针对性测试 + 必要回归；数值主张另取证', 'testing'),
            box('pr', 330, 700, 300, '提交可复核的 PR', '变更理由、测试结果、迁移和未测项', 'contribution-pr'),
            box('review', 355, 805, 250, '维护者审阅通过？', '目标 / 契约 / 兼容 / 证据', 'contribution-pr', 'decision', 108),
            box('revise', 680, 825, 240, '回应意见并补充修改', '补测试、文档或缩小改动范围', 'contribution-pr'),
            box('done', 330, 985, 300, '合并、记录变更与后续维护', '维护者另行决定版本发布；跟进回归', 'contribution-pr', 'terminal'),
        ]
        edges = [
            arrow('feedback','plan','M480,82 V125'),
            arrow('plan','baseline','M480,187 V230'),
            arrow('baseline','algorithm','M480,292 V320 H175 V350'),
            arrow('baseline','math','M480,292 V350'),
            arrow('baseline','fix','M480,292 V320 H785 V350'),
            arrow('algorithm','implement','M175,412 V450 H480 V480'),
            arrow('math','implement','M480,412 V480'),
            arrow('fix','implement','M785,412 V450 H480 V480'),
            arrow('implement','tests','M480,542 V590'),
            arrow('tests','implement','M330,621 H265 V511 H330','未通过，返回修正',70,565,True),
            arrow('tests','pr','M480,652 V700','通过',500,682),
            arrow('pr','review','M480,762 V805'),
            arrow('review','revise','M605,859 H680','需修改',615,839),
            arrow('revise','implement','M800,825 V511 H630','再验证后更新 PR',665,690,True),
            arrow('review','done','M480,913 V985','通过',500,956),
        ]
        background = ('<rect class="framework-group" x="24" y="335" width="912" height="94" rx="12"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 {height}" role="group" aria-labelledby="{prefix}-title {prefix}-desc">'
           f'<title id="{prefix}-title">{escape(title)}</title><desc id="{prefix}-desc">{escape(description)}</desc>'
           f'<defs><marker id="{prefix}-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#54768b"/></marker></defs>'
           + background + ''.join(edges) + ''.join(nodes) + '</svg>')
    footnote = ('图中“符合目标”需结合物理条件及参考误差判断，训练完成或 loss 下降不等于精度达标。'
                if name == 'user' else '三条开发路线按任务选用，不要求全部实施。Issue / PR 是 GitHub 发布后的建议流程；当前未宣称已启用 CI 或自动合并。运行架构移至“架构与源码参考”。')
    return ('<figure class="manual-framework"><figcaption>' + title + '</figcaption>'
            '<p>顺着箭头看完整链路；点击图中方框可进入对应说明。窄屏可横向滚动。</p>'
            '<div class="manual-framework-scroll" tabindex="0" role="region" aria-label="可滚动的整体流程图">' + svg + '</div>'
            '<p class="framework-footnote">' + footnote + '</p></figure>')
