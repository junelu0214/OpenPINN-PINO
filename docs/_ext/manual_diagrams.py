"""Offline, accessible manual diagrams. Links are resolved by the page splitter.

Trees express topic/decision branches, not function calls. Flows express the
documented task order at a deliberately coarser level than the source code.
"""
import html
import re

from manual_sections import MANUAL_PAGES
from manual_frameworks import framework


def node(title, target, note, *children):
    return (title, target, note, children)


# section -> (layout, caption, nodes); fragment targets work offline and are
# rewritten to the appropriate child page by openpinn_manuals.py.
DIAGRAMS = {
    "user": {
        "custom-equation": ("flow", "核心流程：自定义模型 → 训练代码", [
            node("定义计算区域", "geometry", "维度、空间范围；需要时导入轮廓"),
            node("填写控制方程", "custom-equation-details", "模板或公式与物理参数；核对符号、单位与时间维度"),
            node("逐边设置条件", "boundary", "外轮廓 / 孔洞分别设置；动态问题核对初值"),
            node("选择方法与训练预算", "methods", "先用 baseline 和小预算检查；不要跳过训练设置"),
            node("核对生成脚本", "custom-delivery", "坐标、残差、条件与页面一致；修改后重新保存"),
            node("运行与精度验证", "metrics", "短跑检查可执行性；参考误差另行验证"),
        ]),
        "geometry": ("tree", "选哪一种几何入口？", [
            node("直接输入参数", "geometry-details", "一维、矩形、圆、椭圆、圆角矩形"),
            node("编辑平滑闭合曲线", "geometry-details", "控制点 + 张力；张力不等于数学曲率"),
            node("来自图片或坐标", "image-import", "核对物理尺度、外轮廓与孔洞"),
            node("来自工程图纸", "dxf", "二维 ASCII DXF → 单位换算 → 轮廓校验"),
            node("导入后精调", "hole-editor", "顶点、边长、圆心与半径；再核对边界编号"),
        ]),
        "boundary": ("tree", "边界与初值的选择图", [
            node("先确定作用边界", "hole-editor", "一维端点、二维外边、孔洞；按图中编号"),
            node("指定场值", "boundary-details", "Dirichlet：u = c"),
            node("指定外法向导数", "boundary-details", "Neumann：∂u/∂n = q；注意法向符号"),
            node("值与导数组合", "boundary-details", "Robin：αu + β∂u/∂n = c"),
            node("含时间时补初值", "boundary-details", "检查初始时刻与边界条件是否相容"),
        ]),
        "outputs": ("tree", "结果应该看哪一类文件？", [
            node("运行是否完成", "outputs-details", "status、failure_notes 与有限值检查"),
            node("计算结果是什么", "outputs-details", "predictions 与 metrics；保留原始数组"),
            node("训练过程怎样", "metrics", "history 与 loss 图不等于参考误差"),
            node("误差是否可信", "reference", "同坐标、同单位的独立参考对比"),
            node("如何复现与交接", "checkpoint", "脚本、配置、数据、环境及完整结果目录"),
        ]),
    },
    "developer": {
        "architecture": ("tree", "模块职责地图（分支表示职责，不是调用顺序）", [
            node("用户入口与配置", "api-flow", "solve / CLI → 配置展开与校验",
                 node("方法配置与组件创建", "registries", "MethodRegistry ≠ ComponentRegistry")),
            node("数学问题", "problem-contract", "PDEProblem：域、残差、条件",
                 node("几何与法向", "geometry-contract", "域判断、内部 / 边界采样与外法向"),
                 node("边界与初值", "conditions", "selector 与条件残差")),
            node("训练执行", "training", "模型、采样、损失与优化器协作",
                 node("实验组装与产物", "experiment", "Experiment.run 负责训练前后流程")),
            node("网页代码生成", "frontend", "浏览器状态 → Python 文本；不在网页训练"),
            node("评价与协作验收", "evaluation", "独立参考、指标、产物与交接"),
        ]),
        "registries": ("flow", "从方法选择到实际计算对象", [
            node("校验方法与选项", "registries-details", "method_options 检查参数、slot 冲突与顺序"),
            node("展开方法配置", "api-flow", "MethodRegistry 应用默认 / 选项补丁及 finalizer"),
            node("按配置创建组件", "experiment", "ComponentRegistry 查工厂并校验对象契约"),
            node("交给训练器执行", "training", "loss composer 单独组装，不属于七类注册组件"),
        ]),
        "frontend": ("tree", "同一设置如何驱动预览与代码", [
            node("接收输入 / 几何编辑", "import-pipelines", "input / change 或拖动提交；校验和状态更新"),
            node("读取统一设置", "frontend-function-contracts", "readCustomSettings；以下是两个消费分支，不是先求解再导出",
                 node("公式与 SVG 预览", "custom-data-contract", "用于建模核对，不执行 PDE 求解"),
                 node("Python 文本", "frontend-function-contracts", "updateCode 调用 generatedProblemLines；无效参数禁用导出")),
            node("执行生成脚本回归", "custom-regression", "不仅检查页面文本，还做语法、短跑与数值契约测试"),
        ]),
        "testing": ("flow", "验收层次：每一层提供不同证据", [
            node("接口 / 数学单元测试", "testing-details", "形状、域判断、导数、法向及失败输入"),
            node("网页事件与生成文本", "custom-regression", "参数变化须派发事件；验证状态和代码同步"),
            node("生成脚本语法与短跑", "change-acceptance", "证明可执行，不等于精度达标"),
            node("真实浏览器交互", "custom-regression", "缩放、拖拽、回退和布局不能仅由 DOM 测试证明"),
            node("独立参考 / 收敛验证", "evaluation", "按变更风险补精度证据，记录未测项"),
        ]),
    },
}


def render_diagram(kind, title, items, root=None):
    """Use native anchors and CSS connectors; no CDN or JS is needed."""
    def branch(item, number):
        label, target, note, children = item
        link = ('<a class="manual-map-node" href="' + html.escape(target, quote=True)
                + '"><span class="manual-map-icon" aria-hidden="true">' + number
                + '</span><span><strong>' + html.escape(label) + '</strong><small>'
                + html.escape(note) + '</small></span></a>')
        if children:
            link += '<ul>' + ''.join(branch(child, f'{number}.{i}') for i, child in enumerate(children, 1)) + '</ul>'
        return '<li>' + link + '</li>'
    tag = 'ol' if kind == 'flow' else 'ul'
    return ('<figure class="manual-map manual-map-' + kind + '" aria-label="' + html.escape(title, quote=True)
            + '"><figcaption>' + html.escape(title) + '</figcaption>'
            + '<p class="manual-map-hint">点击编号图标或整个节点可跳转；也可用 Tab 选择、Enter 打开。'
            + ('箭头表示步骤顺序。' if kind == 'flow' else '连线表示分类或分支，不表示执行先后。')
            + '</p><div class="manual-map-body">'
            + ('<div class="manual-map-root">' + html.escape(root or title) + '</div>' if kind == 'tree' else '')
            + '<' + tag + ' class="manual-map-branches">'
            + ''.join(branch(item, str(i)) for i, item in enumerate(items, 1))
            + '</' + tag + '></div></figure>')


def overview(name):
    """Lead with an actual process graph; keep the page directory secondary."""
    return framework(name) + '<details class="manual-overview-directory"><summary>按章节查阅（6 个部分）</summary><nav class="manual-overview-links" aria-label="说明书章节">' + ''.join(
        f'<a href="{name}_guide_{slug}.html">{html.escape(title)}</a>'
        for slug, title, _, _ in MANUAL_PAGES[name]
    ) + '</nav></details>'


def enrich(name, body):
    """Inject once before splitting; website and standalone stay identical."""
    for anchor, (kind, title, items) in DIAGRAMS[name].items():
        pattern = r'(<section id="' + re.escape(anchor) + r'"><h2>.*?</h2>)([\s\S]*?)(</section>)'
        def replace(match):
            content = match[2]
            # Keep detailed original call sequences available without showing
            # two diagrams at once. These are evidence, not redundant prose.
            if anchor in {'architecture', 'frontend'}:
                content = re.sub(r'(<pre class="flow">[\s\S]*?</pre>)',
                                 r'<details class="manual-map-detail"><summary>展开原始细化步骤 / 调用说明</summary>\1</details>', content)
            def targets(item):
                return node(item[0], '#' + item[1], item[2], *(targets(c) for c in item[3]))
            diagram = render_diagram(kind, title, [targets(item) for item in items])
            return match[1] + diagram + '<div id="' + anchor + '-details"></div>' + content + match[3]
        body, count = re.subn(pattern, replace, body, count=1)
        if count != 1:
            raise ValueError(f'Diagram section missing: {name}/{anchor}')
    return body
