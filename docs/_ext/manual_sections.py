"""Single source of truth for the website manuals' small page hierarchy."""
import re

MANUAL_PAGES = {
    "user": [
        ("start", "安装与快速上手", "准备环境，完成第一次运行，了解内置模型。", ("scope", "install", "first-run", "builtins")),
        ("modeling", "自定义建模与代码生成", "从几何、控制方程和边界条件到可运行脚本。", ("configurator", "custom-equation", "boundary")),
        ("geometry", "几何创建、导入与编辑", "设置区域，导入图片或 DXF，编辑顶点和圆孔。", ("geometry", "image-import", "dxf", "hole-editor")),
        ("training", "训练设置与运行", "选择方法、调整参数，使用 Python/YAML 或运行双隧道二维模型。", ("methods", "tunnel", "python-yaml", "parameters")),
        ("results", "结果验证与实验交接", "读取输出，对比参考场，检查误差与重复实验。", ("reference", "outputs", "metrics", "checkpoint")),
        ("help", "问题排查与运行核对", "出现问题时按顺序定位，运行前后逐项检查。", ("troubleshooting", "user-checklist")),
    ],
    "developer": [
        ("start", "开始贡献：从反馈到开发计划", "复现问题，明确贡献范围，准备环境与基线。", ("contribution-plan", "dev-scope", "tree")),
        ("runtime", "新增算法与改进方法", "选择扩展点，接入配置与组件，用证据验证新方法。", ("method-recipe", "registries", "api-flow", "models", "samplers", "losses", "training", "experiment", "evaluation", "resume")),
        ("modeling", "扩展方程、几何与边界", "先实现最小数学问题，再按需接入公共入口。", ("extension-problem", "problem-contract", "geometry-contract", "conditions")),
        ("configurator", "修复功能与改进网页", "先复现再修复，保证表单、预览、代码及后端一致。", ("maintenance-recipe", "frontend", "import-pipelines", "extension-method")),
        ("delivery", "验证、提交 PR 与版本维护", "准备验收证据、兼容说明与审阅交接；区分建议与现有机制。", ("testing", "contribution-pr", "collaboration", "security", "manual-maintenance")),
        ("reference", "架构与源码参考", "需要时查询运行架构、模块契约、函数和快照。", ("architecture", "navigation-functions", "functions", "javascript-index", "snapshot")),
    ],
}


def manual_sections(text):
    """Split authored top-level sections, preserving nested HTML verbatim."""
    match = re.search(r"<main>([\s\S]*?)</main>", text)
    if not match:
        raise ValueError("Manual has no main content")
    parts = re.split(r'(?=<section id="[^"]+"><h2>)', match.group(1))
    result = {}
    for part in parts:
        heading = re.match(r'<section id="([^"]+)"><h2>(.*?)</h2>', part)
        if heading:
            result[heading[1]] = (heading[2], part)
        elif part.strip():
            raise ValueError("Unexpected content outside manual sections")
    return result


def section_routes(name, sections):
    """Map every section and nested anchor to its owning page; fail on omissions."""
    expected = [anchor for _, _, _, anchors in MANUAL_PAGES[name] for anchor in anchors]
    if len(expected) != len(set(expected)) or set(expected) != set(sections):
        raise ValueError(f"Manual page coverage mismatch: {name}")
    routes = {}
    for slug, _, _, anchors in MANUAL_PAGES[name]:
        for anchor in anchors:
            for identifier in re.findall(r'\bid="([^"]+)"', sections[anchor][1]):
                if identifier in routes:
                    raise ValueError(f"Duplicate manual anchor: {identifier}")
                routes[identifier] = f"{name}_guide_{slug}.html"
    return routes
