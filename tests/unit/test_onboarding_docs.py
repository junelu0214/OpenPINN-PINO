"""Guard the public project landing page and detailed documentation routes."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def read(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


def test_readme_links_and_prebuild_manuals_exist():
    readme = read("README.md")
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme):
        if target.startswith(("http:", "https:")):
            continue
        if target.startswith("#"):
            assert f'id="{target[1:]}"' in readme
        else:
            assert (ROOT / target).exists(), target
    for name in ("user_manual.html", "developer_manual.html"):
        assert (ROOT / "docs/manuals/public" / name).is_file()
    assert "](docs/_build/" not in readme  # GitHub cannot serve untracked build outputs.
    assert "https://github.com/junelu0214/OpenPINN-PINO" in readme
    assert "OpenPINN-upstream" not in readme
    assert "xiwang0706.github.io" not in readme
    assert readme.index("op.solve") < readme.index("op.train_operator")
    assert "ProblemSpec" in readme and "MethodSpec" in readme
    assert "CapabilityRegistry" in readme and "UnifiedRunResult" in readme
    assert "python scripts/build_docs.py" in readme
    assert "https://openpinn-upstream-demo.pages.dev" not in readme
    assert "out of band" not in readme
    assert "DOCS_DEPLOY_KEY" not in readme


def test_detailed_manual_uses_actual_first_download_name():
    text = read("docs/manuals/_sources/user_content.html")
    assert "python examples/first_poisson_run.py" in text
    assert ".\\.venv\\Scripts\\python.exe examples\\first_poisson_run.py" in text
    assert "重命名" in text
    assert "result.svg" in text and "finite_outputs" in text
    assert "5 步" in text


def test_install_routes_do_not_depend_on_missing_wheel_or_stale_manuals():
    for relative in ("README.md", "docs/install.rst", "docs/manuals/_sources/user_content.html"):
        text = read(relative)
        assert "scripts/build_docs.py" in text
        assert "python -m sphinx" not in text
        assert "openpinn-0.2.0-py3-none-any.whl" not in text
    for relative in ("docs/install.rst", "docs/manuals/_sources/user_content.html"):
        text = read(relative)
        assert "node --version" in text
    manual = read("docs/manuals/_sources/user_content.html")
    assert ".\\.venv\\Scripts\\python.exe my_first_run.py" in manual
    assert '方程编辑模式选“自由组合”' not in manual


def test_developer_checks_build_html_before_dom_tests():
    readme = read("README.md").split("## Development", 1)[-1]
    block = re.search(r"```bash\n([\s\S]*?)```", readme)[1]
    assert block.index("python scripts/build_docs.py") < block.index("npm test")
    source = read("docs/_ext/openpinn_method_catalog.py")
    assert "计算时仍以细分多边形近似圆弧" not in source
    assert "四顶点轴对齐矩形外框＋全部已识别圆孔使用解析圆孔训练" in source
