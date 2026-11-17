"""The index must include raw HTML manuals and remain valid after incremental builds."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("openpinn_search", ROOT / "docs/_ext/openpinn_search.py")
search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search)


def test_rendered_manual_sections_and_split_inline_chinese():
    parser = search.PageSections("user_guide_geometry.html", "几何")
    parser.feed('''<head><title>不应索引</title></head><nav>侧栏</nav>
    <div itemprop="articleBody"><h1>几何</h1>
    <section id="holes"><h2>圆孔<a class="headerlink">¶</a></h2>
    <p><strong>固结</strong>系数与孔压。</p><p>第二段。</p>
    <script>secret()</script><svg><title>图的隐藏文本</title></svg>
    <section id="nested"><h3 id="own">参数</h3><p>a &lt; b</p></section>
    <h2 id="last">末节</h2><p>末尾内容</p></section></div><footer>页脚</footer>''')
    records = parser.finish()
    hole = next(r for r in records if r["heading"] == "圆孔")
    assert hole["href"] == "user_guide_geometry.html#holes"
    assert "固结系数" in hole["text"] and "孔压。 第二段。" in hole["text"]
    assert any(r["href"].endswith("#own") and "a < b" in r["text"] for r in records)
    assert records[-1]["text"].endswith("末尾内容")
    all_text = "".join(r["text"] for r in records)
    for excluded in ["侧栏", "secret", "隐藏文本", "页脚", "不应索引", "¶"]:
        assert excluded not in all_text


def test_index_rebuild_reads_unchanged_pages_and_drops_deleted_pages(tmp_path):
    (tmp_path / "_static").mkdir()
    for name, body in [("one", "固结系数"), ("two", "圆孔"), ("deleted", "旧页")]:
        (tmp_path / (name + ".html")).write_text('<div itemprop="articleBody"><p>' + body + '</p></div>', encoding="utf-8")
    app = SimpleNamespace(outdir=tmp_path,
        env=SimpleNamespace(found_docs={"one", "two"}, titles={}),
        builder=SimpleNamespace(name="html", get_outfilename=lambda name: tmp_path / (name + ".html"),
                                get_target_uri=lambda name: name + ".html"))
    search.write_search_index(app, None)
    target = tmp_path / "_static/openpinn_search_index.js"
    first = target.read_text(encoding="utf-8")
    data = json.loads(first.removeprefix("window.OpenPINNSearchIndex = ").removesuffix(";\n"))
    assert {r["href"] for r in data["records"]} == {"one.html", "two.html"}
    app.env.found_docs = {"two"}
    search.write_search_index(app, None)
    assert "one.html" not in target.read_text(encoding="utf-8")
    saved = target.read_bytes()
    search.write_search_index(app, RuntimeError("failed build"))
    assert target.read_bytes() == saved


def test_upstream_svg_highlighter_is_removed_without_removing_other_scripts():
    asset = SimpleNamespace(filename="_static/sphinx_highlight.js")
    keep = SimpleNamespace(filename="_static/doctools.js")
    context = {"script_files": [asset, keep, "_static/sphinx_highlight.js"]}
    search.replace_highlighter(None, None, None, context, None)
    assert context["script_files"] == [keep]
