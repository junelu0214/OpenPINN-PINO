"""Embed the existing offline manuals in the Sphinx navigation shell."""

from pathlib import Path
import re
import html
import json

from docutils import nodes
from sphinx.util.docutils import SphinxDirective
from manual_sections import MANUAL_PAGES, manual_sections, section_routes
from manual_diagrams import overview


class OpenPINNManualDirective(SphinxDirective):
    required_arguments = 1
    optional_arguments = 1

    def run(self):
        name = self.arguments[0]
        if name not in {"user", "developer"}:
            raise self.error("Manual must be user or developer.")
        source = Path(self.env.srcdir) / "manuals/public" / f"{name}_manual.html"
        self.env.note_dependency(str(source))
        if not source.is_file():
            raise self.error("Run python scripts/build_manuals.py before building docs.")
        text = source.read_text(encoding="utf-8")
        sections = manual_sections(text)
        routes = section_routes(name, sections)
        if len(self.arguments) == 1:
            # Overview pages have real child-page links, not a giant embedded manual.
            data = json.dumps(routes, ensure_ascii=False).replace("<", "\\u003c")
            return [nodes.raw("", overview(name) + '<script type="application/json" data-manual-routes>' + data + '</script>', format="html")]
        slug = self.arguments[1]
        pages = MANUAL_PAGES[name]
        selected = next((page for page in pages if page[0] == slug), None)
        if selected is None:
            raise self.error(f"Unknown manual page: {name}/{slug}")
        current = f"{name}_guide_{slug}.html"
        body = "".join(sections[anchor][1] for anchor in selected[3])
        def rewrite_link(match):
            anchor = match[1]
            if anchor not in routes:
                raise self.error(f"Unknown manual link: {anchor}")
            target = "" if routes[anchor] == current else routes[anchor]
            return 'href="' + target + '#' + anchor + '"'
        body = re.sub(r'href="#([^"]+)"', rewrite_link, body)
        headings = [(anchor, sections[anchor][0]) for anchor in selected[3]]
        contents = "".join(f'<li><a href="#{anchor}">{title}</a></li>' for anchor, title in headings)
        toolbar = (
            '<div class="manual-tools"><label>筛选本页小节 '
            '<input type="search" data-manual-search placeholder="输入关键词，不隐藏正文"></label>'
            '<button type="button" data-manual-expand>展开本页折叠内容</button>'
            '<button type="button" data-manual-collapse>收起</button></div>'
            '<details class="manual-local-nav"><summary>本页内容导航（' + str(len(headings)) + ' 节）</summary>'
            '<nav class="manual-contents" aria-label="本页内容导航"><ul>'
            + contents + '</ul></nav></details>'
        )
        index = pages.index(selected)
        links = [f'<a href="{name}_guide.html">返回说明书总览</a>']
        for label, position in (("上一部分", index - 1), ("下一部分", index + 1)):
            if 0 <= position < len(pages):
                target, title, _, _ = pages[position]
                links.append(f'<a href="{name}_guide_{target}.html">{label}：{html.escape(title)}</a>')
        pager = '<nav class="manual-page-links" aria-label="说明书分页">' + ''.join(links) + '</nav>'
        return [nodes.raw("", '<div class="openpinn-manual">' + pager + toolbar + body + pager + '</div>', format="html")]


def setup(app):
    app.add_directive("openpinn-manual", OpenPINNManualDirective)
    app.add_css_file("manuals.css")
    app.add_css_file("manual_diagrams.css")
    app.add_js_file("manuals.js", defer="defer")
    return {"version": "2.0", "parallel_read_safe": True, "parallel_write_safe": True}
