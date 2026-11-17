"""Index rendered documentation, including raw-HTML manuals, for offline search.

The index is a local script, not a fetched JSON file: file:// browsers can load it.
Rebuild from all current source pages after writing HTML, including incremental
builds, so deleted pages and unchanged pages are handled without a second cache.
"""
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import quote


def plain(text):
    return re.sub(r"\s+", " ", text).strip()


class PageSections(HTMLParser):
    SKIP = {"script", "style", "nav", "svg", "math", "button", "select", "textarea"}
    BLOCK = {"p", "div", "section", "li", "tr", "td", "th", "pre", "br", "dt", "dd"}
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self, href, title):
        super().__init__(convert_charrefs=True)
        self.href, self.title = href, title
        self.stack = []
        self.records = []
        self.heading, self.anchor, self.text = title, "", []
        self.heading_parts = None

    def _inside(self):
        return any(attrs.get("itemprop") == "articleBody" for _, attrs in self.stack)

    def _skip(self):
        return any(tag in self.SKIP or "headerlink" in attrs.get("class", "").split()
                   for tag, attrs in self.stack)

    def _flush(self):
        text = plain("".join(self.text))
        if text:
            target = self.href + ("#" + quote(self.anchor, safe="-._:") if self.anchor else "")
            self.records.append({"href": target, "title": self.title,
                                 "heading": self.heading, "text": text})
        self.text = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag not in self.VOID:
            self.stack.append((tag, attrs))
        if not self._inside() or self._skip():
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._flush()
            self.anchor = attrs.get("id", "") or next(
                (a.get("id", "") for t, a in reversed(self.stack) if t == "section"), "")
            self.heading_parts = []
        elif tag in self.BLOCK:
            self.text.append(" ")

    def handle_endtag(self, tag):
        if self._inside() and not self._skip():
            if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self.heading_parts is not None:
                self.heading = plain("".join(self.heading_parts)) or self.title
                self.heading_parts = None
                self.text.append(" ")
            elif tag in self.BLOCK:
                self.text.append(" ")
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        if self._inside() and not self._skip():
            self.text.append(data)
            if self.heading_parts is not None:
                self.heading_parts.append(data)

    def finish(self):
        self.close()
        self._flush()
        return self.records


def write_search_index(app, exception):
    if exception is not None or app.builder.name != "html":
        return
    records = []
    for name in sorted(app.env.found_docs):
        source = Path(app.builder.get_outfilename(name))
        title = app.env.titles[name].astext() if name in app.env.titles else name
        parser = PageSections(app.builder.get_target_uri(name), title)
        parser.feed(source.read_text(encoding="utf-8"))
        records.extend(parser.finish())
    output = Path(app.outdir) / "_static/openpinn_search_index.js"
    data = json.dumps({"version": 1, "records": records}, ensure_ascii=False, separators=(",", ":"))
    output.write_text("window.OpenPINNSearchIndex = " + data.replace("<", "\\u003c") + ";\n", encoding="utf-8")


def replace_highlighter(app, pagename, templatename, context, doctree):
    # The upstream highlighter mutates SVG title/desc nodes and calls getBBox on
    # them. Our text-only implementation also handles legacy ?highlight= links.
    context["script_files"] = [asset for asset in context.get("script_files", [])
                               if not str(getattr(asset, "filename", asset)).split("?")[0].endswith("/sphinx_highlight.js")]


def setup(app):
    app.add_js_file("documentation_search.js", defer="defer")
    app.add_css_file("documentation_search.css")
    app.connect("html-page-context", replace_highlighter)
    app.connect("build-finished", write_search_index)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
