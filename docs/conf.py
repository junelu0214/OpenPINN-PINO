from __future__ import annotations

import os
import sys
from pathlib import Path

from docutils import nodes


DOCS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(DOCS_DIR / "_ext"))

project = "OpenPINN"
author = "OpenPINN team"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "nbsphinx",
    "openpinn_method_catalog",
    "openpinn_manuals",
    "openpinn_search",
]

# Tutorial outputs are executed and reviewed before they are committed.  Docs
# builds render those stored outputs and never retrain notebooks in CI.
nbsphinx_execute = "never"
nbsphinx_prolog = r"""
.. raw:: html

   <p class="nbsphinx-download">
     <a href="../_sources/{{ env.doc2path(env.docname, base=None) }}.txt"
        download="{{ env.docname.rsplit('/', 1)[-1] }}.ipynb">
       {% if env.config.language == 'en' %}Download this Jupyter notebook (.ipynb){% else %}下载这个 Jupyter notebook（.ipynb）{% endif %}
     </a>
   </p>
"""

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
language = os.environ.get("OPENPINN_DOCS_LANGUAGE", "zh_CN")
locale_dirs = ["locales/"]
gettext_compact = True

html_theme = "sphinx_rtd_theme"
html_title = "OpenPINN Documentation" if language == "en" else "OpenPINN 文档"
html_static_path = ["_static"]
html_css_files = ["document_site.css"]
html_extra_path = ["manuals/public"]
html_favicon = "_static/favicon.svg"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": True,
    "sticky_navigation": True,
}

autodoc_typehints = "description"
add_module_names = False


def _merge_repeated_first_column(app, doctree, docname):
    """Merge consecutive equal cells in tables marked for method grouping."""
    del app, docname
    for table in doctree.findall(nodes.table):
        if "merge-repeated-first-column" not in table.get("classes", []):
            continue

        for tbody in table.findall(nodes.tbody):
            previous_text = None
            first_entry = None
            additional_rows = 0

            for row in tbody.children:
                if not isinstance(row, nodes.row):
                    continue
                entries = [child for child in row.children if isinstance(child, nodes.entry)]
                if not entries:
                    continue

                current_entry = entries[0]
                current_text = current_entry.astext().strip()
                if current_text == previous_text and first_entry is not None:
                    additional_rows += 1
                    first_entry["morerows"] = additional_rows
                    row.remove(current_entry)
                    continue

                previous_text = current_text
                first_entry = current_entry
                additional_rows = 0


def _language_switch_context(app, pagename, templatename, context, doctree):
    """Link a rendered page to the same page in the other language tree."""
    del templatename, doctree
    depth = pagename.count("/")
    current_language = app.config.language
    if current_language == "en":
        context["language_switch_label"] = "中文"
        context["language_switch_url"] = "../" * (depth + 1) + pagename + ".html"
        context["openpinn_language"] = "en"
    else:
        context["language_switch_label"] = "English"
        context["language_switch_url"] = "../" * depth + "en/" + pagename + ".html"
        context["openpinn_language"] = "zh_CN"


def setup(app):
    app.connect("doctree-resolved", _merge_repeated_first_column)
    app.connect("html-page-context", _language_switch_context)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
