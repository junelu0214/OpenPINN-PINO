"""The optional gated demo stays out of the public project landing page."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def read(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


def test_publish_job_deploys_the_gated_pages_project():
    workflow = read(".github/workflows/pages.yml")
    assert "peaceiris/actions-gh-pages" not in workflow
    assert "DOCS_DEPLOY_KEY" not in workflow
    assert "xiwang0706/openpinn-docs" not in workflow
    assert "external_repository" not in workflow
    assert "secrets.INVITE_PASSPHRASE" not in workflow
    assert "secrets.CLOUDFLARE_API_TOKEN" in workflow
    assert "secrets.CLOUDFLARE_ACCOUNT_ID" in workflow
    assert "openpinn-upstream-demo" in workflow
    assert "wrangler@4.153.0" in workflow
    assert "scripts/stage_demo_site.py" in workflow
    assert "build-and-publish:" in workflow
    assert "needs: test" in workflow
    assert "if: github.event_name != 'pull_request'" in workflow
    assert "if: github.event_name == 'pull_request'" in workflow
    assert workflow.count("python scripts/build_docs.py") == 3
    builder = read("scripts/build_docs.py")
    assert '"docs/_build/html"' in builder
    assert '"docs/_build/html/en"' in builder
    assert '"language=en"' in builder
    assert "test ! -d docs/_build/html/_modules" in workflow
    demo = read("docs/demo_site.md")
    assert "https://openpinn-upstream-demo.pages.dev" in demo
    assert "wrangler@4.153.0 pages secret put INVITE_PASSPHRASE" in demo
    assert "out of band" in demo
    assert "openssl rand -base64 24" in demo
    assert "CLOUDFLARE_API_TOKEN" in demo
    assert "Rotate the invite" in demo
    readme = read("README.md")
    assert "https://openpinn-upstream-demo.pages.dev" not in readme
    assert "INVITE_PASSPHRASE" not in readme
    assert "python -m sphinx" not in readme


def test_stage_demo_site_copies_only_the_worker(tmp_path):
    html = tmp_path / "html"
    html.mkdir()
    (html / "method_configurator.html").write_text("<p>configurator</p>", encoding="utf-8")
    subprocess.run(
        [sys.executable, "scripts/stage_demo_site.py", "--html-dir", str(html)],
        cwd=ROOT,
        check=True,
    )
    staged = (html / "_worker.js").read_text(encoding="utf-8")
    assert staged == read("deploy/cloudflare/invite_gate.mjs")
    assert "preview_server" not in staged
    assert "node:" not in staged


def test_stage_demo_site_requires_the_html_directory(tmp_path):
    missing = tmp_path / "missing"
    result = subprocess.run(
        [sys.executable, "scripts/stage_demo_site.py", "--html-dir", str(missing)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "does not exist" in result.stderr


def test_invite_gate_node_suite():
    result = subprocess.run(
        ["node", "--test", "tests/demo/invite_gate.test.mjs"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
