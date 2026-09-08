# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Server-side static checks for browser projection and served assets.

Frontend source contracts live in frontend/ui/*.structure.test.ts.
"""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
FRONTEND_STYLES = FRONTEND / "styles"


def test_bootstrap_advertises_exact_backend_attachment_limits() -> None:
    bootstrap_source = (ROOT / "src/dlightrag/adapters/http/browser/routes/bootstrap.py").read_text(
        encoding="utf-8"
    )

    assert "count_limit=application.config.answer.generation.max_attachments" in bootstrap_source
    assert (
        "attachment_limit = application.config.answer.generation.max_attachment_bytes"
        in bootstrap_source
    )
    assert "image_max_bytes=attachment_limit" in bootstrap_source
    assert "document_max_bytes=attachment_limit" in bootstrap_source


def test_vite_html_has_no_external_script_or_unresolved_theme_placeholder() -> None:
    for name in ("index.html", "login.html", "design-system.html", "product-showcase.html"):
        source = (FRONTEND / name).read_text(encoding="utf-8")
        built = (ROOT / "src/dlightrag/adapters/http/browser/static/app" / name).read_text(
            encoding="utf-8"
        )
        assert 'src="https://' not in source
        assert "__THEME_INIT__" not in built
        assert re.search(r'/static/app/assets/theme-init-[^"/]+\.js', built)


def test_web_shell_bootstraps_theme_preference_before_app_assets() -> None:
    index = (FRONTEND / "index.html").read_text(encoding="utf-8")
    theme = (FRONTEND / "theme-init.ts").read_text(encoding="utf-8")
    built = (ROOT / "src/dlightrag/adapters/http/browser/static/app/index.html").read_text(
        encoding="utf-8"
    )

    html_open = re.search(r"<html\b[^>]*>", index)
    assert html_open is not None
    assert 'lang="en"' in html_open.group(0)
    assert 'data-theme="system"' in html_open.group(0)
    assert 'data-color-mode="dark"' in html_open.group(0)
    assert '<meta name="color-scheme" content="dark light">' in index
    assert "'dlightrag-theme'" in theme
    assert "localStorage.getItem" in theme
    assert "matchMedia('(prefers-color-scheme: dark)')" in theme

    theme_script = built.index("/assets/theme-init-")
    app_script = built.index("/assets/app-")
    stylesheet = built.index('<link rel="stylesheet"')
    assert theme_script < stylesheet
    assert theme_script < app_script


def test_web_static_css_build_keeps_only_served_bundles() -> None:
    static_root = ROOT / "src/dlightrag/adapters/http/browser/static"
    assets = static_root / "app" / "assets"

    assert {path.name for path in static_root.glob("*.css")} == {"pygments.css"}
    styles = [path.name for path in assets.glob("style-*.css")]
    assert len(styles) == 1


def test_pygments_css_matches_generator() -> None:
    generator_path = ROOT / "scripts" / "generate_pygments_css.py"
    spec = importlib.util.spec_from_file_location("generate_pygments_css", generator_path)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    css = (ROOT / "src/dlightrag/adapters/http/browser/static/pygments.css").read_text(
        encoding="utf-8"
    )
    assert generator.generate_css() == css


def test_web_static_js_build_has_no_orphan_chunks() -> None:
    app_root = ROOT / "src/dlightrag/adapters/http/browser/static/app"
    assets = app_root / "assets"
    import_pattern = re.compile(
        r"""(?:import\(`\./([^`]+\.js)`\)|import\(["']\./([^"']+\.js)["']\)|from["']\./([^"']+\.js)["'])"""
    )
    html = "\n".join(
        (app_root / filename).read_text(encoding="utf-8")
        for filename in (
            "index.html",
            "login.html",
            "design-system.html",
            "product-showcase.html",
        )
    )
    roots = set(re.findall(r'/static/app/assets/([^"/]+\.js)', html))
    expected = {path.name for path in assets.glob("*.js")}
    seen: set[str] = set()
    stack = list(roots)

    while stack:
        filename = stack.pop()
        if filename in seen:
            continue
        seen.add(filename)
        content = (assets / filename).read_text(encoding="utf-8")
        for match in import_pattern.finditer(content):
            child = next(part for part in match.groups() if part)
            if child not in seen:
                stack.append(child)

    assert expected == seen


def _presentation_source(*, source_uri: str, download_url: str | None = None):
    from dlightrag.adapters.http.browser.presentation import build_answer_presentation
    from dlightrag.engine.answer.citations.contracts import SourceReferencePayload

    source = SourceReferencePayload(
        id="1",
        title=None,
        source_uri=source_uri,
        download_url=download_url,
        chunks=[],
    )
    return build_answer_presentation(
        answer="Cited [1].",
        sources=[source],
        evidence_images=[],
    ).sources[0]


def test_presentation_preserves_authorized_download_without_nesting_markup() -> None:
    source = _presentation_source(
        source_uri="local://default/notes.md",
        download_url="/web/api/files/raw/doc-notes?workspace=default",
    )
    assert source.download_url == "/web/api/files/raw/doc-notes?workspace=default"
    assert source.title == "Source"


def test_presentation_hides_download_without_caller_permission() -> None:
    source = _presentation_source(source_uri="local://default/notes.md")

    assert source.download_url is None


@pytest.mark.parametrize(
    "source_uri",
    [
        "https://exa.ai/library/weather/gothenburg-sweden?latitude=57.7052&longitude=11.9737",
        "http://www.sgas.ruc.edu.cn/xwgg/yjyxw/f1a3ff59a5894391b7b0db77951c08b4.htm",
    ],
)
def test_presentation_projects_public_web_provenance(source_uri: str) -> None:
    source = _presentation_source(source_uri=source_uri)

    assert source.source_url == source_uri


def test_presentation_rejects_non_public_provenance() -> None:
    for value in (
        "local://default/report.pdf",
        "https://127.0.0.1/private",
        "res-opaque",
    ):
        assert _presentation_source(source_uri=value).source_url is None


def test_answer_presentation_uses_semantic_citations_and_no_legacy_paths() -> None:
    from dlightrag.adapters.http.browser.presentation import build_answer_presentation

    presentation = build_answer_presentation(
        answer="Answer [1].",
        sources=[],
        evidence_images=[],
    )
    assert '<cite class="citation-badge"' in presentation.parts[0].html
    assert "answer_images" not in presentation.model_dump()


def test_source_anchor_allowlist_rejects_unsafe_attributes_and_targets() -> None:
    from dlightrag.adapters.http.browser.safe_html import sanitize_html_fragment

    html = sanitize_html_fragment(
        '<a href="/web/api/files/raw/doc-notes" aria-label="Download source" '
        'onclick="alert(1)" style="display:none" target="_self">Download</a>'
    )

    assert 'aria-label="Download source"' in html
    assert "onclick" not in html
    assert "style=" not in html
    assert "target=" not in html


def _css_blocks() -> list[tuple[str, str]]:
    """Every `selector { declarations }` pair across the served stylesheets."""
    blocks: list[tuple[str, str]] = []
    sheets = [
        *FRONTEND_STYLES.rglob("*.css"),
        *(FRONTEND / "design-system").rglob("*.css"),
    ]
    for sheet in sorted(sheets):
        css = re.sub(r"/\*.*?\*/", "", sheet.read_text(encoding="utf-8"), flags=re.S)
        for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
            blocks.append((selector.strip(), body))
    return blocks


def _declarations(body: str) -> dict[str, str]:
    decls: dict[str, str] = {}
    for line in body.split(";"):
        name, _, value = line.partition(":")
        name, value = name.strip().lower(), value.strip()
        if not name or not value:
            continue
        decls[name] = value
        if name == "border":
            # A base `border: 1px solid X` is what a hover `border-color` must beat.
            decls.setdefault("border-color", value.split()[-1])
    return decls


def test_jinja_template_tree_is_deleted() -> None:
    assert not (ROOT / "src/dlightrag/adapters/http/browser/templates").exists()


def test_production_web_sources_have_no_htmx_contract() -> None:
    sources = list((ROOT / "src/dlightrag/adapters/http/browser").rglob("*.py"))
    for path in sources:
        if "node_modules" in path.parts:
            continue
        source = path.read_text(encoding="utf-8").lower()
        assert "htmx" not in source
        assert not re.search(r"\bhx-[a-z]", source)


def test_button_hover_rules_change_something() -> None:
    """A hover that restates the base is the same as having no hover at all."""
    blocks = _css_blocks()
    base = {sel: _declarations(body) for sel, body in blocks if ":hover" not in sel}

    for selector, body in blocks:
        if ":hover" not in selector:
            continue
        hover = _declarations(body)
        if not hover:
            continue
        for part in selector.split(","):
            root = part.strip().split(":hover")[0].strip()
            if root not in base:
                continue
            changed = any(base[root].get(prop) != value for prop, value in hover.items())
            assert changed, f"{part.strip()} restates {root} and renders no feedback"
