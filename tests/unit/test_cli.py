# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for CLI argument validation."""

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from dlightrag.application.corpus_admin import ingest_kwargs_from_spec, ingest_spec_from_payload

# Load scripts/cli.py as a module (it's a script, not a package)
_cli_path = Path(__file__).resolve().parents[2] / "scripts" / "cli.py"
_spec = importlib.util.spec_from_file_location("cli", _cli_path)
assert _spec is not None and _spec.loader is not None
_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cli)

build_parser = _cli.build_parser
_build_answer_payload = _cli._build_answer_payload
_apply_query_options = _cli._apply_query_options
_validate_ingest_args = _cli._validate_ingest_args
cmd_answer = _cli.cmd_answer
_run_ingest = _cli._run_ingest


def _image_block(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def _parse_ingest(args: list[str]):
    """Parse CLI args for the ingest subcommand."""
    return build_parser().parse_args(["ingest", *args])


def _parse_query(args: list[str]):
    """Parse CLI args for the query subcommand."""
    return build_parser().parse_args(["query", *args])


def _parse_answer(args: list[str]):
    """Parse CLI args for the answer subcommand."""
    return build_parser().parse_args(["answer", *args])


def _parse_chat(args: list[str]):
    """Parse CLI args for the chat subcommand."""
    return build_parser().parse_args(["chat", *args])


# ---------------------------------------------------------------------------
# Test current REST payload options
# ---------------------------------------------------------------------------


def test_query_payload_supports_current_retrieval_options() -> None:
    args = _parse_query(
        [
            "find diagrams",
            "--top-k",
            "8",
            "--chunk-top-k",
            "5",
            "--workspaces",
            "finance",
            "legal",
            "--filters-json",
            '{"title":"Manual"}',
            "--filter-custom-json",
            '{"department":"finance"}',
            "--query-image",
            "data:image/png;base64,abc",
        ]
    )

    assert _apply_query_options({"query": args.query}, args) == {
        "query": "find diagrams",
        "top_k": 8,
        "chunk_top_k": 5,
        "workspaces": ["finance", "legal"],
        "filters": {
            "title": "Manual",
            "custom": {"department": "finance"},
        },
        "query_images": [_image_block("data:image/png;base64,abc")],
    }


def test_answer_payload_supports_current_answer_options() -> None:
    args = _parse_answer(
        [
            "summarize",
            "--chunk-top-k",
            "9",
            "--filters-json",
            '{"author":"Ada"}',
            "--attach-url",
            "https://example.test/chart.pdf",
        ]
    )

    assert _build_answer_payload(args, query=args.query) == {
        "query": "summarize",
        "chunk_top_k": 9,
        "filters": {"author": "Ada"},
        "attachments": [{"url": "https://example.test/chart.pdf"}],
    }


def test_answer_cli_local_attachment_uses_multipart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-body")

    captured: dict[str, Any] = {}

    async def fake_answer(self, payload, *, attachments=(), idempotency_key=None, on_token=None):
        captured["payload"] = payload
        captured["attachments"] = list(attachments)
        return _cli.AnswerResult.from_payload(
            {"answer": "done", "parts": [{"type": "markdown", "text": "done"}]}
        )

    monkeypatch.setattr(_cli.AnswerRunClient, "answer", fake_answer)
    monkeypatch.setattr(_cli.sdk_http, "api_url", lambda: "https://rag.example")
    monkeypatch.setattr(_cli.sdk_http, "auth_headers", lambda: {})
    monkeypatch.setattr(_cli.sdk_http, "client_timeout", lambda: 15)

    args = _parse_answer(
        [
            "summarize",
            "--attach",
            str(source),
            "--attach-url",
            "https://example.test/a.pdf",
        ]
    )
    cmd_answer(args)

    assert captured["payload"]["attachments"] == [{"url": "https://example.test/a.pdf"}]
    upload = captured["attachments"][0]
    assert upload.filename == "report.pdf"
    assert upload.content == b"%PDF-body"


def test_chat_payload_is_stateless_and_preserves_current_answer_options() -> None:
    args = _parse_chat(
        [
            "--chunk-top-k",
            "3",
        ]
    )

    assert _build_answer_payload(args, query="Follow up") == {
        "query": "Follow up",
        "chunk_top_k": 3,
    }


def test_answer_cli_renders_typed_evidence_images(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, Any] = {}

    async def fake_answer(self, payload, *, attachments=(), idempotency_key=None, on_token=None):
        captured["payload"] = payload
        return _cli.AnswerResult.from_payload(
            {
                "answer": "The figure shows the flow [1-1].",
                "parts": [{"type": "markdown", "text": "The figure shows the flow [1-1]."}],
                "references": [{"id": "1", "title": "paper.pdf"}],
                "evidence_images": [
                    {
                        "id": "fig-1",
                        "chunk_id": "chunk-1",
                        "source_ref": "1-1",
                        "label": "paper.pdf",
                        "url": "https://example.test/full.png",
                        "thumbnail_url": "https://example.test/thumb.png",
                    }
                ],
            }
        )

    monkeypatch.setattr(_cli.AnswerRunClient, "answer", fake_answer)
    monkeypatch.setattr(_cli.sdk_http, "api_url", lambda: "https://rag.example")
    monkeypatch.setattr(_cli.sdk_http, "auth_headers", lambda: {})
    monkeypatch.setattr(_cli.sdk_http, "client_timeout", lambda: 15)

    args = _parse_answer(["describe diagram"])
    cmd_answer(args)

    output = capsys.readouterr().out
    assert captured["payload"]["query"] == "describe diagram"
    assert "The figure shows the flow [1-1]." in output
    assert "[evidence image 1-1] paper.pdf https://example.test/thumb.png" in output
    assert "References (1):" in output


async def test_ingest_workspace_override_uses_the_durable_rest_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_ingest(self, payload, *, replace=False, idempotency_key=None):
        captured.update(payload=payload, replace=replace, idempotency_key=idempotency_key)
        return {"action": "ingest", "document_count": 1}

    monkeypatch.setattr(_cli.AnswerRunClient, "ingest", fake_ingest)
    monkeypatch.setattr(_cli.sdk_http, "api_url", lambda: "https://rag.example")
    monkeypatch.setattr(_cli.sdk_http, "auth_headers", lambda: {})
    monkeypatch.setattr(_cli.sdk_http, "client_timeout", lambda: 15)

    result = await _run_ingest(_parse_ingest(["./docs", "--workspace", "finance"]))

    assert result == {"action": "ingest", "document_count": 1}
    assert captured == {
        "payload": {
            "source_type": "local",
            "path": "./docs",
            "replace": False,
            "workspace": "finance",
        },
        "replace": False,
        "idempotency_key": None,
    }


def test_ingest_kwargs_support_document_metadata_options() -> None:
    args = _parse_ingest(
        [
            "./docs/report.pdf",
            "--title",
            "Quarterly Report",
            "--author",
            "Ada",
            "--metadata-json",
            '{"department":"finance"}',
        ]
    )

    assert ingest_kwargs_from_spec(ingest_spec_from_payload(args)) == {
        "path": "./docs/report.pdf",
        "replace": False,
        "title": "Quarterly Report",
        "author": "Ada",
        "metadata": {"department": "finance"},
    }


def test_ingest_kwargs_support_s3_region_and_retention() -> None:
    args = _parse_ingest(
        [
            "--source",
            "s3",
            "--bucket",
            "bucket",
            "--s3-key",
            "docs/report.pdf",
            "--s3-region",
            "eu-north-1",
            "--retain-source-file",
        ]
    )

    assert ingest_kwargs_from_spec(ingest_spec_from_payload(args)) == {
        "bucket": "bucket",
        "s3_key": "docs/report.pdf",
        "s3_region": "eu-north-1",
        "retain_source_file": True,
        "replace": False,
    }


def test_ingest_kwargs_support_url_source() -> None:
    args = _parse_ingest(
        [
            "--source",
            "url",
            "--url",
            "https://cdn.example.com/download?id=asset-1",
            "--filename",
            "asset.pdf",
            "--source-uri",
            "bynder://asset/asset-1",
        ]
    )

    assert ingest_kwargs_from_spec(ingest_spec_from_payload(args)) == {
        "url": "https://cdn.example.com/download?id=asset-1",
        "filename": "asset.pdf",
        "source_uri": "bynder://asset/asset-1",
        "replace": False,
    }


def test_ingest_kwargs_support_url_download_uris() -> None:
    args = _parse_ingest(
        [
            "--source",
            "url",
            "--urls",
            "https://fetch.example.com/a.pdf",
            "https://fetch.example.com/b.pdf",
            "--download-uris",
            "https://cdn.example.com/a.pdf",
            "https://cdn.example.com/b.pdf",
        ]
    )

    _validate_ingest_args(args)

    assert ingest_kwargs_from_spec(ingest_spec_from_payload(args)) == {
        "urls": [
            "https://fetch.example.com/a.pdf",
            "https://fetch.example.com/b.pdf",
        ],
        "download_uris": [
            "https://cdn.example.com/a.pdf",
            "https://cdn.example.com/b.pdf",
        ],
        "replace": False,
    }


def test_ingest_help_explains_signed_url_download_choices(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["ingest", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out.lower()

    assert "queryless durable download uri" in help_text
    assert "signed url" in help_text
    assert "retain" in help_text
    assert "--source url --urls https://example.com/a.pdf https://example.com/b.pdf" in help_text
    assert (
        "--source url --url 'https://fetch.example.com/doc?sig=...' --retain-source-file"
        in help_text
    )
    assert "--download-uri https://cdn.example.com/doc.pdf" in help_text


@pytest.mark.parametrize(
    "args",
    [
        [
            "--source",
            "url",
            "--urls",
            "https://fetch.example.com/a.pdf",
            "https://fetch.example.com/b.pdf",
            "--download-uri",
            "https://cdn.example.com/a.pdf",
        ],
        [
            "--source",
            "url",
            "--urls",
            "https://fetch.example.com/a.pdf",
            "https://fetch.example.com/b.pdf",
            "--download-uris",
            "https://cdn.example.com/a.pdf",
        ],
        [
            "--source",
            "url",
            "--url",
            "https://fetch.example.com/a.pdf",
            "--download-uri",
            "https://cdn.example.com/a.pdf",
            "--download-uris",
            "https://cdn.example.com/a.pdf",
        ],
        ["./docs/report.pdf", "--download-uri", "https://cdn.example.com/a.pdf"],
    ],
)
def test_ingest_rejects_invalid_download_uri_argument_shapes(args: list[str]) -> None:
    with pytest.raises(SystemExit):
        _validate_ingest_args(_parse_ingest(args))


def test_json_object_arg_rejects_non_object_json() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["query", "q", "--filter-custom-json", '["not", "object"]'])


# ---------------------------------------------------------------------------
# TestValidateIngestArgs — local source
# ---------------------------------------------------------------------------


class TestValidateLocal:
    """Validation for local source (default)."""

    @pytest.mark.parametrize(
        "argv",
        [
            pytest.param(["./docs"], id="valid_local"),
            pytest.param(["./docs", "--replace"], id="valid_local_with_flags"),
        ],
    )
    def test_valid(self, argv: list[str]) -> None:
        args = _parse_ingest(argv)
        _validate_ingest_args(args)  # should not raise

    @pytest.mark.parametrize(
        "argv",
        [
            pytest.param([], id="local_requires_path"),
            pytest.param(["./docs", "--container", "c"], id="local_rejects_container"),
            pytest.param(["./docs", "--bucket", "b"], id="local_rejects_bucket"),
        ],
    )
    def test_invalid(self, argv: list[str]) -> None:
        args = _parse_ingest(argv)
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)


# ---------------------------------------------------------------------------
# TestValidateIngestArgs — azure_blob source
# ---------------------------------------------------------------------------


class TestValidateAzureBlob:
    """Validation for azure_blob source."""

    @pytest.mark.parametrize(
        "argv",
        [
            pytest.param(
                ["--source", "azure_blob", "--container", "c"],
                id="valid_azure_container_only",
            ),
            pytest.param(
                ["--source", "azure_blob", "--container", "c", "--prefix", "docs/"],
                id="valid_azure_with_prefix",
            ),
            pytest.param(
                ["--source", "azure_blob", "--container", "c", "--blob-path", "f.pdf"],
                id="valid_azure_with_blob_path",
            ),
        ],
    )
    def test_valid(self, argv: list[str]) -> None:
        args = _parse_ingest(argv)
        _validate_ingest_args(args)

    @pytest.mark.parametrize(
        "argv",
        [
            pytest.param(["--source", "azure_blob"], id="azure_requires_container"),
            pytest.param(
                ["./docs", "--source", "azure_blob", "--container", "c"],
                id="azure_rejects_positional_path",
            ),
            pytest.param(
                [
                    "--source",
                    "azure_blob",
                    "--container",
                    "c",
                    "--blob-path",
                    "f.pdf",
                    "--prefix",
                    "docs/",
                ],
                id="azure_blob_path_and_prefix_mutually_exclusive",
            ),
            pytest.param(
                [
                    "--source",
                    "azure_blob",
                    "--container",
                    "c",
                    "--bucket",
                    "b",
                ],
                id="azure_rejects_bucket",
            ),
        ],
    )
    def test_invalid(self, argv: list[str]) -> None:
        args = _parse_ingest(argv)
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)


# ---------------------------------------------------------------------------
# TestValidateIngestArgs — s3 source
# ---------------------------------------------------------------------------


class TestValidateS3:
    """Validation for s3 source."""

    @pytest.mark.parametrize(
        "argv",
        [
            pytest.param(
                ["--source", "s3", "--bucket", "my-bucket", "--s3-key", "doc.pdf"],
                id="valid_s3_with_key",
            ),
            pytest.param(
                ["--source", "s3", "--bucket", "my-bucket", "--prefix", "docs/"],
                id="valid_s3_with_prefix",
            ),
        ],
    )
    def test_valid(self, argv: list[str]) -> None:
        args = _parse_ingest(argv)
        _validate_ingest_args(args)

    @pytest.mark.parametrize(
        "argv",
        [
            pytest.param(["--source", "s3"], id="s3_requires_bucket"),
            pytest.param(
                ["./docs", "--source", "s3", "--bucket", "b"],
                id="s3_rejects_positional_path",
            ),
            pytest.param(
                [
                    "--source",
                    "s3",
                    "--bucket",
                    "b",
                    "--s3-key",
                    "doc.pdf",
                    "--prefix",
                    "docs/",
                ],
                id="s3_key_and_prefix_mutually_exclusive",
            ),
            pytest.param(
                [
                    "--source",
                    "s3",
                    "--bucket",
                    "b",
                    "--s3-key",
                    "doc.pdf",
                    "--container",
                    "c",
                ],
                id="s3_rejects_container",
            ),
        ],
    )
    def test_invalid(self, argv: list[str]) -> None:
        args = _parse_ingest(argv)
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)
