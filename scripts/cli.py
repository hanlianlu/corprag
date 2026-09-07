#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""CLI for dlightrag — durable ingestion, retrieval, and answers over REST.

Usage:
    # Durable ingestion (requires the API server)
    uv run scripts/cli.py ingest ./docs
    uv run scripts/cli.py ingest ./docs --replace
    uv run scripts/cli.py ingest ./docs --workspace project-a
    uv run scripts/cli.py ingest ./report.pdf --title "Quarterly Report" --metadata-json '{"department":"finance"}'

    # Azure Blob ingestion
    uv run scripts/cli.py ingest --source azure_blob --container my-container
    uv run scripts/cli.py ingest --source azure_blob --container c --blob-path docs/report.pdf
    uv run scripts/cli.py ingest --source azure_blob --container c --prefix reports/

    # S3 ingestion
    uv run scripts/cli.py ingest --source s3 --bucket my-bucket --s3-key docs/report.pdf --s3-region us-east-1
    uv run scripts/cli.py ingest --source s3 --bucket my-bucket --prefix docs/

    # URL ingestion
    uv run scripts/cli.py ingest --source url --url https://example.com/doc.pdf --filename doc.pdf

    # Query & answer (requires API server: docker compose up dlightrag-api)
    uv run scripts/cli.py query "What are the key findings?" --chunk-top-k 30
    uv run scripts/cli.py query "findings?" --workspaces project-a project-b
    uv run scripts/cli.py query "findings?" --filters-json '{"author":"Ada"}'
    uv run scripts/cli.py answer "What are the key findings?"
    uv run scripts/cli.py answer "summarize report" --attach ./report.pdf --attach-url https://example.com/appendix.pdf
    uv run scripts/cli.py chat
    uv run scripts/cli.py chat --workspaces project-a project-b

"""

import argparse
import asyncio
import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from dlightrag.adapters.http.client import (
    AnswerAttachmentUpload,
    AnswerResult,
    AnswerRunClient,
    RunCancelledError,
    RunFailedError,
)
from dlightrag.adapters.http.client import http as sdk_http
from dlightrag.adapters.http.client.requests import query_image_blocks_from_urls
from dlightrag.application.corpus_admin import ingest_spec_from_payload


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def _json_object_arg(value: str) -> dict[str, Any]:
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"expected JSON object: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError("expected JSON object")
    return data


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(2)


def _validate_ingest_args(args: argparse.Namespace) -> None:
    """Reject flags aimed at the wrong source, then defer to the shared spec."""
    if args.source_type != "local" and args.path:
        _die(f"positional path is not used with {args.source_type}")
    if args.source_type != "azure_blob" and (args.container_name or args.blob_path):
        _die("--container, --blob-path are only for azure_blob source")
    if args.source_type != "s3" and (args.bucket or args.s3_key):
        _die("--bucket, --s3-key are only for s3 source")
    try:
        ingest_spec_from_payload(args)
    except ValidationError as exc:
        _die("; ".join(error["msg"].removeprefix("Value error, ") for error in exc.errors()))


def _metadata_filter_payload(args: argparse.Namespace) -> dict[str, Any] | None:
    filters = dict(args.filters_json or {})
    custom = args.filter_custom
    if custom is not None:
        existing = filters.get("custom")
        filters["custom"] = {**existing, **custom} if isinstance(existing, dict) else custom

    return filters or None


def _apply_common_options(
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if getattr(args, "chunk_top_k", None) is not None:
        payload["chunk_top_k"] = args.chunk_top_k
    if args.workspaces:
        payload["workspaces"] = args.workspaces

    filters = _metadata_filter_payload(args)
    if filters is not None:
        payload["filters"] = filters

    return payload


def _apply_query_options(
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    _apply_common_options(payload, args)

    if args.query_images:
        payload["query_images"] = query_image_blocks_from_urls(args.query_images)

    return payload


def _build_answer_payload(
    args: argparse.Namespace,
    *,
    query: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": query}
    _apply_common_options(payload, args)
    attachment_urls = getattr(args, "attachment_urls", None)
    if attachment_urls:
        payload["attachments"] = [{"url": url} for url in attachment_urls]
    return payload


def _format_evidence_image(source_ref: str, label: str, url: str) -> str:
    text = f"[evidence image {source_ref or '?'}]"
    if label:
        text = f"{text} {label}"
    return f"{text} {url}" if url else text


def _render_answer_for_terminal(data: AnswerResult) -> str:
    """Render typed Answer parts and the default Evidence Image region."""
    rendered: list[str] = []
    for part in data.parts:
        if part.type == "markdown":
            rendered.append(part.text)
        elif part.type == "artifact" and part.artifact is not None:
            artifact = part.artifact
            suffix = artifact.uri if artifact.status == "available" else "unavailable"
            rendered.append(f"\n[Artifact: {artifact.label}] {suffix}\n")
        elif part.type == "evidence_image" and part.evidence_image is not None:
            image = part.evidence_image
            rendered.append(
                "\n"
                + _format_evidence_image(
                    image.source_ref, image.label, image.thumbnail_url or image.url
                )
                + "\n"
            )
    inline_images = {
        part.evidence_image.id
        for part in data.parts
        if part.type == "evidence_image" and part.evidence_image is not None
    }
    for image in data.evidence_images:
        if image.id not in inline_images:
            rendered.append(
                "\n"
                + _format_evidence_image(
                    image.source_ref, image.label, image.thumbnail_url or image.url
                )
            )
    return "".join(rendered).strip() or data.answer or "(no answer)"


# ═══════════════════════════════════════════════════════════════════
# ingest
# ═══════════════════════════════════════════════════════════════════


async def _run_ingest(args: argparse.Namespace) -> dict[str, Any]:
    spec = ingest_spec_from_payload(args)
    payload = spec.model_dump(mode="json", exclude_none=True)
    if args.workspace:
        payload["workspace"] = args.workspace
    async with _answer_client() as client:
        return await client.ingest(payload, replace=bool(spec.replace))


def cmd_ingest(args: argparse.Namespace) -> None:
    _validate_ingest_args(args)
    action = "replace" if args.replace else "ingest"
    print(f"API: {sdk_http.api_url()}/runs/corpus/{action}\n")
    try:
        _print_json(asyncio.run(_run_ingest(args)))
    except RunCancelledError:
        _die("Corpus Mutation Run was cancelled")
    except RunFailedError as exc:
        _die(f"Corpus Mutation Run failed ({exc.error_kind}): {exc.public_message}")
    except httpx.HTTPStatusError as exc:
        _die(f"HTTP {exc.response.status_code}: {exc.response.text}")


# ═══════════════════════════════════════════════════════════════════
# query / answer / chat
# ═══════════════════════════════════════════════════════════════════


async def _run_query(args: argparse.Namespace) -> dict[str, Any]:
    async with _answer_client() as client:
        result = await client.retrieve(_apply_query_options({"query": args.query}, args))
    return {
        "contexts": dict(result.contexts),
        "sources": [dict(source) for source in result.sources],
        "trace": dict(result.trace),
        "image_descriptions": list(result.image_descriptions),
    }


def cmd_query(args: argparse.Namespace) -> None:
    print(f"Query: {args.query}")
    if args.workspaces:
        print(f"Workspaces: {', '.join(args.workspaces)}")
    print(f"API: {sdk_http.api_url()}/retrieve\n")

    try:
        _print_json(asyncio.run(_run_query(args)))
    except RunCancelledError:
        _die("retrieval run was cancelled")
    except RunFailedError as exc:
        _die(f"retrieval run failed ({exc.error_kind}): {exc.public_message}")
    except httpx.HTTPStatusError as exc:
        _die(f"HTTP {exc.response.status_code}: {exc.response.text}")


def _attachment_uploads(paths: list[str] | None) -> list[AnswerAttachmentUpload]:
    return [
        AnswerAttachmentUpload(filename=Path(path).name, content=Path(path).read_bytes())
        for path in paths or []
    ]


@asynccontextmanager
async def _answer_client() -> AsyncIterator[AnswerRunClient]:
    """Open the one REST client every answer command shares."""
    async with httpx.AsyncClient(timeout=sdk_http.client_timeout()) as http:
        yield AnswerRunClient(
            http,
            base_url=sdk_http.api_url(),
            headers=sdk_http.auth_headers(),
        )


async def _run_answer(args: argparse.Namespace) -> AnswerResult:
    async with _answer_client() as client:
        return await client.answer(
            _build_answer_payload(args, query=args.query),
            attachments=_attachment_uploads(getattr(args, "attachment_paths", None)),
        )


def cmd_answer(args: argparse.Namespace) -> None:
    print(f"Question: {args.query}")
    if args.workspaces:
        print(f"Workspaces: {', '.join(args.workspaces)}")
    print(f"API: {sdk_http.api_url()}/answer\n")

    try:
        data = asyncio.run(_run_answer(args))
    except RunCancelledError:
        _die("answer run was cancelled")
        return
    except RunFailedError as exc:
        _die(f"answer run failed ({exc.error_kind}): {exc.public_message}")
        return
    except httpx.HTTPStatusError as exc:
        _die(f"HTTP {exc.response.status_code}: {exc.response.text}")
        return

    # Print answer first, then validated references.
    answer = _render_answer_for_terminal(data)
    print(f"Answer:\n{answer}\n")

    if data.references:
        print(f"References ({len(data.references)}):")
        for ref in data.references:
            print(f"  [{ref.get('id', '?')}] {ref.get('title', '')}")


async def _run_chat(args: argparse.Namespace) -> None:
    ws_info = f", workspaces={','.join(args.workspaces)}" if args.workspaces else ""
    print(f"dlightrag chat (API={sdk_http.api_url()}{ws_info})")
    print("Type your question, or /quit to exit. Each request is stateless.\n")

    async with _answer_client() as client:
        while True:
            try:
                question = (await asyncio.to_thread(input, "You: ")).strip()
            except EOFError, KeyboardInterrupt:
                print("\nBye!")
                return

            if not question:
                continue
            if question in ("/quit", "/exit", "/q"):
                print("Bye!")
                return

            try:
                data = await client.answer(_build_answer_payload(args, query=question))
            except httpx.HTTPStatusError as exc:
                print(f"[error] HTTP {exc.response.status_code}: {exc.response.text}\n")
                continue
            except httpx.ConnectError:
                print(f"[error] Connection failed: {sdk_http.api_url()}\n")
                continue
            except RunFailedError as exc:
                print(f"[error] answer run failed ({exc.error_kind}): {exc.public_message}\n")
                continue
            except RunCancelledError:
                print("[error] answer run was cancelled\n")
                continue

            print(f"\nAssistant: {_render_answer_for_terminal(data)}")

            if data.sources:
                titles = {str(s["title"]) for s in data.sources if s.get("title")}
                if titles:
                    print(f"  Sources: {', '.join(sorted(titles))}")
            print()


def cmd_chat(args: argparse.Namespace) -> None:
    asyncio.run(_run_chat(args))


def _add_filter_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--filters-json",
        type=_json_object_arg,
        default=None,
        help="Full metadata filters JSON object sent to the API",
    )
    parser.add_argument(
        "--filter-custom-json",
        type=_json_object_arg,
        default=None,
        dest="filter_custom",
        help="Custom metadata filter JSON object",
    )


def _add_common_options(
    parser: argparse.ArgumentParser,
    *,
    include_chunk_top_k: bool = False,
) -> None:
    parser.add_argument("--top-k", type=int, default=None, dest="top_k")
    if include_chunk_top_k:
        parser.add_argument("--chunk-top-k", type=int, default=None, dest="chunk_top_k")
    parser.add_argument("--workspaces", nargs="+", default=None, help="Workspaces (federation)")
    _add_filter_options(parser)


def _add_retrieval_options(
    parser: argparse.ArgumentParser,
    *,
    include_chunk_top_k: bool = False,
) -> None:
    _add_common_options(parser, include_chunk_top_k=include_chunk_top_k)
    parser.add_argument(
        "--query-image",
        action="append",
        default=None,
        dest="query_images",
        help="User-attached image URL or data URI; repeat up to 3 times",
    )


def _add_answer_options(parser: argparse.ArgumentParser) -> None:
    _add_common_options(parser, include_chunk_top_k=True)
    parser.add_argument(
        "--attach",
        action="append",
        default=None,
        dest="attachment_paths",
        help="Local file to attach as an answer resource; repeatable",
    )
    parser.add_argument(
        "--attach-url",
        action="append",
        default=None,
        dest="attachment_urls",
        help="HTTPS URL to attach as an answer resource; repeatable",
    )


# ═══════════════════════════════════════════════════════════════════
# parser
# ═══════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dlightrag-cli",
        description="dlightrag CLI — durable ingestion, retrieval, and answers over REST",
        suggest_on_error=True,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- ingest --
    p_ingest = sub.add_parser(
        "ingest",
        help="Ingest documents from local, Azure Blob, S3, or URL sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Ingest documents into the RAG knowledge base.\n\n"
            "Source types:\n"
            "  local (default)  Ingest from local filesystem (file or directory)\n"
            "  azure_blob       Ingest from Azure Blob Storage container\n"
            "  s3               Ingest from AWS S3 bucket\n"
            "  url              Ingest from public or signed HTTPS URLs\n\n"
            "Examples:\n"
            "  %(prog)s ./docs                                          # local file/dir\n"
            "  %(prog)s ./docs --replace                                # local with replace\n"
            "  %(prog)s --source azure_blob --container my-container    # entire container\n"
            "  %(prog)s --source azure_blob --container c --prefix rpt/ # by prefix\n"
            "  %(prog)s --source s3 --bucket my-bucket --s3-key doc.pdf    # S3 single object\n"
            "  %(prog)s --source s3 --bucket my-bucket --prefix docs/   # S3 by prefix\n"
            "  %(prog)s --source url --url https://example.com/doc.pdf  # URL single document\n"
            "  %(prog)s --source url --urls https://example.com/a.pdf https://example.com/b.pdf\n"
            "  %(prog)s --source url --url 'https://fetch.example.com/doc?sig=...' --retain-source-file\n"
            "  %(prog)s --source url --url 'https://fetch.example.com/doc?sig=...' "
            "--download-uri https://cdn.example.com/doc.pdf"
        ),
    )
    p_ingest.add_argument("path", nargs="?", default=None, help="Path to file or directory (local)")
    p_ingest.add_argument(
        "--source",
        choices=["local", "azure_blob", "s3", "url"],
        default="local",
        dest="source_type",
        help="Data source type (default: local)",
    )
    p_ingest.add_argument("--container", dest="container_name", help="Azure Blob container name")
    p_ingest.add_argument("--blob-path", dest="blob_path", help="Specific blob (azure_blob)")
    p_ingest.add_argument("--prefix", help="Blob prefix filter (azure_blob/s3)")
    p_ingest.add_argument("--bucket", help="S3 bucket name")
    p_ingest.add_argument("--s3-region", dest="s3_region", help="S3 region name")
    p_ingest.add_argument("--s3-key", dest="s3_key", help="S3 object key")
    p_ingest.add_argument("--url", help="Public or signed HTTPS document URL")
    p_ingest.add_argument("--urls", nargs="+", help="Public or signed HTTPS document URLs")
    p_ingest.add_argument("--filename", help="Parser filename for a single URL")
    p_ingest.add_argument(
        "--source-uri", dest="source_uri", help="Stable source URI for a single URL"
    )
    p_ingest.add_argument(
        "--source-uris",
        nargs="+",
        dest="source_uris",
        help="Stable source URIs for URL batches",
    )
    p_ingest.add_argument(
        "--download-uri",
        dest="download_uri",
        help="Queryless durable download URI for a single URL (required for signed URLs unless retained)",
    )
    p_ingest.add_argument(
        "--download-uris",
        nargs="+",
        dest="download_uris",
        help="Queryless durable download URIs for URL batches",
    )
    p_ingest.add_argument(
        "--retain-source-file",
        action="store_true",
        default=None,
        dest="retain_source_file",
        help="Retain fetched bytes for later download (including signed URL fetches)",
    )
    p_ingest.add_argument("--replace", action="store_true", help="Replace existing documents")
    p_ingest.add_argument("--workspace", default=None, help="Target workspace")
    p_ingest.add_argument("--title", default=None, help="Optional document title metadata")
    p_ingest.add_argument("--author", default=None, help="Optional document author metadata")
    p_ingest.add_argument(
        "--metadata-json",
        type=_json_object_arg,
        default=None,
        dest="metadata",
        help="User metadata JSON object to attach to ingested documents",
    )

    # -- query --
    p_query = sub.add_parser("query", help="Retrieve contexts and sources (no answer)")
    p_query.add_argument("query", help="Search query")
    _add_retrieval_options(p_query, include_chunk_top_k=True)

    # -- answer --
    p_answer = sub.add_parser("answer", help="LLM-generated answer with contexts and sources")
    p_answer.add_argument("query", help="Question to answer")
    _add_answer_options(p_answer)

    # -- chat --
    p_chat = sub.add_parser("chat", help="Interactive multi-turn conversation")
    _add_common_options(p_chat, include_chunk_top_k=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "answer": cmd_answer,
        "chat": cmd_chat,
    }

    try:
        dispatch[args.command](args)
    except httpx.HTTPStatusError as e:
        print(f"HTTP {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except httpx.TimeoutException:
        print(f"Request timed out: {sdk_http.api_url()}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print(f"Connection failed: {sdk_http.api_url()}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
