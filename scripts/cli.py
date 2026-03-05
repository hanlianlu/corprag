#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""CLI for dlightrag — ingestion runs locally, queries go through the REST API.

Usage:
    # Local ingestion (runs directly via RAGService, no API server needed)
    uv run scripts/cli.py ingest ./docs
    uv run scripts/cli.py ingest ./docs --replace
    uv run scripts/cli.py ingest ./docs --workspace project-a

    # Azure Blob ingestion
    uv run scripts/cli.py ingest --source azure_blob --container my-container
    uv run scripts/cli.py ingest --source azure_blob --container c --blob-path docs/report.pdf
    uv run scripts/cli.py ingest --source azure_blob --container c --prefix reports/

    # Snowflake ingestion
    uv run scripts/cli.py ingest --source snowflake --query "SELECT * FROM reports"
    uv run scripts/cli.py ingest --source snowflake --query "SELECT * FROM t" --table reports

    # Query & answer (requires API server: docker compose up dlightrag-api)
    uv run scripts/cli.py query "What are the key findings?"
    uv run scripts/cli.py query "Revenue trends" --mode mix --top-k 30
    uv run scripts/cli.py query "findings?" --workspaces project-a project-b
    uv run scripts/cli.py answer "What are the key findings?"
    uv run scripts/cli.py answer "Summarize the report" --mode mix
    uv run scripts/cli.py chat
    uv run scripts/cli.py chat --workspaces project-a project-b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

import httpx

DEFAULT_API_URL = "http://localhost:8100"


def _get_api_url() -> str:
    return os.environ.get("DLIGHTRAG_API_URL", DEFAULT_API_URL)


def _get_auth_token() -> str | None:
    token = os.environ.get("DLIGHTRAG_API_AUTH_TOKEN")
    if token:
        return token

    from dotenv import dotenv_values, find_dotenv

    env = dotenv_values(find_dotenv(usecwd=True))
    return env.get("DLIGHTRAG_API_AUTH_TOKEN")


def _headers() -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    token = _get_auth_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _print_json(data: dict) -> None:
    print(json.dumps(data, indent=2, default=str))


# ── helpers ──────────────────────────────────────────────────────


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(2)


def _validate_ingest_args(args: argparse.Namespace) -> None:
    """Validate ingest arguments based on source type."""
    source = args.source_type

    if source == "local":
        if not args.path:
            _die(
                "local source requires a path argument.\n"
                "Usage: dlightrag-cli ingest <path> [--replace]"
            )
        if args.container_name or args.blob_path or args.prefix:
            _die("--container, --blob-path, --prefix are only for azure_blob source")
        if args.query or args.table:
            _die("--query, --table are only for snowflake source")

    elif source == "azure_blob":
        if args.path:
            _die(
                "positional path is not used with azure_blob source.\n"
                "Use --blob-path for a specific blob, or --prefix for batch."
            )
        if not args.container_name:
            _die(
                "azure_blob source requires --container.\n"
                "Usage: dlightrag-cli ingest --source azure_blob --container <name> "
                "[--blob-path <path> | --prefix <pfx>]"
            )
        if args.blob_path and args.prefix:
            _die("--blob-path and --prefix are mutually exclusive")
        if args.query or args.table:
            _die("--query, --table are only for snowflake source")

    elif source == "snowflake":
        if args.path:
            _die("positional path is not used with snowflake source")
        if not args.query:
            _die(
                "snowflake source requires --query.\n"
                "Usage: dlightrag-cli ingest --source snowflake --query '<SQL>'"
            )
        if args.container_name or args.blob_path or args.prefix:
            _die("--container, --blob-path, --prefix are only for azure_blob source")
        if args.replace:
            _die("--replace is not supported for snowflake source")


# ── subcommands ──────────────────────────────────────────────────


async def _run_ingest(args: argparse.Namespace) -> None:
    """Run ingestion directly via RAGService (no API server needed)."""
    from dlightrag.config import get_config
    from dlightrag.core.service import RAGService

    source = args.source_type
    kwargs: dict[str, Any] = {}

    if source == "local":
        kwargs["path"] = args.path
        kwargs["replace"] = args.replace
        print(f"Ingesting: {args.path} (replace={args.replace})")

    elif source == "azure_blob":
        kwargs["container_name"] = args.container_name
        if args.blob_path:
            kwargs["blob_path"] = args.blob_path
        if args.prefix is not None:
            kwargs["prefix"] = args.prefix
        kwargs["replace"] = args.replace
        target = args.blob_path or (f"prefix={args.prefix}" if args.prefix else "entire container")
        print(
            f"Ingesting from Azure Blob: container={args.container_name}, target={target} "
            f"(replace={args.replace})"
        )

    elif source == "snowflake":
        kwargs["query"] = args.query
        if args.table:
            kwargs["table"] = args.table
        print(f"Ingesting from Snowflake: query={args.query!r}")
        if args.table:
            print(f"  table metadata: {args.table}")

    config = get_config()
    workspace = args.workspace or config.workspace
    if args.workspace:
        config = config.model_copy(update={"workspace": workspace})
    print(f"Workspace: {workspace}")
    print("Running locally (direct RAGService)\n")

    service = await RAGService.create(config=config)
    try:
        result = await service.aingest(source_type=source, **kwargs)
        _print_json(result)
    finally:
        await service.close()


def cmd_ingest(args: argparse.Namespace) -> None:
    _validate_ingest_args(args)
    asyncio.run(_run_ingest(args))


def cmd_query(args: argparse.Namespace) -> None:
    url = f"{_get_api_url()}/retrieve"
    payload: dict = {"query": args.query, "mode": args.mode}
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.workspaces:
        payload["workspaces"] = args.workspaces

    print(f"Query: {args.query}")
    print(f"Mode: {args.mode}")
    if args.workspaces:
        print(f"Workspaces: {', '.join(args.workspaces)}")
    print(f"API: {url}\n")

    resp = httpx.post(url, json=payload, headers=_headers(), timeout=120)
    resp.raise_for_status()
    _print_json(resp.json())


def cmd_answer(args: argparse.Namespace) -> None:
    url = f"{_get_api_url()}/answer"
    payload: dict = {"query": args.query, "mode": args.mode}
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.workspaces:
        payload["workspaces"] = args.workspaces

    print(f"Question: {args.query}")
    print(f"Mode: {args.mode}")
    if args.workspaces:
        print(f"Workspaces: {', '.join(args.workspaces)}")
    print(f"API: {url}\n")

    resp = httpx.post(url, json=payload, headers=_headers(), timeout=120)
    resp.raise_for_status()
    _print_json(resp.json())


def cmd_chat(args: argparse.Namespace) -> None:
    url = f"{_get_api_url()}/answer"
    history: list[dict[str, str]] = []

    ws_info = f", workspaces={','.join(args.workspaces)}" if args.workspaces else ""
    print(f"dlightrag chat (mode={args.mode}{ws_info}, API={_get_api_url()})")
    print("Type your question, or: /clear to reset history, /quit to exit\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question in ("/quit", "/exit", "/q"):
            print("Bye!")
            break
        if question == "/clear":
            history.clear()
            print("-- history cleared --\n")
            continue

        payload: dict = {"query": question, "mode": args.mode}
        if args.top_k is not None:
            payload["top_k"] = args.top_k
        if args.workspaces:
            payload["workspaces"] = args.workspaces
        if history:
            payload["conversation_history"] = history

        try:
            resp = httpx.post(url, json=payload, headers=_headers(), timeout=120)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            print(f"[error] HTTP {e.response.status_code}: {e.response.text}\n")
            continue
        except httpx.ConnectError:
            print(f"[error] Connection failed: {_get_api_url()}\n")
            continue

        data = resp.json()
        answer_text = data.get("answer") or "(no answer)"

        print(f"\nAssistant: {answer_text}")

        sources = (data.get("raw") or {}).get("sources", [])
        if sources:
            titles = {s["title"] for s in sources if s.get("title")}
            if titles:
                print(f"  Sources: {', '.join(sorted(titles))}")
        print()

        # Append this turn to history
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer_text})


# ── parser ───────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dlightrag-cli",
        description="dlightrag CLI — ingestion runs locally, queries go through the REST API",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser(
        "ingest",
        help="Ingest documents from local, Azure Blob, or Snowflake sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Ingest documents into the RAG knowledge base.\n\n"
            "Source types:\n"
            "  local (default)  Ingest from local filesystem (file or directory)\n"
            "  azure_blob       Ingest from Azure Blob Storage container\n"
            "  snowflake        Ingest from Snowflake via SQL query\n\n"
            "Examples:\n"
            "  %(prog)s ./docs                                          # local file/dir\n"
            "  %(prog)s ./docs --replace                                # local with replace\n"
            "  %(prog)s --source azure_blob --container my-container    # entire container\n"
            "  %(prog)s --source azure_blob --container c --prefix rpt/ # by prefix\n"
            '  %(prog)s --source snowflake --query "SELECT * FROM t"    # snowflake query'
        ),
    )
    p_ingest.add_argument(
        "path", nargs="?", default=None, help="Path to file or directory (local source only)"
    )
    p_ingest.add_argument(
        "--source",
        choices=["local", "azure_blob", "snowflake"],
        default="local",
        dest="source_type",
        help="Data source type (default: local)",
    )
    p_ingest.add_argument(
        "--container", dest="container_name", help="Azure Blob container name (azure_blob source)"
    )
    p_ingest.add_argument(
        "--blob-path",
        dest="blob_path",
        help="Specific blob to ingest (azure_blob, mutually exclusive with --prefix)",
    )
    p_ingest.add_argument(
        "--prefix", help="Blob prefix filter (azure_blob, mutually exclusive with --blob-path)"
    )
    p_ingest.add_argument("--query", help="SQL query (snowflake source)")
    p_ingest.add_argument("--table", help="Table name metadata (snowflake source, optional)")
    p_ingest.add_argument("--replace", action="store_true", help="Replace existing documents")
    p_ingest.add_argument(
        "--workspace", default=None, help="Target workspace (default: from config)"
    )

    # query (retrieve only)
    p_query = sub.add_parser("query", help="Retrieve contexts and sources (no LLM answer)")
    p_query.add_argument("query", help="Search query")
    p_query.add_argument(
        "--mode", default="mix", choices=["local", "global", "hybrid", "naive", "mix"]
    )
    p_query.add_argument("--top-k", type=int, default=None, dest="top_k")
    p_query.add_argument(
        "--workspaces", nargs="+", default=None, help="Workspaces to search (federation)"
    )

    # answer (single-shot LLM answer)
    p_answer = sub.add_parser(
        "answer", help="Get an LLM-generated answer with contexts and sources"
    )
    p_answer.add_argument("query", help="Question to answer")
    p_answer.add_argument(
        "--mode", default="mix", choices=["local", "global", "hybrid", "naive", "mix"]
    )
    p_answer.add_argument("--top-k", type=int, default=None, dest="top_k")
    p_answer.add_argument(
        "--workspaces", nargs="+", default=None, help="Workspaces to search (federation)"
    )

    # chat (multi-turn REPL)
    p_chat = sub.add_parser("chat", help="Interactive multi-turn conversation")
    p_chat.add_argument(
        "--mode", default="mix", choices=["local", "global", "hybrid", "naive", "mix"]
    )
    p_chat.add_argument("--top-k", type=int, default=None, dest="top_k")
    p_chat.add_argument(
        "--workspaces", nargs="+", default=None, help="Workspaces to search (federation)"
    )

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
    except httpx.ConnectError:
        print(f"Connection failed: {_get_api_url()}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
