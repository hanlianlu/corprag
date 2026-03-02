#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""CLI client for the corprag REST API.

Usage:
    uv run scripts/cli.py ingest ./docs
    uv run scripts/cli.py ingest ./docs --replace
    uv run scripts/cli.py query "What are the key findings?"
    uv run scripts/cli.py query "Revenue trends" --mode mix --top-k 30
    uv run scripts/cli.py answer "What are the key findings?"
    uv run scripts/cli.py answer "Summarize the report" --mode mix
    uv run scripts/cli.py chat
    uv run scripts/cli.py chat --mode mix
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

DEFAULT_API_URL = "http://localhost:8100"


def _get_api_url() -> str:
    return os.environ.get("CORPRAG_API_URL", DEFAULT_API_URL)


def _get_auth_token() -> str | None:
    token = os.environ.get("CORPRAG_API_AUTH_TOKEN")
    if token:
        return token

    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("CORPRAG_API_AUTH_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None


def _headers() -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    token = _get_auth_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _print_json(data: dict) -> None:
    print(json.dumps(data, indent=2, default=str))


# ── subcommands ──────────────────────────────────────────────────


def cmd_ingest(args: argparse.Namespace) -> None:
    url = f"{_get_api_url()}/ingest"
    payload = {
        "source_type": "local",
        "path": args.path,
        "replace": args.replace,
    }
    print(f"Ingesting: {args.path} (replace={args.replace})")
    print(f"API: {url}\n")

    resp = httpx.post(url, json=payload, headers=_headers(), timeout=600)
    resp.raise_for_status()
    _print_json(resp.json())


def cmd_query(args: argparse.Namespace) -> None:
    url = f"{_get_api_url()}/retrieve"
    payload: dict = {"query": args.query, "mode": args.mode}
    if args.top_k is not None:
        payload["top_k"] = args.top_k

    print(f"Query: {args.query}")
    print(f"Mode: {args.mode}")
    print(f"API: {url}\n")

    resp = httpx.post(url, json=payload, headers=_headers(), timeout=120)
    resp.raise_for_status()
    _print_json(resp.json())


def cmd_answer(args: argparse.Namespace) -> None:
    url = f"{_get_api_url()}/answer"
    payload: dict = {"query": args.query, "mode": args.mode}
    if args.top_k is not None:
        payload["top_k"] = args.top_k

    print(f"Question: {args.query}")
    print(f"Mode: {args.mode}")
    print(f"API: {url}\n")

    resp = httpx.post(url, json=payload, headers=_headers(), timeout=120)
    resp.raise_for_status()
    _print_json(resp.json())


def cmd_chat(args: argparse.Namespace) -> None:
    url = f"{_get_api_url()}/answer"
    history: list[dict[str, str]] = []

    print(f"corprag chat (mode={args.mode}, API={_get_api_url()})")
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
        prog="corprag-cli",
        description="CLI client for the corprag REST API",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest documents")
    p_ingest.add_argument("path", help="Path inside the container or local filesystem")
    p_ingest.add_argument("--replace", action="store_true", help="Replace existing documents")

    # query (retrieve only)
    p_query = sub.add_parser("query", help="Retrieve contexts and sources (no LLM answer)")
    p_query.add_argument("query", help="Search query")
    p_query.add_argument(
        "--mode", default="mix", choices=["local", "global", "hybrid", "naive", "mix"]
    )
    p_query.add_argument("--top-k", type=int, default=None, dest="top_k")

    # answer (single-shot LLM answer)
    p_answer = sub.add_parser(
        "answer", help="Get an LLM-generated answer with contexts and sources"
    )
    p_answer.add_argument("query", help="Question to answer")
    p_answer.add_argument(
        "--mode", default="mix", choices=["local", "global", "hybrid", "naive", "mix"]
    )
    p_answer.add_argument("--top-k", type=int, default=None, dest="top_k")

    # chat (multi-turn REPL)
    p_chat = sub.add_parser("chat", help="Interactive multi-turn conversation")
    p_chat.add_argument(
        "--mode", default="mix", choices=["local", "global", "hybrid", "naive", "mix"]
    )
    p_chat.add_argument("--top-k", type=int, default=None, dest="top_k")

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
