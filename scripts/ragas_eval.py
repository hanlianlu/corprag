#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""RAGAS evaluation adapter for DlightRAG.

Reuses LightRAG's built-in :class:`RAGEvaluator` — RAGAS metrics
(Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision),
two-stage pipeline concurrency, progress bars, and CSV/JSON export.

Only :meth:`generate_rag_response` is overridden to call DlightRAG's
``/answer`` instead of LightRAG's ``/query``.

When ``EVAL_LLM_BINDING_API_KEY`` is not set, the adapter auto-resolves
eval credentials from DlightRAG's own OpenAI-compatible query config,
cascading down to the embedding config when provider-compatible.
No extra ``.env`` entries needed in the common case.

Usage::

    uv sync --group eval
    uv run python scripts/ragas_eval.py --dataset my_questions.json

See `docs/evaluation.md <../docs/evaluation.md>`_ for full guide.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, cast

import httpx
from dotenv import load_dotenv
from lightrag.evaluation.eval_rag_quality import RAGEvaluator
from lightrag.utils import logger

from dlightrag.adapters.http.client import (
    AnswerRunClient,
    RunCancelledError,
    RunFailedError,
)
from dlightrag.engine.ai.settings import ModelSettings


class EvalError(RuntimeError):
    """A DlightRAG API call failed while generating an evaluation response."""


# ═══════════════════════════════════════════════════════════════════
# Auto-resolve eval credentials from DlightRAG config
# ═══════════════════════════════════════════════════════════════════

# OpenAI v1 embedding protocols — their api_key + base_url work with
# langchain's OpenAIEmbeddings. Native transports are excluded.
_OPENAI_COMPATIBLE_LLM_PROVIDERS = frozenset({"openai"})
_OPENAI_COMPATIBLE_EMBED_PROVIDERS = frozenset({"openai", "openai_compatible"})
DEFAULT_RESULTS_DIR = Path("ragas_eval_results")


def _resolve_eval_env() -> None:
    """Set EVAL_* env vars from DlightRAG config when not explicitly configured.

    Cascade (each level only applies when the env var is unset):

    Eval LLM:
      1. EVAL_LLM_BINDING_API_KEY  ← config.models.chat.roles.query.api_key
                                    ← config.models.chat.default.api_key
      2. EVAL_LLM_MODEL            ← query role model  ← default model
      3. EVAL_LLM_BINDING_HOST     ← query role base_url ← default base_url

    Eval embeddings:
      4. EVAL_EMBEDDING_BINDING_API_KEY  ← EVAL_LLM_BINDING_API_KEY
                                          ← DlightRAG embedding key (if OpenAI-compatible)
      5. EVAL_EMBEDDING_BINDING_HOST     ← EVAL_LLM_BINDING_HOST
                                          ← DlightRAG embedding base_url (if OpenAI-compatible)

    DlightRAG connection:
      6. DLIGHTRAG_API_URL              ← config.interfaces.api.host:api_port
      7. DLIGHTRAG_API_TOKEN            ← config.access.api_token (simple)
    """
    llm_key_set = bool(os.getenv("EVAL_LLM_BINDING_API_KEY"))
    embed_key_set = bool(os.getenv("EVAL_EMBEDDING_BINDING_API_KEY"))
    eval_llm_available = llm_key_set or bool(os.getenv("OPENAI_API_KEY"))
    eval_embedding_available = embed_key_set or eval_llm_available
    if (
        eval_llm_available
        and eval_embedding_available
        and os.getenv("EVAL_LLM_MODEL")
        and os.getenv("DLIGHTRAG_API_URL")
    ):
        return

    try:
        from dlightrag.application.config import DlightragConfig
    except ImportError:
        logger.warning("DlightRAG not importable — skipping eval/API auto-resolution")
        return

    try:
        config = DlightragConfig()  # pyright: ignore[reportCallIssue]
    except Exception:
        logger.warning(
            "DlightRAG config failed to load — set missing eval/API values explicitly "
            "via EVAL_LLM_BINDING_API_KEY and --api / DLIGHTRAG_API_URL. "
            "Run from the repo root where config.yaml exists for auto-resolution.",
            exc_info=True,
        )
        return
    chat = config.models.chat
    resolver = getattr(chat, "resolve", None)
    query_cfg = cast(
        ModelSettings,
        resolver("query") if callable(resolver) else chat.roles.query or chat.default,
    )
    query_is_openai_compatible = query_cfg.provider in _OPENAI_COMPATIBLE_LLM_PROVIDERS

    # -- Eval LLM --
    if not llm_key_set and query_is_openai_compatible:
        if query_cfg.api_key:
            os.environ["EVAL_LLM_BINDING_API_KEY"] = query_cfg.api_key
            logger.info("Eval LLM key: auto-resolved from DlightRAG query/default role")
        elif os.getenv("OPENAI_API_KEY"):
            logger.info("Eval LLM key: using OPENAI_API_KEY")
        else:
            logger.warning(
                "No eval LLM key found — set EVAL_LLM_BINDING_API_KEY, "
                "DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY, or OPENAI_API_KEY"
            )
    elif not eval_llm_available:
        logger.warning(
            "DlightRAG query provider '%s' is not OpenAI-compatible — set "
            "EVAL_LLM_BINDING_API_KEY or OPENAI_API_KEY for RAGAS evaluation",
            query_cfg.provider,
        )

    if query_is_openai_compatible and not os.getenv("EVAL_LLM_MODEL"):
        os.environ["EVAL_LLM_MODEL"] = query_cfg.model
        logger.info("Eval LLM model: %s (from DlightRAG config)", query_cfg.model)

    if query_is_openai_compatible and not os.getenv("EVAL_LLM_BINDING_HOST") and query_cfg.base_url:
        os.environ["EVAL_LLM_BINDING_HOST"] = query_cfg.base_url
        logger.info("Eval LLM host: %s (from DlightRAG config)", query_cfg.base_url)

    # -- Eval Embeddings --
    if not embed_key_set:
        # Cascade: EVAL_LLM key → DlightRAG embedding key (if OpenAI-compatible provider)
        resolved_embed_key = os.getenv("EVAL_LLM_BINDING_API_KEY")
        if (
            not resolved_embed_key
            and config.models.embedding.provider in _OPENAI_COMPATIBLE_EMBED_PROVIDERS
        ):
            resolved_embed_key = config.models.embedding.api_key
        if resolved_embed_key:
            os.environ["EVAL_EMBEDDING_BINDING_API_KEY"] = resolved_embed_key
            logger.info("Eval embedding key: cascaded from eval LLM or DlightRAG embedding config")

    if not os.getenv("EVAL_EMBEDDING_BINDING_HOST"):
        # Cascade: EVAL_LLM host → DlightRAG embedding host (if OpenAI-compatible)
        llm_host = os.getenv("EVAL_LLM_BINDING_HOST")
        embed_cfg = config.models.embedding
        if llm_host:
            os.environ["EVAL_EMBEDDING_BINDING_HOST"] = llm_host
        elif embed_cfg.provider in _OPENAI_COMPATIBLE_EMBED_PROVIDERS and embed_cfg.base_url:
            os.environ["EVAL_EMBEDDING_BINDING_HOST"] = embed_cfg.base_url

    # -- DlightRAG API URL (--api / $DLIGHTRAG_API_URL / config) --
    if not os.getenv("DLIGHTRAG_API_URL") and config.interfaces.api.host:
        os.environ["DLIGHTRAG_API_URL"] = (
            f"http://{config.interfaces.api.host}:{config.interfaces.api.port}"
        )
        logger.info(
            "DlightRAG API URL: auto-resolved from config (%s)", os.environ["DLIGHTRAG_API_URL"]
        )

    # -- DlightRAG API token (simple auth only; JWT tokens come from the issuer) --
    if not os.getenv("DLIGHTRAG_API_TOKEN"):
        if config.access.auth_mode == "simple" and config.access.api_token:
            os.environ["DLIGHTRAG_API_TOKEN"] = config.access.api_token
            logger.info("DlightRAG API token: auto-resolved from config (simple auth)")


# ═══════════════════════════════════════════════════════════════════
# Adapter
# ═══════════════════════════════════════════════════════════════════


class DlightRAGAdapterEvaluator(RAGEvaluator):
    """RAGEvaluator wired to a DlightRAG ``/answer`` endpoint.

    Inherits everything — RAGAS metrics, concurrency, tqdm, CSV/JSON export —
    and only overrides the API-call method to speak DlightRAG's response format.
    """

    def __init__(
        self,
        test_dataset_path: str | None = None,
        rag_api_url: str | None = None,
        *,
        api_key: str | None = None,
    ) -> None:
        # CLI --api-key > $DLIGHTRAG_API_TOKEN (auto-resolved in _resolve_eval_env)
        self._dlightrag_api_key = api_key or os.getenv("DLIGHTRAG_API_TOKEN")
        super().__init__(  # pyright: ignore[reportArgumentType]
            test_dataset_path=test_dataset_path,  # type: ignore[arg-type]
            rag_api_url=rag_api_url,  # type: ignore[arg-type]
        )

    # ---------------------------------------------------------------- #
    #  The ONLY overridden method — format translation                 #
    # ---------------------------------------------------------------- #

    async def generate_rag_response(
        self,
        question: str,
        client: httpx.AsyncClient,
    ) -> dict[str, Any]:
        """Run one durable DlightRAG answer and translate it to LightRAG format.

        DlightRAG canonical result::

            {"answer": "...",
             "contexts": {"chunks": [{"content": "...", "chunk_id": "..."}]},
             "references": [...], "sources": [...], "trace": {...}}

        Translated to LightRAG RAGEvaluator format::

            {"answer": "...",
             "contexts": ["chunk text 1", "chunk text 2", ...]}
        """
        headers: dict[str, str] = {}
        if self._dlightrag_api_key:
            headers["Authorization"] = f"Bearer {self._dlightrag_api_key}"
        runs = AnswerRunClient(client, base_url=self.rag_api_url, headers=headers)
        try:
            result = await runs.answer(
                {
                    "query": question,
                    "top_k": int(os.getenv("EVAL_QUERY_TOP_K", "10")),
                }
            )
        except RunCancelledError as exc:
            raise EvalError(f"DlightRAG answer run was cancelled for: {question[:80]}") from exc
        except RunFailedError as exc:
            raise EvalError(
                f"DlightRAG answer run failed ({exc.error_kind}): {exc.public_message}"
            ) from exc
        except httpx.ConnectError as exc:
            raise EvalError(
                f"Cannot connect to DlightRAG API at {self.rag_api_url}/answer\n"
                f"  Make sure DlightRAG is running: docker compose up -d\n"
                f"  Error: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise EvalError(
                f"DlightRAG API error {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.ReadTimeout as exc:
            raise EvalError(
                f"Request timeout waiting for DlightRAG response\n"
                f"  Question: {question[:100]}...\n"
                f"  Error: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise EvalError(f"DlightRAG API request failed: {type(exc).__name__}: {exc}") from exc

        answer = result.answer or "No response generated"
        chunks = result.contexts.get("chunks", [])

        # RAGAS scores textual grounding; display-only media blocks stay out.
        contexts: list[str] = []
        for chunk in chunks:
            content = chunk.get("content", "")
            if isinstance(content, str) and content.strip():
                contexts.append(content)

        if not contexts:
            logger.warning("Eval query returned no chunk content for: %s", question[:80])

        return {"answer": answer, "contexts": contexts}


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAGAS evaluation for DlightRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        suggest_on_error=True,
        epilog="""
Examples:
  # Zero-config for no-auth/simple-auth setups
  python scripts/ragas_eval.py --dataset my_questions.json

  # Explicit API URL (when running outside the repo or remote)
  python scripts/ragas_eval.py --api http://localhost:8100 --dataset my_questions.json

  # Explicit eval model overrides (overrides auto-resolution)
  EVAL_LLM_MODEL=gpt-4o EVAL_EMBEDDING_MODEL=text-embedding-3-large \\
    python scripts/ragas_eval.py --dataset my_questions.json
        """,
    )

    parser.add_argument(
        "--api",
        type=str,
        default=os.getenv("DLIGHTRAG_API_URL"),
        help="DlightRAG API base URL (auto-resolved from $DLIGHTRAG_API_URL or config). "
        "Example: http://localhost:8100",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("DLIGHTRAG_API_TOKEN"),
        help="Bearer token when auth_mode is 'simple' or 'jwt' (default: $DLIGHTRAG_API_TOKEN).",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help='Required test dataset JSON file. Format: {"test_cases": '
        '[{"question": "...", "ground_truth": "..."}]}',
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory for CSV/JSON results (default: ./ragas_eval_results/).",
    )

    return parser


# ═══════════════════════════════════════════════════════════════════
# Shared env reference (informational — shown on startup)
# ═══════════════════════════════════════════════════════════════════

_EXPECTED_ENV_VARS: dict[str, str] = {
    "EVAL_LLM_MODEL": "LLM for RAGAS scoring (default: from DlightRAG config, or gpt-4o-mini)",
    "EVAL_LLM_BINDING_API_KEY": "API key for eval LLM (auto-resolved from DlightRAG config)",
    "EVAL_LLM_BINDING_HOST": "Custom endpoint for eval LLM (auto-resolved from DlightRAG config)",
    "EVAL_EMBEDDING_MODEL": "Embedding model for RAGAS (default: text-embedding-3-large)",
    "EVAL_EMBEDDING_BINDING_API_KEY": "API key for eval embeddings (cascaded from eval LLM key)",
    "EVAL_EMBEDDING_BINDING_HOST": "Custom endpoint for eval embeddings (cascaded from eval LLM host)",
    "DLIGHTRAG_API_URL": "DlightRAG API base URL (default for --api)",
    "DLIGHTRAG_API_TOKEN": "Bearer token (auto from simple config, explicit for JWT)",
    "EVAL_QUERY_TOP_K": "top_k sent to DlightRAG /answer (default: 10)",
    "EVAL_MAX_CONCURRENT": "RAGAS evaluation concurrency (default: 2)",
    "EVAL_LLM_MAX_RETRIES": "Max retries for eval LLM calls (default: 5)",
    "EVAL_LLM_TIMEOUT": "Timeout per eval LLM call, seconds (default: 180)",
}


def _check_env() -> None:
    """Print configured environment variables (informational)."""
    logger.info("Environment variables (set → value, unset → <auto>):")
    for var, description in _EXPECTED_ENV_VARS.items():
        value = os.getenv(var)
        if value:
            display = value if "KEY" not in var and "TOKEN" not in var else "***"
            logger.info("  %-34s = %s  # %s", var, display, description)
        else:
            logger.info("  %-34s   <auto>  # %s", var, description)
    logger.info("")


async def _run() -> None:
    # Load .env before argparse reads env-backed defaults; resolve project config
    # after argparse so `--help` exits before credential auto-resolution logs.
    load_dotenv(dotenv_path=".env", override=False)
    args = build_parser().parse_args()
    if args.api:
        os.environ["DLIGHTRAG_API_URL"] = args.api
    if args.api_key:
        os.environ["DLIGHTRAG_API_TOKEN"] = args.api_key
    _resolve_eval_env()
    args.api = args.api or os.getenv("DLIGHTRAG_API_URL")

    if not args.api:
        print(
            "DlightRAG API URL not found. Run from the repo root so config.yaml is "
            "visible, or set --api / $DLIGHTRAG_API_URL.",
            file=sys.stderr,
        )
        sys.exit(1)

    _check_env()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error("No test dataset found at %s", dataset_path)
        logger.error(
            "Pass --dataset <your_questions.json> with ingested documents.\n"
            '  Format: {"test_cases": [{"question": "...", "ground_truth": "..."}]}\n'
            "  See docs/evaluation.md for the full guide."
        )
        sys.exit(1)

    evaluator = DlightRAGAdapterEvaluator(
        test_dataset_path=str(dataset_path),
        rag_api_url=args.api.rstrip("/"),
        api_key=args.api_key,
    )

    evaluator.results_dir = Path(args.output_dir) if args.output_dir else DEFAULT_RESULTS_DIR
    evaluator.results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("DlightRAG API: %s/answer", evaluator.rag_api_url)
    logger.info("Eval LLM:     %s", evaluator.eval_model)
    logger.info("Eval Embed:   %s", evaluator.eval_embedding_model)
    logger.info("Results dir:  %s", evaluator.results_dir.absolute())
    logger.info("")

    await evaluator.run()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
