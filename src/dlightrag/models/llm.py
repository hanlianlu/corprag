# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""LLM, embedding, vision, and rerank model factories for RAGAnything integration.

Self-contained — all models are constructed from DlightragConfig without
external dependencies on backend.llm or backend.agents.

Supports multiple LLM providers:
- Category A (OpenAI-compatible): openai, azure_openai, qwen, minimax
- Category B (Dedicated class): anthropic, google_gemini
"""
# pyright: reportCallIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false

from __future__ import annotations

import base64
import logging
import time
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any

from lightrag.utils import EmbeddingFunc

from dlightrag.config import DlightragConfig

logger = logging.getLogger(__name__)

LLMFunc = Callable[..., Awaitable[str]]

# OpenAI-compatible providers that can use ChatOpenAI with custom base_url
_OPENAI_COMPATIBLE_PROVIDERS = {
    "openai",
    "azure_openai",
    "qwen",
    "minimax",
    "ollama",
    "xinference",
    "openrouter",
}


def _ensure_bytes(data: bytes | bytearray | str) -> bytes | None:
    """Normalize image data to bytes from various input formats."""
    if isinstance(data, bytes | bytearray):
        return bytes(data)
    if isinstance(data, str):
        s = data.strip()
        if s.startswith("data:image"):
            try:
                _, b64data = s.split(",", 1)
                return base64.b64decode(b64data)
            except Exception:
                logger.warning("Failed to decode image data URI")
                return None
        try:
            return base64.b64decode(s, validate=True)
        except Exception:
            logger.debug("String image_data is not valid base64; ignoring")
            return None
    return None


# ═══════════════════════════════════════════════════════════════════
# Chat Model Factory
# ═══════════════════════════════════════════════════════════════════



def get_llm_model_func(
    config: DlightragConfig | None = None,
    model_name: str | None = None,
    provider: str | None = None,
) -> LLMFunc:
    """Build a LightRAG-compatible LLM function using native _if_cache functions.

    Returns a partial() with model/api_key/base_url bound. Works both when
    called directly (RAGAnything modal processing) and when called through
    LightRAG's pipeline (which injects hashing_kv).
    """
    from dlightrag.config import get_config

    cfg = config or get_config()
    prov = provider or cfg.llm_provider
    model = model_name or cfg.chat_model_name
    api_key = cfg._get_provider_api_key(prov)
    base_url = cfg._get_url(f"{prov}_base_url")

    if prov in ("openai", "qwen", "minimax", "openrouter", "xinference"):
        from lightrag.llm.openai import openai_complete_if_cache

        return partial(openai_complete_if_cache, model=model, api_key=api_key, base_url=base_url)

    if prov == "azure_openai":
        from lightrag.llm.azure_openai import azure_openai_complete_if_cache

        return partial(
            azure_openai_complete_if_cache,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )

    if prov == "anthropic":
        from lightrag.llm.anthropic import anthropic_complete_if_cache

        return partial(anthropic_complete_if_cache, model=model, api_key=api_key)

    if prov == "google_gemini":
        from lightrag.llm.gemini import gemini_complete_if_cache

        return partial(gemini_complete_if_cache, model=model, api_key=api_key)

    if prov == "ollama":
        from lightrag.llm.ollama import _ollama_model_if_cache

        host = (cfg.ollama_base_url or "http://localhost:11434").removesuffix("/v1")
        return partial(_ollama_model_if_cache, model=model, host=host)

    raise ValueError(f"Unsupported LLM provider: {prov}")


def get_ingestion_llm_model_func(config: DlightragConfig | None = None) -> LLMFunc:
    """Dedicated ingestion LLM — uses ingestion_model_name."""
    from dlightrag.config import get_config

    cfg = config or get_config()
    return get_llm_model_func(cfg, model_name=cfg.ingestion_model_name)


# ═══════════════════════════════════════════════════════════════════
# Vision Model
# ═══════════════════════════════════════════════════════════════════


def get_vision_model_func(
    config: DlightragConfig | None = None,
) -> Callable[..., Awaitable[str]] | None:
    """Vision-language model func for RAGAnything / LightRAG.

    Dispatches to provider-specific builders based on effective_vision_provider.
    """
    from dlightrag.config import get_config

    cfg = config or get_config()
    vision_provider = cfg.effective_vision_provider
    vision_deployment = cfg.vision_model_name
    if not vision_deployment:
        logger.warning("No vision model configured; disabling VLM.")
        return None

    extra = cfg.vision_model_kwargs

    if vision_provider in _OPENAI_COMPATIBLE_PROVIDERS:
        return _build_openai_vision_func(cfg, vision_deployment, vision_provider, extra)
    if vision_provider == "anthropic":
        return _build_anthropic_vision_func(cfg, vision_deployment, extra)
    if vision_provider == "google_gemini":
        return _build_google_vision_func(cfg, vision_deployment, extra)

    logger.warning("Vision not supported for provider: %s", vision_provider)
    return None


def _build_openai_vision_func(
    cfg: DlightragConfig,
    vision_deployment: str,
    provider: str,
    model_kwargs: dict[str, Any] | None = None,
) -> Callable[..., Awaitable[str]]:
    """OpenAI-compatible vision func (covers openai, azure_openai, qwen, minimax)."""
    from openai import AsyncOpenAI

    api_key = cfg._get_provider_api_key(provider)
    base_url = cfg._get_url(f"{provider}_base_url")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=cfg.llm_request_timeout,
        max_retries=cfg.llm_max_retries,
    )
    vision_temp = cfg.vision_temperature
    extra = model_kwargs or {}

    async def vision_model_func(
        prompt: str,
        *,
        system_prompt: str | None = None,
        history_messages: list | None = None,
        image_data: bytes | bytearray | str | None = None,
        messages: list | None = None,
        **_kwargs: object,
    ) -> str:
        """Vision model function compatible with RAG-Anything."""
        # Pattern 1: Pre-formatted messages from RAG-Anything
        if messages is not None:
            t0 = time.perf_counter()
            try:
                resp = await client.chat.completions.create(
                    model=vision_deployment,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=vision_temp,
                    extra_body=extra,
                )
                result = resp.choices[0].message.content or "" if resp.choices else ""
                elapsed = time.perf_counter() - t0
                preview = result[:120].replace("\n", " ")
                logger.info("Vision %.1fs %dc [%s]", elapsed, len(result), preview)
                return result
            except Exception:
                logger.exception("Vision failed %.1fs", time.perf_counter() - t0)
                return ""

        # Pattern 2: Individual image_data parameter
        if image_data is None:
            logger.warning("vision_model_func called without image_data or messages")
            return ""

        raw_bytes = _ensure_bytes(image_data)
        if raw_bytes is None:
            return ""

        b64 = base64.b64encode(raw_bytes).decode("utf-8")
        image_url = f"data:image/png;base64,{b64}"

        msg_list: list[dict[str, Any]] = []
        if system_prompt:
            msg_list.append({"role": "system", "content": system_prompt})
        if history_messages:
            msg_list.extend(history_messages)
        msg_list.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        )

        t0 = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=vision_deployment,
                messages=msg_list,  # type: ignore[arg-type]
                temperature=vision_temp,
                extra_body=extra,
            )
            result = resp.choices[0].message.content or "" if resp.choices else ""
            elapsed = time.perf_counter() - t0
            preview = result[:120].replace("\n", " ")
            logger.info("Vision %.1fs %dc [%s]", elapsed, len(result), preview)
            return result
        except Exception:
            logger.exception("Vision failed %.1fs", time.perf_counter() - t0)
            return ""

    return vision_model_func


def _convert_openai_to_anthropic_messages(
    openai_messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Convert OpenAI-format messages to Anthropic format.

    Returns (messages, system_prompt) since Anthropic separates system messages.
    """
    system_parts: list[str] = []
    anthropic_messages: list[dict[str, Any]] = []

    for msg in openai_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_parts.append(content if isinstance(content, str) else str(content))
            continue

        if isinstance(content, str):
            anthropic_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            blocks: list[dict[str, Any]] = []
            for block in content:
                if block.get("type") == "text":
                    blocks.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "image_url":
                    url = block["image_url"]["url"]
                    if url.startswith("data:"):
                        media_type, _, b64data = url.partition(";base64,")
                        media_type = media_type.replace("data:", "")
                        blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64data,
                                },
                            }
                        )
            if blocks:
                anthropic_messages.append({"role": role, "content": blocks})

    return anthropic_messages, "\n".join(system_parts)


def _build_anthropic_vision_func(
    cfg: DlightragConfig,
    vision_deployment: str,
    model_kwargs: dict[str, Any] | None = None,
) -> Callable[..., Awaitable[str]]:
    """Anthropic vision func using AsyncAnthropic client."""
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(
        api_key=cfg.anthropic_api_key,
        timeout=float(cfg.llm_request_timeout),
        max_retries=cfg.llm_max_retries,
    )
    vision_temp = cfg.vision_temperature

    async def vision_model_func(
        prompt: str,
        *,
        system_prompt: str | None = None,
        history_messages: list | None = None,
        image_data: bytes | bytearray | str | None = None,
        messages: list | None = None,
        **_kwargs: object,
    ) -> str:
        # Pattern 1: Pre-formatted messages from RAG-Anything
        if messages is not None:
            try:
                ant_msgs, sys = _convert_openai_to_anthropic_messages(messages)
                resp = await client.messages.create(
                    model=vision_deployment,
                    messages=ant_msgs,  # type: ignore[arg-type]
                    system=sys or (system_prompt or ""),
                    max_tokens=4096,
                    temperature=vision_temp,
                )
                return resp.content[0].text if resp.content else ""
            except Exception:
                logger.exception("Anthropic vision call failed (messages mode)")
                return ""

        # Pattern 2: Individual image_data
        if image_data is None:
            logger.warning("vision_model_func called without image_data or messages")
            return ""

        raw_bytes = _ensure_bytes(image_data)
        if raw_bytes is None:
            return ""

        b64 = base64.b64encode(raw_bytes).decode("utf-8")
        ant_msgs_list: list[dict[str, Any]] = []
        if history_messages:
            for hm in history_messages:
                ant_msgs_list.append(
                    {
                        "role": hm.get("role", "user"),
                        "content": str(hm.get("content", "")),
                    }
                )
        ant_msgs_list.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                ],
            }
        )

        try:
            resp = await client.messages.create(
                model=vision_deployment,
                messages=ant_msgs_list,  # type: ignore[arg-type]
                system=system_prompt or "",
                max_tokens=4096,
                temperature=vision_temp,
            )
            return resp.content[0].text if resp.content else ""
        except Exception:
            logger.exception("Anthropic vision call failed (image_data mode)")
            return ""

    return vision_model_func


def _build_google_vision_func(
    cfg: DlightragConfig,
    vision_deployment: str,
    model_kwargs: dict[str, Any] | None = None,
) -> Callable[..., Awaitable[str]]:
    """Google Gemini vision func using google-genai client."""
    from google import genai
    from google.genai.types import Content, Part

    client = genai.Client(api_key=cfg.google_gemini_api_key)
    vision_temp = cfg.vision_temperature

    async def vision_model_func(
        prompt: str,
        *,
        system_prompt: str | None = None,
        history_messages: list | None = None,
        image_data: bytes | bytearray | str | None = None,
        messages: list | None = None,
        **_kwargs: object,
    ) -> str:
        # Pattern 1: Pre-formatted messages from RAG-Anything
        if messages is not None:
            try:
                contents: list[Content] = []
                for msg in messages:
                    role = "user" if msg.get("role") in ("user", "system") else "model"
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        contents.append(Content(role=role, parts=[Part.from_text(text=content)]))
                    elif isinstance(content, list):
                        parts: list[Part] = []
                        for block in content:
                            if block.get("type") == "text":
                                parts.append(Part.from_text(text=block["text"]))
                            elif block.get("type") == "image_url":
                                url = block["image_url"]["url"]
                                if url.startswith("data:"):
                                    _, _, b64data = url.partition(";base64,")
                                    img_bytes = base64.b64decode(b64data)
                                    parts.append(
                                        Part.from_bytes(data=img_bytes, mime_type="image/png")
                                    )
                        if parts:
                            contents.append(Content(role=role, parts=parts))

                resp = await client.aio.models.generate_content(
                    model=vision_deployment,
                    contents=contents,
                    config={"temperature": vision_temp},
                )
                return resp.text or ""
            except Exception:
                logger.exception("Google vision call failed (messages mode)")
                return ""

        # Pattern 2: Individual image_data
        if image_data is None:
            logger.warning("vision_model_func called without image_data or messages")
            return ""

        raw_bytes = _ensure_bytes(image_data)
        if raw_bytes is None:
            return ""

        parts_list: list[Part] = []
        if system_prompt:
            parts_list.append(Part.from_text(text=system_prompt))
        parts_list.append(Part.from_text(text=prompt))
        parts_list.append(Part.from_bytes(data=raw_bytes, mime_type="image/png"))

        try:
            resp = await client.aio.models.generate_content(
                model=vision_deployment,
                contents=Content(parts=parts_list),
                config={"temperature": vision_temp},
            )
            return resp.text or ""
        except Exception:
            logger.exception("Google vision call failed (image_data mode)")
            return ""

    return vision_model_func


# ═══════════════════════════════════════════════════════════════════
# Embedding
# ═══════════════════════════════════════════════════════════════════


def get_embedding_func(config: DlightragConfig | None = None) -> EmbeddingFunc:
    """Get embedding function using LightRAG native embed functions.

    Dispatches based on effective_embedding_provider.
    """
    from dlightrag.config import get_config

    cfg = config or get_config()
    emb_provider = cfg.effective_embedding_provider
    api_key = cfg._get_provider_api_key(emb_provider)
    base_url = cfg._get_url(f"{emb_provider}_base_url")

    if emb_provider == "google_gemini":
        from lightrag.llm.gemini import gemini_embed
        raw_fn = partial(gemini_embed.func, model=cfg.embedding_model, api_key=api_key)
    elif emb_provider == "ollama":
        from lightrag.llm.ollama import ollama_embed
        host = (cfg.ollama_base_url or "http://localhost:11434").removesuffix("/v1")
        raw_fn = partial(ollama_embed.func, embed_model=cfg.embedding_model, host=host)
    else:
        # OpenAI-compatible: openai, qwen, minimax, xinference, openrouter, azure_openai
        from lightrag.llm.openai import openai_embed
        raw_fn = partial(
            openai_embed.func, model=cfg.embedding_model,
            api_key=api_key, base_url=base_url,
        )

    return EmbeddingFunc(
        embedding_dim=cfg.embedding_dim,
        max_token_size=8192,
        func=raw_fn,
        model_name=cfg.embedding_model,
    )


# ═══════════════════════════════════════════════════════════════════
# Reranking
# ═══════════════════════════════════════════════════════════════════


def get_rerank_func(config: DlightragConfig | None = None) -> Callable:
    """Return rerank function based on configured backend.

    Backends:
    - "llm": LLM-based listwise reranker (uses current llm_provider)
    - "cohere": Cohere-compatible API (works with Cohere, Xinference, LiteLLM, etc.)
    - "jina": Jina-compatible API
    - "aliyun": Aliyun DashScope API
    - "azure_cohere": Azure AI Foundry Cohere deployment (custom auth header)

    The cohere/jina/aliyun backends can target any compatible endpoint via
    DLIGHTRAG_RERANK_BASE_URL, following the same pattern as LightRAG's
    --rerank-binding + --rerank-binding-host.
    """
    from dlightrag.config import get_config

    cfg = config or get_config()

    if cfg.rerank_backend in ("cohere", "jina", "aliyun"):
        from lightrag.rerank import ali_rerank, cohere_rerank, jina_rerank

        rerank_functions = {
            "cohere": cohere_rerank,
            "jina": jina_rerank,
            "aliyun": ali_rerank,
        }
        selected_func = rerank_functions[cfg.rerank_backend]
        model = cfg.effective_rerank_model
        api_key = cfg.effective_rerank_api_key

        kwargs: dict[str, Any] = {"api_key": api_key, "model": model}
        if cfg.rerank_base_url:
            kwargs["base_url"] = cfg.rerank_base_url

        logger.info(
            "Reranker: backend=%s, model=%s, base_url=%s",
            cfg.rerank_backend,
            model,
            cfg.rerank_base_url or "(provider default)",
        )
        return partial(selected_func, **kwargs)

    if cfg.rerank_backend == "azure_cohere":
        logger.info("Reranker: backend=azure_cohere, model=%s", cfg.effective_rerank_model)
        return _build_azure_cohere_rerank_func(cfg)

    logger.info(
        "Reranker: backend=llm (%s), model=%s",
        cfg.effective_rerank_llm_provider,
        cfg.effective_rerank_model,
    )
    return _build_llm_rerank_func(cfg)


def _json_kwargs_for_provider(provider: str) -> dict[str, Any]:
    """Return provider-specific kwargs for JSON output mode."""
    if provider in ("openai", "azure_openai", "qwen", "minimax", "openrouter", "xinference"):
        return {"response_format": {"type": "json_object"}}
    if provider == "ollama":
        return {"format": "json"}
    if provider == "google_gemini":
        return {"generation_config": {"response_mime_type": "application/json"}}
    return {}  # anthropic: prompt-only


def _extract_json(text: str) -> str:
    """Extract JSON from LLM response (handles markdown fences)."""
    import re

    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    # Try to find raw JSON object — match outermost braces
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return text[start:]
    return text


def _build_llm_rerank_func(config: DlightragConfig) -> Callable:
    """LLM-based listwise reranker using native LLM function + Pydantic parsing.

    3-layer defense:
    1. Native JSON mode where available (via provider-specific kwargs)
    2. Prompt always instructs JSON format (universal fallback)
    3. Pydantic model_validate_json() as safety net
    """
    from dlightrag.models.schemas import RerankResult

    provider = config.effective_rerank_llm_provider
    json_kwargs = _json_kwargs_for_provider(provider)

    llm_func = get_llm_model_func(
        config,
        model_name=config.effective_rerank_model,
        provider=provider,
    )

    default_domain_knowledge = config.domain_knowledge_hints

    async def rerank_func(
        query: str,
        documents: list[str],
        domain_knowledge: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        if not documents:
            return []

        effective_domain_knowledge = domain_knowledge or default_domain_knowledge
        doc_lines = "\n".join([f"[{idx}] {doc}" for idx, doc in enumerate(documents)])

        system_parts = [
            "You are a reranker. Given a query and a list of chunks, rank them by relevance.",
            '\nRespond with JSON only: {"ranked_chunks": [{"index": 0, "relevance_score": 0.95}, ...]}',
        ]
        if effective_domain_knowledge:
            system_parts.append(f"\n{effective_domain_knowledge}")

        user_content = f"Query: {query}\n\nChunks:\n{doc_lines}"

        try:
            result_str = await llm_func(
                user_content,
                system_prompt="".join(system_parts),
                **json_kwargs,
            )

            parsed = RerankResult.model_validate_json(_extract_json(result_str))

            seen: set[int] = set()
            results = []
            for chunk in parsed.ranked_chunks:
                if 0 <= chunk.index < len(documents) and chunk.index not in seen:
                    seen.add(chunk.index)
                    results.append({"index": chunk.index, "relevance_score": chunk.relevance_score})

            # Append any chunks LLM didn't return (preserves all chunks, just sorted)
            if len(results) < len(documents):
                min_score = results[-1]["relevance_score"] if results else 0.5
                for idx in range(len(documents)):
                    if idx not in seen:
                        results.append(
                            {"index": idx, "relevance_score": max(0.0, min_score - 0.01)}
                        )

            return results or _fallback_ranking(len(documents))
        except Exception as exc:
            logger.warning("Rerank failed: %s", exc)
            return _fallback_ranking(len(documents))

    return rerank_func


def _fallback_ranking(n: int) -> list[dict[str, float]]:
    """Return original order ranking as fallback."""
    return [{"index": i, "relevance_score": 1.0 - i * 0.01} for i in range(n)]


def _build_azure_cohere_rerank_func(config: DlightragConfig) -> Callable:
    """Azure AI Services Cohere reranker via REST API."""
    import httpx

    endpoint = config._get_url("rerank_base_url")
    api_key = config.rerank_api_key
    deployment = config.effective_rerank_model

    if not endpoint or not api_key:
        raise ValueError("rerank_base_url and rerank_api_key required for azure_cohere backend")
    rerank_url = (
        endpoint
        if "/providers/cohere/" in endpoint or endpoint.endswith("/rerank")
        else f"{endpoint}/providers/cohere/v2/rerank"
    )

    async def azure_cohere_rerank(
        query: str,
        documents: list[str],
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        if not documents:
            return []

        async with httpx.AsyncClient(timeout=30.0) as http_client:
            try:
                response = await http_client.post(
                    rerank_url,
                    headers={"api-key": api_key, "Content-Type": "application/json"},
                    json={"model": deployment, "query": query, "documents": documents},
                )
                response.raise_for_status()
                return response.json().get("results", [])
            except httpx.HTTPStatusError as e:
                logger.error(f"Azure Cohere rerank error: {e.response.status_code}")
                raise
            except Exception as e:
                logger.exception(f"Azure Cohere rerank failed: {e}")
                raise

    return azure_cohere_rerank


__all__ = [
    "LLMFunc",
    "get_embedding_func",
    "get_ingestion_llm_model_func",
    "get_llm_model_func",
    "get_rerank_func",
    "get_vision_model_func",
]
