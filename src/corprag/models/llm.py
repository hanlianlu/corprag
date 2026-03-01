# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""LLM, embedding, vision, and rerank model factories for RAGAnything integration.

Self-contained — all models are constructed from CorpragConfig without
external dependencies on backend.llm or backend.agents.
"""

from __future__ import annotations

import base64
from collections.abc import Awaitable, Callable
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from lightrag.utils import EmbeddingFunc
import numpy as np
from pydantic import SecretStr

from corprag.config import CorpragConfig

logger = logging.getLogger(__name__)

LLMFunc = Callable[..., Awaitable[str]]


# ═══════════════════════════════════════════════════════════════════
# LLM Factories
# ═══════════════════════════════════════════════════════════════════


def _build_chat_openai(
    config: CorpragConfig,
    model: str,
    temperature: float | None = None,
) -> ChatOpenAI:
    """Build a ChatOpenAI instance from config."""
    return ChatOpenAI(  # type: ignore[call-arg]
        model=model,
        api_key=SecretStr(config.unified_api_key),
        base_url=config.unified_base_url,
        temperature=temperature if temperature is not None else config.llm_temperature,
        timeout=config.llm_request_timeout,
        max_retries=config.llm_max_retries,
    )


def _build_llm_model_func(config: CorpragConfig, deployment: str, temperature: float | None = None) -> LLMFunc:
    """Build a LightRAG-compatible LLM function from config."""
    llm = _build_chat_openai(config, deployment, temperature)

    async def llm_model_func(
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        **_: Any,
    ) -> str:
        """LLM func accepted by LightRAG & RAGAnything."""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        try:
            model = llm.bind(max_tokens=max_tokens) if max_tokens else llm
            resp = await model.ainvoke(messages)
            return str(resp.content) if resp.content else ""
        except Exception:
            logger.exception("LLM call failed")
            return ""

    return llm_model_func


def get_llm_model_func(config: CorpragConfig | None = None) -> LLMFunc:
    """Primary chat LLM (retrieval/generation)."""
    from corprag.config import get_config

    cfg = config or get_config()
    return _build_llm_model_func(cfg, cfg.chat_model_name)


def get_ingestion_llm_model_func(config: CorpragConfig | None = None) -> LLMFunc:
    """Dedicated ingestion LLM with lower temperature for deterministic extraction."""
    from corprag.config import get_config

    cfg = config or get_config()
    return _build_llm_model_func(
        cfg,
        cfg.ingestion_model_name,
        temperature=cfg.ingestion_temperature,
    )


# ═══════════════════════════════════════════════════════════════════
# Vision Model
# ═══════════════════════════════════════════════════════════════════


def get_vision_model_func(config: CorpragConfig | None = None) -> Callable[..., Awaitable[str]] | None:
    """Vision-language model func for RAGAnything / LightRAG.

    Uses raw OpenAI client for multimodal message format support.
    """
    from corprag.config import get_config

    cfg = config or get_config()
    vision_deployment = cfg.vision_model_name
    if not vision_deployment:
        logger.warning("No vision model configured; disabling VLM.")
        return None

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=cfg.unified_api_key,
        base_url=cfg.unified_base_url,
        timeout=cfg.llm_request_timeout,
        max_retries=cfg.llm_max_retries,
    )
    vision_temp = cfg.vision_temperature

    def _ensure_bytes(data: bytes | bytearray | str) -> bytes | None:
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

    async def vision_model_func(
        prompt: str,
        *,
        system_prompt: str | None = None,
        history_messages: list | None = None,
        image_data: bytes | bytearray | str | None = None,
        messages: list | None = None,
        **kwargs: object,
    ) -> str:
        """Vision model function compatible with RAG-Anything."""
        # Pattern 1: Pre-formatted messages from RAG-Anything
        if messages is not None:
            try:
                resp = await client.chat.completions.create(
                    model=vision_deployment,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=vision_temp,
                )
                return resp.choices[0].message.content or "" if resp.choices else ""
            except Exception:
                logger.exception("Vision call failed (messages mode)")
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

        try:
            resp = await client.chat.completions.create(
                model=vision_deployment,
                messages=msg_list,  # type: ignore[arg-type]
                temperature=vision_temp,
            )
            return resp.choices[0].message.content or "" if resp.choices else ""
        except Exception:
            logger.exception("Vision call failed (image_data mode)")
            return ""

    return vision_model_func


# ═══════════════════════════════════════════════════════════════════
# Embedding
# ═══════════════════════════════════════════════════════════════════


def get_embedding_func(config: CorpragConfig | None = None) -> EmbeddingFunc:
    """Get embedding function for RAGAnything."""
    from corprag.config import get_config

    cfg = config or get_config()
    embeddings = OpenAIEmbeddings(  # type: ignore[call-arg]
        model=cfg.embedding_model,
        api_key=SecretStr(cfg.unified_api_key),
        base_url=cfg.unified_base_url,
        dimensions=cfg.embedding_dim,
    )

    async def embed_func(texts: list[str]) -> np.ndarray:
        try:
            vectors = await embeddings.aembed_documents(texts)
            return np.asarray(vectors, dtype=np.float32)
        except Exception:
            logger.exception("Embedding call failed")
            return np.array([])

    return EmbeddingFunc(
        embedding_dim=cfg.embedding_dim,
        max_token_size=8192,
        func=embed_func,
    )


# ═══════════════════════════════════════════════════════════════════
# Reranking
# ═══════════════════════════════════════════════════════════════════


def get_rerank_func(config: CorpragConfig | None = None) -> Callable:
    """Return rerank function based on configured provider.

    Providers:
    - "llm": LLM-based listwise reranker
    - "cohere": Cohere REST API
    - "azure_cohere": Azure AI Services Cohere deployment
    """
    from corprag.config import get_config

    cfg = config or get_config()

    if cfg.rerank_provider == "cohere":
        from functools import partial

        from lightrag.rerank import cohere_rerank

        logger.info("Reranker: provider=cohere, model=%s", cfg.cohere_rerank_model)
        return partial(
            cohere_rerank,
            api_key=cfg.cohere_api_key,
            model=cfg.cohere_rerank_model,
        )

    if cfg.rerank_provider == "azure_cohere":
        logger.info("Reranker: provider=azure_cohere, model=%s", cfg.azure_cohere_deployment)
        return _build_azure_cohere_rerank_func(cfg)

    logger.info("Reranker: provider=llm, model=%s", cfg.rerank_model)
    return _build_llm_rerank_func(cfg)


def _build_llm_rerank_func(config: CorpragConfig) -> Callable:
    """LLM-based listwise reranker using structured output."""
    from corprag.models.schemas import RerankResult

    llm = _build_chat_openai(config, config.rerank_model, temperature=config.rerank_temperature)
    structured_llm = llm.with_structured_output(RerankResult)
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
            "You are a reranker. Given a query and a list of chunks, rank them by relevance."
        ]
        if effective_domain_knowledge:
            system_parts.append(f"\n{effective_domain_knowledge}")

        user_content = f"Query: {query}\n\nChunks:\n{doc_lines}"

        try:
            result: RerankResult = await structured_llm.ainvoke(
                [
                    SystemMessage(content="".join(system_parts)),
                    HumanMessage(content=user_content),
                ]
            )  # type: ignore[assignment]

            seen: set[int] = set()
            results = []
            for chunk in result.ranked_chunks:
                if 0 <= chunk.index < len(documents) and chunk.index not in seen:
                    seen.add(chunk.index)
                    results.append(
                        {"index": chunk.index, "relevance_score": chunk.relevance_score}
                    )

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


def _build_azure_cohere_rerank_func(config: CorpragConfig) -> Callable:
    """Azure AI Services Cohere reranker via REST API."""
    import httpx

    endpoint = config.azure_cohere_endpoint
    api_key = config.azure_cohere_api_key
    deployment = config.azure_cohere_deployment

    if not endpoint or not api_key:
        raise ValueError(
            "azure_cohere_endpoint and azure_cohere_api_key required for azure_cohere"
        )

    endpoint = endpoint.rstrip("/")
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
