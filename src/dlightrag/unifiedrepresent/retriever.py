# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Visual retrieval pipeline for unified representational RAG.

Handles query processing: LightRAG KG retrieval -> visual chunk resolution ->
visual reranking -> VLM answer generation.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any, Literal, cast

import httpx
from lightrag import LightRAG, QueryParam
from lightrag.constants import GRAPH_FIELD_SEP

QueryMode = Literal["local", "global", "hybrid", "naive", "mix", "bypass"]

logger = logging.getLogger(__name__)


class VisualRetriever:
    """Query pipeline for unified representational RAG (Phases 2-4)."""

    def __init__(
        self,
        lightrag: LightRAG,
        visual_chunks: Any,  # BaseKVStorage instance
        config: Any,  # DlightragConfig
        vision_model_func: Callable | None = None,
        rerank_model: str | None = None,
        rerank_base_url: str | None = None,
        rerank_api_key: str | None = None,
        rerank_backend: str | None = None,
    ) -> None:
        self.lightrag = lightrag
        self.visual_chunks = visual_chunks
        self.config = config
        self.vision_model_func = vision_model_func
        self.rerank_model = rerank_model
        self.rerank_base_url = rerank_base_url
        self.rerank_api_key = rerank_api_key
        self.rerank_backend = rerank_backend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        mode: str = "mix",
        top_k: int = 60,
        chunk_top_k: int = 10,
    ) -> dict:
        """Run Phase 1-3: LightRAG retrieval -> visual resolution -> reranking.

        Returns dict with keys:
            contexts: {entities, relationships, chunks}
            raw: {sources, media}
        """
        # Phase 1: LightRAG retrieval
        param = QueryParam(
            mode=cast(QueryMode, mode),
            only_need_context=True,
            top_k=top_k,
            enable_rerank=False,  # We handle reranking ourselves
        )
        result = await self.lightrag.aquery_data(query, param=param)

        # Extract chunk_ids from result
        data = result.get("data", {})
        chunk_ids: set[str] = set()

        # From chunks section
        for chunk in data.get("chunks", []):
            if chunk.get("chunk_id"):
                chunk_ids.add(chunk["chunk_id"])

        # From entities source_id
        for entity in data.get("entities", []):
            source_id = entity.get("source_id", "")
            if source_id:
                for cid in source_id.split(GRAPH_FIELD_SEP):
                    cid = cid.strip()
                    if cid:
                        chunk_ids.add(cid)

        # From relationships source_id
        for rel in data.get("relationships", []):
            source_id = rel.get("source_id", "")
            if source_id:
                for cid in source_id.split(GRAPH_FIELD_SEP):
                    cid = cid.strip()
                    if cid:
                        chunk_ids.add(cid)

        # Phase 2: Visual resolution
        chunk_id_list = list(chunk_ids)
        visual_data = await self.visual_chunks.get_by_ids(chunk_id_list)
        # visual_data is a list (same order as input); filter out None/missing.
        # Some KV backends (e.g., PG) may return JSONB as raw strings — parse them.
        resolved: dict[str, dict] = {}
        for cid, vd in zip(chunk_id_list, visual_data, strict=False):
            if vd is None:
                continue
            if isinstance(vd, str):
                try:
                    vd = json.loads(vd)
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Skipping unparseable visual_chunk %s", cid)
                    continue
            if isinstance(vd, dict):
                resolved[cid] = vd

        # Phase 3: Visual reranking (optional)
        if self.rerank_backend == "llm" and self.vision_model_func and resolved:
            resolved = await self._llm_visual_rerank(query, resolved, chunk_top_k)
        elif self.rerank_base_url and self.rerank_model and resolved:
            resolved = await self._visual_rerank(query, resolved, chunk_top_k)
        else:
            # No reranking — just take top chunk_top_k by insertion order
            resolved = dict(list(resolved.items())[:chunk_top_k])

        # Build return dict
        sources: dict[str, dict] = {}
        media: list[dict] = []
        for cid, vd in resolved.items():
            doc_id = vd.get("doc_id", "")
            if doc_id not in sources:
                sources[doc_id] = {
                    "doc_id": doc_id,
                    "title": vd.get("doc_title", ""),
                    "author": vd.get("doc_author", ""),
                    "path": vd.get("source_file", ""),
                }
            media.append(
                {
                    "chunk_id": cid,
                    "page_index": vd.get("page_index"),
                    "image_data": vd.get("image_data"),
                    "relevance_score": vd.get("relevance_score"),
                }
            )

        return {
            "contexts": {
                "entities": data.get("entities", []),
                "relationships": data.get("relationships", []),
                "chunks": [
                    {
                        "reference_id": cid,
                        "page_index": vd.get("page_index"),
                        "relevance_score": vd.get("relevance_score"),
                        "content": vd.get("content", ""),
                    }
                    for cid, vd in resolved.items()
                ],
            },
            "raw": {
                "sources": list(sources.values()),
                "media": media,
            },
        }

    async def answer(
        self,
        query: str,
        mode: str = "mix",
        top_k: int = 60,
        chunk_top_k: int = 10,
    ) -> dict:
        """Run Phase 1-4: retrieve + VLM answer generation.

        Returns dict with keys: answer, contexts, raw
        """
        retrieval = await self.retrieve(query, mode, top_k, chunk_top_k)

        if not self.vision_model_func:
            return {"answer": None, **retrieval}

        # Phase 4: Build multimodal prompt and call VLM
        from dlightrag.unifiedrepresent.prompts import UNIFIED_ANSWER_SYSTEM_PROMPT

        kg_context = self._format_kg_context(retrieval["contexts"])
        user_prompt = f"Knowledge Graph Context:\n{kg_context}\n\nQuestion: {query}"

        messages = self._build_vlm_messages(
            UNIFIED_ANSWER_SYSTEM_PROMPT, user_prompt, retrieval["raw"]["media"]
        )
        answer_text = await self.vision_model_func(
            user_prompt,
            messages=messages,
        )

        return {"answer": answer_text, **retrieval}

    async def answer_stream(
        self,
        query: str,
        mode: str = "mix",
        top_k: int = 60,
        chunk_top_k: int = 10,
    ) -> tuple[dict, dict, AsyncIterator[str] | None]:
        """Run Phase 1-3 batch + Phase 4 streaming.

        Returns (contexts, raw, token_iterator).
        """
        retrieval = await self.retrieve(query, mode, top_k, chunk_top_k)

        if not self.vision_model_func:
            return retrieval["contexts"], retrieval["raw"], None

        from dlightrag.unifiedrepresent.prompts import UNIFIED_ANSWER_SYSTEM_PROMPT

        kg_context = self._format_kg_context(retrieval["contexts"])
        user_prompt = f"Knowledge Graph Context:\n{kg_context}\n\nQuestion: {query}"

        messages = self._build_vlm_messages(
            UNIFIED_ANSWER_SYSTEM_PROMPT, user_prompt, retrieval["raw"]["media"]
        )
        token_iterator = await self.vision_model_func(
            user_prompt,
            messages=messages,
            stream=True,
        )

        return retrieval["contexts"], retrieval["raw"], token_iterator

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _llm_visual_rerank(
        self,
        query: str,
        resolved: dict[str, dict],
        top_k: int,
    ) -> dict[str, dict]:
        """Rerank visual chunks using VLM pointwise scoring.

        Sends each page image to vision_model_func with a scoring prompt.
        Pages are scored 0-10, sorted descending, top_k returned.
        """
        import asyncio

        from dlightrag.unifiedrepresent.prompts import VISUAL_RERANK_PROMPT

        if not resolved or not self.vision_model_func:
            return dict(list(resolved.items())[:top_k])

        vision_model_func = self.vision_model_func
        prompt = VISUAL_RERANK_PROMPT.format(query=query)
        sem = asyncio.Semaphore(4)
        chunk_ids = list(resolved.keys())
        logger.info("[Visual Rerank] Scoring %d chunks (pointwise VLM)", len(chunk_ids))

        async def _score_one(cid: str) -> tuple[str, float]:
            vd = resolved[cid]
            img_data = vd.get("image_data")
            if not img_data:
                return cid, 0.0
            async with sem:
                try:
                    img_bytes = base64.b64decode(img_data)
                    resp = await vision_model_func(prompt, image_data=img_bytes)
                    return cid, self._parse_rerank_score(resp)
                except Exception:
                    logger.warning("VLM rerank failed for chunk %s", cid, exc_info=True)
                    return cid, 0.0

        results = await asyncio.gather(*[_score_one(cid) for cid in chunk_ids])
        scored = sorted(results, key=lambda x: x[1], reverse=True)

        reranked: dict[str, dict] = {}
        for cid, score in scored[:top_k]:
            resolved[cid]["relevance_score"] = score
            reranked[cid] = resolved[cid]
        scores_str = ", ".join(f"{s:.1f}" for _, s in scored[:top_k])
        logger.info("[Visual Rerank] Top %d scores: [%s]", top_k, scores_str)
        return reranked

    @staticmethod
    def _parse_rerank_score(response: str) -> float:
        """Parse VLM response to a 0-10 relevance score."""
        try:
            score = float(response.strip())
            return max(0.0, min(10.0, score))
        except (ValueError, TypeError):
            logger.warning("Could not parse rerank score from: %r", response)
            return 0.0

    async def _visual_rerank(
        self,
        query: str,
        resolved: dict[str, dict],
        top_k: int,
    ) -> dict[str, dict]:
        """Rerank resolved visual chunks using multimodal reranker API.

        Calls OpenAI-compatible rerank endpoint with query + page images.
        """
        if not resolved:
            return resolved

        chunk_ids = list(resolved.keys())
        documents: list[dict | str] = []
        for cid in chunk_ids:
            vd = resolved[cid]
            img_data = vd.get("image_data")
            if img_data:
                documents.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_data}"},
                    }
                )
            else:
                # Fallback to text content if no image
                documents.append(vd.get("content", ""))

        if self.rerank_base_url is None:
            raise RuntimeError("rerank_base_url is required for visual reranking")
        url = f"{self.rerank_base_url.rstrip('/')}/rerank"
        payload = {
            "model": self.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": min(top_k, len(documents)),
        }
        headers: dict[str, str] = {}
        if self.rerank_api_key:
            headers["Authorization"] = f"Bearer {self.rerank_api_key}"

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
            results = resp.json().get("results", [])

            # Sort by relevance_score descending
            results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

            reranked: dict[str, dict] = {}
            for item in results[:top_k]:
                idx = item["index"]
                cid = chunk_ids[idx]
                resolved[cid]["relevance_score"] = item.get("relevance_score")
                reranked[cid] = resolved[cid]
            return reranked

        except Exception:
            logger.warning(
                "Visual reranking failed, returning unranked results",
                exc_info=True,
            )
            return dict(list(resolved.items())[:top_k])

    @staticmethod
    def _build_vlm_messages(system_prompt: str, user_prompt: str, media: list[dict]) -> list[dict]:
        """Build OpenAI-format multimodal messages with inline base64 images."""
        content: list[dict] = []
        for item in media:
            img_data = item.get("image_data")
            if img_data:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_data}"},
                    }
                )
        content.append({"type": "text", "text": user_prompt})
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    def _format_kg_context(self, contexts: dict) -> str:
        """Format KG context (entities + relationships) as text for VLM prompt."""
        parts: list[str] = []

        entities = contexts.get("entities", [])
        if entities:
            parts.append("## Entities")
            for e in entities[:20]:  # Limit to avoid prompt overflow
                name = e.get("entity_name", "")
                etype = e.get("entity_type", "")
                desc = e.get("description", "")
                parts.append(f"- **{name}** ({etype}): {desc}")

        rels = contexts.get("relationships", [])
        if rels:
            parts.append("\n## Relationships")
            for r in rels[:20]:
                src = r.get("src_id", "")
                tgt = r.get("tgt_id", "")
                desc = r.get("description", "")
                parts.append(f"- {src} -> {tgt}: {desc}")

        return "\n".join(parts) if parts else "No knowledge graph context available."
