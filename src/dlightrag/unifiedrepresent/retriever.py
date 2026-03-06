# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Visual retrieval pipeline for unified representational RAG.

Handles query processing: LightRAG KG retrieval -> visual chunk resolution ->
visual reranking -> VLM answer generation.
"""

from __future__ import annotations

import base64
import io
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
from lightrag import LightRAG, QueryParam
from lightrag.constants import GRAPH_FIELD_SEP

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
    ) -> None:
        self.lightrag = lightrag
        self.visual_chunks = visual_chunks
        self.config = config
        self.vision_model_func = vision_model_func
        self.rerank_model = rerank_model
        self.rerank_base_url = rerank_base_url
        self.rerank_api_key = rerank_api_key

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
            mode=mode,
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
        # visual_data is a list (same order as input); filter out None/missing
        resolved = {
            cid: vd
            for cid, vd in zip(chunk_id_list, visual_data, strict=False)
            if vd is not None
        }

        # Phase 3: Visual reranking (optional)
        if self.rerank_model and self.rerank_base_url and resolved:
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
            media.append({
                "chunk_id": cid,
                "page_index": vd.get("page_index"),
                "image_data": vd.get("image_data"),
                "relevance_score": vd.get("relevance_score"),
            })

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
        from PIL import Image

        from dlightrag.unifiedrepresent.prompts import UNIFIED_ANSWER_SYSTEM_PROMPT

        # Build KG text context
        kg_context = self._format_kg_context(retrieval["contexts"])

        # Collect page images from media
        images = []
        for item in retrieval["raw"]["media"]:
            img_data = item.get("image_data")
            if img_data:
                img_bytes = base64.b64decode(img_data)
                img = Image.open(io.BytesIO(img_bytes))
                images.append(img)

        # Build prompt
        user_prompt = f"Knowledge Graph Context:\n{kg_context}\n\nQuestion: {query}"

        # Call VLM with system prompt + user prompt + images
        answer_text = await self.vision_model_func(
            user_prompt,
            system_prompt=UNIFIED_ANSWER_SYSTEM_PROMPT,
            images=images if images else None,
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

        from PIL import Image

        from dlightrag.unifiedrepresent.prompts import UNIFIED_ANSWER_SYSTEM_PROMPT

        kg_context = self._format_kg_context(retrieval["contexts"])
        images = []
        for item in retrieval["raw"]["media"]:
            img_data = item.get("image_data")
            if img_data:
                img_bytes = base64.b64decode(img_data)
                images.append(Image.open(io.BytesIO(img_bytes)))

        user_prompt = f"Knowledge Graph Context:\n{kg_context}\n\nQuestion: {query}"

        # Call VLM with stream=True
        token_iterator = await self.vision_model_func(
            user_prompt,
            system_prompt=UNIFIED_ANSWER_SYSTEM_PROMPT,
            images=images if images else None,
            stream=True,
        )

        return retrieval["contexts"], retrieval["raw"], token_iterator

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
                documents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_data}"},
                })
            else:
                # Fallback to text content if no image
                documents.append(vd.get("content", ""))

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
            results.sort(
                key=lambda x: x.get("relevance_score", 0), reverse=True
            )

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
