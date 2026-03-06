# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Entity extraction for unified representational RAG.

Uses VLM to generate text descriptions from page images, then feeds
them into LightRAG's entity extraction pipeline to build the knowledge graph.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from lightrag.operate import extract_entities, merge_nodes_and_edges
from lightrag.utils import compute_mdhash_id

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extract entities from page images via VLM + LightRAG pipeline."""

    def __init__(
        self,
        lightrag: Any,  # LightRAG instance
        entity_types: list[str],
        vision_model_func: Callable,
        max_concurrent_vlm: int = 4,
    ) -> None:
        self.lightrag = lightrag
        self.entity_types = entity_types
        self.vision_model_func = vision_model_func
        self._vlm_semaphore = asyncio.Semaphore(max_concurrent_vlm)

    async def extract_from_pages(
        self,
        images: list[Any],  # list[PIL.Image.Image]
        doc_id: str,
        file_path: str,
    ) -> list[dict]:
        """Extract entities from page images and build KG.

        Returns list of dicts with keys: chunk_id, page_index, content
        (VLM description). These are used to populate text_chunks and
        visual_chunks.
        """
        # 1. Generate VLM descriptions for all pages (semaphore-controlled)
        description_tasks = [
            self._describe_page(image, page_index)
            for page_index, image in enumerate(images)
        ]
        descriptions = await asyncio.gather(*description_tasks)

        # 2. Build chunks dict for LightRAG
        chunk_ids: list[str] = []
        chunks_dict: dict[str, dict[str, Any]] = {}
        for page_index, description in enumerate(descriptions):
            chunk_id = compute_mdhash_id(
                f"{doc_id}:page:{page_index}", prefix="chunk-"
            )
            chunk_ids.append(chunk_id)
            chunks_dict[chunk_id] = {
                "content": description,
                "full_doc_id": doc_id,
                "tokens": len(description.split()),
                "chunk_order_index": page_index,
                "file_path": file_path,
            }

        # 3. Call extract_entities
        chunk_results = await extract_entities(
            chunks=chunks_dict,
            global_config=self.lightrag.__dict__,
            llm_response_cache=self.lightrag.llm_response_cache,
            text_chunks_storage=self.lightrag.text_chunks,
        )

        # 4. Call merge_nodes_and_edges
        await merge_nodes_and_edges(
            chunk_results=chunk_results,
            knowledge_graph_inst=self.lightrag.chunk_entity_relation_graph,
            entity_vdb=self.lightrag.entities_vdb,
            relationships_vdb=self.lightrag.relationships_vdb,
            global_config=self.lightrag.__dict__,
            full_entities_storage=self.lightrag.full_entities,
            full_relations_storage=self.lightrag.full_relations,
            doc_id=doc_id,
            llm_response_cache=self.lightrag.llm_response_cache,
            file_path=file_path,
        )

        # 5. Return page info list
        return [
            {
                "chunk_id": chunk_id,
                "page_index": page_index,
                "content": description,
            }
            for page_index, (chunk_id, description) in enumerate(
                zip(chunk_ids, descriptions, strict=True)
            )
        ]

    async def _describe_page(self, image: Any, page_index: int) -> str:
        """Call VLM to generate text description of a page image."""
        from dlightrag.unifiedrepresent.prompts import PAGE_DESCRIPTION_PROMPT

        prompt = PAGE_DESCRIPTION_PROMPT.format(
            entity_types=", ".join(self.entity_types)
        )

        async with self._vlm_semaphore:
            # vision_model_func expects: (prompt, images=[image])
            # The exact calling convention follows DlightRAG's existing
            # vision model pattern
            description = await self.vision_model_func(
                prompt,
                images=[image],
            )

        if not description or not description.strip():
            logger.warning(
                "VLM returned empty description for page %d", page_index
            )
            return f"[Page {page_index + 1}: no content extracted]"

        return description.strip()
