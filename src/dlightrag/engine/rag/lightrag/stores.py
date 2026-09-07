# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Single boundary for LightRAG private storage access.

**LightRAG coupling surface (lightrag-hku>=1.5.7):**

This module depends on the following LightRAG internals that are NOT part
of the public API.  A LightRAG major-version bump may break these:

Storage attributes (copied from the LightRAG instance):
    ``chunks_vdb``
    ``text_chunks``
    ``full_docs``
    ``doc_status``

LightRAG API methods (public, but shape-dependent):
    ``LightRAG.aquery_data()``          — returns {data: {...}, status: ...}
    ``apipeline_enqueue_documents()``   — enqueues for processing
    ``apipeline_process_enqueue_documents()`` — processes queue

Backend-specific chunk operations are delegated through ``CorpusChunkStore``.
When upgrading lightrag-hku, verify these surfaces still exist and behave as
expected. The host contract guard provides runtime contract checks.
"""

from collections.abc import AsyncIterator, Sequence
from typing import Any

from dlightrag.engine.rag.retrieval import MetadataFilter, MetadataScope
from dlightrag.engine.rag.retrieval.ports import CorpusChunkStore


class LightRAGStores:
    """Typed accessor for the LightRAG storage surface DlightRAG touches directly."""

    chunks_vdb: Any
    text_chunks: Any
    full_docs: Any
    doc_status: Any

    def __init__(self, lightrag: Any, *, chunk_store: CorpusChunkStore) -> None:
        self.raw = lightrag
        self.chunks_vdb = lightrag.chunks_vdb
        self.text_chunks = lightrag.text_chunks
        self.full_docs = lightrag.full_docs
        self.doc_status = lightrag.doc_status
        self._chunk_store = chunk_store

    async def get_doc_status(self, doc_id: str) -> dict[str, Any] | None:
        return await self.doc_status.get_by_id(doc_id)

    async def iter_doc_status_pages(
        self,
        statuses: Sequence[Any],
        *,
        page_size: int = 200,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield strict keyset pages without materializing a status bucket."""
        if not statuses:
            return
        if isinstance(page_size, bool) or page_size < 1:
            raise ValueError("page_size must be a positive integer")

        from lightrag.base import CURSOR_END, CURSOR_START

        cursor = CURSOR_START
        while True:
            page = await self.doc_status.get_docs_by_statuses_page(
                list(statuses),
                limit=page_size,
                position=cursor,
                strict=True,
            )
            yield dict(page.docs)
            next_cursor = page.next_position
            if next_cursor is CURSOR_END:
                return
            if next_cursor == cursor:
                raise RuntimeError("LightRAG status paging cursor did not advance")
            cursor = next_cursor

    async def get_full_doc_statuses(self, doc_ids: list[str]) -> dict[str, Any]:
        """Hydrate full status rows for one bounded page of known ids."""
        if not doc_ids:
            return {}
        return await self.doc_status.get_full_docs_by_ids(doc_ids, strict=True)

    async def get_full_doc(self, doc_id: str) -> dict[str, Any] | None:
        return await self.full_docs.get_by_id(doc_id)

    async def get_full_docs(self, doc_ids: list[str]) -> list[Any]:
        """Fetch full-document KV rows aligned with the requested ids."""
        if not doc_ids:
            return []
        return await self.full_docs.get_by_ids(doc_ids)

    async def get_text_chunks(self, chunk_ids: list[str]) -> list[Any]:
        if not chunk_ids:
            return []
        return await self.text_chunks.get_by_ids(chunk_ids)

    async def context_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch LightRAG text chunks and format them as retrieval context rows."""
        if not chunk_ids:
            return []
        inject_ids = list(dict.fromkeys(chunk_ids))
        raw_contents = await self.get_text_chunks(inject_ids)

        chunks: list[dict[str, Any]] = []
        for cid, content_raw in zip(inject_ids, raw_contents, strict=False):
            if content_raw is None:
                continue
            if isinstance(content_raw, str):
                content = content_raw
                file_path = ""
                full_doc_id = None
            else:
                content = content_raw.get("content", "")
                file_path = content_raw.get("file_path", "") or ""
                full_doc_id = content_raw.get("full_doc_id")
            chunk = {
                "chunk_id": cid,
                "content": content,
                "reference_id": "",
                "file_path": file_path,
            }
            if full_doc_id:
                chunk["full_doc_id"] = str(full_doc_id)
            if not isinstance(content_raw, str):
                for key in ("sidecar", "sidecar_location", "page_number"):
                    if content_raw.get(key) is not None:
                        chunk[key] = content_raw[key]
            chunks.append(chunk)
        return chunks

    async def overwrite_chunk_vectors(
        self,
        vectors: dict[str, list[float]],
        *,
        embedding_dim: int,
    ) -> None:
        await self._chunk_store.overwrite_chunk_vectors(vectors, embedding_dim=embedding_dim)

    async def resolve_scope(self, filters: MetadataFilter) -> MetadataScope:
        """Resolve filter facts plus the bounded matching-chunk probe."""
        return await self._chunk_store.resolve_scope(filters)

    async def fetch_chunk_contents(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        return await self._chunk_store.fetch_chunk_contents(chunk_ids)

    async def update_chunk_bm25_languages(self, labels: dict[str, str]) -> None:
        await self._chunk_store.update_chunk_bm25_languages(labels)
