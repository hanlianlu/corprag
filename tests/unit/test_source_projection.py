# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for client-safe retrieval source and context projection."""

import logging

import pytest

from dlightrag.engine.answer.citations.contracts import ChunkSnippet, SourceReference
from dlightrag.engine.rag.retrieval import RetrievalResult


def _internal_source(*, chunks: list[ChunkSnippet] | None = None) -> SourceReference:
    return SourceReference(
        id="1",
        title="report.pdf",
        source_uri="local://default/report.pdf",
        workspace="default",
        document_id="doc-report",
        download_locator="/private/report.pdf",
        chunks=chunks,
    )


def test_project_source_payloads_resolves_and_hides_raw_locator(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from dlightrag.engine.answer.citations.sources import (
        SourceDownloadLinkBuilder,
        project_source_payloads,
    )

    source = SourceReference(
        id="1",
        source_uri="s3://bucket/report.pdf",
        workspace="finance",
        document_id="doc-report",
        download_locator="s3://bucket/report.pdf",
    )

    with caplog.at_level(logging.INFO, logger="dlightrag.engine.answer.citations.sources"):
        projected = project_source_payloads([source], resolver=SourceDownloadLinkBuilder())[0]

    assert projected.download_url == "/files/raw/doc-report?workspace=finance"
    assert source.download_locator not in projected.download_url
    payload = projected.model_dump()
    assert "download_locator" not in payload
    assert "workspace" not in payload
    record = next(
        record
        for record in caplog.records
        if record.message == "source_download_projection_outcome"
    )
    assert getattr(record, "outcome", None) == "resolved"
    assert getattr(record, "source_id", None) == "1"
    assert source.download_locator not in caplog.text


def test_project_source_payloads_rejects_invalid_locator_without_logging_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from dlightrag.engine.answer.citations.sources import (
        SourceDownloadInvariantError,
        SourceDownloadLinkBuilder,
        project_source_payloads,
    )

    source = SourceReference(
        id="unsafe\nsource",
        source_uri="bynder://asset/1",
        workspace="finance",
        document_id="",
        download_locator="file://secret-host/private/report.pdf",
    )

    with (
        caplog.at_level(logging.INFO, logger="dlightrag.engine.answer.citations.sources"),
        pytest.raises(SourceDownloadInvariantError, match=r"unsafe\\nsource"),
    ):
        project_source_payloads([source], resolver=SourceDownloadLinkBuilder())

    record = next(
        record
        for record in caplog.records
        if record.message == "source_download_projection_outcome"
    )
    assert getattr(record, "outcome", None) == "invalid"
    assert getattr(record, "source_id", None) == r"unsafe\nsource"
    assert source.download_locator not in caplog.text


def test_project_source_payloads_omits_link_without_download_permission() -> None:
    from dlightrag.engine.answer.citations.sources import (
        SourceDownloadLinkBuilder,
        project_source_payloads,
    )

    source = SourceReference(
        id="1",
        source_uri="s3://bucket/report.pdf",
        workspace="finance",
        document_id="doc-report",
        download_locator="s3://bucket/report.pdf",
    )

    projected = project_source_payloads(
        [source],
        resolver=SourceDownloadLinkBuilder(),
        downloadable_workspaces=set(),
    )[0]

    assert projected.download_url is None


def test_public_context_projection_strips_internal_source_metadata() -> None:
    from dlightrag.engine.answer.citations.sources import project_contexts_for_client

    contexts = {
        "chunks": [
            {
                "chunk_id": "c1",
                "reference_id": "r1",
                "file_path": "/srv/dlightrag/inputs/finance/report.pdf",
                "content": "text",
                "metadata": {
                    "source_uri": "bynder://asset/1",
                    "source_download_locator": "https://cdn.example.com/assets/1.pdf",
                    "category": "research",
                },
            }
        ],
        "entities": [],
        "relationships": [],
    }

    projected = project_contexts_for_client(contexts)

    assert projected["chunks"][0]["metadata"] == {"category": "research"}
    assert projected["chunks"][0]["file_path"] == "report.pdf"


def test_retrieval_projector_hides_composer_cache_and_vector_fields() -> None:
    from dlightrag.application.retrieval import RetrieveProjection
    from dlightrag.application.retrieval._answer_projection import project_answer_retrieval

    private_fields = {
        "_cache_key",
        "cache_chunk_id",
        "embedding_signature",
        "embedding_vector",
    }
    result = RetrievalResult(
        contexts={
            "chunks": [
                {
                    "chunk_id": "composer-att-1",
                    "reference_id": "composer_ref_1",
                    "full_doc_id": "att-1",
                    "file_path": "report.pdf",
                    "content": "Composer evidence",
                    "_workspace": "__web_attachment__",
                    "_cache_key": object(),
                    "cache_chunk_id": "private-cache-id",
                    "embedding_signature": "private-signature",
                    "embedding_vector": [0.25, 0.75],
                    "metadata": {
                        "source_type": "web_attachment",
                        "source_uri": "web-attachment://att-1",
                        "source_download_locator": "web-attachment://att-1",
                    },
                }
            ],
            "entities": [],
            "relationships": [],
        }
    )

    projected = project_answer_retrieval(
        result,
        RetrieveProjection(
            downloadable_workspaces=frozenset(),
            visual_workspaces=frozenset(),
            image_url_prefix=None,
        ),
    )
    payload = {
        "contexts": projected.contexts,
        "sources": list(projected.sources),
    }

    assert private_fields.isdisjoint(payload["contexts"]["chunks"][0])
    assert private_fields.isdisjoint(payload["sources"][0])
    assert private_fields.isdisjoint(payload["sources"][0]["chunks"][0])


def test_project_contexts_for_client_strips_inline_images_and_adds_image_urls() -> None:
    from dlightrag.engine.answer.citations.sources import project_contexts_for_client

    contexts = {
        "chunks": [
            {
                "chunk_id": "image chunk/1",
                "reference_id": "1",
                "file_path": "/private/report.pdf",
                "content": "Figure evidence",
                "page_number": 2,
                "image_mime_type": "image/png",
                "relevance_score": 0.87,
                "metadata": {"department": "finance"},
                "image_data": "base64-payload",
                "_workspace": "workspace a",
                "full_doc_id": "doc-internal",
                "score": 0.22,
                "rerank_score": 0.91,
                "distance": 0.78,
                "bm25_profile": "english",
                "sidecar": {"type": "drawing"},
                "sidecar_location": "file:///tmp/report.parsed",
            }
        ],
        "entities": [{"entity_name": "ACME"}],
    }

    public = project_contexts_for_client(contexts)

    chunk = public["chunks"][0]
    assert "image_data" not in chunk
    assert chunk["image_url"] == "/images/workspace%20a/image%20chunk%2F1?size=full"
    assert chunk["thumbnail_url"] == "/images/workspace%20a/image%20chunk%2F1?size=thumb"
    assert chunk["page_number"] == 2
    assert "page_idx" not in chunk
    assert "bbox" not in chunk
    assert chunk["metadata"] == {"department": "finance"}
    assert "full_doc_id" not in chunk
    assert "score" not in chunk
    assert "rerank_score" not in chunk
    assert "distance" not in chunk
    assert "bm25_profile" not in chunk
    assert "sidecar" not in chunk
    assert "sidecar_location" not in chunk
    assert public["entities"] == [{"entity_name": "ACME"}]
    assert "image_data" in contexts["chunks"][0]


@pytest.mark.parametrize("workspace", ["__attachment__", "__web_search__"])
def test_request_owned_visuals_do_not_require_corpus_acl(workspace: str) -> None:
    from dlightrag.engine.answer.citations.sources import project_contexts_for_client

    public = project_contexts_for_client(
        {
            "chunks": [
                {
                    "chunk_id": "visual-1",
                    "content": "Request evidence",
                    "image_data": "base64-payload",
                    "_workspace": workspace,
                }
            ]
        },
        visual_workspaces=set(),
    )

    assert public["chunks"][0]["image_url"].startswith(f"/images/{workspace}/")


def test_real_double_underscore_workspace_visuals_still_require_acl() -> None:
    from dlightrag.engine.answer.citations.sources import project_contexts_for_client

    public = project_contexts_for_client(
        {
            "chunks": [
                {
                    "chunk_id": "visual-1",
                    "content": "Corpus evidence",
                    "image_data": "base64-payload",
                    "image_url": "/images/__private/visual-1?size=full",
                    "thumbnail_url": "/images/__private/visual-1?size=thumb",
                    "image_mime_type": "image/png",
                    "_workspace": "__private",
                }
            ]
        },
        visual_workspaces=set(),
    )

    chunk = public["chunks"][0]
    assert "image_url" not in chunk
    assert "thumbnail_url" not in chunk
    assert "image_mime_type" not in chunk


def test_project_contexts_for_client_accepts_lightrag_id_alias() -> None:
    from dlightrag.engine.answer.citations.sources import project_contexts_for_client

    public = project_contexts_for_client({"chunks": [{"id": "c1", "content": "Evidence"}]})

    assert public["chunks"] == [
        {"chunk_id": "c1", "reference_id": "", "file_path": "", "content": "Evidence"}
    ]


def test_project_contexts_for_client_adds_visual_chunk_urls_without_inline_image_data() -> None:
    from dlightrag.engine.answer.citations.sources import project_contexts_for_client

    public = project_contexts_for_client(
        {
            "chunks": [
                {
                    "chunk_id": "doc-1-mm-drawing-001",
                    "reference_id": "1",
                    "file_path": "/private/report.pdf",
                    "content": "[Image Name]architecture",
                    "_workspace": "default",
                    "sidecar": {"type": "drawing"},
                },
                {
                    "chunk_id": "doc-1-chunk-001",
                    "reference_id": "1",
                    "file_path": "/private/report.pdf",
                    "content": "plain text",
                    "_workspace": "default",
                },
            ]
        }
    )

    assert public["chunks"][0]["image_url"] == "/images/default/doc-1-mm-drawing-001?size=full"
    assert "image_url" not in public["chunks"][1]


def test_project_contexts_for_client_skips_chunks_without_public_id() -> None:
    from dlightrag.engine.answer.citations.sources import project_contexts_for_client

    public = project_contexts_for_client({"chunks": [{"content": "orphan"}]})

    assert public["chunks"] == []


def test_evidence_image_helper_derives_cited_visuals() -> None:
    from dlightrag.engine.answer.media import evidence_images_from_sources

    sources = [
        _internal_source(
            chunks=[
                ChunkSnippet(
                    chunk_id="fig-1",
                    chunk_idx=1,
                    content="Figure evidence",
                    image_url="/images/default/fig-1?size=full",
                    thumbnail_url="/images/default/fig-1?size=thumb",
                )
            ]
        )
    ]

    images = evidence_images_from_sources(sources)

    assert images == [
        {
            "id": "fig-1",
            "chunk_id": "fig-1",
            "source_ref": "1-1",
            "url": "/images/default/fig-1?size=full",
            "thumbnail_url": "/images/default/fig-1?size=thumb",
            "label": "report.pdf",
            "answer_image_sent": True,
        }
    ]
