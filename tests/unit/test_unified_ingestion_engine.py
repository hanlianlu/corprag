# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for unified LightRAG sidecar ingestion engine."""

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from lightrag.base import DocStatus
from lightrag.parser.routing import FilenameParserHintError
from lightrag.utils import compute_mdhash_id
from lightrag.utils_pipeline import normalize_document_file_path
from PIL import Image

from dlightrag.engine.rag.corpus.ingestion.document_embedding import (
    DocumentEmbeddingInput,
    DocumentEmbeddingTrace,
    DocumentEmbeddingVector,
)
from dlightrag.engine.rag.corpus.ingestion.engine import (
    _FINALIZATION_COMPLETE_KEY,
    PreparedIngestFile,
    UnifiedIngestionEngine,
    _prepare_ingest_item,
    _raw_path_source_uri,
)


def _sha256(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _make_engine(**overrides):
    lightrag = AsyncMock()
    lightrag.apipeline_enqueue_documents.return_value = "track-1"
    lightrag.adelete_by_doc_id.return_value = SimpleNamespace(status="success")
    stores = AsyncMock()
    stores.fetch_chunk_contents.return_value = []
    stores.get_doc_status.return_value = {
        "status": "processed",
        "chunks_list": ["chunk-a"],
        "content_hash": "sha256:abc",
    }
    stores.get_full_doc_statuses.side_effect = lambda doc_ids: {
        doc_id: {
            "status": "processed",
            "chunks_list": ["chunk-a"],
            "content_hash": "sha256:abc",
        }
        for doc_id in doc_ids
    }
    stores.get_full_doc.return_value = {
        "parse_engine": "mineru",
        "process_options": "iteP",
        "chunk_options": {"paragraph_semantic": {"chunk_token_size": 2000}},
        "sidecar_location": "file:///tmp/sample.parsed/",
    }
    document_embedder = AsyncMock()
    document_embedder.image_enabled = True
    document_embedder.dimension = 3
    document_embedder.aembed_documents.return_value = (
        [],
        DocumentEmbeddingTrace(fused=0, text=0, fused_to_text_fallback=0, failed=0),
    )
    defaults = {
        "lightrag": lightrag,
        "stores": stores,
        "metadata_index": AsyncMock(),
        "document_embedder": document_embedder,
        "workspace": "default",
        "parser_rules": "docx:native-iteP,*:mineru-iteP",
        "chunk_options": {},
    }
    defaults.update(overrides)
    defaults["metadata_index"].get.return_value = None
    return UnifiedIngestionEngine(**defaults), defaults


async def test_replace_false_keeps_idempotent_skip(tmp_path: Path) -> None:
    content = b"%PDF-1.4"
    source = tmp_path / "sample.pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-a"],
        "content_hash": _sha256(content),
        "status": "processed",
    }
    deps["metadata_index"].get.return_value = {_FINALIZATION_COMPLETE_KEY: True}

    result = await engine.aingest_file(source, replace=False)

    assert result["source_kind"] == "skipped"
    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_replace_true_bypasses_idempotent_skip(tmp_path: Path) -> None:
    content = b"%PDF-1.4"
    source = tmp_path / "sample.pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [
        {"chunks_list": ["old-chunk"], "content_hash": _sha256(content), "status": "processed"},
        {"chunks_list": ["old-chunk"], "content_hash": _sha256(content), "status": "processed"},
        {"chunks_list": ["new-chunk"], "content_hash": _sha256(content), "status": "processed"},
    ]

    result = await engine.aingest_file(source, replace=True)

    assert result["source_kind"] == "document"
    assert result["chunks"] == ["new-chunk"]
    deps["lightrag"].adelete_by_doc_id.assert_awaited_once()
    deps["lightrag"].apipeline_enqueue_documents.assert_awaited_once()


async def test_batch_replace_true_bypasses_idempotent_skip(tmp_path: Path) -> None:
    content = b"%PDF-1.4"
    source = tmp_path / "sample.pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [
        {"chunks_list": ["old-chunk"], "content_hash": _sha256(content), "status": "processed"},
        {"chunks_list": ["old-chunk"], "content_hash": _sha256(content), "status": "processed"},
        {"chunks_list": ["new-chunk"], "content_hash": _sha256(content), "status": "processed"},
    ]

    result = await engine.aingest_files([source], replace=True)

    assert result["processed"] == 1
    assert result["results"][0]["chunks"] == ["new-chunk"]
    deps["lightrag"].adelete_by_doc_id.assert_awaited_once()
    deps["lightrag"].apipeline_enqueue_documents.assert_awaited_once()


async def test_document_ingest_resolves_lightrag_parser_rules(tmp_path: Path) -> None:
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()

    await engine.aingest_file(source, replace=False)

    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["docs_format"] == "pending_parse"
    assert "ids" not in kwargs
    assert kwargs["parse_engine"] == ["mineru"]
    assert kwargs["process_options"] == ["iteP"]
    deps["lightrag"].apipeline_process_enqueue_documents.assert_awaited_once()
    assert deps["metadata_index"].upsert.await_count == 3
    assert deps["metadata_index"].upsert.await_args_list[0].args[1] == {
        _FINALIZATION_COMPLETE_KEY: False
    }


async def test_ingest_waits_when_processing_is_queued_behind_busy_owner(
    tmp_path: Path,
) -> None:
    source = tmp_path / "queued.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    statuses: dict[str, dict[str, object]] = {}
    processing_call_returned = asyncio.Event()

    async def enqueue(**_kwargs: object) -> str:
        statuses[doc_id] = {"status": "pending", "chunks_list": []}
        return "track-queued"

    async def queue_behind_busy_owner() -> None:
        processing_call_returned.set()

    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc_statuses.side_effect = lambda doc_ids: {
        doc_id: statuses[doc_id] for doc_id in doc_ids if doc_id in statuses
    }
    deps["lightrag"].apipeline_enqueue_documents.side_effect = enqueue
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = queue_behind_busy_owner

    task = asyncio.create_task(engine.aingest_files([source]))
    await asyncio.wait_for(processing_call_returned.wait(), timeout=1)
    await asyncio.sleep(0)

    assert not task.done()

    statuses[doc_id] = {"status": "processed", "chunks_list": ["chunk-queued"]}
    result = await asyncio.wait_for(task, timeout=1)

    assert result["processed"] == 1
    assert result["errors"] == []
    assert result["results"][0]["chunks"] == ["chunk-queued"]


async def test_ingest_redrives_queue_after_busy_owner_exits_abnormally(
    tmp_path: Path,
) -> None:
    source = tmp_path / "retry-queued.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    statuses: dict[str, dict[str, object]] = {}
    processing_calls = 0

    async def enqueue(**_kwargs: object) -> str:
        statuses[doc_id] = {"status": "pending", "chunks_list": []}
        return "track-retry-queued"

    async def process_or_observe_busy_owner() -> None:
        nonlocal processing_calls
        processing_calls += 1
        if processing_calls == 2:
            statuses[doc_id] = {
                "status": "processed",
                "chunks_list": ["chunk-retry-queued"],
            }

    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc_statuses.side_effect = lambda doc_ids: {
        doc_id: statuses[doc_id] for doc_id in doc_ids if doc_id in statuses
    }
    deps["lightrag"].apipeline_enqueue_documents.side_effect = enqueue
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process_or_observe_busy_owner

    result = await asyncio.wait_for(engine.aingest_files([source]), timeout=1)

    assert processing_calls == 2
    assert result["processed"] == 1
    assert result["errors"] == []
    assert result["results"][0]["chunks"] == ["chunk-retry-queued"]


async def test_ingest_does_not_wait_forever_on_unknown_pipeline_status(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unknown-status.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    unknown_status = {"status": "future-state", "chunks_list": []}

    deps["stores"].get_doc_status.return_value = unknown_status
    deps["stores"].get_full_doc_statuses.side_effect = lambda _doc_ids: {doc_id: unknown_status}

    result = await asyncio.wait_for(engine.aingest_files([source]), timeout=1)

    assert result["processed"] == 0
    assert result["errors"] == ["unknown-status.pdf: document processing failed"]


async def test_document_ingest_persists_lightrag_archived_source_locator(
    tmp_path: Path,
) -> None:
    source = tmp_path / "inputs" / "default" / "report.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-1.4")
    archived = source.parent / "__parsed__" / source.name
    engine, deps = _make_engine()

    async def archive_source() -> None:
        archived.parent.mkdir()
        source.replace(archived)

    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = archive_source

    await engine.aingest_file(source, replace=False)

    _, saved = deps["metadata_index"].upsert.await_args.args
    assert saved["download_locator"] == str(archived.resolve())


async def test_document_ingest_raises_when_pipeline_finishes_failed(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [
        None,
        {
            "status": DocStatus.FAILED,
            "chunks_list": [],
            "content_hash": None,
            "content_summary": "PDF parser failed on page 3",
            "error_msg": None,
        },
    ]

    with pytest.raises(RuntimeError, match="PDF parser failed on page 3"):
        await engine.aingest_file(source, replace=False)

    assert deps["metadata_index"].upsert.await_count == 1
    deps["stores"].overwrite_chunk_vectors.assert_not_awaited()


async def test_document_ingest_preserves_lightrag_parser_engine_params(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample.[mineru(page_range=1-3)-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()

    await engine.aingest_file(source, replace=False)

    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["parse_engine"] == ["mineru(page_range=1-3)"]


@pytest.mark.parametrize(
    "fault_phase",
    [
        pytest.param("visual", id="required-visual-fusion"),
        pytest.param("bm25", id="required-bm25-labels"),
        pytest.param("metadata_source_readiness", id="metadata-source-and-readiness-marker"),
    ],
)
async def test_required_product_document_finalizer_fault_replays_before_readiness(
    fault_phase: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed required finalizer cannot publish; retry replays the same document."""
    engine, deps = _make_engine()
    metadata = {
        "filename": "report.pdf",
        "source_uri": "local://default/report.pdf",
        "download_locator": "/shared/default/report.pdf",
        _FINALIZATION_COMPLETE_KEY: False,
    }
    visual = AsyncMock()
    bm25 = AsyncMock()
    monkeypatch.setattr(engine, "_overwrite_sidecar_image_vectors", visual)
    monkeypatch.setattr(engine, "_label_bm25_languages", bm25)
    if fault_phase == "visual":
        visual.side_effect = [RuntimeError("visual finalizer unavailable"), None]
    elif fault_phase == "bm25":
        bm25.side_effect = [RuntimeError("BM25 finalizer unavailable"), None]
    else:
        deps["metadata_index"].upsert.side_effect = [
            RuntimeError("metadata/source/readiness commit unavailable"),
            None,
        ]

    with pytest.raises(RuntimeError):
        await engine._finalize_ingested_document(
            doc_id="doc-report",
            metadata_record=metadata,
            parse_engine="mineru",
            process_options="iteP",
        )

    # No compensating write marks an unfinished document ready. The exact same
    # idempotent finalization can then complete without replaying LightRAG ingest.
    result = await engine._finalize_ingested_document(
        doc_id="doc-report",
        metadata_record=metadata,
        parse_engine="mineru",
        process_options="iteP",
    )

    assert result["doc_id"] == "doc-report"
    assert deps["lightrag"].apipeline_enqueue_documents.await_count == 0
    committed = deps["metadata_index"].upsert.await_args.args[1]
    assert committed[_FINALIZATION_COMPLETE_KEY] is True
    assert committed["source_uri"] == "local://default/report.pdf"


async def test_document_ingest_labels_bm25_chunk_languages(tmp_path: Path) -> None:
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")

    class FakeClassifier:
        def detect(self, content: str) -> str:
            return {"现金流 风险": "zh", "risk factors": "en"}.get(content, "simple")

    engine, deps = _make_engine(bm25_language_classifier=FakeClassifier())
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-zh", "chunk-en"],
        "content_hash": "sha256:abc",
        "status": "processed",
    }
    deps["stores"].fetch_chunk_contents.return_value = [
        {"id": "chunk-zh", "content": "现金流 风险"},
        {"id": "chunk-en", "content": "risk factors"},
    ]

    await engine.aingest_file(source, replace=False)

    deps["stores"].fetch_chunk_contents.assert_awaited_once_with(["chunk-zh", "chunk-en"])
    deps["stores"].update_chunk_bm25_languages.assert_awaited_once_with(
        {"chunk-zh": "zh", "chunk-en": "en"}
    )


async def test_batch_document_ingest_uses_lightrag_staged_pipeline(tmp_path: Path) -> None:
    pdf = tmp_path / "b[mineru-iteP].pdf"
    docx = tmp_path / "a.docx"
    pdf.write_bytes(b"%PDF-1.4")
    docx.write_bytes(b"fake-docx")
    engine, deps = _make_engine()
    pdf_doc_id = compute_mdhash_id(normalize_document_file_path(pdf), prefix="doc-")
    docx_doc_id = compute_mdhash_id(normalize_document_file_path(docx), prefix="doc-")
    deps["stores"].get_doc_status.side_effect = [
        None,
        None,
        {"chunks_list": ["chunk-docx"], "content_hash": "sha256:docx", "status": "processed"},
        {"chunks_list": ["chunk-pdf"], "content_hash": "sha256:pdf", "status": "processed"},
    ]
    deps["stores"].get_full_doc.side_effect = [
        {
            "parse_engine": "native",
            "process_options": "iteP",
            "chunk_options": {},
            "sidecar_location": None,
        },
        {
            "parse_engine": "mineru",
            "process_options": "iteP",
            "chunk_options": {},
            "sidecar_location": None,
        },
    ]

    result = await engine.aingest_files([docx, pdf], replace=False)

    assert result["processed"] == 2
    assert [item["doc_id"] for item in result["results"]] == [docx_doc_id, pdf_doc_id]
    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["input"] == ["", ""]
    assert kwargs["file_paths"] == [str(docx), str(pdf)]
    assert kwargs["parse_engine"] == ["native", "mineru"]
    assert kwargs["process_options"] == ["iteP", "iteP"]
    deps["lightrag"].apipeline_process_enqueue_documents.assert_awaited_once()
    assert deps["metadata_index"].upsert.await_count == 4


async def test_batch_document_ingest_preserves_per_file_chunk_params(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "b.[mineru-iteP(chunk_ts=1234,drop_rf=true)].pdf"
    docx = tmp_path / "a.docx"
    pdf.write_bytes(b"%PDF-1.4")
    docx.write_bytes(b"fake-docx")
    engine, deps = _make_engine(
        parser_rules="docx:native-iteP,*:mineru-iteP",
        chunk_options={"paragraph_semantic": {"chunk_overlap_token_size": 99}},
    )
    deps["stores"].get_doc_status.side_effect = [
        None,
        None,
        {"chunks_list": ["chunk-docx"], "content_hash": "sha256:docx", "status": "processed"},
        {"chunks_list": ["chunk-pdf"], "content_hash": "sha256:pdf", "status": "processed"},
    ]
    deps["stores"].get_full_doc.side_effect = [
        {
            "parse_engine": "native",
            "process_options": "iteP",
            "chunk_options": {},
            "sidecar_location": None,
        },
        {
            "parse_engine": "mineru",
            "process_options": "iteP",
            "chunk_options": {},
            "sidecar_location": None,
        },
    ]

    await engine.aingest_files([docx, pdf], replace=False)

    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["chunk_options"] == [
        {"paragraph_semantic": {"chunk_overlap_token_size": 99}},
        {
            "paragraph_semantic": {
                "chunk_overlap_token_size": 99,
                "chunk_token_size": 1234,
                "drop_references": True,
            }
        },
    ]


async def test_prepared_batch_uses_explicit_download_locator(tmp_path: Path) -> None:
    parser_source = tmp_path / "report__s3_abcd1234.pdf"
    parser_source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [
        None,
        {"chunks_list": ["chunk-report"], "content_hash": "sha256:pdf", "status": "processed"},
    ]
    deps["stores"].get_full_doc.return_value = {
        "parse_engine": "mineru",
        "process_options": "iteP",
        "chunk_options": {},
        "sidecar_location": None,
    }

    result = await engine.aingest_files(
        [
            PreparedIngestFile(
                parser_path=parser_source,
                source_uri="s3://bucket/team-a/report.pdf",
                download_locator="s3://bucket/team-a/report.pdf",
                display_filename="report.pdf",
            )
        ],
        replace=False,
    )

    assert result["processed"] == 1
    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["file_paths"] == [str(parser_source)]
    _, saved = deps["metadata_index"].upsert.await_args.args
    assert saved["source_uri"] == "s3://bucket/team-a/report.pdf"
    assert saved["download_locator"] == "s3://bucket/team-a/report.pdf"
    assert saved["filename"] == "report.pdf"
    assert saved["filename_stem"] == "report"
    assert saved["file_extension"] == "pdf"


def test_raw_path_preparation_uses_collision_safe_local_identity(tmp_path: Path) -> None:
    first = tmp_path / "team-a" / "report.pdf"
    second = tmp_path / "team-b" / "report.pdf"
    first.parent.mkdir()
    second.parent.mkdir()

    first_item = _prepare_ingest_item(first, workspace="finance_team")
    second_item = _prepare_ingest_item(second, workspace="finance_team")

    assert first_item.source_uri.startswith("local://finance_team/")
    assert second_item.source_uri.startswith("local://finance_team/")
    assert first_item.source_uri.endswith("/report.pdf")
    assert second_item.source_uri.endswith("/report.pdf")
    assert first_item.source_uri != second_item.source_uri
    assert str(tmp_path) not in first_item.source_uri
    assert str(tmp_path) not in second_item.source_uri
    assert first_item.download_locator == str(first)
    assert second_item.download_locator == str(second)


async def test_single_file_forwards_explicit_source_contract_to_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, _deps = _make_engine()
    prepare_metadata = MagicMock(wraps=engine._prepare_metadata_record)
    monkeypatch.setattr(engine, "_prepare_metadata_record", prepare_metadata)

    await engine.aingest_file(
        source,
        source_uri="local://default/docs/sample.pdf",
        download_locator=str(source),
    )

    assert prepare_metadata.call_args.kwargs["source_uri"] == ("local://default/docs/sample.pdf")
    assert prepare_metadata.call_args.kwargs["download_locator"] == str(source)


async def test_metadata_only_update_forwards_explicit_source_contract(
    tmp_path: Path, monkeypatch
) -> None:
    content = b"%PDF-1.4"
    source = tmp_path / "sample.pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-a"],
        "content_hash": _sha256(content),
        "status": "processed",
    }
    deps["metadata_index"].get.return_value = {_FINALIZATION_COMPLETE_KEY: True}
    prepare_metadata = MagicMock(wraps=engine._prepare_metadata_record)
    monkeypatch.setattr(engine, "_prepare_metadata_record", prepare_metadata)

    result = await engine.aingest_file(
        source,
        source_uri="local://default/docs/sample.pdf",
        download_locator=str(source),
        title="Updated title",
    )

    assert result["source_kind"] == "metadata_updated"
    assert prepare_metadata.call_args.kwargs["source_uri"] == ("local://default/docs/sample.pdf")
    assert prepare_metadata.call_args.kwargs["download_locator"] == str(source)


@pytest.mark.parametrize(
    ("title", "expected_source_kind"),
    [
        pytest.param(None, "skipped", id="unchanged_metadata"),
        pytest.param("Updated title", "metadata_updated", id="updated_metadata"),
    ],
)
async def test_single_hash_match_bypasses_parser_directives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    title: str | None,
    expected_source_kind: str,
) -> None:
    content = b"%PDF-1.4"
    source = tmp_path / "sample.pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-a"],
        "content_hash": _sha256(content),
        "status": "processed",
    }
    deps["metadata_index"].get.return_value = {_FINALIZATION_COMPLETE_KEY: True}

    def fail_parser_directives(_path: Path) -> tuple[str, str, dict[str, object] | None]:
        raise AssertionError("parser directives should not be resolved for hash-match fast path")

    monkeypatch.setattr(engine, "_parser_directives_for", fail_parser_directives)

    result = await engine.aingest_file(source, replace=False, title=title)

    assert result["source_kind"] == expected_source_kind


async def test_batch_hash_match_skip_does_not_resolve_invalid_parser_directives(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4"
    source = tmp_path / "bad.[unknown-iteP].pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-a"],
        "content_hash": _sha256(content),
        "status": "processed",
    }
    deps["metadata_index"].get.return_value = {_FINALIZATION_COMPLETE_KEY: True}

    result = await engine.aingest_files([source], replace=False)
    single_result = await engine.aingest_file(source, replace=False)

    assert result == {
        "processed": 1,
        "errors": [],
        "results": [
            {
                "doc_id": compute_mdhash_id(
                    normalize_document_file_path(source),
                    prefix="doc-",
                ),
                "source_kind": "skipped",
                "reason": "content_hash_match",
                "chunks": ["chunk-a"],
            }
        ],
    }
    assert result["results"][0] == single_result
    deps["metadata_index"].upsert.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_batch_replace_validates_all_enqueue_candidates_before_cleanup(
    tmp_path: Path,
) -> None:
    good = tmp_path / "good.pdf"
    bad = tmp_path / "bad.[unknown-iteP].pdf"
    good.write_bytes(b"%PDF-1.4 good")
    bad.write_bytes(b"%PDF-1.4 bad")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [
        {"chunks_list": ["chunk-good"], "content_hash": "sha256:good", "status": "processed"},
        {"chunks_list": ["chunk-bad"], "content_hash": "sha256:bad", "status": "processed"},
    ]

    with pytest.raises(FilenameParserHintError):
        await engine.aingest_files([good, bad], replace=True)

    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()
    deps["metadata_index"].delete.assert_not_awaited()
    deps["metadata_index"].upsert.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()
    deps["lightrag"].apipeline_process_enqueue_documents.assert_not_awaited()


async def test_batch_hash_match_metadata_update_waits_for_enqueue_validation(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4"
    first = tmp_path / "first.pdf"
    bad = tmp_path / "bad.[unknown-iteP].pdf"
    first.write_bytes(content)
    bad.write_bytes(b"%PDF-1.4 bad")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [
        {"chunks_list": ["chunk-first"], "content_hash": _sha256(content), "status": "processed"},
        None,
    ]
    deps["metadata_index"].get.return_value = {
        "filename": "first.pdf",
        "filename_stem": "first",
        "file_path": str(first),
        "source_uri": "local://default/first.pdf",
        "download_locator": str(first),
        "file_extension": "pdf",
        "title": "Old title",
        "custom_metadata": {},
    }

    with pytest.raises(FilenameParserHintError):
        await engine.aingest_files(
            [
                PreparedIngestFile(
                    parser_path=first,
                    source_uri="local://default/first.pdf",
                    download_locator=str(first),
                    title="Updated title",
                ),
                bad,
            ],
            replace=False,
        )

    deps["metadata_index"].get.assert_awaited_once()
    deps["metadata_index"].upsert.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()
    deps["lightrag"].apipeline_process_enqueue_documents.assert_not_awaited()


async def test_failed_document_cleanup_requires_documented_delete_success(
    tmp_path: Path,
) -> None:
    source = tmp_path / "failed.pdf"
    source.write_bytes(b"%PDF-1.4 failed")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": [],
        "content_hash": None,
        "status": "failed",
        "error_msg": "parser failed",
    }
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["lightrag"].adelete_by_doc_id.return_value = None

    with pytest.raises(RuntimeError, match="deletion was not acknowledged"):
        await engine.aingest_files([source], replace=False)

    deps["metadata_index"].delete.assert_not_awaited()
    assert deps["metadata_index"].upsert.await_args.args[1] == {_FINALIZATION_COMPLETE_KEY: False}
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_failed_document_ingest_fails_closed_when_status_snapshot_disappears(
    tmp_path: Path,
) -> None:
    source = tmp_path / "failed.pdf"
    source.write_bytes(b"%PDF-1.4 failed")
    engine, deps = _make_engine()
    failed_status = {"chunks_list": [], "content_hash": None, "status": "failed"}
    deps["stores"].get_doc_status.side_effect = [failed_status, None]

    with pytest.raises(RuntimeError, match="status snapshot is unavailable"):
        await engine.aingest_files([source], replace=False)

    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()
    deps["metadata_index"].delete.assert_not_awaited()
    assert deps["metadata_index"].upsert.await_args.args[1] == {_FINALIZATION_COMPLETE_KEY: False}


async def test_failed_document_retry_cancellation_restores_discoverability(
    tmp_path: Path,
) -> None:
    source = tmp_path / "failed.pdf"
    source.write_bytes(b"%PDF-1.4 failed")
    engine, deps = _make_engine()
    original_status = {
        "chunks_list": [],
        "content_hash": None,
        "status": "failed",
        "error_msg": "parser failed",
    }
    original_metadata = {
        "filename": "failed.pdf",
        "source_uri": "local://default/failed.pdf",
        "download_locator": str(source),
        "title": "Preserved title",
        "custom_metadata": {"department": "finance"},
    }
    deps["stores"].get_doc_status.side_effect = [original_status, original_status, None]
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["metadata_index"].get.return_value = original_metadata
    deps["lightrag"].adelete_by_doc_id.return_value = SimpleNamespace(status="success")
    deps["lightrag"].apipeline_enqueue_documents.side_effect = asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await engine.aingest_files([source], replace=False)

    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    deps["stores"].doc_status.upsert.assert_not_awaited()
    first_metadata_write = deps["metadata_index"].upsert.await_args_list[0]
    assert first_metadata_write.args == (
        doc_id,
        {"_dlightrag_finalization_complete": False},
    )


async def test_concurrent_single_file_replacements_serialize_cleanup(
    tmp_path: Path,
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    statuses = {doc_id: {"status": "processed", "chunks_list": ["old"]}}
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    active_deletes = 0
    max_active_deletes = 0

    async def delete(current: str, **_kwargs: object) -> SimpleNamespace:
        nonlocal active_deletes, max_active_deletes
        active_deletes += 1
        max_active_deletes = max(max_active_deletes, active_deletes)
        await asyncio.sleep(0.01)
        statuses.pop(current, None)
        active_deletes -= 1
        return SimpleNamespace(status="success")

    async def process() -> None:
        statuses[doc_id] = {"status": "processed", "chunks_list": ["new"]}

    deps["lightrag"].adelete_by_doc_id.side_effect = delete
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process

    first, second = await asyncio.gather(
        engine.aingest_file(source, replace=True),
        engine.aingest_file(source, replace=True),
    )

    assert first["doc_id"] == doc_id
    assert second["doc_id"] == doc_id
    assert max_active_deletes == 1
    assert deps["lightrag"].adelete_by_doc_id.await_count == 2


async def test_delete_time_cancellation_after_status_delete_does_not_restore_zombie(
    tmp_path: Path,
) -> None:
    source = tmp_path / "failed.pdf"
    source.write_bytes(b"%PDF-1.4 failed")
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    status = {"status": "failed", "chunks_list": ["stale"], "content_hash": None}
    statuses = {doc_id: dict(status)}
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}

    async def cancel_delete(current: str, **_kwargs: object) -> None:
        statuses.pop(current, None)
        raise asyncio.CancelledError

    async def upsert(rows: dict[str, dict[str, object]]) -> None:
        statuses.update({key: dict(value) for key, value in rows.items()})

    deps["lightrag"].adelete_by_doc_id.side_effect = cancel_delete
    deps["stores"].doc_status.upsert.side_effect = upsert

    with pytest.raises(asyncio.CancelledError):
        await engine.aingest_files([source])

    assert doc_id not in statuses
    deps["stores"].doc_status.upsert.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_cancellation_after_processing_commit_keeps_upstream_processed_and_hidden(
    tmp_path: Path,
) -> None:
    source = tmp_path / "failed.pdf"
    source.write_bytes(b"%PDF-1.4 failed")
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    original = {"status": "failed", "chunks_list": [], "content_hash": None}
    statuses = {doc_id: dict(original)}
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}

    async def upsert_status(rows: dict[str, dict[str, object]]) -> None:
        statuses.update({key: dict(value) for key, value in rows.items()})

    deps["stores"].doc_status.upsert.side_effect = upsert_status

    async def delete(current: str, **_kwargs: object) -> SimpleNamespace:
        statuses.pop(current, None)
        return SimpleNamespace(status="success")

    async def process() -> None:
        statuses[doc_id] = {"status": "processed", "chunks_list": ["new-chunk"]}

    metadata_writes = 0

    async def upsert_metadata(*_args: object) -> None:
        nonlocal metadata_writes
        metadata_writes += 1
        if metadata_writes == 3:
            raise asyncio.CancelledError

    deps["lightrag"].adelete_by_doc_id.side_effect = delete
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process
    deps["metadata_index"].upsert.side_effect = upsert_metadata

    with pytest.raises(asyncio.CancelledError):
        await engine.aingest_files([source])

    assert statuses[doc_id]["status"] == "processed"
    assert statuses[doc_id]["chunks_list"] == ["new-chunk"]
    assert "error_msg" not in statuses[doc_id]
    deps["stores"].doc_status.upsert.assert_not_awaited()
    assert (
        deps["metadata_index"].upsert.await_args_list[0].args[1][_FINALIZATION_COMPLETE_KEY]
        is False
    )


async def test_remote_locator_replacement_deletes_old_metadata_only_after_new_commit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "renamed.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    new_doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_doc_id = "doc-old-locator"
    statuses = {
        old_doc_id: {"status": "processed", "chunks_list": ["old-chunk"]},
    }
    metadata = {
        old_doc_id: {
            "download_locator": "s3://bucket/report.pdf",
            "source_uri": "s3://bucket/report.pdf",
        }
    }
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def delete(current: str, **_kwargs: object) -> SimpleNamespace:
        statuses.pop(current, None)
        return SimpleNamespace(status="success")

    async def process() -> None:
        statuses[new_doc_id] = {"status": "processed", "chunks_list": ["new-chunk"]}

    async def delete_metadata(current: str) -> None:
        assert statuses.get(new_doc_id, {}).get("status") == "processed"
        metadata.pop(current, None)

    deps["lightrag"].adelete_by_doc_id.side_effect = delete
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process
    deps["metadata_index"].delete.side_effect = delete_metadata

    result = await engine.aingest_files(
        [
            PreparedIngestFile(
                parser_path=source,
                source_uri="s3://bucket/report.pdf",
                download_locator="s3://bucket/report.pdf",
                replacement_doc_ids=(old_doc_id,),
                replacement_ownership=(
                    (old_doc_id, "s3://bucket/report.pdf", "s3://bucket/report.pdf"),
                ),
            )
        ],
        replace=True,
    )

    assert result["processed"] == 1
    assert statuses[new_doc_id]["status"] == "processed"
    assert old_doc_id not in metadata
    deps["metadata_index"].delete.assert_awaited_once_with(old_doc_id)


async def test_batch_partial_cleanup_waits_for_enqueue_validation(
    tmp_path: Path,
) -> None:
    good = tmp_path / "good.pdf"
    bad = tmp_path / "bad.[unknown-iteP].pdf"
    good.write_bytes(b"%PDF-1.4 good")
    bad.write_bytes(b"%PDF-1.4 bad")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [
        {"chunks_list": ["chunk-good"], "content_hash": "sha256:stale", "status": "analyzing"},
        None,
    ]

    with pytest.raises(FilenameParserHintError):
        await engine.aingest_files([good, bad], replace=False)

    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()
    deps["metadata_index"].delete.assert_not_awaited()
    deps["metadata_index"].upsert.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()
    deps["lightrag"].apipeline_process_enqueue_documents.assert_not_awaited()


async def test_single_hash_match_source_contract_change_updates_metadata(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4"
    source = tmp_path / "sample.pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-a"],
        "content_hash": _sha256(content),
        "status": "processed",
    }
    deps["metadata_index"].get.return_value = {
        "filename": "sample.pdf",
        "filename_stem": "sample",
        "file_path": "https://cdn.example.com/old-sample.pdf",
        "source_uri": "bynder://asset/old",
        "download_locator": "https://cdn.example.com/old-sample.pdf",
        "file_extension": "pdf",
        "custom_metadata": {},
        _FINALIZATION_COMPLETE_KEY: True,
    }

    result = await engine.aingest_file(
        source,
        source_uri="bynder://asset/new",
        download_locator="https://cdn.example.com/new-sample.pdf",
        display_filename="renamed-sample.pdf",
        replace=False,
    )

    assert result == {
        "doc_id": compute_mdhash_id(normalize_document_file_path(source), prefix="doc-"),
        "source_kind": "metadata_updated",
        "reason": "content_hash_match",
        "chunks": ["chunk-a"],
    }
    deps["metadata_index"].upsert.assert_awaited_once()
    _, saved = deps["metadata_index"].upsert.await_args.args
    assert saved["filename"] == "renamed-sample.pdf"
    assert saved["filename_stem"] == "renamed-sample"
    assert saved["source_uri"] == "bynder://asset/new"
    assert saved["download_locator"] == "https://cdn.example.com/new-sample.pdf"
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_single_hash_match_local_noop_checks_finalization_marker(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4"
    source = tmp_path / "sample.pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-a"],
        "content_hash": _sha256(content),
        "status": "processed",
    }
    deps["metadata_index"].get.return_value = {_FINALIZATION_COMPLETE_KEY: True}

    result = await engine.aingest_file(source, replace=False)

    assert result == {
        "doc_id": compute_mdhash_id(normalize_document_file_path(source), prefix="doc-"),
        "source_kind": "skipped",
        "reason": "content_hash_match",
        "chunks": ["chunk-a"],
    }
    deps["metadata_index"].get.assert_awaited_once()
    deps["metadata_index"].upsert.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_single_hash_match_internal_local_contract_checks_finalization_marker(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4"
    source = tmp_path / "sample.pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-a"],
        "content_hash": _sha256(content),
        "status": "processed",
    }
    deps["metadata_index"].get.return_value = {_FINALIZATION_COMPLETE_KEY: True}

    result = await engine.aingest_file(
        source,
        source_uri=_raw_path_source_uri(source, workspace="default"),
        download_locator=str(source.resolve()),
        source_uri_explicit=False,
        download_locator_explicit=False,
        replace=False,
    )

    assert result == {
        "doc_id": compute_mdhash_id(normalize_document_file_path(source), prefix="doc-"),
        "source_kind": "skipped",
        "reason": "content_hash_match",
        "chunks": ["chunk-a"],
    }
    deps["metadata_index"].get.assert_awaited_once()
    deps["metadata_index"].upsert.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_single_hash_match_explicit_default_source_contract_updates_metadata(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4"
    source = tmp_path / "sample.pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-a"],
        "content_hash": _sha256(content),
        "status": "processed",
    }
    deps["metadata_index"].get.return_value = {
        "filename": "old-name.pdf",
        "filename_stem": "old-name",
        "file_path": "https://cdn.example.com/old-sample.pdf",
        "source_uri": "bynder://asset/old",
        "download_locator": "https://cdn.example.com/old-sample.pdf",
        "file_extension": "pdf",
        "custom_metadata": {},
        _FINALIZATION_COMPLETE_KEY: True,
    }

    result = await engine.aingest_file(
        source,
        source_uri=_raw_path_source_uri(source, workspace="default"),
        download_locator=str(source.resolve()),
        display_filename=source.name,
        replace=False,
    )

    assert result == {
        "doc_id": compute_mdhash_id(normalize_document_file_path(source), prefix="doc-"),
        "source_kind": "metadata_updated",
        "reason": "content_hash_match",
        "chunks": ["chunk-a"],
    }
    deps["metadata_index"].get.assert_awaited_once()
    deps["metadata_index"].upsert.assert_awaited_once()
    _, saved = deps["metadata_index"].upsert.await_args.args
    assert saved["filename"] == "sample.pdf"
    assert saved["filename_stem"] == "sample"
    assert saved["source_uri"] == _raw_path_source_uri(source, workspace="default")
    assert saved["download_locator"] == str(source.resolve())
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_batch_metadata_only_update_preserves_source_contract_and_chunks(
    tmp_path: Path, monkeypatch
) -> None:
    content = b"%PDF-1.4"
    parser_source = tmp_path / "report__s3_abcd1234.pdf"
    parser_source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-report"],
        "content_hash": _sha256(content),
        "status": "processed",
    }
    deps["metadata_index"].get.return_value = {_FINALIZATION_COMPLETE_KEY: True}
    prepare_metadata = MagicMock(wraps=engine._prepare_metadata_record)
    monkeypatch.setattr(engine, "_prepare_metadata_record", prepare_metadata)

    result = await engine.aingest_files(
        [
            PreparedIngestFile(
                parser_path=parser_source,
                source_uri="bynder://asset/1",
                download_locator="https://cdn.example.com/assets/1.pdf",
                display_filename="report.pdf",
                title="Updated title",
                author="Updated author",
                metadata={"category": "finance"},
            )
        ],
        replace=False,
    )

    assert result["processed"] == 1
    assert result["errors"] == []
    assert result["results"] == [
        {
            "doc_id": compute_mdhash_id(
                normalize_document_file_path(parser_source),
                prefix="doc-",
            ),
            "source_kind": "metadata_updated",
            "reason": "content_hash_match",
            "chunks": ["chunk-report"],
        }
    ]
    assert prepare_metadata.call_args.kwargs["source_uri"] == "bynder://asset/1"
    assert prepare_metadata.call_args.kwargs["download_locator"] == (
        "https://cdn.example.com/assets/1.pdf"
    )
    _, saved = deps["metadata_index"].upsert.await_args.args
    assert saved["source_uri"] == "bynder://asset/1"
    assert saved["download_locator"] == "https://cdn.example.com/assets/1.pdf"
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()
    deps["lightrag"].apipeline_process_enqueue_documents.assert_not_awaited()
    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()


async def test_document_ingest_uses_lightrag_canonical_doc_id(tmp_path: Path) -> None:
    source = tmp_path / "1912.09363v3.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()

    result = await engine.aingest_file(source, replace=False)

    expected_doc_id = compute_mdhash_id(
        normalize_document_file_path(source),
        prefix="doc-",
    )
    assert result["doc_id"] == expected_doc_id
    assert deps["metadata_index"].upsert.await_count == 3
    assert all(
        call.args[0] == expected_doc_id for call in deps["metadata_index"].upsert.await_args_list
    )


async def test_pending_metadata_is_persisted_before_parser_enqueue_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = None
    persisted: list[dict] = []

    async def save_metadata(_doc_id: str, metadata: dict) -> None:
        persisted.append(metadata)

    async def fail_enqueue(**_kwargs) -> None:
        assert persisted
        raise RuntimeError("parser enqueue failed")

    deps["metadata_index"].upsert = AsyncMock(side_effect=save_metadata)
    deps["lightrag"].apipeline_enqueue_documents = AsyncMock(side_effect=fail_enqueue)

    with pytest.raises(RuntimeError, match="parser enqueue failed"):
        await engine.aingest_file(
            source,
            source_uri="bynder://asset/1",
            download_locator="https://cdn.example.com/assets/1.pdf",
        )

    assert persisted == [
        {
            "filename": "1.pdf",
            "filename_stem": "1",
            "source_uri": "bynder://asset/1",
            "download_locator": "https://cdn.example.com/assets/1.pdf",
            "file_extension": "pdf",
            "custom_metadata": {},
            _FINALIZATION_COMPLETE_KEY: False,
        }
    ]


async def test_batch_pending_metadata_is_persisted_before_parser_enqueue_failure(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"%PDF-1.4 first")
    second.write_bytes(b"%PDF-1.4 second")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [None, None]
    persisted: list[tuple[str, dict]] = []

    async def save_metadata(doc_id: str, metadata: dict) -> None:
        persisted.append((doc_id, metadata))

    async def fail_enqueue(**_kwargs) -> None:
        assert len(persisted) == 2
        raise RuntimeError("batch parser enqueue failed")

    deps["metadata_index"].upsert = AsyncMock(side_effect=save_metadata)
    deps["lightrag"].apipeline_enqueue_documents = AsyncMock(side_effect=fail_enqueue)

    with pytest.raises(RuntimeError, match="batch parser enqueue failed"):
        await engine.aingest_files(
            [
                PreparedIngestFile(
                    parser_path=first,
                    source_uri="bynder://asset/1",
                    download_locator="https://cdn.example.com/assets/1.pdf",
                ),
                PreparedIngestFile(
                    parser_path=second,
                    source_uri="bynder://asset/2",
                    download_locator="s3://documents/assets/2.pdf",
                ),
            ]
        )

    assert [metadata["source_uri"] for _, metadata in persisted] == [
        "bynder://asset/1",
        "bynder://asset/2",
    ]
    assert [metadata["download_locator"] for _, metadata in persisted] == [
        "https://cdn.example.com/assets/1.pdf",
        "s3://documents/assets/2.pdf",
    ]


async def test_post_processing_failure_keeps_processed_status_hidden_for_retry(
    tmp_path: Path,
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    processed = {"status": "processed", "chunks_list": ["chunk-a"], "content_hash": "hash"}
    deps["stores"].get_doc_status.side_effect = [None, processed]
    writes = 0

    async def metadata_write(*_args: object) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise RuntimeError("metadata index unavailable")

    deps["metadata_index"].upsert.side_effect = metadata_write

    result = await engine.aingest_files([source])

    assert result["processed"] == 0
    assert result["errors"] == ["report.pdf: document processing failed"]
    assert processed["status"] == "processed"
    deps["stores"].doc_status.upsert.assert_not_awaited()
    assert (
        deps["metadata_index"].upsert.await_args_list[0].args[1][_FINALIZATION_COMPLETE_KEY]
        is False
    )


async def test_batch_finalization_aggregates_failed_and_processed_documents(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"%PDF-1.4 first")
    second.write_bytes(b"%PDF-1.4 second")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [
        None,
        None,
        {
            "status": DocStatus.FAILED,
            "chunks_list": [],
            "error_msg": "private parser detail",
        },
        {
            "status": DocStatus.PROCESSED,
            "chunks_list": ["chunk-second"],
            "content_hash": "sha256:second",
        },
    ]

    result = await engine.aingest_files(
        [
            PreparedIngestFile(
                parser_path=first,
                source_uri="bynder://asset/1",
                download_locator="https://cdn.example.com/assets/1.pdf",
                display_filename="first.pdf",
            ),
            PreparedIngestFile(
                parser_path=second,
                source_uri="bynder://asset/2",
                download_locator="https://cdn.example.com/assets/2.pdf",
                display_filename="second.pdf",
            ),
        ]
    )

    assert result["processed"] == 1
    assert result["errors"] == ["first.pdf: document processing failed"]
    assert len(result["results"]) == 1
    assert result["results"][0]["chunks"] == ["chunk-second"]
    assert deps["stores"].get_doc_status.await_count == 4
    assert deps["metadata_index"].upsert.await_count == 3


async def test_document_ingest_delegates_non_sidecar_parser_route(tmp_path: Path) -> None:
    """LightRAG routing is the ingestability boundary.

    DlightRAG enqueues the LightRAG-resolved parser route and skips sidecar
    vector overrides when that route does not produce a sidecar location.
    """
    source = tmp_path / "notes.docx"
    source.write_bytes(b"fake docx")
    engine, deps = _make_engine()
    deps["stores"].get_full_doc.return_value = {
        "parse_engine": "native",
        "process_options": "iteP",
        "chunk_options": {},
        "sidecar_location": None,
    }

    result = await engine.aingest_file(source, replace=False)

    assert result["doc_id"] is not None
    assert result["parse_engine"] == "native"
    assert result["process_options"] == "iteP"
    assert result["chunks"] == ["chunk-a"]
    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["parse_engine"] == ["native"]
    assert kwargs["process_options"] == ["iteP"]
    deps["stores"].overwrite_chunk_vectors.assert_not_awaited()


async def test_document_ingest_accepts_explicit_user_metadata(tmp_path: Path) -> None:
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()

    await engine.aingest_file(
        source,
        replace=False,
        metadata={"reviewer": " Ada Lovelace ", "project": "Analytical Engine"},
    )

    _, saved = deps["metadata_index"].upsert.await_args.args
    assert saved["custom_metadata"] == {
        "reviewer": " Ada Lovelace ",
        "project": "Analytical Engine",
    }


async def test_prepared_file_metadata_overlays_batch_metadata(tmp_path: Path) -> None:
    source = tmp_path / "asset.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()

    await engine.aingest_files(
        [
            PreparedIngestFile(
                parser_path=source,
                source_uri="local://default/asset.pdf",
                download_locator=str(source),
                metadata={"department": " Legal ", "asset_id": "A-123"},
            )
        ],
        replace=False,
        metadata={"source_system": "Bynder", "department": "Marketing"},
    )

    _, saved = deps["metadata_index"].upsert.await_args.args
    assert saved["custom_metadata"] == {
        "source_system": "Bynder",
        "department": " Legal ",
        "asset_id": "A-123",
    }


async def test_image_file_ingest_delegates_to_lightrag_parser(
    tmp_path: Path,
) -> None:
    from PIL import Image

    source = tmp_path / "image.png"
    Image.new("RGB", (1, 1), "white").save(source)
    engine, deps = _make_engine()

    result = await engine.aingest_file(source)

    assert result["source_kind"] == "document"
    deps["stores"].overwrite_chunk_vectors.assert_not_awaited()
    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["docs_format"] == "pending_parse"
    assert kwargs["parse_engine"] == ["mineru"]
    assert kwargs["process_options"] == ["iteP"]


async def test_document_ingest_cleans_up_partial_before_reingest(tmp_path: Path) -> None:
    """When a doc exists with status 'analyzing' (interrupted MinerU run),
    re-ingesting must clean up the partial record and proceed normally."""
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    artifact_dir = tmp_path / "old.parsed"
    artifact_dir.mkdir()
    events: list[str] = []

    # Simulate a partial record from an interrupted ingest.
    partial_status = {
        "chunks_list": ["old-chunk-1"],
        "content_hash": "sha256:deadbeef",
        "status": "analyzing",
    }
    deps["stores"].get_doc_status.side_effect = [
        partial_status,
        partial_status,
        {"chunks_list": ["chunk-a"], "content_hash": "sha256:abc", "status": "processed"},
    ]

    async def get_full_doc(doc_id_arg: str) -> dict | None:
        assert doc_id_arg == doc_id
        events.append("get_full_doc")
        if "adelete_by_doc_id" in events:
            return {
                "parse_engine": "mineru",
                "process_options": "iteP",
                "chunk_options": {},
                "sidecar_location": None,
            }
        return {
            "parse_engine": "mineru",
            "process_options": "iteP",
            "chunk_options": {},
            "sidecar_location": artifact_dir.as_uri(),
        }

    async def delete_doc(doc_id_arg: str, *, delete_llm_cache: bool) -> object:
        assert doc_id_arg == doc_id
        assert delete_llm_cache is True
        events.append("adelete_by_doc_id")
        return type("DeletionResult", (), {"status": "success"})()

    deps["stores"].get_full_doc = AsyncMock(side_effect=get_full_doc)
    deps["lightrag"].adelete_by_doc_id = AsyncMock(side_effect=delete_doc)

    result = await engine.aingest_file(source, replace=False)

    # Must have cleaned up the old partial record.
    deps["lightrag"].adelete_by_doc_id.assert_awaited_once_with(doc_id, delete_llm_cache=True)
    deps["metadata_index"].delete.assert_not_awaited()
    deps["metadata_index"].upsert.assert_awaited()
    assert events[:2] == ["get_full_doc", "adelete_by_doc_id"]
    assert not artifact_dir.exists()

    # Must have proceeded with normal ingest.
    assert result["doc_id"] == doc_id
    deps["lightrag"].apipeline_enqueue_documents.assert_awaited_once()
    deps["lightrag"].apipeline_process_enqueue_documents.assert_awaited_once()


async def test_document_ingest_replaces_processed_hash_mismatch(tmp_path: Path) -> None:
    """Pinned LightRAG rejects duplicate IDs, so changed content is cleaned first."""
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-1"],
        "content_hash": "sha256:abc",
        "status": "processed",
    }

    await engine.aingest_file(source, replace=False)

    deps["lightrag"].adelete_by_doc_id.assert_awaited_once()
    deps["metadata_index"].delete.assert_not_awaited()


async def test_document_ingest_first_time_no_cleanup(tmp_path: Path) -> None:
    """When no prior doc_status exists, ingest proceeds without cleanup."""
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [
        None,
        {"chunks_list": ["chunk-a"], "content_hash": "sha256:abc", "status": "processed"},
    ]

    result = await engine.aingest_file(source, replace=False)

    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()
    assert result["doc_id"] is not None


async def test_parser_image_sidecar_overwrites_lightrag_mm_chunk_vector(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    mm_chunk_id = f"{doc_id}-mm-drawing-000"
    artifact_dir = tmp_path / "sample.parsed"
    assets_dir = artifact_dir / "sample.blocks.assets"
    assets_dir.mkdir(parents=True)
    (artifact_dir / "sample.blocks.jsonl").write_text("", encoding="utf-8")
    image_path = assets_dir / "fig.png"
    Image.new("RGB", (128, 128), "white").save(image_path)
    (artifact_dir / "sample.drawings.json").write_text(
        """
        {
          "drawings": {
            "fig-1": {
              "id": "fig-1",
              "path": "sample.blocks.assets/fig.png",
              "llm_analyze_result": {
                "status": "success",
                "name": "Harness QR",
                "type": "QR code",
                "description": "hallucinated harness lifecycle description"
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )
    document_embedder = AsyncMock()
    document_embedder.image_enabled = True
    document_embedder.dimension = 3
    document_embedder.aembed_documents.return_value = (
        [DocumentEmbeddingVector(mm_chunk_id, [0.1, 0.2, 0.3], "fused")],
        DocumentEmbeddingTrace(fused=1, text=0, fused_to_text_fallback=0, failed=0),
    )
    engine, deps = _make_engine(document_embedder=document_embedder)
    deps["stores"].fetch_chunk_contents.return_value = [
        {"id": mm_chunk_id, "content": "public/private sector mapping chart"}
    ]
    deps["stores"].get_doc_status.side_effect = [
        None,
        {
            "chunks_list": ["chunk-a", mm_chunk_id],
            "content_hash": "sha256:parsed",
            "status": "processed",
        },
    ]
    deps["stores"].get_full_doc.return_value = {
        "parse_engine": "mineru",
        "process_options": "iteP",
        "chunk_options": {},
        "sidecar_location": artifact_dir.as_uri(),
    }

    result = await engine.aingest_file(source, replace=False)

    assert result["chunks"] == ["chunk-a", mm_chunk_id]
    deps["stores"].overwrite_chunk_vectors.assert_awaited_once()
    vectors = deps["stores"].overwrite_chunk_vectors.await_args.args[0]
    assert vectors == {mm_chunk_id: [0.1, 0.2, 0.3]}
    document_embedder.aembed_documents.assert_awaited_once_with(
        [
            DocumentEmbeddingInput(
                key=mm_chunk_id,
                text="public/private sector mapping chart",
                image_path=image_path,
            )
        ]
    )


async def test_parser_image_sidecar_skips_vector_overwrite_when_direct_embedding_disabled(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    mm_chunk_id = f"{doc_id}-mm-drawing-000"
    artifact_dir = tmp_path / "sample.parsed"
    assets_dir = artifact_dir / "sample.blocks.assets"
    assets_dir.mkdir(parents=True)
    image_path = assets_dir / "fig.png"
    Image.new("RGB", (128, 128), "white").save(image_path)
    (artifact_dir / "sample.drawings.json").write_text(
        """
        {
          "drawings": {
            "fig-1": {
              "id": "fig-1",
              "path": "sample.blocks.assets/fig.png",
              "llm_analyze_result": {
                "status": "success",
                "description": "LightRAG semantic visual chunk"
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )
    document_embedder = AsyncMock()
    document_embedder.image_enabled = False
    document_embedder.dimension = 3
    engine, deps = _make_engine(
        document_embedder=document_embedder,
    )
    deps["stores"].get_doc_status.side_effect = [
        None,
        {
            "chunks_list": ["chunk-a", mm_chunk_id],
            "content_hash": "sha256:parsed",
            "status": "processed",
        },
    ]
    deps["stores"].get_full_doc.return_value = {
        "parse_engine": "mineru",
        "process_options": "iteP",
        "chunk_options": {},
        "sidecar_location": artifact_dir.as_uri(),
    }

    result = await engine.aingest_file(source, replace=False)

    assert result["chunks"] == ["chunk-a", mm_chunk_id]
    document_embedder.aembed_documents.assert_not_awaited()
    deps["stores"].overwrite_chunk_vectors.assert_not_awaited()


async def test_concurrent_ingest_of_same_doc_is_serialized(tmp_path: Path) -> None:
    """Two concurrent ingests of the same failed doc must NOT both clean up.
    The per-doc lock ensures the second sees the first's state changes."""
    import asyncio

    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()

    status_iter = iter(
        [
            {"chunks_list": [], "content_hash": "sha256:dead", "status": "failed"},
            {"chunks_list": ["chunk-1"], "content_hash": "sha256:abc", "status": "processing"},
        ]
    )

    async def status_side_effect(doc_id_arg: str) -> dict | None:
        assert doc_id_arg == compute_mdhash_id(
            normalize_document_file_path(source),
            prefix="doc-",
        )
        try:
            return next(status_iter)
        except StopIteration:
            return {
                "chunks_list": ["chunk-1"],
                "content_hash": _sha256(b"%PDF-1.4"),
                "status": "processed",
            }

    deps["stores"].get_doc_status = AsyncMock(side_effect=status_side_effect)
    deps["stores"].get_full_doc.return_value = {
        "parse_engine": "mineru",
        "process_options": "iteP",
        "chunk_options": {},
        "sidecar_location": "file:///tmp/sample.parsed/",
    }

    async def slow_delete(doc_id_arg: str, *, delete_llm_cache: bool) -> object:
        assert doc_id_arg == compute_mdhash_id(
            normalize_document_file_path(source),
            prefix="doc-",
        )
        assert delete_llm_cache is True
        await asyncio.sleep(0.03)
        return type("DeletionResult", (), {"status": "success"})()

    deps["lightrag"].adelete_by_doc_id = AsyncMock(side_effect=slow_delete)

    async def ingest() -> dict:
        return await engine.aingest_file(source, replace=False)

    results = await asyncio.gather(ingest(), ingest())
    assert len(results) == 2
    # Cleanup must have been called exactly once (not twice).
    assert deps["lightrag"].adelete_by_doc_id.await_count == 1


async def test_reingest_skips_when_content_hash_matches(tmp_path: Path) -> None:
    """Re-ingesting finalized content with the same hash returns early."""
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")

    current_hash = _file_sha256_static(source)
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-1", "chunk-2"],
        "content_hash": current_hash,
        "status": "processed",
    }
    deps["metadata_index"].get.return_value = {_FINALIZATION_COMPLETE_KEY: True}

    result = await engine.aingest_file(source, replace=False)

    assert result["doc_id"] == doc_id
    assert result["source_kind"] == "skipped"
    assert result["reason"] == "content_hash_match"
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_reingest_hash_check_runs_off_event_loop(tmp_path: Path, monkeypatch) -> None:
    import asyncio

    import dlightrag.engine.rag.corpus.ingestion.engine as engine_module

    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-1"],
        "content_hash": _file_sha256_static(source),
        "status": "processed",
    }
    calls = []

    async def fake_to_thread(func, *args, **kwargs):
        calls.append(func)
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    await engine.aingest_file(source, replace=False)

    assert engine_module._file_sha256 in calls


def _file_sha256_static(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


async def test_reingest_proceeds_when_content_hash_differs(tmp_path: Path) -> None:
    """Re-ingesting with different content_hash must proceed normally."""
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-1"],
        "content_hash": "sha256:different_hash",
        "status": "processed",
    }

    result = await engine.aingest_file(source, replace=False)

    assert result.get("source_kind") != "skipped"
    deps["lightrag"].apipeline_enqueue_documents.assert_awaited_once()


async def test_reingest_proceeds_when_not_processed(tmp_path: Path) -> None:
    """Re-ingesting a failed doc must proceed even if hash matches."""
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()

    current_hash = _file_sha256_static(source)
    failed_status = {
        "chunks_list": [],
        "content_hash": current_hash,
        "status": "failed",
    }
    deps["stores"].get_doc_status.side_effect = [
        failed_status,
        failed_status,
        {"chunks_list": ["chunk-a"], "content_hash": current_hash, "status": "processed"},
    ]

    result = await engine.aingest_file(source, replace=False)

    assert result.get("source_kind") != "skipped"
    deps["lightrag"].apipeline_enqueue_documents.assert_awaited_once()


async def test_sidecar_image_vectors_delegate_document_inputs(tmp_path: Path) -> None:
    import json

    artifact_dir = tmp_path / "sample.parsed"
    artifact_dir.mkdir()
    (artifact_dir / "sample.blocks.jsonl").write_text("{}\n", encoding="utf-8")
    image_path = artifact_dir / "chart.png"
    Image.new("RGB", (128, 128), color=(255, 0, 0)).save(image_path)
    (artifact_dir / "sample.drawings.json").write_text(
        json.dumps(
            {
                "drawings": {
                    "im-1": {
                        "path": "chart.png",
                        "llm_analyze_result": {"status": "success"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    engine, deps = _make_engine()
    deps["stores"].overwrite_chunk_vectors = AsyncMock()
    deps["stores"].fetch_chunk_contents = AsyncMock(
        return_value=[{"id": "doc-1-mm-drawing-000", "content": "chart"}]
    )
    deps["document_embedder"].dimension = 2
    deps["document_embedder"].aembed_documents.return_value = (
        [DocumentEmbeddingVector("doc-1-mm-drawing-000", [0.1, 0.2], "fused")],
        DocumentEmbeddingTrace(fused=1, text=0, fused_to_text_fallback=0, failed=0),
    )

    await engine._overwrite_sidecar_image_vectors(
        doc_id="doc-1",
        sidecar_location=artifact_dir.as_uri(),
        chunk_ids={"doc-1-mm-drawing-000"},
    )

    deps["document_embedder"].aembed_documents.assert_awaited_once_with(
        [
            DocumentEmbeddingInput(
                key="doc-1-mm-drawing-000",
                text="chart",
                image_path=image_path,
            )
        ]
    )
    deps["stores"].overwrite_chunk_vectors.assert_awaited_once()


async def test_sidecar_image_embed_failure_fails_finalization(tmp_path: Path) -> None:
    import json

    artifact_dir = tmp_path / "sample.parsed"
    artifact_dir.mkdir()
    (artifact_dir / "sample.blocks.jsonl").write_text("{}\n", encoding="utf-8")
    Image.new("RGB", (128, 128), color=(0, 128, 255)).save(artifact_dir / "chart.png")
    (artifact_dir / "sample.drawings.json").write_text(
        json.dumps(
            {
                "drawings": {
                    "im-1": {
                        "path": "chart.png",
                        "llm_analyze_result": {"status": "success"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    engine, deps = _make_engine()
    deps["stores"].overwrite_chunk_vectors = AsyncMock()
    deps["stores"].fetch_chunk_contents = AsyncMock(
        return_value=[{"id": "doc-1-mm-drawing-000", "content": "chart"}]
    )
    deps["document_embedder"].aembed_documents = AsyncMock(
        side_effect=RuntimeError("provider rejected oversized image")
    )

    with pytest.raises(RuntimeError, match="provider rejected"):
        await engine._overwrite_sidecar_image_vectors(
            doc_id="doc-1",
            sidecar_location=artifact_dir.as_uri(),
            chunk_ids={"doc-1-mm-drawing-000"},
        )

    deps["stores"].overwrite_chunk_vectors.assert_not_awaited()


async def test_sidecar_unreadable_image_fails_required_visual_fusion(tmp_path: Path) -> None:
    import json

    artifact_dir = tmp_path / "sample.parsed"
    artifact_dir.mkdir()
    (artifact_dir / "sample.blocks.jsonl").write_text("{}\n", encoding="utf-8")
    Image.new("RGB", (128, 128), color=(0, 200, 0)).save(artifact_dir / "good.png")
    (artifact_dir / "bad.png").write_bytes(b"not a real image")
    (artifact_dir / "sample.drawings.json").write_text(
        json.dumps(
            {
                "drawings": {
                    "im-1": {"path": "good.png", "llm_analyze_result": {"status": "success"}},
                    "im-2": {"path": "bad.png", "llm_analyze_result": {"status": "success"}},
                }
            }
        ),
        encoding="utf-8",
    )
    good_chunk = "doc-1-mm-drawing-000"
    bad_chunk = "doc-1-mm-drawing-001"
    engine, deps = _make_engine()
    deps["stores"].fetch_chunk_contents = AsyncMock(
        return_value=[
            {"id": good_chunk, "content": "keep"},
            {"id": bad_chunk, "content": "boom"},
        ]
    )
    deps["document_embedder"].dimension = 2
    deps["document_embedder"].aembed_documents.return_value = (
        [
            DocumentEmbeddingVector(good_chunk, [0.5, 0.6], "fused"),
            DocumentEmbeddingVector(bad_chunk, [0.7, 0.8], "text"),
        ],
        DocumentEmbeddingTrace(fused=1, text=1, fused_to_text_fallback=0, failed=0),
    )

    with pytest.raises(RuntimeError, match="required sidecar visual fusion"):
        await engine._overwrite_sidecar_image_vectors(
            doc_id="doc-1",
            sidecar_location=artifact_dir.as_uri(),
            chunk_ids={good_chunk, bad_chunk},
        )

    deps["stores"].overwrite_chunk_vectors.assert_not_awaited()


def test_resolve_sidecar_uri_handles_file_scheme() -> None:
    from pathlib import Path

    from lightrag.utils_pipeline import resolve_sidecar_uri

    assert resolve_sidecar_uri("file:///tmp/sample.parsed/") == Path("/tmp/sample.parsed")
    assert resolve_sidecar_uri("file:///tmp/path%20with%20spaces/") == Path("/tmp/path with spaces")


def test_resolve_sidecar_uri_rejects_everything_that_is_not_a_local_sidecar() -> None:
    """The unknown sentinel must never resolve: engine cleanup rmtree's the result."""
    from lightrag.utils_pipeline import SIDECAR_LOCATION_UNKNOWN, resolve_sidecar_uri

    assert resolve_sidecar_uri(SIDECAR_LOCATION_UNKNOWN) is None
    assert resolve_sidecar_uri("s3://bucket/key/parsed/") is None
    assert resolve_sidecar_uri("azure://container/path/") is None
    assert resolve_sidecar_uri("/tmp/local/path") is None
    assert resolve_sidecar_uri(None) is None
    assert resolve_sidecar_uri("") is None


async def test_batch_rejects_duplicate_canonical_ids_before_mutation(tmp_path: Path) -> None:
    first = tmp_path / "a" / "report.pdf"
    second = tmp_path / "b" / "report.pdf"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    engine, deps = _make_engine()

    for paths in ([first, first], [first, second]):
        with pytest.raises(ValueError, match="duplicate canonical document IDs"):
            await engine.aingest_files(paths)

    deps["stores"].get_doc_status.assert_not_awaited()
    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()
    deps["metadata_index"].upsert.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_final_delete_failure_with_missing_status_never_restores_zombie(
    tmp_path: Path,
) -> None:
    source = tmp_path / "failed.pdf"
    source.write_bytes(b"failed")
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    statuses = {doc_id: {"status": "failed", "chunks_list": ["old"]}}
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}

    async def final_stage_failure(current: str, **_kwargs: object) -> SimpleNamespace:
        statuses.pop(current, None)
        return SimpleNamespace(status="fail")

    deps["lightrag"].adelete_by_doc_id.side_effect = final_stage_failure
    deps["lightrag"].apipeline_enqueue_documents.side_effect = RuntimeError("enqueue down")

    with pytest.raises(RuntimeError, match="enqueue down"):
        await engine.aingest_files([source])

    assert doc_id not in statuses
    deps["stores"].doc_status.upsert.assert_not_awaited()


async def test_metadata_only_remote_tombstone_recovers_and_retires_old_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_id = "doc-old"
    locator = "s3://bucket/item.pdf"
    source_uri = "bynder://asset/1"
    statuses: dict[str, dict[str, object]] = {}
    metadata = {old_id: {"download_locator": locator, "source_uri": source_uri}}
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def process() -> None:
        statuses[new_id] = {"status": "processed", "chunks_list": ["new"]}

    async def delete_metadata(current: str) -> None:
        metadata.pop(current, None)

    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process
    deps["metadata_index"].delete.side_effect = delete_metadata
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=(old_id,),
        replacement_ownership=((old_id, locator, source_uri),),
    )

    result = await engine.aingest_files([item], replace=True)

    assert result["processed"] == 1
    assert statuses[new_id]["status"] == "processed"
    assert old_id not in metadata
    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()


async def test_failed_metadata_tombstone_replacement_keeps_only_new_failed_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_id = "doc-old"
    locator = "s3://bucket/item.pdf"
    source_uri = "bynder://asset/1"
    statuses: dict[str, dict[str, object]] = {}
    metadata = {old_id: {"download_locator": locator, "source_uri": source_uri}}
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def process() -> None:
        statuses[new_id] = {"status": "failed", "chunks_list": [], "error_msg": "parse"}

    async def delete_metadata(current: str) -> None:
        metadata.pop(current, None)

    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process
    deps["metadata_index"].delete.side_effect = delete_metadata
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=(old_id,),
        replacement_ownership=((old_id, locator, source_uri),),
    )

    result = await engine.aingest_files([item], replace=True)

    assert result["processed"] == 0
    assert set(statuses) == {new_id}
    assert statuses[new_id]["status"] == "failed"
    assert old_id not in metadata


async def test_replacement_revalidates_locator_ownership_after_lock_wait(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    old_id = "doc-old"
    locator = "s3://bucket/item.pdf"
    source_uri = "bynder://asset/1"
    metadata = {old_id: {"download_locator": locator, "source_uri": source_uri}}
    deps["stores"].get_doc_status.return_value = None
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=(old_id,),
        replacement_ownership=((old_id, locator, source_uri),),
    )
    lock = engine._get_ingest_lock(old_id)
    await lock.acquire()
    task = asyncio.create_task(engine.aingest_files([item], replace=True))
    await asyncio.sleep(0)
    metadata[old_id] = {
        "download_locator": "s3://other/unrelated.pdf",
        "source_uri": "bynder://asset/other",
    }
    lock.release()

    with pytest.raises(RuntimeError, match="ownership changed"):
        await task
    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()
    deps["metadata_index"].delete.assert_not_awaited()


@pytest.mark.parametrize("boundary", ["vectors", "bm25"])
async def test_finalization_cancellation_keeps_upstream_processed_and_unpublished(
    tmp_path: Path, boundary: str
) -> None:
    source = tmp_path / f"{boundary}.pdf"
    source.write_bytes(b"content")
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    statuses: dict[str, dict[str, object]] = {}
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}

    async def process() -> None:
        statuses[doc_id] = {"status": "processed", "chunks_list": ["chunk"]}

    async def upsert_status(rows: dict[str, dict[str, object]]) -> None:
        statuses.update({key: dict(value) for key, value in rows.items()})

    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process
    deps["stores"].doc_status.upsert.side_effect = upsert_status
    if boundary == "vectors":
        engine._overwrite_sidecar_image_vectors = AsyncMock(  # type: ignore[method-assign]
            side_effect=asyncio.CancelledError
        )
    else:
        engine._label_bm25_languages = AsyncMock(  # type: ignore[method-assign]
            side_effect=asyncio.CancelledError
        )

    with pytest.raises(asyncio.CancelledError):
        await engine.aingest_files([source])

    assert statuses[doc_id]["status"] == "processed"
    assert "error_msg" not in statuses[doc_id]
    deps["stores"].doc_status.upsert.assert_not_awaited()
    assert (
        deps["metadata_index"].upsert.await_args_list[0].args[1][_FINALIZATION_COMPLETE_KEY]
        is False
    )


async def test_failed_old_to_new_replacement_restores_only_old_failed_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_id = "doc-old"
    locator = "s3://bucket/item.pdf"
    source_uri = "bynder://asset/1"
    statuses = {old_id: {"status": "processed", "chunks_list": ["old"]}}
    metadata: dict[str, dict[str, object]] = {
        old_id: {
            "filename": "old.pdf",
            "download_locator": locator,
            "source_uri": source_uri,
        }
    }
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def delete(current: str, **_kwargs: object) -> SimpleNamespace:
        statuses.pop(current, None)
        return SimpleNamespace(status="success")

    async def process() -> None:
        statuses[new_id] = {"status": "failed", "chunks_list": [], "error_msg": "parse"}

    async def upsert_status(rows: dict[str, dict[str, object]]) -> None:
        statuses.update({key: dict(value) for key, value in rows.items()})

    async def upsert_metadata(current: str, row: dict[str, object]) -> None:
        metadata[current] = dict(row)

    async def delete_metadata(current: str) -> None:
        metadata.pop(current, None)

    deps["lightrag"].adelete_by_doc_id.side_effect = delete
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process
    deps["stores"].doc_status.upsert.side_effect = upsert_status
    deps["metadata_index"].upsert.side_effect = upsert_metadata
    deps["metadata_index"].delete.side_effect = delete_metadata
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=(old_id,),
        replacement_ownership=((old_id, locator, source_uri),),
    )

    result = await engine.aingest_files([item], replace=True)

    assert result["processed"] == 0
    assert statuses == {}
    assert set(metadata) == {old_id}
    assert metadata[old_id]["_dlightrag_finalization_complete"] is False
    assert deps["lightrag"].adelete_by_doc_id.await_count == 2


@pytest.mark.parametrize("failure", [RuntimeError("enqueue failed"), asyncio.CancelledError()])
async def test_outer_enqueue_failure_settles_partial_candidate_and_old_anchor(
    tmp_path: Path, failure: BaseException
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_id = "doc-old"
    locator = "s3://bucket/item.pdf"
    source_uri = "bynder://asset/1"
    statuses = {old_id: {"status": "processed", "chunks_list": ["old"]}}
    metadata: dict[str, dict[str, object]] = {
        old_id: {"download_locator": locator, "source_uri": source_uri}
    }
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def delete(current: str, **_kwargs: object) -> SimpleNamespace:
        statuses.pop(current, None)
        return SimpleNamespace(status="success")

    async def enqueue(**_kwargs: object) -> None:
        statuses[new_id] = {"status": "pending", "chunks_list": []}
        raise failure

    async def upsert_status(rows: dict[str, dict[str, object]]) -> None:
        statuses.update({key: dict(value) for key, value in rows.items()})

    async def upsert_metadata(current: str, row: dict[str, object]) -> None:
        metadata[current] = dict(row)

    async def delete_metadata(current: str) -> None:
        metadata.pop(current, None)

    deps["lightrag"].adelete_by_doc_id.side_effect = delete
    deps["lightrag"].apipeline_enqueue_documents.side_effect = enqueue
    deps["stores"].doc_status.upsert.side_effect = upsert_status
    deps["metadata_index"].upsert.side_effect = upsert_metadata
    deps["metadata_index"].delete.side_effect = delete_metadata
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=(old_id,),
        replacement_ownership=((old_id, locator, source_uri),),
    )

    with pytest.raises(type(failure)):
        await engine.aingest_files([item], replace=True)

    assert statuses == {}
    assert set(metadata) == {old_id}
    assert metadata[old_id]["_dlightrag_finalization_complete"] is False


async def test_old_metadata_retirement_failure_does_not_publish_deleted_candidate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_id = "doc-old"
    locator = "s3://bucket/item.pdf"
    source_uri = "bynder://asset/1"
    statuses = {old_id: {"status": "processed", "chunks_list": ["old"]}}
    metadata: dict[str, dict[str, object]] = {
        old_id: {"download_locator": locator, "source_uri": source_uri}
    }
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def delete(current: str, **_kwargs: object) -> SimpleNamespace:
        statuses.pop(current, None)
        return SimpleNamespace(status="success")

    async def process() -> None:
        statuses[new_id] = {"status": "processed", "chunks_list": ["new"]}

    async def upsert_status(rows: dict[str, dict[str, object]]) -> None:
        statuses.update({key: dict(value) for key, value in rows.items()})

    async def upsert_metadata(current: str, row: dict[str, object]) -> None:
        metadata[current] = dict(row)

    async def delete_metadata(current: str) -> None:
        if current == old_id:
            raise RuntimeError("metadata retirement failed")
        metadata.pop(current, None)

    deps["lightrag"].adelete_by_doc_id.side_effect = delete
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process
    deps["stores"].doc_status.upsert.side_effect = upsert_status
    deps["metadata_index"].upsert.side_effect = upsert_metadata
    deps["metadata_index"].delete.side_effect = delete_metadata
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=(old_id,),
        replacement_ownership=((old_id, locator, source_uri),),
    )

    result = await engine.aingest_files([item], replace=True)

    assert result["processed"] == 0
    assert result["results"] == []
    assert result["errors"]
    assert statuses == {}
    assert set(metadata) == {old_id}
    assert metadata[old_id]["_dlightrag_finalization_complete"] is False


async def test_incomplete_finalization_marker_replays_without_reenqueue(
    tmp_path: Path,
) -> None:
    source = tmp_path / "report.pdf"
    content = b"content"
    source.write_bytes(content)
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    deps["stores"].get_doc_status.return_value = {
        "status": "processed",
        "chunks_list": ["chunk"],
        "content_hash": _sha256(content),
    }
    deps["metadata_index"].get.return_value = {
        "filename": "report.pdf",
        _FINALIZATION_COMPLETE_KEY: False,
    }
    engine._overwrite_sidecar_image_vectors = AsyncMock()  # type: ignore[method-assign]
    engine._label_bm25_languages = AsyncMock()  # type: ignore[method-assign]

    result = await engine.aingest_file(source)

    assert result["doc_id"] == doc_id
    assert result["source_kind"] == "document"
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()
    deps["lightrag"].apipeline_process_enqueue_documents.assert_not_awaited()
    engine._overwrite_sidecar_image_vectors.assert_awaited_once()  # type: ignore[attr-defined]
    engine._label_bm25_languages.assert_awaited_once_with(["chunk"])  # type: ignore[attr-defined]
    deps["stores"].doc_status.upsert.assert_not_awaited()
    _, completed = deps["metadata_index"].upsert.await_args.args
    assert completed[_FINALIZATION_COMPLETE_KEY] is True


@pytest.mark.parametrize("original", [RuntimeError("vectors failed"), asyncio.CancelledError()])
async def test_finalizer_failure_preserves_upstream_status_and_original_error(
    tmp_path: Path, original: BaseException
) -> None:

    source = tmp_path / "report.pdf"
    source.write_bytes(b"content")
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    statuses = {doc_id: {"status": "processed", "chunks_list": ["chunk"]}}
    deps["stores"].get_doc_status.side_effect = [None, statuses[doc_id]]
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = lambda: None
    engine._overwrite_sidecar_image_vectors = AsyncMock(  # type: ignore[method-assign]
        side_effect=original
    )
    deps["stores"].doc_status.upsert.side_effect = RuntimeError("status store down")

    expected = (
        asyncio.CancelledError if isinstance(original, asyncio.CancelledError) else RuntimeError
    )
    with pytest.raises(expected):
        await engine.aingest_file(source)

    assert statuses[doc_id]["status"] == "processed"
    first_metadata = deps["metadata_index"].upsert.await_args_list[0].args[1]
    assert first_metadata[_FINALIZATION_COMPLETE_KEY] is False
    deps["stores"].doc_status.upsert.assert_not_awaited()


async def test_metadata_only_old_tombstone_enqueue_failure_removes_candidate_intent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_id = "doc-old"
    locator = "s3://bucket/item.pdf"
    source_uri = "bynder://asset/1"
    metadata: dict[str, dict[str, object]] = {
        old_id: {"download_locator": locator, "source_uri": source_uri}
    }
    deps["stores"].get_doc_status.return_value = None
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def upsert_metadata(current: str, row: dict[str, object]) -> None:
        metadata[current] = dict(row)

    async def delete_metadata(current: str) -> None:
        metadata.pop(current, None)

    deps["metadata_index"].upsert.side_effect = upsert_metadata
    deps["metadata_index"].delete.side_effect = delete_metadata
    deps["lightrag"].apipeline_enqueue_documents.side_effect = RuntimeError("enqueue failed")
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=(old_id,),
        replacement_ownership=((old_id, locator, source_uri),),
    )

    with pytest.raises(RuntimeError, match="enqueue failed"):
        await engine.aingest_files([item], replace=True)

    assert set(metadata) == {old_id}
    assert new_id not in metadata


async def test_unrelated_preexisting_candidate_fails_before_replacement_cleanup(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_id = "doc-old"
    incoming_locator = "s3://bucket/incoming.pdf"
    incoming_source = "bynder://asset/incoming"
    candidate_metadata = {
        "download_locator": "s3://other/existing.pdf",
        "source_uri": "bynder://asset/existing",
        "filename": "existing.pdf",
    }
    metadata: dict[str, dict[str, object]] = {
        new_id: dict(candidate_metadata),
        old_id: {
            "download_locator": incoming_locator,
            "source_uri": incoming_source,
        },
    }
    statuses: dict[str, dict[str, object]] = {
        new_id: {"status": "processed", "chunks_list": ["old-chunk"]}
    }
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    item = PreparedIngestFile(
        source,
        incoming_source,
        incoming_locator,
        replacement_doc_ids=(old_id,),
        replacement_ownership=((old_id, incoming_locator, incoming_source),),
    )

    with pytest.raises(RuntimeError, match="candidate ownership changed"):
        await engine.aingest_files([item], replace=True)

    assert metadata[new_id] == candidate_metadata
    assert statuses[new_id]["status"] == "processed"
    assert metadata[old_id]["download_locator"] == incoming_locator
    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()
    deps["metadata_index"].upsert.assert_not_awaited()
    deps["metadata_index"].delete.assert_not_awaited()


async def test_multiple_metadata_tombstones_collapse_to_one_owner_on_enqueue_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_ids = ("doc-old-a", "doc-old-b")
    locator = "s3://bucket/incoming.pdf"
    source_uri = "bynder://asset/incoming"
    metadata: dict[str, dict[str, object]] = {
        old_id: {"download_locator": locator, "source_uri": source_uri} for old_id in old_ids
    }
    deps["stores"].get_doc_status.return_value = None
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def upsert_metadata(current: str, row: dict[str, object]) -> None:
        metadata[current] = dict(row)

    async def delete_metadata(current: str) -> None:
        metadata.pop(current, None)

    deps["metadata_index"].upsert.side_effect = upsert_metadata
    deps["metadata_index"].delete.side_effect = delete_metadata
    deps["lightrag"].apipeline_enqueue_documents.side_effect = RuntimeError("enqueue failed")
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=old_ids,
        replacement_ownership=tuple((old_id, locator, source_uri) for old_id in old_ids),
    )

    with pytest.raises(RuntimeError, match="enqueue failed"):
        await engine.aingest_files([item], replace=True)

    assert set(metadata) == {old_ids[0]}
    assert new_id not in metadata


async def test_replacement_completion_marker_commits_after_old_metadata_retirement(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_id = "doc-old"
    locator = "s3://bucket/incoming.pdf"
    source_uri = "bynder://asset/incoming"
    statuses: dict[str, dict[str, object]] = {
        old_id: {"status": "processed", "chunks_list": ["old"]}
    }
    metadata: dict[str, dict[str, object]] = {
        old_id: {"download_locator": locator, "source_uri": source_uri}
    }
    events: list[tuple[str, str, object]] = []
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def delete_doc(current: str, **_kwargs: object) -> SimpleNamespace:
        statuses.pop(current, None)
        return SimpleNamespace(status="success")

    async def process() -> None:
        statuses[new_id] = {"status": "processed", "chunks_list": ["new"]}

    async def upsert_metadata(current: str, row: dict[str, object]) -> None:
        metadata[current] = dict(row)
        events.append(("upsert", current, row.get(_FINALIZATION_COMPLETE_KEY)))

    async def delete_metadata(current: str) -> None:
        events.append(("delete", current, None))
        metadata.pop(current, None)

    deps["lightrag"].adelete_by_doc_id.side_effect = delete_doc
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process
    deps["metadata_index"].upsert.side_effect = upsert_metadata
    deps["metadata_index"].delete.side_effect = delete_metadata
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=(old_id,),
        replacement_ownership=((old_id, locator, source_uri),),
    )

    result = await engine.aingest_files([item], replace=True)

    assert result["processed"] == 1
    retirement = events.index(("delete", old_id, None))
    completion = events.index(("upsert", new_id, True))
    assert retirement < completion
    assert all(
        marker is not True
        for operation, current, marker in events[:completion]
        if operation == "upsert" and current == new_id
    )
    assert metadata[new_id][_FINALIZATION_COMPLETE_KEY] is True
    assert old_id not in metadata


async def test_recovered_incomplete_replacement_retires_remaining_owner_before_commit(
    tmp_path: Path,
) -> None:
    content = b"new"
    source = tmp_path / "new.pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_id = "doc-old"
    locator = "s3://bucket/incoming.pdf"
    source_uri = "bynder://asset/incoming"
    statuses: dict[str, dict[str, object]] = {
        new_id: {
            "status": "processed",
            "chunks_list": ["new"],
            "content_hash": _sha256(content),
        }
    }
    metadata: dict[str, dict[str, object]] = {
        new_id: {
            "download_locator": locator,
            "source_uri": source_uri,
            _FINALIZATION_COMPLETE_KEY: False,
        },
        old_id: {"download_locator": locator, "source_uri": source_uri},
    }
    events: list[tuple[str, str, object]] = []
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def upsert_metadata(current: str, row: dict[str, object]) -> None:
        metadata[current] = dict(row)
        events.append(("upsert", current, row.get(_FINALIZATION_COMPLETE_KEY)))

    async def delete_metadata(current: str) -> None:
        events.append(("delete", current, None))
        metadata.pop(current, None)

    deps["metadata_index"].upsert.side_effect = upsert_metadata
    deps["metadata_index"].delete.side_effect = delete_metadata
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=(old_id,),
        replacement_ownership=((old_id, locator, source_uri),),
    )

    result = await engine.aingest_files([item], replace=False)

    assert result["processed"] == 1
    assert events.index(("delete", old_id, None)) < events.index(("upsert", new_id, True))
    assert old_id not in metadata
    assert metadata[new_id][_FINALIZATION_COMPLETE_KEY] is True
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()
    deps["lightrag"].apipeline_process_enqueue_documents.assert_not_awaited()


async def test_replacement_marker_failure_after_retirement_leaves_retriable_candidate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_id = "doc-old"
    locator = "s3://bucket/incoming.pdf"
    source_uri = "bynder://asset/incoming"
    statuses: dict[str, dict[str, object]] = {
        old_id: {"status": "processed", "chunks_list": ["old"]}
    }
    metadata: dict[str, dict[str, object]] = {
        old_id: {"download_locator": locator, "source_uri": source_uri}
    }
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def delete_doc(current: str, **_kwargs: object) -> SimpleNamespace:
        statuses.pop(current, None)
        return SimpleNamespace(status="success")

    async def process() -> None:
        statuses[new_id] = {"status": "processed", "chunks_list": ["new"]}

    async def upsert_status(rows: dict[str, dict[str, object]]) -> None:
        statuses.update(rows)

    async def upsert_metadata(current: str, row: dict[str, object]) -> None:
        if current == new_id and row.get(_FINALIZATION_COMPLETE_KEY) is True:
            raise RuntimeError("completion marker unavailable")
        metadata[current] = dict(row)

    async def delete_metadata(current: str) -> None:
        metadata.pop(current, None)

    deps["lightrag"].adelete_by_doc_id.side_effect = delete_doc
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process
    deps["stores"].doc_status.upsert.side_effect = upsert_status
    deps["metadata_index"].upsert.side_effect = upsert_metadata
    deps["metadata_index"].delete.side_effect = delete_metadata
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=(old_id,),
        replacement_ownership=((old_id, locator, source_uri),),
    )

    result = await engine.aingest_files([item], replace=True)

    assert result["processed"] == 0
    assert result["errors"] == ["incoming.pdf: document processing failed"]
    assert old_id not in metadata
    assert metadata[new_id][_FINALIZATION_COMPLETE_KEY] is False
    assert statuses[new_id]["status"] == "processed"
    assert "error_msg" not in statuses[new_id]
    deps["stores"].doc_status.upsert.assert_not_awaited()


async def test_unrelated_candidate_with_metadata_tombstones_collapses_only_safe_surplus(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_ids = ("doc-old-a", "doc-old-b")
    locator = "s3://bucket/incoming.pdf"
    source_uri = "bynder://asset/incoming"
    candidate = {
        "download_locator": "s3://other/existing.pdf",
        "source_uri": "bynder://asset/existing",
        "filename": "existing.pdf",
    }
    metadata: dict[str, dict[str, object]] = {
        new_id: dict(candidate),
        **{old_id: {"download_locator": locator, "source_uri": source_uri} for old_id in old_ids},
    }
    statuses = {new_id: {"status": "processed", "chunks_list": ["existing"]}}
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def delete_metadata(current: str) -> None:
        metadata.pop(current, None)

    deps["metadata_index"].delete.side_effect = delete_metadata
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=old_ids,
        replacement_ownership=tuple((old_id, locator, source_uri) for old_id in old_ids),
    )

    with pytest.raises(RuntimeError, match="candidate ownership changed"):
        await engine.aingest_files([item], replace=True)

    assert metadata[new_id] == candidate
    assert statuses[new_id] == {"status": "processed", "chunks_list": ["existing"]}
    assert set(metadata) == {new_id, old_ids[0]}
    deps["metadata_index"].delete.assert_awaited_once_with(old_ids[1])
    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_unrelated_candidate_does_not_collapse_when_external_owner_has_status(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_ids = ("doc-old-a", "doc-old-b")
    locator = "s3://bucket/incoming.pdf"
    source_uri = "bynder://asset/incoming"
    candidate = {
        "download_locator": "s3://other/existing.pdf",
        "source_uri": "bynder://asset/existing",
    }
    metadata: dict[str, dict[str, object]] = {
        new_id: dict(candidate),
        **{old_id: {"download_locator": locator, "source_uri": source_uri} for old_id in old_ids},
    }
    statuses = {
        new_id: {"status": "processed", "chunks_list": ["existing"]},
        old_ids[1]: {"status": "failed", "chunks_list": []},
    }
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=old_ids,
        replacement_ownership=tuple((old_id, locator, source_uri) for old_id in old_ids),
    )

    with pytest.raises(RuntimeError, match="candidate ownership changed"):
        await engine.aingest_files([item], replace=True)

    assert metadata == {
        new_id: candidate,
        old_ids[0]: {"download_locator": locator, "source_uri": source_uri},
        old_ids[1]: {"download_locator": locator, "source_uri": source_uri},
    }
    deps["metadata_index"].delete.assert_not_awaited()
    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()
