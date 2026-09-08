"""Tests for build_sources() shared utility."""

from typing import Any

import pytest

from dlightrag.engine.answer.citations.contracts import SourceReference
from dlightrag.engine.answer.citations.source_builder import (
    SourceBuildInvariantError,
    build_sources,
    build_sources_from_chunks,
)


def _chunk(
    chunk_id: str,
    reference_id: str,
    file_path: str = "/data/report.pdf",
    content: str = "text",
    image_data: str | None = None,
    page_number: int | None = None,
) -> dict[str, Any]:
    file_name = file_path.rsplit("/", 1)[-1]
    return {
        "chunk_id": chunk_id,
        "reference_id": reference_id,
        "full_doc_id": f"doc-{reference_id}",
        "file_path": file_path,
        "content": content,
        "image_data": image_data,
        "page_number": page_number,
        "metadata": {
            "file_name": file_name,
            "source_uri": f"local://default/{file_name}",
            "source_download_locator": file_path,
        },
        "_workspace": "default",
    }


def _chunks(source: SourceReference):
    assert source.chunks is not None
    return source.chunks


class TestBuildSources:
    def test_uses_dedicated_source_metadata_contract(self) -> None:
        chunk = _chunk("c1", "ref-1", file_path="parser-name.pdf")
        chunk["metadata"] = {
            "source_uri": "bynder://asset/1",
            "source_download_locator": "https://cdn.example.com/assets/1.pdf",
            "source_file_name": "report.pdf",
        }
        chunk["_workspace"] = "finance"

        source = build_sources({"chunks": [chunk]})[0]

        assert source.source_uri == "bynder://asset/1"
        assert source.workspace == "finance"
        assert source.document_id == "doc-ref-1"
        assert source.download_locator == "https://cdn.example.com/assets/1.pdf"

    def test_groups_by_reference_id(self) -> None:
        contexts = {
            "chunks": [
                _chunk("c1", "ref-1", page_number=1),
                _chunk("c2", "ref-1", page_number=2),
                _chunk("c3", "ref-2", file_path="/data/chart.xlsx"),
            ],
            "entities": [],
            "relationships": [],
        }
        sources = build_sources(contexts)

        assert len(sources) == 2
        assert sources[0].id == "ref-1"
        assert len(_chunks(sources[0])) == 2
        assert sources[1].id == "ref-2"
        assert len(_chunks(sources[1])) == 1

    def test_chunks_keep_citation_index_order(self) -> None:
        contexts = {
            "chunks": [
                _chunk("c2", "ref-1", page_number=3),
                _chunk("c1", "ref-1", page_number=1),
                _chunk("c3", "ref-1", page_number=2),
            ],
        }
        sources = build_sources(contexts)

        assert [c.chunk_id for c in _chunks(sources[0])] == ["c2", "c1", "c3"]
        assert [c.chunk_idx for c in _chunks(sources[0])] == [1, 2, 3]

    def test_source_title_from_filename(self) -> None:
        contexts = {"chunks": [_chunk("c1", "ref-1", file_path="/long/path/report.pdf")]}
        sources = build_sources(contexts)

        assert sources[0].title == "report.pdf"
        assert sources[0].source_uri == "local://default/report.pdf"

    def test_projects_image_urls_without_exposing_image_data(self) -> None:
        chunk = _chunk("c1", "ref-1", image_data="base64data", page_number=1)
        chunk["_workspace"] = "default"
        contexts = {"chunks": [chunk]}
        sources = build_sources(contexts)

        assert _chunks(sources[0])[0].image_url == "/images/default/c1?size=full"
        assert _chunks(sources[0])[0].thumbnail_url == "/images/default/c1?size=thumb"

    def test_projects_visual_chunk_urls_without_inline_image_data(self) -> None:
        chunk = _chunk("doc-1-mm-drawing-001", "ref-1", image_data=None, page_number=1)
        chunk["_workspace"] = "default"
        chunk["sidecar"] = {"type": "drawing"}
        contexts = {"chunks": [chunk]}
        sources = build_sources(contexts)

        assert _chunks(sources[0])[0].image_url == "/images/default/doc-1-mm-drawing-001?size=full"
        assert (
            _chunks(sources[0])[0].thumbnail_url
            == "/images/default/doc-1-mm-drawing-001?size=thumb"
        )

    def test_empty_contexts(self) -> None:
        assert build_sources({}) == []
        assert build_sources({"chunks": []}) == []

    def test_source_order_by_first_appearance(self) -> None:
        contexts = {
            "chunks": [
                _chunk("c1", "ref-2"),
                _chunk("c2", "ref-1"),
                _chunk("c3", "ref-2"),
            ],
        }
        sources = build_sources(contexts)

        assert sources[0].id == "ref-2"
        assert sources[1].id == "ref-1"

    def test_chunk_idx_assigned_sequentially(self) -> None:
        contexts = {
            "chunks": [
                _chunk("c1", "ref-1", page_number=1),
                _chunk("c2", "ref-1", page_number=2),
            ],
        }
        sources = build_sources(contexts)

        assert _chunks(sources[0])[0].chunk_idx == 1
        assert _chunks(sources[0])[1].chunk_idx == 2

    def test_missing_page_number_does_not_reorder_citation_index(self) -> None:
        contexts = {
            "chunks": [
                _chunk("c1", "ref-1", page_number=None),
                _chunk("c2", "ref-1", page_number=1),
            ],
        }
        sources = build_sources(contexts)

        assert [c.chunk_id for c in _chunks(sources[0])] == ["c1", "c2"]
        assert [c.chunk_idx for c in _chunks(sources[0])] == [1, 2]

    def test_page_number_falls_back_to_metadata(self) -> None:
        chunk = _chunk("c1", "ref-1", page_number=None)
        chunk["metadata"]["page_number"] = 7
        sources = build_sources({"chunks": [chunk]})

        assert _chunks(sources[0])[0].page_number == 7

    def test_cited_subset_preserves_source_catalog_fields(self) -> None:
        chunks = [
            _chunk("c1", "ref-1", file_path="/raw/report.pdf"),
            _chunk("c2", "ref-1", file_path="/raw/report.pdf"),
        ]
        catalog = [
            SourceReference(
                id="ref-1",
                title="Resolved Report",
                type="pdf",
                source_uri="bynder://asset/report",
                workspace="resolved",
                document_id="doc-resolved",
                download_locator="s3://bucket/resolved/report.pdf",
            )
        ]

        sources = build_sources_from_chunks(
            chunks,
            cited_chunks={"ref-1": ["c2"]},
            source_catalog=catalog,
        )

        assert len(sources) == 1
        assert sources[0].title == "Resolved Report"
        assert sources[0].type == "pdf"
        assert sources[0].source_uri == "bynder://asset/report"
        assert sources[0].workspace == "resolved"
        assert sources[0].document_id == "doc-resolved"
        assert sources[0].download_locator == "s3://bucket/resolved/report.pdf"
        assert sources[0].cited_chunk_ids == ["c2"]
        assert [c.chunk_id for c in _chunks(sources[0])] == ["c2"]

    def test_prefers_dedicated_source_metadata_for_identity_and_title(self) -> None:
        chunk = _chunk("c1", "ref-1", file_path="report__a1b2c3d4e5f6.pdf")
        chunk["metadata"] = {
            "source_uri": "bynder://asset/1",
            "source_download_locator": "s3://my-bucket/reports/report.pdf",
            "source_file_name": "report.pdf",
        }
        sources = build_sources({"chunks": [chunk]})

        assert sources[0].title == "report.pdf"
        assert sources[0].source_uri == "bynder://asset/1"
        assert sources[0].download_locator == "s3://my-bucket/reports/report.pdf"

    def test_rejects_missing_dedicated_source_metadata(self) -> None:
        chunk = _chunk("c1", "ref-1")
        chunk["metadata"].pop("source_download_locator")

        with pytest.raises(SourceBuildInvariantError, match="Source ref-1 is missing"):
            build_sources({"chunks": [chunk]})

    def test_invariant_error_uses_safe_source_id(self) -> None:
        chunk = _chunk("c1", "unsafe\nref")
        chunk["metadata"].pop("source_download_locator")

        with pytest.raises(SourceBuildInvariantError) as exc_info:
            build_sources({"chunks": [chunk]})

        assert "unsafe\\nref" in str(exc_info.value)
        assert "unsafe\nref" not in str(exc_info.value)

    def test_rejects_missing_workspace(self) -> None:
        chunk = _chunk("c1", "ref-1")
        chunk.pop("_workspace")

        with pytest.raises(SourceBuildInvariantError, match="Source ref-1 is missing workspace"):
            build_sources({"chunks": [chunk]})

    def test_skips_chunks_without_reference_id(self) -> None:
        contexts = {
            "chunks": [
                _chunk("c1", ""),
                _chunk("c2", "ref-1"),
            ],
        }
        sources = build_sources(contexts)

        assert len(sources) == 1
        assert sources[0].id == "ref-1"
