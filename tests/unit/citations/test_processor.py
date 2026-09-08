# tests/unit/citations/test_processor.py
from dlightrag.engine.answer.citations.contracts import SourceReference
from dlightrag.engine.answer.citations.processor import CitationProcessor, CitationResult


def _metadata(file_name: str, file_path: str, **extra):
    return {
        "file_name": file_name,
        "source_uri": f"local://default/{file_name}",
        "source_download_locator": file_path,
        **extra,
    }


def _source(
    source_id: str,
    file_path: str,
    *,
    title: str,
    source_type: str | None = None,
) -> SourceReference:
    return SourceReference(
        id=source_id,
        title=title,
        type=source_type,
        source_uri=f"local://default/{file_path.rsplit('/', 1)[-1]}",
        workspace="default",
        document_id=f"doc-{source_id}",
        download_locator=file_path,
    )


def _make_contexts():
    """Contexts simulating DlightRAG RetrievalResult.contexts flattened."""
    return [
        {
            "chunk_id": "c1",
            "reference_id": "1",
            "full_doc_id": "doc-1",
            "file_path": "/docs/report.pdf",
            "content": "Market growth reached 15% this year.",
            "_workspace": "default",
            "metadata": _metadata("report.pdf", "/docs/report.pdf", page_number=3),
        },
        {
            "chunk_id": "c2",
            "reference_id": "1",
            "full_doc_id": "doc-1",
            "file_path": "/docs/report.pdf",
            "content": "Revenue exceeded expectations.",
            "_workspace": "default",
            "metadata": _metadata("report.pdf", "/docs/report.pdf", page_number=5),
        },
        {
            "chunk_id": "c3",
            "reference_id": "2",
            "full_doc_id": "doc-2",
            "file_path": "/docs/analysis.xlsx",
            "content": "Technical indicators show positive trend.",
            "_workspace": "default",
            "metadata": _metadata("analysis.xlsx", "/docs/analysis.xlsx", page_number=1),
        },
    ]


def _make_sources():
    return [
        _source("1", "/docs/report.pdf", title="Report", source_type="pdf"),
        _source("2", "/docs/analysis.xlsx", title="Analysis", source_type="xlsx"),
    ]


def test_process_basic():
    contexts = _make_contexts()
    sources = _make_sources()
    proc = CitationProcessor(contexts=contexts, available_sources=sources)
    result = proc.process("Growth was strong [1-1] and trends are positive [2-1].")

    assert isinstance(result, CitationResult)
    assert "[1-1]" in result.answer
    assert "[2-1]" in result.answer
    assert len(result.sources) == 2
    src1 = next(s for s in result.sources if s.id == "1")
    assert src1.chunks is not None
    assert src1.chunks[0].chunk_id == "c1"


def test_process_removes_invalid_citations():
    contexts = _make_contexts()
    sources = _make_sources()
    proc = CitationProcessor(contexts=contexts, available_sources=sources)
    result = proc.process("Valid [1-1] and hallucinated [3-1] and [1-99].")

    assert "[1-1]" in result.answer
    assert "[3-1]" not in result.answer
    assert "[1-99]" not in result.answer


def test_process_empty_answer():
    proc = CitationProcessor(contexts=[], available_sources=[])
    result = proc.process("")
    assert result.answer == ""
    assert result.sources == []


def test_cited_chunks_populated():
    contexts = _make_contexts()
    sources = _make_sources()
    proc = CitationProcessor(contexts=contexts, available_sources=sources)
    result = proc.process("See [1-1].")
    assert "1" in result.cited_chunks
    assert "c1" in result.cited_chunks["1"]


def test_doc_level_and_chunk_level_sources_follow_inline_order():
    contexts = _make_contexts()
    sources = _make_sources()
    proc = CitationProcessor(contexts=contexts, available_sources=sources)
    result = proc.process("Overall trend is positive [2]. Growth was strong [1-1].")

    assert [s.id for s in result.sources] == ["2", "1"]


def test_image_only_chunk_is_citable():
    contexts = [
        {
            "chunk_id": "img1",
            "reference_id": "1",
            "content": "",
            "image_data": "base64-page-image",
            "file_path": "/data/chart.pdf",
            "page_number": 4,
            "_workspace": "default",
            "metadata": _metadata("chart.pdf", "/data/chart.pdf"),
        }
    ]
    sources = [_source("1", "/data/chart.pdf", title="chart.pdf")]

    proc = CitationProcessor(contexts=contexts, available_sources=sources)
    result = proc.process("The chart shows the trend [1-1].")

    assert result.answer == "The chart shows the trend [1-1]."
    assert result.sources[0].chunks is not None
    assert result.sources[0].chunks[0].image_url == "/images/default/img1?size=full"


def test_source_catalog_metadata_is_preserved():
    contexts = [
        {
            "chunk_id": "c1",
            "reference_id": "1",
            "content": "Evidence.",
            "file_path": "/raw/doc.pdf",
            "_workspace": "raw",
            "metadata": _metadata("raw.pdf", "/raw/doc.pdf", file_type="raw"),
        }
    ]
    sources = [
        SourceReference(
            id="1",
            title="Catalog title",
            type="pdf",
            source_uri="bynder://asset/1",
            workspace="resolved",
            document_id="doc-catalog",
            download_locator="s3://bucket/doc.pdf",
        )
    ]

    proc = CitationProcessor(contexts=contexts, available_sources=sources)
    result = proc.process("Evidence [1-1].")

    assert result.sources[0].title == "Catalog title"
    assert result.sources[0].type == "pdf"
    assert result.sources[0].source_uri == "bynder://asset/1"
    assert result.sources[0].workspace == "resolved"
    assert result.sources[0].download_locator == "s3://bucket/doc.pdf"


class TestCitationProcessorFlatPageNumber:
    """Test that page_number is read from flat context metadata."""

    def test_page_number_from_flat_field(self):
        contexts = [
            {
                "chunk_id": "c1",
                "reference_id": "1",
                "content": "Page one content",
                "file_path": "/data/doc.pdf",
                "page_number": 1,
                "_workspace": "default",
                "metadata": _metadata("doc.pdf", "/data/doc.pdf"),
            },
            {
                "chunk_id": "c2",
                "reference_id": "1",
                "content": "Page two content",
                "file_path": "/data/doc.pdf",
                "page_number": 2,
                "_workspace": "default",
                "metadata": _metadata("doc.pdf", "/data/doc.pdf"),
            },
        ]
        sources = [
            _source("1", "/data/doc.pdf", title="doc.pdf"),
        ]

        processor = CitationProcessor(contexts=contexts, available_sources=sources)
        result = processor.process("According to [1], the data shows...")

        assert len(result.sources) == 1
        src = result.sources[0]
        assert src.chunks is not None
        assert len(src.chunks) == 2
        assert src.chunks[0].page_number == 1
        assert src.chunks[1].page_number == 2
