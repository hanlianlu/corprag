import dlightrag.engine.answer.citations.contracts as schemas
from dlightrag.engine.answer.citations.contracts import ChunkSnippet, SourceReference


def test_source_reference_and_public_payload_have_distinct_contracts() -> None:
    assert {"source_uri", "workspace", "document_id", "download_locator"} <= set(
        SourceReference.model_fields
    )
    assert {"path", "url", "download_url"}.isdisjoint(SourceReference.model_fields)

    payload_type = getattr(schemas, "SourceReferencePayload", None)
    assert payload_type is not None
    assert {"source_uri", "download_url"} <= set(payload_type.model_fields)
    assert {"workspace", "document_id", "download_locator", "path", "url"}.isdisjoint(
        payload_type.model_fields
    )


def test_chunk_snippet_minimal():
    cs = ChunkSnippet(chunk_id="abc123", content="some text")
    assert cs.chunk_id == "abc123"
    assert cs.chunk_idx is None
    assert cs.page_number is None
    assert cs.highlight_phrases is None


def test_chunk_snippet_full():
    cs = ChunkSnippet(
        chunk_id="abc123",
        chunk_idx=2,
        page_number=3,
        content="market growth reached 15%",
        image_url="/images/default/abc123?size=full",
        thumbnail_url="/images/default/abc123?size=thumb",
        highlight_phrases=["15%"],
    )
    assert cs.chunk_idx == 2
    assert cs.page_number == 3
    assert cs.image_url == "/images/default/abc123?size=full"
    assert cs.thumbnail_url == "/images/default/abc123?size=thumb"
    assert cs.highlight_phrases == ["15%"]


def test_source_reference_minimal():
    sr = SourceReference(
        id="1",
        source_uri="local://default/report.pdf",
        workspace="default",
        document_id="doc-report",
        download_locator="/docs/report.pdf",
    )
    assert sr.id == "1"
    assert sr.title is None
    assert sr.chunks is None


def test_source_reference_with_chunks():
    chunk = ChunkSnippet(chunk_id="c1", chunk_idx=1, content="text")
    sr = SourceReference(
        id="1",
        title="Report",
        type="pdf",
        source_uri="local://default/report.pdf",
        workspace="default",
        document_id="doc-report",
        download_locator="/docs/report.pdf",
        chunks=[chunk],
        cited_chunk_ids=["c1"],
    )
    assert sr.chunks is not None
    assert len(sr.chunks) == 1
    assert sr.cited_chunk_ids == ["c1"]
