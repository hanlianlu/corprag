import asyncio
import json
import logging

import pytest

from dlightrag.engine.answer.citations.contracts import ChunkSnippet, SourceReference
from dlightrag.engine.answer.citations.highlight import (
    HighlightExtractor,
    HighlightPhrases,
    extract_all_citing_sentences,
    extract_highlights_for_sources,
)


def _report_source(chunks: list[ChunkSnippet]) -> SourceReference:
    return SourceReference(
        id="1",
        title="report.pdf",
        source_uri="local://default/report.pdf",
        workspace="default",
        document_id="doc-report",
        download_locator="/docs/report.pdf",
        chunks=chunks,
    )


def test_extract_citing_sentences_basic():
    text = "Market grew fast. Growth was 15% [1-1]. Tech improved too [2-1]."
    result = extract_all_citing_sentences(text)
    assert "1-1" in result
    assert "2-1" in result
    assert any("15%" in s for s in result["1-1"])


def test_extract_citing_sentences_chinese_punctuation():
    text = "市场增长了15%[1-1]。技术也提升了[2-1]。"
    result = extract_all_citing_sentences(text)
    assert "1-1" in result
    assert "2-1" in result


def test_extract_citing_sentences_multiple_citations_one_sentence():
    text = "Both growth [1-1] and tech [2-1] improved."
    result = extract_all_citing_sentences(text)
    assert "1-1" in result
    assert "2-1" in result


def test_extract_citing_sentences_doc_level():
    text = "Revenue grew 15% [1]. Tech improved [2]."
    result = extract_all_citing_sentences(text)
    assert "1" in result
    assert "2" in result
    assert any("15%" in s for s in result["1"])


def test_extract_citing_sentences_doc_level_before_chinese_text():
    text = "翟东升观点见《货币、权力与人》[1]和《中国为什么有前途》[3]。"
    result = extract_all_citing_sentences(text)
    assert "1" in result
    assert "3" in result


def test_extract_citing_sentences_mixed_levels():
    text = "Revenue grew 15% [1-1]. Overall the report [1] is positive."
    result = extract_all_citing_sentences(text)
    assert "1-1" in result
    assert "1" in result


def test_extract_citing_sentences_attachment_doc_and_chunk_levels():
    text = "The attachment supports this [att-1]. Exact evidence is here [att-1-2]."

    assert extract_all_citing_sentences(text) == {
        "att-1": ["The attachment supports this [att-1]."],
        "att-1-2": ["Exact evidence is here [att-1-2]."],
    }


def test_extract_citing_sentences_supports_named_chunk_references():
    text = "Generic evidence [report_deadbeef-4]. Other evidence [project_alpha-2]."

    assert extract_all_citing_sentences(text) == {
        "report_deadbeef-4": ["Generic evidence [report_deadbeef-4]."],
        "project_alpha-2": ["Other evidence [project_alpha-2]."],
    }


def test_extract_citing_sentences_ignores_generated_references_section():
    text = "Revenue grew 15% [1-1].\n\n### References\n- [1] report.pdf"
    result = extract_all_citing_sentences(text)
    assert result == {"1-1": ["Revenue grew 15% [1-1]."]}


def test_extract_citing_sentences_empty():
    assert extract_all_citing_sentences("no citations here") == {}


def test_highlight_phrases_model():
    hp = HighlightPhrases(phrases=["market growth", "15%"], confidence=0.9)
    assert len(hp.phrases) == 2
    assert hp.confidence == 0.9


def test_highlight_phrases_defaults():
    hp = HighlightPhrases()
    assert hp.phrases == []
    assert hp.confidence == 1.0


class TestHighlightExtractor:
    @pytest.fixture()
    def mock_llm(self):
        async def llm_func(*, messages, **kwargs) -> str:
            return '{"items": [{"id": "0", "phrases": ["market growth"], "confidence": 0.9}]}'

        return llm_func

    @pytest.fixture()
    def extractor(self, mock_llm):
        return HighlightExtractor(llm_func=mock_llm, cache_size=10)

    @pytest.mark.asyncio
    async def test_batch_cache_hit(self, extractor):
        item = ("c1", "chunk content with data", "sentence")
        [(_, r1)] = await extractor.extract_highlight_batch([item])
        [(_, r2)] = await extractor.extract_highlight_batch([item])
        assert r1.phrases == r2.phrases

    @pytest.mark.asyncio
    async def test_doc_level_citation_triggers_highlights(self, mock_llm):
        """Doc-level [n] citations should trigger highlights for all chunks of that source."""
        from dlightrag.engine.answer.citations.contracts import ChunkSnippet
        from dlightrag.engine.answer.citations.highlight import extract_highlights_for_sources

        sources = [
            _report_source(
                [
                    ChunkSnippet(
                        chunk_id="c1",
                        chunk_idx=1,
                        content="Reports show market growth reached 15% in 2025.",
                    ),
                ]
            ),
        ]
        answer_text = "The market growth was impressive [1]."
        result = await extract_highlights_for_sources(sources, answer_text, mock_llm)
        # Doc-level [1] should match source id="1" and trigger highlights
        assert result[0].chunks is not None
        assert result[0].chunks[0].highlight_phrases is not None
        assert len(result[0].chunks[0].highlight_phrases) > 0

    @pytest.mark.asyncio
    async def test_extract_highlights_logs_summary(
        self,
        mock_llm,
        caplog: pytest.LogCaptureFixture,
    ):
        from dlightrag.engine.answer.citations.contracts import ChunkSnippet
        from dlightrag.engine.answer.citations.highlight import extract_highlights_for_sources

        sources = [
            _report_source(
                [
                    ChunkSnippet(
                        chunk_id="c1",
                        chunk_idx=1,
                        content="Reports show market growth reached 15% in 2025.",
                    ),
                ]
            ),
        ]
        answer_text = "The market growth was impressive [1]."

        with caplog.at_level(logging.INFO, logger="dlightrag.engine.answer.citations.highlight"):
            await extract_highlights_for_sources(sources, answer_text, mock_llm)

        assert "[Highlight] complete:" in caplog.text
        assert "sources=1" in caplog.text
        assert "chunks=1" in caplog.text
        assert "tasks=1" in caplog.text
        assert "task_errors=0" in caplog.text
        assert "highlighted_chunks=1" in caplog.text
        assert "phrases=1" in caplog.text

    @pytest.mark.asyncio
    async def test_invalid_phrases_filtered(self):
        async def bad_llm(*, messages, **kwargs) -> str:
            return (
                '{"items": [{"id": "0", "phrases": '
                '["hallucinated phrase", "actual text"], "confidence": 0.8}]}'
            )

        ext = HighlightExtractor(llm_func=bad_llm, cache_size=10)
        [(_, result)] = await ext.extract_highlight_batch(
            [("c1", "This has actual text in it.", "The actual text matters.")]
        )
        assert "actual text" in result.phrases
        assert "hallucinated phrase" not in result.phrases

    @pytest.mark.asyncio
    async def test_extract_highlights_for_sources_batches_items(self):
        """Highlight enrichment should batch citation/chunk pairs per LLM call."""
        batch_sizes: list[int] = []

        async def batch_llm(*, messages, **kwargs) -> str:
            user_prompt = messages[-1]["content"]
            items_json = user_prompt.split("Items:\n", 1)[1]
            items = json.loads(items_json)
            batch_sizes.append(len(items))
            return json.dumps(
                {
                    "items": [
                        {
                            "id": item["id"],
                            "phrases": [f"Evidence {item['id']}"],
                            "confidence": 0.9,
                        }
                        for item in items
                    ]
                }
            )

        chunks = [
            ChunkSnippet(
                chunk_id=f"c{i}",
                chunk_idx=i + 1,
                content=f"Evidence {i % 8} supports the answer for chunk {i}.",
            )
            for i in range(9)
        ]
        sources = [_report_source(chunks)]
        answer_text = " ".join(f"Chunk {i} is supported [1-{i}]." for i in range(1, 10))

        highlighted = await extract_highlights_for_sources(
            sources,
            answer_text,
            batch_llm,
            batch_size=8,
            max_concurrency=2,
        )

        assert batch_sizes == [8, 1]
        assert highlighted[0].chunks is not None
        assert highlighted[0].chunks[0].highlight_phrases == ["Evidence 0"]
        assert highlighted[0].chunks[8].highlight_phrases == ["Evidence 0"]

    @pytest.mark.asyncio
    async def test_extract_highlights_for_sources_runs_batches_concurrently(self):
        """max_concurrency should bound concurrent batch calls, not single-item calls."""
        active = 0
        peak_active = 0

        async def batch_llm(*, messages, **kwargs) -> str:
            nonlocal active, peak_active
            user_prompt = messages[-1]["content"]
            items = json.loads(user_prompt.split("Items:\n", 1)[1])
            active += 1
            peak_active = max(peak_active, active)
            try:
                await asyncio.sleep(0.01)
                return json.dumps(
                    {
                        "items": [
                            {
                                "id": item["id"],
                                "phrases": [f"Evidence {item['id']}"],
                                "confidence": 0.9,
                            }
                            for item in items
                        ]
                    }
                )
            finally:
                active -= 1

        chunks = [
            ChunkSnippet(
                chunk_id=f"c{i}",
                chunk_idx=i + 1,
                content=f"Evidence {i % 8} supports the answer for chunk {i}.",
            )
            for i in range(16)
        ]
        sources = [_report_source(chunks)]
        answer_text = " ".join(f"Chunk {i} is supported [1-{i}]." for i in range(1, 17))

        await extract_highlights_for_sources(
            sources,
            answer_text,
            batch_llm,
            batch_size=8,
            max_concurrency=2,
        )

        assert peak_active == 2

    @pytest.mark.asyncio
    async def test_doc_level_citation_fan_out_is_capped(self):
        """A doc-level [n] citation highlights only the first few chunks, not all."""
        seen_items: list[str] = []

        async def batch_llm(*, messages, **kwargs) -> str:
            user_prompt = messages[-1]["content"]
            items = json.loads(user_prompt.split("Items:\n", 1)[1])
            seen_items.extend(item["id"] for item in items)
            return json.dumps(
                {
                    "items": [
                        {"id": item["id"], "phrases": ["hit"], "confidence": 0.9} for item in items
                    ]
                }
            )

        chunks = [
            ChunkSnippet(chunk_id=f"c{i}", chunk_idx=i + 1, content=f"Content hit for chunk {i}.")
            for i in range(10)
        ]
        sources = [_report_source(chunks)]
        answer_text = "The report supports the answer [1]."

        highlighted = await extract_highlights_for_sources(
            sources,
            answer_text,
            batch_llm,
            batch_size=8,
            max_concurrency=2,
        )

        # Doc-level [1] fans out to only the first few chunks, not all ten.
        assert len(seen_items) == 3
        assert highlighted[0].chunks is not None
        highlighted_count = sum(1 for c in highlighted[0].chunks if c.highlight_phrases)
        assert highlighted_count == 3
