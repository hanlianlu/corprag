"""Semantic highlight extraction — uses LLM to find relevant phrases in chunks.

Ported from sandbox_agent. LangChain replaced with raw async LLM function call.
"""

import hashlib
import json
import logging
import re
import time
from collections import OrderedDict, defaultdict
from collections.abc import Awaitable, Callable, Sequence

from pydantic import BaseModel, Field

from dlightrag.engine.ai.concurrency import bounded_map
from dlightrag.engine.answer.citations.contracts import HighlightSource
from dlightrag.engine.answer.prompts import (
    HIGHLIGHT_BATCH_USER_PROMPT,
    HIGHLIGHT_SYSTEM_PROMPT,
)

from .parser import CITATION_PATTERN, DOC_CITATION_PATTERN, strip_generated_references_section

logger = logging.getLogger(__name__)

_MAX_INPUT_CHARS = 4096
_MAX_CONCURRENT_HIGHLIGHTS = 8
_DEFAULT_HIGHLIGHT_BATCH_SIZE = 8
# Doc-level [n] citations otherwise fan out across every chunk of the document,
# so a single [n] on a large doc becomes chunks x citing-sentences LLM items.
# Cap doc-level sentences to the first few (highest-ranked) chunks per doc.
_MAX_DOC_LEVEL_HIGHLIGHT_CHUNKS = 3


class HighlightPhrases(BaseModel):
    phrases: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class HighlightBatchItem(BaseModel):
    id: str
    phrases: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class HighlightBatchResponse(BaseModel):
    items: list[HighlightBatchItem] = Field(default_factory=list)


_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?\u3002\uff01\uff1f])\s+|(?<=[.!?\u3002\uff01\uff1f])(?=\S)"
)


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text


def _chunks[T](items: Sequence[T], size: int) -> list[list[T]]:
    size = max(1, size)
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def extract_all_citing_sentences(answer_text: str) -> dict[str, list[str]]:
    """Extract all sentences that contain each citation."""
    answer_text = strip_generated_references_section(answer_text)
    sentences = _SENTENCE_SPLIT_RE.split(answer_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    result: dict[str, list[str]] = {}
    for sentence in sentences:
        positions = [
            *((match.start(), match.group(1)) for match in DOC_CITATION_PATTERN.finditer(sentence)),
            *(
                (match.start(), f"{match.group(1)}-{match.group(2)}")
                for match in CITATION_PATTERN.finditer(sentence)
            ),
        ]
        for _, key in sorted(positions):
            result.setdefault(key, [])
            if sentence not in result[key]:
                result[key].append(sentence)

    return result


class HighlightExtractor:
    """Extract semantic highlight phrases using LLM with LRU cache."""

    def __init__(
        self,
        llm_func: Callable[..., Awaitable[str]],
        cache_size: int = 500,
        max_input_chars: int = _MAX_INPUT_CHARS,
    ) -> None:
        self._llm_func = llm_func
        self._cache: OrderedDict[str, HighlightPhrases] = OrderedDict()
        self._cache_size = cache_size
        self._max_input_chars = max_input_chars

    def _cache_key(self, citing_sentence: str, chunk_id: str) -> str:
        raw = f"{citing_sentence}||{chunk_id}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_get(self, key: str) -> HighlightPhrases | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: str, value: HighlightPhrases) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._cache_size:
                self._cache.popitem(last=False)
        self._cache[key] = value

    def _validate_phrases(
        self,
        result: HighlightPhrases,
        chunk_content: str,
    ) -> HighlightPhrases:
        chunk_lower = chunk_content.lower()
        validated = []
        for phrase in result.phrases:
            words = phrase.split()
            if len(words) < 1 or len(words) > 25:
                continue
            if phrase in chunk_content:
                validated.append(phrase)
            elif phrase.lower() in chunk_lower:
                idx = chunk_lower.index(phrase.lower())
                validated.append(chunk_content[idx : idx + len(phrase)])
        result.phrases = validated
        return result

    async def extract_highlight_batch(
        self,
        items: Sequence[tuple[str, str, str]],
    ) -> list[tuple[str, HighlightPhrases]]:
        """Extract highlights for a batch of ``(chunk_id, content, sentence)`` items."""
        results: list[HighlightPhrases | None] = [None] * len(items)
        pending: list[tuple[int, str, str, str, str, str]] = []

        for idx, (chunk_id, chunk_content, citing_sentence) in enumerate(items):
            cache_key = self._cache_key(citing_sentence, chunk_id)
            cached = self._cache_get(cache_key)
            if cached is not None:
                results[idx] = cached
                continue
            item_id = str(len(pending))
            pending.append((idx, item_id, cache_key, chunk_id, chunk_content, citing_sentence))

        if pending:
            payload = [
                {
                    "id": item_id,
                    "citing_sentence": citing_sentence[: self._max_input_chars],
                    "source_chunk": chunk_content[: self._max_input_chars],
                }
                for _, item_id, _, _, chunk_content, citing_sentence in pending
            ]
            user_prompt = HIGHLIGHT_BATCH_USER_PROMPT.format(
                items_json=json.dumps(payload, ensure_ascii=False)
            )
            messages = [
                {"role": "system", "content": HIGHLIGHT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            parsed_items: dict[str, HighlightPhrases] = {}
            try:
                raw_response = await self._llm_func(messages=messages)
                parsed = json.loads(_strip_json_fences(raw_response))
                if isinstance(parsed, dict) and "items" in parsed:
                    batch = HighlightBatchResponse(**parsed)
                    parsed_items = {
                        item.id: HighlightPhrases(
                            phrases=item.phrases,
                            confidence=item.confidence,
                        )
                        for item in batch.items
                    }
            except Exception:
                logger.debug("Highlight batch extraction failed", exc_info=True)

            for idx, item_id, cache_key, _, chunk_content, _ in pending:
                result = parsed_items.get(item_id, HighlightPhrases())
                result = self._validate_phrases(result, chunk_content)
                self._cache_put(cache_key, result)
                results[idx] = result

        return [
            (chunk_id, result if result is not None else HighlightPhrases())
            for result, (chunk_id, _, _) in zip(results, items, strict=True)
        ]


async def extract_highlights_for_sources[SourceT: HighlightSource](
    sources: list[SourceT],
    answer_text: str,
    llm_func: Callable[..., Awaitable[str]],
    *,
    max_concurrency: int = _MAX_CONCURRENT_HIGHLIGHTS,
    batch_size: int = _DEFAULT_HIGHLIGHT_BATCH_SIZE,
    max_input_chars: int = _MAX_INPUT_CHARS,
    cache_size: int = 500,
) -> list[SourceT]:
    """Extract highlights for all sources in parallel."""
    started = time.monotonic()
    extractor = HighlightExtractor(
        llm_func=llm_func,
        cache_size=cache_size,
        max_input_chars=max_input_chars,
    )
    citing_sentences = extract_all_citing_sentences(answer_text)

    items: list[tuple[str, str, str]] = []
    chunk_count = 0
    text_chunk_count = 0
    for src in sources:
        if not src.chunks:
            continue
        chunk_count += len(src.chunks)
        doc_level_sentences = citing_sentences.get(src.id, [])
        doc_level_chunks_used = 0
        for chunk in src.chunks:
            if not chunk.content:
                continue
            text_chunk_count += 1
            # Chunk-level [n-m] citations always highlight their own chunk.
            chunk_key = f"{src.id}-{chunk.chunk_idx}"
            sentences = list(citing_sentences.get(chunk_key, []))
            # Doc-level [n] citations fan out only to the first few chunks of the
            # doc, bounding total items at O(chunk-level + N x doc-sentences).
            if doc_level_sentences and doc_level_chunks_used < _MAX_DOC_LEVEL_HIGHLIGHT_CHUNKS:
                added_doc_level = False
                for s in doc_level_sentences:
                    if s not in sentences:
                        sentences.append(s)
                        added_doc_level = True
                if added_doc_level:
                    doc_level_chunks_used += 1
            for sentence in sentences:
                items.append((chunk.chunk_id, chunk.content, sentence))

    if not items:
        logger.info(
            "[Highlight] complete: sources=%d chunks=%d text_chunks=%d citing_keys=%d "
            "tasks=0 task_errors=0 batches=0 batch_errors=0 "
            "highlighted_chunks=0 phrases=0 duration=%.2fs",
            len(sources),
            chunk_count,
            text_chunk_count,
            len(citing_sentences),
            time.monotonic() - started,
        )
        return sources

    batches = _chunks(items, batch_size)

    async def _extract_batch(
        batch: list[tuple[str, str, str]],
    ) -> list[tuple[str, HighlightPhrases]]:
        return await extractor.extract_highlight_batch(batch)

    batch_results = await bounded_map(
        batches,
        _extract_batch,
        max_concurrent=max_concurrency,
        task_name="highlight-batch",
    )

    chunk_phrases: defaultdict[str, set[str]] = defaultdict(set)
    task_errors = 0
    batch_errors = 0
    for batch, res in zip(batches, batch_results, strict=True):
        if isinstance(res, BaseException):
            batch_errors += 1
            task_errors += len(batch)
            logger.debug("Highlight batch failed: %s", res)
            continue
        for chunk_id, hp in res:
            if hp.phrases:
                chunk_phrases[chunk_id].update(hp.phrases)

    highlighted_chunks = 0
    phrase_count = 0
    for src in sources:
        if not src.chunks:
            continue
        for chunk in src.chunks:
            phrases = chunk_phrases.get(chunk.chunk_id)
            if phrases:
                phrase_list = sorted(phrases)
                chunk.highlight_phrases = phrase_list
                highlighted_chunks += 1
                phrase_count += len(phrase_list)

    logger.info(
        "[Highlight] complete: sources=%d chunks=%d text_chunks=%d citing_keys=%d "
        "tasks=%d task_errors=%d batches=%d batch_errors=%d "
        "highlighted_chunks=%d phrases=%d duration=%.2fs",
        len(sources),
        chunk_count,
        text_chunk_count,
        len(citing_sentences),
        len(items),
        task_errors,
        len(batches),
        batch_errors,
        highlighted_chunks,
        phrase_count,
        time.monotonic() - started,
    )

    return sources
