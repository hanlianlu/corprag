# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared semantic highlight enrichment for answer sources."""

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from dlightrag.engine.ai.completion import CompletionModel
from dlightrag.engine.ai.telemetry import Telemetry
from dlightrag.engine.answer.citations.contracts import HighlightSource
from dlightrag.engine.answer.citations.highlight import extract_highlights_for_sources

logger = logging.getLogger(__name__)

type SemanticHighlightModelFactory = Callable[[], tuple[CompletionModel, Telemetry]]


@dataclass(frozen=True, slots=True)
class SemanticHighlightSettings:
    enabled: bool
    timeout: float
    max_concurrency: int
    batch_size: int
    max_input_chars: int
    cache_size: int


async def enrich_semantic_highlights[SourceT: HighlightSource](
    sources: list[SourceT],
    *,
    answer_text: str | None,
    settings: SemanticHighlightSettings,
    model_factory: SemanticHighlightModelFactory,
) -> list[SourceT]:
    """Return sources with optional semantic highlight phrases."""
    text_chunk_count = _text_chunk_count(sources)
    if not settings.enabled or not answer_text or text_chunk_count == 0:
        return sources

    llm_model: CompletionModel | None = None
    try:
        llm_model, telemetry = model_factory()
        async with telemetry.observe(
            "semantic_highlights",
            as_type="chain",
            metadata={"source_count": len(sources), "text_chunk_count": text_chunk_count},
        ) as trace:
            highlighted = await asyncio.wait_for(
                extract_highlights_for_sources(
                    sources=sources,
                    answer_text=answer_text,
                    llm_func=llm_model,
                    max_concurrency=settings.max_concurrency,
                    batch_size=settings.batch_size,
                    max_input_chars=settings.max_input_chars,
                    cache_size=settings.cache_size,
                ),
                timeout=settings.timeout,
            )
            trace.update(output=_highlight_output(highlighted))
            return highlighted
    except TimeoutError:
        logger.warning(
            "Semantic highlight extraction timed out (%.1fs), skipping",
            settings.timeout,
        )
    except Exception:
        logger.warning("Semantic highlight extraction failed", exc_info=True)
    finally:
        if llm_model is not None:
            try:
                await llm_model.aclose()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Failed to close semantic highlight model", exc_info=True)
    return sources


def _text_chunk_count(sources: Sequence[HighlightSource]) -> int:
    return sum(1 for source in sources if source.chunks for chunk in source.chunks if chunk.content)


def _highlight_output(sources: Sequence[HighlightSource]) -> dict[str, int]:
    highlighted_source_count = 0
    highlighted_chunk_count = 0
    for source in sources:
        source_has_highlight = False
        for chunk in source.chunks or []:
            if chunk.highlight_phrases:
                source_has_highlight = True
                highlighted_chunk_count += 1
        if source_has_highlight:
            highlighted_source_count += 1
    return {
        "highlighted_source_count": highlighted_source_count,
        "highlighted_chunk_count": highlighted_chunk_count,
    }


__all__ = ["SemanticHighlightSettings", "enrich_semantic_highlights"]
