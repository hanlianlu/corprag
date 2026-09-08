# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Final answer synthesis.

Receives merged retrieval/evidence contexts from any path and generates the
single final answer with proper citations. Lives in the Answer domain
level -- shared across all workspaces.

The synthesizer accepts a single ``model_func`` callable that follows the
messages-first interface: it receives ``messages=`` (OpenAI-format list) and an
optional ``stream=`` keyword argument.  Images are inlined as ``image_url``
content blocks so there is no separate VLM path -- the provider decides how to
handle multimodal content.

Both streaming and non-streaming paths use the same freetext system prompt and
the same evidence preparation. Sources are projected from validated inline
citation markers. Input packing is bounded by the answer model's resolved
profile and the current immutable context policy.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, cast

from dlightrag.engine.agent.context import ContextContribution, ContextProjector
from dlightrag.engine.agent.session.fold import PriorTurns
from dlightrag.engine.ai.capacity import CONTEXT_POLICY, ContextPolicy, ModelProfile
from dlightrag.engine.ai.tokens import estimate_content_tokens, estimate_messages_tokens
from dlightrag.engine.answer.citations.indexer import CitationIndexer
from dlightrag.engine.answer.citations.streaming import AnswerStream, aclose_answer_stream
from dlightrag.engine.answer.errors import (
    AnswerInputOverflowError,
    CurrentImagePayloadError,
)
from dlightrag.engine.answer.excerpts import build_excerpt_lane_blocks, format_kg_context
from dlightrag.engine.answer.images import AnswerImageBudget, AnswerImagePolicy
from dlightrag.engine.answer.memory import standing_memory_message
from dlightrag.engine.answer.prompts import answer_core
from dlightrag.engine.answer.synthesis_context import AnswerContextPacker
from dlightrag.engine.rag.retrieval import RetrievalContexts

logger = logging.getLogger(__name__)

NO_CONTEXT_DISCLAIMER = (
    "**General Knowledge Notice:** The answer below is NOT grounded in your knowledge base."
)


@dataclass
class _PreparedAnswerPrompt:
    contexts: RetrievalContexts
    user_prompt: str
    kg_context: str
    indexer: CitationIndexer
    chunk_image_blocks: dict[str, dict[str, Any]]
    trace: dict[str, Any]


@dataclass
class _PreparedModelCall:
    contexts: RetrievalContexts
    messages: list[dict[str, Any]]
    indexer: CitationIndexer
    trace: dict[str, Any]
    no_context: bool
    max_output_tokens: int | None


class AnswerSynthesizer:
    """Mode-agnostic final answer generator with citation support.

    Accepts a single ``model_func`` that speaks the messages-first interface.
    Current-request and chunk images are inlined as ``image_url`` content blocks
    under one budget -- no separate VLM routing is needed.

    ``generate_stream()`` uses the unified freetext system prompt and identical
    evidence preparation. Sources are projected from validated inline ``[n]``
    and ``[n-m]`` markers.
    """

    def __init__(
        self,
        *,
        image_policy: AnswerImagePolicy,
        model_profile: ModelProfile,
        context_policy: ContextPolicy = CONTEXT_POLICY,
        model_func: Callable[..., Any] | None = None,
    ) -> None:
        self.model_func = model_func
        self._image_policy = image_policy
        self._model_profile = model_profile
        self._context_policy = context_policy

    def history_input_measure(
        self,
        query: str,
        memory_text: str = "",
        episodic_summary: str = "",
        current_images: list[dict[str, Any]] | None = None,
    ) -> Callable[..., int]:
        """Return the exact zero-evidence final-call serializer for history fitting."""

        def measure(
            history: list[dict[str, Any]],
            projected_summary: str = "",
        ) -> int:
            budget = self._image_policy.new_budget()
            current_image_blocks = self._prepare_current_image_blocks(
                current_images,
                image_budget=budget,
            )
            empty_contexts: RetrievalContexts = {
                "chunks": [],
                "entities": [],
                "relationships": [],
            }
            prepared = self._prepare_prompt_context(
                query,
                empty_contexts,
                image_budget=budget,
            )
            excerpt_blocks = self._build_excerpt_blocks(
                prepared.contexts,
                prepared.indexer,
                image_blocks_by_context_key=prepared.chunk_image_blocks,
            )
            messages = self._compose_user_messages(
                answer_core(),
                prepared.user_prompt,
                excerpt_blocks,
                current_image_blocks=current_image_blocks,
                history_messages=history,
                episodic_summary="\n\n".join(
                    part for part in (episodic_summary, projected_summary) if part.strip()
                ),
                memory_text=memory_text,
            )
            return estimate_messages_tokens(messages)

        return measure

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_stream(
        self,
        query: str,
        contexts: RetrievalContexts,
        conversation_history: PriorTurns | None = None,
        memory_text: str = "",
        current_images: list[dict[str, Any]] | None = None,
    ) -> tuple[RetrievalContexts, AsyncIterator[str] | None]:
        """Streaming final answer generation.

        Uses the same freetext prompt and identical evidence preparation as
        ``generate()``. Wraps the token stream with :class:`AnswerStream` for
        post-stream citation index validation.
        """
        if self.model_func is None:
            logger.info("[AS] generate_stream: no model_func, returning None")
            return contexts, None

        prepared = await asyncio.to_thread(
            self._prepare_model_call,
            query,
            contexts,
            conversation_history=conversation_history,
            memory_text=memory_text,
            current_images=current_images,
        )

        logger.info(
            "[AS] generate_stream: input_chunks=%d packed_chunks=%d images_sent=%d "
            "images_skipped=%d query=%s",
            len(contexts.get("chunks", [])),
            prepared.trace["answer_context_chunks"],
            prepared.trace["answer_context_images_sent"],
            prepared.trace["answer_context_images_skipped"],
            query[:60],
        )

        usage: dict[str, Any] = {}
        prepared.trace["usage"] = usage
        call_kwargs: dict[str, Any] = {
            "messages": prepared.messages,
            "stream": True,
            "usage_holder": usage,
            "model_profile": self._model_profile,
        }
        if prepared.max_output_tokens is not None:
            call_kwargs["max_tokens"] = prepared.max_output_tokens
        token_iterator = await self.model_func(**call_kwargs)
        if prepared.no_context:
            token_iterator = _prepend_no_context_stream(token_iterator)

        if hasattr(token_iterator, "__aiter__"):
            token_iterator = AnswerStream(token_iterator, indexer=prepared.indexer)
            cast(Any, token_iterator).trace = prepared.trace

        return prepared.contexts, token_iterator

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_model_call(
        self,
        query: str,
        contexts: RetrievalContexts,
        *,
        conversation_history: PriorTurns | None = None,
        memory_text: str = "",
        current_images: list[dict[str, Any]] | None = None,
    ) -> _PreparedModelCall:
        prior_turns = conversation_history or PriorTurns()
        original_history = list(prior_turns.messages)

        def build(
            history: list[dict[str, Any]],
            candidate_contexts: RetrievalContexts,
        ) -> tuple[_PreparedModelCall, int, int]:
            budget = self._image_policy.new_budget()
            current_image_blocks = self._prepare_current_image_blocks(
                current_images,
                image_budget=budget,
            )
            prepared = self._prepare_prompt_context(
                query,
                candidate_contexts,
                image_budget=budget,
            )
            no_context = not any(
                prepared.contexts.get(key) for key in ("chunks", "entities", "relationships")
            )
            if no_context:
                prepared.trace["answer_no_context"] = True
            self._apply_image_trace(
                prepared.trace,
                budget=budget,
                current_image_count=len(current_images or ()),
            )
            excerpt_blocks = self._build_excerpt_blocks(
                prepared.contexts,
                prepared.indexer,
                image_blocks_by_context_key=prepared.chunk_image_blocks,
            )
            messages = self._compose_user_messages(
                answer_core(),
                prepared.user_prompt,
                excerpt_blocks,
                current_image_blocks=current_image_blocks,
                history_messages=history,
                episodic_summary=prior_turns.episodic_summary,
                memory_text=memory_text,
            )
            evidence_tokens = estimate_content_tokens(excerpt_blocks) + estimate_content_tokens(
                prepared.kg_context
            )
            total_tokens = estimate_messages_tokens(messages)
            call = _PreparedModelCall(
                contexts=prepared.contexts,
                messages=messages,
                indexer=prepared.indexer,
                trace=prepared.trace,
                no_context=no_context,
                max_output_tokens=None,
            )
            return call, evidence_tokens, total_tokens

        input_limit = self._context_policy.hard_input_limit(self._model_profile)
        retrieved_chunk_count = len(contexts.get("chunks", []))
        result, evidence_tokens, total_tokens = build(original_history, contexts)
        capacity_chunks = [dict(chunk) for chunk in result.contexts.get("chunks", [])]
        base_contexts: RetrievalContexts = {
            key: [dict(item) for item in value] for key, value in result.contexts.items()
        }
        capacity_dropped_chunk_count = 0
        while total_tokens > input_limit and capacity_chunks:
            capacity_chunks.pop()
            capacity_dropped_chunk_count += 1
            candidate_contexts: RetrievalContexts = {
                key: [dict(item) for item in value] for key, value in base_contexts.items()
            }
            candidate_contexts["chunks"] = [dict(chunk) for chunk in capacity_chunks]
            result, evidence_tokens, total_tokens = build(
                original_history,
                candidate_contexts,
            )
        if total_tokens > input_limit:
            raise AnswerInputOverflowError(
                "Answer input exceeds the resolved model input limit after all whole "
                "capacity-adjustable chunks were removed; the retained non-chunk context "
                f"and fixed envelope use {total_tokens} > {input_limit} estimated input tokens"
            )
        admitted_chunk_count = len(result.contexts.get("chunks", []))
        overhead_tokens = total_tokens - evidence_tokens
        evidence_capacity = max(0, input_limit - overhead_tokens)
        result.max_output_tokens = self._context_policy.output_allowance(
            self._model_profile,
            input_tokens=total_tokens,
        )
        result.trace.update(
            {
                "answer_input_limit_tokens": input_limit,
                "context_policy_revision": self._context_policy.revision,
                "answer_evidence_tokens": evidence_tokens,
                "answer_evidence_capacity_tokens": evidence_capacity,
                "answer_input_tokens": total_tokens,
                "answer_history_messages": len(original_history),
                "answer_retrieved_chunk_count": retrieved_chunk_count,
                "answer_capacity_admitted_chunk_count": admitted_chunk_count,
                "answer_capacity_dropped_chunk_count": capacity_dropped_chunk_count,
            }
        )
        if result.max_output_tokens is not None:
            result.trace["answer_output_allowance_tokens"] = result.max_output_tokens
        return result

    def _compose_user_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        excerpt_blocks: list[dict[str, Any]],
        *,
        current_image_blocks: list[dict[str, Any]] | None = None,
        history_messages: list[dict[str, Any]],
        episodic_summary: str = "",
        memory_text: str = "",
    ) -> list[dict[str, Any]]:
        """Place budgeted image blocks into the final message structure.

        The standing memory block rides as its own user-role message after the
        current request — never inside the system prompt (Pi/Kimi convention).
        """
        content: list[dict[str, Any]] = []
        content.extend(current_image_blocks or ())
        content.extend(excerpt_blocks)
        content.append({"type": "text", "text": user_prompt})
        contributions = [
            ContextContribution(
                source="answer.system",
                authority="system",
                messages=({"role": "system", "content": system_prompt},),
                compressible=False,
            )
        ]
        if episodic_summary:
            contributions.append(
                ContextContribution(
                    source="conversation.episodic",
                    authority="conversation",
                    messages=({"role": "user", "content": episodic_summary},),
                )
            )
        if history_messages:
            contributions.append(
                ContextContribution(
                    source="conversation.tail",
                    authority="conversation",
                    messages=tuple(history_messages),
                )
            )
        contributions.append(
            ContextContribution(
                source="answer.evidence" if excerpt_blocks else "answer.question",
                authority="evidence" if excerpt_blocks else "user",
                messages=({"role": "user", "content": content},),
                citable=bool(excerpt_blocks),
                compressible=False,
            )
        )
        memory_message = standing_memory_message(memory_text)
        if memory_message is not None:
            contributions.append(
                ContextContribution(
                    source="profile.memory",
                    authority="profile",
                    messages=(memory_message,),
                )
            )
        return list(ContextProjector().project(contributions).messages)

    @staticmethod
    def _prepare_current_image_blocks(
        current_images: list[dict[str, Any]] | None,
        *,
        image_budget: AnswerImageBudget,
    ) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for index, image in enumerate(current_images or (), start=1):
            bounded = image_budget.add_user_image(image, label=f"current_image_{index}")
            if bounded is None:
                raise CurrentImagePayloadError(
                    f"current image current_image_{index} could not fit the answer image budget"
                )
            blocks.extend(
                (
                    {"type": "text", "text": f"[current image {index}]"},
                    bounded,
                )
            )
        return blocks

    @staticmethod
    def _apply_image_trace(
        trace: dict[str, Any],
        *,
        budget: AnswerImageBudget,
        current_image_count: int,
    ) -> None:
        rag_context = int(trace.get("answer_context_images_sent", 0))
        trace["answer_images_current"] = current_image_count
        trace["answer_images_rag"] = rag_context
        trace["answer_images_total"] = current_image_count + rag_context
        trace["answer_image_budget_used_bytes"] = budget.used_bytes

    def _prepare_prompt_context(
        self,
        query: str,
        contexts: RetrievalContexts,
        *,
        image_budget: AnswerImageBudget | None = None,
    ) -> _PreparedAnswerPrompt:
        if image_budget is None:
            image_budget = self._image_policy.new_budget()
        packed = AnswerContextPacker().pack(
            contexts,
            image_budget=image_budget,
        )
        # Fast and Research use the same Evidence ledger for citation identity;
        # Fast remains a lightweight invocation and never creates an Agent Session.
        from dlightrag.engine.answer.evidence import EvidenceLedger

        evidence = EvidenceLedger()
        evidence.add_contexts(packed.contexts)
        _blocks, indexer = evidence.render_blocks(
            image_blocks_by_context_key=packed.image_blocks_by_context_key
        )
        kg_context = self._format_kg_context(packed.contexts, indexer=indexer)
        user_prompt = "\n\n".join(
            [
                f"## Knowledge Graph Context\n{kg_context}",
                f"## Question\n{query}",
            ]
        )
        trace = dict(packed.trace)
        return _PreparedAnswerPrompt(
            contexts=packed.contexts,
            user_prompt=user_prompt,
            kg_context=kg_context,
            indexer=indexer,
            chunk_image_blocks=packed.image_blocks_by_context_key,
            trace=trace,
        )

    @staticmethod
    def _build_citation_indexer(contexts: RetrievalContexts) -> CitationIndexer:
        """Flatten contexts and build a CitationIndexer."""
        flat: list[dict[str, Any]] = []
        for items in contexts.values():
            if isinstance(items, list):
                flat.extend(items)
        indexer = CitationIndexer()
        indexer.build_index(flat)
        return indexer

    @staticmethod
    def _format_kg_context(
        contexts: RetrievalContexts,
        indexer: CitationIndexer | None = None,
    ) -> str:
        """Format entities/relationships as markdown text (max 20 each).

        When *indexer* is provided, each entity/relationship is annotated with
        citation tags derived from its ``source_id``, so the LLM knows which
        document each KG fact originated from.
        """
        return format_kg_context(contexts, indexer)

    @staticmethod
    def _build_excerpt_blocks(
        contexts: RetrievalContexts,
        indexer: CitationIndexer | None = None,
        image_blocks_by_context_key: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build lane-labelled per-document blocks with interleaved images."""
        chunks = contexts.get("chunks", [])
        if not chunks:
            return []

        attachment_chunks: list[dict[str, Any]] = []
        rag_chunks: list[dict[str, Any]] = []
        for chunk in chunks:
            source_type = str((chunk.get("metadata") or {}).get("source_type") or "")
            if source_type == "web_attachment":
                attachment_chunks.append(chunk)
            else:
                rag_chunks.append(chunk)

        blocks: list[dict[str, Any]] = []
        if attachment_chunks:
            blocks.append({"type": "text", "text": "## User-attached documents"})
            blocks.extend(
                build_excerpt_lane_blocks(
                    attachment_chunks,
                    indexer=indexer,
                    image_blocks_by_context_key=image_blocks_by_context_key,
                )
            )
        if rag_chunks:
            blocks.append({"type": "text", "text": "## Knowledge-base evidence"})
            blocks.extend(
                build_excerpt_lane_blocks(
                    rag_chunks,
                    indexer=indexer,
                    image_blocks_by_context_key=image_blocks_by_context_key,
                )
            )
        return blocks


async def _prepend_no_context_stream(token_iterator: Any) -> AsyncIterator[str]:
    yield f"{NO_CONTEXT_DISCLAIMER}\n\n"
    if isinstance(token_iterator, str):
        yield token_iterator
        return
    if token_iterator is None:
        return
    try:
        async for token in token_iterator:
            yield token
    finally:
        await aclose_answer_stream(token_iterator)


__all__ = ["NO_CONTEXT_DISCLAIMER", "AnswerSynthesizer"]
