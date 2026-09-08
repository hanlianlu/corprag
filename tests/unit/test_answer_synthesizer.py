# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the AnswerSynthesizer messages-first interface."""

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from dlightrag.engine.agent.session.fold import PriorTurns
from dlightrag.engine.ai.capacity import ContextPolicy, ModelProfile
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.answer.citations.finalization import finalize_answer
from dlightrag.engine.answer.citations.streaming import AnswerStream
from dlightrag.engine.answer.errors import (
    AnswerInputOverflowError,
    CurrentImagePayloadError,
)
from dlightrag.engine.answer.memory import reserved_auto_recall_text
from dlightrag.engine.answer.synthesizer import NO_CONTEXT_DISCLAIMER, AnswerSynthesizer
from dlightrag.engine.rag.retrieval import RetrievalContexts
from tests.unit.conftest import answer_image_policy, answer_model_profile

_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _image_block(payload: str = _PNG_B64) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}}


def _source_metadata(file_path: str, **extra: object) -> dict[str, object]:
    file_name = file_path.rsplit("/", 1)[-1]
    return {
        "source_uri": f"local://default/{file_name}",
        "source_download_locator": file_path,
        **extra,
    }


def _text_contexts() -> RetrievalContexts:
    return {
        "chunks": [
            {
                "chunk_id": "c1",
                "reference_id": "1",
                "file_path": "/docs/report.pdf",
                "content": "Revenue grew 15%.",
                "page_number": 3,
                "_workspace": "default",
                "metadata": _source_metadata("/docs/report.pdf"),
            },
        ],
        "entities": [
            {
                "entity_name": "Revenue",
                "entity_type": "Metric",
                "description": "Total revenue",
                "source_id": "c1",
            },
        ],
        "relationships": [],
    }


def _image_contexts() -> RetrievalContexts:
    return {
        "chunks": [
            {
                "chunk_id": "c1",
                "reference_id": "1",
                "file_path": "/docs/chart.pdf",
                "content": "Chart showing growth",
                "page_number": 1,
                "image_data": _PNG_B64,
                "_workspace": "default",
                "metadata": _source_metadata("/docs/chart.pdf"),
            },
        ],
        "entities": [],
        "relationships": [],
    }


def _multi_doc_contexts() -> RetrievalContexts:
    return {
        "chunks": [
            {
                "chunk_id": "c1",
                "reference_id": "1",
                "file_path": "/docs/report.pdf",
                "content": "Revenue data.",
                "page_number": 3,
                "image_data": _PNG_B64,
                "_workspace": "default",
                "metadata": _source_metadata("/docs/report.pdf", title="2025 Annual Report"),
            },
            {
                "chunk_id": "c2",
                "reference_id": "1",
                "file_path": "/docs/report.pdf",
                "content": "Expenses data.",
                "page_number": 7,
                "image_data": _PNG_B64,
                "_workspace": "default",
                "metadata": _source_metadata("/docs/report.pdf"),
            },
            {
                "chunk_id": "c3",
                "reference_id": "2",
                "file_path": "/docs/other.pdf",
                "content": "Other info.",
                "page_number": 1,
                "_workspace": "default",
                "metadata": _source_metadata("/docs/other.pdf"),
            },
        ],
        "entities": [],
        "relationships": [],
    }


def _capacity_contexts(
    count: int,
    *,
    graph_source: str | None = None,
) -> RetrievalContexts:
    chunks = [
        {
            "chunk_id": f"capacity-{index}",
            "reference_id": "1",
            "file_path": "/docs/capacity.pdf",
            "content": f"CAPACITY-MARKER-{index}",
            "_workspace": "default",
            "metadata": _source_metadata("/docs/capacity.pdf"),
        }
        for index in range(1, count + 1)
    ]
    source = graph_source or (str(chunks[0]["chunk_id"]) if chunks else "")
    return {
        "chunks": chunks,
        "entities": [
            {
                "entity_name": "Capacity",
                "description": "Graph context remains corpus-level during tail admission.",
                "source_id": source,
                "_workspace": "default",
            }
        ],
        "relationships": [
            {
                "src_id": "Capacity",
                "tgt_id": "Reserve",
                "description": "Uses",
                "source_id": source,
                "_workspace": "default",
            }
        ],
    }


def _capacity_marker_count(messages: object) -> int:
    rendered = repr(messages)
    return sum(rendered.count(f"CAPACITY-MARKER-{index}") for index in range(1, 10))


def _capacity_synthesizer() -> AnswerSynthesizer:
    return AnswerSynthesizer(
        image_policy=answer_image_policy(),
        model_profile=ModelProfile(context_window_tokens=101, max_input_tokens=100),
        context_policy=ContextPolicy(
            requested_output_reserve_tokens=0,
            dynamic_context_reserve_tokens=40,
            safety_reserve_tokens=0,
            minimum_input_tokens=0,
        ),
    )


# ---------------------------------------------------------------------------
# TestAnswerSynthesizerPolicy
# ---------------------------------------------------------------------------


def _stream_func(*tokens: str) -> AsyncMock:
    """A model callable that streams *tokens*, like every real answer provider."""

    async def _tokens() -> AsyncIterator[str]:
        for token in tokens:
            yield token

    return AsyncMock(side_effect=lambda **_kwargs: _tokens())


async def _drain(token_iter: object) -> str:
    return "".join([token async for token in cast(AsyncIterator[str], token_iter)])


class TestAnswerSynthesizerPolicy:
    """Evidence packing, image budgets, and citation cleanup on the one path."""

    @pytest.mark.asyncio
    async def test_evidence_is_prepared_off_the_event_loop(self, monkeypatch) -> None:
        import dlightrag.engine.answer.synthesizer as answer_module

        calls = []

        async def fake_to_thread(func, *args, **kwargs):
            calls.append(func.__name__)
            return func(*args, **kwargs)

        monkeypatch.setattr(
            answer_module,
            "asyncio",
            SimpleNamespace(to_thread=fake_to_thread),
            raising=False,
        )
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(),
            model_func=_stream_func("a"),
        )

        await synth.generate_stream("query", _image_contexts())

        assert "_prepare_model_call" in calls

    @pytest.mark.asyncio
    async def test_images_are_sent_when_the_policy_allows_them(self) -> None:
        model_func = _stream_func("ok")
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(max_images=6),
            model_profile=answer_model_profile(),
            model_func=model_func,
        )

        await synth.generate_stream("describe", _image_contexts())

        user_content = model_func.call_args.kwargs["messages"][1]["content"]
        assert any(item.get("type") == "image_url" for item in user_content)
        assert any(item.get("type") == "text" for item in user_content)

    @pytest.mark.asyncio
    async def test_current_images_share_the_final_serializer_budget_and_trace(self) -> None:
        model_func = _stream_func("ok")
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(max_images=3),
            model_profile=answer_model_profile(),
            model_func=model_func,
        )
        current_images = [_image_block(), _image_block()]

        _, stream = await synth.generate_stream(
            "describe",
            _image_contexts(),
            conversation_history=PriorTurns(
                [{"role": "user", "content": "Earlier text-only question"}]
            ),
            current_images=current_images,
        )

        messages = model_func.call_args.kwargs["messages"]
        image_blocks = [
            block
            for message in messages
            for block in (
                message.get("content") if isinstance(message.get("content"), list) else []
            )
            if block.get("type") == "image_url"
        ]
        current_message = messages[-1]["content"]
        assert len(image_blocks) == 3
        assert [block["text"] for block in current_message if block.get("type") == "text"][:2] == [
            "[current image 1]",
            "[current image 2]",
        ]
        assert (
            sum(
                "## Question\ndescribe" in str(block.get("text", ""))
                for message in messages
                for block in (
                    message.get("content") if isinstance(message.get("content"), list) else []
                )
            )
            == 1
        )
        assert cast(Any, stream).trace["answer_images_current"] == 2
        assert cast(Any, stream).trace["answer_images_rag"] == 1
        assert cast(Any, stream).trace["answer_images_total"] == 3

    def test_history_measure_uses_the_exact_current_image_serializer(self) -> None:
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(max_images=2),
            model_profile=answer_model_profile(),
        )
        current_images = [_image_block(), _image_block()]

        measured = synth.history_input_measure(
            "question",
            current_images=current_images,
        )([])
        prepared = synth._prepare_model_call(
            "question",
            {"chunks": [], "entities": [], "relationships": []},
            current_images=current_images,
        )

        assert measured == prepared.trace["answer_input_tokens"]
        assert [
            block["type"]
            for block in prepared.messages[-1]["content"]
            if block.get("type") in {"text", "image_url"}
        ][:4] == ["text", "image_url", "text", "image_url"]

    @pytest.mark.asyncio
    async def test_current_images_are_all_or_error_before_provider_output(self) -> None:
        model_func = _stream_func("must not run")
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(max_images=1),
            model_profile=answer_model_profile(),
            model_func=model_func,
        )

        with pytest.raises(CurrentImagePayloadError, match="current_image_2"):
            await synth.generate_stream(
                "describe",
                {"chunks": [], "entities": [], "relationships": []},
                current_images=[_image_block(), _image_block()],
            )

        model_func.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_each_call_gets_a_fresh_budget_from_one_policy(self) -> None:
        policy = answer_image_policy(max_images=1)
        synth = AnswerSynthesizer(
            image_policy=policy,
            model_profile=answer_model_profile(),
            model_func=_stream_func("ok [1-1]."),
        )

        _, first = await synth.generate_stream("describe", _image_contexts())
        _, second = await synth.generate_stream("describe", _image_contexts())

        assert cast(Any, first).trace["answer_images_total"] == 1
        assert cast(Any, second).trace["answer_images_total"] == 1
        assert policy.max_images == 1

    @pytest.mark.asyncio
    async def test_image_budget_omission_is_not_reported_as_capacity_drop(self) -> None:
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(max_images=0),
            model_profile=answer_model_profile(),
            model_func=_stream_func("answer"),
        )
        contexts: RetrievalContexts = {
            "chunks": [
                {
                    "chunk_id": "visual-only",
                    "reference_id": "1",
                    "file_path": "/docs/figures.pdf",
                    "content": "",
                    "image_data": _PNG_B64,
                    "_workspace": "default",
                    "metadata": _source_metadata("/docs/figures.pdf"),
                },
                *_text_contexts()["chunks"],
            ],
            "entities": [],
            "relationships": [],
        }

        packed, token_iter = await synth.generate_stream("q", contexts)

        assert packed is not contexts
        assert [c["chunk_id"] for c in packed["chunks"]] == ["c1"]
        trace = cast(Any, token_iter).trace
        assert trace["answer_context_images_skipped"] == 1
        assert trace["answer_retrieved_chunk_count"] == 2
        assert trace["answer_capacity_admitted_chunk_count"] == 1
        assert trace["answer_capacity_dropped_chunk_count"] == 0

    @pytest.mark.asyncio
    async def test_the_settled_answer_drops_a_model_generated_references_tail(self) -> None:
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(),
            model_func=_stream_func("Growth is 15% [1-1].", "\n\n### References\n- [1] x.pdf"),
        )

        _, token_iter = await synth.generate_stream("query", _text_contexts())
        await _drain(token_iter)

        settled = cast(Any, token_iter).answer
        assert "Growth is 15%" in settled
        assert "### References" not in settled


# ---------------------------------------------------------------------------
# TestAnswerSynthesizerStream
# ---------------------------------------------------------------------------


class TestAnswerSynthesizerStream:
    @pytest.mark.asyncio
    async def test_generate_stream_wraps_with_answer_stream(self) -> None:
        async def mock_tokens():
            for token in ["Hello", " world"]:
                yield token

        model_func = AsyncMock(return_value=mock_tokens())
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(),
            model_func=model_func,
        )

        _ctx, token_iter = await synth.generate_stream("test", _text_contexts())

        from dlightrag.engine.answer.citations.streaming import AnswerStream

        assert isinstance(token_iter, AnswerStream)

    @pytest.mark.asyncio
    async def test_generate_stream_no_model_func(self) -> None:
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(),
            model_func=None,
        )
        contexts: RetrievalContexts = {"chunks": []}
        ctx, token_iter = await synth.generate_stream("test", contexts)
        assert token_iter is None
        assert ctx is contexts

    @pytest.mark.asyncio
    async def test_generate_stream_empty_context_disclaims_general_knowledge(self) -> None:
        async def mock_stream():
            yield "I am DlightRAG."

        model_func = AsyncMock(return_value=mock_stream())
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(),
            model_func=model_func,
        )
        contexts: RetrievalContexts = {"chunks": [], "entities": [], "relationships": []}

        ctx, token_iter = await synth.generate_stream("who are u", contexts)

        assert ctx == contexts
        assert token_iter is not None
        assert [token async for token in token_iter] == [
            f"{NO_CONTEXT_DISCLAIMER}\n\n",
            "I am DlightRAG.",
        ]

    @pytest.mark.asyncio
    async def test_closing_no_context_answer_releases_scheduled_model_stream(self) -> None:
        scheduler = ModelScheduler(max_concurrency=1)
        source_closed = asyncio.Event()
        competing_started = asyncio.Event()

        async def source():
            try:
                yield "first"
                await asyncio.Event().wait()
            finally:
                source_closed.set()

        async def model_func(**_kwargs: Any):
            return scheduler.stream(source)

        async def competing() -> None:
            competing_started.set()

        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(),
            model_func=model_func,
        )
        contexts: RetrievalContexts = {"chunks": [], "entities": [], "relationships": []}
        _ctx, token_iter = await synth.generate_stream("who are u", contexts)
        assert isinstance(token_iter, AnswerStream)
        assert await anext(token_iter) == f"{NO_CONTEXT_DISCLAIMER}\n\n"
        assert await anext(token_iter) == "first"
        waiting = asyncio.create_task(scheduler.run(competing))
        await asyncio.sleep(0)
        assert not competing_started.is_set()

        await token_iter.aclose()

        assert source_closed.is_set()
        await waiting
        assert competing_started.is_set()

    @pytest.mark.asyncio
    async def test_generate_stream_returns_packed_contexts_and_tokens(self) -> None:
        async def mock_stream():
            for token in ["Hello", " ", "world"]:
                yield token

        model_func = AsyncMock(return_value=mock_stream())
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(),
            model_func=model_func,
        )

        result_contexts, token_iter = await synth.generate_stream("query", _text_contexts())

        assert result_contexts is not _text_contexts()
        assert token_iter is not None
        assert [t async for t in token_iter] == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_generate_stream_no_response_format_and_streams(self) -> None:
        async def mock_stream():
            yield "text"

        model_func = AsyncMock(return_value=mock_stream())
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(),
            model_func=model_func,
        )

        await synth.generate_stream("query", _text_contexts())

        call_kwargs = model_func.call_args.kwargs
        assert "response_format" not in call_kwargs
        assert call_kwargs.get("stream") is True
        assert isinstance(call_kwargs.get("usage_holder"), dict)

    @pytest.mark.asyncio
    async def test_generate_stream_passes_messages_and_indexer(self) -> None:
        async def mock_stream():
            yield "token"

        model_func = AsyncMock(return_value=mock_stream())
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(),
            model_func=model_func,
        )

        _, token_iter = await synth.generate_stream("query", _text_contexts())

        from dlightrag.engine.answer.citations.streaming import AnswerStream

        messages = model_func.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert isinstance(token_iter, AnswerStream)
        assert token_iter._indexer is not None


# ---------------------------------------------------------------------------
# TestAnswerSynthesizerCapacity
# ---------------------------------------------------------------------------


class TestAnswerSynthesizerCapacity:
    def test_history_measure_grows_when_memory_is_reserved(self) -> None:
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(),
        )
        empty = synth.history_input_measure("query")
        reserved = synth.history_input_measure("query", memory_text=reserved_auto_recall_text())
        assert reserved([]) > empty([])

    def test_fast_prompt_projects_accepted_episodic_summary_before_tail(self) -> None:
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(),
        )

        prepared = synth._prepare_model_call(
            "question",
            _text_contexts(),
            conversation_history=PriorTurns(
                [{"role": "user", "content": "recent turn"}],
                episodic_summary="older accepted turns",
            ),
        )

        contents = [message.get("content") for message in prepared.messages]
        assert contents.index("older accepted turns") < contents.index("recent turn")

    def test_evidence_capacity_uses_the_full_residual_model_input(self) -> None:
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=ModelProfile(
                context_window_tokens=10_000,
                max_input_tokens=9_000,
                max_output_tokens=1_000,
            ),
        )

        prepared = synth._prepare_model_call(
            "question", _text_contexts(), conversation_history=PriorTurns()
        )

        assert prepared.trace["answer_input_limit_tokens"] == 7_976
        assert prepared.trace["context_policy_revision"] == "agent-v4-dynamic-context"
        assert prepared.trace["answer_evidence_capacity_tokens"] == 7_976 - (
            prepared.trace["answer_input_tokens"] - prepared.trace["answer_evidence_tokens"]
        )
        assert prepared.trace["answer_evidence_capacity_tokens"] > 6_000

    def test_empty_retrieval_context_remains_no_context(self) -> None:
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(),
        )
        contexts: RetrievalContexts = {"chunks": [], "entities": [], "relationships": []}

        prepared = synth._prepare_model_call("question", contexts)

        assert prepared.contexts == contexts
        assert prepared.no_context is True
        assert prepared.trace["answer_no_context"] is True

    def test_oversized_single_chunk_is_removed_whole_without_mutating_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dlightrag.engine.answer.synthesizer as answer_module

        contexts = _capacity_contexts(1)
        original_content = contexts["chunks"][0]["content"]
        monkeypatch.setattr(
            answer_module,
            "estimate_messages_tokens",
            lambda messages: 90 + 20 * _capacity_marker_count(messages),
        )
        synth = _capacity_synthesizer()

        prepared = synth._prepare_model_call(
            "question", contexts, conversation_history=PriorTurns()
        )

        assert prepared.contexts["chunks"] == []
        assert prepared.trace["answer_retrieved_chunk_count"] == 1
        assert prepared.trace["answer_capacity_admitted_chunk_count"] == 0
        assert prepared.trace["answer_capacity_dropped_chunk_count"] == 1
        assert contexts["chunks"][0]["content"] == original_content

    @pytest.mark.parametrize(
        ("fixed_tokens", "expected_ids"),
        [(50, ["capacity-1", "capacity-2"]), (70, ["capacity-1"])],
    )
    def test_exact_tail_admission_drops_one_or_multiple_whole_chunks(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fixed_tokens: int,
        expected_ids: list[str],
    ) -> None:
        import dlightrag.engine.answer.synthesizer as answer_module

        contexts = _capacity_contexts(3, graph_source="capacity-3")
        original_chunks = [dict(chunk) for chunk in contexts["chunks"]]
        monkeypatch.setattr(
            answer_module,
            "estimate_messages_tokens",
            lambda messages: fixed_tokens + 20 * _capacity_marker_count(messages),
        )

        prepared = _capacity_synthesizer()._prepare_model_call("question", contexts)

        assert [chunk["chunk_id"] for chunk in prepared.contexts["chunks"]] == expected_ids
        assert [chunk["content"] for chunk in prepared.contexts["chunks"]] == [
            chunk["content"] for chunk in original_chunks[: len(expected_ids)]
        ]
        assert prepared.indexer.get_max_chunk_idx("1") == len(expected_ids)
        assert prepared.indexer.get_chunk_id("1", len(expected_ids)) == expected_ids[-1]
        assert prepared.indexer.get_chunk_id("1", len(expected_ids) + 1) is None
        finalized = finalize_answer(
            "Grounded in the first survivor [1-1].",
            prepared.contexts,
            indexer=prepared.indexer,
        )
        assert len(finalized.sources) == 1
        assert finalized.sources[0].cited_chunk_ids == ["capacity-1"]
        assert prepared.contexts["entities"][0]["source_id"] == "capacity-3"
        assert prepared.contexts["relationships"][0]["source_id"] == "capacity-3"
        assert prepared.trace["answer_retrieved_chunk_count"] == 3
        assert prepared.trace["answer_capacity_admitted_chunk_count"] == len(expected_ids)
        assert prepared.trace["answer_capacity_dropped_chunk_count"] == 3 - len(expected_ids)
        assert contexts["chunks"] == original_chunks

    def test_exact_boundary_fit_does_not_drop_a_chunk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dlightrag.engine.answer.synthesizer as answer_module

        contexts = _capacity_contexts(3)
        monkeypatch.setattr(
            answer_module,
            "estimate_messages_tokens",
            lambda messages: 40 + 20 * _capacity_marker_count(messages),
        )

        prepared = _capacity_synthesizer()._prepare_model_call("question", contexts)

        assert prepared.trace["answer_input_tokens"] == 100
        assert prepared.trace["answer_capacity_admitted_chunk_count"] == 3
        assert prepared.trace["answer_capacity_dropped_chunk_count"] == 0

    def test_image_block_does_not_change_exact_text_capacity(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlightrag.engine.answer.excerpts as excerpts_module

        monkeypatch.setattr(excerpts_module, "build_image_label", lambda **_kwargs: "")
        policy = ContextPolicy(
            requested_output_reserve_tokens=0,
            dynamic_context_reserve_tokens=0,
            safety_reserve_tokens=0,
            minimum_input_tokens=0,
        )
        plain_contexts = _text_contexts()
        image_contexts: RetrievalContexts = {
            key: [dict(item) for item in value] for key, value in plain_contexts.items()
        }
        image_contexts["chunks"][0]["image_data"] = _PNG_B64
        probe = AnswerSynthesizer(
            image_policy=answer_image_policy(max_images=6),
            model_profile=ModelProfile(context_window_tokens=10_000),
            context_policy=policy,
        )

        plain = probe._prepare_model_call("question", plain_contexts)
        with_image = probe._prepare_model_call("question", image_contexts)

        assert with_image.trace["answer_context_images_sent"] == 1
        assert with_image.trace["answer_input_tokens"] == plain.trace["answer_input_tokens"]
        exact_input = with_image.trace["answer_input_tokens"]
        exact = AnswerSynthesizer(
            image_policy=answer_image_policy(max_images=6),
            model_profile=ModelProfile(
                context_window_tokens=exact_input + 1,
                max_input_tokens=exact_input,
            ),
            context_policy=policy,
        )._prepare_model_call("question", image_contexts)
        assert [chunk["chunk_id"] for chunk in exact.contexts["chunks"]] == ["c1"]
        assert exact.trace["answer_capacity_dropped_chunk_count"] == 0

    def test_non_chunk_context_overflow_after_all_chunks_are_removed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dlightrag.engine.answer.synthesizer as answer_module

        contexts = _capacity_contexts(2)
        monkeypatch.setattr(answer_module, "estimate_messages_tokens", lambda _messages: 101)

        with pytest.raises(AnswerInputOverflowError, match="retained non-chunk context"):
            _capacity_synthesizer()._prepare_model_call("question", contexts)

        assert [chunk["chunk_id"] for chunk in contexts["chunks"]] == [
            "capacity-1",
            "capacity-2",
        ]

    def test_pinned_history_is_not_locally_trimmed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import dlightrag.engine.answer.synthesizer as answer_module

        monkeypatch.setattr(answer_module, "answer_core", lambda: "SYS")

        contexts: RetrievalContexts = {
            "chunks": [
                {
                    "chunk_id": "rag-1",
                    "reference_id": "rag-doc",
                    "file_path": "rag.pdf",
                    "content": "workspace evidence",
                    "metadata": {
                        "source_uri": "file:///rag.pdf",
                        "source_download_locator": "rag.pdf",
                    },
                    "_workspace": "default",
                }
            ],
            "entities": [],
            "relationships": [],
        }
        original_chunks = [dict(contexts["chunks"][0])]
        # A window whose input budget only fits recent history plus fixed input.
        synth = AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(context_window_tokens=500),
        )

        history = PriorTurns(
            [
                {"role": "user", "content": "OLD-HISTORY " + ("old " * 1_000)},
                {"role": "assistant", "content": "old answer " * 60},
                {"role": "user", "content": "RECENT-HISTORY follow-up"},
                {"role": "assistant", "content": "recent answer"},
            ]
        )

        with pytest.raises(AnswerInputOverflowError):
            synth._prepare_model_call(
                "current question",
                contexts,
                conversation_history=history,
            )

        assert len(history) == 4
        assert contexts["chunks"] == original_chunks


# ---------------------------------------------------------------------------
# TestAnswerSynthesizerHelpers
# ---------------------------------------------------------------------------


class TestAnswerSynthesizerHelpers:
    def test_format_kg_context_with_entities_and_rels(self) -> None:
        contexts: RetrievalContexts = {
            "chunks": [],
            "entities": [
                {
                    "entity_name": "Acme",
                    "entity_type": "Company",
                    "description": "A company",
                    "source_id": "s1",
                },
            ],
            "relationships": [
                {
                    "src_id": "Acme",
                    "tgt_id": "Revenue",
                    "description": "generates",
                    "source_id": "s1",
                },
            ],
        }
        result = AnswerSynthesizer._format_kg_context(contexts)
        assert "## Entities" in result
        assert "**Acme**" in result
        assert "## Relationships" in result
        assert "Acme -> Revenue" in result

    def test_format_kg_context_empty(self) -> None:
        contexts: RetrievalContexts = {"chunks": [], "entities": [], "relationships": []}
        assert (
            AnswerSynthesizer._format_kg_context(contexts)
            == "No knowledge graph context available."
        )

    def test_format_kg_context_includes_doc_level_tags(self) -> None:
        from dlightrag.engine.answer.citations.indexer import CitationIndexer

        contexts: RetrievalContexts = {
            "chunks": [
                {
                    "chunk_id": "c1",
                    "reference_id": "1",
                    "file_path": "/docs/report.pdf",
                    "content": "Revenue data.",
                    "page_number": 1,
                },
            ],
            "entities": [
                {
                    "entity_name": "Revenue",
                    "entity_type": "Metric",
                    "description": "Total revenue grew 15%",
                    "source_id": "c1",
                },
            ],
            "relationships": [
                {
                    "src_id": "Acme",
                    "tgt_id": "Revenue",
                    "description": "reports",
                    "source_id": "c1",
                },
            ],
        }
        flat: list = []
        for items in contexts.values():
            if isinstance(items, list):
                flat.extend(items)
        indexer = CitationIndexer()
        indexer.build_index(flat)

        result = AnswerSynthesizer._format_kg_context(contexts, indexer=indexer)
        assert "(from [1])" in result

    def test_build_citation_indexer(self) -> None:
        indexer = AnswerSynthesizer._build_citation_indexer(_text_contexts())
        assert indexer.get_max_chunk_idx("1") > 0


# ---------------------------------------------------------------------------
# TestBuildExcerptBlocks
# ---------------------------------------------------------------------------


class TestBuildExcerptBlocks:
    def test_groups_chunks_by_document(self) -> None:
        from dlightrag.engine.answer.citations.indexer import CitationIndexer

        contexts = _multi_doc_contexts()
        indexer = CitationIndexer()
        indexer.build_index(list(contexts["chunks"]))

        blocks = AnswerSynthesizer._build_excerpt_blocks(contexts, indexer=indexer)

        all_text = "\n".join(b["text"] for b in blocks if b.get("type") == "text")
        assert "Document [1]" in all_text
        assert "Document [2]" in all_text
        assert "report.pdf" in all_text
        assert "other.pdf" in all_text

    def test_images_interleaved_with_document(self) -> None:
        contexts = _multi_doc_contexts()
        blocks = AnswerSynthesizer._build_excerpt_blocks(contexts)

        image_blocks = [b for b in blocks if b.get("type") == "image_url"]
        assert len(image_blocks) == 2

    def test_empty_chunks_returns_empty_blocks(self) -> None:
        assert AnswerSynthesizer._build_excerpt_blocks({"chunks": []}) == []
