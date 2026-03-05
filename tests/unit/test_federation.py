# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for cross-workspace federated retrieval."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dlightrag.retrieval.engine import RetrievalResult
from dlightrag.retrieval.federation import (
    federated_answer,
    federated_retrieve,
    merge_results,
)


def _make_result(
    chunks: list[dict] | None = None,
    sources: list[dict] | None = None,
    answer: str | None = None,
) -> RetrievalResult:
    """Helper to create a RetrievalResult with given data."""
    return RetrievalResult(
        answer=answer,
        contexts={
            "chunks": chunks or [],
            "entities": [],
            "relationships": [],
        },
        raw={
            "sources": sources or [],
            "media": [],
        },
    )


class TestMergeResults:
    """Test round-robin merge logic."""

    def test_round_robin_interleaves_chunks(self) -> None:
        r1 = _make_result(chunks=[{"id": "a1"}, {"id": "a2"}, {"id": "a3"}])
        r2 = _make_result(chunks=[{"id": "b1"}, {"id": "b2"}])

        merged = merge_results([r1, r2], ["ws-a", "ws-b"])
        chunk_ids = [c["id"] for c in merged.contexts["chunks"]]

        assert chunk_ids == ["a1", "b1", "a2", "b2", "a3"]

    def test_chunks_tagged_with_workspace(self) -> None:
        r1 = _make_result(chunks=[{"id": "c1"}])
        r2 = _make_result(chunks=[{"id": "c2"}])

        merged = merge_results([r1, r2], ["legal", "finance"])

        assert merged.contexts["chunks"][0]["_workspace"] == "legal"
        assert merged.contexts["chunks"][1]["_workspace"] == "finance"

    def test_chunk_top_k_truncates(self) -> None:
        r1 = _make_result(chunks=[{"id": f"a{i}"} for i in range(10)])
        r2 = _make_result(chunks=[{"id": f"b{i}"} for i in range(10)])

        merged = merge_results([r1, r2], ["ws-a", "ws-b"], chunk_top_k=5)

        assert len(merged.contexts["chunks"]) == 5

    def test_sources_merged_and_tagged(self) -> None:
        r1 = _make_result(sources=[{"id": "s1", "title": "Doc A"}])
        r2 = _make_result(sources=[{"id": "s2", "title": "Doc B"}])

        merged = merge_results([r1, r2], ["ws-a", "ws-b"])
        sources = merged.raw["sources"]

        assert len(sources) == 2
        assert sources[0]["_workspace"] == "ws-a"
        assert sources[1]["_workspace"] == "ws-b"

    def test_answers_concatenated(self) -> None:
        r1 = _make_result(answer="Answer from A")
        r2 = _make_result(answer="Answer from B")

        merged = merge_results([r1, r2], ["ws-a", "ws-b"])

        assert "Answer from A" in merged.answer
        assert "Answer from B" in merged.answer

    def test_none_answers_skipped(self) -> None:
        r1 = _make_result(answer=None)
        r2 = _make_result(answer="Only B answered")

        merged = merge_results([r1, r2], ["ws-a", "ws-b"])

        assert merged.answer == "Only B answered"

    def test_empty_results(self) -> None:
        merged = merge_results([], [])
        assert merged.contexts["chunks"] == []
        assert merged.raw["sources"] == []
        assert merged.answer is None

    def test_workspaces_recorded_in_raw(self) -> None:
        r1 = _make_result()
        r2 = _make_result()

        merged = merge_results([r1, r2], ["ws-a", "ws-b"])

        assert merged.raw["workspaces"] == ["ws-a", "ws-b"]


class TestFederatedRetrieve:
    """Test federated_retrieve orchestration."""

    @pytest.mark.asyncio
    async def test_single_workspace_no_federation(self) -> None:
        mock_svc = AsyncMock()
        mock_svc.aretrieve.return_value = _make_result(
            chunks=[{"id": "c1"}], sources=[{"id": "s1"}]
        )

        async def get_svc(ws: str):
            return mock_svc

        result = await federated_retrieve("test query", ["ws-only"], get_svc)

        mock_svc.aretrieve.assert_awaited_once()
        assert result.contexts["chunks"][0]["_workspace"] == "ws-only"
        assert result.raw["workspaces"] == ["ws-only"]

    @pytest.mark.asyncio
    async def test_multi_workspace_parallel(self) -> None:
        svc_a = AsyncMock()
        svc_a.aretrieve.return_value = _make_result(chunks=[{"id": "a1"}])
        svc_b = AsyncMock()
        svc_b.aretrieve.return_value = _make_result(chunks=[{"id": "b1"}])

        services = {"ws-a": svc_a, "ws-b": svc_b}

        async def get_svc(ws: str):
            return services[ws]

        result = await federated_retrieve("query", ["ws-a", "ws-b"], get_svc)

        assert len(result.contexts["chunks"]) == 2
        assert result.contexts["chunks"][0]["_workspace"] == "ws-a"
        assert result.contexts["chunks"][1]["_workspace"] == "ws-b"

    @pytest.mark.asyncio
    async def test_failed_workspace_excluded(self) -> None:
        svc_ok = AsyncMock()
        svc_ok.aretrieve.return_value = _make_result(chunks=[{"id": "ok1"}])

        svc_fail = AsyncMock()
        svc_fail.aretrieve.side_effect = RuntimeError("DB down")

        services = {"ws-ok": svc_ok, "ws-fail": svc_fail}

        async def get_svc(ws: str):
            return services[ws]

        result = await federated_retrieve("query", ["ws-ok", "ws-fail"], get_svc)

        assert len(result.contexts["chunks"]) == 1
        assert result.contexts["chunks"][0]["_workspace"] == "ws-ok"

    @pytest.mark.asyncio
    async def test_all_workspaces_fail(self) -> None:
        svc = AsyncMock()
        svc.aretrieve.side_effect = RuntimeError("fail")

        async def get_svc(ws: str):
            return svc

        result = await federated_retrieve("query", ["ws-a", "ws-b"], get_svc)

        assert result.contexts["chunks"] == []
        assert "errors" in result.raw

    @pytest.mark.asyncio
    async def test_workspace_filter_rbac(self) -> None:
        svc = AsyncMock()
        svc.aretrieve.return_value = _make_result(chunks=[{"id": "c1"}])

        async def get_svc(ws: str):
            return svc

        async def only_allow_a(requested: list[str]) -> list[str]:
            return [ws for ws in requested if ws == "ws-a"]

        result = await federated_retrieve(
            "query", ["ws-a", "ws-b"], get_svc, workspace_filter=only_allow_a
        )

        svc.aretrieve.assert_awaited_once()
        assert result.contexts["chunks"][0]["_workspace"] == "ws-a"

    @pytest.mark.asyncio
    async def test_workspace_filter_denies_all(self) -> None:
        svc = AsyncMock()

        async def get_svc(ws: str):
            return svc

        async def deny_all(requested: list[str]) -> list[str]:
            return []

        result = await federated_retrieve("query", ["ws-a"], get_svc, workspace_filter=deny_all)

        svc.aretrieve.assert_not_awaited()
        assert result.contexts["chunks"] == []

    @pytest.mark.asyncio
    async def test_empty_workspaces_list(self) -> None:
        async def get_svc(ws: str):
            raise AssertionError("Should not be called")

        result = await federated_retrieve("query", [], get_svc)

        assert result.contexts["chunks"] == []


class TestFederatedAnswer:
    """Test federated_answer orchestration."""

    @pytest.mark.asyncio
    async def test_single_workspace_uses_aanswer(self) -> None:
        mock_svc = AsyncMock()
        mock_svc.aanswer.return_value = _make_result(chunks=[{"id": "c1"}], answer="The answer")

        async def get_svc(ws: str):
            return mock_svc

        result = await federated_answer("query", ["ws-only"], get_svc)

        mock_svc.aanswer.assert_awaited_once()
        assert result.answer == "The answer"

    @pytest.mark.asyncio
    async def test_multi_workspace_merges_answers(self) -> None:
        svc_a = AsyncMock()
        svc_a.aanswer.return_value = _make_result(answer="Answer A")
        svc_b = AsyncMock()
        svc_b.aanswer.return_value = _make_result(answer="Answer B")

        services = {"a": svc_a, "b": svc_b}

        async def get_svc(ws: str):
            return services[ws]

        result = await federated_answer("query", ["a", "b"], get_svc)

        assert "Answer A" in result.answer
        assert "Answer B" in result.answer
