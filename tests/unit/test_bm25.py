# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for PostgreSQL BM25 retrieval."""

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.adapters.postgres.core._pool import PGPool
from dlightrag.adapters.postgres.corpus.corpus_bm25 import (
    BM25_LANGUAGE_COLUMN,
    BM25IndexOptions,
    PGBM25ProfileSearch,
    build_bm25_sql,
    create_postgres_bm25,
    rebuild_postgres_bm25,
    required_postgres_extensions,
)
from dlightrag.engine.rag.retrieval import MetadataFilter, MetadataScope
from dlightrag.engine.rag.retrieval.bm25 import (
    BM25_PROFILE_FALLBACK,
    BM25Profile,
    ProfiledBM25Search,
    profiles_from_config,
)
from dlightrag.engine.rag.retrieval.filtering import metadata_filter_scope
from dlightrag.engine.rag.retrieval.language import BM25LanguageClassifier


def _scope(
    *, candidate_count: int = 12, candidate_count_exact: bool = True, doc_exists: bool = True
) -> MetadataScope:
    return MetadataScope(
        filters=MetadataFilter(filename="x.pdf"),
        filename_mode="exact",
        doc_exists=doc_exists,
        candidate_count=candidate_count,
        candidate_count_exact=candidate_count_exact,
    )


def _metadata_conditions(workspace: str = "default") -> tuple[str, ...]:
    from dlightrag.adapters.postgres.corpus.pg_metadata_index import metadata_match_conditions

    conditions, _params = metadata_match_conditions(
        workspace,
        MetadataFilter(filename="x.pdf"),
        filename_mode="exact",
        start_index=3,
    )
    return tuple(conditions)


def _profiled_bm25(
    searcher: Any,
    *,
    profiles: tuple[BM25Profile, ...] = (BM25_PROFILE_FALLBACK,),
    top_k: int = 40,
) -> ProfiledBM25Search:
    return ProfiledBM25Search(
        searcher,
        workspace="default",
        profiles=profiles,
        top_k=top_k,
    )


def test_bm25_sql_filters_candidates() -> None:
    sql = build_bm25_sql(
        index_name="idx_lightrag_doc_chunks_bm25_en",
        scoped=True,
        limit=20,
        language="en",
        metadata_conditions=_metadata_conditions(),
    )

    assert "full_doc_id IN (SELECT doc_id FROM dlightrag_doc_metadata" in sql
    assert "workspace = $3" in sql
    assert "LIMIT $5" in sql
    assert "to_bm25query" in sql
    assert "idx_lightrag_doc_chunks_bm25_en" in sql
    assert "dlightrag_bm25_language = 'en'" in sql
    assert "ANY(" not in sql


def test_bm25_sql_has_no_candidate_clause_when_unfiltered() -> None:
    sql = build_bm25_sql(
        index_name="idx_lightrag_doc_chunks_bm25_simple",
        scoped=False,
        limit=20,
        language=None,
    )

    assert "full_doc_id IN" not in sql
    assert "LIMIT $3" in sql


def test_bm25_sql_rejects_scoped_without_metadata_conditions() -> None:
    with pytest.raises(ValueError, match="metadata conditions"):
        build_bm25_sql(
            index_name="idx_lightrag_doc_chunks_bm25_simple",
            scoped=True,
            limit=20,
        )


def test_bm25_sql_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        build_bm25_sql(
            index_name="idx_lightrag_doc_chunks_bm25_simple",
            scoped=False,
            limit=0,
        )


def test_bm25_index_options_render_pg_textsearch_with_tuning() -> None:
    profile = BM25Profile(name="en", text_config="english", languages=("en",))
    options = BM25IndexOptions(profile=profile, k1=1.4, b=0.65)

    assert options.create_index_sql() == (
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_en "
        "ON LIGHTRAG_DOC_CHUNKS USING bm25(content) "
        "WITH (text_config='english', k1=1.4, b=0.65) "
        "WHERE dlightrag_bm25_language = 'en'"
    )


def test_bm25_index_options_render_qualified_text_config() -> None:
    profile = BM25Profile(name="zh", text_config="public.jiebacfg", languages=("zh",))
    options = BM25IndexOptions(profile=profile)

    assert "text_config='public.jiebacfg'" in options.create_index_sql()


def test_bm25_language_profile_requires_one_language_bucket() -> None:
    with pytest.raises(ValueError, match="exactly one language"):
        BM25Profile(name="mixed", text_config="simple", languages=("de", "sv"))

    with pytest.raises(ValueError, match="exactly one language"):
        BM25Profile(name="empty", text_config="simple")


def test_bm25_fallback_profile_rejects_languages() -> None:
    with pytest.raises(ValueError, match="fallback profile must not declare languages"):
        BM25Profile(name="simple", text_config="simple", languages=("en",), fallback=True)


def test_bm25_profiles_from_config_preserves_runtime_fields() -> None:
    profiles = profiles_from_config(
        [
            SimpleNamespace(
                name="zh",
                text_config="public.jiebacfg",
                languages=["zh-CN"],
                fallback=False,
            ),
            SimpleNamespace(
                name="simple",
                text_config="simple",
                languages=[],
                fallback=True,
            ),
        ]
    )

    assert profiles == (
        BM25Profile(name="zh", text_config="public.jiebacfg", languages=("zh",)),
        BM25_PROFILE_FALLBACK,
    )


def test_bm25_required_extensions_follow_profile_text_configs() -> None:
    assert required_postgres_extensions(
        [BM25Profile(name="en", text_config="english", languages=("en",))]
    ) == ("pg_textsearch",)
    assert required_postgres_extensions(
        [BM25Profile(name="zh", text_config="public.jiebacfg", languages=("zh",))]
    ) == ("pg_textsearch", "pg_jieba")


def _bm25_config(*, enabled: bool, is_reader: bool = False):
    return SimpleNamespace(
        corpus=SimpleNamespace(
            retrieval=SimpleNamespace(
                bm25_enabled=enabled,
                bm25_profiles=[],
                bm25_k1=1.2,
                bm25_b=0.75,
            )
        ),
        deployment=SimpleNamespace(workspace="default"),
        is_reader=is_reader,
    )


async def test_create_postgres_bm25_returns_none_when_disabled() -> None:
    config = _bm25_config(enabled=False)

    assert await create_postgres_bm25(config) is None


@pytest.mark.parametrize("is_reader", [False, True])
async def test_create_postgres_bm25_provisions_for_service_role(
    monkeypatch: pytest.MonkeyPatch,
    is_reader: bool,
) -> None:
    from dlightrag.adapters.postgres.corpus import corpus_bm25 as module

    instance = SimpleNamespace(
        ensure_indexes=AsyncMock(),
        verify_indexes=AsyncMock(),
        search_profile=AsyncMock(return_value=[]),
    )
    constructor = MagicMock(return_value=instance)
    monkeypatch.setattr(module, "PGBM25ProfileSearch", constructor)
    profiles = (BM25_PROFILE_FALLBACK,)
    config = _bm25_config(enabled=True, is_reader=is_reader)
    config.deployment.workspace = "research"
    config.corpus.retrieval.bm25_k1 = 1.4
    config.corpus.retrieval.bm25_b = 0.65

    result = await create_postgres_bm25(config, profiles=profiles)

    assert isinstance(result, ProfiledBM25Search)
    await result.search("query", scope=None)
    instance.search_profile.assert_awaited_once()
    constructor.assert_called_once_with(
        workspace="research",
        profiles=profiles,
    )
    expected = instance.verify_indexes if is_reader else instance.ensure_indexes
    unexpected = instance.ensure_indexes if is_reader else instance.verify_indexes
    expected.assert_awaited_once_with(k1=1.4, b=0.65)
    unexpected.assert_not_awaited()


async def test_rebuild_postgres_bm25_provisions_then_relabels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.adapters.postgres.corpus import corpus_bm25 as module

    adapter = SimpleNamespace(
        relabel_chunk_languages=AsyncMock(return_value={"processed_chunks": 3, "updated_chunks": 3})
    )
    provision = AsyncMock(return_value=(adapter, (BM25_PROFILE_FALLBACK,)))
    monkeypatch.setattr(module, "_provision_postgres_bm25", provision)
    config = _bm25_config(enabled=True, is_reader=False)
    stats = await rebuild_postgres_bm25(
        config,
        batch_size=25,
    )

    assert stats == {"processed_chunks": 3, "updated_chunks": 3}
    provision.assert_awaited_once_with(config)
    assert adapter.relabel_chunk_languages.await_count == 1
    assert callable(adapter.relabel_chunk_languages.await_args.args[0])
    assert adapter.relabel_chunk_languages.await_args.kwargs == {"batch_size": 25}


async def test_rebuild_postgres_bm25_skips_when_disabled() -> None:
    config = _bm25_config(enabled=False)

    assert await rebuild_postgres_bm25(config) == {
        "processed_chunks": 0,
        "updated_chunks": 0,
    }


async def test_rebuild_postgres_bm25_rejects_reader_role() -> None:
    config = _bm25_config(enabled=True, is_reader=True)

    with pytest.raises(RuntimeError, match="writer service role"):
        await rebuild_postgres_bm25(config)


def test_bm25_index_options_reject_unsafe_text_config() -> None:
    with pytest.raises(ValueError, match="unsafe BM25 text_config"):
        BM25IndexOptions(
            profile=BM25Profile(
                name="en",
                text_config="english'; DROP TABLE x; --",
                languages=("en",),
            )
        ).create_index_sql()


def test_bm25_index_options_match_real_pg_indexdef_format() -> None:
    profile = BM25Profile(name="en", text_config="english", languages=("en",))
    options = BM25IndexOptions(profile=profile, k1=1.2, b=0.75)

    assert options.matches_indexdef(
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_en ON public.lightrag_doc_chunks "
        "USING bm25 (content) WITH (text_config=english, k1='1.2', b='0.75') "
        "WHERE ((dlightrag_bm25_language)::text = 'en'::text)"
    )


def test_bm25_index_options_keeps_simple_fallback_full_table() -> None:
    options = BM25IndexOptions(profile=BM25_PROFILE_FALLBACK)

    assert options.create_index_sql() == (
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_simple "
        "ON LIGHTRAG_DOC_CHUNKS USING bm25(content) "
        "WITH (text_config='simple', k1=1.2, b=0.75)"
    )


async def test_bm25_search_empty_candidate_set_short_circuits() -> None:
    searcher = SimpleNamespace(search_profile=AsyncMock())
    bm25 = _profiled_bm25(searcher)

    assert await bm25.search("query", scope=_scope(doc_exists=False, candidate_count=0)) == []
    searcher.search_profile.assert_not_awaited()


async def test_unscoped_bm25_reports_bounded_visibility_pushdown() -> None:
    searcher = SimpleNamespace(search_profile=AsyncMock(return_value=[]))
    bm25 = _profiled_bm25(searcher)

    async with metadata_filter_scope(None) as stats:
        await bm25.search("query", scope=None)

    assert stats.visibility_strategy == "bounded_pushdown"


async def test_bm25_search_maps_rows() -> None:
    conn = AsyncMock()
    conn.fetch.return_value = [
        {
            "id": "chunk-a",
            "content": "hello world",
            "file_path": "a.md",
            "full_doc_id": "doc-a",
            "score": 1.5,
        }
    ]
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    bm25 = PGBM25ProfileSearch(
        pool=pool,
        workspace="default",
        profiles=(BM25Profile(name="en", text_config="english", fallback=True),),
    )

    rows = await bm25.search_profile(
        "hello",
        profile_name="en",
        language=None,
        scope=_scope(),
        limit=3,
    )

    args = conn.fetch.await_args.args
    assert args[1] == "hello"
    assert args[2] == "default"
    assert args[3] == "default"  # the bound metadata-side workspace
    assert args[4] == "x.pdf"
    assert args[5] == 3
    assert "full_doc_id IN (SELECT doc_id FROM dlightrag_doc_metadata" in args[0]
    assert rows == [
        {
            "chunk_id": "chunk-a",
            "content": "hello world",
            "file_path": "a.md",
            "full_doc_id": "doc-a",
            "bm25_profile": "en",
            "score": 1.5,
        }
    ]


async def test_bm25_search_uses_default_pool_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = AsyncMock()
    conn.fetch.return_value = []

    async def run(_self: PGPool, operation: Any) -> Any:
        return await operation(conn)

    monkeypatch.setattr(PGPool, "run", run)
    bm25 = PGBM25ProfileSearch(
        workspace="default",
        profiles=(BM25_PROFILE_FALLBACK,),
    )

    assert (
        await bm25.search_profile(
            "hello",
            profile_name="simple",
            language=None,
            scope=None,
            limit=3,
        )
        == []
    )
    conn.fetch.assert_awaited_once()


async def test_bm25_relabels_existing_chunks_in_workspace_batches() -> None:
    conn = AsyncMock()
    conn.fetch.side_effect = [
        [
            {"id": "chunk-a", "content": "hello"},
            {"id": "chunk-b", "content": "现金流"},
        ],
        [{"id": "chunk-c", "content": "ambiguous"}],
    ]
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    bm25 = PGBM25ProfileSearch(
        pool=pool,
        workspace="research",
        profiles=(
            BM25Profile(name="en", text_config="english", languages=("en",)),
            BM25_PROFILE_FALLBACK,
        ),
    )
    classify = MagicMock(side_effect=["en", "simple", "simple"])

    stats = await bm25.relabel_chunk_languages(classify, batch_size=2)

    assert stats == {"processed_chunks": 3, "updated_chunks": 3}
    assert [call.args[1:] for call in conn.fetch.await_args_list] == [
        ("research", "", 2),
        ("research", "chunk-b", 2),
    ]
    assert "ORDER BY id" in conn.fetch.await_args_list[0].args[0]
    assert [call.args[1:] for call in conn.execute.await_args_list] == [
        ("research", ["chunk-a", "chunk-b"], ["en", "simple"]),
        ("research", ["chunk-c"], ["simple"]),
    ]
    assert "FROM UNNEST" in conn.execute.await_args_list[0].args[0]


async def test_bm25_relabel_rejects_non_positive_batch_size() -> None:
    bm25 = PGBM25ProfileSearch(
        pool=AsyncMock(),
        workspace="default",
        profiles=(BM25_PROFILE_FALLBACK,),
    )

    with pytest.raises(ValueError, match="batch_size must be positive"):
        await bm25.relabel_chunk_languages(lambda _text: "simple", batch_size=0)


async def test_bm25_catalog_queries_use_shared_chunk_table_identity() -> None:
    conn = AsyncMock()
    conn.fetchval.return_value = 1
    conn.fetch.return_value = []

    await PGBM25ProfileSearch._verify_schema(conn)
    await PGBM25ProfileSearch._drop_stale_indexes(conn, set())

    assert conn.fetchval.await_args.args[1:] == (
        "lightrag_doc_chunks",
        BM25_LANGUAGE_COLUMN,
    )
    assert conn.fetch.await_args.args[1:] == (
        "lightrag_doc_chunks",
        "idx_lightrag_doc_chunks_bm25%",
    )


async def test_bm25_search_logs_profile_routing_and_results(
    caplog: pytest.LogCaptureFixture,
) -> None:
    searcher = SimpleNamespace(
        search_profile=AsyncMock(
            return_value=[
                {
                    "chunk_id": "chunk-a",
                    "content": "hello world",
                    "file_path": "a.md",
                    "bm25_profile": "en",
                    "score": 1.5,
                }
            ]
        )
    )
    bm25 = _profiled_bm25(
        searcher,
        top_k=3,
        profiles=(BM25Profile(name="en", text_config="english", fallback=True),),
    )

    with caplog.at_level(logging.INFO, logger="dlightrag.engine.rag.retrieval.bm25"):
        await bm25.search("hello", scope=_scope())

    assert "[BM25] search" in caplog.text
    assert "workspace=default" in caplog.text
    assert "query='hello'" in caplog.text
    assert "profiles=en" in caplog.text
    assert "candidate_scope=12chunk" in caplog.text
    assert "returned=1" in caplog.text
    assert "top=chunk-a:en:1.500" in caplog.text


async def test_bm25_ensure_index_rebuilds_when_options_change() -> None:
    conn = AsyncMock()
    conn.fetchval.side_effect = [
        1,
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_en ON public.lightrag_doc_chunks "
        "USING bm25 (content) WITH (text_config=english, k1=1.2, b=0.75) "
        "WHERE ((dlightrag_bm25_language)::text = 'en'::text)",
        1,
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_simple ON public.lightrag_doc_chunks "
        "USING bm25 (content) WITH (text_config=simple, k1=1.4, b=0.65)",
    ]
    conn.fetch.return_value = []
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    profile = BM25Profile(name="en", text_config="english", languages=("en",))
    bm25 = PGBM25ProfileSearch(
        pool=pool,
        workspace="default",
        profiles=(profile, BM25_PROFILE_FALLBACK),
    )

    await bm25.ensure_indexes(k1=1.4, b=0.65)

    executed = [call.args[0] for call in conn.execute.await_args_list]
    assert "ALTER TABLE LIGHTRAG_DOC_CHUNKS ADD COLUMN IF NOT EXISTS " in executed[0]
    assert BM25_LANGUAGE_COLUMN in executed[0]
    assert "DROP INDEX IF EXISTS idx_lightrag_doc_chunks_bm25_en" in executed
    assert (
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_en "
        "ON LIGHTRAG_DOC_CHUNKS USING bm25(content) "
        "WITH (text_config='english', k1=1.4, b=0.65) "
        "WHERE dlightrag_bm25_language = 'en'"
    ) in executed


async def test_bm25_ensure_index_keeps_matching_index() -> None:
    conn = AsyncMock()
    conn.fetchval.side_effect = [
        1,
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_en ON public.lightrag_doc_chunks "
        "USING bm25 (content) WITH (text_config=english, k1=1.4, b=0.65) "
        "WHERE ((dlightrag_bm25_language)::text = 'en'::text)",
        1,
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_simple ON public.lightrag_doc_chunks "
        "USING bm25 (content) WITH (text_config=simple, k1=1.4, b=0.65)",
    ]
    conn.fetch.return_value = []
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    profile = BM25Profile(name="en", text_config="english", languages=("en",))
    bm25 = PGBM25ProfileSearch(
        pool=pool,
        workspace="default",
        profiles=(profile, BM25_PROFILE_FALLBACK),
    )

    await bm25.ensure_indexes(k1=1.4, b=0.65)

    executed = [call.args[0] for call in conn.execute.await_args_list]
    assert len(executed) == 3
    assert executed[0].startswith("ALTER TABLE LIGHTRAG_DOC_CHUNKS ADD COLUMN IF NOT EXISTS")
    assert executed[1].startswith(
        "CREATE INDEX IF NOT EXISTS idx_lightrag_doc_chunks_dlightrag_bm25_language"
    )
    assert executed[2].startswith(
        "CREATE INDEX IF NOT EXISTS idx_lightrag_doc_chunks_dlightrag_full_doc_id"
    )


async def test_bm25_ensure_index_verifies_qualified_text_config_by_schema() -> None:
    conn = AsyncMock()
    conn.fetchval.side_effect = [1, None]
    conn.fetch.return_value = []
    pool = MagicMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    profile = BM25Profile(name="zh", text_config="public.jiebacfg", fallback=True)
    bm25 = PGBM25ProfileSearch(pool=pool, workspace="default", profiles=(profile,))

    await bm25.ensure_indexes()

    first_fetch = conn.fetchval.await_args_list[0]
    assert first_fetch.args[1:] == ("public", "jiebacfg")
    executed = [call.args[0] for call in conn.execute.await_args_list]
    assert (
        "CREATE INDEX idx_lightrag_doc_chunks_bm25_zh "
        "ON LIGHTRAG_DOC_CHUNKS USING bm25(content) "
        "WITH (text_config='public.jiebacfg', k1=1.2, b=0.75)"
    ) in executed


async def test_bm25_routes_chinese_query_to_jieba_profile_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BM25LanguageClassifier, "detect", lambda *_args: "zh")
    searcher = SimpleNamespace(
        search_profile=AsyncMock(
            return_value=[
                {
                    "chunk_id": "zh-hit",
                    "content": "现金流",
                    "file_path": "cn.md",
                    "bm25_profile": "zh",
                    "score": 2.0,
                }
            ]
        )
    )
    bm25 = _profiled_bm25(
        searcher,
        profiles=(
            BM25Profile(name="zh", text_config="public.jiebacfg", languages=("zh",)),
            BM25Profile(name="en", text_config="english", languages=("en",)),
            BM25_PROFILE_FALLBACK,
        ),
    )

    rows = await bm25.search("现金流", scope=None, top_k=5)

    assert searcher.search_profile.await_count == 1
    assert searcher.search_profile.await_args.kwargs == {
        "profile_name": "zh",
        "language": "zh",
        "scope": None,
        "limit": 5,
    }
    assert {row["chunk_id"] for row in rows} == {"zh-hit"}


def test_bm25_profile_rejects_unsafe_language_code() -> None:
    with pytest.raises(ValueError, match="unsafe BM25 language code"):
        BM25Profile(name="bad", text_config="simple", languages=("en';drop",))


async def test_bm25_routes_configured_language_to_matching_profile_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BM25LanguageClassifier, "detect", lambda *_args: "de")
    searcher = SimpleNamespace(
        search_profile=AsyncMock(
            return_value=[{"chunk_id": "de-hit", "content": "Umsatz", "score": 2.0}]
        )
    )
    bm25 = _profiled_bm25(
        searcher,
        profiles=(
            BM25Profile(name="de", text_config="german", languages=("de",)),
            BM25Profile(name="en", text_config="english", languages=("en",)),
            BM25_PROFILE_FALLBACK,
        ),
    )

    rows = await bm25.search("Wie hoch ist der Umsatz im letzten Quartal?", scope=None)

    assert searcher.search_profile.await_count == 1
    assert searcher.search_profile.await_args.kwargs["profile_name"] == "de"
    assert searcher.search_profile.await_args.kwargs["language"] == "de"
    assert {row["chunk_id"] for row in rows} == {"de-hit"}


async def test_bm25_routes_region_language_tag_to_profile_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BM25LanguageClassifier, "detect", lambda *_args: "de")
    searcher = SimpleNamespace(
        search_profile=AsyncMock(
            return_value=[{"chunk_id": "de-hit", "content": "Umsatz", "score": 2.0}]
        )
    )
    bm25 = _profiled_bm25(
        searcher,
        profiles=(
            BM25Profile(name="de", text_config="german", languages=("de-DE",)),
            BM25Profile(name="en", text_config="english", languages=("en-US",)),
            BM25_PROFILE_FALLBACK,
        ),
    )

    rows = await bm25.search("Wie hoch ist der Umsatz im letzten Quartal?", scope=None)

    assert searcher.search_profile.await_count == 1
    assert searcher.search_profile.await_args.kwargs["profile_name"] == "de"
    assert searcher.search_profile.await_args.kwargs["language"] == "de"
    assert {row["chunk_id"] for row in rows} == {"de-hit"}


async def test_bm25_routes_unknown_language_to_simple_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    searcher = SimpleNamespace(
        search_profile=AsyncMock(
            return_value=[{"chunk_id": "simple-hit", "content": "query", "score": 1.0}]
        )
    )
    bm25 = _profiled_bm25(
        searcher,
        profiles=(
            BM25Profile(name="de", text_config="german", languages=("de",)),
            BM25Profile(name="en", text_config="english", languages=("en",)),
            BM25_PROFILE_FALLBACK,
        ),
    )
    monkeypatch.setattr(
        BM25LanguageClassifier,
        "detect",
        lambda *_args, **_kwargs: "simple",
    )

    rows = await bm25.search("unsupported", scope=None)

    assert searcher.search_profile.await_count == 1
    assert searcher.search_profile.await_args.kwargs["profile_name"] == "simple"
    assert searcher.search_profile.await_args.kwargs["language"] is None
    assert {row["chunk_id"] for row in rows} == {"simple-hit"}
