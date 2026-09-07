# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL implementation of language-profiled BM25 corpus search."""

import asyncio
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from dlightrag.adapters.postgres.core._operations import ConnectionPool, PostgresOperationRunner
from dlightrag.adapters.postgres.core.identifiers import pg_identifier, pg_qualified_identifier
from dlightrag.adapters.postgres.corpus._corpus_schema import (
    BM25_LANGUAGE_COLUMN,
    CHUNK_DOCUMENT_SCOPE_INDEX,
    LIGHTRAG_CHUNKS_TABLE,
)
from dlightrag.adapters.postgres.corpus.corpus_languages import update_chunk_bm25_languages
from dlightrag.adapters.postgres.corpus.pg_metadata_index import (
    METADATA_TABLE,
    metadata_match_conditions,
    metadata_visibility_condition,
)
from dlightrag.engine.rag.retrieval import ContextRow, MetadataScope
from dlightrag.engine.rag.retrieval.bm25 import (
    BM25_PROFILE_FALLBACK,
    BM25Profile,
    ProfiledBM25Search,
    profile_languages,
    profiles_from_config,
)
from dlightrag.engine.rag.retrieval.language import (
    BM25_FALLBACK_LANGUAGE,
    BM25LanguageClassifier,
    normalize_language_code,
)
from dlightrag.engine.rag.retrieval.visibility import bounded_visibility_candidate_limit

BM25_INDEX_PREFIX = pg_identifier("idx_lightrag_doc_chunks_bm25")
BM25_LANGUAGE_INDEX = pg_identifier("idx_lightrag_doc_chunks_dlightrag_bm25_language")
BM25_DOC_INDEX = pg_identifier(CHUNK_DOCUMENT_SCOPE_INDEX)
BM25_TABLE = pg_identifier(LIGHTRAG_CHUNKS_TABLE)
_VERIFY_SCHEMA_SQL = (
    "SELECT 1 FROM information_schema.columns WHERE table_name = $1 AND column_name = $2 LIMIT 1"
)
_STALE_INDEXES_SQL = "SELECT indexname FROM pg_indexes WHERE tablename = $1 AND indexname LIKE $2"


def _format_float(value: float) -> str:
    return f"{float(value):g}"


def _validated_language_code(language: str) -> str:
    return pg_identifier(normalize_language_code(language))


def validate_profile_text_config(text_config: str) -> str:
    value = str(text_config).strip()
    try:
        return pg_qualified_identifier(value)
    except ValueError as exc:
        raise ValueError(f"unsafe BM25 text_config: {text_config!r}") from exc


def _index_name(profile_name: str) -> str:
    return pg_identifier(f"{BM25_INDEX_PREFIX}_{pg_identifier(profile_name)}")


def required_postgres_extensions(profiles: Iterable[BM25Profile]) -> tuple[str, ...]:
    extensions = ["pg_textsearch"]
    if any(profile.text_config == "public.jiebacfg" for profile in profiles):
        extensions.append("pg_jieba")
    return tuple(extensions)


@dataclass(frozen=True)
class BM25IndexOptions:
    profile: BM25Profile = BM25_PROFILE_FALLBACK
    k1: float = 1.2
    b: float = 0.75

    def __post_init__(self) -> None:
        if self.k1 <= 0:
            raise ValueError("BM25 k1 must be positive")
        if not 0 <= self.b <= 1:
            raise ValueError("BM25 b must be between 0 and 1")

    def create_index_sql(self) -> str:
        index_name = _index_name(self.profile.name)
        text_config = validate_profile_text_config(self.profile.text_config)
        sql = (
            f"CREATE INDEX {index_name} ON {BM25_TABLE} USING bm25(content) "
            f"WITH (text_config='{text_config}', "
            f"k1={_format_float(self.k1)}, b={_format_float(self.b)})"
        )
        if self.profile.language_bucket is not None:
            sql += (
                f" WHERE {BM25_LANGUAGE_COLUMN} = "
                f"'{_validated_language_code(self.profile.language_bucket)}'"
            )
        return sql

    def matches_indexdef(self, indexdef: str | None) -> bool:
        if not indexdef:
            return False
        normalized = re.sub(r"\s+", "", indexdef.lower().replace('"', "").replace("'", ""))
        text_config = validate_profile_text_config(self.profile.text_config).lower()
        index_name = _index_name(self.profile.name)
        matches = (
            "usingbm25(content)" in normalized
            and index_name.lower() in normalized
            and (
                f"text_config={text_config}" in normalized
                or f"text_config={text_config}::regconfig" in normalized
            )
            and f"k1={_format_float(self.k1)}" in normalized
            and f"b={_format_float(self.b)}" in normalized
        )
        if not matches:
            return False
        if self.profile.language_bucket is None:
            return "where" not in normalized
        return (
            "where" in normalized
            and BM25_LANGUAGE_COLUMN.lower() in normalized
            and self.profile.language_bucket.lower() in normalized
        )


def build_bm25_sql(
    *,
    index_name: str,
    scoped: bool,
    limit: int,
    language: str | None = None,
    metadata_conditions: tuple[str, ...] = (),
) -> str:
    safe_index = pg_identifier(index_name)
    limit_value = int(limit)
    if limit_value < 1:
        raise ValueError("BM25 limit must be positive")
    if scoped and not metadata_conditions:
        raise ValueError("scoped BM25 requires metadata conditions")
    language_clause = (
        f"AND {BM25_LANGUAGE_COLUMN} = '{_validated_language_code(language)}'" if language else ""
    )
    if scoped:
        # One selective user-filter semi-join. The shared conditions also carry
        # the publication marker without adding a bound parameter.
        candidate_clause = (
            f"AND full_doc_id IN (SELECT doc_id FROM {METADATA_TABLE} "  # noqa: S608
            f"WHERE {' AND '.join(metadata_conditions)})"
        )
        highest_placeholder = max(
            (
                int(slot)
                for condition in metadata_conditions
                for slot in re.findall(r"\$(\d+)", condition)
            ),
            default=2,
        )
        limit_placeholder = f"${highest_placeholder + 1}"
        return (
            f"SELECT id, content, file_path, full_doc_id, "  # noqa: S608
            f"-(content <@> to_bm25query($1, '{safe_index}')) AS score "
            f"FROM {BM25_TABLE} "
            "WHERE workspace = $2 "
            f"{candidate_clause} "
            f"{language_clause} "
            f"ORDER BY content <@> to_bm25query($1, '{safe_index}') "
            f"LIMIT {limit_placeholder}"
        )

    # With no user filter, never form the all-visible document-id set. Rank a
    # bounded lexical window first and correlate only those rows to metadata.
    return (
        f"WITH ranked AS MATERIALIZED ("  # noqa: S608
        f"SELECT id, workspace, content, file_path, full_doc_id, "
        f"-(content <@> to_bm25query($1, '{safe_index}')) AS score, "
        f"content <@> to_bm25query($1, '{safe_index}') AS rank_distance "
        f"FROM {BM25_TABLE} "
        "WHERE workspace = $2 "
        f"{language_clause} "
        f"ORDER BY content <@> to_bm25query($1, '{safe_index}') LIMIT $3"
        ") "
        "SELECT r.id, r.content, r.file_path, r.full_doc_id, r.score "
        "FROM ranked r WHERE EXISTS ("
        f"SELECT 1 FROM {METADATA_TABLE} m "
        "WHERE m.workspace = r.workspace AND m.doc_id = r.full_doc_id "
        f"AND {metadata_visibility_condition('m')}"
        ") ORDER BY r.rank_distance LIMIT $4"
    )


class PGBM25ProfileSearch(PostgresOperationRunner):
    """Own pg_textsearch SQL, schema verification, and chunk relabeling."""

    def __init__(
        self,
        *,
        pool: ConnectionPool | None = None,
        workspace: str,
        profiles: tuple[BM25Profile, ...],
    ) -> None:
        super().__init__(pool=pool)
        self._workspace = workspace
        self._profiles = profiles

    async def ensure_indexes(self, *, k1: float = 1.2, b: float = 0.75) -> None:
        options_by_profile = [
            BM25IndexOptions(profile=profile, k1=k1, b=b) for profile in self._profiles
        ]

        async def operation(conn: Any) -> None:
            await self._ensure_schema(conn)
            for options in options_by_profile:
                await self._verify_text_config(conn, options.profile.text_config)
                index_name = _index_name(options.profile.name)
                indexdef = await self._fetch_indexdef(conn, index_name)
                if options.matches_indexdef(indexdef):
                    continue
                if indexdef:
                    await conn.execute(f"DROP INDEX IF EXISTS {index_name}")
                await conn.execute(options.create_index_sql())
            await self._drop_stale_indexes(
                conn,
                {_index_name(option.profile.name) for option in options_by_profile},
            )

        await self._run(operation)

    async def verify_indexes(self, *, k1: float = 1.2, b: float = 0.75) -> None:
        options_by_profile = [
            BM25IndexOptions(profile=profile, k1=k1, b=b) for profile in self._profiles
        ]

        async def operation(conn: Any) -> None:
            await self._verify_schema(conn)
            for options in options_by_profile:
                await self._verify_text_config(conn, options.profile.text_config)
                index_name = _index_name(options.profile.name)
                indexdef = await self._fetch_indexdef(conn, index_name)
                if not options.matches_indexdef(indexdef):
                    raise RuntimeError(
                        f"BM25 index {index_name} is missing or does not match configured "
                        "options; initialize it on the writer first"
                    )

        await self._run(operation)

    async def search_profile(
        self,
        query: str,
        *,
        profile_name: str,
        language: str | None,
        scope: MetadataScope | None,
        limit: int,
    ) -> list[ContextRow]:
        if scope is not None:
            conditions, params = metadata_match_conditions(
                self._workspace,
                scope.filters,
                filename_mode=scope.filename_mode,
                start_index=3,
            )
            sql = build_bm25_sql(
                index_name=_index_name(profile_name),
                scoped=True,
                limit=limit,
                language=language,
                metadata_conditions=tuple(conditions),
            )

            async def operation(conn: Any) -> list[Any]:
                return await conn.fetch(sql, query, self._workspace, *params, int(limit))

        else:
            sql = build_bm25_sql(
                index_name=_index_name(profile_name),
                scoped=False,
                limit=limit,
                language=language,
            )

            async def operation(conn: Any) -> list[Any]:
                return await conn.fetch(
                    sql,
                    query,
                    self._workspace,
                    bounded_visibility_candidate_limit(limit),
                    int(limit),
                )

        rows = await self._run(operation)
        return [self._row_to_chunk(row, profile_name=profile_name) for row in rows]

    async def relabel_chunk_languages(
        self,
        classify: Callable[[str], str],
        *,
        batch_size: int = 500,
    ) -> dict[str, int]:
        batch_limit = int(batch_size)
        if batch_limit < 1:
            raise ValueError("BM25 batch_size must be positive")

        async def operation(conn: Any) -> dict[str, int]:
            stats = {"processed_chunks": 0, "updated_chunks": 0}
            cursor = ""
            while True:
                rows = await conn.fetch(
                    f"""
                    SELECT id, content
                    FROM {BM25_TABLE}
                    WHERE workspace = $1 AND id > $2
                    ORDER BY id
                    LIMIT $3
                    """,  # noqa: S608 - private table constant.
                    self._workspace,
                    cursor,
                    batch_limit,
                )
                if not rows:
                    break
                chunk_ids = [str(row["id"]) for row in rows]
                contents = [str(row["content"] or "") for row in rows]
                languages = await asyncio.to_thread(
                    lambda texts=contents: [classify(text) for text in texts]
                )
                updated = await update_chunk_bm25_languages(
                    conn,
                    workspace=self._workspace,
                    labels=dict(zip(chunk_ids, languages, strict=True)),
                )
                stats["processed_chunks"] += len(rows)
                stats["updated_chunks"] += updated
                cursor = chunk_ids[-1]
                if len(rows) < batch_limit:
                    break
            return stats

        return await self._run(operation)

    @staticmethod
    async def _verify_schema(conn: Any) -> None:
        exists = await conn.fetchval(_VERIFY_SCHEMA_SQL, BM25_TABLE.lower(), BM25_LANGUAGE_COLUMN)
        if not exists:
            raise RuntimeError(
                f"{BM25_TABLE}.{BM25_LANGUAGE_COLUMN} is missing; initialize it on the writer first"
            )

    @staticmethod
    async def _ensure_schema(conn: Any) -> None:
        await conn.execute(
            f"ALTER TABLE {BM25_TABLE} ADD COLUMN IF NOT EXISTS "
            f"{BM25_LANGUAGE_COLUMN} TEXT NOT NULL DEFAULT '{BM25_FALLBACK_LANGUAGE}'"
        )
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS {BM25_LANGUAGE_INDEX} "
            f"ON {BM25_TABLE}(workspace, {BM25_LANGUAGE_COLUMN})"
        )
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS {BM25_DOC_INDEX} ON {BM25_TABLE}(workspace, full_doc_id)"
        )

    @staticmethod
    async def _verify_text_config(conn: Any, text_config: str) -> None:
        safe_config = validate_profile_text_config(text_config)
        if "." in safe_config:
            schema, name = safe_config.split(".", maxsplit=1)
            exists = await conn.fetchval(
                """
                SELECT 1
                FROM pg_ts_config c
                JOIN pg_namespace n ON n.oid = c.cfgnamespace
                WHERE n.nspname = $1 AND c.cfgname = $2
                LIMIT 1
                """,
                schema,
                name,
            )
        else:
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_ts_config WHERE cfgname = $1 LIMIT 1",
                safe_config,
            )
        if not exists:
            raise RuntimeError(f"PostgreSQL text search config {safe_config!r} is missing")

    @staticmethod
    async def _fetch_indexdef(conn: Any, index_name: str) -> str | None:
        return await conn.fetchval(
            "SELECT indexdef FROM pg_indexes WHERE indexname = $1", index_name
        )

    @staticmethod
    async def _drop_stale_indexes(conn: Any, desired_indexes: set[str]) -> None:
        rows = await conn.fetch(
            _STALE_INDEXES_SQL,
            BM25_TABLE.lower(),
            f"{BM25_INDEX_PREFIX}%",
        )
        for row in rows:
            index_name = str(row["indexname"])
            if index_name not in desired_indexes:
                await conn.execute(f"DROP INDEX IF EXISTS {pg_identifier(index_name)}")

    @staticmethod
    def _row_to_chunk(row: Any, *, profile_name: str) -> ContextRow:
        chunk = {
            "chunk_id": row["id"],
            "content": row["content"],
            "file_path": row["file_path"],
            "bm25_profile": profile_name,
            "score": float(row["score"]),
        }
        if row.get("full_doc_id"):
            chunk["full_doc_id"] = row["full_doc_id"]
        return chunk


async def _provision_postgres_bm25(
    config: Any,
    *,
    profiles: tuple[BM25Profile, ...] | None = None,
) -> tuple[PGBM25ProfileSearch, tuple[BM25Profile, ...]] | None:
    if not config.corpus.retrieval.bm25_enabled:
        return None
    runtime_profiles = profiles or profiles_from_config(config.corpus.retrieval.bm25_profiles)
    adapter = PGBM25ProfileSearch(
        workspace=config.deployment.workspace,
        profiles=runtime_profiles,
    )
    if config.is_reader:
        await adapter.verify_indexes(
            k1=config.corpus.retrieval.bm25_k1, b=config.corpus.retrieval.bm25_b
        )
    else:
        await adapter.ensure_indexes(
            k1=config.corpus.retrieval.bm25_k1, b=config.corpus.retrieval.bm25_b
        )
    return adapter, runtime_profiles


async def create_postgres_bm25(
    config: Any,
    *,
    profiles: tuple[BM25Profile, ...] | None = None,
) -> ProfiledBM25Search | None:
    provisioned = await _provision_postgres_bm25(config, profiles=profiles)
    if provisioned is None:
        return None
    adapter, runtime_profiles = provisioned
    return ProfiledBM25Search(
        adapter,
        workspace=config.deployment.workspace,
        profiles=runtime_profiles,
    )


async def rebuild_postgres_bm25(
    config: Any,
    *,
    batch_size: int = 500,
) -> dict[str, int]:
    if config.corpus.retrieval.bm25_enabled and config.is_reader:
        raise RuntimeError("BM25 rebuild requires the writer service role")
    provisioned = await _provision_postgres_bm25(config)
    if provisioned is None:
        return {"processed_chunks": 0, "updated_chunks": 0}
    adapter, profiles = provisioned
    classifier = BM25LanguageClassifier(profile_languages(profiles))
    return await adapter.relabel_chunk_languages(classifier.detect, batch_size=batch_size)


__all__ = [
    "BM25IndexOptions",
    "BM25_LANGUAGE_COLUMN",
    "PGBM25ProfileSearch",
    "build_bm25_sql",
    "create_postgres_bm25",
    "rebuild_postgres_bm25",
    "required_postgres_extensions",
]
