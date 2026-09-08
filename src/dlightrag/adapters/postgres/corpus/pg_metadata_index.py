# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL-backed document metadata index for structured queries."""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from dlightrag.adapters.postgres.core._migrations import (
    Migration,
    TableRequirement,
    apply_migrations,
    verify_migrations,
)
from dlightrag.adapters.postgres.core._operations import PostgresOperationRunner
from dlightrag.adapters.postgres.core.identifiers import pg_identifier
from dlightrag.adapters.postgres.corpus.partition_foundation import (
    PartitionedTableSpec,
    default_child_name,
    ensure_partitioned_tables,
    verify_partitioned_tables,
)
from dlightrag.engine.rag.retrieval import MetadataFilter, MetadataScope
from dlightrag.engine.rag.retrieval.metadata_fields import (
    FILTER_FIELD_COLUMNS,
    INGEST_FINALIZATION_COMPLETE_FIELD,
    METADATA_FIELD_IDS,
)
from dlightrag.engine.rag.workspace.ports import CorpusSchemaError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _PGMetadataColumn:
    field_id: str
    pg_type: str
    indexed: bool = False


_PG_FIELD_TYPES = {
    "filename": "VARCHAR(512)",
    "filename_stem": "VARCHAR(512)",
    "source_uri": "TEXT",
    "download_locator": "TEXT",
    "file_extension": "VARCHAR(32)",
    "title": "TEXT",
    "author": "VARCHAR(255)",
    # RAG normalizes values to naive UTC before binding.
    "creation_date": "TIMESTAMP",
    "ingested_at": "TIMESTAMPTZ DEFAULT NOW()",
    "custom_metadata": "JSONB DEFAULT '{}'",
}
_PG_INDEXED_FIELDS = frozenset(
    {"filename", "filename_stem", "file_extension", "title", "author", "creation_date"}
)
# Application-owned crash journal. It is deliberately outside the public RAG
# metadata registry: callers cannot filter on or supply it as custom metadata.
_FINALIZATION_COMPLETE_COLUMN = INGEST_FINALIZATION_COMPLETE_FIELD
if set(_PG_FIELD_TYPES) != set(METADATA_FIELD_IDS):
    raise RuntimeError("PostgreSQL metadata columns do not match the RAG metadata field registry")
_PG_METADATA_COLUMNS = tuple(
    _PGMetadataColumn(
        field_id,
        _PG_FIELD_TYPES[field_id],
        indexed=field_id in _PG_INDEXED_FIELDS,
    )
    for field_id in METADATA_FIELD_IDS
)
_FILTERABLE_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(column for columns in FILTER_FIELD_COLUMNS.values() for column in columns)
)


def _build_create_table() -> str:
    cols = [
        "workspace       VARCHAR(255) NOT NULL",
        "doc_id          VARCHAR(255) NOT NULL",
    ]
    for f in _PG_METADATA_COLUMNS:
        cols.append(f"    {f.field_id}    {f.pg_type}")
    cols.append(f"    {_FINALIZATION_COMPLETE_COLUMN}    BOOLEAN NOT NULL DEFAULT FALSE")
    cols.append("    PRIMARY KEY (workspace, doc_id)")
    return (
        "CREATE TABLE IF NOT EXISTS dlightrag_doc_metadata (\n"
        + ",\n".join(cols)
        + "\n) PARTITION BY LIST (workspace)"
    )


_CREATE_TABLE = _build_create_table()


# One canonical comparison for every text match: neither case nor padding is a
# meaningful difference in a filter, and both sides must fold identically.
def _canonical(expr: str) -> str:
    return f"LOWER(TRIM({expr}))"


def _index_clause(field_id: str, pg_type: str, indexed: bool) -> str | None:
    if not indexed:
        return None
    if field_id == "title":
        # ``title`` is unbounded TEXT: a plain B-tree on the value can fail on
        # values larger than one index page (~2704 bytes). A fixed-width MD5 of
        # the canonical value keeps the index key bounded; the equality recheck
        # in ``metadata_match_conditions`` makes the match exact regardless of
        # hash collisions.
        return f" (workspace, MD5({_canonical(field_id)}))"
    if _is_string_pg_type(pg_type):
        return f" (workspace, {_canonical(field_id)})"
    return f" (workspace, {field_id})"


def _is_string_pg_type(pg_type: str) -> bool:
    normalized = pg_type.upper()
    return normalized.startswith(("TEXT", "VARCHAR", "CHAR", "CHARACTER"))


# The one canonicalization contract for ``custom_metadata_search``: top-level
# keys plus every scalar/text representation are folded to trimmed lowercase
# text. This intentionally preserves the existing ``custom_metadata ->> key``
# comparison semantics, where JSON numbers/booleans and equivalent strings
# compare identically. Both write paths and the later runtime ``@>`` predicate
# fold through this same immutable SQL function, so stored and bound values can
# never drift apart. ORDER BY resolves duplicate folded keys deterministically.
_CREATE_CANONICAL_CUSTOM_FN = """
CREATE OR REPLACE FUNCTION dlightrag_canonical_custom_metadata(meta jsonb)
RETURNS jsonb
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT COALESCE(jsonb_object_agg(
        lower(trim(key)),
        to_jsonb(lower(trim(COALESCE(value #>> '{}', 'null'))))
        ORDER BY lower(trim(key)), key
    ), '{}'::jsonb)
    FROM jsonb_each(COALESCE(meta, '{}'::jsonb))
$$
"""

_BACKFILL_CUSTOM_SEARCH = """
UPDATE dlightrag_doc_metadata
SET custom_metadata_search = dlightrag_canonical_custom_metadata(custom_metadata)
WHERE custom_metadata_search
      IS DISTINCT FROM dlightrag_canonical_custom_metadata(custom_metadata)
"""

# Rows written before the publication journal existed were completed under the
# legacy contract. Publish only rows LightRAG durably marked processed before
# the journal migration; post-migration interrupted commits remain hidden.
_PUBLISH_LEGACY_PROCESSED_DOCUMENTS = f"""
DO $$
BEGIN
    IF to_regclass('public.lightrag_doc_status') IS NOT NULL THEN
        UPDATE dlightrag_doc_metadata AS metadata
        SET {_FINALIZATION_COMPLETE_COLUMN} = TRUE
        WHERE metadata.{_FINALIZATION_COMPLETE_COLUMN} IS FALSE
          AND COALESCE(metadata.ingested_at, '-infinity'::timestamptz) <= (
              SELECT migration.applied_at
              FROM dlightrag_schema_migrations AS migration
              WHERE migration.scope = 'doc_metadata'
                AND migration.version = 'product_document_visibility'
          )
          AND EXISTS (
              SELECT 1
              FROM lightrag_doc_status AS status
              WHERE status.workspace = metadata.workspace
                AND status.id = metadata.doc_id
                AND status.status = 'processed'
          );
    END IF;
END
$$
"""  # noqa: S608 - fixed internal column

_METADATA_TABLE = "dlightrag_doc_metadata"
_SEARCH_COLUMN = "custom_metadata_search"

# Public identities the other corpus adapters reuse so every retrieval leg
# matches against the same table and column.
METADATA_TABLE = _METADATA_TABLE
METADATA_SEARCH_COLUMN = _SEARCH_COLUMN


def metadata_visibility_condition(alias: str | None = None) -> str:
    """Render the one fail-closed Product Document publication predicate."""
    prefix = f"{pg_identifier(alias)}." if alias else ""
    return f"{prefix}{_FINALIZATION_COMPLETE_COLUMN} IS TRUE"


_FIELD_STATS_TABLE = "dlightrag_metadata_field_stats"
_CUSTOM_SCHEMA_KEY_LIMIT = 128
_CREATE_FIELD_STATS_TABLE = f"""
CREATE TABLE IF NOT EXISTS {_FIELD_STATS_TABLE} (
    workspace       VARCHAR(255) NOT NULL,
    field_id        TEXT         NOT NULL,
    document_count  BIGINT       NOT NULL,
    PRIMARY KEY (workspace, field_id),
    CONSTRAINT dlightrag_metadata_field_stats_count_check
        CHECK (document_count >= 0)
)
"""


def _presence_rows(record: str) -> str:
    builtins = ",\n            ".join(
        f"('{column}', {record}.{column} IS NOT NULL)" for column in _FILTERABLE_COLUMNS
    )
    return f"""
        SELECT {record}.workspace AS workspace, fields.field_id
        FROM (VALUES
            {builtins}
        ) AS fields(field_id, present)
        WHERE fields.present
        UNION
        SELECT {record}.workspace, custom.key
        FROM jsonb_object_keys(
            COALESCE({record}.custom_metadata, '{{}}'::jsonb)
        ) AS custom(key)
    """  # noqa: S608 - record and columns are fixed migration SQL


def _presence_difference(left: str, right: str) -> str:
    return f"""
        SELECT workspace, field_id FROM ({_presence_rows(left)}) AS left_fields
        EXCEPT
        SELECT workspace, field_id FROM ({_presence_rows(right)}) AS right_fields
    """  # noqa: S608 - composes only fixed trigger fragments


def _increment_field_stats(rows: str) -> str:
    return f"""
        INSERT INTO {_FIELD_STATS_TABLE} (workspace, field_id, document_count)
        SELECT present.workspace, present.field_id, 1
        FROM ({rows}) AS present
        ORDER BY present.workspace, present.field_id
        ON CONFLICT (workspace, field_id) DO UPDATE
        SET document_count = {_FIELD_STATS_TABLE}.document_count + 1;
    """  # noqa: S608 - composes only fixed trigger fragments


def _decrement_field_stats(rows: str) -> str:
    return f"""
        INSERT INTO {_FIELD_STATS_TABLE} (workspace, field_id, document_count)
        SELECT removed.workspace, removed.field_id, 0
        FROM ({rows}) AS removed
        ORDER BY removed.workspace, removed.field_id
        ON CONFLICT (workspace, field_id) DO UPDATE
        SET document_count = {_FIELD_STATS_TABLE}.document_count - 1;

        DELETE FROM {_FIELD_STATS_TABLE} AS stats
        USING ({rows}) AS removed
        WHERE stats.workspace = removed.workspace
          AND stats.field_id = removed.field_id
          AND stats.document_count = 0;
    """  # noqa: S608 - composes only fixed trigger fragments


_CREATE_FIELD_STATS_TRIGGER_FN = f"""
CREATE OR REPLACE FUNCTION dlightrag_sync_metadata_field_stats()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.{_FINALIZATION_COMPLETE_COLUMN} IS TRUE THEN
            {_increment_field_stats(_presence_rows("NEW"))}
        END IF;
    ELSIF TG_OP = 'UPDATE' THEN
        IF OLD.{_FINALIZATION_COMPLETE_COLUMN} IS TRUE
           AND NEW.{_FINALIZATION_COMPLETE_COLUMN} IS TRUE THEN
            {_increment_field_stats(_presence_difference("NEW", "OLD"))}
            {_decrement_field_stats(_presence_difference("OLD", "NEW"))}
        ELSIF OLD.{_FINALIZATION_COMPLETE_COLUMN} IS TRUE THEN
            {_decrement_field_stats(_presence_rows("OLD"))}
        ELSIF NEW.{_FINALIZATION_COMPLETE_COLUMN} IS TRUE THEN
            {_increment_field_stats(_presence_rows("NEW"))}
        END IF;
    ELSE
        IF OLD.{_FINALIZATION_COMPLETE_COLUMN} IS TRUE THEN
            {_decrement_field_stats(_presence_rows("OLD"))}
        END IF;
    END IF;
    RETURN NULL;
END
$$
"""

_DROP_FIELD_STATS_TRIGGER = (
    "DROP TRIGGER IF EXISTS dlightrag_metadata_field_stats ON dlightrag_doc_metadata"
)
_CREATE_FIELD_STATS_TRIGGER = """
CREATE TRIGGER dlightrag_metadata_field_stats
AFTER INSERT OR UPDATE OR DELETE ON dlightrag_doc_metadata
FOR EACH ROW EXECUTE FUNCTION dlightrag_sync_metadata_field_stats()
"""


def _field_stats_source(where: str = "") -> str:
    builtins = ",\n                ".join(
        f"('{column}', metadata.{column} IS NOT NULL)" for column in _FILTERABLE_COLUMNS
    )
    predicate = f"WHERE {where}" if where else ""
    return f"""
        SELECT metadata.workspace, metadata.doc_id, fields.field_id
        FROM dlightrag_doc_metadata AS metadata
        CROSS JOIN LATERAL (VALUES
                {builtins}
        ) AS fields(field_id, present)
        WHERE fields.present {f"AND {where}" if where else ""}
        UNION
        SELECT metadata.workspace, metadata.doc_id, custom.key
        FROM dlightrag_doc_metadata AS metadata
        CROSS JOIN LATERAL jsonb_object_keys(
            COALESCE(metadata.custom_metadata, '{{}}'::jsonb)
        ) AS custom(key)
        {predicate}
    """  # noqa: S608 - predicate and columns are fixed migration SQL


def _backfill_field_stats(where: str = "") -> str:
    return f"""
INSERT INTO {_FIELD_STATS_TABLE} (workspace, field_id, document_count)
SELECT present.workspace, present.field_id, COUNT(*)::bigint
FROM ({_field_stats_source(where)}) AS present
GROUP BY present.workspace, present.field_id
"""  # noqa: S608 - composes only fixed backfill fragments


_VISIBLE_STATS_PREDICATE = metadata_visibility_condition("metadata")
_BACKFILL_FIELD_STATS = _backfill_field_stats(_VISIBLE_STATS_PREDICATE)
_BACKFILL_WORKSPACE_FIELD_STATS = _backfill_field_stats(
    f"metadata.workspace = $1 AND {_VISIBLE_STATS_PREDICATE}"
)


def _metadata_partition_spec() -> PartitionedTableSpec:
    return PartitionedTableSpec(
        name=_METADATA_TABLE,
        # Foundation validation runs before append-only migrations. New
        # storage-internal columns therefore belong to _SCHEMA_TABLES below,
        # not this pre-migration compatibility gate.
        required_columns=("workspace", "doc_id", *METADATA_FIELD_IDS, _SEARCH_COLUMN),
        primary_key=("workspace", "doc_id"),
        required_indexes=(
            "idx_dm_workspace_download_locator",
            *(f"idx_dm_{f.field_id}" for f in _PG_METADATA_COLUMNS if f.indexed),
            "idx_dm_custom_metadata_search",
            "idx_dm_filename_trgm",
        ),
        missing_ok=True,
        convert_empty_plain=False,
    )


def _build_schema_migrations() -> tuple[Migration, ...]:
    """Add declared metadata columns; never remove what is no longer declared.

    Every statement is derived from the registry rather than recorded as history.
    Indexes are rebuilt in place. Columns are never dropped automatically, but a
    registry change that leaves an undeclared ledger version requires a full
    development-data reset before this revision can start.
    """
    migrations = [
        Migration(
            "document_metadata",
            "Create document metadata table partitioned by workspace",
            (_CREATE_TABLE,),
        ),
        Migration(
            "partition_default_child",
            "Attach the shared DEFAULT child to the metadata parent",
            (
                f"CREATE TABLE IF NOT EXISTS {default_child_name(_METADATA_TABLE)} "
                f"PARTITION OF {_METADATA_TABLE} DEFAULT",
            ),
        ),
        Migration(
            "function_canonical_custom_metadata",
            "Install the shared custom-metadata canonicalization function",
            (_CREATE_CANONICAL_CUSTOM_FN,),
        ),
        Migration(
            "column_custom_metadata_search",
            "Add the storage-internal canonical search JSONB column",
            (
                "ALTER TABLE dlightrag_doc_metadata "
                "ADD COLUMN IF NOT EXISTS custom_metadata_search "
                "JSONB NOT NULL DEFAULT '{}'",
            ),
        ),
        Migration(
            "backfill_custom_metadata_search",
            "Deterministically backfill canonical custom-metadata search values",
            (_BACKFILL_CUSTOM_SEARCH,),
        ),
    ]
    for f in _PG_METADATA_COLUMNS:
        migrations.append(
            Migration(
                f"column_{f.field_id}",
                f"Ensure document metadata column {f.field_id}",
                (
                    "ALTER TABLE dlightrag_doc_metadata "
                    f"ADD COLUMN IF NOT EXISTS {pg_identifier(f.field_id)} {f.pg_type}",
                ),
            )
        )
    migrations.append(
        Migration(
            "index_workspace_download_locator",
            "Index exact source download ownership lookups",
            (
                "CREATE INDEX IF NOT EXISTS idx_dm_workspace_download_locator "
                "ON dlightrag_doc_metadata (workspace, download_locator)",
            ),
        )
    )
    for f in _PG_METADATA_COLUMNS:
        idx_clause = _index_clause(f.field_id, f.pg_type, f.indexed)
        if idx_clause is None:
            continue
        # Versioned by the expression: an index only serves a match whose
        # canonical form it was built with, so changing one must rebuild it.
        migrations.append(
            Migration(
                f"index_{f.field_id}_canonical",
                f"Ensure document metadata index {f.field_id}",
                (
                    f"DROP INDEX IF EXISTS idx_dm_{f.field_id}",
                    f"CREATE INDEX IF NOT EXISTS idx_dm_{f.field_id} "
                    f"ON dlightrag_doc_metadata{idx_clause}",
                ),
            )
        )
    migrations.append(
        Migration(
            "index_custom_metadata_search_gin",
            "GIN containment index for canonical custom metadata",
            (
                "CREATE INDEX IF NOT EXISTS idx_dm_custom_metadata_search "
                "ON dlightrag_doc_metadata "
                "USING GIN (custom_metadata_search jsonb_path_ops)",
            ),
        )
    )
    migrations.append(
        Migration(
            "index_filename_trgm",
            "Trigram GIN index for literal filename substring matching",
            (
                "CREATE INDEX IF NOT EXISTS idx_dm_filename_trgm "
                "ON dlightrag_doc_metadata "
                "USING GIN (LOWER(TRIM(filename)) gin_trgm_ops)",
            ),
        )
    )
    migrations.append(
        Migration(
            "column_finalization_complete",
            "Persist the application-owned ingestion finalization journal",
            (
                "ALTER TABLE dlightrag_doc_metadata "
                f"ADD COLUMN IF NOT EXISTS {_FINALIZATION_COMPLETE_COLUMN} "
                "BOOLEAN DEFAULT FALSE",
            ),
        )
    )
    migrations.append(
        Migration(
            "metadata_field_stats",
            "Maintain bounded planner field availability counts",
            (
                _CREATE_FIELD_STATS_TABLE,
                _CREATE_FIELD_STATS_TRIGGER_FN,
                _DROP_FIELD_STATS_TRIGGER,
                _CREATE_FIELD_STATS_TRIGGER,
                f"TRUNCATE TABLE {_FIELD_STATS_TABLE}",
                _BACKFILL_FIELD_STATS,
            ),
        )
    )
    migrations.append(
        Migration(
            "product_document_visibility",
            "Enforce finalized-only Product Document visibility and statistics",
            (
                "ALTER TABLE dlightrag_doc_metadata "
                f"ADD COLUMN IF NOT EXISTS {_FINALIZATION_COMPLETE_COLUMN} "
                "BOOLEAN DEFAULT FALSE",
                "UPDATE dlightrag_doc_metadata "  # noqa: S608 - fixed internal column
                f"SET {_FINALIZATION_COMPLETE_COLUMN} = FALSE "
                f"WHERE {_FINALIZATION_COMPLETE_COLUMN} IS NULL",
                "ALTER TABLE dlightrag_doc_metadata "
                f"ALTER COLUMN {_FINALIZATION_COMPLETE_COLUMN} SET DEFAULT FALSE",
                "ALTER TABLE dlightrag_doc_metadata "
                f"ALTER COLUMN {_FINALIZATION_COMPLETE_COLUMN} SET NOT NULL",
                "ALTER TABLE dlightrag_doc_metadata "
                "ADD CONSTRAINT dlightrag_doc_metadata_finalization_not_null "
                f"CHECK ({_FINALIZATION_COMPLETE_COLUMN} IS NOT NULL)",
                "CREATE INDEX IF NOT EXISTS idx_dm_visible_doc "
                "ON dlightrag_doc_metadata (workspace, doc_id) "
                f"WHERE {metadata_visibility_condition()}",
                _CREATE_FIELD_STATS_TRIGGER_FN,
                _DROP_FIELD_STATS_TRIGGER,
                _CREATE_FIELD_STATS_TRIGGER,
                f"TRUNCATE TABLE {_FIELD_STATS_TABLE}",
                _BACKFILL_FIELD_STATS,
            ),
        )
    )
    migrations.append(
        Migration(
            "publish_legacy_processed_documents",
            "Publish verified processed documents written before the visibility journal",
            (_PUBLISH_LEGACY_PROCESSED_DOCUMENTS,),
        )
    )
    return tuple(migrations)


_SCHEMA_MIGRATIONS = _build_schema_migrations()

_SCHEMA_TABLES = (
    TableRequirement(
        name="dlightrag_doc_metadata",
        columns=(
            "workspace",
            "doc_id",
            *METADATA_FIELD_IDS,
            "custom_metadata_search",
            _FINALIZATION_COMPLETE_COLUMN,
        ),
        primary_key=("workspace", "doc_id"),
        indexes=(
            "idx_dm_workspace_download_locator",
            *(f"idx_dm_{f.field_id}" for f in _PG_METADATA_COLUMNS if f.indexed),
            "idx_dm_custom_metadata_search",
            "idx_dm_filename_trgm",
            "idx_dm_visible_doc",
        ),
        checks=("dlightrag_doc_metadata_finalization_not_null",),
        partitioned_by=("workspace",),
        required_child_partitions=(default_child_name("dlightrag_doc_metadata"),),
    ),
    TableRequirement(
        name=_FIELD_STATS_TABLE,
        columns=("workspace", "field_id", "document_count"),
        primary_key=("workspace", "field_id"),
        checks=("dlightrag_metadata_field_stats_count_check",),
    ),
)

_CUSTOM = "custom_metadata"
_UPSERT_FIELD_IDS = (
    *(field_id for field_id in METADATA_FIELD_IDS if field_id != "ingested_at"),
    _FINALIZATION_COMPLETE_COLUMN,
)


def _field_assignment(field_id: str, placeholder: str, table_qualified: str) -> str:
    if field_id == _CUSTOM:
        # `||` yields NULL if either side is: a merge must never erase the column.
        return (
            f"{field_id} = COALESCE({table_qualified}, '{{}}'::jsonb) "
            f"|| COALESCE({placeholder}::jsonb, '{{}}'::jsonb)"
        )
    return f"{field_id} = COALESCE({placeholder}, {table_qualified})"


def _custom_placeholder_index(columns: tuple[str, ...]) -> int:
    """Return the $-placeholder index of the custom_metadata value in ``columns``."""
    return columns.index(_CUSTOM) + 1


def _search_assignment(placeholder: str, table_qualified: str) -> str:
    """Recompute the canonical search column from the merged raw custom metadata."""
    return (
        f"{_SEARCH_COLUMN} = dlightrag_canonical_custom_metadata("
        f"COALESCE({table_qualified}, '{{}}'::jsonb) "
        f"|| COALESCE({placeholder}::jsonb, '{{}}'::jsonb))"
    )


def _build_upsert() -> str:
    value_columns = ("workspace", "doc_id", *_UPSERT_FIELD_IDS)
    columns = (*value_columns, _SEARCH_COLUMN)
    insert_columns = ", ".join(columns)
    marker_slot = 3 + _UPSERT_FIELD_IDS.index(_FINALIZATION_COMPLETE_COLUMN)
    value_expressions = [f"${idx}" for idx in range(1, len(value_columns) + 1)]
    # A new partial row is unpublished, while a conflict update can still
    # distinguish omitted marker (preserve) from explicit False (hide).
    value_expressions[marker_slot - 1] = f"COALESCE(${marker_slot}, FALSE)"
    custom_placeholder = f"${_custom_placeholder_index(columns)}"
    value_expressions.append(
        f"dlightrag_canonical_custom_metadata(COALESCE({custom_placeholder}::jsonb, '{{}}'::jsonb))"
    )
    placeholders = ",".join(value_expressions)
    updates = [
        "    "
        + _field_assignment(
            field_id,
            (
                f"${marker_slot}"
                if field_id == _FINALIZATION_COMPLETE_COLUMN
                else f"EXCLUDED.{field_id}"
            ),
            f"dlightrag_doc_metadata.{field_id}",
        )
        for field_id in _UPSERT_FIELD_IDS
    ]
    updates.append(
        "    "
        + _search_assignment(
            f"EXCLUDED.{_CUSTOM}",
            "dlightrag_doc_metadata.custom_metadata",
        )
    )
    return (
        "INSERT INTO dlightrag_doc_metadata\n"
        f"    ({insert_columns})\n"
        f"VALUES ({placeholders})\n"
        "ON CONFLICT (workspace, doc_id) DO UPDATE SET\n" + ",\n".join(updates)
    )


def _build_update() -> str:
    """Same assignments as the upsert, but a missing document stays missing."""
    assignments = [
        "    " + _field_assignment(field_id, f"${idx}", field_id)
        for idx, field_id in enumerate(_UPSERT_FIELD_IDS, start=3)
    ]
    custom_placeholder = f"${3 + _UPSERT_FIELD_IDS.index(_CUSTOM)}"
    assignments.append("    " + _search_assignment(custom_placeholder, _CUSTOM))
    return (
        "UPDATE dlightrag_doc_metadata SET\n"
        + ",\n".join(assignments)
        + "\nWHERE workspace = $1 AND doc_id = $2 "
        + f"AND {metadata_visibility_condition()}"
    )


def _build_params(workspace: str, doc_id: str, metadata: dict[str, Any]) -> list[Any]:
    custom = metadata.get(_CUSTOM)
    return [
        workspace,
        doc_id,
        *(
            json.dumps(custom if isinstance(custom, dict) else {})
            if field_id == _CUSTOM
            else metadata.get(field_id)
            for field_id in _UPSERT_FIELD_IDS
        ),
    ]


_UPSERT = _build_upsert()
_UPDATE = _build_update()


def _build_field_schema() -> str:
    """Read populated filter columns and the most prevalent custom keys."""
    populated = ",\n    ".join(
        f"bool_or(field_id = '{column}') AS {column}" for column in _FILTERABLE_COLUMNS
    )
    builtins = ", ".join(f"'{column}'" for column in _FILTERABLE_COLUMNS)
    return f"""
WITH combined AS (
    SELECT field_id, SUM(document_count)::bigint AS document_count
    FROM {_FIELD_STATS_TABLE}
    WHERE workspace = ANY($1::text[])
      AND document_count > 0
    GROUP BY field_id
)
SELECT
    {populated},
    (
        SELECT array_agg(field_id ORDER BY document_count DESC, field_id)
        FROM (
            SELECT field_id, document_count
            FROM combined
            WHERE field_id NOT IN ({builtins})
            ORDER BY document_count DESC, field_id
            LIMIT {_CUSTOM_SCHEMA_KEY_LIMIT}
        ) AS custom
    ) AS custom_keys
FROM combined
"""  # noqa: S608 - field names come from the registry, never from input


_FIELD_SCHEMA = _build_field_schema()


# A named file is matched against both the full name and the stem, so a caller
# who omits the extension still hits the functional workspace-leading indexes on
# both.
_FILENAME_EXACT_CONDITION = (
    "({canonical_filename} = {canonical_value} OR {canonical_stem} = {canonical_value}) "
)


# A caller types a name, not a pattern, so the widened match is a literal
# substring search. The full pattern (wildcard framing plus escaping) is built
# by ``like_contains_pattern`` and bound as a bare parameter: a parameterized
# ``LIKE`` is the one shape the planner recognizes against the pg_trgm GIN
# expression index, while the ``ESCAPE`` clause keeps caller ``%``, ``_``, and
# ``\`` characters literal.
_FILENAME_CONTAINS_CONDITION = "{canonical_filename} LIKE LOWER(${idx}) ESCAPE '\\'"


def like_contains_pattern(value: str) -> str:
    r"""Build one literal-substring LIKE pattern for caller-supplied text.

    ``%``, ``_``, and ``\`` in the value stay literal characters: they are
    escaped against the ``ESCAPE '\'`` clause, and only this helper's own
    ``%`` framing acts as a wildcard. PostgreSQL lowercases the bound pattern
    in the predicate so non-ASCII folding uses the same database collation as
    the indexed ``LOWER(TRIM(filename))`` expression.
    """
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


_FILENAME_MODES = frozenset({"exact", "contains"})


def _filename_condition(
    column_prefix: str,
    value: str,
    *,
    filename_mode: str,
    idx: int,
) -> tuple[str, Any]:
    """Render the one filename clause the selected mode demands."""
    canonical_filename = _canonical(f"{column_prefix}filename")
    if filename_mode == "contains":
        condition = _FILENAME_CONTAINS_CONDITION.format(
            canonical_filename=canonical_filename, idx=idx
        )
        return condition, like_contains_pattern(value)
    condition = _FILENAME_EXACT_CONDITION.format(
        canonical_filename=canonical_filename,
        canonical_stem=_canonical(f"{column_prefix}filename_stem"),
        canonical_value=_canonical(f"${idx}"),
    )
    return condition, value


def metadata_match_conditions(
    workspace: str,
    filters: MetadataFilter,
    *,
    filename_mode: str = "exact",
    start_index: int = 1,
    alias: str | None = None,
) -> tuple[list[str], list[Any]]:
    """Build the shared workspace-and-filter conditions for one metadata match.

    Every retrieval leg (bounded scope preflight, exact/HNSW vector, BM25, and
    the graph-scoped chunk read) renders its predicates through this one
    builder, so their semantics can never drift apart. ``start_index`` lets
    each statement place the conditions after its own leading parameters, and
    ``alias`` qualifies columns when the conditions run in a subquery against
    an aliased metadata table.

    ``filename_mode`` selects the filename clause: ``exact`` (name or stem
    equality) or ``contains`` (literal substring). The widened clause is chosen
    once by the caller, so the internal full-set query and a paged traversal
    share identical SQL semantics instead of drifting apart.

    All custom key/value equalities collapse into one canonical JSONB
    containment object evaluated through the same SQL function the write path
    uses, so strings, numbers, booleans, nulls, and nested text representations
    match the storage contract exactly. No raw custom scan runs on this path.
    """
    if filename_mode not in _FILENAME_MODES:
        raise ValueError("metadata match filename mode is invalid")
    column_prefix = f"{alias}." if alias else ""
    conditions: list[str] = [
        f"{column_prefix}workspace = ${start_index}",
        metadata_visibility_condition(alias),
    ]
    params: list[Any] = [workspace]
    idx = start_index + 1

    for attr in ("file_extension", "author"):
        value = getattr(filters, attr, None)
        if value is None:
            continue
        conditions.append(f"{_canonical(f'{column_prefix}{attr}')} = {_canonical(f'${idx}')}")
        params.append(value)
        idx += 1

    if filters.title:
        # Unbounded TEXT: the index key is the fixed-width MD5 of the canonical
        # value, so the predicate carries the MD5 equality for the index plus
        # the full equality recheck that makes matches exact on hash collision.
        conditions.append(
            f"MD5({_canonical(f'{column_prefix}title')}) = MD5({_canonical(f'${idx}')}) "
            f"AND {_canonical(f'{column_prefix}title')} = {_canonical(f'${idx}')}"
        )
        params.append(filters.title)
        idx += 1

    if filters.filename:
        condition, param = _filename_condition(
            column_prefix,
            filters.filename,
            filename_mode=filename_mode,
            idx=idx,
        )
        conditions.append(condition)
        params.append(param)
        idx += 1

    # Date range
    if filters.creation_date_from:
        conditions.append(f"{column_prefix}creation_date >= ${idx}")
        params.append(filters.creation_date_from)
        idx += 1
    if filters.creation_date_to:
        conditions.append(f"{column_prefix}creation_date <= ${idx}")
        params.append(filters.creation_date_to)
        idx += 1

    # Custom key/value equalities are one canonical containment object: the
    # bound JSONB folds keys and scalar/text representations through the same
    # immutable SQL function the storage column was written with, so the
    # comparison and the GIN jsonb_path_ops index agree by construction.
    if filters.custom:
        conditions.append(
            f"{column_prefix}{_SEARCH_COLUMN} @> dlightrag_canonical_custom_metadata(${idx}::jsonb)"
        )
        params.append(json.dumps(filters.custom))
        idx += 1

    return conditions, params


# Deletion resolves a name, so it matches the full name only, never the stem.
_FIND_BY_FILENAME = (
    "SELECT doc_id FROM dlightrag_doc_metadata "  # noqa: S608 - fixed text; only $-params
    f"WHERE workspace=$1 AND {_canonical('filename')} = {_canonical('$2')}"
)


def _decoded_row(row: Any) -> dict[str, Any]:
    """Decode public metadata and remove storage-internal search projections."""
    decoded = dict(row)
    decoded.pop(_SEARCH_COLUMN, None)
    raw = decoded.get("custom_metadata")
    decoded["custom_metadata"] = json.loads(raw) if isinstance(raw, str) else (raw or {})
    return decoded


async def rebuild_metadata_field_stats_for_workspace(conn: Any, workspace: str) -> None:
    """Recount one workspace after its rows move between physical partitions."""
    await conn.execute(
        f"DELETE FROM {_FIELD_STATS_TABLE} WHERE workspace = $1",  # noqa: S608
        workspace,
    )
    await conn.execute(_BACKFILL_WORKSPACE_FIELD_STATS, workspace)


class PGMetadataIndex(PostgresOperationRunner):
    """PostgreSQL-backed document metadata index.

    Stores system-extracted and user-defined metadata per document.
    """

    def __init__(self, workspace: str = "default") -> None:
        super().__init__()
        self._workspace = workspace

    async def initialize(self, *, validate_only: bool = False) -> None:
        """Create/convert the partitioned table and indexes, or validate (reader).

        The partition foundation runs before migrations so an old unpartitioned
        (or populated) corpus fails loudly with the one-time development reset
        message instead of hitting a raw PostgreSQL DDL error.
        """

        async def _operation(conn: Any) -> None:
            if validate_only:
                await verify_partitioned_tables(conn, specs=(_metadata_partition_spec(),))
                await verify_migrations(
                    conn,
                    scope="doc_metadata",
                    migrations=_SCHEMA_MIGRATIONS,
                    tables=_SCHEMA_TABLES,
                    schema_error=CorpusSchemaError,
                )
                return
            await ensure_partitioned_tables(conn, specs=(_metadata_partition_spec(),))
            await apply_migrations(
                conn,
                scope="doc_metadata",
                migrations=_SCHEMA_MIGRATIONS,
                schema_error=CorpusSchemaError,
                require_applied_prefix=False,
            )

        await self._run(_operation)

    async def upsert(self, doc_id: str, metadata: dict[str, Any]) -> None:
        """Insert or update document metadata."""
        params = _build_params(self._workspace, doc_id, metadata)

        async def _operation(conn: Any) -> None:
            await conn.execute(_UPSERT, *params)

        await self._run(_operation)

    async def merge_custom_metadata(self, doc_id: str, metadata: dict[str, Any]) -> bool:
        """Update an existing document, reporting whether one was there to update."""
        params = _build_params(self._workspace, doc_id, metadata)

        async def _operation(conn: Any) -> str:
            return await conn.execute(_UPDATE, *params)

        return (await self._run(_operation)) != "UPDATE 0"

    async def query(self, filters: MetadataFilter) -> list[str]:
        """Query for doc_ids matching the given filters.

        Explicit metadata-search/admin API only: the retrieval runtime path
        resolves scopes through ``MetadataScopeStore.resolve_scope`` and never
        materializes a document-id set on the request path.
        """
        conditions, params = metadata_match_conditions(
            self._workspace,
            filters,
            filename_mode="exact",
        )
        doc_ids = await self._select_doc_ids(conditions, params)
        if doc_ids or not filters.filename:
            return doc_ids

        # The caller named a file the corpus does not carry verbatim. A planner
        # cannot know whether a name is complete, and a human rarely types one,
        # so widen that single clause rather than returning nothing.
        conditions, params = metadata_match_conditions(
            self._workspace,
            filters,
            filename_mode="contains",
        )
        return await self._select_doc_ids(conditions, params)

    async def _select_doc_ids(self, conditions: list[str], params: list[Any]) -> list[str]:
        where = " AND ".join(conditions)
        sql = f"SELECT doc_id FROM dlightrag_doc_metadata WHERE {where}"  # noqa: S608

        async def _operation(conn: Any) -> list[Any]:
            return await conn.fetch(sql, *params)

        rows = await self._run(_operation)

        return [r["doc_id"] for r in rows]

    async def get(self, doc_id: str) -> dict[str, Any] | None:
        """Get metadata for a single document."""

        async def _operation(conn: Any) -> Any:
            return await conn.fetchrow(
                "SELECT * FROM dlightrag_doc_metadata WHERE workspace=$1 AND doc_id=$2",
                self._workspace,
                doc_id,
            )

        row = await self._run(_operation)
        if not row:
            return None
        return _decoded_row(row)

    async def get_many(self, doc_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Get metadata for multiple documents in one query."""
        unique_doc_ids = list(dict.fromkeys(str(doc_id) for doc_id in doc_ids if doc_id))
        if not unique_doc_ids:
            return {}

        async def _operation(conn: Any) -> list[Any]:
            return await conn.fetch(
                "SELECT * FROM dlightrag_doc_metadata WHERE workspace=$1 AND doc_id = ANY($2::text[])",
                self._workspace,
                unique_doc_ids,
            )

        rows = await self._run(_operation)
        return {str(row["doc_id"]): _decoded_row(row) for row in rows}

    async def is_visible(self, doc_id: str) -> bool:
        """Return whether one document has the exact publication marker."""

        async def _operation(conn: Any) -> Any:
            return await conn.fetchval(
                "SELECT 1 FROM dlightrag_doc_metadata "  # noqa: S608 - fixed internal column
                "WHERE workspace=$1 AND doc_id=$2 "
                f"AND {metadata_visibility_condition()}",
                self._workspace,
                doc_id,
            )

        return bool(await self._run(_operation))

    async def visible_subset(
        self,
        doc_ids: Sequence[str],
        *,
        scope: MetadataScope | None = None,
    ) -> frozenset[str]:
        """Apply publication and optional metadata scope to caller-bounded ids."""
        unique_doc_ids = list(dict.fromkeys(str(doc_id) for doc_id in doc_ids if doc_id))
        if not unique_doc_ids or (scope is not None and not scope):
            return frozenset()

        filters = scope.filters if scope is not None else MetadataFilter()
        filename_mode = scope.filename_mode if scope is not None else "exact"
        conditions, params = metadata_match_conditions(
            self._workspace,
            filters,
            filename_mode=filename_mode,
            start_index=2,
            alias="m",
        )
        where = " AND ".join(conditions)
        sql = (
            f"SELECT m.doc_id FROM dlightrag_doc_metadata m "  # noqa: S608
            f"WHERE m.doc_id = ANY($1::text[]) AND {where}"
        )

        async def _operation(conn: Any) -> list[Any]:
            return await conn.fetch(sql, unique_doc_ids, *params)

        rows = await self._run(_operation)
        return frozenset(str(row["doc_id"]) for row in rows)

    async def delete(self, doc_id: str) -> None:
        """Delete metadata for a document."""

        async def _operation(conn: Any) -> None:
            await conn.execute(
                "DELETE FROM dlightrag_doc_metadata WHERE workspace=$1 AND doc_id=$2",
                self._workspace,
                doc_id,
            )

        await self._run(_operation)

    async def clear(self) -> None:
        """Delete all metadata for this workspace."""

        async def _operation(conn: Any) -> str:
            return await conn.execute(
                "DELETE FROM dlightrag_doc_metadata WHERE workspace=$1",
                self._workspace,
            )

        result = await self._run(_operation)
        logger.info("PGMetadataIndex cleared for workspace %s: %s", self._workspace, result)

    async def get_field_schema(
        self,
        *,
        workspaces: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Report the filters this workspace can actually satisfy.

        Returns the planner-facing filter field names backed by at least one
        populated column, plus the custom metadata keys in use. Formats and
        meanings are static and live in the planner prompt; only availability
        is workspace-dependent.
        """
        workspace_scope = list(dict.fromkeys(workspaces or (self._workspace,)))

        async def _operation(conn: Any) -> Any:
            return await conn.fetchrow(_FIELD_SCHEMA, workspace_scope)

        row = await self._run(_operation)
        if row is None:
            return {"filters": [], "custom_keys": []}

        filters = [
            field
            for field, columns in FILTER_FIELD_COLUMNS.items()
            if any(row[column] for column in columns)
        ]
        custom_keys = list(row["custom_keys"] or ())
        if custom_keys:
            # The planner is told every other field must stay null, so a key it
            # may filter on is useless unless `custom` is named as available.
            filters.append("custom")
        return {"filters": filters, "custom_keys": custom_keys}

    async def find_by_filename(self, name: str) -> list[str]:
        """Find doc_ids by case-insensitive filename match."""

        async def _operation(conn: Any) -> list[Any]:
            return await conn.fetch(
                _FIND_BY_FILENAME,
                self._workspace,
                name,
            )

        rows = await self._run(_operation)
        return [r["doc_id"] for r in rows]

    async def find_by_download_locator(self, download_locator: str) -> list[str]:
        """Find doc_ids owning one exact download locator in this workspace."""

        async def _operation(conn: Any) -> list[Any]:
            return await conn.fetch(
                "SELECT doc_id FROM dlightrag_doc_metadata "
                "WHERE workspace=$1 AND download_locator=$2",
                self._workspace,
                download_locator,
            )

        rows = await self._run(_operation)
        return [r["doc_id"] for r in rows]
