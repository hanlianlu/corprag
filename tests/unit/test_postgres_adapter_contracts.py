# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Ownership contracts for the root PostgreSQL adapters."""

import ast
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOTS = (_ROOT / "src", _ROOT / "packages")
_POSTGRES_ADAPTER = _ROOT / "src/dlightrag/adapters/postgres"
_MEMORY_PG_ADAPTER = _ROOT / "packages/memory/src/dlightrag_memory/_storage"
_RAG_CORE = _ROOT / "src/dlightrag/engine/rag"
_RAG_OFFLINE_REBUILD = {
    _RAG_CORE / "corpus" / "rebuild_bm25.py",
    _RAG_CORE / "corpus" / "rebuild_vdb.py",
}
_RAW_SQL_RE = re.compile(
    r"\b(?:SELECT\b.{0,500}\bFROM|INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|"
    r"CREATE\s+(?:TABLE|INDEX)|ALTER\s+TABLE|DROP\s+INDEX)\b",
    re.IGNORECASE | re.DOTALL,
)
_POSTGRES_IDENTIFIERS = (
    "LIGHTRAG_DOC_CHUNKS",
    "dlightrag_doc_metadata",
    "dlightrag_ingest_jobs",
    "dlightrag_bm25_language",
)
_WEB_RECORD_NAMES = {
    "AnswerTurnCreation",
    "ConversationHead",
    "ConversationHistoryPage",
    "RecoveryTurnBatch",
    "SubmissionSeed",
    "ConversationSubmissionConflict",
    "LinkedTurn",
}


def _python_files() -> list[Path]:
    return [path for root in _SOURCE_ROOTS for path in root.rglob("*.py")]


def test_file_panel_page_queries_match_the_exact_partial_index_contract() -> None:
    from dlightrag.adapters.postgres.corpus import file_panel

    index = " ".join(file_panel._CREATE_PAGE_INDEX.split())
    first = " ".join(file_panel._LIST_FIRST_PAGE.split())
    after_null = " ".join(file_panel._LIST_AFTER_NULL.split())
    after_timestamp = " ".join(file_panel._LIST_AFTER_TIMESTAMP.split())

    assert "(workspace, updated_at DESC NULLS FIRST, id ASC)" in index
    assert "WHERE status = 'processed'" in index
    assert "INNER JOIN dlightrag_doc_metadata m" in first
    assert "m._dlightrag_finalization_complete IS TRUE" in first
    assert "ORDER BY s.updated_at DESC NULLS FIRST, s.id ASC LIMIT $2" in first
    assert "(s.updated_at IS NULL AND s.id > $2) OR s.updated_at IS NOT NULL" in after_null
    assert "s.updated_at < $2::timestamp" in after_timestamp
    assert "s.updated_at = $2::timestamp AND s.id > $3" in after_timestamp
    assert "ORDER BY s.updated_at DESC NULLS FIRST, s.id ASC" in after_timestamp
    assert all("OFFSET" not in query.upper() for query in (first, after_null, after_timestamp))


def test_web_conversation_page_query_matches_the_covering_index_contract() -> None:
    from dlightrag.adapters.postgres.web import web_conversations

    first = " ".join(web_conversations._LIST_CONVERSATIONS_FIRST_PAGE.split())
    after = " ".join(web_conversations._LIST_CONVERSATIONS_AFTER.split())
    index = " ".join(web_conversations._CREATE_CONVERSATION_INDEXES[0].split())

    assert "(principal_id, updated_at DESC, conversation_id DESC)" in index
    assert "WHERE principal_id = $1" in first
    assert "ORDER BY updated_at DESC, conversation_id DESC LIMIT $2" in first
    assert "(updated_at, conversation_id) < ($2::timestamptz, $3::uuid)" in after
    assert "ORDER BY updated_at DESC, conversation_id DESC LIMIT $4" in after
    assert "OFFSET" not in first.upper()
    assert "OFFSET" not in after.upper()


def test_metadata_search_page_sql_matches_the_doc_id_keyset_contract() -> None:
    from dlightrag.adapters.postgres.corpus import pg_metadata_search
    from dlightrag.adapters.postgres.corpus.pg_metadata_index import metadata_match_conditions
    from dlightrag.engine.rag.retrieval import MetadataFilter

    conditions, params = metadata_match_conditions(
        "finance",
        MetadataFilter(filename="Report", file_extension=".pdf"),
        filename_mode="exact",
    )
    first = " ".join(pg_metadata_search._paged_sql(conditions, params, after_doc_id=None).split())
    after = " ".join(
        pg_metadata_search._paged_sql(conditions, params, after_doc_id="doc-9").split()
    )

    assert "ORDER BY doc_id ASC LIMIT $4" in first
    assert "doc_id > $4" in after
    assert "ORDER BY doc_id ASC LIMIT $5" in after
    assert "OFFSET" not in first.upper()
    assert "OFFSET" not in after.upper()


def test_child_roster_page_queries_match_the_exact_roster_index_contract() -> None:
    from dlightrag.adapters.postgres.runtime import run_store

    first = " ".join(run_store._SELECT_CHILD_SESSIONS_FIRST_PAGE.split())
    after = " ".join(run_store._SELECT_CHILD_SESSIONS_AFTER.split())
    index_statements = [
        " ".join(statement.split())
        for statement in run_store._CREATE_INDEXES
        if "idx_answer_child_sessions_roster" in statement
    ]

    assert index_statements, "roster index must be declared in _CREATE_INDEXES"
    assert "(owner_id, run_id, created_at DESC, child_session_id DESC)" in index_statements[0]
    assert "WHERE owner_id = $1 AND run_id = $2" in first
    assert "ORDER BY created_at DESC, child_session_id DESC LIMIT $3" in first
    assert (
        "created_at < $3::timestamptz" in after
        and "created_at = $3::timestamptz AND child_session_id < $4::uuid" in after
    )
    assert "ORDER BY created_at DESC, child_session_id DESC LIMIT $5" in after
    assert all("OFFSET" not in query.upper() for query in (first, after))


def test_web_turn_pages_select_limit_plus_one_identities_before_run_joins() -> None:
    from dlightrag.adapters.postgres.web import web_conversations

    history = " ".join(web_conversations._GET_TURNS_PAGE.split()).upper()
    oldest = " ".join(web_conversations._GET_RECOVERY_OLDEST.split()).upper()
    assert "WITH SELECTED_TURNS AS" in history
    assert history.index("LIMIT $4") < history.index("JOIN DLIGHTRAG_RUNS")
    assert "T.TURN_NUMBER < $3" in history
    assert "ORDER BY T.TURN_NUMBER DESC" in history
    assert "LIMIT $5" in oldest
    assert oldest.index("LIMIT $5") < oldest.index("JOIN DLIGHTRAG_RUNS")
    assert "ORDER BY T.TURN_NUMBER ASC" in oldest
    assert "OFFSET" not in history and "OFFSET" not in oldest


def test_web_turn_numbers_use_the_locked_conversation_revision_and_attachment_seed_is_safe() -> (
    None
):
    from dlightrag.adapters.postgres.web import web_conversations

    insert = " ".join(web_conversations._INSERT_TURN.split()).upper()
    attachments = " ".join(web_conversations._GET_CARRIED_ATTACHMENTS.split()).upper()

    assert "MAX(TURN_NUMBER)" not in insert
    assert "$4" in insert
    assert "ORDER BY T.TURN_NUMBER DESC, SOURCE_ORDINAL ASC" in attachments
    assert "ATTACHMENT.VALUE->>'DIGEST' IS NOT NULL" in attachments


def test_asyncpg_is_private_to_the_postgres_adapter() -> None:
    offenders: list[Path] = []
    for path in _python_files():
        if path.is_relative_to(_POSTGRES_ADAPTER) or path.is_relative_to(_MEMORY_PG_ADAPTER):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            (isinstance(node, ast.Import) and any(alias.name == "asyncpg" for alias in node.names))
            or (isinstance(node, ast.ImportFrom) and node.module == "asyncpg")
            for node in ast.walk(tree)
        ):
            offenders.append(path.relative_to(_ROOT))

    assert offenders == []


def test_owner_modules_do_not_import_the_postgres_adapter() -> None:
    owner_roots = (
        _ROOT / "src/dlightrag/application",
        _ROOT / "src/dlightrag/engine/runtime",
        _RAG_CORE,
    )
    owner_files = [
        path
        for root in owner_roots
        for path in root.rglob("*.py")
        if path not in _RAG_OFFLINE_REBUILD
    ]
    owner_files.append(_ROOT / "src/dlightrag/adapters/http/rest/routes/status.py")

    offenders: list[Path] = []
    for path in owner_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and node.module.startswith("dlightrag.adapters.postgres")
            )
            or (
                isinstance(node, ast.Import)
                and any(
                    alias.name.startswith("dlightrag.adapters.postgres") for alias in node.names
                )
            )
            for node in ast.walk(tree)
        ):
            offenders.append(path.relative_to(_ROOT))

    assert offenders == []


def test_rag_core_contains_no_postgres_schema_or_raw_sql() -> None:
    offenders: list[Path] = []
    assert _RAG_CORE.is_dir()
    for path in _RAG_CORE.rglob("*.py"):
        if path in _RAG_OFFLINE_REBUILD:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        docstrings: set[int] = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if not body or not isinstance(body, list):
                continue
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                docstrings.add(id(first.value))
        literals = (
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        )
        if any(
            _RAW_SQL_RE.search(literal)
            or any(identifier in literal for identifier in _POSTGRES_IDENTIFIERS)
            for literal in literals
        ):
            offenders.append(path.relative_to(_ROOT))

    assert offenders == []


def test_callers_take_web_records_from_the_application_owner() -> None:
    offenders: list[tuple[Path, str]] = []
    for path in _python_files() + list((_ROOT / "tests").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.ImportFrom)
                and node.module == "dlightrag.adapters.http.browser.conversation_models"
            ):
                continue
            offenders.extend(
                (path.relative_to(_ROOT), alias.name)
                for alias in node.names
                if alias.name in _WEB_RECORD_NAMES
            )

    assert offenders == []


def test_workspace_catalog_page_query_rides_the_primary_key_without_offset() -> None:
    from dlightrag.adapters.postgres.corpus import workspaces

    query = " ".join(workspaces._LIST_PAGE.split())

    assert "SELECT workspace, display_name, embedding_model, created_at, updated_at" in query
    assert "FROM dlightrag_workspace_meta" in query
    assert "WHERE workspace > $1" in query
    assert "ORDER BY workspace ASC" in query
    assert "LIMIT $2" in query
    assert "OFFSET" not in query.upper()
