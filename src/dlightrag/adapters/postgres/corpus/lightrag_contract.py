# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Centralized startup validation for LightRAG coupling assumptions.

DlightRAG extends LightRAG through wrapping (FilteredVectorStorage), monkey
patching (_lightrag_patches), and direct DB access (PGMetadataIndex,
LightRAG text_chunks/chunks_vdb/doc_status). This guard validates all
coupling assumptions once at startup and fails fast with a complete error
report if anything has drifted between LightRAG releases.

Called by the PostgreSQL corpus adapter after storage attachment and before
chunks_vdb is wrapped by the storage-neutral filtering layer.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

READ_ONLY_STORAGE_ATTRS = (
    "full_docs",
    "text_chunks",
    "full_entities",
    "full_relations",
    "entity_chunks",
    "relation_chunks",
    "entities_vdb",
    "relationships_vdb",
    "chunks_vdb",
    "chunk_entity_relation_graph",
    "llm_response_cache",
    "doc_status",
)


class PGLightRAGContractGuard:
    """Validates LightRAG internal API assumptions at startup.

    Collects all errors before raising, producing one report instead of
    failing on the first issue. PostgreSQL is the only supported backend.
    """

    _REQUIRED_CALLABLES = (
        "initialize_storages",
        "finalize_storages",
        "aquery_data",
        "apipeline_enqueue_documents",
        "apipeline_process_enqueue_documents",
        "adelete_by_doc_id",
        "aget_docs_by_track_id",
    )
    _REQUIRED_ATTRIBUTES = ("chunks_vdb", "text_chunks", "full_docs", "doc_status")
    _REQUIRED_DOC_STATUS_CALLABLES = (
        "get_docs_by_statuses_page",
        "get_full_docs_by_ids",
    )
    _CHUNKS_VDB_COLUMNS = {"id", "content", "content_vector", "workspace", "file_path"}
    _BM25_TABLE = "lightrag_doc_chunks"
    _BM25_COLUMNS = {"id", "content", "file_path", "workspace"}
    _CLIENT_MANAGER_CONFIG_PARAMS = ("vector_storage",)
    _CLIENT_MANAGER_BUILD_SIGNATURE_PARAMS = ("config", "vector_storage")
    _CLIENT_MANAGER_ASSERT_SIGNATURE_PARAMS = ("requested_signature",)
    _NAMESPACE_TO_TABLE_NAME_PARAMS = ("namespace",)

    def __init__(self, lightrag: Any) -> None:
        self._lightrag = lightrag

    def verify_surface(self) -> None:
        """Fail before storage initialization when the consumed runtime surface drifted."""
        errors = [
            f"LightRAG missing callable {name!r}"
            for name in self._REQUIRED_CALLABLES
            if not callable(getattr(self._lightrag, name, None))
        ]
        errors.extend(
            f"LightRAG missing attribute {name!r}"
            for name in self._REQUIRED_ATTRIBUTES
            if not hasattr(self._lightrag, name)
        )
        doc_status = getattr(self._lightrag, "doc_status", None)
        if doc_status is not None:
            errors.extend(
                f"LightRAG doc_status missing callable {name!r}"
                for name in self._REQUIRED_DOC_STATUS_CALLABLES
                if not callable(getattr(doc_status, name, None))
            )
        if errors:
            raise RuntimeError(
                f"LightRAG runtime contract check failed ({len(errors)} issue(s)):\n"
                + "\n".join(f"  - {error}" for error in errors)
            )

    async def verify_all(self, *, vector_storage: str = "PGVectorStorage") -> None:
        """Validate the PostgreSQL KV/BM25 leg and the selected vector contract."""
        errors: list[str] = []
        self._require_pg_text_chunks(errors)
        if not errors:
            await self._check_bm25_table(errors)
            if vector_storage == "PGVectorStorage":
                self._require_pg_vector(errors)
                if not errors:
                    await self._check_chunks_table_schema(errors)
        if errors:
            raise RuntimeError(
                f"LightRAG contract check failed "
                f"({len(errors)} issue(s)):\n" + "\n".join(f"  - {e}" for e in errors)
            )
        logger.info("LightRAG contract check passed (vector=%s)", vector_storage)

    def verify_read_only_attach_contract(self) -> None:
        """Validate the private surfaces the read-only attach adapter relies on."""
        errors: list[str] = []
        self._check_read_only_attach_contract(errors)
        if errors:
            raise RuntimeError(
                f"LightRAG contract check failed "
                f"({len(errors)} issue(s)):\n" + "\n".join(f"  - {e}" for e in errors)
            )

    def _require_pg_text_chunks(self, errors: list[str]) -> None:
        """Require the fixed PostgreSQL chunk-KV leg used by BM25 and metadata reads."""
        chunks = getattr(self._lightrag, "text_chunks", None)
        if chunks is None:
            errors.append("text_chunks missing (PGKVStorage required)")
            return
        db = getattr(chunks, "db", None)
        if db is None:
            errors.append("text_chunks.db missing (PGKVStorage required)")
            return
        if not hasattr(db, "pool") or getattr(db, "pool", None) is None:
            errors.append("text_chunks.db.pool missing (PGKVStorage required)")

    def _require_pg_vector(self, errors: list[str]) -> None:
        """Require PostgreSQL vector internals only for explicit PGVectorStorage."""
        vdb = getattr(self._lightrag, "chunks_vdb", None)
        if vdb is None:
            errors.append("chunks_vdb missing (PGVectorStorage required)")
            return
        db = getattr(vdb, "db", None)
        if db is None:
            errors.append("chunks_vdb.db missing (PGVectorStorage required)")
            return
        if not hasattr(db, "pool") or getattr(db, "pool", None) is None:
            errors.append("chunks_vdb.db.pool missing (PGVectorStorage required)")

    async def _check_chunks_table_schema(self, errors: list[str]) -> None:
        """Check A: chunks_vdb table has all columns we depend on."""
        vdb = self._lightrag.chunks_vdb
        table_name = getattr(vdb, "table_name", None)
        if not table_name:
            errors.append("chunks_vdb missing 'table_name' attribute (PG path)")
            return
        pool = vdb.db.pool
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = $1",
                table_name.lower(),
            )
        actual = {r["column_name"] for r in rows}
        missing = self._CHUNKS_VDB_COLUMNS - actual
        if missing:
            errors.append(f"chunks_vdb table '{table_name}' missing columns: {missing}")

    async def _check_bm25_table(self, errors: list[str]) -> None:
        """Check B: the PostgreSQL text-chunk/BM25 table has required columns."""
        pool = self._lightrag.text_chunks.db.pool
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = $1",
                self._BM25_TABLE.lower(),
            )
        if not rows:
            errors.append(f"BM25 table '{self._BM25_TABLE}' does not exist")
            return
        actual = {r["column_name"] for r in rows}
        missing = self._BM25_COLUMNS - actual
        if missing:
            errors.append(f"BM25 table '{self._BM25_TABLE}' missing columns: {missing}")

    def _check_read_only_attach_contract(self, errors: list[str]) -> None:
        """Check E: reader attach adapter surfaces remain available."""
        import inspect

        keyword_compatible_kinds = (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        positional_compatible_kinds = (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )

        def _matches_expected_prefix_with_optional_suffix(
            signature: inspect.Signature,
            expected: tuple[str, ...],
            required_kinds: tuple[tuple[inspect._ParameterKind, ...], ...],
        ) -> bool:
            parameters = tuple(signature.parameters.values())
            param_names = tuple(parameter.name for parameter in parameters)
            if param_names[: len(expected)] != expected:
                return False
            for parameter, allowed_kinds in zip(
                parameters[: len(expected)], required_kinds, strict=True
            ):
                if parameter.kind not in allowed_kinds:
                    return False
            for parameter in parameters[len(expected) :]:
                if parameter.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                if parameter.default is inspect.Parameter.empty:
                    return False
            return True

        try:
            from lightrag.kg.postgres_impl import ClientManager, namespace_to_table_name
        except ImportError as e:
            errors.append(f"Cannot import reader attach surfaces: {e}")
            return

        for attr in READ_ONLY_STORAGE_ATTRS:
            if not hasattr(self._lightrag, attr):
                errors.append(f"LightRAG missing '{attr}' storage attribute for reader attach")

        required_client_attrs = (
            "get_config",
            "_build_vector_signature",
            "_assert_compatible_vector_signature",
            "_lock",
            "_instances",
        )
        for attr in required_client_attrs:
            if not hasattr(ClientManager, attr):
                errors.append(f"ClientManager.{attr} missing for reader attach")

        instances = getattr(ClientManager, "_instances", None)
        if instances is not None:
            for key in ("db", "ref_count", "vector_signature"):
                if key not in instances:
                    errors.append(f"ClientManager._instances missing key '{key}'")

        signature_checks = (
            (
                "ClientManager.get_config",
                getattr(ClientManager, "get_config", None),
                self._CLIENT_MANAGER_CONFIG_PARAMS,
                (keyword_compatible_kinds,),
            ),
            (
                "ClientManager._build_vector_signature",
                getattr(ClientManager, "_build_vector_signature", None),
                self._CLIENT_MANAGER_BUILD_SIGNATURE_PARAMS,
                (positional_compatible_kinds, positional_compatible_kinds),
            ),
            (
                "ClientManager._assert_compatible_vector_signature",
                getattr(ClientManager, "_assert_compatible_vector_signature", None),
                self._CLIENT_MANAGER_ASSERT_SIGNATURE_PARAMS,
                (positional_compatible_kinds,),
            ),
            (
                "namespace_to_table_name",
                namespace_to_table_name,
                self._NAMESPACE_TO_TABLE_NAME_PARAMS,
                (positional_compatible_kinds,),
            ),
        )
        for name, value, expected, required_kinds in signature_checks:
            if value is None or not callable(value):
                continue
            try:
                signature = inspect.signature(value)
                params = tuple(signature.parameters.keys())
            except (ValueError, TypeError) as e:
                errors.append(f"Cannot inspect {name}: {e}")
                continue
            if not _matches_expected_prefix_with_optional_suffix(
                signature, expected, required_kinds
            ):
                errors.append(f"{name} signature changed: expected prefix {expected}, got {params}")


__all__ = ["PGLightRAGContractGuard", "READ_ONLY_STORAGE_ATTRS"]
