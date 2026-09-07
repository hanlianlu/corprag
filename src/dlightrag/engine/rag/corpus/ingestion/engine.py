# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unified ingestion over LightRAG parser sidecars and image vector overrides."""

import asyncio
import hashlib
import logging
import shutil
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lightrag.constants import FULL_DOCS_FORMAT_PENDING_PARSE
from lightrag.parser.routing import (
    chunk_strategy_key,
    encode_parse_engine,
    resolve_chunk_options,
    resolve_parser_directives,
)
from lightrag.utils import compute_mdhash_id
from lightrag.utils_pipeline import normalize_document_file_path, resolve_sidecar_uri

from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY, Telemetry
from dlightrag.engine.rag.corpus.ingestion.document_embedding import (
    DocumentEmbeddingInput,
    RobustDocumentEmbedder,
)
from dlightrag.engine.rag.corpus.ingestion.errors import RetryOutcomeUncertainError
from dlightrag.engine.rag.corpus.ingestion.lightrag_sidecar import collect_lightrag_drawing_assets
from dlightrag.engine.rag.corpus.ingestion.paths import lightrag_archived_source_path
from dlightrag.engine.rag.corpus.sources.source_contract import (
    local_source_uri,
    safe_source_filename,
)
from dlightrag.engine.rag.retrieval.metadata_fields import (
    INGEST_FINALIZATION_COMPLETE_FIELD,
    extract_system_metadata,
    normalize_user_metadata,
)
from dlightrag.engine.rag.retrieval.visibility import ingest_finalization_complete
from dlightrag.engine.rag.workspace.lifecycle import await_shared_cleanup, defer_cancellation

logger = logging.getLogger(__name__)

_FINALIZATION_COMPLETE_KEY = INGEST_FINALIZATION_COMPLETE_FIELD
_ACTIVE_INGEST_STATUSES = frozenset(
    {"pending", "parsing", "analyzing", "processing", "preprocessed"}
)
_STATUS_POLL_INITIAL_SECONDS = 0.05
_STATUS_POLL_MAX_SECONDS = 1.0


@dataclass(frozen=True)
class PreparedIngestFile:
    """Prepared parser input with explicit provenance and download identity.

    ``parser_path`` must be a local file because LightRAG pending-parse
    ingestion requires local parser input. ``source_uri`` is stable provenance;
    ``download_locator`` is the durable location used to reacquire the bytes.
    """

    parser_path: Path
    source_uri: str
    download_locator: str
    display_filename: str | None = None
    title: str | None = None
    author: str | None = None
    metadata: Mapping[str, Any] | None = None
    source_uri_explicit: bool = True
    download_locator_explicit: bool = True
    display_filename_explicit: bool = False
    replacement_doc_ids: tuple[str, ...] = ()
    # (doc_id, accepted download locator, accepted stable source URI).
    # The engine revalidates this ownership under the per-document lock.
    replacement_ownership: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "parser_path", Path(self.parser_path))


@dataclass(frozen=True)
class _PendingDocumentIngest:
    index: int
    parser_path: Path
    doc_id: str
    metadata_record: dict[str, Any]
    metadata_update_requested: bool
    parse_engine: str | None
    process_options: str | None
    chunk_options: dict[str, Any] | None
    replacement_doc_ids: tuple[str, ...]
    replacement_ownership: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class _DocumentIngestDecision:
    enqueue: bool
    cleanup_kind: str | None = None
    result: dict[str, Any] | None = None
    metadata_update: bool = False
    finalize_only: bool = False


class UnifiedIngestionEngine:
    """One ingestion path over LightRAG parser/routing."""

    def __init__(
        self,
        *,
        lightrag: Any,
        stores: Any,
        metadata_index: Any,
        document_embedder: RobustDocumentEmbedder,
        workspace: str,
        parser_rules: str,
        chunk_options: dict[str, Any] | None,
        bm25_language_classifier: Any | None = None,
        telemetry: Telemetry = NOOP_TELEMETRY,
    ) -> None:
        self._lightrag = lightrag
        self._stores = stores
        self._metadata_index = metadata_index
        self._document_embedder = document_embedder
        self._workspace = workspace
        self._parser_rules = parser_rules
        self._chunk_options = chunk_options or {}
        self._bm25_language_classifier = bm25_language_classifier
        self._telemetry = telemetry
        self._ingest_locks: dict[str, asyncio.Lock] = {}

    async def _process_enqueued(self, doc_ids: list[str]) -> None:
        """Drive the shared queue and wait until this accepted cohort settles.

        LightRAG returns immediately when another owner already holds its single
        processing reservation. The enqueue is still durable and wakes that owner,
        so an immediate finalization read would misclassify the PENDING row as a
        processing failure. Poll the accepted cohort while rows report a known
        active state; missing or unknown rows fall through to finalization's
        existing consistency error instead of waiting forever.
        """
        async with self._telemetry.observe(
            "ingest_pipeline",
            as_type="chain",
            metadata={"document_count": len(doc_ids), "doc_ids": doc_ids},
        ):
            await self._lightrag.apipeline_process_enqueue_documents()
            pending = list(dict.fromkeys(doc_ids))
            delay = _STATUS_POLL_INITIAL_SECONDS
            while pending:
                statuses = await self._stores.get_full_doc_statuses(pending)
                pending = [
                    doc_id
                    for doc_id in pending
                    if (status := statuses.get(doc_id)) is not None
                    and _normalized_status(status) in _ACTIVE_INGEST_STATUSES
                ]
                if not pending:
                    return
                await asyncio.sleep(delay)
                delay = min(delay * 2, _STATUS_POLL_MAX_SECONDS)
                # A prior owner can abort after restoring its cohort to PENDING.
                # Re-drive the queue so this waiter does not depend on a future,
                # unrelated ingest to provide LightRAG's next explicit trigger.
                await self._lightrag.apipeline_process_enqueue_documents()

    async def aingest_file(
        self,
        path: str | Path,
        *,
        source_uri: str | None = None,
        download_locator: str | None = None,
        display_filename: str | None = None,
        source_uri_explicit: bool | None = None,
        download_locator_explicit: bool | None = None,
        display_filename_explicit: bool | None = None,
        replace: bool = False,
        title: str | None = None,
        author: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        track_id: str | None = None,
    ) -> dict[str, Any]:
        """Ingest one file through the same locked compensation core as batches."""
        file_path = Path(path)
        item = PreparedIngestFile(
            parser_path=file_path,
            source_uri=source_uri or _raw_path_source_uri(file_path, workspace=self._workspace),
            download_locator=download_locator or str(file_path.resolve()),
            display_filename=display_filename,
            title=title,
            author=author,
            metadata=metadata,
            source_uri_explicit=(
                source_uri is not None if source_uri_explicit is None else source_uri_explicit
            ),
            download_locator_explicit=(
                download_locator is not None
                if download_locator_explicit is None
                else download_locator_explicit
            ),
            display_filename_explicit=(
                display_filename is not None
                if display_filename_explicit is None
                else display_filename_explicit
            ),
        )
        batch = await self.aingest_files(
            [item], replace=replace, track_id=track_id, _raise_finalization_errors=True
        )
        results = batch.get("results")
        if isinstance(results, list) and results:
            result = results[0]
            if isinstance(result, dict):
                return result
        errors = batch.get("errors")
        if isinstance(errors, list) and errors:
            raise RuntimeError(str(errors[0]))
        raise RuntimeError("document ingestion produced no result")

    async def aingest_files(
        self,
        paths: Sequence[str | Path | PreparedIngestFile],
        *,
        replace: bool = False,
        title: str | None = None,
        author: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        track_id: str | None = None,
        _raise_finalization_errors: bool = False,
    ) -> dict[str, Any]:
        """Ingest local files as one LightRAG staged batch.

        LightRAG 1.5 accepts per-document parser directives, so a single
        enqueue can mix native DOCX parsing and MinerU PDF/image parsing.
        DlightRAG keeps only the product-layer metadata and sidecar vector
        overrides around that native batch pipeline.
        """
        if not paths:
            return {"processed": 0, "errors": [], "results": []}

        entries: list[_PendingDocumentIngest] = []
        for index, item in enumerate(
            _prepare_ingest_item(path, workspace=self._workspace) for path in paths
        ):
            entries.append(
                self._prepare_pending_document(
                    index=index,
                    item=item,
                    title=title,
                    author=author,
                    metadata=metadata,
                    resolve_parser_directives=False,
                )
            )

        doc_ids = [entry.doc_id for entry in entries]
        duplicate_doc_ids = sorted(doc_id for doc_id in set(doc_ids) if doc_ids.count(doc_id) > 1)
        if duplicate_doc_ids:
            raise ValueError(
                "ingest batch contains duplicate canonical document IDs: "
                + ", ".join(duplicate_doc_ids)
            )

        results_by_index: dict[int, dict[str, Any]] = {}
        errors: list[str] = []
        deferred_metadata_updates: list[tuple[_PendingDocumentIngest, dict[str, Any]]] = []
        deferred_finalizations: list[_PendingDocumentIngest] = []
        to_enqueue: list[tuple[_PendingDocumentIngest, _DocumentIngestDecision]] = []

        async with AsyncExitStack() as stack:
            locked_doc_ids = {
                doc_id for entry in entries for doc_id in (entry.doc_id, *entry.replacement_doc_ids)
            }
            for doc_id in sorted(locked_doc_ids):
                await stack.enter_async_context(self._get_ingest_lock(doc_id))

            for entry in entries:
                decision = await self._decide_document_ingest(entry, replace=replace)
                if decision.enqueue:
                    to_enqueue.append((entry, decision))
                    continue
                if decision.finalize_only:
                    deferred_finalizations.append(self._ensure_enqueue_entry(entry))
                    continue
                if decision.metadata_update and decision.result is not None:
                    deferred_metadata_updates.append((entry, decision.result))
                    continue
                if decision.result is not None:
                    results_by_index[entry.index] = decision.result
            validated_to_enqueue = [
                (self._ensure_enqueue_entry(entry), decision) for entry, decision in to_enqueue
            ]

            cleanup_snapshots: dict[str, tuple[dict[str, Any] | None, dict[str, Any] | None]] = {}
            cleanup_ids_by_entry: dict[int, tuple[str, ...]] = {}
            try:
                for entry, decision in validated_to_enqueue:
                    if any(old_doc_id != entry.doc_id for old_doc_id in entry.replacement_doc_ids):
                        await self._validate_candidate_ownership_if_present(entry)
                    cleanup_ids = tuple(
                        dict.fromkeys(
                            (
                                *((entry.doc_id,) if decision.cleanup_kind is not None else ()),
                                *entry.replacement_doc_ids,
                            )
                        )
                    )
                    cleanup_ids_by_entry[entry.index] = cleanup_ids
                    for cleanup_doc_id in cleanup_ids:
                        if cleanup_doc_id in cleanup_snapshots:
                            continue
                        external_replacement = cleanup_doc_id in entry.replacement_doc_ids
                        if external_replacement:
                            await self._validate_replacement_ownership(entry, cleanup_doc_id)
                        cleanup_snapshots[cleanup_doc_id] = await self._cleanup_partial_doc(
                            cleanup_doc_id,
                            allow_metadata_tombstone=external_replacement,
                        )

                for entry, result in deferred_metadata_updates:
                    finalized_metadata = dict(entry.metadata_record)
                    finalized_metadata[_FINALIZATION_COMPLETE_KEY] = True
                    await self._metadata_index.upsert(entry.doc_id, finalized_metadata)
                    results_by_index[entry.index] = result

                for entry in deferred_finalizations:
                    parse_engine, process_options = _required_enqueue_fields(entry)
                    external_ids = tuple(
                        dict.fromkeys(
                            doc_id for doc_id in entry.replacement_doc_ids if doc_id != entry.doc_id
                        )
                    )
                    recovered_snapshots: dict[
                        str, tuple[dict[str, Any] | None, dict[str, Any] | None]
                    ] = {}
                    try:
                        if external_ids:
                            await self._validate_candidate_ownership_if_present(entry)
                        present_external_ids: list[str] = []
                        for external_id in external_ids:
                            current = await self._metadata_index.get(external_id)
                            if current is None:
                                if await self._stores.get_doc_status(external_id) is not None:
                                    raise RuntimeError("replacement document ownership changed")
                                continue
                            await self._validate_replacement_ownership(entry, external_id)
                            present_external_ids.append(external_id)
                        for external_id in present_external_ids:
                            recovered_snapshots[external_id] = await self._cleanup_partial_doc(
                                external_id,
                                allow_metadata_tombstone=True,
                            )
                        durable_result = await self._finalize_ingested_document(
                            doc_id=entry.doc_id,
                            metadata_record=entry.metadata_record,
                            parse_engine=parse_engine,
                            process_options=process_options,
                            commit_complete=not present_external_ids,
                        )
                        for external_id in present_external_ids:
                            await self._metadata_index.delete(external_id)
                            recovered_snapshots.pop(external_id, None)
                        if present_external_ids:
                            await self._commit_finalization_complete(
                                entry.doc_id,
                                entry.metadata_record,
                            )
                        results_by_index[entry.index] = durable_result
                    except BaseException as error:
                        await self._restore_cleanup_after_error(
                            recovered_snapshots,
                            error,
                        )
                        raise

                if validated_to_enqueue:
                    enqueue_entries = [entry for entry, _decision in validated_to_enqueue]
                    chunk_options = self._batch_chunk_options(enqueue_entries)
                    for entry in enqueue_entries:
                        await self._metadata_index.upsert(entry.doc_id, entry.metadata_record)
                    await self._lightrag.apipeline_enqueue_documents(
                        input=[""] * len(enqueue_entries),
                        file_paths=[str(entry.parser_path) for entry in enqueue_entries],
                        docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
                        parse_engine=[entry.parse_engine for entry in enqueue_entries],
                        process_options=[entry.process_options for entry in enqueue_entries],
                        chunk_options=chunk_options,
                        track_id=track_id,
                    )
                    await self._process_enqueued([entry.doc_id for entry in enqueue_entries])

                    for entry in enqueue_entries:
                        try:
                            parse_engine, process_options = _required_enqueue_fields(entry)
                            external_cleanup_ids = tuple(
                                cleanup_doc_id
                                for cleanup_doc_id in cleanup_ids_by_entry.get(entry.index, ())
                                if cleanup_doc_id != entry.doc_id
                            )
                            durable_result = await self._finalize_ingested_document(
                                doc_id=entry.doc_id,
                                metadata_record=entry.metadata_record,
                                parse_engine=parse_engine,
                                process_options=process_options,
                                commit_complete=not external_cleanup_ids,
                            )
                            for cleanup_doc_id in cleanup_ids_by_entry.get(entry.index, ()):
                                if cleanup_doc_id != entry.doc_id:
                                    await self._metadata_index.delete(cleanup_doc_id)
                                cleanup_snapshots.pop(cleanup_doc_id, None)
                            if external_cleanup_ids:
                                # Replacement completion is authoritative only
                                # after every old locator owner is retired.
                                await self._commit_finalization_complete(
                                    entry.doc_id,
                                    entry.metadata_record,
                                )
                            # Publish only after old locator metadata retirement
                            # and the final completion marker commit.
                            results_by_index[entry.index] = durable_result
                        except BaseException as error:  # noqa: BLE001
                            entry_snapshots = {
                                cleanup_doc_id: cleanup_snapshots[cleanup_doc_id]
                                for cleanup_doc_id in cleanup_ids_by_entry.get(entry.index, ())
                                if cleanup_doc_id in cleanup_snapshots
                            }
                            await self._settle_failed_replacement(entry, entry_snapshots, error)
                            for cleanup_doc_id in entry_snapshots:
                                cleanup_snapshots.pop(cleanup_doc_id, None)
                            if isinstance(error, asyncio.CancelledError):
                                raise
                            if _raise_finalization_errors:
                                raise
                            filename = safe_source_filename(
                                str(entry.metadata_record.get("filename") or entry.parser_path.name)
                            )
                            logger.warning(
                                "Document finalization failed for %s", filename, exc_info=True
                            )
                            errors.append(f"{filename}: document processing failed")
            except BaseException as error:
                # Enqueue/process can partially create candidate rows before it
                # raises. Settle every affected replacement, rather than merely
                # restoring old snapshots and leaving two locator owners.
                for entry, _decision in validated_to_enqueue:
                    entry_snapshots = {
                        cleanup_doc_id: cleanup_snapshots[cleanup_doc_id]
                        for cleanup_doc_id in cleanup_ids_by_entry.get(entry.index, ())
                        if cleanup_doc_id in cleanup_snapshots
                    }
                    if not entry_snapshots:
                        continue
                    await self._settle_failed_replacement(entry, entry_snapshots, error)
                    for cleanup_doc_id in entry_snapshots:
                        cleanup_snapshots.pop(cleanup_doc_id, None)
                await self._restore_cleanup_after_error(cleanup_snapshots, error)
                raise

        return {
            "processed": len(results_by_index),
            "errors": errors,
            "results": [results_by_index[index] for index in sorted(results_by_index)],
        }

    def _prepare_pending_document(
        self,
        *,
        index: int,
        item: PreparedIngestFile,
        title: str | None = None,
        author: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        resolve_parser_directives: bool = True,
    ) -> _PendingDocumentIngest:
        effective_title = item.title if item.title is not None else title
        effective_author = item.author if item.author is not None else author
        effective_metadata = _overlay_metadata(metadata, item.metadata)
        parse_engine: str | None = None
        process_options: str | None = None
        chunk_options: dict[str, Any] | None = None
        if resolve_parser_directives:
            parse_engine, process_options, chunk_options = self._parser_directives_for(
                item.parser_path
            )
        return _PendingDocumentIngest(
            index=index,
            parser_path=item.parser_path,
            doc_id=_canonical_file_doc_id(item.parser_path),
            metadata_record=self._prepare_metadata_record(
                item.parser_path,
                source_uri=item.source_uri,
                download_locator=item.download_locator,
                display_filename=item.display_filename,
                title=effective_title,
                author=effective_author,
                metadata=effective_metadata,
            ),
            metadata_update_requested=(
                _source_contract_update_requested(item)
                or effective_title is not None
                or effective_author is not None
                or effective_metadata is not None
            ),
            parse_engine=parse_engine,
            process_options=process_options,
            chunk_options=chunk_options,
            replacement_doc_ids=tuple(dict.fromkeys(item.replacement_doc_ids)),
            replacement_ownership=tuple(dict.fromkeys(item.replacement_ownership)),
        )

    def _ensure_enqueue_entry(self, entry: _PendingDocumentIngest) -> _PendingDocumentIngest:
        if entry.parse_engine is not None and entry.process_options is not None:
            return entry
        parse_engine, process_options, chunk_options = self._parser_directives_for(
            entry.parser_path
        )
        return _PendingDocumentIngest(
            index=entry.index,
            parser_path=entry.parser_path,
            doc_id=entry.doc_id,
            metadata_record=entry.metadata_record,
            metadata_update_requested=entry.metadata_update_requested,
            parse_engine=parse_engine,
            process_options=process_options,
            chunk_options=chunk_options,
            replacement_doc_ids=entry.replacement_doc_ids,
            replacement_ownership=entry.replacement_ownership,
        )

    async def _decide_document_ingest(
        self,
        entry: _PendingDocumentIngest,
        *,
        replace: bool,
    ) -> _DocumentIngestDecision:
        existing_status = await self._stores.get_doc_status(entry.doc_id)
        if existing_status is None:
            return _DocumentIngestDecision(enqueue=True)

        if replace:
            return _DocumentIngestDecision(enqueue=True, cleanup_kind="replace")

        hash_match_decision = await self._hash_match_decision(entry, existing_status)
        if hash_match_decision is not None:
            return hash_match_decision

        if _normalized_status(existing_status) == "processed":
            # LightRAG rejects an existing canonical ID as a duplicate. A hash
            # mismatch must therefore delete the old corpus before enqueue.
            return _DocumentIngestDecision(enqueue=True, cleanup_kind="replace")

        return _DocumentIngestDecision(
            enqueue=True,
            cleanup_kind="partial",
        )

    async def _hash_match_decision(
        self,
        entry: _PendingDocumentIngest,
        existing_status: Mapping[str, Any] | None,
    ) -> _DocumentIngestDecision | None:
        stored_hash = _mapping_get(existing_status, "content_hash")
        if _normalized_status(existing_status) != "processed" or not stored_hash:
            return None
        current_hash = await asyncio.to_thread(_file_sha256, entry.parser_path)
        if current_hash != stored_hash:
            return None

        chunks = list(_mapping_get(existing_status, "chunks_list") or [])
        persisted = await self._metadata_index.get(entry.doc_id)
        if not ingest_finalization_complete(persisted):
            # LightRAG may have committed PROCESSED immediately before a hard
            # crash. Replay only DlightRAG's idempotent required finalization.
            return _DocumentIngestDecision(enqueue=False, finalize_only=True)

        if not await self._metadata_update_required(entry, persisted=persisted):
            return _DocumentIngestDecision(
                enqueue=False,
                result={
                    "doc_id": entry.doc_id,
                    "source_kind": "skipped",
                    "reason": "content_hash_match",
                    "chunks": chunks,
                },
            )

        return _DocumentIngestDecision(
            enqueue=False,
            metadata_update=True,
            result={
                "doc_id": entry.doc_id,
                "source_kind": "metadata_updated",
                "reason": "content_hash_match",
                "chunks": chunks,
            },
        )

    async def _metadata_update_required(
        self,
        entry: _PendingDocumentIngest,
        *,
        persisted: Mapping[str, Any] | None = None,
    ) -> bool:
        if not entry.metadata_update_requested:
            return False
        if persisted is None:
            persisted = await self._metadata_index.get(entry.doc_id)
        if not isinstance(persisted, Mapping):
            return True
        return _hash_match_metadata_record(entry.metadata_record) != _hash_match_metadata_record(
            persisted
        )

    def _prepare_metadata_record(
        self,
        file_path: Path,
        *,
        source_uri: str,
        download_locator: str,
        display_filename: str | None = None,
        title: str | None,
        author: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        normalized_metadata = normalize_user_metadata(metadata)
        system_metadata = extract_system_metadata(
            download_locator,
            display_filename=display_filename,
            source_uri=source_uri,
            download_locator=download_locator,
        )
        if title is not None:
            system_metadata["title"] = title
        if author is not None:
            system_metadata["author"] = author
        system_metadata.update(normalized_metadata.system)
        return {
            **system_metadata,
            "custom_metadata": normalized_metadata.custom_metadata,
            # Application-owned crash journal; user metadata cannot override it.
            _FINALIZATION_COMPLETE_KEY: False,
        }

    def _get_ingest_lock(self, doc_id: str) -> asyncio.Lock:
        """Return a per-doc async lock to serialize concurrent ingests of the same file."""
        lock = self._ingest_locks.get(doc_id)
        if lock is None:
            lock = asyncio.Lock()
            self._ingest_locks[doc_id] = lock
        return lock

    async def _settle_failed_replacement(
        self,
        entry: _PendingDocumentIngest,
        snapshots: dict[str, tuple[dict[str, Any] | None, dict[str, Any] | None]],
        error: BaseException,
    ) -> None:
        """Leave exactly one retry identity after an old-ID replacement fails."""
        external = tuple(doc_id for doc_id in entry.replacement_doc_ids if doc_id != entry.doc_id)
        if not external:
            await self._restore_cleanup_after_error(snapshots, error)
            return

        candidate_status = await self._stores.get_doc_status(entry.doc_id)
        # A metadata-only old tombstone cannot safely be recreated. Once the
        # candidate has a status, it is the authoritative failed identity.
        if any(snapshots.get(doc_id, (None, None))[0] is None for doc_id in external):
            if candidate_status is not None:
                for old_doc_id in external:
                    await self._metadata_index.delete(old_doc_id)
                return

            # Enqueue failed before it created a candidate status. The intent
            # metadata written immediately before enqueue must not overwrite a
            # pre-existing canonical candidate that owned another locator.
            candidate_snapshot = snapshots.get(entry.doc_id)
            if candidate_snapshot is not None and candidate_snapshot[0] is not None:
                await self._restore_cleanup_snapshots({entry.doc_id: candidate_snapshot})
            else:
                await self._metadata_index.delete(entry.doc_id)

            # Every external ID was ownership-validated for the incoming
            # locator under its lock. Retain one deterministic retry tombstone
            # and retire only surplus owners, restoring its status when one was
            # available before cleanup.
            authoritative_old = external[0]
            authoritative_snapshot = snapshots.get(authoritative_old)
            if authoritative_snapshot is not None and authoritative_snapshot[0] is not None:
                await self._restore_cleanup_snapshots({authoritative_old: authoritative_snapshot})
            for old_doc_id in external[1:]:
                await self._metadata_index.delete(old_doc_id)
            return

        if candidate_status is None:
            await self._metadata_index.delete(entry.doc_id)
            await self._restore_cleanup_after_error(snapshots, error)
            return

        cancellation = (
            defer_cancellation(None, error) if isinstance(error, asyncio.CancelledError) else None
        )
        delete_task = asyncio.create_task(
            self._lightrag.adelete_by_doc_id(entry.doc_id, delete_llm_cache=True)
        )
        try:
            deletion = await await_shared_cleanup(delete_task)
        except asyncio.CancelledError as exc:
            cancellation = defer_cancellation(cancellation, exc)
            deletion = None
        except Exception:
            deletion = None
        deletion_status = str(_mapping_get(deletion, "status") or "").lower()
        candidate_gone = await self._stores.get_doc_status(entry.doc_id) is None
        if deletion_status in {"success", "not_found"} or candidate_gone:
            await self._metadata_index.delete(entry.doc_id)
            await self._restore_cleanup_snapshots(snapshots)
        else:
            # Candidate remains authoritative; retire only the already-deleted
            # old locator tombstones so the next retry cannot see two owners.
            for old_doc_id in external:
                await self._metadata_index.delete(old_doc_id)
        # The caller re-raises the original cancellation after removing these
        # settled snapshots from the outer compensation set.

    async def _validate_candidate_ownership_if_present(self, entry: _PendingDocumentIngest) -> None:
        """Never replace an unrelated pre-existing canonical candidate."""
        current = await self._metadata_index.get(entry.doc_id)
        if current is None:
            if await self._stores.get_doc_status(entry.doc_id) is not None:
                raise RuntimeError("replacement candidate ownership changed")
            return
        if not isinstance(current, Mapping):
            raise RuntimeError("replacement candidate ownership changed")
        expected_locator = entry.metadata_record.get("download_locator")
        expected_source = entry.metadata_record.get("source_uri")
        if (
            current.get("download_locator") != expected_locator
            or current.get("source_uri") != expected_source
        ):
            await self._collapse_safe_external_tombstones(entry)
            raise RuntimeError("replacement candidate ownership changed")

    async def _collapse_safe_external_tombstones(
        self,
        entry: _PendingDocumentIngest,
    ) -> None:
        """Retain one proven metadata-only owner without touching corpus rows."""
        external = tuple(
            dict.fromkeys(doc_id for doc_id in entry.replacement_doc_ids if doc_id != entry.doc_id)
        )
        if len(external) < 2:
            return

        # Validate every owner and every status before the first mutation. If
        # any row is uncertain or status-bearing, candidate collision remains
        # a pure fail-closed no-op.
        for doc_id in external:
            await self._validate_replacement_ownership(entry, doc_id)
            if await self._stores.get_doc_status(doc_id) is not None:
                return
        for surplus_doc_id in external[1:]:
            await self._metadata_index.delete(surplus_doc_id)

    async def _validate_replacement_ownership(
        self, entry: _PendingDocumentIngest, doc_id: str
    ) -> None:
        """Prove locator ownership under the already-acquired document lock."""
        accepted = tuple(
            (locator, source_uri)
            for owned_id, locator, source_uri in entry.replacement_ownership
            if owned_id == doc_id
        )
        if not accepted:
            raise RuntimeError("replacement document ownership is unavailable")
        current = await self._metadata_index.get(doc_id)
        if not isinstance(current, Mapping):
            raise RuntimeError("replacement document ownership changed")
        if not any(
            current.get("download_locator") == locator and current.get("source_uri") == source_uri
            for locator, source_uri in accepted
        ):
            raise RuntimeError("replacement document ownership changed")

    async def _restore_cleanup_snapshots(
        self,
        snapshots: dict[str, tuple[dict[str, Any] | None, dict[str, Any] | None]],
    ) -> None:
        """Retain only a DlightRAG-owned hidden repair identity.

        LightRAG document status is upstream state and is never reconstructed
        from a local snapshot after a destructive handoff.
        """
        for doc_id, (_status_snapshot, metadata_snapshot) in snapshots.items():
            if metadata_snapshot is None:
                continue
            current_metadata = await self._metadata_index.get(doc_id)
            if current_metadata is None:
                unpublished_snapshot = dict(metadata_snapshot)
                unpublished_snapshot[_FINALIZATION_COMPLETE_KEY] = False
                await self._metadata_index.upsert(doc_id, unpublished_snapshot)

    async def _restore_cleanup_after_error(
        self,
        snapshots: dict[str, tuple[dict[str, Any] | None, dict[str, Any] | None]],
        error: BaseException,
    ) -> None:
        cancellation = (
            defer_cancellation(None, error) if isinstance(error, asyncio.CancelledError) else None
        )
        cleanup_task = asyncio.create_task(self._restore_cleanup_snapshots(snapshots))
        try:
            await await_shared_cleanup(cleanup_task)
        except asyncio.CancelledError as exc:
            cancellation = defer_cancellation(cancellation, exc)
        except Exception:
            logger.warning("Failed to restore cleaned document snapshots", exc_info=True)
        if cancellation is not None:
            raise cancellation from None

    async def _cleanup_partial_doc(
        self,
        doc_id: str,
        *,
        allow_metadata_tombstone: bool = False,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Remove a document, accepting only a validated metadata-only tombstone."""
        status_snapshot = await self._stores.get_doc_status(doc_id)
        metadata_snapshot = await self._metadata_index.get(doc_id)
        # Publication must stop before the first destructive upstream effect.
        await self._metadata_index.upsert(
            doc_id,
            {_FINALIZATION_COMPLETE_KEY: False},
        )
        if not isinstance(status_snapshot, Mapping):
            if allow_metadata_tombstone and isinstance(metadata_snapshot, Mapping):
                # A prior owner committed deletion but crashed before enqueue.
                # Ownership was revalidated under this doc lock by the caller.
                return None, dict(metadata_snapshot)
            raise RuntimeError("document status snapshot is unavailable")
        full_doc = await self._stores.get_full_doc(doc_id)
        sidecar_uri = _mapping_get(full_doc, "sidecar_location")

        try:
            result = await self._lightrag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
        except BaseException as error:
            current_status = await self._stores.get_doc_status(doc_id)
            if current_status is not None:
                await self._restore_cleanup_after_error(
                    {
                        doc_id: (
                            dict(status_snapshot),
                            dict(metadata_snapshot)
                            if isinstance(metadata_snapshot, Mapping)
                            else None,
                        )
                    },
                    error,
                )
            raise
        raw_status = _mapping_get(result, "status")
        status = str(getattr(raw_status, "value", raw_status) or "").strip().lower()
        if status != "success":
            current_status = await self._stores.get_doc_status(doc_id)
            if current_status is None:
                # Pinned LightRAG deletes doc_status first at its final stage and
                # intentionally treats a later retry as already deleted. Never
                # recreate that row as a zombie.
                return (
                    None,
                    dict(metadata_snapshot) if isinstance(metadata_snapshot, Mapping) else None,
                )
            await self._restore_cleanup_snapshots(
                {
                    doc_id: (
                        dict(status_snapshot),
                        dict(metadata_snapshot) if isinstance(metadata_snapshot, Mapping) else None,
                    )
                }
            )
            raise RuntimeError("LightRAG document deletion was not acknowledged")

        # Keep source metadata until the replacement record is durably written.
        # A hard process exit after deletion can then reconstruct the same ID.
        artifact_dir = resolve_sidecar_uri(sidecar_uri)
        if artifact_dir is not None and artifact_dir.exists():
            shutil.rmtree(artifact_dir, ignore_errors=True)
        return (
            dict(status_snapshot),
            dict(metadata_snapshot) if isinstance(metadata_snapshot, Mapping) else None,
        )

    async def _finalize_ingested_document(
        self,
        *,
        doc_id: str,
        metadata_record: dict[str, Any],
        parse_engine: str,
        process_options: str,
        commit_complete: bool = True,
    ) -> dict[str, Any]:
        doc_status = await self._stores.get_doc_status(doc_id)
        if _normalized_status(doc_status) != "processed":
            error_summary = (doc_status or {}).get("error_msg") or (doc_status or {}).get(
                "content_summary"
            )
            raise RuntimeError(str(error_summary or "LightRAG document processing failed"))

        full_doc = await self._stores.get_full_doc(doc_id)
        light_chunks = list((doc_status or {}).get("chunks_list") or [])
        finalized_metadata = _with_finalized_local_download_locator(metadata_record)
        finalized_metadata[_FINALIZATION_COMPLETE_KEY] = commit_complete
        await self._overwrite_sidecar_image_vectors(
            doc_id=doc_id,
            sidecar_location=_mapping_get(full_doc, "sidecar_location"),
            chunk_ids=set(light_chunks),
        )
        await self._label_bm25_languages(light_chunks)
        # The DlightRAG-owned marker is the only publication commit point.
        # LightRAG's upstream PROCESSED status remains untouched when any
        # idempotent finalizer fails, so retry re-enters finalization only.
        await self._metadata_index.upsert(doc_id, finalized_metadata)

        return {
            "doc_id": doc_id,
            "source_kind": "document",
            "chunks": light_chunks,
            "parse_engine": parse_engine,
            "process_options": process_options,
        }

    async def _commit_finalization_complete(
        self,
        doc_id: str,
        metadata_record: dict[str, Any],
    ) -> None:
        """Commit replacement completion only after old-owner retirement."""
        try:
            doc_status = await self._stores.get_doc_status(doc_id)
        except Exception as error:
            raise RetryOutcomeUncertainError(
                "replacement finalization status is uncertain"
            ) from error
        if _normalized_status(doc_status) != "processed":
            raise RetryOutcomeUncertainError("replacement finalization status is uncertain")
        completed = _with_finalized_local_download_locator(metadata_record)
        completed[_FINALIZATION_COMPLETE_KEY] = True
        await self._metadata_index.upsert(doc_id, completed)

    def _parser_directives_for(self, file_path: Path) -> tuple[str, str, dict[str, Any] | None]:
        directives = resolve_parser_directives(
            file_path,
            parser_rules=self._parser_rules,
            require_external_endpoint=False,
        )
        parse_engine = encode_parse_engine(directives.engine, directives.engine_params)
        chunk_options = self._chunk_options_for_directives(
            directives.process_options,
            directives.chunk_params,
        )
        return parse_engine, directives.process_options, chunk_options

    def _chunk_options_for_directives(
        self,
        process_options: str,
        chunk_params: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not chunk_params:
            return None

        merged = self._base_chunk_options(process_options)
        for selector, params in chunk_params.items():
            key = chunk_strategy_key(selector)
            current = merged.get(key)
            merged[key] = {
                **(current if isinstance(current, dict) else {}),
                **params,
            }
        return merged

    def _base_chunk_options(self, process_options: str) -> dict[str, Any]:
        if self._chunk_options:
            return deepcopy(self._chunk_options)
        return resolve_chunk_options(
            getattr(self._lightrag, "addon_params", None),
            process_options=process_options,
        )

    def _batch_chunk_options(
        self,
        entries: Sequence[_PendingDocumentIngest],
    ) -> list[dict[str, Any]] | dict[str, Any] | None:
        if not any(entry.chunk_options is not None for entry in entries):
            return deepcopy(self._chunk_options) if self._chunk_options else None
        options: list[dict[str, Any]] = []
        for entry in entries:
            if entry.chunk_options is not None:
                options.append(entry.chunk_options)
                continue
            _, process_options = _required_enqueue_fields(entry)
            options.append(self._base_chunk_options(process_options))
        return options

    async def _overwrite_sidecar_image_vectors(
        self,
        *,
        doc_id: str,
        sidecar_location: str | None,
        chunk_ids: set[str],
    ) -> None:
        if not self._document_embedder.image_enabled:
            return
        # A unified multimodal embedder (image support probed at startup) fuses the
        # VLM description with the image into one vector, keeping the visual chunk
        # reachable by text queries.
        artifact_dir = resolve_sidecar_uri(sidecar_location)
        if artifact_dir is None or not artifact_dir.exists():
            return

        assets = [
            asset
            for asset in collect_lightrag_drawing_assets(artifact_dir, doc_id=doc_id)
            if asset.chunk_id in chunk_ids and asset.image_path.exists()
        ]
        if not assets:
            return

        descriptions = await self._fetch_chunk_descriptions([a.chunk_id for a in assets])
        inputs = [
            DocumentEmbeddingInput(
                key=asset.chunk_id,
                text=descriptions.get(asset.chunk_id, ""),
                image_path=asset.image_path,
            )
            for asset in assets
        ]
        embedded, trace = await self._document_embedder.aembed_documents(inputs)
        if (
            trace.fused != len(inputs)
            or trace.text
            or trace.fused_to_text_fallback
            or trace.failed
            or any(item.mode != "fused" for item in embedded)
            or {item.key for item in embedded} != {item.key for item in inputs}
        ):
            raise RuntimeError("required sidecar visual fusion did not complete")
        logger.debug(
            "Sidecar document embedding outcomes: fused=%d text=%d fallback=%d failed=%d",
            trace.fused,
            trace.text,
            trace.fused_to_text_fallback,
            trace.failed,
        )
        vectors = {item.key: item.vector for item in embedded}
        if vectors:
            await self._stores.overwrite_chunk_vectors(
                vectors,
                embedding_dim=self._document_embedder.dimension,
            )

    async def _fetch_chunk_descriptions(self, chunk_ids: list[str]) -> dict[str, str]:
        """Return {chunk_id: VLM description} to fuse into visual-chunk vectors."""
        rows = await self._stores.fetch_chunk_contents(chunk_ids)
        return {str(row["id"]): str(row.get("content") or "") for row in rows if row.get("id")}

    async def _label_bm25_languages(self, chunk_ids: list[str]) -> None:
        classifier = self._bm25_language_classifier
        if classifier is None or not chunk_ids:
            return
        rows = await self._stores.fetch_chunk_contents(chunk_ids)

        # lingua n-gram detection is CPU-bound and runs over every chunk of the
        # document, so keep it off the loop serving concurrent queries.
        def _detect() -> dict[str, str]:
            return {
                str(row["id"]): classifier.detect(str(row.get("content") or ""))
                for row in rows
                if row.get("id")
            }

        labels = await asyncio.to_thread(_detect)
        if labels:
            await self._stores.update_chunk_bm25_languages(labels)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _prepare_ingest_item(
    path: str | Path | PreparedIngestFile, *, workspace: str
) -> PreparedIngestFile:
    if isinstance(path, PreparedIngestFile):
        return path
    parser_path = Path(path)
    return PreparedIngestFile(
        parser_path=parser_path,
        source_uri=_raw_path_source_uri(parser_path, workspace=workspace),
        download_locator=str(parser_path.resolve()),
        source_uri_explicit=False,
        download_locator_explicit=False,
    )


def _with_finalized_local_download_locator(
    metadata_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist the exact source location after LightRAG archives parser input."""
    finalized = dict(metadata_record)
    locator = finalized.get("download_locator")
    if not isinstance(locator, str) or "://" in locator:
        return finalized

    source_path = Path(locator)
    archived_path = lightrag_archived_source_path(source_path)
    if source_path.is_file() or not archived_path.is_file():
        return finalized

    resolved = str(archived_path.resolve())
    finalized["download_locator"] = resolved
    return finalized


def _hash_match_metadata_record(metadata_record: Mapping[str, Any]) -> dict[str, Any]:
    comparable = {
        "filename": metadata_record.get("filename"),
        "filename_stem": metadata_record.get("filename_stem"),
        "source_uri": metadata_record.get("source_uri"),
        "download_locator": metadata_record.get("download_locator"),
        "file_extension": metadata_record.get("file_extension"),
        "title": metadata_record.get("title"),
        "author": metadata_record.get("author"),
        "creation_date": metadata_record.get("creation_date"),
        "custom_metadata": deepcopy(metadata_record.get("custom_metadata")) or {},
    }
    comparable["download_locator"] = _canonicalize_local_metadata_locator(
        comparable["download_locator"]
    )
    return comparable


def _canonicalize_local_metadata_locator(locator: Any) -> Any:
    if not isinstance(locator, str) or "://" in locator:
        return locator
    return str(lightrag_archived_source_path(Path(locator)).resolve())


def _raw_path_source_uri(path: Path, *, workspace: str) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
    return local_source_uri(workspace, Path(digest) / path.name)


def _source_contract_update_requested(item: PreparedIngestFile) -> bool:
    return (
        item.source_uri_explicit or item.download_locator_explicit or item.display_filename_explicit
    )


def _overlay_metadata(
    defaults: Mapping[str, Any] | None,
    overlay: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if defaults is None:
        return overlay
    if overlay is None:
        return defaults
    return {**defaults, **overlay}


def _mapping_get(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _normalized_status(value: Any) -> str:
    raw_status = _mapping_get(value, "status")
    return str(getattr(raw_status, "value", raw_status) or "").lower()


def _required_enqueue_fields(entry: _PendingDocumentIngest) -> tuple[str, str]:
    if entry.parse_engine is None or entry.process_options is None:
        raise RuntimeError("ingest decision resolved without parser directives")
    return entry.parse_engine, entry.process_options


def _canonical_file_doc_id(path: Path) -> str:
    """Match LightRAG's file-backed document id derivation."""
    return compute_mdhash_id(normalize_document_file_path(path), prefix="doc-")


__all__ = [
    "PreparedIngestFile",
    "UnifiedIngestionEngine",
]
