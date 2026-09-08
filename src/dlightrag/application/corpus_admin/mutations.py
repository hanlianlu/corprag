# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable Corpus Mutation acceptance and execution on the common RunRuntime."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import os
import shutil
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast
from uuid import UUID, uuid7

from dlightrag.application.runs import (
    IdempotencyKeyConflict,
    RunAdmissionLimitExceededError,
    RunCreation,
    RunRuntimeUnavailableError,
)
from dlightrag.engine.dependencies import (
    DependencyComponent,
    classify_transient_dependency,
    next_dependency_retry,
)
from dlightrag.engine.rag.corpus.ingestion.errors import RetryOutcomeUncertainError
from dlightrag.engine.rag.corpus.ingestion.uploads import safe_upload_relative_path
from dlightrag.engine.rag.workspace.pool import WorkspacePool
from dlightrag.engine.rag.workspace.ports import CorpusMaintenanceStore, WorkspaceWriteFencedError
from dlightrag.engine.rag.workspace.workspaces import require_canonical_workspace_id
from dlightrag.engine.runtime.coordinator import RunExecutor, RunSession
from dlightrag.engine.runtime.policy import CORPUS_MUTATION_RUN_RETENTION_SECONDS
from dlightrag.engine.runtime.records import (
    Deferred,
    Failed,
    PreparedInputTooLargeError,
    PreparedRunEnvelope,
    RunAccessScope,
    RunExecutionOutcome,
    Succeeded,
    WaitingForRepair,
    require_prepared_input_bounds,
    run_request_fingerprint,
)
from dlightrag.engine.runtime.records import (
    IdempotencyKeyConflict as RuntimeIdempotencyKeyConflict,
)
from dlightrag.engine.runtime.records import (
    RunAdmissionLimitExceededError as RuntimeRunAdmissionLimitExceededError,
)

from .errors import UnsafeUploadNameError, UploadTooLargeError
from .service import IngestSpec, safe_upload_basename

type CorpusMutationAction = Literal["ingest", "replace", "delete", "retry", "reset"]
type RetrySelector = Literal["all_retryable"]

_REPAIR_REASON = "The upstream corpus outcome is not safe to repeat automatically."
_REPAIR_REMEDY = "Inspect the public LightRAG state, repair it, then resume this Run."
_MAX_RESULT_DOCUMENTS = 100
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_DEFER_BASE_SECONDS = 2
_DEFER_MAX_SECONDS = 60


class CorpusMutationStore(Protocol):
    async def replay_run(
        self,
        *,
        owner_id: str,
        idempotency_key: str,
        idempotency_fingerprint: str,
        run_kind: Literal["corpus_mutation"],
    ) -> Any: ...

    async def accept_run(self, *, envelope: PreparedRunEnvelope, run_id: str) -> Any: ...

    async def record_corpus_window(
        self,
        *,
        run_id: str,
        workspace: str,
        window_number: int,
        docs: int,
        chunks: int,
    ) -> bool: ...


class CorpusMutationScheduler(Protocol):
    @property
    def is_started(self) -> bool: ...

    def admission(self) -> Any: ...
    def wake(self) -> None: ...


@dataclass(frozen=True, slots=True)
class StagedCorpusSource:
    """One bounded upload atomically committed to a Run-exclusive source path."""

    path: Path
    filename: str
    content_sha256: str
    size_bytes: int


class CorpusMutationService:
    """Accept all product corpus writes as generic ``corpus_mutation`` Runs."""

    def __init__(
        self,
        *,
        input_root: Path,
        store: CorpusMutationStore,
        coordinator: CorpusMutationScheduler,
    ) -> None:
        self._input_root = Path(input_root)
        self._store = store
        self._coordinator = coordinator

    async def replay(
        self,
        *,
        submitted_by: str,
        idempotency_key: str | None,
        normalized_request: Mapping[str, Any],
    ) -> RunCreation | None:
        """Resolve a supplied digest/key before receiving an upload body."""
        if idempotency_key is None:
            return None
        fingerprint = run_request_fingerprint(normalized_request)
        try:
            replay = await self._store.replay_run(
                owner_id=submitted_by,
                idempotency_key=idempotency_key,
                idempotency_fingerprint=fingerprint,
                run_kind="corpus_mutation",
            )
        except RuntimeIdempotencyKeyConflict as exc:
            raise IdempotencyKeyConflict(str(exc)) from exc
        return RunCreation.from_runtime(replay) if replay is not None else None

    async def create_ingest(
        self,
        *,
        workspace: str,
        spec: IngestSpec,
        submitted_by: str,
        idempotency_key: str | None = None,
    ) -> RunCreation:
        action: CorpusMutationAction = "replace" if bool(spec.replace) else "ingest"
        request = {
            "action": action,
            "workspace": require_canonical_workspace_id(workspace),
            "source": spec.model_dump(mode="json", exclude_none=True),
        }
        if spec.source_type != "local":
            replay = await self.replay(
                submitted_by=submitted_by,
                idempotency_key=idempotency_key,
                normalized_request=request,
            )
            if replay is not None:
                return replay

        run_id = str(uuid7())
        execution_spec = spec
        staged_root: Path | None = None
        staged_sources: list[dict[str, Any]] = []
        if spec.source_type == "local":
            execution_spec, staged_root, staged_sources = await asyncio.to_thread(
                self._snapshot_local_spec, run_id, workspace, spec
            )
        normalized_request = {
            **request,
            **(
                {
                    "staged_sources": [
                        {
                            "content_sha256": item["content_sha256"],
                            "size_bytes": item["size_bytes"],
                        }
                        for item in staged_sources
                    ]
                }
                if staged_sources
                else {}
            ),
        }
        if staged_root is not None:
            replay = await self.replay(
                submitted_by=submitted_by,
                idempotency_key=idempotency_key,
                normalized_request=normalized_request,
            )
            if replay is not None:
                await asyncio.to_thread(shutil.rmtree, staged_root, True)
                return replay
        payload = {
            **request,
            "source": execution_spec.model_dump(mode="json", exclude_none=True),
            "staged_sources": staged_sources,
            "track_id": _track_id(run_id),
        }
        try:
            creation = await self._accept(
                run_id=run_id,
                workspace=workspace,
                submitted_by=submitted_by,
                idempotency_key=idempotency_key,
                normalized_request=normalized_request,
                payload=payload,
            )
        except BaseException:
            if staged_root is not None:
                await asyncio.to_thread(shutil.rmtree, staged_root, True)
            raise
        if staged_root is not None and creation.replayed and creation.run.run_id != run_id:
            await asyncio.to_thread(shutil.rmtree, staged_root, True)
        return creation

    async def create_staged_ingest(
        self,
        *,
        workspace: str,
        staged: StagedCorpusSource,
        submitted_by: str,
        idempotency_key: str | None = None,
        title: str | None = None,
        author: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        replace: bool = False,
    ) -> RunCreation:
        run_id = staged.path.parents[1].name
        action: CorpusMutationAction = "replace" if replace else "ingest"
        source_identity = {
            "source_type": "local",
            "filename": staged.filename,
            "content_sha256": staged.content_sha256,
            "size_bytes": staged.size_bytes,
            **({"title": title} if title is not None else {}),
            **({"author": author} if author is not None else {}),
            **({"metadata": dict(metadata)} if metadata is not None else {}),
        }
        request = {
            "action": action,
            "workspace": require_canonical_workspace_id(workspace),
            "source": source_identity,
        }
        payload = {
            **request,
            "source": {
                "source_type": "local",
                "path": str(staged.path),
                "replace": replace,
                **({"title": title} if title is not None else {}),
                **({"author": author} if author is not None else {}),
                **({"metadata": dict(metadata)} if metadata is not None else {}),
            },
            "staged_sources": [_staged_source_record(staged)],
            "track_id": _track_id(run_id),
        }
        try:
            creation = await self._accept(
                run_id=run_id,
                workspace=workspace,
                submitted_by=submitted_by,
                idempotency_key=idempotency_key,
                normalized_request=request,
                payload=payload,
            )
        except BaseException:
            await asyncio.to_thread(shutil.rmtree, staged.path.parents[1], True)
            raise
        if creation.replayed and creation.run.run_id != run_id:
            await asyncio.to_thread(shutil.rmtree, staged.path.parents[1], True)
        return creation

    async def create_staged_batch(
        self,
        *,
        workspace: str,
        staged: Sequence[StagedCorpusSource],
        submitted_by: str,
        idempotency_key: str | None = None,
        replace: bool = False,
    ) -> RunCreation:
        """Accept one already-staged multipart cohort as one ingest or replace Run."""
        if not staged:
            raise ValueError("at least one staged source is required")
        run_id = staged[0].path.parents[1].name
        if any(item.path.parents[1].name != run_id for item in staged):
            raise ValueError("staged sources do not belong to one Run")
        action: CorpusMutationAction = "replace" if replace else "ingest"
        request = {
            "action": action,
            "workspace": require_canonical_workspace_id(workspace),
            "sources": [
                {
                    "filename": item.filename,
                    "content_sha256": item.content_sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in staged
            ],
        }
        payload = {
            **request,
            "source": {
                "source_type": "local",
                "path": str(staged[0].path.parent),
                "replace": replace,
            },
            "staged_sources": [_staged_source_record(item) for item in staged],
            "track_id": _track_id(run_id),
        }
        run_root = staged[0].path.parents[1]
        try:
            creation = await self._accept(
                run_id=run_id,
                workspace=workspace,
                submitted_by=submitted_by,
                idempotency_key=idempotency_key,
                normalized_request=request,
                payload=payload,
            )
        except BaseException:
            await asyncio.to_thread(shutil.rmtree, run_root, True)
            raise
        if creation.replayed and creation.run.run_id != run_id:
            await asyncio.to_thread(shutil.rmtree, run_root, True)
        return creation

    async def create_delete(
        self,
        *,
        workspace: str,
        submitted_by: str,
        file_paths: Sequence[str] = (),
        filenames: Sequence[str] = (),
        document_ids: Sequence[str] = (),
        idempotency_key: str | None = None,
    ) -> RunCreation:
        selectors = {
            "file_paths": _bounded_unique(file_paths),
            "filenames": _bounded_unique(filenames),
            "document_ids": _bounded_unique(document_ids),
        }
        if not any(selectors.values()):
            raise ValueError("at least one exact document identifier is required")
        return await self._create_action(
            action="delete",
            workspace=workspace,
            submitted_by=submitted_by,
            idempotency_key=idempotency_key,
            fields=selectors,
        )

    async def create_retry(
        self,
        *,
        workspace: str,
        submitted_by: str,
        document_ids: Sequence[str] = (),
        selector: RetrySelector | None = None,
        idempotency_key: str | None = None,
    ) -> RunCreation:
        ids = _bounded_unique(document_ids)
        if bool(ids) == bool(selector):
            raise ValueError("provide document_ids or selector='all_retryable', but not both")
        if selector not in {None, "all_retryable"}:
            raise ValueError("unknown retry selector")
        return await self._create_action(
            action="retry",
            workspace=workspace,
            submitted_by=submitted_by,
            idempotency_key=idempotency_key,
            fields={"document_ids": ids, "selector": selector},
        )

    async def create_reset(
        self,
        *,
        workspace: str,
        submitted_by: str,
        supersedes_run_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> RunCreation:
        return await self._create_action(
            action="reset",
            workspace=workspace,
            submitted_by=submitted_by,
            idempotency_key=idempotency_key,
            fields={"supersedes_run_id": supersedes_run_id},
        )

    async def _create_action(
        self,
        *,
        action: CorpusMutationAction,
        workspace: str,
        submitted_by: str,
        idempotency_key: str | None,
        fields: Mapping[str, Any],
    ) -> RunCreation:
        canonical = require_canonical_workspace_id(workspace)
        request = {"action": action, "workspace": canonical, **dict(fields)}
        replay = await self.replay(
            submitted_by=submitted_by,
            idempotency_key=idempotency_key,
            normalized_request=request,
        )
        if replay is not None:
            return replay
        run_id = str(uuid7())
        return await self._accept(
            run_id=run_id,
            workspace=canonical,
            submitted_by=submitted_by,
            idempotency_key=idempotency_key,
            normalized_request=request,
            payload={**request, "track_id": _track_id(run_id)},
        )

    async def _accept(
        self,
        *,
        run_id: str,
        workspace: str,
        submitted_by: str,
        idempotency_key: str | None,
        normalized_request: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> RunCreation:
        try:
            require_prepared_input_bounds(payload)
        except PreparedInputTooLargeError as exc:
            raise ValueError(str(exc)) from exc
        coordinator = self._coordinator
        if not coordinator.is_started:
            raise RunRuntimeUnavailableError("Corpus Mutation runtime is unavailable")
        envelope = PreparedRunEnvelope(
            run_kind="corpus_mutation",
            lane="corpus_mutation",
            submitted_by=submitted_by,
            access_scope=RunAccessScope(kind="workspace", scope_id=workspace),
            submission_key=idempotency_key or run_id,
            request_fingerprint=run_request_fingerprint(normalized_request),
            payload=payload,
            accepted_input={
                "action": str(payload["action"]),
                "workspace": workspace,
                **_accepted_selector(payload),
            },
            retention_seconds=CORPUS_MUTATION_RUN_RETENTION_SECONDS,
            supersedes_run_id=(
                str(payload["supersedes_run_id"])
                if payload.get("supersedes_run_id") is not None
                else None
            ),
        )
        try:
            async with coordinator.admission() as available:
                if not available:
                    raise RunRuntimeUnavailableError("Corpus Mutation runtime is unavailable")
                creation = await self._store.accept_run(envelope=envelope, run_id=run_id)
                coordinator.wake()
        except RuntimeIdempotencyKeyConflict as exc:
            raise IdempotencyKeyConflict(str(exc)) from exc
        except RuntimeRunAdmissionLimitExceededError as exc:
            raise RunAdmissionLimitExceededError(str(exc)) from exc
        return RunCreation.from_runtime(creation)

    async def stage_upload(
        self,
        *,
        workspace: str,
        run_id: str,
        filename: str,
        reader: Any,
        max_bytes: int,
        content_sha256: str | None = None,
    ) -> StagedCorpusSource:
        """Stream, hash, bound, and atomically commit one source outside Run blobs."""
        canonical = require_canonical_workspace_id(workspace)
        try:
            safe_path = safe_upload_relative_path(filename)
        except ValueError:
            raise UnsafeUploadNameError(f"Unsafe filename: {filename!r}") from None
        expected = content_sha256.lower() if content_sha256 else None
        if expected is not None and (
            len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected)
        ):
            raise ValueError("content_sha256 must be a lowercase or uppercase SHA-256 hex digest")

        run_root = self._input_root / canonical / ".runs" / run_id
        source_root = run_root / "sources"
        staging_root = self._input_root / canonical / ".staging"
        source_root.mkdir(parents=True, exist_ok=True)
        staging_root.mkdir(parents=True, exist_ok=True)
        temporary = staging_root / f"{run_id}.part"
        target = source_root / safe_path
        if target.exists():
            raise ValueError("upload contains duplicate source filenames")
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        try:
            with temporary.open("xb") as stream:
                while True:
                    chunk = await reader.read(_UPLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise UploadTooLargeError(f"upload exceeds {max_bytes} bytes")
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            actual = digest.hexdigest()
            if expected is not None and actual != expected:
                raise ValueError("content_sha256 does not match the uploaded bytes")
            os.replace(temporary, target)
            return StagedCorpusSource(
                path=target,
                filename=safe_path.as_posix(),
                content_sha256=actual,
                size_bytes=size,
            )
        except BaseException:
            temporary.unlink(missing_ok=True)
            await asyncio.to_thread(shutil.rmtree, run_root, True)
            raise

    async def discard_staged_run(self, *, workspace: str, run_id: str) -> None:
        """Delete one unaccepted Run-exclusive upload stage without leaking layout."""
        canonical = require_canonical_workspace_id(workspace)
        safe_run_id = str(UUID(run_id))
        run_root = self._input_root / canonical / ".runs" / safe_run_id
        await asyncio.to_thread(shutil.rmtree, run_root, True)

    def _snapshot_local_spec(
        self, run_id: str, workspace: str, spec: IngestSpec
    ) -> tuple[IngestSpec, Path, list[dict[str, Any]]]:
        canonical = require_canonical_workspace_id(workspace)
        workspace_root = (self._input_root / canonical).resolve()
        run_root = workspace_root / ".runs" / run_id
        source_root = run_root / "sources"
        source_root.mkdir(parents=True, exist_ok=False)
        manifest: list[dict[str, Any]] = []

        def record_file(path: Path) -> None:
            if len(manifest) >= _MAX_RESULT_DOCUMENTS:
                raise ValueError(
                    f"local corpus source contains more than {_MAX_RESULT_DOCUMENTS} files"
                )
            manifest.append(
                {
                    "path": str(path),
                    "content_sha256": _file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )

        def copy_source(raw: str, ordinal: int) -> str:
            source = Path(raw).resolve(strict=True)
            source.relative_to(workspace_root)
            if source.is_symlink():
                raise ValueError("local corpus sources cannot be symlinks")
            name = f"{ordinal:04d}-{safe_upload_basename(source.name)}"
            target = source_root / name
            if source.is_dir():
                shutil.copytree(source, target, symlinks=False)
                for child in sorted(target.rglob("*")):
                    if child.is_symlink():
                        raise ValueError("local corpus sources cannot contain symlinks")
                    if child.is_file():
                        record_file(child)
            elif source.is_file():
                shutil.copy2(source, target)
                record_file(target)
            else:
                raise ValueError("local corpus source is not a regular file or directory")
            return str(target)

        try:
            if spec.documents is not None:
                documents = [
                    document.model_copy(update={"path": copy_source(cast(str, document.path), i)})
                    for i, document in enumerate(spec.documents)
                ]
                return spec.model_copy(update={"documents": documents}), run_root, manifest
            copied = copy_source(cast(str, spec.path), 0)
            return spec.model_copy(update={"path": copied}), run_root, manifest
        except BaseException:
            shutil.rmtree(run_root, ignore_errors=True)
            raise


class _TrackedPipelineNotSettled(RuntimeError):
    """A recoverable tracked LightRAG cohort has not reached a terminal status."""


class CorpusMutationExecutor(RunExecutor):
    """Recoverable five-action executor using public Workspace/LightRAG operations."""

    def __init__(
        self,
        *,
        pool: WorkspacePool,
        maintenance: CorpusMaintenanceStore,
        store: CorpusMutationStore,
        now: Callable[[], datetime.datetime] | None = None,
    ) -> None:
        self._pool = pool
        self._maintenance = maintenance
        self._store = store
        self._now = now or (lambda: datetime.datetime.now(datetime.UTC))

    async def execute(self, session: RunSession) -> RunExecutionOutcome:
        raw = session.prepared_input
        if not isinstance(raw, Mapping):
            return Failed("invalid_corpus_mutation", "Corpus Mutation input is unavailable.")
        try:
            action, workspace = validate_corpus_mutation_prepared_input(raw)
        except ValueError:
            return Failed("invalid_corpus_mutation", "Corpus Mutation input is invalid.")
        if workspace != session.owner_id:
            return Failed("invalid_corpus_mutation", "Corpus Mutation scope does not match input.")

        recovered_after_handoff = session.handoff_started
        checkpoint = dict(session.checkpoint or {})
        checkpoint.update(
            action=action,
            workspace=workspace,
            track_id=str(raw.get("track_id") or _track_id(session.run_id)),
        )
        try:
            runtime = await self._pool.acquire(workspace)
            if action in {"ingest", "replace", "retry"}:
                await session.enter_phase("reconciling_upstream")
                upstream = await runtime.lightrag.aget_docs_by_track_id(checkpoint["track_id"])
                checkpoint["upstream_documents"] = _public_upstream_state(upstream)
                await session.checkpoint_state(checkpoint, phase="reconciled")

            if recovered_after_handoff and _requires_repair_resume(action, checkpoint):
                if checkpoint.get("repair_resume_confirmed") is not True:
                    return WaitingForRepair(_repair_checkpoint(checkpoint, ()))
                checkpoint["repair_resume_confirmed"] = False
                checkpoint["phase"] = "repair_attempt_started"
                await session.checkpoint_state(checkpoint, phase="repair_attempt_started")

            if action in {"ingest", "replace"}:
                return await self._ingest(session, runtime, raw, checkpoint)
            if action == "delete":
                return await self._delete(session, runtime, raw, checkpoint)
            if action == "retry":
                return await self._retry(session, runtime, raw, checkpoint)
            return await self._reset(session, runtime, raw, checkpoint)
        except WorkspaceWriteFencedError, _TrackedPipelineNotSettled:
            return _deferred(checkpoint, "corpus_storage", now=self._now)
        except RetryOutcomeUncertainError:
            if action in {"replace", "delete", "retry", "reset"} and session.handoff_started:
                return WaitingForRepair(_repair_checkpoint(checkpoint, ()))
            return _deferred(checkpoint, "corpus_storage", now=self._now)
        except FileNotFoundError:
            return Failed(
                "corpus_source_unavailable",
                "A complete accepted corpus source is no longer available.",
                result=_result(action, (), checkpoint),
            )
        except ValueError:
            return Failed(
                "invalid_corpus_mutation",
                "Corpus Mutation input failed validation.",
                result=_result(action, (), checkpoint),
            )
        except Exception as exc:
            if action in {"replace", "delete", "retry", "reset"} and session.handoff_started:
                return WaitingForRepair(_repair_checkpoint(checkpoint, ()))
            component = classify_transient_dependency(exc)
            if component is not None:
                return _deferred(checkpoint, component, now=self._now)
            raise

    async def _ingest(
        self,
        session: RunSession,
        runtime: Any,
        raw: Mapping[str, Any],
        checkpoint: dict[str, Any],
    ) -> RunExecutionOutcome:
        source = raw.get("source")
        if not isinstance(source, Mapping):
            return Failed("invalid_corpus_mutation", "Corpus source is unavailable.")
        if checkpoint.get("operation_settled") is True:
            return await self._settle_ingest(session, str(raw["action"]), checkpoint)

        kwargs = dict(source)
        source_type = str(kwargs.pop("source_type", ""))
        upstream = list(checkpoint.get("upstream_documents") or ())
        if session.handoff_started and upstream:
            result = await self._reconcile_tracked_ingest(session, runtime, checkpoint)
            documents = _retry_outcomes(
                result,
                [str(item.get("document_id") or "") for item in upstream],
            )
        else:
            if source_type == "local" and not await asyncio.to_thread(
                _local_source_complete,
                kwargs,
                raw.get("staged_sources"),
            ):
                raise FileNotFoundError
            kwargs["replace"] = raw.get("action") == "replace"
            kwargs["_track_id"] = checkpoint["track_id"]
            await session.checkpoint_state(
                {**checkpoint, "phase": "source_staged"}, phase="source_staged"
            )
            if not session.handoff_started:
                await session.begin_handoff({**checkpoint, "phase": "handoff_started"})
            await session.enter_phase("upstream_pipeline")
            async with self._maintenance.workspace_write_gate(session.owner_id):
                result = await _join_public_operation(runtime.aingest(source_type, **kwargs))
            documents = _document_outcomes(result)
        errors = result.get("errors") if isinstance(result, Mapping) else None
        processed = (
            int(result.get("processed") or len(documents)) if isinstance(result, Mapping) else 0
        )
        checkpoint.update(
            document_outcomes=documents,
            operation_settled=True,
            result_had_errors=bool(errors),
            processed_count=max(0, processed),
        )
        await session.checkpoint_state(checkpoint, phase="documents_reconciled")
        return await self._settle_ingest(session, str(raw["action"]), checkpoint)

    async def _settle_ingest(
        self,
        session: RunSession,
        action: str,
        checkpoint: Mapping[str, Any],
    ) -> RunExecutionOutcome:
        documents = _checkpoint_documents(checkpoint)
        public = _result(action, documents, checkpoint)
        if checkpoint.get("result_had_errors") is True or any(
            item.get("status") == "failed" for item in documents
        ):
            return Failed(
                "corpus_mutation_document_failed",
                "One or more documents did not become ready.",
                result=public,
            )
        await self._store.record_corpus_window(
            run_id=session.run_id,
            workspace=session.owner_id,
            window_number=1,
            docs=_nonnegative_checkpoint_int(checkpoint, "processed_count"),
            chunks=sum(_chunk_count(item.get("chunks")) for item in documents),
        )
        return Succeeded(public)

    async def _reconcile_tracked_ingest(
        self,
        session: RunSession,
        runtime: Any,
        checkpoint: dict[str, Any],
    ) -> Mapping[str, Any]:
        """Settle one already-handed-off cohort without replaying replace deletion."""
        states = {
            str(item.get("document_id") or ""): str(item.get("status") or "").lower()
            for item in checkpoint.get("upstream_documents") or ()
            if isinstance(item, Mapping) and item.get("document_id")
        }
        if any(status not in {"processed", "failed"} for status in states.values()):
            await session.enter_phase("recovering_upstream_pipeline")
            async with self._maintenance.workspace_write_gate(session.owner_id):
                await _join_public_operation(runtime.lightrag.apipeline_process_enqueue_documents())
            refreshed = await runtime.lightrag.aget_docs_by_track_id(checkpoint["track_id"])
            checkpoint["upstream_documents"] = _public_upstream_state(refreshed)
            await session.checkpoint_state(checkpoint, phase="reconciled")
            states = {
                str(item.get("document_id") or ""): str(item.get("status") or "").lower()
                for item in checkpoint["upstream_documents"]
                if isinstance(item, Mapping) and item.get("document_id")
            }
        if not states or any(status not in {"processed", "failed"} for status in states.values()):
            raise _TrackedPipelineNotSettled
        await session.enter_phase("finalizing_tracked_documents")
        async with self._maintenance.workspace_write_gate(session.owner_id):
            return await _join_public_operation(
                runtime.aretry_failed_docs(
                    cohort_doc_ids=tuple(states),
                    track_id=checkpoint["track_id"],
                )
            )

    async def _delete(
        self,
        session: RunSession,
        runtime: Any,
        raw: Mapping[str, Any],
        checkpoint: dict[str, Any],
    ) -> RunExecutionOutcome:
        file_paths = [str(value) for value in raw.get("file_paths") or ()]
        filenames = [str(value) for value in raw.get("filenames") or ()]
        document_ids = [str(value) for value in raw.get("document_ids") or ()]
        identifiers = [*file_paths, *filenames, *document_ids]
        if checkpoint.get("operation_settled") is True:
            return _settled_delete_outcome(checkpoint)
        if "resolved_documents" not in checkpoint:
            preview = await runtime.adelete_files(
                file_paths=[*file_paths, *document_ids], filenames=filenames, dry_run=True
            )
            checkpoint["resolved_documents"] = [
                {
                    "identifier": str(item.get("identifier") or ""),
                    "document_ids": [str(value) for value in item.get("matched_doc_ids") or ()][
                        :_MAX_RESULT_DOCUMENTS
                    ],
                    "file_paths": [str(value) for value in item.get("matched_file_paths") or ()][
                        :_MAX_RESULT_DOCUMENTS
                    ],
                }
                for item in preview
                if isinstance(item, Mapping)
            ][:_MAX_RESULT_DOCUMENTS]
            await session.checkpoint_state(checkpoint, phase="identities_resolved")
        if not session.handoff_started:
            await session.begin_handoff({**checkpoint, "phase": "handoff_started"})
        await session.enter_phase("deleting_upstream")
        async with self._maintenance.workspace_write_gate(session.owner_id):
            results = await _join_public_operation(
                runtime.adelete_files(
                    file_paths=[*file_paths, *document_ids], filenames=filenames, dry_run=False
                )
            )
        documents = [dict(item) for item in results if isinstance(item, Mapping)]
        if any(str(item.get("status")) == "waiting_for_repair" for item in documents):
            return WaitingForRepair(_repair_checkpoint(checkpoint, documents))
        if not identifiers:
            return Failed("invalid_corpus_mutation", "No delete identifiers were accepted.")
        checkpoint.update(document_outcomes=documents, operation_settled=True)
        await session.checkpoint_state(checkpoint, phase="delete_settled")
        return _settled_delete_outcome(checkpoint)

    async def _retry(
        self,
        session: RunSession,
        runtime: Any,
        raw: Mapping[str, Any],
        checkpoint: dict[str, Any],
    ) -> RunExecutionOutcome:
        cohort = [str(value) for value in checkpoint.get("cohort_doc_ids") or ()]
        if checkpoint.get("operation_settled") is True:
            return await self._settle_retry(session, checkpoint)
        if not checkpoint.get("cohort_sealed"):
            requested = [str(value) for value in raw.get("document_ids") or ()]
            if requested:
                cohort = list(dict.fromkeys(requested))
            else:
                cohort = list(await runtime.aretryable_document_ids())
            checkpoint.update(cohort_doc_ids=cohort, cohort_sealed=True, phase="cohort_sealed")
            await session.checkpoint_state(checkpoint, phase="cohort_sealed")
        if not cohort:
            return Succeeded(_result("retry", (), checkpoint))
        if not session.handoff_started:
            await session.begin_handoff({**checkpoint, "phase": "handoff_started"})
        await session.enter_phase("retrying_documents")
        async with self._maintenance.workspace_write_gate(session.owner_id):
            result = await _join_public_operation(
                runtime.aretry_failed_docs(
                    cohort_doc_ids=cohort,
                    track_id=checkpoint["track_id"],
                )
            )
        documents = _retry_outcomes(result, cohort)
        checkpoint.update(
            document_outcomes=documents,
            operation_settled=True,
            retry_failed_count=(
                max(0, int(result.get("failed") or 0))
                if isinstance(result, Mapping)
                else len(documents)
            ),
            retry_succeeded_count=(
                max(0, int(result.get("succeeded") or 0)) if isinstance(result, Mapping) else 0
            ),
        )
        await session.checkpoint_state(checkpoint, phase="retry_settled")
        return await self._settle_retry(session, checkpoint)

    async def _settle_retry(
        self,
        session: RunSession,
        checkpoint: Mapping[str, Any],
    ) -> RunExecutionOutcome:
        documents = _checkpoint_documents(checkpoint)
        public = _result("retry", documents, checkpoint)
        if _nonnegative_checkpoint_int(checkpoint, "retry_failed_count") > 0 or any(
            item.get("status") == "failed" for item in documents
        ):
            return Failed(
                "corpus_retry_document_failed",
                "One or more retry documents did not become ready.",
                result=public,
            )
        await self._store.record_corpus_window(
            run_id=session.run_id,
            workspace=session.owner_id,
            window_number=1,
            docs=_nonnegative_checkpoint_int(checkpoint, "retry_succeeded_count"),
            chunks=0,
        )
        return Succeeded(public)

    async def _reset(
        self,
        session: RunSession,
        runtime: Any,
        raw: Mapping[str, Any],
        checkpoint: dict[str, Any],
    ) -> RunExecutionOutcome:
        if checkpoint.get("operation_settled") is not True:
            if not session.handoff_started:
                await session.begin_handoff({**checkpoint, "phase": "handoff_started"})
            await session.enter_phase("resetting_corpus")
            async with self._maintenance.workspace_write_gate(session.owner_id):
                result = await _join_public_operation(
                    runtime.areset(
                        dry_run=False,
                        preserve_run_sources_after=session.run_id,
                    )
                )
            documents = [dict(result)] if isinstance(result, Mapping) else []
            if not isinstance(result, Mapping) or result.get("errors"):
                return WaitingForRepair(_repair_checkpoint(checkpoint, documents))
            checkpoint.update(document_outcomes=documents, operation_settled=True)
            await session.checkpoint_state(checkpoint, phase="reset_settled")
        await self._pool.evict(session.owner_id)
        return Succeeded(_result("reset", _checkpoint_documents(checkpoint), checkpoint))


async def _join_public_operation[T](operation: Awaitable[T]) -> T:
    """Never let task cancellation abandon an in-process upstream operation."""
    task = asyncio.ensure_future(operation)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Handoff already occurred before every public operation joined here.
        # Finish and classify the authoritative outcome instead of labelling a
        # completed destructive effect as cancelled.
        return await asyncio.shield(task)


def validate_corpus_mutation_prepared_input(
    raw: Mapping[str, Any],
) -> tuple[CorpusMutationAction, str]:
    """Validate the closed durable action schema used for recovery compatibility."""
    action = str(raw.get("action") or "")
    if action not in {"ingest", "replace", "delete", "retry", "reset"}:
        raise ValueError("unknown Corpus Mutation action")
    workspace = require_canonical_workspace_id(str(raw.get("workspace") or ""))
    track_id = str(raw.get("track_id") or "")
    prefix = "dlightrag-corpus-"
    if not track_id.startswith(prefix):
        raise ValueError("Corpus Mutation track_id is invalid")
    try:
        if UUID(track_id.removeprefix(prefix)).version != 7:
            raise ValueError
    except ValueError:
        raise ValueError("Corpus Mutation track_id is invalid") from None
    common = {"action", "workspace", "track_id"}
    action_fields = {
        "ingest": {"source", "staged_sources"},
        "replace": {"source", "staged_sources"},
        "delete": {"file_paths", "filenames", "document_ids"},
        "retry": {"document_ids", "selector"},
        "reset": {"supersedes_run_id"},
    }
    expected_fields = common | action_fields[action]
    allowed_field_sets = {frozenset(expected_fields)}
    if action in {"ingest", "replace"}:
        allowed_field_sets.add(frozenset(expected_fields | {"sources"}))
    if frozenset(raw) not in allowed_field_sets:
        raise ValueError("Corpus Mutation input fields do not match its action")
    if action in {"ingest", "replace"}:
        source = raw.get("source")
        if not isinstance(source, Mapping):
            raise ValueError("Corpus Mutation source is unavailable")
        spec = IngestSpec.model_validate(source)
        if bool(spec.replace) != (action == "replace"):
            raise ValueError("Corpus Mutation source action does not match replace mode")
        staged_sources = raw.get("staged_sources")
        if not isinstance(staged_sources, list) or len(staged_sources) > _MAX_RESULT_DOCUMENTS:
            raise ValueError("Corpus Mutation staged source manifest is invalid")
        for item in staged_sources:
            if not isinstance(item, Mapping) or set(item) != {
                "path",
                "content_sha256",
                "size_bytes",
            }:
                raise ValueError("Corpus Mutation staged source manifest is invalid")
            path = item.get("path")
            digest = item.get("content_sha256")
            size = item.get("size_bytes")
            if (
                not isinstance(path, str)
                or not path
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest)
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
            ):
                raise ValueError("Corpus Mutation staged source manifest is invalid")
        if spec.source_type == "local" and not staged_sources:
            raise ValueError("local Corpus Mutation source manifest is unavailable")
        if spec.source_type != "local" and staged_sources:
            raise ValueError("remote Corpus Mutation cannot contain staged sources")
        sources = raw.get("sources")
        if sources is not None and (
            not isinstance(sources, list)
            or len(sources) != len(staged_sources)
            or any(
                not isinstance(item, Mapping)
                or set(item) != {"filename", "content_sha256", "size_bytes"}
                for item in sources
            )
        ):
            raise ValueError("Corpus Mutation source identity list is invalid")
    elif action == "delete":
        selectors = tuple(
            _prepared_string_list(raw.get(key), field=key)
            for key in ("file_paths", "filenames", "document_ids")
        )
        if not any(selectors):
            raise ValueError("delete requires an exact identifier")
    elif action == "retry":
        document_ids = _prepared_string_list(raw.get("document_ids"), field="document_ids")
        selector = raw.get("selector")
        if bool(document_ids) == bool(selector) or selector not in {None, "all_retryable"}:
            raise ValueError("retry requires one cohort selector")
    else:
        supersedes = raw.get("supersedes_run_id")
        if supersedes is not None and (not isinstance(supersedes, str) or not supersedes.strip()):
            raise ValueError("supersedes_run_id is invalid")
    return cast(CorpusMutationAction, action), workspace


def _prepared_string_list(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > _MAX_RESULT_DOCUMENTS:
        raise ValueError(f"{field} must be a bounded list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field} contains an invalid identifier")
    normalized = tuple(item.strip() for item in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} contains duplicate identifiers")
    return normalized


def _track_id(run_id: str) -> str:
    return f"dlightrag-corpus-{run_id}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_UPLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _staged_source_record(source: StagedCorpusSource) -> dict[str, Any]:
    return {
        "path": str(source.path),
        "content_sha256": source.content_sha256,
        "size_bytes": source.size_bytes,
    }


def _requires_repair_resume(
    action: CorpusMutationAction,
    checkpoint: Mapping[str, Any],
) -> bool:
    """Fence recovered destructive work unless durable evidence makes replay unnecessary."""
    if checkpoint.get("operation_settled") is True:
        return False
    if action == "ingest":
        return False
    if action == "replace" and bool(checkpoint.get("upstream_documents")):
        return False
    return True


def _checkpoint_documents(checkpoint: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = checkpoint.get("document_outcomes")
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)][:_MAX_RESULT_DOCUMENTS]


def _nonnegative_checkpoint_int(checkpoint: Mapping[str, Any], key: str) -> int:
    value = checkpoint.get(key)
    return max(0, value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _settled_delete_outcome(checkpoint: Mapping[str, Any]) -> RunExecutionOutcome:
    documents = _checkpoint_documents(checkpoint)
    public = _result("delete", documents, checkpoint)
    if any(str(item.get("status")) in {"failed", "rejected"} for item in documents):
        return Failed(
            "corpus_delete_rejected",
            "One or more documents could not be deleted.",
            result=public,
        )
    return Succeeded(public)


def _deferred(
    checkpoint: Mapping[str, Any],
    component: DependencyComponent,
    *,
    now: Callable[[], datetime.datetime],
) -> Deferred:
    retry_checkpoint, delay = next_dependency_retry(
        checkpoint,
        component,
        base_seconds=_DEFER_BASE_SECONDS,
        max_seconds=_DEFER_MAX_SECONDS,
    )
    return Deferred(
        checkpoint={
            **dict(checkpoint),
            **retry_checkpoint,
            "phase": "deferred_dependency",
        },
        next_attempt_at=now() + datetime.timedelta(seconds=delay),
    )


def _bounded_unique(values: Sequence[str]) -> list[str]:
    normalized = list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    if len(normalized) > _MAX_RESULT_DOCUMENTS:
        raise ValueError(f"at most {_MAX_RESULT_DOCUMENTS} document identifiers are allowed")
    return normalized


def _accepted_selector(payload: Mapping[str, Any]) -> dict[str, Any]:
    action = str(payload.get("action") or "")
    if action in {"delete", "retry", "reset"}:
        return {
            key: value
            for key, value in payload.items()
            if key
            in {
                "file_paths",
                "filenames",
                "document_ids",
                "selector",
                "supersedes_run_id",
            }
            and value is not None
        }
    source = payload.get("source")
    return {
        "source_type": str(source.get("source_type") or "") if isinstance(source, Mapping) else ""
    }


def _local_source_complete(source: Mapping[str, Any], manifest: Any) -> bool:
    paths = [str(source.get("path") or "")]
    documents = source.get("documents")
    if isinstance(documents, list):
        paths = [str(item.get("path") or "") for item in documents if isinstance(item, Mapping)]
    if not paths or not all(path and Path(path).exists() for path in paths):
        return False
    if not isinstance(manifest, list) or not manifest:
        return False
    for item in manifest:
        if not isinstance(item, Mapping):
            return False
        path = Path(str(item.get("path") or ""))
        digest = str(item.get("content_sha256") or "")
        size = item.get("size_bytes")
        if not isinstance(size, int):
            return False
        try:
            if not path.is_file() or path.stat().st_size != size or _file_sha256(path) != digest:
                return False
        except OSError, TypeError, ValueError:
            return False
    return True


def _public_upstream_state(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, Mapping):
        return []
    rows: list[dict[str, str]] = []
    for doc_id, status in list(value.items())[:_MAX_RESULT_DOCUMENTS]:
        raw = (
            status.get("status") if isinstance(status, Mapping) else getattr(status, "status", None)
        )
        rows.append(
            {
                "document_id": str(doc_id),
                "status": str(getattr(raw, "value", raw) or "unknown"),
            }
        )
    return rows


def _document_outcomes(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, Mapping):
        return [{"status": "failed", "phase": "pipeline"}]
    raw_results = result.get("results")
    rows = (
        [dict(item) for item in raw_results if isinstance(item, Mapping)]
        if isinstance(raw_results, list)
        else ([dict(result)] if result.get("doc_id") else [])
    )
    for row in rows:
        row.setdefault("status", "ready")
        row.setdefault("phase", "finalized")
    errors = [str(error)[:256] for error in result.get("errors") or ()]
    for error in errors[: max(0, _MAX_RESULT_DOCUMENTS - len(rows))]:
        rows.append({"status": "failed", "phase": "pipeline", "error": error})
    return rows[:_MAX_RESULT_DOCUMENTS]


def _retry_outcomes(result: Any, cohort: Sequence[str]) -> list[dict[str, Any]]:
    if not isinstance(result, Mapping):
        return [{"document_id": value, "status": "failed"} for value in cohort]
    succeeded = {
        str(item.get("doc_id"))
        for item in result.get("succeeded_docs") or ()
        if isinstance(item, Mapping)
    }
    failed = {
        str(item.get("doc_id"))
        for item in result.get("failed_docs") or ()
        if isinstance(item, Mapping)
    }
    return [
        {
            "document_id": doc_id,
            "status": "ready" if doc_id in succeeded and doc_id not in failed else "failed",
            "phase": "finalized" if doc_id in succeeded and doc_id not in failed else "retry",
        }
        for doc_id in cohort[:_MAX_RESULT_DOCUMENTS]
    ]


def _chunk_count(value: Any) -> int:
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value)
    return 0


def _public_document_outcome(item: Mapping[str, Any]) -> dict[str, Any]:
    """Project bounded result evidence without source locators or diagnostics."""
    projected: dict[str, Any] = {}
    document_id = item.get("document_id") or item.get("doc_id")
    if document_id:
        projected["document_id"] = str(document_id)[:256]
    if item.get("identifier"):
        projected["identifier"] = str(item["identifier"])[:256]
    for key in ("status", "phase", "source_kind", "reason"):
        if item.get(key) is not None:
            projected[key] = str(item[key])[:256]
    chunks = item.get("chunks")
    if chunks is not None:
        projected["chunk_count"] = _chunk_count(chunks)
    for key in (
        "replacement_count",
        "documents_deleted",
        "chunks_deleted",
        "entities_deleted",
        "relationships_deleted",
        "local_files_removed",
        "orphan_tables_cleaned",
    ):
        value = item.get(key)
        if isinstance(value, int):
            projected[key] = max(0, value)
    if not projected:
        projected = {"status": "completed"}
    return projected


def _result(
    action: str, documents: Sequence[Mapping[str, Any]], checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "action": action,
        "documents": [_public_document_outcome(item) for item in documents[:_MAX_RESULT_DOCUMENTS]],
        "document_count": min(len(documents), _MAX_RESULT_DOCUMENTS),
        "details_truncated": len(documents) > _MAX_RESULT_DOCUMENTS,
        "track_id": str(checkpoint.get("track_id") or ""),
    }


def _repair_checkpoint(
    checkpoint: Mapping[str, Any], documents: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        **dict(checkpoint),
        "phase": "waiting_for_repair",
        "repair_reason": _REPAIR_REASON,
        "repair_remedy": _REPAIR_REMEDY,
        "documents": [dict(item) for item in documents[:_MAX_RESULT_DOCUMENTS]],
    }


__all__ = [
    "CorpusMutationAction",
    "CorpusMutationExecutor",
    "CorpusMutationService",
    "RetrySelector",
    "StagedCorpusSource",
    "validate_corpus_mutation_prepared_input",
]
