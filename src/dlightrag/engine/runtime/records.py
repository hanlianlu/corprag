# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Storage-neutral records for one durable run lifecycle."""

import datetime
import hashlib
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from dlightrag.engine.agent.session.ids import SessionId
from dlightrag.engine.agent.session.repository import AgentSessionRepository
from dlightrag.engine.runtime.contracts import RunKind, RunLane, RunPhase, RunStatus
from dlightrag.engine.runtime.policy import MAX_RECLAIMS_WITHOUT_PROGRESS
from dlightrag.engine.runtime.progress import RunProgressStore
from dlightrag.engine.runtime.settlements import EffectHostUpdate
from dlightrag.engine.runtime.workspace import WorkspaceStore

# Event labels are executor-owned. RunRuntime orders and persists them without
# importing an operation-specific vocabulary.
type RunEventType = str
type ArtifactReferenceKind = Literal[
    "current_attachment",
    "history_attachment",
    "fetched_resource",
    "published_artifact",
]
#: How a graceful shutdown left one owned run.
type ShutdownOutcome = Literal["requeued", "cancelled", "lease_lost"]

_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
MAX_PREPARED_INPUT_BYTES = 8 * 1024 * 1024


class PreparedInputTooLargeError(ValueError):
    """A generic prepared Run envelope exceeds its durable size bound."""

    def __init__(self, *, encoded_bytes: int) -> None:
        self.encoded_bytes = encoded_bytes
        super().__init__(
            "prepared_input_too_large: "
            f"{encoded_bytes} bytes exceed the {MAX_PREPARED_INPUT_BYTES} byte bound"
        )


def canonical_run_request_json(request: Mapping[str, Any]) -> str:
    """Serialize a run request to its one durable JSON representation."""
    return json.dumps(
        dict(request),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def run_request_fingerprint(request: Mapping[str, Any]) -> str:
    """Digest one canonical public request for submission replay comparison."""
    encoded = canonical_run_request_json(request).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_prepared_input_bounds(prepared_input: Mapping[str, Any]) -> None:
    """Enforce the one generic canonical bound for durable execution input."""
    encoded = canonical_run_request_json(prepared_input).encode("utf-8")
    if len(encoded) > MAX_PREPARED_INPUT_BYTES:
        raise PreparedInputTooLargeError(encoded_bytes=len(encoded))


def artifact_digest(content: bytes) -> str:
    """Content address for one immutable run artifact."""
    return hashlib.sha256(content).hexdigest()


def parse_run_id(run_id: str) -> uuid.UUID | None:
    """Parse an opaque run id; malformed caller input reads as unknown."""
    try:
        return uuid.UUID(str(run_id))
    except ValueError:
        return None


class IdempotencyKeyConflict(RuntimeError):
    """One submitter reused a submission key with different normalized input."""


class RunCapacityExceededError(RuntimeError):
    """A lane's deployment-wide nonterminal admission fuse is full."""


@dataclass(frozen=True, slots=True)
class RunAccessScope:
    """The authorization scope required to observe or cancel one run."""

    kind: Literal["owner", "workspace"]
    scope_id: str


@dataclass(frozen=True, slots=True)
class PreparedRunEnvelope:
    """Generic durable input submitted by an Application use case.

    ``payload`` is opaque to RunRuntime. ``accepted_input`` is the bounded
    terminal-surviving projection. ``supersedes_run_id`` is an explicit generic
    acceptance relationship, never parsed from an operation payload. A
    submission key is mandatory even when the external caller did not provide
    one; the Application then uses the run id.
    """

    run_kind: RunKind
    lane: RunLane
    submitted_by: str
    access_scope: RunAccessScope
    submission_key: str
    request_fingerprint: str
    payload: Mapping[str, Any]
    accepted_input: Mapping[str, Any]
    retention_seconds: int
    supersedes_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Authoritative lifecycle state of one durable run.

    Durable progress replaces the checkpoint-era turn and recovery counters:
    ``durable_progress_version`` advances only on live fenced work — model
    turn appends, compaction appends, live effect settlements, and Fast stage
    settlements. Recovery prelude (interrupted ``never`` intents, contract
    changes, workspace-epoch handoff) does not advance it.
    """

    run_id: str
    run_kind: RunKind
    lane: RunLane
    submitted_by: str
    access_scope: RunAccessScope
    submission_key: str
    request_fingerprint: str
    prepared_input: Mapping[str, Any] | None
    status: RunStatus
    phase: RunPhase | None
    stop_reason: str | None
    cancel_requested_at: datetime.datetime | None
    lease_owner: str | None
    lease_expires_at: datetime.datetime | None
    fencing_epoch: int
    durable_progress_version: int
    last_reclaim_progress_version: int
    reclaims_without_progress: int
    next_event_sequence: int
    events_trimmed_at: datetime.datetime | None
    result: Mapping[str, Any] | None
    error_kind: str | None
    error_message: str | None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    started_at: datetime.datetime | None
    finished_at: datetime.datetime | None
    purge_after: datetime.datetime | None = None
    next_attempt_at: datetime.datetime | None = None
    active_permit: bool = False
    checkpoint: Mapping[str, Any] | None = None
    handoff_started_at: datetime.datetime | None = None
    superseded_by_run_id: str | None = None
    agent_workspace_epoch: int | None = None
    #: The bounded public envelope (query, workspaces, mode, attachment
    #: identities) that survives the terminal transition: prepared_input_json
    #: is cleared at finish, so post-terminal readers project from this.
    accepted_input: Mapping[str, Any] | None = None

    @property
    def owner_id(self) -> str:
        """Answer's owner scope during Slice 1; callers should use access_scope."""
        return self.access_scope.scope_id

    def request_input(self) -> Mapping[str, Any]:
        """The run's public request for projection: envelope first, then input.

        Terminal transitions clear ``prepared_input_json``; the accepted
        envelope is the durable public face readers use for history, workspace
        download authorization, and turn projection.
        """
        return self.accepted_input or self.prepared_input or {}

    @property
    def cancel_requested(self) -> bool:
        return self.cancel_requested_at is not None

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES


@dataclass(frozen=True, slots=True)
class ReclaimState:
    """The three durable-progress counters one reclaim consults and updates."""

    durable_progress_version: int
    last_reclaim_progress_version: int
    reclaims_without_progress: int


@dataclass(frozen=True, slots=True)
class ReclaimDecision:
    """What one expired-lease reclaim decided: claimable or abandoned."""

    abandoned: bool
    reclaims_without_progress: int
    last_reclaim_progress_version: int


def advance_reclaim(
    state: ReclaimState,
    *,
    max_reclaims: int = MAX_RECLAIMS_WITHOUT_PROGRESS,
) -> ReclaimDecision:
    """Advance the reclaim counters for one expired-lease reclaim.

    Progress since the last reclaim resets the no-progress counter to one (this
    reclaim itself). Consecutive reclaims without durable progress abandon the
    run once the declared bound is reached: the fourth such reclaim abandons
    under the default bound.
    """
    if max_reclaims < 1:
        raise ValueError("max_reclaims must be positive")
    if state.durable_progress_version > state.last_reclaim_progress_version:
        return ReclaimDecision(
            abandoned=False,
            reclaims_without_progress=1,
            last_reclaim_progress_version=state.durable_progress_version,
        )
    reclaims = state.reclaims_without_progress + 1
    return ReclaimDecision(
        abandoned=reclaims >= max_reclaims,
        reclaims_without_progress=reclaims,
        last_reclaim_progress_version=state.last_reclaim_progress_version,
    )


@dataclass(frozen=True, slots=True)
class RunEvent:
    """One durable event in a run's gap-free sequence."""

    sequence: int
    event_type: RunEventType
    payload: Mapping[str, Any]
    created_at: datetime.datetime


@dataclass(frozen=True, slots=True)
class RunCreation:
    """Result of an owner-scoped create, including idempotent replays."""

    run: RunRecord
    replayed: bool


@dataclass(frozen=True, slots=True, init=False)
class RunExecutionContext:
    """The immutable claim-bound execution surface one worker receives.

    The PostgreSQL adapter creates this binding at claim/reclaim: owner id,
    run id, worker id, lease owner, and fencing epoch are embedded. Answer-owned
    facilities fail explicitly when an executor for another run kind asks for
    them, while Answer consumers retain their non-optional bound interfaces.
    """

    owner_id: str
    run_id: str
    worker_id: str
    lease_owner: str
    fencing_epoch: int
    _session_repository: AgentSessionRepository[EffectHostUpdate] | None
    _progress_store: RunProgressStore | None
    workspace_store: WorkspaceStore | None

    def __init__(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        lease_owner: str,
        fencing_epoch: int,
        session_repository: AgentSessionRepository[EffectHostUpdate] | None = None,
        progress_store: RunProgressStore | None = None,
        workspace_store: WorkspaceStore | None = None,
    ) -> None:
        object.__setattr__(self, "owner_id", owner_id)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "worker_id", worker_id)
        object.__setattr__(self, "lease_owner", lease_owner)
        object.__setattr__(self, "fencing_epoch", fencing_epoch)
        object.__setattr__(self, "_session_repository", session_repository)
        object.__setattr__(self, "_progress_store", progress_store)
        object.__setattr__(self, "workspace_store", workspace_store)

    @property
    def session_repository(self) -> AgentSessionRepository[EffectHostUpdate]:
        """Return the Answer-owned Session binding for this claimed run."""
        if self._session_repository is None:
            raise RuntimeError("claimed run has no Agent Session repository")
        return self._session_repository

    @property
    def progress_store(self) -> RunProgressStore:
        """Return the Answer-owned progress binding for this claimed run."""
        if self._progress_store is None:
            raise RuntimeError("claimed run has no Answer progress store")
        return self._progress_store


@dataclass(frozen=True, slots=True)
class ClaimedRun:
    """A run this worker now owns, with its claim-bound execution surface."""

    run: RunRecord
    execution: RunExecutionContext
    pinned_session_id: SessionId | None = None


@dataclass(frozen=True, slots=True)
class LeaseRenewal:
    """Whether a fenced worker still owns its run and its cancellation state."""

    renewed: bool
    cancel_requested: bool


@dataclass(frozen=True, slots=True)
class TerminalOutcome:
    """Result of a fenced terminal transition and its single terminal event."""

    committed: bool
    status: RunStatus | None
    event_sequence: int | None


@dataclass(frozen=True, slots=True)
class Succeeded:
    """Execution succeeded; the coordinator still owns the terminal write."""

    result: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Failed:
    """Execution failed with an optional durable partial result."""

    error_kind: str
    error_message: str
    result: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Deferred:
    """Execution yielded its permit until a durable retry time."""

    checkpoint: Mapping[str, Any]
    next_attempt_at: datetime.datetime


@dataclass(frozen=True, slots=True)
class WaitingForRepair:
    """Execution needs operator repair while retaining its mutation barrier."""

    checkpoint: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class AlreadyCommittedTerminal:
    """Execution atomically committed a known terminal row and event."""

    terminal: TerminalOutcome

    def __post_init__(self) -> None:
        terminal = self.terminal
        if (
            not terminal.committed
            or terminal.status not in _TERMINAL_STATUSES
            or terminal.event_sequence is None
            or terminal.event_sequence < 1
        ):
            raise ValueError("already-committed execution outcome requires a known terminal")


type RunExecutionOutcome = (
    Succeeded | Failed | Deferred | WaitingForRepair | AlreadyCommittedTerminal
)


@dataclass(frozen=True, slots=True)
class CancellationOutcome:
    """Result of a scoped cancellation request.

    ``rejected`` means execution crossed its durable external handoff and the
    runtime deliberately did not set ``cancel_requested_at``.
    """

    outcome: Literal["unknown", "cancelled", "pending", "already_terminal", "rejected"]
    run: RunRecord | None


@dataclass(frozen=True, slots=True)
class SweepOutcome:
    """Rows the slot-free sweeper finalized in one pass."""

    cancelled: int
    abandoned: int


@dataclass(frozen=True, slots=True)
class RunDeletion:
    """Rows and now-unreferenced blobs removed by deletion or retention."""

    runs: int
    artifacts: int


@dataclass(frozen=True, slots=True)
class PendingPublication:
    """Staged workspace bytes to attach at successful terminal commit."""

    resource_id: str
    reference_kind: ArtifactReferenceKind
    filename: str
    mime_type: str
    content: bytes


@dataclass(frozen=True, slots=True)
class PendingArtifact:
    """Immutable bytes in one owner's content-addressed namespace."""

    content: bytes

    @property
    def digest(self) -> str:
        return artifact_digest(self.content)


@dataclass(frozen=True, slots=True)
class PendingArtifactReference:
    """One ordered run input or discovered resource pointing at stored bytes."""

    resource_id: str
    reference_kind: ArtifactReferenceKind
    ordinal: int
    digest: str
    filename: str
    mime_type: str
    transform_locator: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunFetchedResource:
    """Durable catalog entry for one fixed Web or tool-attachment snapshot."""

    resource_id: str
    ordinal: int
    digest: str
    filename: str
    mime_type: str
    source_locator: bytes
    capabilities: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RunArtifactReference:
    """A stored run-artifact reference read back with its creation time."""

    resource_id: str
    reference_kind: ArtifactReferenceKind
    ordinal: int
    digest: str
    filename: str
    mime_type: str
    transform_locator: Mapping[str, Any]
    created_at: datetime.datetime


__all__ = [
    "RunEvent",
    "RunEventType",
    "RunRecord",
    "ArtifactReferenceKind",
    "CancellationOutcome",
    "ClaimedRun",
    "Deferred",
    "Failed",
    "IdempotencyKeyConflict",
    "LeaseRenewal",
    "MAX_PREPARED_INPUT_BYTES",
    "PendingArtifact",
    "PendingPublication",
    "PendingArtifactReference",
    "PreparedInputTooLargeError",
    "PreparedRunEnvelope",
    "ReclaimDecision",
    "ReclaimState",
    "RunAccessScope",
    "RunArtifactReference",
    "RunCapacityExceededError",
    "RunFetchedResource",
    "RunCreation",
    "RunDeletion",
    "RunExecutionContext",
    "ShutdownOutcome",
    "SweepOutcome",
    "TerminalOutcome",
    "WaitingForRepair",
    "Succeeded",
    "advance_reclaim",
    "run_request_fingerprint",
    "artifact_digest",
    "canonical_run_request_json",
    "parse_run_id",
    "require_prepared_input_bounds",
]
