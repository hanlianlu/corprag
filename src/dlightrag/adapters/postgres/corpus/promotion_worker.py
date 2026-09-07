# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Automatic hot-workspace promotion worker (Commit 3 control plane).

One background task per writer process claims durable promotion jobs with the
existing lease fencing, then drives one workspace through an idempotent,
recoverable state machine:

    claim (generation N) -> durable write fence -> exclusive write gate
    -> reconcile/verify -> staged copy per table -> verify checksums
    -> ONE transaction: recheck lease+fence, delete the DEFAULT copies,
       ATTACH every staging partition, flip registry tier/state, release the
       fence, mark the job done.

Crash anywhere before the cutover transaction leaves the workspace fully
shared: staging relations are deterministic, dropped on the next attempt, and
the fence expires or is released by the next claim. The cutover transaction is
all-or-nothing across every retrieval-critical parent, so there is no
half-promoted state. A crash after the cutover commits is reconciled by the
next attempt (or a later admin read) because the attached dedicated partition
is detected and only the remaining bookkeeping is completed.

Partitioning stays physical: every query keeps its authenticated
``workspace = $n`` predicate and application roles keep querying parents.
"""

import asyncio
import datetime
import hashlib
import logging
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any

from dlightrag.adapters.postgres.core._pool import pg_pool
from dlightrag.adapters.postgres.core.identifiers import pg_identifier, pg_qualified_identifier
from dlightrag.adapters.postgres.corpus.partition_foundation import (
    child_partition_name,
    default_child_name,
)
from dlightrag.adapters.postgres.corpus.pg_metadata_index import (
    rebuild_metadata_field_stats_for_workspace,
)
from dlightrag.adapters.postgres.corpus.promotion_jobs import PGPromotionJobStore
from dlightrag.adapters.postgres.corpus.workspace_write_gate import workspace_write_gate
from dlightrag.adapters.postgres.corpus.workspaces import PGWorkspaceRegistry

logger = logging.getLogger(__name__)

# Retrieval-critical parents: the DlightRAG metadata table, LightRAG's chunk
# table (and therefore its BM25 child indexes), and every attached chunk-vector
# table (discovered dynamically by the vector-name prefix).
_METADATA_TABLE = "dlightrag_doc_metadata"
_CHUNKS_TABLE = "LIGHTRAG_DOC_CHUNKS"
_VECTOR_TABLE_PREFIX = "lightrag_vdb_chunks"

_PROMOTION_ERROR_PREFIX = "promotion failed"

_LEASE_RENEW_INTERVAL_DIVISOR = 3  # renew when one third of the lease remains

# Guarded failure transition: the job row proves the attempt still owns its
# unexpired lease; the registry update additionally proves the fence owner.
_FAIL_JOB_GUARDED = """
UPDATE dlightrag_promotion_jobs
SET state = 'failed',
    last_error = $4,
    next_retry_at = $5::timestamptz,
    lease_owner = NULL,
    lease_until = NULL,
    updated_at = NOW()
WHERE job_id = $1
  AND state = 'promoting'
  AND lease_owner = $2
  AND lease_generation = $3
  AND lease_until > NOW()
RETURNING 1
"""

_FAIL_REGISTRY_GUARDED = """
UPDATE dlightrag_workspace_meta
SET promotion_state = 'failed',
    promotion_last_error = $3,
    promotion_retry_count = promotion_retry_count + 1,
    promotion_next_retry_at = $4::timestamptz,
    write_fence_owner = NULL,
    write_fence_until = NULL,
    updated_at = NOW()
WHERE workspace = $1
  AND write_fence_owner = $2
RETURNING 1
"""

# Guarded success flips: both must affect exactly one row or the cutover
# transaction rolls back (stale attempts can never flip a newer state).
_FLIP_HOT_GUARDED = """
UPDATE dlightrag_workspace_meta
SET storage_tier = 'hot',
    promotion_state = 'none',
    promotion_last_error = NULL,
    promotion_next_retry_at = NULL,
    write_fence_owner = NULL,
    write_fence_until = NULL,
    updated_at = NOW()
WHERE workspace = $1
  AND write_fence_owner = $2
  AND write_fence_until > NOW()
"""

_DONE_GUARDED = """
UPDATE dlightrag_promotion_jobs
SET state = 'done',
    promoted_at = NOW(),
    last_error = NULL,
    next_retry_at = NULL,
    lease_owner = NULL,
    lease_until = NULL,
    updated_at = NOW()
WHERE job_id = $1
  AND state = 'promoting'
  AND lease_owner = $2
  AND lease_generation = $3
  AND lease_until > NOW()
"""


@dataclass(frozen=True, slots=True)
class PromotionJobClaim:
    job_id: int
    workspace: str
    attempt_count: int
    lease_generation: int
    owner: str


class PromotionAttemptError(RuntimeError):
    """One promotion attempt failed; the job transitions to failed/retry."""


class StalePromotionAttempt(RuntimeError):
    """This attempt's lease or fence is no longer current; abandon quietly."""


def _raise_if_renewal_lost(renewal_lost: asyncio.Event) -> None:
    if renewal_lost.is_set():
        raise StalePromotionAttempt("promotion lease/fence was lost during copy")


def staging_partition_name(table_name: str, workspace: str) -> str:
    """Deterministic internal name for one detached pre-attach staging table."""
    parent_digest = hashlib.sha256(pg_identifier(table_name).lower().encode("utf-8")).hexdigest()[
        :10
    ]
    workspace_digest = hashlib.sha256(str(workspace).encode("utf-8")).hexdigest()[:16]
    return f"s_{parent_digest}_w_{workspace_digest}"


class PGPromotionWorker:
    """Leased background driver for automatic hot-workspace promotion."""

    def __init__(
        self,
        *,
        job_store: PGPromotionJobStore,
        registry: PGWorkspaceRegistry,
        lease_seconds: int = 1800,
        retry_backoff_seconds: int = 600,
        claim_poll_seconds: float = 5.0,
    ) -> None:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        if retry_backoff_seconds < 1:
            raise ValueError("retry_backoff_seconds must be positive")
        if claim_poll_seconds <= 0:
            raise ValueError("claim_poll_seconds must be positive")
        self._job_store = job_store
        self._registry = registry
        self._lease_seconds = lease_seconds
        self._retry_backoff_seconds = retry_backoff_seconds
        self._claim_poll_seconds = claim_poll_seconds
        self._owner = f"dlightrag-promotion-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._task: asyncio.Task[None] | None = None
        self._closing = False
        self._current_attempt_gated = False

    def start(self) -> None:
        """Begin the background claim loop (idempotent)."""
        if self._task is not None and not self._task.done():
            return
        self._closing = False
        self._task = asyncio.create_task(self._loop(), name="dlightrag-promotion-worker")

    async def aclose(self) -> None:
        self._closing = True
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def run_once(self) -> bool:
        """Claim and run at most one due job; True when work was performed."""
        lease_until = datetime.datetime.now(datetime.UTC) + datetime.timedelta(
            seconds=self._lease_seconds
        )
        row = await self._job_store.claim_next(owner=self._owner, lease_until=lease_until)
        if row is None:
            return False
        claim = PromotionJobClaim(
            job_id=int(row["job_id"]),
            workspace=str(row["workspace"]),
            attempt_count=int(row["attempt_count"]),
            lease_generation=int(row["lease_generation"]),
            owner=self._owner,
        )
        logger.info(
            "Promotion attempt %d (generation %d) claimed for workspace '%s'",
            claim.attempt_count,
            claim.lease_generation,
            claim.workspace,
        )
        # Per-attempt state must reset before fence acquisition: an exception
        # there must not inherit the previous run's gated=True value and skip
        # its guarded failure transition.
        self._current_attempt_gated = False
        try:
            await self._run_attempt(claim)
        except StalePromotionAttempt:
            logger.warning(
                "Promotion attempt for workspace '%s' lost its lease/fence; yielding",
                claim.workspace,
            )
        except asyncio.CancelledError:
            # The in-gate handler already cleaned artifacts and performed the
            # guarded failed transition when applicable.
            raise
        except Exception as exc:
            logger.warning("Promotion attempt failed: %s", exc, exc_info=True)
            if not self._current_attempt_gated:
                # Pre-gate failure (e.g. the fence was unavailable): no
                # artifacts exist yet, so the guarded failed transition is
                # the only cleanup needed.
                await self._fail_attempt(
                    claim,
                    f"{_PROMOTION_ERROR_PREFIX}: {type(exc).__name__}: {exc}",
                )
        return True

    async def _loop(self) -> None:
        while not self._closing:
            try:
                performed = await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Promotion claim loop iteration failed", exc_info=True)
                performed = False
            if not performed:
                await asyncio.sleep(self._claim_poll_seconds)

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    async def _run_attempt(self, claim: PromotionJobClaim) -> None:
        workspace = claim.workspace
        fence_owner = f"{self._owner}#{claim.lease_generation}"

        acquired = await self._registry.acquire_write_fence(
            workspace=workspace,
            owner=fence_owner,
            until=datetime.datetime.now(datetime.UTC)
            + datetime.timedelta(seconds=self._lease_seconds),
        )
        if not acquired:
            raise PromotionAttemptError("write fence unavailable")
        state_set = await self._registry.set_promotion_state(
            workspace=workspace,
            state="promoting",
            expected_fence_owner=fence_owner,
        )
        if not state_set:
            await self._registry.release_write_fence(
                workspace=workspace,
                owner=fence_owner,
            )
            raise StalePromotionAttempt("promotion fence changed before state transition")

        renewal_lost = asyncio.Event()
        renewal_task = asyncio.create_task(
            self._renew_while(
                claim=claim,
                fence_owner=fence_owner,
                renewal_lost=renewal_lost,
            )
        )
        self._current_attempt_gated = False
        try:
            async with workspace_write_gate(workspace, exclusive=True) as conn:
                self._current_attempt_gated = True
                try:
                    await self._copy_and_cutover(
                        conn,
                        claim,
                        fence_owner,
                        renewal_lost,
                    )
                except BaseException as exc:
                    # Any normal failure or cancellation while still inside
                    # the exclusive gate: clean the current attempt's
                    # artifacts BEFORE the unlock. Only when that cleanup
                    # succeeds may the job transition to failed/retry and the
                    # fence be released; a failed cleanup leaves job and
                    # registry 'promoting' for lease recovery — the next
                    # reclaimed worker removes the artifacts itself.
                    cleanup_ok = False
                    try:
                        await self._cleanup_artifacts_on(conn, workspace)
                        cleanup_ok = True
                    except Exception:
                        logger.error(
                            "Promotion artifact cleanup for workspace '%s' failed; "
                            "leaving job and registry promoting for lease recovery",
                            workspace,
                            exc_info=True,
                        )
                    if cleanup_ok and not isinstance(exc, StalePromotionAttempt):
                        await self._fail_attempt(
                            claim,
                            f"{_PROMOTION_ERROR_PREFIX}: {type(exc).__name__}: {exc}",
                        )
                    raise
        finally:
            renewal_task.cancel()
            try:
                await renewal_task
            except asyncio.CancelledError, Exception:
                logger.debug(
                    "Promotion lease renewal task ended for workspace '%s'",
                    workspace,
                )

    async def _renew_while(
        self,
        *,
        claim: PromotionJobClaim,
        fence_owner: str,
        renewal_lost: asyncio.Event,
    ) -> None:
        """Renew the attempt, signaling the copy loop when ownership is lost."""
        interval = max(1.0, self._lease_seconds / _LEASE_RENEW_INTERVAL_DIVISOR)
        try:
            while True:
                await asyncio.sleep(interval)
                lease_until = datetime.datetime.now(datetime.UTC) + datetime.timedelta(
                    seconds=self._lease_seconds
                )
                renewed = await self._job_store.renew_lease(
                    job_id=claim.job_id,
                    owner=claim.owner,
                    lease_generation=claim.lease_generation,
                    lease_until=lease_until,
                )
                if not renewed:
                    renewal_lost.set()
                    return
                fence_renewed = await self._registry.acquire_write_fence(
                    workspace=claim.workspace,
                    owner=fence_owner,
                    until=lease_until,
                )
                if not fence_renewed:
                    renewal_lost.set()
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            renewal_lost.set()
            logger.warning(
                "Promotion lease/fence renewal failed for workspace '%s'",
                claim.workspace,
                exc_info=True,
            )

    async def _copy_and_cutover(
        self,
        conn: Any,
        claim: PromotionJobClaim,
        fence_owner: str,
        renewal_lost: asyncio.Event,
    ) -> None:
        workspace = claim.workspace
        _raise_if_renewal_lost(renewal_lost)
        await self._recheck_current(conn, claim, fence_owner)

        tables = await _discover_retrieval_parents(conn)
        if not tables:
            raise PromotionAttemptError("no partitioned retrieval parents discovered")

        staged: list[tuple[str, str, str]] = []  # (parent, staging, child)
        for parent in tables:
            _raise_if_renewal_lost(renewal_lost)
            child = child_partition_name(parent, workspace)
            if await _child_is_attached(conn, parent, child, workspace):
                # A previous cutover already attached this parent's dedicated
                # partition (crash after commit). Verify it and reconcile.
                await _verify_attached_partition(conn, parent, child, workspace)
                continue
            staging = staging_partition_name(parent, workspace)
            # Crash recovery while we hold the exclusive gate: a newer worker
            # cannot have entered, so our deterministic staging/exclusion
            # leftovers are safe to remove before rebuilding.
            await _drop_stale_artifacts(conn, parent, workspace)
            await _create_staging(conn, parent, staging, workspace)
            await _copy_workspace_rows(conn, parent, staging, workspace)
            await _verify_copy_checksums(conn, parent, staging, workspace)
            _raise_if_renewal_lost(renewal_lost)
            await _build_staging_indexes(conn, parent, staging)
            _raise_if_renewal_lost(renewal_lost)
            staged.append((parent, staging, child))

        # Phase 1 (autocommit, inside the exclusive gate): commit each
        # deterministic DEFAULT exclusion proof outside the cutover
        # transaction, so its AccessExclusive DDL lock releases immediately
        # instead of covering the later multi-table scans.
        workspace_literal = await conn.fetchval("SELECT quote_literal($1)", workspace)
        for parent, _staging, _child in staged:
            await _add_exclusion_check(conn, parent, workspace, workspace_literal)
        if _PHASE1_PAUSE_HOOK is not None:
            await _PHASE1_PAUSE_HOOK(conn)
        _raise_if_renewal_lost(renewal_lost)

        # Phase 2: the one atomic cutover.
        async with conn.transaction():
            await self._recheck_current(conn, claim, fence_owner, for_update=True)
            # DELETE + VALIDATE for ALL parents before any ATTACH lock: the
            # VALIDATE scans run under ShareUpdateExclusive (reads and
            # other-workspace RowExclusive DML keep flowing) and are never
            # interleaved with an early ATTACH.
            for parent, _staging, _child in staged:
                await conn.execute(
                    f"DELETE FROM ONLY {pg_identifier(default_child_name(parent))} "  # noqa: S608
                    "WHERE workspace = $1",
                    workspace,
                )
                await conn.execute(
                    f"ALTER TABLE ONLY {pg_identifier(default_child_name(parent))} "  # noqa: S608
                    f"VALIDATE CONSTRAINT {_exclusion_constraint_name(parent, workspace)}"
                )
            if _CUTOVER_PAUSE_HOOK is not None:
                await _CUTOVER_PAUSE_HOOK(conn)
            # Only now the short RENAME/ATTACH loop.
            for parent, staging, child in staged:
                await conn.execute(
                    f"ALTER TABLE {pg_identifier(staging)} "  # noqa: S608 - deterministic
                    f"RENAME TO {child}"
                )
                await conn.execute(
                    f"ALTER TABLE {pg_qualified_identifier(parent)} "  # noqa: S608
                    f"ATTACH PARTITION {child} FOR VALUES IN ({workspace_literal})"
                )
                await conn.execute(
                    f"ALTER TABLE ONLY {pg_identifier(default_child_name(parent))} "  # noqa: S608
                    f"DROP CONSTRAINT {_exclusion_constraint_name(parent, workspace)}"
                )
            # Copying into detached staging tables does not fire the metadata
            # trigger; deleting DEFAULT copies fires the parent's cloned trigger.
            # Recount after ATTACH so the logical move leaves availability intact.
            await rebuild_metadata_field_stats_for_workspace(conn, workspace)
            flipped = await conn.execute(_FLIP_HOT_GUARDED, workspace, fence_owner)
            if flipped == "UPDATE 0":
                raise StalePromotionAttempt("registry flip refused: fence not current")
            done = await conn.execute(
                _DONE_GUARDED,
                claim.job_id,
                claim.owner,
                claim.lease_generation,
            )
            if done == "UPDATE 0":
                raise StalePromotionAttempt("job completion refused: lease not current")
        logger.info(
            "Workspace '%s' promoted to dedicated partitions across %d table(s)",
            workspace,
            len(staged) or len(tables),
        )

    async def _recheck_current(
        self,
        conn: Any,
        claim: PromotionJobClaim,
        fence_owner: str,
        *,
        for_update: bool = False,
    ) -> None:
        """Prove this attempt still owns the lease and the fence.

        The cutover calls this inside its own transaction with row locks, so a
        stale worker can never attach partitions or flip registry state after
        its lease was reclaimed.
        """
        lock = " FOR UPDATE" if for_update else ""
        job_row = await conn.fetchrow(
            f"""
            SELECT state, lease_owner, lease_generation, lease_until
            FROM dlightrag_promotion_jobs
            WHERE job_id = $1{lock}
            """,  # noqa: S608 - suffix is a fixed literal
            claim.job_id,
        )
        if (
            job_row is None
            or job_row["state"] != "promoting"
            or job_row["lease_owner"] != claim.owner
            or int(job_row["lease_generation"]) != claim.lease_generation
            or job_row["lease_until"] is None
            or job_row["lease_until"] <= datetime.datetime.now(datetime.UTC)
        ):
            raise StalePromotionAttempt("promotion lease is not current")
        fence_row = await conn.fetchrow(
            f"""
            SELECT write_fence_owner, write_fence_until
            FROM dlightrag_workspace_meta
            WHERE workspace = $1{lock}
            """,  # noqa: S608 - suffix is a fixed literal
            claim.workspace,
        )
        if (
            fence_row is None
            or fence_row["write_fence_owner"] != fence_owner
            or fence_row["write_fence_until"] is None
            or fence_row["write_fence_until"] <= datetime.datetime.now(datetime.UTC)
        ):
            raise StalePromotionAttempt("promotion write fence is not current")

    async def _fail_attempt(self, claim: PromotionJobClaim, error: str) -> None:
        """Fail one attempt in one guarded transaction.

        Job transition, registry failed/retry observability, and the
        owned-fence release commit together, guarded by job_id + owner +
        generation + unexpired lease and a matching fence owner. A stale
        attempt (lease reclaimed or fence taken over) mutates nothing — in
        particular it can never clobber a newer worker's 'promoting'
        observability — and it does not touch staging the newer worker owns.
        """
        workspace = claim.workspace
        fence_owner = f"{self._owner}#{claim.lease_generation}"
        next_retry = datetime.datetime.now(datetime.UTC) + datetime.timedelta(
            seconds=self._retry_backoff_seconds
        )

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                job_guard = await conn.fetchval(
                    _FAIL_JOB_GUARDED,
                    claim.job_id,
                    claim.owner,
                    claim.lease_generation,
                    error,
                    next_retry,
                )
                if not int(job_guard or 0):
                    return False
                registry_guard = await conn.fetchval(
                    _FAIL_REGISTRY_GUARDED,
                    workspace,
                    fence_owner,
                    error,
                    next_retry,
                )
                if not int(registry_guard or 0):
                    # Fence was taken over (expired): roll the job transition
                    # back — nothing we own may be mutated.
                    raise StalePromotionAttempt(
                        "promotion fence was taken over during failure handling"
                    )
                return True

        try:
            transitioned = await pg_pool.run_once(_operation)
        except StalePromotionAttempt:
            logger.warning(
                "Promotion attempt for workspace '%s' lost its fence during "
                "failure handling; yielding without mutation",
                workspace,
            )
            return
        if not transitioned:
            logger.warning(
                "Promotion attempt for workspace '%s' was reclaimed before the "
                "failure transition; yielding without mutation",
                workspace,
            )
            return

    async def _cleanup_artifacts_on(self, conn: Any, workspace: str) -> None:
        """Drop this attempt's deterministic artifacts without schema assumptions.

        Cleanup must remain possible when strict parent discovery is the thing
        that failed (for example a missing DEFAULT child). Catalog scans use
        only the reserved staging prefix and this workspace's internal digest,
        so a current worker can still honor the contract: clean/quarantine its
        artifacts, release the fence, and leave the workspace fully shared.
        The exclusive gate prevents a newer generation from creating same-name
        artifacts until this method returns.
        """
        workspace_digest = hashlib.sha256(str(workspace).encode("utf-8")).hexdigest()[:16]
        staging_suffix = f"_w_{workspace_digest}"
        staging_rows = await conn.fetch(
            """
            SELECT c.relname AS name
            FROM pg_catalog.pg_class c
            JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public'
              AND c.relkind = 'r'
              AND left(c.relname, 2) = 's_'
              AND right(c.relname, length($1)) = $1
            ORDER BY c.relname
            """,
            staging_suffix,
        )
        for row in staging_rows:
            await conn.execute(f"DROP TABLE {pg_identifier(str(row['name']))}")  # noqa: S608

        constraint_suffix = f"_{workspace_digest}_excl"
        constraint_rows = await conn.fetch(
            """
            SELECT table_rel.relname AS table_name, con.conname AS constraint_name
            FROM pg_catalog.pg_constraint con
            JOIN pg_catalog.pg_class table_rel ON table_rel.oid = con.conrelid
            JOIN pg_catalog.pg_namespace n ON n.oid = table_rel.relnamespace
            WHERE n.nspname = 'public'
              AND con.contype = 'c'
              AND right(con.conname, length($1)) = $1
            ORDER BY table_rel.relname, con.conname
            """,
            constraint_suffix,
        )
        for row in constraint_rows:
            await conn.execute(
                f"ALTER TABLE ONLY {pg_identifier(str(row['table_name']))} "  # noqa: S608
                f"DROP CONSTRAINT {pg_identifier(str(row['constraint_name']))}"
            )


# ---------------------------------------------------------------------------
# Database helpers (run inside the exclusive write gate)
# ---------------------------------------------------------------------------


async def _discover_retrieval_parents(conn: Any) -> list[str]:
    """Return every partitioned retrieval-critical parent, failing loudly on gaps.

    The metadata parent, the chunks parent, and at least one prefix-exact
    chunk-vector parent are mandatory; every parent's DEFAULT child must exist
    too. A promotion must never silently cover a subset of the
    retrieval-critical set. The vector prefix is matched with ``position()``
    so ``_`` never acts as a wildcard, and a prefix-matching relation of the
    wrong relation kind (a broken/plain vector parent) fails loudly instead of
    being silently excluded.
    """
    rows = await conn.fetch(
        """
        SELECT c.relname AS name, c.relkind
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relkind NOT IN ('i', 'I')  -- index relations share names with
                                           -- the vector parent's PK and are
                                           -- never broken parents themselves
          AND (lower(c.relname) = $1 OR lower(c.relname) = $2
               OR position($3 in lower(c.relname)) = 1)
          AND EXISTS (
              SELECT 1 FROM pg_catalog.pg_attribute a
              WHERE a.attrelid = c.oid AND a.attname = 'workspace' AND a.attnum > 0
          )
        ORDER BY c.relname
        """,
        _METADATA_TABLE.lower(),
        _CHUNKS_TABLE.lower(),
        _VECTOR_TABLE_PREFIX,
    )
    required_names = {_METADATA_TABLE.lower(), _CHUNKS_TABLE.lower()}
    vector_names: list[str] = []
    parents: list[str] = []
    for row in rows:
        table = str(row["name"])
        is_vector = (
            table.lower().startswith(_VECTOR_TABLE_PREFIX) and table.lower() not in required_names
        )
        relkind = row["relkind"]
        if isinstance(relkind, bytes):
            relkind = relkind.decode("ascii")
        if is_vector and relkind != "p":
            raise PromotionAttemptError(
                f"chunk-vector relation {table} is not a partitioned parent (relkind {relkind!r})"
            )
        if is_vector:
            vector_names.append(table)
        default_child = default_child_name(table)
        has_default = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM pg_catalog.pg_inherits i
                JOIN pg_catalog.pg_class child ON child.oid = i.inhrelid
                JOIN pg_catalog.pg_class parent ON parent.oid = i.inhparent
                WHERE parent.oid = to_regclass($1) AND child.relname = $2
            )
            """,
            table,
            default_child,
        )
        if not has_default:
            raise PromotionAttemptError(f"retrieval parent {table} has no shared DEFAULT partition")
        parents.append(table)
    missing_required = [
        required
        for required in sorted(required_names)
        if required not in {table.lower() for table in parents}
    ]
    if missing_required:
        raise PromotionAttemptError(
            "required retrieval parents are missing: " + ", ".join(missing_required)
        )
    if not vector_names:
        raise PromotionAttemptError(
            "no partitioned chunk-vector parent discovered; promotion must never "
            "cover a subset of the retrieval-critical set"
        )
    return parents


async def _relation_exists(conn: Any, table_name: str) -> bool:
    return bool(
        await conn.fetchval(
            "SELECT to_regclass($1) IS NOT NULL",
            table_name,
        )
    )


async def _drop_relation(conn: Any, table_name: str) -> None:
    if await _relation_exists(conn, table_name):
        await conn.execute(f"DROP TABLE {pg_identifier(table_name)}")  # noqa: S608


async def _parent_columns(conn: Any, parent: str) -> list[str]:
    rows = await conn.fetch(
        """
        SELECT a.attname AS name
        FROM pg_catalog.pg_attribute a
        WHERE a.attrelid = to_regclass($1)
          AND a.attnum > 0
          AND NOT a.attisdropped
        ORDER BY a.attnum
        """,
        parent,
    )
    return [str(row["name"]) for row in rows]


async def _create_staging(conn: Any, parent: str, staging: str, workspace: str) -> None:
    """Detached copy of the parent's shape plus the exact partition bound.

    Parent CHECK constraints are required on the detached table before it can
    be attached as a partition. The explicit workspace constraint matches the
    future partition bound, so ATTACH PARTITION skips the whole-table scan.
    """
    await conn.execute(
        f"CREATE TABLE {pg_identifier(staging)} ("  # noqa: S608
        f"LIKE {pg_qualified_identifier(parent)} "
        "INCLUDING DEFAULTS INCLUDING CONSTRAINTS INCLUDING STORAGE INCLUDING COMPRESSION"
        ")"
    )
    workspace_literal = await conn.fetchval("SELECT quote_literal($1)", workspace)
    await conn.execute(
        f"ALTER TABLE {pg_identifier(staging)} "  # noqa: S608
        f"ADD CONSTRAINT {pg_identifier(f'{staging}_workspace_chk')} "
        f"CHECK (workspace = {workspace_literal})"
    )


async def _copy_workspace_rows(
    conn: Any,
    parent: str,
    staging: str,
    workspace: str,
) -> None:
    """Copy only this workspace's rows out of the shared DEFAULT partition.

    Reading through ONLY the DEFAULT child keeps the copy correct even when a
    dedicated partition already exists (that case is reconciled before we get
    here) and never touches other workspaces' rows.
    """
    columns = await _parent_columns(conn, parent)
    column_list = ", ".join(pg_identifier(column) for column in columns)
    default_child = default_child_name(parent)
    await conn.execute(
        f"INSERT INTO {pg_identifier(staging)} ({column_list}) "  # noqa: S608
        f"SELECT {column_list} FROM ONLY {pg_identifier(default_child)} WHERE workspace = $1",
        workspace,
    )


async def _verify_copy_checksums(
    conn: Any,
    parent: str,
    staging: str,
    workspace: str,
) -> None:
    """Compare row counts and streamed row checksums on both sides."""
    default_child = default_child_name(parent)
    row = await conn.fetchrow(
        f"""
        SELECT
            (SELECT COUNT(*)::bigint FROM ONLY {pg_identifier(default_child)}
             WHERE workspace = $1) AS expected_count,
            (SELECT COUNT(*)::bigint FROM {pg_identifier(staging)}) AS staged_count,
            (SELECT COALESCE(SUM(hashtextextended(md5((t.*)::text), 0)), 0)::numeric
             FROM ONLY {pg_identifier(default_child)} t WHERE t.workspace = $1)
                AS expected_checksum,
            (SELECT COALESCE(SUM(hashtextextended(md5((t.*)::text), 0)), 0)::numeric
             FROM {pg_identifier(staging)} t) AS staged_checksum
        """,  # noqa: S608 - deterministic identifiers
        workspace,
    )
    if row is None or int(row["expected_count"]) != int(row["staged_count"]):
        raise PromotionAttemptError(f"copy verification failed for {parent}: row counts differ")
    if int(row["expected_checksum"]) != int(row["staged_checksum"]):
        raise PromotionAttemptError(f"copy verification failed for {parent}: row checksums differ")


async def _build_staging_indexes(conn: Any, parent: str, staging: str) -> None:
    """Recreate every parent index on the detached staging table.

    PostgreSQL reuses definition-compatible child indexes and constraints at
    ATTACH time, so pre-building BM25/GIN/HNSW indexes *and the primary key*
    keeps the parent ACCESS EXCLUSIVE window short instead of building a
    10M/100M-row child PK under the parent lock. The PK is rebuilt as a plain
    UNIQUE index and converted into the staging PRIMARY KEY constraint via
    ``USING INDEX``, which is the identity ATTACH reuses.
    """
    rows = await conn.fetch(
        """
        SELECT i.relname AS indexname, pg_get_indexdef(i.oid) AS definition,
               idx.indisprimary
        FROM pg_catalog.pg_index idx
        JOIN pg_catalog.pg_class i ON i.oid = idx.indexrelid
        JOIN pg_catalog.pg_class t ON t.oid = idx.indrelid
        JOIN pg_catalog.pg_namespace n ON n.oid = t.relnamespace
        WHERE n.nspname = 'public' AND t.relname = $1
          AND idx.indisvalid
          AND idx.indisready
        """,
        parent,
    )
    primary_index: str | None = None
    for row in rows:
        index_name = str(row["indexname"])
        # Short deterministic staging index name: prefixing the parent name
        # would exceed PostgreSQL's 63-byte identifier limit.
        digest = hashlib.sha256(f"{staging}:{parent}:{index_name}".encode()).hexdigest()[:12]
        staged_index = pg_identifier(f"si_{digest}")
        definition = str(row["definition"])
        # Re-target the table reference (quoted or unquoted, ON or ON ONLY)
        # at the staging relation, then rename the index itself.
        retargeted = re.sub(
            r"( ON ONLY | ON )public\.[^\s]+ USING ",
            f" ON {pg_identifier(staging)} USING ",
            definition,
            count=1,
        )
        if retargeted == definition:
            raise PromotionAttemptError(
                f"could not re-target index definition for {parent}.{index_name}"
            )
        retargeted = retargeted.replace(f"INDEX {index_name} ", f"INDEX {staged_index} ", 1)
        await conn.execute(retargeted)  # noqa: S608 - rewritten trusted definition
        if bool(row["indisprimary"]):
            primary_index = staged_index
    if primary_index is None:
        raise PromotionAttemptError(f"parent {parent} exposes no primary key index")
    await conn.execute(
        f"ALTER TABLE {pg_identifier(staging)} "  # noqa: S608 - deterministic
        f"ADD CONSTRAINT {pg_identifier(f'{staging}_pkey')} "
        f"PRIMARY KEY USING INDEX {primary_index}"
    )


# Test seams: called between protocol phases so real-PG tests can prove the
# lock-phase ordering (reads/DML continue after phase 1 commits and during the
# cutover's pre-ATTACH validation window). None in production.
_PHASE1_PAUSE_HOOK: Any | None = None
_CUTOVER_PAUSE_HOOK: Any | None = None


async def _drop_stale_artifacts(conn: Any, parent: str, workspace: str) -> None:
    """Remove a crashed attempt's staging table and exclusion proof.

    Runs only while this worker holds the exclusive gate (a newer worker
    cannot have entered), so the deterministic names always refer to our own
    leftovers.
    """
    await _drop_relation(conn, staging_partition_name(parent, workspace))
    default_child = default_child_name(parent)
    await conn.execute(
        f"ALTER TABLE ONLY {pg_identifier(default_child)} "  # noqa: S608
        f"DROP CONSTRAINT IF EXISTS {_exclusion_constraint_name(parent, workspace)}"
    )


async def _add_exclusion_check(
    conn: Any,
    parent: str,
    workspace: str,
    workspace_literal: str,
) -> None:
    """Commit one DEFAULT exclusion proof (autocommit) for the final cutover.

    ``ADD CONSTRAINT ... CHECK (...) NOT VALID`` takes AccessExclusiveLock on
    the DEFAULT child and retains it until commit — running it inside the
    final cutover transaction would block reads and other-workspace DML on
    that child for the whole multi-table scan. Each statement therefore runs
    standalone (the exclusive advisory gate still fences this workspace's
    writers), so the DDL lock commits and releases immediately. The proof is
    persistent: a crash between this phase and the cutover leaves it behind
    for the next reclaimed worker to drop and rebuild.
    """
    default_child = default_child_name(parent)
    tmp_constraint = _exclusion_constraint_name(parent, workspace)
    await conn.execute(
        f"ALTER TABLE ONLY {pg_identifier(default_child)} "  # noqa: S608
        f"ADD CONSTRAINT {tmp_constraint} "
        f"CHECK (workspace <> {workspace_literal}) NOT VALID"
    )


def _exclusion_constraint_name(parent: str, workspace: str) -> str:
    """The deterministic temporary exclusion constraint used in the cutover."""
    workspace_hash = hashlib.sha256(str(workspace).encode()).hexdigest()[:16]
    return pg_identifier(f"{default_child_name(parent)}_{workspace_hash}_excl")


async def _child_is_attached(
    conn: Any,
    parent: str,
    child: str,
    workspace: str,
) -> bool:
    """Return whether ``child`` is a real attached partition of ``parent``.

    A detached relation that merely reuses the deterministic child name is
    NOT proof of attachment: the pg_inherits link and the exact LIST bound for
    the canonical workspace must both hold. A conflicting relation fails
    loudly instead of being silently skipped or overwritten.
    """
    exists = await conn.fetchval(
        "SELECT to_regclass($1) IS NOT NULL",
        child,
    )
    if not exists:
        return False
    workspace_literal = await conn.fetchval("SELECT quote_literal($1)", workspace)
    expected_bound = f"FOR VALUES IN ({workspace_literal})"
    link = await conn.fetchrow(
        """
        SELECT pg_get_expr(child.relpartbound, child.oid) AS bound
        FROM pg_catalog.pg_inherits i
        JOIN pg_catalog.pg_class child ON child.oid = i.inhrelid
        JOIN pg_catalog.pg_class parent ON parent.oid = i.inhparent
        WHERE parent.oid = to_regclass($1) AND child.relname = $2
        """,
        parent,
        child,
    )
    if link is None:
        raise PromotionAttemptError(
            f"relation {child} exists but is not attached to parent {parent}"
        )
    if str(link["bound"] or "") != expected_bound:
        raise PromotionAttemptError(
            f"attached partition {child} bound {link['bound']!r} != {expected_bound!r}"
        )
    return True


async def _verify_attached_partition(
    conn: Any,
    parent: str,
    child: str,
    workspace: str,
) -> None:
    """Reconcile a dedicated partition a previous cutover already attached.

    The reconciliation proof is strict: the child must contain no rows for
    other workspaces (a conflicting fill is corruption, not bookkeeping), and
    the DEFAULT child must no longer hold this workspace's copies. An empty
    child beside a populated DEFAULT would lose data on the flip, so it is a
    hard failure instead of a silent reconciliation.
    """
    row = await conn.fetchrow(
        f"""
        SELECT
            (SELECT COUNT(*)::bigint FROM ONLY {pg_identifier(child)}
             WHERE workspace <> $1) AS foreign_count,
            (SELECT COUNT(*)::bigint FROM ONLY {pg_identifier(child)}) AS child_count,
            (SELECT COUNT(*)::bigint FROM ONLY {pg_identifier(default_child_name(parent))}
             WHERE workspace = $1) AS default_count
        """,  # noqa: S608
        workspace,
    )
    if row is None:
        raise PromotionAttemptError(f"attached partition verification failed for {parent}")
    if int(row["foreign_count"]) != 0:
        raise PromotionAttemptError(f"attached partition {child} holds rows for other workspaces")
    if int(row["child_count"]) == 0 and int(row["default_count"]) > 0:
        raise PromotionAttemptError(
            f"attached partition for {parent} is empty while DEFAULT still holds rows"
        )
    if int(row["default_count"]) > 0:
        raise PromotionAttemptError(
            f"attached partition for {parent} exists while DEFAULT still holds "
            f"{row['default_count']} copied row(s)"
        )


__all__ = [
    "PGPromotionWorker",
    "PromotionAttemptError",
    "PromotionJobClaim",
    "StalePromotionAttempt",
    "_exclusion_constraint_name",
    "staging_partition_name",
]
