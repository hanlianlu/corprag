#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Accept durable Reset Corpus Mutation Runs for explicit Workspaces."""

import argparse
import asyncio
import logging
import sys
from collections.abc import Mapping, Sequence
from uuid import uuid7

from dlightrag.application.access import AccessAction, AccessGate, AllowAllAccessControl
from dlightrag.application.corpus_admin import CorpusAdmin, CorpusMutationService
from dlightrag.application.runs import RunService, RunView
from dlightrag.engine.rag.workspace.workspaces import normalize_workspace

_SUBMITTED_BY = "operator:reset-workspace"
_TERMINAL = frozenset({"succeeded", "failed", "cancelled"})
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dlightrag-reset-workspace",
        description="Submit durable Reset Corpus Mutation Runs",
        suggest_on_error=True,
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--workspace", default=None, help="Target Corpus Workspace")
    scope.add_argument(
        "--all",
        dest="reset_all",
        action="store_true",
        help="Reset every authorized Corpus Workspace",
    )
    return parser


async def _authorized_reset_scope(
    corpora: CorpusAdmin,
    gate: AccessGate,
    *,
    workspace: str | None,
    reset_all: bool,
    default_workspace: str,
) -> tuple[str, ...]:
    if reset_all:
        records = await gate.filter_workspace_records(
            AccessAction.WORKSPACE_RESET,
            await corpora.alist_workspace_records(),
        )
        workspace_ids = tuple(record["workspace"] for record in records)
        if not workspace_ids:
            raise ValueError("No authorized Corpus Workspaces are available to reset")
        return workspace_ids

    raw_workspace = workspace if workspace is not None else default_workspace
    if not raw_workspace.strip() or raw_workspace.strip() == "*":
        raise ValueError("--workspace requires a concrete workspace name")
    workspace_id = normalize_workspace(raw_workspace)
    await gate.check(AccessAction.WORKSPACE_RESET, workspace=workspace_id)
    return (workspace_id,)


async def _submit_reset_runs(
    corpora: CorpusAdmin,
    mutations: CorpusMutationService,
    gate: AccessGate,
    *,
    workspace: str | None,
    reset_all: bool,
    default_workspace: str,
) -> tuple[RunView, ...]:
    workspace_ids = await _authorized_reset_scope(
        corpora,
        gate,
        workspace=workspace,
        reset_all=reset_all,
        default_workspace=default_workspace,
    )
    accepted = []
    for workspace_id in workspace_ids:
        creation = await mutations.create_reset(
            workspace=workspace_id,
            submitted_by=_SUBMITTED_BY,
            idempotency_key=f"reset:{workspace_id}:{uuid7()}",
        )
        accepted.append(creation.run)
    return tuple(accepted)


async def _wait_for_runs(runs: RunService, accepted: Sequence[RunView]) -> tuple[RunView, ...]:
    pending = {run.run_id: run for run in accepted}
    complete: dict[str, RunView] = {}
    while pending:
        for run_id, accepted_run in tuple(pending.items()):
            current = await runs.get(
                owner_id=accepted_run.access_scope_id,
                run_id=run_id,
            )
            if current is not None and current.status in _TERMINAL:
                complete[run_id] = current
                del pending[run_id]
        if pending:
            await asyncio.sleep(0.25)
    return tuple(complete[run.run_id] for run in accepted)


def _print_run(run: RunView) -> None:
    print(f"\n  [{run.access_scope_id}] {run.status} ({run.run_id})")
    if run.error_message:
        print(f"    {run.error_message}")
    result = run.result if isinstance(run.result, Mapping) else {}
    documents = result.get("documents") if isinstance(result, Mapping) else None
    if isinstance(documents, list) and documents:
        stats = documents[0]
        if isinstance(stats, Mapping):
            for key in (
                "documents_deleted",
                "chunks_deleted",
                "entities_deleted",
                "relationships_deleted",
                "local_files_removed",
                "orphan_tables_cleaned",
            ):
                if key in stats:
                    print(f"    {key.replace('_', ' ').title()}: {stats[key]}")


async def _run(*, workspace: str | None, reset_all: bool) -> tuple[RunView, ...]:
    from dlightrag import create_application
    from dlightrag.application.config import get_config

    config = get_config()
    application = await create_application(config)
    try:
        accepted = await _submit_reset_runs(
            application.corpora,
            application.corpus_mutations,
            AccessGate(AllowAllAccessControl(), None),
            workspace=workspace,
            reset_all=reset_all,
            default_workspace=config.deployment.workspace,
        )
        for run in accepted:
            print(f"Accepted Reset Run {run.run_id} for {run.access_scope_id}")
        return await _wait_for_runs(application.runs, accepted)
    finally:
        await application.aclose()


def main() -> int:
    args = build_parser().parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    if not args.yes:
        scope = (
            "every authorized Corpus Workspace"
            if args.reset_all
            else (args.workspace or "the default Corpus Workspace")
        )
        print(f"\nWARNING: This will permanently delete Corpus data in {scope}.")
        print("Type 'yes' to proceed: ", end="")
        try:
            if input().strip().lower() != "yes":
                print("Cancelled.")
                return 1
        except EOFError, KeyboardInterrupt:
            print("\nCancelled.")
            return 1

    completed = asyncio.run(_run(workspace=args.workspace, reset_all=args.reset_all))
    for run in completed:
        _print_run(run)
    failures = sum(run.status != "succeeded" for run in completed)
    print(f"\nDone. Failed Runs: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
