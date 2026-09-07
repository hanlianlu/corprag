# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable-run lifecycle policy shared by Runtime and its store adapter."""

#: Default floor for terminal run and event-log retention: at least 365 days
#: after finish before the maintenance sweep may reclaim them. Deployments
#: override through RuntimeConfig.run_retention_days.
DEFAULT_RUN_RETENTION_SECONDS = 365 * 24 * 3600
#: Top-level Retrieval is intentionally shorter-lived than Answer history.
RETRIEVAL_RUN_RETENTION_SECONDS = 7 * 24 * 3600
#: Corpus Mutation lifecycle and accepted inputs are retained for seven days.
CORPUS_MUTATION_RUN_RETENTION_SECONDS = 7 * 24 * 3600
#: A worker renews this window while it holds a run; expiry makes the row reclaimable.
RUN_LEASE_SECONDS = 60
#: Consecutive expired-lease reclaims allowed without durable progress before abandon.
MAX_RECLAIMS_WITHOUT_PROGRESS = 4
RUN_ABANDONED_ERROR_KIND = "run_abandoned"

__all__ = [
    "RUN_LEASE_SECONDS",
    "DEFAULT_RUN_RETENTION_SECONDS",
    "RETRIEVAL_RUN_RETENTION_SECONDS",
    "CORPUS_MUTATION_RUN_RETENTION_SECONDS",
    "MAX_RECLAIMS_WITHOUT_PROGRESS",
    "RUN_ABANDONED_ERROR_KIND",
]
