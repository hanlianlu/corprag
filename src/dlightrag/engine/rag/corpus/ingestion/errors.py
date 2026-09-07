# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Corpus ingestion execution errors independent of lifecycle persistence."""


class RetryOutcomeUncertainError(RuntimeError):
    """A retry may have committed, but its authoritative status is unavailable."""


__all__ = ["RetryOutcomeUncertainError"]
