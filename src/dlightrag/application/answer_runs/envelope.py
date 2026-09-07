# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer-owned terminal-surviving accepted input projection."""

from collections.abc import Mapping
from typing import Any


def accepted_input_envelope(prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the terminal-surviving public envelope from a prepared input.

    Continuations are ordinary newly authorized runs, but they must be able to
    reconstruct the selected run's accepted context after execution-private
    prepared input is cleared. Pinned model facts and resource manifests remain
    execution-only; normalized history, retrieval controls, and input-resource
    identities remain in this bounded public envelope.
    """
    envelope = {
        "query": str(prepared.get("query") or ""),
        "workspaces": [str(value) for value in prepared.get("workspaces") or ()],
        "history": [dict(item) for item in prepared.get("history") or ()],
        "episodic_summary": str(prepared.get("episodic_summary") or ""),
        "top_k": prepared.get("top_k"),
        "chunk_top_k": prepared.get("chunk_top_k"),
        "federated_rerank": bool(prepared.get("federated_rerank")),
        "filters": (
            dict(prepared["filters"]) if isinstance(prepared.get("filters"), Mapping) else None
        ),
        "semantic_highlights": bool(prepared.get("semantic_highlights")),
        "mode": str(prepared["mode"]) if prepared.get("mode") else None,
        "links": [dict(item) for item in prepared.get("links") or ()],
        "attachments": [dict(item) for item in prepared.get("attachments") or ()],
        "history_attachments": [dict(item) for item in prepared.get("history_attachments") or ()],
        "agent_session_id": str(prepared.get("agent_session_id") or ""),
        "agent_lane_id": str(prepared.get("agent_lane_id") or "main"),
        "source_lane_id": (
            str(prepared["source_lane_id"]) if prepared.get("source_lane_id") else None
        ),
    }
    if prepared.get("parent_run_id"):
        envelope["parent_run_id"] = str(prepared["parent_run_id"])
        envelope["continuation_kind"] = str(prepared.get("continuation_kind") or "")
    return envelope


__all__ = ["accepted_input_envelope"]
