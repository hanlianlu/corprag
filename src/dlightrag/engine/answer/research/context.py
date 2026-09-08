# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Assemble one research request from the run's memory under one capacity."""

import asyncio
from typing import Any

from dlightrag.engine.agent.context import ContextContribution, ContextProjector
from dlightrag.engine.agent.session.fold import PriorTurns, WorkingContextProjection
from dlightrag.engine.ai.capacity import CONTEXT_POLICY, ContextPolicy, ModelProfile
from dlightrag.engine.ai.tokens import estimate_messages_tokens
from dlightrag.engine.answer.citations.indexer import CitationIndexer
from dlightrag.engine.answer.errors import AnswerInputOverflowError
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.memory import standing_memory_message
from dlightrag.engine.answer.prompts import agent_control_prompt, control_turn_instruction
from dlightrag.engine.answer.resources.models import ResourceManifestEntry
from dlightrag.engine.rag.corpus.sources.source_contract import safe_source_filename


class ContextAssembler:
    """Build each turn of one request from the stores, never by extending the last turn.

    Each control turn replays the active Agent Session projection and packs the
    ledger. The terminal in-loop assistant text is the Research answer; citation
    and source finalization remain deterministic outside model generation.
    """

    def __init__(
        self,
        *,
        model_profile: ModelProfile,
        context_policy: ContextPolicy = CONTEXT_POLICY,
        query: str,
        history: PriorTurns,
        query_images: list[dict[str, Any]] | None,
        resource_manifest: tuple[ResourceManifestEntry, ...],
        memory_text: str = "",
        contributions: tuple[ContextContribution, ...] = (),
        tool_guidance: tuple[str, ...] = (),
        profile_memory_write: bool = False,
        artifact_publication: bool = False,
    ) -> None:
        self._model_profile = model_profile
        self._context_policy = context_policy
        self._input_limit = context_policy.hard_input_limit(model_profile)
        self._control_target = context_policy.compaction_trigger(model_profile)
        self._history = history
        self._question = _question_message(query, query_images, resource_manifest)
        self._memory_text = memory_text
        self._contributions = contributions
        self._tool_guidance = tool_guidance
        self._profile_memory_write = profile_memory_write
        self._artifact_publication = artifact_publication
        self._control_instruction = control_turn_instruction(
            artifact_publication=artifact_publication
        )

    async def control_turn(
        self,
        *,
        evidence: EvidenceLedger,
        working: WorkingContextProjection,
        tool_schema_tokens: int,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._build_control_turn,
            evidence,
            working,
            tool_schema_tokens,
        )

    def measure_control_input(
        self,
        *,
        evidence: EvidenceLedger,
        working: WorkingContextProjection,
    ) -> int:
        """Measure the exact control-turn messages without enforcing the limit."""
        return estimate_messages_tokens(self._compose_control_turn(evidence, working))

    def observation_residual(
        self,
        transcript_with_assistant: list[dict[str, Any]],
        *,
        tool_schema_tokens: int,
    ) -> int:
        """Return capacity left for the next model-visible tool-result batch."""
        fixed = list(transcript_with_assistant)
        if len(fixed) >= 2 and _is_control_evidence_message(fixed[-2], self._control_instruction):
            fixed.pop(-2)
        assistant = fixed[-1] if fixed else {}
        tool_messages = [_empty_tool_message(call) for call in assistant.get("tool_calls") or ()]
        instruction = {
            "role": "user",
            "content": [{"type": "text", "text": self._control_instruction}],
        }
        used = estimate_messages_tokens([*fixed, *tool_messages, instruction]) + tool_schema_tokens
        return max(0, self._control_target - used)

    def output_allowance(
        self,
        messages: list[dict[str, Any]],
        *,
        additional_input_tokens: int = 0,
    ) -> int | None:
        """Preflight one exact model request and return its provider output cap."""
        if additional_input_tokens < 0:
            raise ValueError("additional_input_tokens cannot be negative")
        input_tokens = estimate_messages_tokens(messages) + additional_input_tokens
        self._check_input_tokens(input_tokens)
        return self._context_policy.output_allowance(
            self._model_profile,
            input_tokens=input_tokens,
        )

    def control_output_allowance(
        self,
        messages: list[dict[str, Any]],
        *,
        tool_schema_tokens: int,
    ) -> int:
        """Bound control output while preserving dynamic space for tool results."""
        provider_allowance = self.output_allowance(
            messages,
            additional_input_tokens=tool_schema_tokens,
        )
        accumulation_gap = self._input_limit - self._control_target
        control_allowance = accumulation_gap - tool_schema_tokens
        if control_allowance <= 0:
            raise AnswerInputOverflowError(
                "Research tool schemas leave no model residual for a control completion"
            )
        return (
            control_allowance
            if provider_allowance is None
            else min(provider_allowance, control_allowance)
        )

    def _build_control_turn(
        self,
        evidence: EvidenceLedger,
        working: WorkingContextProjection,
        tool_schema_tokens: int,
    ) -> list[dict[str, Any]]:
        # The orchestrator owns the proactive H trigger and compacts before
        # composing; this assembler enforces only the hard L limit later via
        # ``output_allowance`` / ``control_output_allowance``.
        return self._compose_control_turn(
            evidence,
            working,
            tool_schema_tokens=tool_schema_tokens,
        )

    def _compose_control_turn(
        self,
        evidence: EvidenceLedger,
        working: WorkingContextProjection,
        *,
        tool_schema_tokens: int = 0,
    ) -> list[dict[str, Any]]:
        system = {
            "role": "system",
            "content": agent_control_prompt(
                profile_memory_write=self._profile_memory_write,
                artifact_publication=self._artifact_publication,
            ),
        }
        head = self._head(system, working.messages())
        tail: list[ContextContribution] = []
        if self._tool_guidance:
            tail.append(
                ContextContribution(
                    source="answer.tools",
                    authority="reference",
                    messages=(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Tool usage guidance:\n"
                                    + "\n".join(self._tool_guidance),
                                }
                            ],
                        },
                    ),
                )
            )
        if evidence.row_count:
            blocks, _ = self._pack(
                evidence,
                head=head,
                tool_schema_tokens=tool_schema_tokens,
            )
            tail.append(
                ContextContribution(
                    source="answer.evidence",
                    authority="evidence",
                    messages=({"role": "user", "content": blocks},),
                    citable=True,
                )
            )
        memory_message = standing_memory_message(self._memory_text)
        if memory_message is not None:
            tail.append(
                ContextContribution(
                    source="profile.memory",
                    authority="profile",
                    messages=(memory_message,),
                )
            )
        tail.extend(self._contributions)
        return [*head, *ContextProjector().project(tail).messages]

    def _head(
        self,
        system: dict[str, Any],
        carried: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        contributions = [
            ContextContribution(
                source="answer.system",
                authority="system",
                messages=(system,),
                compressible=False,
            )
        ]
        if self._history.episodic_summary:
            contributions.append(
                ContextContribution(
                    source="conversation.episodic",
                    authority="conversation",
                    messages=(
                        {
                            "role": "user",
                            "content": self._history.episodic_summary,
                        },
                    ),
                )
            )
        if self._history.messages:
            contributions.append(
                ContextContribution(
                    source="conversation.tail",
                    authority="conversation",
                    messages=tuple(self._history.messages),
                )
            )
        contributions.extend(
            (
                ContextContribution(
                    source="answer.question",
                    authority="user",
                    messages=(self._question,),
                    compressible=False,
                ),
                ContextContribution(
                    source="agent.session",
                    authority="working",
                    messages=tuple(carried),
                ),
            )
        )
        return list(ContextProjector().project(contributions).messages)

    def _pack(
        self,
        evidence: EvidenceLedger,
        *,
        head: list[dict[str, Any]],
        tool_schema_tokens: int = 0,
    ) -> tuple[list[dict[str, Any]], CitationIndexer]:
        instruction_block = {"type": "text", "text": self._control_instruction}
        fixed_input_tokens = estimate_messages_tokens(
            [*head, {"role": "user", "content": [instruction_block]}]
        )
        target = self._control_target
        residual = max(0, target - tool_schema_tokens - fixed_input_tokens)
        while True:
            blocks, indexer = evidence.transform(residual_tokens=residual)
            content = [*blocks, instruction_block]
            rendered_tokens = (
                estimate_messages_tokens([*head, {"role": "user", "content": content}])
                + tool_schema_tokens
            )
            if rendered_tokens <= target:
                return content, indexer
            if residual == 0:
                raise AnswerInputOverflowError(
                    "Fixed research evidence handles exceed the resolved model input target"
                )
            residual = max(0, residual - (rendered_tokens - target))

    def _check_input_tokens(self, input_tokens: int) -> None:
        if input_tokens > self._input_limit:
            raise AnswerInputOverflowError(
                "Research input exceeds the resolved model input limit: "
                f"{input_tokens} > {self._input_limit} estimated input tokens"
            )


def _question_message(
    query: str,
    query_images: list[dict[str, Any]] | None,
    resource_manifest: tuple[ResourceManifestEntry, ...],
) -> dict[str, Any]:
    manifest = _resource_manifest_context(resource_manifest)
    if not query_images and not manifest:
        return {"role": "user", "content": query}
    content: list[dict[str, Any]] = [{"type": "text", "text": query}]
    if manifest:
        content.append({"type": "text", "text": manifest})
    content.extend(query_images or [])
    return {"role": "user", "content": content}


def _is_control_evidence_message(message: dict[str, Any], instruction: str) -> bool:
    content = message.get("content")
    return bool(
        message.get("role") == "user"
        and isinstance(content, list)
        and content
        and isinstance(content[-1], dict)
        and content[-1].get("text") == instruction
    )


def _empty_tool_message(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    return {
        "role": "tool",
        "tool_call_id": str(call.get("id") or ""),
        "name": str(function.get("name") or ""),
        "content": "",
    }


def _resource_manifest_context(manifest: tuple[ResourceManifestEntry, ...]) -> str:
    if not manifest:
        return ""
    lines = ["## Registered run-scoped Resources"]
    for entry in manifest:
        filename = safe_source_filename(entry.filename or "resource")
        kind = "image" if (entry.declared_mime or "").lower().startswith("image/") else "resource"
        lines.append(f"- [resource: {entry.resource_id}] {filename} ({kind})")
    lines.append("Use only these opaque resource ids with read or inspect.")
    return "\n".join(lines)


__all__ = ["ContextAssembler"]
