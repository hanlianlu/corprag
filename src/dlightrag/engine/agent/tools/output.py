# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Memory-bounded streaming output with optional durable full-result staging."""

from __future__ import annotations

import codecs
from dataclasses import dataclass
from typing import Protocol

from dlightrag.engine.agent.environment.local import ProcessChunk
from dlightrag.engine.agent.tools.contracts import CommittedOutput

_SINGLE_CHARACTER_LINE_BREAKS = frozenset("\v\f\x1c\x1d\x1e\x85\u2028\u2029")


class OutputStage(Protocol):
    """One uncommitted full-output staging object."""

    def append(self, data: bytes) -> None: ...

    async def commit(self) -> CommittedOutput: ...

    def discard(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ToolOutputSnapshot:
    """One bounded model view plus full-output continuation metadata."""

    text: str
    total_bytes: int
    total_lines: int
    truncated: bool
    receipt: CommittedOutput | None = None


class StreamingToolOutput:
    """Decode process chunks, stage the full stream, and retain a bounded tail."""

    def __init__(
        self,
        *,
        stage: OutputStage | None,
        max_bytes: int,
        max_lines: int,
    ) -> None:
        if max_bytes < 1 or max_lines < 1:
            raise ValueError("streaming output bounds must be positive")
        self._stage = stage
        self._max_bytes = max_bytes
        self._max_lines = max_lines
        self._decoders = {
            "stdout": codecs.getincrementaldecoder("utf-8")(errors="replace"),
            "stderr": codecs.getincrementaldecoder("utf-8")(errors="replace"),
        }
        self._tail = ""
        self._total_bytes = 0
        self._line_breaks = 0
        self._has_text = False
        self._ends_with_line_break = False
        self._last_character_was_cr = False
        self._finished = False

    def append(self, chunk: ProcessChunk) -> ToolOutputSnapshot:
        if self._finished:
            raise RuntimeError("streaming output is already finished")
        text = self._decoders[chunk.stream].decode(chunk.data, final=False)
        self._append_text(text)
        return self.snapshot()

    def snapshot(self) -> ToolOutputSnapshot:
        return ToolOutputSnapshot(
            text=self._tail,
            total_bytes=self._total_bytes,
            total_lines=self._total_lines,
            truncated=self._is_truncated,
        )

    def abort(self) -> None:
        """Discard staging synchronously; safe on cancellation paths."""
        if self._finished:
            return
        self._finished = True
        if self._stage is not None:
            self._stage.discard()

    async def finish(
        self,
        *,
        reserve_bytes: int = 0,
        reserve_lines: int = 0,
    ) -> ToolOutputSnapshot:
        """Finish while reserving final-result space for mandatory tool framing."""
        if self._finished:
            raise RuntimeError("streaming output is already finished")
        if not 0 <= reserve_bytes < self._max_bytes or not 0 <= reserve_lines < self._max_lines:
            raise ValueError("streaming output reserves must be smaller than their bounds")
        for decoder in self._decoders.values():
            self._append_text(decoder.decode(b"", final=True))
        self._finished = True
        final_max_bytes = self._max_bytes - reserve_bytes
        final_max_lines = self._max_lines - reserve_lines
        truncated = self._total_bytes > final_max_bytes or self._total_lines > final_max_lines
        tail = _bounded_complete_line_tail(
            self._tail,
            max_bytes=final_max_bytes,
            max_lines=final_max_lines,
        )
        receipt: CommittedOutput | None = None
        if truncated:
            if self._stage is not None:
                receipt = await self._stage.commit()
        elif self._stage is not None:
            self._stage.discard()
        return ToolOutputSnapshot(
            text=tail,
            total_bytes=self._total_bytes,
            total_lines=self._total_lines,
            truncated=truncated,
            receipt=receipt,
        )

    @property
    def _total_lines(self) -> int:
        if not self._has_text:
            return 0
        return self._line_breaks + (0 if self._ends_with_line_break else 1)

    @property
    def _is_truncated(self) -> bool:
        return self._total_bytes > self._max_bytes or self._total_lines > self._max_lines

    def _append_text(self, text: str) -> None:
        if not text:
            return
        data = text.encode("utf-8")
        if self._stage is not None:
            self._stage.append(data)
        self._total_bytes += len(data)
        for character in text:
            if character == "\n":
                if not self._last_character_was_cr:
                    self._line_breaks += 1
            elif character == "\r" or character in _SINGLE_CHARACTER_LINE_BREAKS:
                self._line_breaks += 1
            self._last_character_was_cr = character == "\r"
            self._ends_with_line_break = (
                character in _SINGLE_CHARACTER_LINE_BREAKS or character in {"\r", "\n"}
            )
        self._has_text = True
        self._tail = _bounded_complete_line_tail(
            self._tail + text,
            max_bytes=self._max_bytes,
            max_lines=self._max_lines,
        )


def _bounded_complete_line_tail(text: str, *, max_bytes: int, max_lines: int) -> str:
    lines = text.splitlines(keepends=True)
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    while lines and len("".join(lines).encode("utf-8")) > max_bytes:
        lines.pop(0)
    return "".join(lines)


__all__ = ["OutputStage", "StreamingToolOutput", "ToolOutputSnapshot"]
