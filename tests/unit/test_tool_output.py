# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded streaming tool output with durable full-result staging."""

from dlightrag.engine.agent.environment.local import ProcessChunk
from dlightrag.engine.agent.tools.contracts import CommittedOutput
from dlightrag.engine.agent.tools.output import OutputStage, StreamingToolOutput


class _Stage(OutputStage):
    def __init__(self) -> None:
        self.data = bytearray()
        self.committed = False
        self.discarded = False

    def append(self, data: bytes) -> None:
        self.data.extend(data)

    async def commit(self) -> CommittedOutput:
        self.committed = True
        return CommittedOutput(
            resource_id="spill-1",
            content_digest="a" * 64,
            size_bytes=len(self.data),
        )

    def discard(self) -> None:
        self.discarded = True


async def test_streaming_output_spills_full_text_and_keeps_complete_tail_lines() -> None:
    stage = _Stage()
    output = StreamingToolOutput(stage=stage, max_bytes=13, max_lines=2)

    output.append(ProcessChunk("stdout", "α-first\n".encode()))
    output.append(ProcessChunk("stderr", b"second\nthird\n"))
    final = await output.finish()

    assert bytes(stage.data).decode() == "α-first\nsecond\nthird\n"
    assert stage.committed is True
    assert stage.discarded is False
    assert final.text == "second\nthird\n"
    assert final.truncated is True
    assert final.total_lines == 3
    assert final.receipt is not None
    assert final.receipt.resource_id == "spill-1"


async def test_non_lf_line_boundaries_use_the_same_accounting_as_tail_trimming() -> None:
    stage = _Stage()
    output = StreamingToolOutput(stage=stage, max_bytes=100, max_lines=3)
    output.append(ProcessChunk("stdout", b"one\rtwo\r"))
    output.append(ProcessChunk("stdout", b"\nthree\vfour\x1efive"))

    final = await output.finish()

    assert final.total_lines == 5
    assert final.text.splitlines() == ["three", "four", "five"]
    assert final.truncated is True
    assert final.receipt is not None
    assert stage.committed is True


async def test_finish_reserves_tail_space_and_commits_the_complete_stream() -> None:
    stage = _Stage()
    output = StreamingToolOutput(stage=stage, max_bytes=100, max_lines=5)
    output.append(ProcessChunk("stdout", b"one\ntwo\nthree\nfour\nfive\n"))

    final = await output.finish(reserve_bytes=20, reserve_lines=2)

    assert final.text == "three\nfour\nfive\n"
    assert final.truncated is True
    assert final.receipt is not None
    assert bytes(stage.data) == b"one\ntwo\nthree\nfour\nfive\n"


async def test_small_streaming_output_discards_staging_file() -> None:
    stage = _Stage()
    output = StreamingToolOutput(stage=stage, max_bytes=100, max_lines=10)

    output.append(ProcessChunk("stdout", b"complete"))
    final = await output.finish()

    assert final.text == "complete"
    assert final.truncated is False
    assert final.receipt is None
    assert stage.discarded is True
