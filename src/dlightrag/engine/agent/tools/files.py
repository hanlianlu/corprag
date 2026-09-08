# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Generic read/write/edit/grep/bash factories over an ExecutionEnvironment."""

from __future__ import annotations

import asyncio
import base64
import codecs
import difflib
import glob
import hashlib
import json
import os
import stat
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dlightrag.engine.agent.environment.access import (
    AccessScheduler,
    PathAccess,
    WorkspaceAccess,
)
from dlightrag.engine.agent.environment.child import build_child_environment
from dlightrag.engine.agent.environment.errors import (
    TOOL_RESULT_MAX_BYTES,
    TOOL_RESULT_MAX_LINES,
    TOOL_RESULT_PREVIEW_BYTES,
    FullOutputUnavailable,
    PathRejected,
    WorkspaceQuotaExceeded,
)
from dlightrag.engine.agent.environment.execution import ExecutionEnvironment
from dlightrag.engine.agent.environment.local import ProcessChunk
from dlightrag.engine.agent.environment.text import decode_workspace_text, encode_workspace_text
from dlightrag.engine.agent.environment.toolchain import SearchToolchain
from dlightrag.engine.agent.tool_content import ToolResourceAttachmentPart, ToolTextPart
from dlightrag.engine.agent.tools.contracts import (
    AgentTool,
    CommittedOutput,
    ResourceAttachmentBytes,
    ToolEffects,
    ToolResult,
    ToolRuntime,
    WorkspaceInventoryFacts,
    WorkspacePathFact,
)
from dlightrag.engine.agent.tools.listing import directory_page as _directory_page
from dlightrag.engine.agent.tools.listing import escape_path as _escape_path
from dlightrag.engine.agent.tools.output import OutputStage, StreamingToolOutput, ToolOutputSnapshot
from dlightrag.engine.ai.media import ImagePayloadBudget, decode_image_base64, detect_image_mime

type SpillWriter = Callable[[str], Awaitable[CommittedOutput]]
type OutputStageFactory = Callable[[str], OutputStage]

_PATH_MAX_CHARS = 4096
_CURSOR_MAX_CHARS = 8192
_VIOLATION_PATH_PREVIEW_BYTES = 256


@dataclass(frozen=True, slots=True)
class PreparedImageAttachment:
    """One provider-safe derivative of a verified source image."""

    data: bytes
    media_type: str
    transformed: bool


type ImagePreparer = Callable[[bytes, str], PreparedImageAttachment | None]


class HttpReadOptions(BaseModel):
    """Representation headers allowed on the first direct URL acquisition."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_agent: str | None = Field(default=None, min_length=1, max_length=256)
    accept: str | None = Field(default=None, min_length=1, max_length=512)
    accept_language: str | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def _single_line(self) -> HttpReadOptions:
        if any("\r" in value or "\n" in value for value in self.model_dump().values() if value):
            raise ValueError("HTTP representation headers must be single-line values")
        return self


class ReadWithoutUrlArgs(BaseModel):
    """Read contract for Hosts that do not provide public-URL admission."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    path: str | None = Field(
        default=None,
        max_length=_PATH_MAX_CHARS,
        description="Workspace-relative path to read.",
    )
    resource_id: str | None = Field(default=None, description="Opaque durable resource id.")
    offset: int | None = Field(default=None, ge=1, description="1-based line offset.")
    limit: int | None = Field(
        default=None,
        ge=1,
        le=100_000,
        description="Maximum lines to return.",
    )
    focus: str | None = Field(
        default=None,
        min_length=1,
        description="Optional relevance focus for a durable resource.",
    )
    cursor: str | None = Field(
        default=None,
        max_length=_CURSOR_MAX_CHARS,
        description="Continuation cursor.",
    )

    @model_validator(mode="after")
    def _exactly_one_target(self) -> ReadWithoutUrlArgs:
        if (self.path is None) == (self.resource_id is None):
            raise ValueError("read requires exactly one of path or resource_id")
        if self.path is not None and self.focus is not None:
            raise ValueError("read focus is available only for resources")
        if self.path is not None and self.cursor is not None and self.offset is not None:
            raise ValueError("read path cursor and offset are mutually exclusive")
        if self.path is None and (self.offset is not None or self.limit is not None):
            raise ValueError("read offset and limit are available only for paths")
        return self


class ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    path: str | None = Field(
        default=None,
        max_length=_PATH_MAX_CHARS,
        description="Workspace-relative path to read.",
    )
    resource_id: str | None = Field(default=None, description="Opaque durable resource id.")
    url: str | None = Field(default=None, description="Anonymous public HTTP(S) URL to read.")
    http: HttpReadOptions | None = Field(
        default=None,
        description="Optional representation headers for the first direct URL acquisition.",
    )
    offset: int | None = Field(default=None, ge=1, description="1-based line offset.")
    limit: int | None = Field(
        default=None,
        ge=1,
        le=100_000,
        description="Maximum lines to return.",
    )
    focus: str | None = Field(
        default=None,
        min_length=1,
        description="Optional relevance focus for a durable resource.",
    )
    cursor: str | None = Field(
        default=None,
        max_length=_CURSOR_MAX_CHARS,
        description="Continuation cursor.",
    )

    @model_validator(mode="after")
    def _exactly_one_target(self) -> ReadArgs:
        targets = sum(value is not None for value in (self.path, self.resource_id, self.url))
        if targets != 1:
            raise ValueError("read requires exactly one of path, resource_id, or url")
        if self.path is not None and self.focus is not None:
            raise ValueError("read focus is available only for resources")
        if self.path is not None and self.cursor is not None and self.offset is not None:
            raise ValueError("read path cursor and offset are mutually exclusive")
        if self.path is None and (self.offset is not None or self.limit is not None):
            raise ValueError("read offset and limit are available only for paths")
        if self.url is not None and self.cursor is not None:
            raise ValueError("continue a URL read with its returned resource_id")
        if self.url is None and self.http is not None:
            raise ValueError("read http options are available only for url")
        return self


@dataclass(frozen=True, slots=True)
class ResourceReadRequest:
    resource_id: str | None
    url: str | None
    focus: str | None
    cursor: str | None
    user_agent: str | None = None
    accept: str | None = None
    accept_language: str | None = None


type ResourceReader = Callable[[ResourceReadRequest, ToolRuntime], Awaitable[ToolResult]]


class WriteArgs(BaseModel):
    path: str = Field(max_length=_PATH_MAX_CHARS, description="Workspace-relative path to write.")
    content: str = Field(description="Full UTF-8 file contents.")


class EditOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    old_text: str = Field(min_length=1, description="Unique exact text in the original file.")
    new_text: str = Field(description="Replacement text.")


class EditArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    path: str = Field(max_length=_PATH_MAX_CHARS, description="Workspace-relative path to edit.")
    edits: list[EditOperation] = Field(
        min_length=1,
        description="Non-overlapping replacements, all matched against the original file.",
    )


class GrepArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    pattern: str = Field(
        min_length=1,
        max_length=65_536,
        description="Regex (or literal with literal=true).",
    )
    path: str = Field(
        default=".",
        max_length=_PATH_MAX_CHARS,
        description="Workspace path to search.",
    )
    glob: str | None = Field(default=None, description="Optional glob filter.")
    ignore_case: bool = Field(default=False, description="Case-insensitive matching.")
    literal: bool = Field(
        default=False, description="Treat pattern as a literal string, not a regex."
    )
    context: int | None = Field(
        default=None,
        ge=0,
        le=1000,
        description="Context lines shown around each match.",
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=100_000,
        description="Maximum matching lines to return.",
    )


class FindArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    pattern: str = Field(min_length=1, max_length=4096, description="Glob pattern to match.")
    path: str = Field(
        default=".",
        max_length=_PATH_MAX_CHARS,
        description="Workspace subtree to search.",
    )
    limit: int = Field(
        default=1000,
        ge=1,
        le=100_000,
        description="Maximum matches to return.",
    )


class LsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    path: str = Field(
        default=".",
        max_length=_PATH_MAX_CHARS,
        description="Workspace directory to list.",
    )
    limit: int = Field(default=500, ge=1, le=10_000, description="Maximum entries to return.")
    cursor: str | None = Field(
        default=None,
        max_length=_CURSOR_MAX_CHARS,
        description="Continuation cursor from a prior page.",
    )


class BashArgs(BaseModel):
    command: str = Field(description="Bash command to run.")
    timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        le=2_147_483.647,
        allow_inf_nan=False,
        description="Optional finite process timeout in seconds.",
    )


def bound_tool_text(text: str, *, spill: SpillWriter | None) -> str:
    """Apply the unified byte/line guard. Spill if available, else raise."""
    if _within_result_bounds(text):
        return text
    if spill is None:
        raise FullOutputUnavailable("oversized tool result has no spill or cursor backing")
    raise FullOutputUnavailable("spill writer must be awaited by the tool, not bound_tool_text")


async def preview_or_spill(
    text: str,
    *,
    spill: SpillWriter | None,
    tool: str,
    preview: Literal["head", "tail"] = "head",
) -> tuple[str, CommittedOutput | None]:
    """Return (model text, optional committed-spill receipt)."""
    if _within_result_bounds(text):
        return text, None
    if spill is None:
        raise FullOutputUnavailable("oversized tool result has no spill or cursor backing")
    receipt = await spill(text)
    resource_id = receipt.resource_id
    excerpt = _utf8_excerpt(text, preview=preview)
    rendered = (
        f"{tool} output exceeded {TOOL_RESULT_MAX_BYTES} UTF-8 bytes or "
        f"{TOOL_RESULT_MAX_LINES} lines ({len(text.encode('utf-8'))} bytes). "
        f"Full output: read(resource_id={resource_id!r}, cursor=...)\n{excerpt}"
    )
    return rendered, receipt


def path_tools(
    environment: ExecutionEnvironment,
    *,
    scheduler: AccessScheduler,
    fd: str = "fd",
    ripgrep: str = "rg",
    search_toolchain: SearchToolchain | None = None,
    image_preparer: ImagePreparer | None = None,
    resource_reader: ResourceReader | None = None,
    spill: SpillWriter | None = None,
    output_stage_factory: OutputStageFactory | None = None,
) -> list[AgentTool]:
    """Return Pi-shaped path tools bound to one rooted environment."""
    toolchain = search_toolchain or SearchToolchain(fd=fd, ripgrep=ripgrep)
    return [
        read_tool(
            environment,
            scheduler,
            resource_reader=resource_reader,
            spill=spill,
            image_preparer=image_preparer,
        ),
        bash_tool(environment, scheduler, output_stage_factory=output_stage_factory),
        edit_tool(environment, scheduler, spill=spill),
        write_tool(environment, scheduler),
        grep_tool(
            environment,
            scheduler,
            search_toolchain=toolchain,
            output_stage_factory=output_stage_factory,
        ),
        find_tool(environment, scheduler, search_toolchain=toolchain, spill=spill),
        ls_tool(environment, scheduler),
    ]


def read_tool(
    environment: ExecutionEnvironment | None,
    scheduler: AccessScheduler,
    *,
    resource_reader: ResourceReader | None = None,
    spill: SpillWriter | None = None,
    image_preparer: ImagePreparer | None = None,
) -> AgentTool:
    """Build ``read`` with whichever branches the host actually has."""

    async def execute(args: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = cast(ReadArgs | ReadWithoutUrlArgs, args)
        url = args.url if isinstance(args, ReadArgs) else None
        if args.resource_id is not None or url is not None:
            if resource_reader is None:
                return ToolResult.text("resource read is not available", is_error=True)
            target = args.resource_id or url or "resource"
            options = (
                args.http or HttpReadOptions() if isinstance(args, ReadArgs) else HttpReadOptions()
            )
            async with scheduler.hold(PathAccess(path=target, kind="read")):
                return await resource_reader(
                    ResourceReadRequest(
                        resource_id=args.resource_id,
                        url=url,
                        focus=args.focus,
                        cursor=args.cursor,
                        user_agent=options.user_agent,
                        accept=options.accept,
                        accept_language=options.accept_language,
                    ),
                    runtime,
                )
        if environment is None or args.path is None:
            return ToolResult.text("path read requires an execution environment", is_error=True)
        if blocked := _integrity_blocked(environment):
            return blocked
        try:
            path = environment.resolve(args.path)
        except PathRejected as exc:
            return ToolResult.text(str(exc), is_error=True)
        canonical_path = _workspace_relative_path(environment.root, path)
        async with scheduler.hold(PathAccess(path=str(path), kind="read")):
            if blocked := _integrity_blocked(environment):
                return blocked
            kind = environment.stat_kind(path)
            if kind == "directory":
                if args.offset is not None:
                    return ToolResult.text(
                        "read directory pages use cursor, not offset",
                        is_error=True,
                    )
                return _directory_page(
                    environment.list_directory(path),
                    path=canonical_path,
                    cursor=args.cursor,
                    limit=args.limit or 500,
                    tool="read",
                )
            if kind == "missing":
                return ToolResult.text(f"file not found: {_escape_path(args.path)}", is_error=True)
            if args.cursor is not None:
                return ToolResult.text("read path cursor requires a directory", is_error=True)
            raw = environment.read_bytes(path)
            media_type = _sniff_image_media_type(raw)
            if media_type is not None:
                prepare = image_preparer or _default_image_preparer
                prepared = prepare(raw, canonical_path)
                if prepared is None:
                    return ToolResult.text(
                        f"image cannot fit the model payload budget: {_escape_path(args.path)}",
                        is_error=True,
                    )
                return _image_attachment_result(
                    raw,
                    source_media_type=media_type,
                    prepared=prepared,
                    path=canonical_path,
                )
            try:
                decoded = decode_workspace_text(raw)
            except ValueError as exc:
                return ToolResult.text(str(exc), is_error=True)
            body, continuation, _remaining = _paginate_lines(
                decoded.text,
                path=canonical_path,
                offset=args.offset,
                limit=args.limit,
                notice=(
                    "[mixed line endings preserved; not normalized]"
                    if decoded.mixed_newlines
                    else ""
                ),
            )
            body, committed = await preview_or_spill(body, spill=spill, tool="read")
            return ToolResult.text(
                body,
                protected_text=continuation,
                effects=ToolEffects(
                    committed_outputs=((committed,) if committed is not None else ())
                ),
            )

    url_enabled = resource_reader is not None
    description = (
        "Read exactly one target: a workspace path, a durable resource_id, or an "
        "anonymous public HTTP(S) url. URL reads accept only optional http.user_agent, "
        "http.accept, and http.accept_language representation preferences; continue "
        "with the returned resource_id and cursor."
        if url_enabled
        else "Read one workspace path or Host-provided durable resource_id."
    )
    guidance = (
        "read: one of path, resource_id, or url; file pages carry an offset while "
        "directory/resource pages carry opaque cursors. Follow the printed continuation "
        "instead of re-reading the whole target."
        if url_enabled
        else (
            "read: one of path or resource_id; files page by offset and directories/resources "
            "by opaque cursor. Follow the printed continuation."
        )
    )
    return AgentTool(
        name="read",
        description=description,
        input_model=ReadArgs if url_enabled else ReadWithoutUrlArgs,
        execute=execute,
        replay_policy="replayable",
        contract_version=4 if url_enabled else 3,
        guidance=guidance,
    )


def write_tool(environment: ExecutionEnvironment, scheduler: AccessScheduler) -> AgentTool:
    async def execute(args: BaseModel, _runtime: ToolRuntime) -> ToolResult:
        args = cast(WriteArgs, args)
        if blocked := _integrity_blocked(environment):
            return blocked
        try:
            path = environment.resolve(args.path)
        except PathRejected as exc:
            return ToolResult.text(str(exc), is_error=True)
        async with scheduler.hold(PathAccess(path=str(path), kind="write")):
            if blocked := _integrity_blocked(environment):
                return blocked
            try:
                environment.write_bytes(path, args.content.encode("utf-8"))
            except WorkspaceQuotaExceeded as exc:
                return ToolResult.text(str(exc), is_error=True)
            except PathRejected as exc:
                return ToolResult.text(str(exc), is_error=True)
            inventory = _inventory_facts(environment.root, path)
        return ToolResult.text(
            f"wrote {args.path} ({len(args.content.encode('utf-8'))} bytes)",
            effects=ToolEffects(workspace_inventory=inventory),
        )

    return AgentTool(
        name="write",
        description="Create or overwrite a UTF-8 workspace file.",
        input_model=WriteArgs,
        execute=execute,
        replay_policy="never",
        contract_version=3,
        guidance="write: replaces the whole file; the success line reports UTF-8 byte size.",
    )


def edit_tool(
    environment: ExecutionEnvironment,
    scheduler: AccessScheduler,
    *,
    spill: SpillWriter | None = None,
) -> AgentTool:
    async def execute(args: BaseModel, _runtime: ToolRuntime) -> ToolResult:
        edit_args = cast(EditArgs, args)
        if blocked := _integrity_blocked(environment):
            return blocked
        try:
            path = environment.resolve(edit_args.path)
        except PathRejected as exc:
            return ToolResult.text(str(exc), is_error=True)
        async with scheduler.hold(PathAccess(path=str(path), kind="readwrite")):
            if blocked := _integrity_blocked(environment):
                return blocked
            if environment.stat_kind(path) != "file":
                return ToolResult.text(
                    f"file not found: {_escape_path(edit_args.path)}",
                    is_error=True,
                )
            try:
                decoded = decode_workspace_text(environment.read_bytes(path))
            except ValueError as exc:
                return ToolResult.text(str(exc), is_error=True)
            spans: list[tuple[int, int, str]] = []
            for index, operation in enumerate(edit_args.edits, start=1):
                if operation.old_text == operation.new_text:
                    return ToolResult.text(
                        f"edit {index} rejected: old_text and new_text are identical",
                        is_error=True,
                    )
                count = decoded.text.count(operation.old_text)
                if count != 1:
                    return ToolResult.text(
                        f"edit {index} old_text matches {count} times; each match must be unique",
                        is_error=True,
                    )
                start = decoded.text.index(operation.old_text)
                spans.append((start, start + len(operation.old_text), operation.new_text))
            ordered = sorted(spans)
            if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:], strict=False)):
                return ToolResult.text("edit ranges overlap in the original file", is_error=True)
            updated = decoded.text
            for start, end, replacement in reversed(ordered):
                updated = updated[:start] + replacement + updated[end:]
            try:
                environment.write_bytes(path, encode_workspace_text(decoded, updated))
            except (WorkspaceQuotaExceeded, PathRejected) as exc:
                return ToolResult.text(str(exc), is_error=True)
            inventory = _inventory_facts(environment.root, path)
        patch = "\n".join(
            difflib.unified_diff(
                decoded.text.splitlines(),
                updated.splitlines(),
                fromfile=edit_args.path,
                tofile=edit_args.path,
                lineterm="",
            )
        )
        first_line = decoded.text.count("\n", 0, ordered[0][0]) + 1
        body = (
            f"edited {_escape_path(edit_args.path)} ({len(ordered)} edits; "
            f"first change line {first_line})\n{patch}"
        )
        body, committed = await preview_or_spill(body, spill=spill, tool="edit")
        return ToolResult.text(
            body,
            effects=ToolEffects(
                committed_outputs=((committed,) if committed is not None else ()),
                workspace_inventory=inventory,
            ),
        )

    return AgentTool(
        name="edit",
        description="Replace exact text in a workspace file.",
        input_model=EditArgs,
        execute=execute,
        replay_policy="never",
        contract_version=3,
        guidance=(
            "edit: every old_text must match exactly once in the current file; all edits "
            "apply atomically or none do. Read the file first when a match fails."
        ),
    )


def grep_tool(
    environment: ExecutionEnvironment,
    scheduler: AccessScheduler,
    *,
    ripgrep: str = "rg",
    search_toolchain: SearchToolchain | None = None,
    output_stage_factory: OutputStageFactory | None = None,
) -> AgentTool:
    toolchain = search_toolchain or SearchToolchain(ripgrep=ripgrep)

    async def execute(args: BaseModel, runtime: ToolRuntime) -> ToolResult:
        grep_args = cast(GrepArgs, args)
        if blocked := _integrity_blocked(environment):
            return blocked
        try:
            root = (
                environment.root if grep_args.path == "." else environment.resolve(grep_args.path)
            )
        except PathRejected as exc:
            return ToolResult.text(str(exc), is_error=True)
        target = root.relative_to(environment.root).as_posix() if root != environment.root else "."
        try:
            ripgrep = await toolchain.path("rg")
        except RuntimeError as exc:
            return ToolResult.text(str(exc), is_error=True)
        argv = [
            ripgrep,
            "--json",
            "--no-config",
            "--hidden",
            "--no-require-git",
            "--glob",
            "!.git",
            "--max-count",
            str(grep_args.limit + 1),
        ]
        if grep_args.ignore_case:
            argv.append("--ignore-case")
        if grep_args.literal:
            argv.append("--fixed-strings")
        if grep_args.context is not None:
            argv.extend(["--context", str(grep_args.context)])
        if grep_args.glob:
            argv.extend(["--glob", grep_args.glob])
        argv.extend(["-e", grep_args.pattern, "--", target])
        output = _streaming_output("grep", output_stage_factory)
        collector = _GrepJsonCollector(
            output=output,
            workspace_root=environment.root,
            limit=grep_args.limit,
        )

        async def capture(chunk: ProcessChunk) -> None:
            collector.feed(chunk)

        try:
            async with scheduler.hold(PathAccess(path=str(root), kind="search")):
                if blocked := _integrity_blocked(environment):
                    output.abort()
                    return blocked
                home, tmp = environment.prepare_process_directories()
                completed = await environment.run(
                    argv,
                    env=build_child_environment(home=home, tmp=tmp),
                    cwd=environment.root,
                    on_output=capture,
                )
            collector.finish()
            if completed.returncode == 1 and collector.matches == 0:
                output.append(ProcessChunk("stdout", b"(no matches)"))
            if collector.truncated:
                output.append(
                    ProcessChunk(
                        "stdout",
                        f"\n[limited to {grep_args.limit} matching lines]".encode(),
                    )
                )
            if collector.parse_errors:
                output.append(
                    ProcessChunk(
                        "stderr",
                        f"\n[ripgrep JSON protocol errors: {collector.parse_errors}]".encode(),
                    )
                )
            final = await output.finish()
        except (OSError, PathRejected, WorkspaceQuotaExceeded) as exc:
            output.abort()
            return ToolResult.text(str(exc), is_error=True)
        except asyncio.CancelledError:
            output.abort()
            raise
        except BaseException:
            output.abort()
            raise
        result = _stream_result("grep", final)
        return ToolResult.text(
            result.text_content,
            details={
                **(result.details or {}),
                "matches": collector.matches,
                "parse_errors": collector.parse_errors,
            },
            protected_text=result.protected_text,
            is_error=completed.returncode not in {0, 1} or collector.parse_errors > 0,
            effects=result.effects,
        )

    return AgentTool(
        name="grep",
        description="Search workspace files with ripgrep.",
        input_model=GrepArgs,
        execute=execute,
        replay_policy="replayable",
        contract_version=3,
        guidance=(
            "grep: regex by default (literal=true for plain text); limit caps matching "
            "lines, not context lines; hidden files are searched while ignore rules apply."
        ),
    )


def bash_tool(
    environment: ExecutionEnvironment,
    scheduler: AccessScheduler,
    *,
    output_stage_factory: OutputStageFactory | None = None,
) -> AgentTool:
    async def execute(args: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = cast(BashArgs, args)
        if blocked := _integrity_blocked(environment):
            return blocked
        output = _streaming_output("bash", output_stage_factory)
        last_update = 0.0

        async def capture(chunk: ProcessChunk) -> None:
            nonlocal last_update
            snapshot = output.append(chunk)
            now = time.monotonic()
            if now - last_update >= 0.1:
                last_update = now
                await runtime.emit_update(_stream_result("bash", snapshot, transient=True))

        try:
            async with scheduler.hold(WorkspaceAccess()):
                if blocked := _integrity_blocked(environment):
                    output.abort()
                    return blocked
                try:
                    home, tmp = environment.prepare_process_directories()
                except (OSError, PathRejected, WorkspaceQuotaExceeded) as exc:
                    output.abort()
                    return ToolResult.text(str(exc), is_error=True)
                try:
                    completed = await environment.run(
                        ["/bin/bash", "--noprofile", "--norc", "-c", args.command],
                        env=build_child_environment(home=home, tmp=tmp),
                        cwd=environment.root,
                        timeout_seconds=args.timeout_seconds,
                        on_output=capture,
                    )
                finally:
                    violations = environment.refresh_integrity()
                quota_violation = environment.quota_violation
                inventory = (
                    None if violations or quota_violation else _scan_inventory(environment.root)
                )
            status = "timeout" if completed.timed_out else f"exit {completed.returncode}"
            output.append(ProcessChunk("stdout", f"\n{status}".encode()))
            notices: list[str] = []
            if violations:
                notices.append(
                    "bash left unsafe or unreadable workspace entries: "
                    f"{_render_violations(violations)}; the workspace is latched until "
                    "external cleanup"
                )
            if quota_violation is not None:
                notices.append(
                    f"{quota_violation}; the workspace is latched until external cleanup"
                )
            reserve_bytes, reserve_lines = _process_notice_reserve(notices)
            final = await output.finish(
                reserve_bytes=reserve_bytes,
                reserve_lines=reserve_lines,
            )
        except asyncio.CancelledError:
            output.abort()
            raise
        except BaseException:
            output.abort()
            raise
        streamed = _stream_result("bash", final, notices=notices)
        failed = (
            completed.timed_out
            or completed.returncode != 0
            or bool(violations)
            or quota_violation is not None
        )
        return ToolResult.text(
            streamed.text_content,
            details={
                **(streamed.details or {}),
                "integrity_violations": list(violations),
                "quota_violation": quota_violation,
            },
            protected_text=streamed.protected_text,
            is_error=failed,
            effects=ToolEffects(
                committed_outputs=streamed.effects.committed_outputs,
                workspace_inventory=inventory,
            ),
        )

    return AgentTool(
        name="bash",
        description="Run a bash command in the workspace.",
        input_model=BashArgs,
        execute=execute,
        replay_policy="never",
        contract_version=3,
        guidance=(
            "bash: output streams live and stays bounded; timed-out or failing commands "
            "still return partial output as errors. Never leave symlinks, FIFOs, sockets, "
            "device files, or quota overflow behind: the workspace latches until external cleanup."
        ),
    )


def _sniff_image_media_type(data: bytes) -> str | None:
    """Return the original media type for a verified image snapshot, else None."""
    if len(data) > 104_857_600:
        return None
    signatures = (
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"\xff\xd8\xff", "image/jpeg"),
        (b"GIF87a", "image/gif"),
        (b"GIF89a", "image/gif"),
        (b"RIFF", "image/webp"),
    )
    media_type = next((mime for magic, mime in signatures if data.startswith(magic)), None)
    if media_type is None:
        return None
    if media_type == "image/webp" and data[8:12] != b"WEBP":
        return None
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        return media_type
    except Exception:
        return None


def _default_image_preparer(data: bytes, path: str) -> PreparedImageAttachment | None:
    budget = ImagePayloadBudget(
        max_total_bytes=3 * 1024 * 1024,
        max_bytes_per_image=3 * 1024 * 1024,
        max_pixels=40_000_000,
        max_px=1536,
        min_px=256,
        quality=89,
        min_quality=65,
        max_images=1,
    )
    encoded = base64.b64encode(data).decode("ascii")
    bounded = budget.add_base64(encoded, label=path)
    if bounded is None:
        return None
    uri, _byte_count = bounded
    prepared, declared_media_type = decode_image_base64(uri)
    media_type = detect_image_mime(prepared, fallback=declared_media_type)
    return PreparedImageAttachment(
        data=prepared,
        media_type=media_type,
        transformed=prepared != data,
    )


def _image_attachment_result(
    data: bytes,
    *,
    source_media_type: str,
    prepared: PreparedImageAttachment,
    path: str,
) -> ToolResult:
    """Persist the source snapshot but expose only its provider-bounded derivative."""
    source_digest = hashlib.sha256(data).hexdigest()
    prepared_digest = hashlib.sha256(prepared.data).hexdigest()
    source_resource_id = f"att_{source_digest[:32]}"
    model_resource_id = (
        f"att_model_{prepared_digest[:32]}" if prepared.transformed else source_resource_id
    )
    safe_name = path.rsplit("/", 1)[-1] or "image"
    attachment = ToolResourceAttachmentPart(
        resource_id=model_resource_id,
        safe_name=safe_name,
        media_type=prepared.media_type,
        content_digest=prepared_digest,
        size_bytes=len(prepared.data),
        data=prepared.data,
    )
    durable_resources = [
        ResourceAttachmentBytes(
            resource_id=source_resource_id,
            filename=safe_name,
            mime_type=source_media_type,
            source_locator=path,
            content=data,
        )
    ]
    if prepared.transformed:
        durable_resources.append(
            ResourceAttachmentBytes(
                resource_id=model_resource_id,
                filename=f"model-{safe_name}",
                mime_type=prepared.media_type,
                source_locator=f"{path}#model-derivative",
                content=prepared.data,
            )
        )
    transformation = "resized/re-encoded derivative" if prepared.transformed else "bounded original"
    return ToolResult(
        parts=(
            ToolTextPart(
                f"image attachment: {_escape_path(path)} ({prepared.media_type}, "
                f"{len(prepared.data)} model bytes, source_resource_id={source_resource_id!r}); "
                f"the model receives a {transformation} and durable evidence retains the source"
            ),
            attachment,
        ),
        effects=ToolEffects(attached_resources=tuple(durable_resources)),
    )


def _render_violations(violations: tuple[str, ...]) -> str:
    shown = [
        _utf8_prefix(_escape_path(path), max_bytes=_VIOLATION_PATH_PREVIEW_BYTES)
        for path in violations[:20]
    ]
    if len(violations) > len(shown):
        shown.append(f"… ({len(violations) - len(shown)} more)")
    return ", ".join(shown)


def _integrity_blocked(environment: ExecutionEnvironment) -> ToolResult | None:
    violations = environment.integrity_violations
    quota_violation = environment.quota_violation
    if not violations and quota_violation is None:
        return None
    reasons: list[str] = []
    if violations:
        reasons.append(f"unsafe or unreadable workspace entries: {_render_violations(violations)}")
    if quota_violation is not None:
        reasons.append(quota_violation)
    return ToolResult.text(
        f"workspace integrity latched: {'; '.join(reasons)}; external cleanup is required",
        is_error=True,
    )


def find_tool(
    environment: ExecutionEnvironment,
    scheduler: AccessScheduler,
    *,
    fd: str = "fd",
    search_toolchain: SearchToolchain | None = None,
    spill: SpillWriter | None = None,
) -> AgentTool:
    toolchain = search_toolchain or SearchToolchain(fd=fd)

    async def execute(args: BaseModel, _runtime: ToolRuntime) -> ToolResult:
        find_args = cast(FindArgs, args)
        if blocked := _integrity_blocked(environment):
            return blocked
        try:
            root = (
                environment.root if find_args.path == "." else environment.resolve(find_args.path)
            )
            if environment.stat_kind(root) != "directory":
                return ToolResult.text(
                    f"find path is not a directory: {_escape_path(find_args.path)}",
                    is_error=True,
                )
            fd = await toolchain.path("fd")
        except (PathRejected, OSError, RuntimeError) as exc:
            return ToolResult.text(str(exc), is_error=True)

        argv = [
            fd,
            "--color",
            "never",
            "--hidden",
            "--no-require-git",
            "--exclude",
            ".git",
            "--print0",
        ]
        pattern = find_args.pattern
        if "/" in pattern:
            argv.append("--full-path")
            pattern = f"{glob.escape(str(root))}/{pattern.removeprefix('./')}"
        argv.extend(["--glob", "--", pattern, str(root)])
        collector = _NulPathCollector(root=root)
        try:
            async with scheduler.hold(PathAccess(path=str(root), kind="search")):
                if blocked := _integrity_blocked(environment):
                    return blocked
                home, tmp = environment.prepare_process_directories()
                completed = await environment.run(
                    argv,
                    env=build_child_environment(home=home, tmp=tmp),
                    cwd=root,
                    on_output=collector.feed,
                )
            collector.finish()
        except (OSError, ValueError) as exc:
            return ToolResult.text(str(exc), is_error=True)
        if completed.returncode != 0:
            return ToolResult.text(
                collector.stderr or f"fd exited with status {completed.returncode}",
                is_error=True,
            )

        ordered = sorted(collector.paths, key=lambda value: (value.casefold(), value))
        truncated = len(ordered) > find_args.limit
        shown = ordered[: find_args.limit]
        body = "\n".join(_escape_path(value) for value in shown) or "(no matches)"
        if truncated:
            body += f"\n[limited to {find_args.limit} matches; more matches exist]"
        body, committed = await preview_or_spill(body, spill=spill, tool="find")
        return ToolResult.text(
            body,
            effects=ToolEffects(committed_outputs=((committed,) if committed is not None else ())),
        )

    return AgentTool(
        name="find",
        description="Find workspace paths recursively with fd glob semantics.",
        input_model=FindArgs,
        execute=execute,
        replay_policy="replayable",
        contract_version=2,
        guidance=(
            "find: fd --glob semantics relative to the requested search root; hidden paths "
            "are included, .git and active ignore rules are respected, and symlinks are not followed."
        ),
    )


def ls_tool(environment: ExecutionEnvironment, scheduler: AccessScheduler) -> AgentTool:
    async def execute(args: BaseModel, _runtime: ToolRuntime) -> ToolResult:
        ls_args = cast(LsArgs, args)
        if blocked := _integrity_blocked(environment):
            return blocked
        try:
            root = environment.root if ls_args.path == "." else environment.resolve(ls_args.path)
            if environment.stat_kind(root) != "directory":
                return ToolResult.text(
                    f"ls path is not a directory: {_escape_path(ls_args.path)}",
                    is_error=True,
                )
            async with scheduler.hold(PathAccess(path=str(root), kind="read")):
                if blocked := _integrity_blocked(environment):
                    return blocked
                entries = environment.list_directory(root)
        except (PathRejected, OSError) as exc:
            return ToolResult.text(str(exc), is_error=True)
        return _directory_page(
            entries,
            path=_workspace_relative_path(environment.root, root),
            cursor=ls_args.cursor,
            limit=ls_args.limit,
            tool="ls",
        )

    return AgentTool(
        name="ls",
        description="List one workspace directory without following symlinks.",
        input_model=LsArgs,
        execute=execute,
        replay_policy="replayable",
        contract_version=2,
        guidance=(
            "ls: one sorted directory level, kind/size/name per entry; continue large "
            "listings with the opaque cursor. Symlinks are listed, never followed."
        ),
    )


class _GrepJsonCollector:
    """Incrementally normalize ripgrep's stable JSON-lines event protocol."""

    _MAX_EVENT_CHARS = 4 * 1024 * 1024

    def __init__(
        self,
        *,
        output: StreamingToolOutput,
        workspace_root: Path,
        limit: int,
        max_line_chars: int = 2000,
    ) -> None:
        self._output = output
        self._workspace_root = workspace_root
        self._limit = limit
        self._max_line_chars = max_line_chars
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._buffer = ""
        self._dropping_event = False
        self.matches = 0
        self.parse_errors = 0
        self.truncated = False

    def feed(self, chunk: ProcessChunk) -> None:
        if chunk.stream == "stderr":
            self._output.append(chunk)
            return
        try:
            decoded = self._decoder.decode(chunk.data, final=False)
        except UnicodeDecodeError:
            self.parse_errors += 1
            self._decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
            self._buffer = ""
            self._dropping_event = True
            return
        if self._dropping_event:
            _discarded, separator, decoded = decoded.partition("\n")
            if not separator:
                return
            self._dropping_event = False
        self._buffer += decoded
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if len(line) > self._MAX_EVENT_CHARS:
                self.parse_errors += 1
            else:
                self._consume(line)
        if len(self._buffer) > self._MAX_EVENT_CHARS:
            self.parse_errors += 1
            self._buffer = ""
            self._dropping_event = True

    def finish(self) -> None:
        try:
            self._buffer += self._decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            self.parse_errors += 1
            self._buffer = ""
        if self._buffer:
            if len(self._buffer) > self._MAX_EVENT_CHARS:
                self.parse_errors += 1
            else:
                self._consume(self._buffer)
            self._buffer = ""

    def _consume(self, line: str) -> None:
        try:
            event = json.loads(line)
            event_type = event["type"]
            data = event["data"]
            if not isinstance(data, dict):
                raise TypeError
        except json.JSONDecodeError, KeyError, TypeError:
            self.parse_errors += 1
            return
        if event_type not in {"match", "context"}:
            return
        if event_type == "match":
            if self.matches >= self._limit:
                self.truncated = True
                return
            self.matches += 1
        elif self.matches >= self._limit and self.truncated:
            return
        try:
            path = self._normalize_path(_json_text(data["path"]))
            line_number = int(data["line_number"])
            source = _json_text(data["lines"]).rstrip("\r\n")
        except KeyError, TypeError, ValueError:
            self.parse_errors += 1
            return
        separator = ":" if event_type == "match" else "-"
        for index, source_line in enumerate(source.splitlines() or [""]):
            clipped = source_line
            if len(clipped) > self._max_line_chars:
                clipped = clipped[: self._max_line_chars] + "…[line truncated]"
            rendered = f"{_escape_path(path)}{separator}{line_number + index}{separator}{clipped}\n"
            self._output.append(ProcessChunk("stdout", rendered.encode("utf-8")))

    def _normalize_path(self, value: str) -> str:
        path = Path(value)
        if path.is_absolute():
            try:
                value = path.relative_to(self._workspace_root).as_posix()
            except ValueError as exc:
                raise ValueError("ripgrep returned a path outside the workspace") from exc
        return value.removeprefix("./") or "."


def _json_text(value: object) -> str:
    if not isinstance(value, dict):
        raise TypeError
    text = value.get("text")
    if isinstance(text, str):
        return text
    encoded = value.get("bytes")
    if isinstance(encoded, str):
        return base64.b64decode(encoded, validate=True).decode("utf-8", errors="replace")
    raise TypeError


class _NulPathCollector:
    """Bound and parse fd's unambiguous NUL-delimited output."""

    _MAX_OUTPUT_BYTES = 64 * 1024 * 1024
    _MAX_STDERR_CHARS = 64 * 1024

    def __init__(self, *, root: Path) -> None:
        self._root = root
        self._stdout = bytearray()
        self._total_stdout = 0
        self._stderr_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.paths: list[str] = []
        self.stderr = ""

    async def feed(self, chunk: ProcessChunk) -> None:
        if chunk.stream == "stderr":
            remaining = self._MAX_STDERR_CHARS - len(self.stderr)
            if remaining > 0:
                self.stderr += self._stderr_decoder.decode(chunk.data, final=False)[:remaining]
            return
        self._total_stdout += len(chunk.data)
        if self._total_stdout > self._MAX_OUTPUT_BYTES:
            raise ValueError("fd output exceeded its 64 MiB safety limit")
        self._stdout.extend(chunk.data)
        while (separator := self._stdout.find(0)) >= 0:
            raw = bytes(self._stdout[:separator])
            del self._stdout[: separator + 1]
            self.paths.append(self._normalize(raw))

    def finish(self) -> None:
        self.stderr += self._stderr_decoder.decode(b"", final=True)[
            : self._MAX_STDERR_CHARS - len(self.stderr)
        ]
        if self._stdout:
            raise ValueError("fd returned an unterminated path")

    def _normalize(self, raw: bytes) -> str:
        value = os.fsdecode(raw)
        directory_suffix = "/" if value.endswith("/") else ""
        candidate = Path(value.rstrip("/"))
        if candidate.is_absolute():
            try:
                candidate = candidate.relative_to(self._root)
            except ValueError as exc:
                raise ValueError("fd returned a path outside the search root") from exc
        normalized = candidate.as_posix().removeprefix("./")
        if not normalized or ".." in candidate.parts:
            raise ValueError("fd returned an unsafe path")
        return normalized + directory_suffix


def _streaming_output(
    tool: str,
    factory: OutputStageFactory | None,
) -> StreamingToolOutput:
    return StreamingToolOutput(
        stage=(factory(tool) if factory is not None else None),
        max_bytes=TOOL_RESULT_MAX_BYTES,
        max_lines=TOOL_RESULT_MAX_LINES,
    )


def _process_notice_reserve(notices: Sequence[str]) -> tuple[int, int]:
    return (
        sum(len(f"\n{notice}".encode()) for notice in notices),
        sum(len(notice.splitlines()) or 1 for notice in notices),
    )


def _stream_result(
    tool: str,
    snapshot: ToolOutputSnapshot,
    *,
    transient: bool = False,
    notices: Sequence[str] = (),
) -> ToolResult:
    details: dict[str, object] = {
        "output_bytes": snapshot.total_bytes,
        "output_lines": snapshot.total_lines,
        "spill_state": "committed"
        if snapshot.receipt
        else "staging"
        if snapshot.truncated
        else "none",
    }
    protected = ""
    prefix = ""
    if snapshot.truncated and not transient:
        if snapshot.receipt is None:
            raise FullOutputUnavailable("oversized process output has no durable spill backing")
        receipt = snapshot.receipt
        resource_id = receipt.resource_id
        protected = f"Full output: read(resource_id={resource_id!r}, cursor=...)"
        prefix = f"{tool} output required a bounded continuation. {protected}\n"
    body = _compose_bounded_process_result(
        prefix=prefix,
        tail=snapshot.text,
        notices=notices,
    )
    return ToolResult.text(
        body,
        details=details,
        protected_text=protected,
        effects=ToolEffects(
            committed_outputs=((snapshot.receipt,) if snapshot.receipt is not None else ())
        ),
    )


def _compose_bounded_process_result(
    *,
    prefix: str,
    tail: str,
    notices: Sequence[str],
) -> str:
    tail_lines = tail.splitlines(keepends=True)

    def compose(drop: int) -> str:
        body = prefix + "".join(tail_lines[drop:])
        for notice in notices:
            body = f"{body}\n{notice}" if body else notice
        return body

    candidate = compose(0)
    if _within_result_bounds(candidate):
        return candidate
    low = 1
    high = len(tail_lines)
    while low < high:
        middle = (low + high) // 2
        if _within_result_bounds(compose(middle)):
            high = middle
        else:
            low = middle + 1
    candidate = compose(low)
    if not _within_result_bounds(candidate):
        raise FullOutputUnavailable("process result framing exceeded its bounded reserve")
    return candidate


def _within_result_bounds(text: str) -> bool:
    return (
        len(text.encode("utf-8")) <= TOOL_RESULT_MAX_BYTES
        and len(text.splitlines()) <= TOOL_RESULT_MAX_LINES
    )


def _utf8_prefix(text: str, *, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore") + "…"


def _utf8_excerpt(text: str, *, preview: Literal["head", "tail"]) -> str:
    lines = text.splitlines(keepends=True)
    selected = lines if preview == "head" else list(reversed(lines))
    kept: list[str] = []
    size = 0
    for line in selected:
        line_size = len(line.encode("utf-8"))
        if size + line_size > TOOL_RESULT_PREVIEW_BYTES:
            break
        kept.append(line)
        size += line_size
    if preview == "tail":
        kept.reverse()
    return "".join(kept)


def _workspace_relative_path(root: Path, path: Path) -> str:
    relative = path.relative_to(root).as_posix()
    return relative or "."


def _inventory_facts(root: object, path: object) -> WorkspaceInventoryFacts:
    from pathlib import Path

    file_path = Path(path)  # type: ignore[arg-type]
    root_path = Path(root)  # type: ignore[arg-type]
    data = file_path.read_bytes()
    record = WorkspacePathFact(
        relative_path=str(file_path.relative_to(root_path)),
        entry_type="file",
        size_bytes=len(data),
        mode=file_path.stat().st_mode,
        content_digest=hashlib.sha256(data).hexdigest(),
    )
    return WorkspaceInventoryFacts(upserts=(record,))


def _scan_inventory(root: object) -> WorkspaceInventoryFacts:
    from pathlib import Path

    root_path = Path(root)  # type: ignore[arg-type]
    upserts: list[WorkspacePathFact] = []
    for current, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [name for name in dirnames if not (Path(current) / name).is_symlink()]
        for name in filenames:
            file_path = Path(current) / name
            if file_path.is_symlink():
                continue
            try:
                metadata = file_path.stat()
            except OSError:
                continue
            if not stat.S_ISREG(metadata.st_mode):
                continue
            upserts.append(
                WorkspacePathFact(
                    relative_path=str(file_path.relative_to(root_path)),
                    entry_type="file",
                    size_bytes=metadata.st_size,
                    mode=metadata.st_mode,
                )
            )
    return WorkspaceInventoryFacts(upserts=tuple(upserts), replace_all=True)


def _paginate_lines(
    text: str,
    *,
    path: str,
    offset: int | None,
    limit: int | None,
    notice: str = "",
) -> tuple[str, str, int]:
    """Return one page whose complete continuation participates in its bounds."""
    lines = text.splitlines()
    start = (offset or 1) - 1
    page_size = min(limit or TOOL_RESULT_MAX_LINES, TOOL_RESULT_MAX_LINES)
    end = min(start + page_size, len(lines))

    def render(candidate_end: int) -> tuple[str, str, int]:
        continuation = ""
        remaining = len(lines) - candidate_end
        chunks = ["\n".join(lines[start:candidate_end])]
        if notice:
            chunks.append(notice)
        if remaining > 0:
            continuation = f"read(path={path!r}, offset={candidate_end + 1})"
            chunks.append(f"[{remaining} more lines; {continuation}]")
        return "\n".join(chunk for chunk in chunks if chunk), continuation, remaining

    rendered = render(end)
    if _within_result_bounds(rendered[0]) or end <= start + 1:
        return rendered

    # Find the largest advancing page that leaves room for its complete notice
    # and continuation. The search examines at most the global line ceiling,
    # regardless of a model-supplied limit or file size.
    low = start + 1
    high = end - 1
    best = low
    while low <= high:
        middle = (low + high) // 2
        candidate = render(middle)
        if _within_result_bounds(candidate[0]):
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return render(best)


__all__ = [
    "BashArgs",
    "EditArgs",
    "EditOperation",
    "FindArgs",
    "GrepArgs",
    "HttpReadOptions",
    "ImagePreparer",
    "LsArgs",
    "PreparedImageAttachment",
    "ReadArgs",
    "ResourceReadRequest",
    "OutputStageFactory",
    "ResourceReader",
    "SpillWriter",
    "WriteArgs",
    "bash_tool",
    "bound_tool_text",
    "edit_tool",
    "find_tool",
    "grep_tool",
    "ls_tool",
    "path_tools",
    "preview_or_spill",
    "read_tool",
    "write_tool",
]
