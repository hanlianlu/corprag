# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded, versioned directory-listing pages shared by read and ls."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Sequence
from typing import cast

from dlightrag.engine.agent.environment.errors import TOOL_RESULT_MAX_BYTES, TOOL_RESULT_MAX_LINES
from dlightrag.engine.agent.environment.local import DirectoryEntry
from dlightrag.engine.agent.tools.contracts import ToolResult

_CURSOR_MAX_CHARS = 1024


def directory_page(
    entries: Sequence[DirectoryEntry],
    *,
    path: str,
    cursor: str | None,
    limit: int,
    tool: str,
) -> ToolResult:
    """Render one stable page and reject cursors after listing changes."""
    items = tuple(entries)
    fingerprint = _directory_fingerprint(items)
    try:
        start = _decode_directory_cursor(cursor, path=path, fingerprint=fingerprint)
    except ValueError as exc:
        return ToolResult.text(str(exc), is_error=True)
    if start > len(items):
        return ToolResult.text("directory cursor is beyond the current listing", is_error=True)

    if not items:
        return ToolResult.text("(empty directory)")

    rows: list[str] = []
    end = start
    requested_end = min(len(items), start + limit)
    while end < requested_end:
        entry = items[end]
        row = f"{entry.kind}\t{entry.size}\t{escape_path(entry.name)}"
        if not _fits_result([*rows, row]):
            break
        rows.append(row)
        end += 1

    protected = ""
    if end < len(items):
        # The complete continuation—not a fixed allowance—participates in both
        # byte and line accounting. Remove trailing rows until the cursor-bearing
        # result fits; filenames themselves are bounded by the host filesystem.
        while True:
            next_cursor = _encode_directory_cursor(
                path=path,
                fingerprint=fingerprint,
                offset=end,
            )
            protected = f"{tool}(path={path!r}, cursor={next_cursor!r}, limit={limit})"
            marker = f"[{len(items) - end} more entries; continue with {protected}]"
            if rows and _fits_result([*rows, marker]):
                rows.append(marker)
                break
            if not rows:
                return ToolResult.text(
                    "directory continuation exceeds the result byte limit",
                    is_error=True,
                )
            rows.pop()
            end -= 1
    if not rows:
        return ToolResult.text("directory entry exceeds the result byte limit", is_error=True)
    return ToolResult.text(
        "\n".join(rows),
        protected_text=protected,
        details={"offset": start, "returned": end - start, "total_entries": len(items)},
    )


def escape_path(path: str) -> str:
    """Render one path as a single model-safe line without quoting ordinary names."""
    escaped = json.dumps(path, ensure_ascii=False)[1:-1]
    return escaped.encode("utf-8", errors="backslashreplace").decode("utf-8")


def _fits_result(rows: Sequence[str]) -> bool:
    if len(rows) > TOOL_RESULT_MAX_LINES:
        return False
    return len("\n".join(rows).encode("utf-8")) <= TOOL_RESULT_MAX_BYTES


def _directory_fingerprint(entries: Sequence[DirectoryEntry]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry.name.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(entry.kind.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry.size).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()[:24]


def _path_fingerprint(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:24]


def _encode_directory_cursor(*, path: str, fingerprint: str, offset: int) -> str:
    payload = json.dumps(
        {"v": 1, "p": _path_fingerprint(path), "f": fingerprint, "o": offset},
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_directory_cursor(
    cursor: str | None,
    *,
    path: str,
    fingerprint: str,
) -> int:
    if cursor is None:
        return 0
    if len(cursor) > _CURSOR_MAX_CHARS:
        raise ValueError("directory cursor is invalid or the directory changed")
    try:
        padding = "=" * (-len(cursor) % 4)
        payload = base64.b64decode(cursor + padding, altchars=b"-_", validate=True)
        decoded = json.loads(payload)
        if (
            not isinstance(decoded, dict)
            or decoded.get("v") != 1
            or decoded.get("p") != _path_fingerprint(path)
            or decoded.get("f") != fingerprint
            or not isinstance(decoded.get("o"), int)
            or decoded["o"] < 0
        ):
            raise ValueError
    except ValueError, TypeError, json.JSONDecodeError:
        raise ValueError("directory cursor is invalid or the directory changed") from None
    return cast(int, decoded["o"])


__all__ = ["directory_page", "escape_path"]
