# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Local trusted execution: rooted files plus explicit child processes."""

from __future__ import annotations

import asyncio
import os
import signal
import stat
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from dlightrag.engine.agent.environment.errors import (
    WORKSPACE_MAX_BYTES,
    WORKSPACE_MAX_ENTRIES,
    PathRejected,
    WorkspaceQuotaExceeded,
)


@dataclass(frozen=True, slots=True)
class DirectoryEntry:
    """One listing row: relative name, type, and size in bytes."""

    name: str
    kind: str
    size: int


@dataclass(frozen=True, slots=True)
class ProcessChunk:
    """One ordered stdout or stderr byte chunk from a child process."""

    stream: Literal["stdout", "stderr"]
    data: bytes


type ProcessOutputSink = Callable[[ProcessChunk], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class CompletedProcess:
    """One child-process terminal status; output is delivered incrementally."""

    returncode: int
    timed_out: bool = False


class LocalExecutionEnvironment:
    """POSIX workspace rooted at one directory. Not a security boundary."""

    def __init__(self, root: Path) -> None:
        resolved = root.expanduser().resolve()
        if not resolved.is_absolute():
            raise ValueError("execution environment root must be absolute")
        resolved.mkdir(parents=True, exist_ok=True)
        self._root = resolved
        (
            self._integrity_violations,
            self._usage_entries,
            self._usage_bytes,
        ) = self._scan_workspace_state()
        self._quota_violation = self._quota_error(
            entries=self._usage_entries,
            total_bytes=self._usage_bytes,
        )

    @property
    def root(self) -> Path:
        return self._root

    @property
    def integrity_violations(self) -> tuple[str, ...]:
        """Unsafe or unreadable entries left by unrestricted workspace changes."""
        return self._integrity_violations

    @property
    def quota_violation(self) -> str | None:
        """Latched workspace quota failure left by unrestricted Bash."""
        return self._quota_violation

    def refresh_integrity(self) -> tuple[str, ...]:
        """Rescan special entries and quota usage after unrestricted execution."""
        (
            self._integrity_violations,
            self._usage_entries,
            self._usage_bytes,
        ) = self._scan_workspace_state()
        self._quota_violation = self._quota_error(
            entries=self._usage_entries,
            total_bytes=self._usage_bytes,
        )
        return self._integrity_violations

    def _scan_workspace_state(self) -> tuple[tuple[str, ...], int, int]:
        """Scan at most one quota envelope; Bash cannot force unbounded state growth."""
        violations: list[str] = []
        entries = 0
        total_bytes = 0
        stack = [self._root]
        while stack:
            directory = stack.pop()
            try:
                children = os.scandir(directory)
            except OSError:
                if len(violations) < 100:
                    relative = (
                        "."
                        if directory == self._root
                        else directory.relative_to(self._root).as_posix()
                    )
                    violations.append(f"{relative} [unreadable]")
                continue
            with children:
                for child in children:
                    entries += 1
                    try:
                        metadata = child.stat(follow_symlinks=False)
                    except OSError:
                        if len(violations) < 100:
                            relative = Path(child.path).relative_to(self._root).as_posix()
                            violations.append(f"{relative} [unreadable]")
                        continue
                    mode = metadata.st_mode
                    relative = Path(child.path).relative_to(self._root).as_posix()
                    if stat.S_ISLNK(mode):
                        if len(violations) < 100:
                            violations.append(relative)
                    elif stat.S_ISDIR(mode):
                        stack.append(Path(child.path))
                    elif stat.S_ISREG(mode):
                        total_bytes += metadata.st_size
                    elif (
                        stat.S_ISFIFO(mode)
                        or stat.S_ISCHR(mode)
                        or stat.S_ISBLK(mode)
                        or stat.S_ISSOCK(mode)
                    ) and len(violations) < 100:
                        violations.append(relative)
                    if entries > WORKSPACE_MAX_ENTRIES or total_bytes > WORKSPACE_MAX_BYTES:
                        stack.clear()
                        break
        return (
            tuple(sorted(violations, key=lambda item: (item.casefold(), item))),
            entries,
            total_bytes,
        )

    def prepare_process_directories(self) -> tuple[Path, Path]:
        """Create HOME/TMP inside the workspace and account for them incrementally."""
        tmp = self.resolve("tmp")
        home = self.resolve("tmp/home")
        missing = self._missing_parent_directories(home)
        if error := self._quota_error(
            entries=self._usage_entries + len(missing),
            total_bytes=self._usage_bytes,
        ):
            raise WorkspaceQuotaExceeded(error)
        if missing:
            try:
                home.mkdir(parents=True, exist_ok=True)
                self._fsync_directory(home)
                for directory in reversed(missing):
                    self._fsync_directory(directory.parent)
            except Exception:
                self._usage_entries, self._usage_bytes = self._scan_workspace_usage()
                self._quota_violation = self._quota_error(
                    entries=self._usage_entries,
                    total_bytes=self._usage_bytes,
                )
                raise
            self._usage_entries += len(missing)
        return home, tmp

    def resolve(self, relative: str) -> Path:
        candidate = relative.strip()
        if not candidate or candidate.startswith(("/", "~")) or "\x00" in candidate:
            raise PathRejected("path must be a relative workspace path")
        parts = Path(candidate).parts
        if any(part in {".", ".."} for part in parts if part == ".."):
            raise PathRejected("path must not escape the workspace")
        if ".." in parts:
            raise PathRejected("path must not escape the workspace")
        current = self._root
        for part in parts:
            current = current / part
            if current.is_symlink():
                raise PathRejected("path tools never follow symbolic links")
        resolved = (self._root / candidate).resolve()
        if not resolved.is_relative_to(self._root):
            raise PathRejected("path must stay inside the workspace")
        if resolved.exists() and not (resolved.is_file() or resolved.is_dir()):
            raise PathRejected("path must name a regular file or directory")
        if resolved.exists() and stat.S_ISLNK(resolved.lstat().st_mode) is False:
            mode = resolved.lstat().st_mode
            if (
                stat.S_ISFIFO(mode)
                or stat.S_ISCHR(mode)
                or stat.S_ISBLK(mode)
                or stat.S_ISSOCK(mode)
            ):
                raise PathRejected("path must name a regular file or directory")
        return resolved

    def stat_kind(self, path: Path) -> str:
        if not path.exists():
            return "missing"
        if path.is_dir():
            return "directory"
        if path.is_file():
            return "file"
        raise PathRejected("path must name a regular file or directory")

    def list_directory(self, path: Path) -> tuple[DirectoryEntry, ...]:
        if not path.is_dir():
            raise PathRejected("directory listing requires a directory")
        entries: list[DirectoryEntry] = []
        for child in path.iterdir():
            try:
                if child.is_symlink():
                    kind, size = "symlink", 0
                elif child.is_dir():
                    kind, size = "directory", 0
                elif child.is_file():
                    kind, size = "file", child.stat().st_size
                else:
                    kind, size = "special", 0
            except OSError:
                # One concurrently removed or unreadable entry must not hide
                # the rest of an otherwise readable directory.
                continue
            entries.append(DirectoryEntry(name=child.name, kind=kind, size=size))
        entries.sort(key=lambda entry: (entry.name.casefold(), entry.name))
        return tuple(entries)

    def read_bytes(self, path: Path) -> bytes:
        if not path.is_file():
            raise PathRejected("read requires a regular file")
        return path.read_bytes()

    def write_bytes(self, path: Path, data: bytes) -> None:
        if path.is_symlink():
            raise PathRejected("path tools never follow symbolic links")
        if path.exists() and path.is_dir():
            raise PathRejected("cannot overwrite a directory")

        existed = path.is_file()
        old_size = path.stat().st_size if existed else 0
        metadata = self._snapshot_file_metadata(path) if existed else None
        missing_parents = self._missing_parent_directories(path.parent)
        proposed_entries = self._usage_entries + len(missing_parents) + (0 if existed else 1)
        proposed_bytes = self._usage_bytes - old_size + len(data)
        if error := self._quota_error(entries=proposed_entries, total_bytes=proposed_bytes):
            raise WorkspaceQuotaExceeded(error)

        descriptor: int | None = None
        tmp: Path | None = None
        replaced = False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, tmp_name = tempfile.mkstemp(prefix=".dlightrag-write-", dir=path.parent)
            tmp = Path(tmp_name)
            handle = os.fdopen(descriptor, "wb")
            descriptor = None
            with handle:
                handle.write(data)
                if metadata is not None:
                    os.fchmod(handle.fileno(), metadata[0])
                handle.flush()
                os.fsync(handle.fileno())
            if metadata is not None:
                self._restore_extended_attributes(tmp, metadata[1])
                self._fsync_file(tmp)
            tmp.replace(path)
            replaced = True
            self._fsync_directory(path.parent)
            for directory in reversed(missing_parents):
                self._fsync_directory(directory.parent)
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            if not replaced:
                if tmp is not None:
                    tmp.unlink(missing_ok=True)
                for directory in reversed(missing_parents):
                    try:
                        directory.rmdir()
                    except OSError:
                        break
            self.refresh_integrity()
            raise

        self._usage_entries = proposed_entries
        self._usage_bytes = proposed_bytes
        self._quota_violation = None

    async def run(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        cwd: Path | None = None,
        timeout_seconds: float | None = None,
        on_output: ProcessOutputSink | None = None,
    ) -> CompletedProcess:
        if not argv:
            raise ValueError("process argv cannot be empty")
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd or self._root),
            env=dict(env),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        async def discard(_chunk: ProcessChunk) -> None:
            return None

        sink = on_output or discard

        async def pump(
            reader: asyncio.StreamReader | None,
            stream: Literal["stdout", "stderr"],
        ) -> None:
            if reader is None:
                return
            while chunk := await reader.read(64 * 1024):
                await sink(ProcessChunk(stream=stream, data=chunk))

        async def wait_and_terminate_descendants() -> int:
            returncode = await process.wait()
            # A successful shell may leave redirected background jobs in its
            # process group. End them before Bash releases WorkspaceAccess and
            # performs its final integrity/quota scan.
            self._terminate_group(process)
            return returncode

        tasks = (
            asyncio.create_task(wait_and_terminate_descendants()),
            asyncio.create_task(pump(process.stdout, "stdout")),
            asyncio.create_task(pump(process.stderr, "stderr")),
        )
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout_seconds)
        except TimeoutError:
            self._terminate_group(process)
            await asyncio.shield(process.wait())
            await asyncio.gather(*tasks[1:], return_exceptions=True)
            return CompletedProcess(
                returncode=process.returncode or -signal.SIGKILL,
                timed_out=True,
            )
        except asyncio.CancelledError:
            self._terminate_group(process)
            await asyncio.shield(process.wait())
            for task in tasks[1:]:
                task.cancel()
            await asyncio.gather(*tasks[1:], return_exceptions=True)
            raise
        except BaseException:
            self._terminate_group(process)
            await asyncio.shield(process.wait())
            for task in tasks[1:]:
                task.cancel()
            await asyncio.gather(*tasks[1:], return_exceptions=True)
            raise
        return CompletedProcess(returncode=process.returncode or 0)

    def _terminate_group(self, process: object) -> None:
        pid = getattr(process, "pid", None)
        if pid is None:
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError, PermissionError:
            return
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError, PermissionError:
            return

    def _scan_workspace_usage(self) -> tuple[int, int]:
        _violations, entries, total_bytes = self._scan_workspace_state()
        return entries, total_bytes

    @staticmethod
    def _quota_error(*, entries: int, total_bytes: int) -> str | None:
        if entries > WORKSPACE_MAX_ENTRIES:
            return f"workspace quota exceeded: {entries} entries > {WORKSPACE_MAX_ENTRIES} entries"
        if total_bytes > WORKSPACE_MAX_BYTES:
            return f"workspace quota exceeded: {total_bytes} bytes > {WORKSPACE_MAX_BYTES} bytes"
        return None

    def _missing_parent_directories(self, parent: Path) -> list[Path]:
        missing: list[Path] = []
        current = parent
        while current != self._root and not current.exists():
            missing.append(current)
            current = current.parent
        missing.reverse()
        return missing

    @staticmethod
    def _snapshot_file_metadata(path: Path) -> tuple[int, dict[str, bytes]]:
        mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
        attributes: dict[str, bytes] = {}
        list_xattrs = cast(Callable[..., list[str]] | None, getattr(os, "listxattr", None))
        get_xattr = cast(Callable[..., bytes] | None, getattr(os, "getxattr", None))
        if callable(list_xattrs) and callable(get_xattr):
            for name in list_xattrs(path, follow_symlinks=False):
                attributes[name] = get_xattr(path, name, follow_symlinks=False)
        return mode, attributes

    @staticmethod
    def _restore_extended_attributes(path: Path, attributes: Mapping[str, bytes]) -> None:
        set_xattr = cast(Callable[..., None] | None, getattr(os, "setxattr", None))
        if attributes and not callable(set_xattr):
            raise OSError("platform cannot preserve file extended attributes")
        if not callable(set_xattr):
            return
        for name, value in attributes.items():
            set_xattr(path, name, value, follow_symlinks=False)

    @staticmethod
    def _fsync_file(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = [
    "CompletedProcess",
    "DirectoryEntry",
    "LocalExecutionEnvironment",
    "ProcessChunk",
    "ProcessOutputSink",
]
