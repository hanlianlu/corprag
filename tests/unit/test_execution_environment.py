# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Path policy, encoding, atomic writes, and process cleanup."""

import asyncio
import os
import signal
import sys
from pathlib import Path

import pytest

from dlightrag.engine.agent.environment import PathRejected, TrustExecutionAdapter
from dlightrag.engine.agent.environment.local import LocalExecutionEnvironment, ProcessChunk
from dlightrag.engine.agent.environment.text import decode_workspace_text, encode_workspace_text


def test_rejects_absolute_parent_and_symlink_escape(tmp_path: Path) -> None:
    env = LocalExecutionEnvironment(tmp_path)
    with pytest.raises(PathRejected):
        env.resolve("/etc/passwd")
    with pytest.raises(PathRejected):
        env.resolve("../secret")
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("nope", encoding="utf-8")
    (tmp_path / "link").symlink_to(outside)
    with pytest.raises(PathRejected):
        env.resolve("link")


def test_write_is_atomic_and_creates_parents(tmp_path: Path) -> None:
    env = LocalExecutionEnvironment(tmp_path)
    target = env.resolve("notes/hello.txt")
    env.write_bytes(target, b"hello")
    assert target.read_text(encoding="utf-8") == "hello"
    leftovers = list(target.parent.glob(".dlightrag-write-*"))
    assert leftovers == []


def test_atomic_overwrite_preserves_mode_and_extended_attributes(tmp_path: Path) -> None:
    target = tmp_path / "script.sh"
    target.write_bytes(b"old")
    target.chmod(0o751)
    xattr_name = "user.dlightrag-test"
    xattrs_supported = False
    set_xattr = getattr(os, "setxattr", None)
    get_xattr = getattr(os, "getxattr", None)
    if callable(set_xattr) and callable(get_xattr):
        try:
            set_xattr(target, xattr_name, b"kept")
            xattrs_supported = True
        except OSError:
            pass
    env = LocalExecutionEnvironment(tmp_path)

    env.write_bytes(target, b"new")

    assert target.read_bytes() == b"new"
    assert target.stat().st_mode & 0o777 == 0o751
    if xattrs_supported and callable(get_xattr):
        assert get_xattr(target, xattr_name) == b"kept"


def test_write_failure_after_parent_creation_restores_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = LocalExecutionEnvironment(tmp_path)

    def fail_mkstemp(*_args: object, **_kwargs: object) -> tuple[int, str]:
        raise OSError("temporary creation failed")

    monkeypatch.setattr("tempfile.mkstemp", fail_mkstemp)
    with pytest.raises(OSError, match="temporary creation failed"):
        env.write_bytes(env.resolve("new/parent/file.txt"), b"content")

    assert not (tmp_path / "new").exists()
    assert env._usage_entries == 0
    assert env._usage_bytes == 0
    assert env.integrity_violations == ()
    assert env.quota_violation is None


def test_write_quota_uses_incremental_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = LocalExecutionEnvironment(tmp_path)

    def fail_rescan() -> tuple[int, int]:
        raise AssertionError("ordinary writes must not rescan the workspace")

    monkeypatch.setattr(env, "_scan_workspace_usage", fail_rescan)
    env.write_bytes(env.resolve("one.txt"), b"one")
    env.write_bytes(env.resolve("nested/two.txt"), b"two")
    assert (tmp_path / "nested" / "two.txt").read_bytes() == b"two"


def test_utf8_and_bom_tagged_utf16_round_trip() -> None:
    utf8 = decode_workspace_text("café\n".encode())
    assert utf8.text == "café\n"
    tagged = decode_workspace_text(b"\xff\xfeh\x00i\x00")
    assert tagged.text == "hi"
    with pytest.raises(ValueError, match="not UTF-8"):
        decode_workspace_text(b"\x80\x81not-utf8")
    crlf = decode_workspace_text(b"a\r\nb\r\n")
    assert crlf.newline == "\r\n"
    assert crlf.text == "a\nb\n"
    assert encode_workspace_text(crlf, "a\nb\n") == b"a\r\nb\r\n"


def test_directory_listing_is_sorted_one_level(tmp_path: Path) -> None:
    env = LocalExecutionEnvironment(tmp_path)
    (tmp_path / "b").mkdir()
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    names = [entry.name for entry in env.list_directory(tmp_path)]
    assert names == ["a.txt", "b"]


async def test_process_run_streams_output_before_exit(tmp_path: Path) -> None:
    env = LocalExecutionEnvironment(tmp_path)
    first = asyncio.Event()
    chunks: list[ProcessChunk] = []

    async def record(chunk: ProcessChunk) -> None:
        chunks.append(chunk)
        if b"first" in chunk.data:
            first.set()

    task = asyncio.create_task(
        env.run(
            (
                sys.executable,
                "-c",
                "import time; print('first', flush=True); time.sleep(0.2); print('second')",
            ),
            env=os.environ,
            on_output=record,
        )
    )

    await asyncio.wait_for(first.wait(), timeout=1)
    assert not task.done()
    completed = await task
    assert completed.returncode == 0
    assert completed.timed_out is False
    assert b"first" in b"".join(chunk.data for chunk in chunks)
    assert b"second" in b"".join(chunk.data for chunk in chunks)
    assert {chunk.stream for chunk in chunks} == {"stdout"}


async def test_successful_process_run_terminates_background_process_group(tmp_path: Path) -> None:
    env = LocalExecutionEnvironment(tmp_path)
    late_path = tmp_path / "late.txt"
    command = f"(sleep 0.2; printf late > {late_path!s}) >/dev/null 2>&1 &"

    completed = await env.run(("/bin/bash", "-lc", command), env=os.environ)
    await asyncio.sleep(0.4)

    assert completed.returncode == 0
    assert not late_path.exists()


async def test_timed_out_process_run_terminates_its_process_group(tmp_path: Path) -> None:
    env = LocalExecutionEnvironment(tmp_path)
    parent_path = tmp_path / "timeout-parent-pid"
    child_path = tmp_path / "timeout-child-pid"
    script = (
        "import os,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f"open({str(parent_path)!r}, 'w').write(str(os.getpid())); "
        f"open({str(child_path)!r}, 'w').write(str(child.pid)); "
        "time.sleep(60)"
    )

    completed = await env.run(
        (sys.executable, "-c", script),
        env=os.environ,
        timeout_seconds=0.5,
    )

    assert completed.timed_out is True
    assert parent_path.exists() and child_path.exists()
    pids = (
        int(parent_path.read_text(encoding="utf-8")),
        int(child_path.read_text(encoding="utf-8")),
    )
    alive: list[int] = []
    for _ in range(100):
        alive = []
        for pid in pids:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            alive.append(pid)
        if not alive:
            break
        await asyncio.sleep(0.01)
    assert alive == []


async def test_timed_out_process_run_drains_output_before_returning(tmp_path: Path) -> None:
    env = LocalExecutionEnvironment(tmp_path)
    pid_path = tmp_path / "drain-parent-pid"
    first_chunk = asyncio.Event()
    release_sink = asyncio.Event()
    chunks: list[ProcessChunk] = []

    async def slow_sink(chunk: ProcessChunk) -> None:
        chunks.append(chunk)
        if not first_chunk.is_set():
            first_chunk.set()
            await release_sink.wait()

    async def release_after_process_exit() -> None:
        await first_chunk.wait()
        while not pid_path.exists():
            await asyncio.sleep(0)
        pid = int(pid_path.read_text(encoding="utf-8"))
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(0.005)
        release_sink.set()

    release = asyncio.create_task(release_after_process_exit())
    try:
        completed = await env.run(
            (
                sys.executable,
                "-c",
                (
                    "import os,time; "
                    f"open({str(pid_path)!r}, 'w').write(str(os.getpid())); "
                    "print('first', flush=True); "
                    "time.sleep(0.05); "
                    "print('second', flush=True); "
                    "time.sleep(60)"
                ),
            ),
            env=os.environ,
            timeout_seconds=0.2,
            on_output=slow_sink,
        )
    finally:
        release_sink.set()
        if not release.done():
            release.cancel()
        await asyncio.gather(release, return_exceptions=True)

    assert completed.timed_out is True
    output = b"".join(chunk.data for chunk in chunks)
    assert b"first" in output
    assert b"second" in output


async def test_descendant_holding_output_pipe_cannot_block_process_completion(
    tmp_path: Path,
) -> None:
    env = LocalExecutionEnvironment(tmp_path)
    chunks: list[ProcessChunk] = []

    async def record(chunk: ProcessChunk) -> None:
        chunks.append(chunk)

    pid_path = tmp_path / "escaped-pipe-holder-pid"
    script = (
        "import subprocess,sys; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], "
        "start_new_session=True); "
        f"open({str(pid_path)!r}, 'w').write(str(child.pid)); "
        "print('spawned', flush=True)"
    )
    try:
        completed = await asyncio.wait_for(
            env.run(
                (sys.executable, "-c", script),
                env=os.environ,
                on_output=record,
            ),
            timeout=2,
        )
        assert completed.returncode == 0
        assert b"spawned" in b"".join(chunk.data for chunk in chunks)
    finally:
        if pid_path.exists():
            try:
                os.kill(int(pid_path.read_text(encoding="utf-8")), signal.SIGKILL)
            except ProcessLookupError:
                pass


async def test_cancelling_process_run_terminates_its_process_group(tmp_path: Path) -> None:
    env = LocalExecutionEnvironment(tmp_path)
    parent_path = tmp_path / "parent-pid"
    child_path = tmp_path / "child-pid"
    script = (
        "import os,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f"open({str(parent_path)!r}, 'w').write(str(os.getpid())); "
        f"open({str(child_path)!r}, 'w').write(str(child.pid)); "
        "time.sleep(60)"
    )
    task = asyncio.create_task(env.run((sys.executable, "-c", script), env=os.environ))
    for _ in range(100):
        if parent_path.exists() and child_path.exists():
            break
        await asyncio.sleep(0.01)
    assert parent_path.exists() and child_path.exists()
    pids = (
        int(parent_path.read_text(encoding="utf-8")),
        int(child_path.read_text(encoding="utf-8")),
    )

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    alive: list[int] = []
    for _ in range(100):
        alive = []
        for pid in pids:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            alive.append(pid)
        if not alive:
            break
        await asyncio.sleep(0.01)
    assert alive == []


async def test_closing_trust_adapter_terminates_and_reaps_active_process(
    tmp_path: Path,
) -> None:
    adapter = TrustExecutionAdapter()
    env = adapter.create(tmp_path)
    pid_path = tmp_path / "adapter-close-pid"
    task = asyncio.create_task(
        env.run(
            (
                sys.executable,
                "-c",
                (
                    "import os,time; "
                    f"open({str(pid_path)!r}, 'w').write(str(os.getpid())); "
                    "time.sleep(60)"
                ),
            ),
            env=os.environ,
        )
    )
    for _ in range(100):
        if pid_path.exists():
            break
        await asyncio.sleep(0.01)
    assert pid_path.exists()
    pid = int(pid_path.read_text(encoding="utf-8"))

    await adapter.aclose()

    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    with pytest.raises(RuntimeError, match="closed"):
        adapter.create(tmp_path / "later")
    with pytest.raises(RuntimeError, match="closed"):
        await env.run((sys.executable, "-c", "pass"), env=os.environ)


async def test_cancelling_process_run_terminates_and_reaps_process(tmp_path: Path) -> None:
    env = LocalExecutionEnvironment(tmp_path)
    pid_path = tmp_path / "pid"
    task = asyncio.create_task(
        env.run(
            (
                sys.executable,
                "-c",
                (
                    "import os,time; "
                    f"open({str(pid_path)!r}, 'w').write(str(os.getpid())); "
                    "time.sleep(60)"
                ),
            ),
            env=os.environ,
        )
    )
    for _ in range(100):
        if pid_path.exists():
            break
        await asyncio.sleep(0.01)
    assert pid_path.exists()
    pid = int(pid_path.read_text(encoding="utf-8"))

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
