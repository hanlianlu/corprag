# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Path tool behavior: edit matches, directory read, oversized guard, bash env."""

import re
import shlex
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from dlightrag.engine.agent.environment import AccessScheduler, FullOutputUnavailable
from dlightrag.engine.agent.environment.errors import TOOL_RESULT_MAX_BYTES
from dlightrag.engine.agent.environment.local import LocalExecutionEnvironment
from dlightrag.engine.agent.environment.toolchain import SearchToolchain
from dlightrag.engine.agent.tools.contracts import CommittedOutput
from dlightrag.engine.agent.tools.files import (
    BashArgs,
    EditArgs,
    EditOperation,
    FindArgs,
    GrepArgs,
    LsArgs,
    ReadArgs,
    WriteArgs,
    bash_tool,
    edit_tool,
    find_tool,
    grep_tool,
    ls_tool,
    path_tools,
    preview_or_spill,
    read_tool,
    write_tool,
)
from tests.tool_helpers import tool_runtime


def _env(tmp_path: Path) -> tuple[LocalExecutionEnvironment, AccessScheduler]:
    return LocalExecutionEnvironment(tmp_path), AccessScheduler()


def _compatible_binary(tmp_path: Path, name: str, reported_version: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        pytest.skip(f"{name} not installed")
    wrapper = tmp_path / ".test-bin" / name
    wrapper.parent.mkdir(exist_ok=True)
    wrapper.write_text(
        "#!/bin/sh\n"
        f"if [ \"$1\" = --version ]; then printf '%s\\n' {shlex.quote(reported_version)}; "
        f'else exec {shlex.quote(executable)} "$@"; fi\n',
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return str(wrapper)


@pytest.mark.asyncio
async def test_batch_edit_is_unique_atomic_and_returns_a_patch(tmp_path: Path) -> None:
    env, scheduler = _env(tmp_path)
    writer = write_tool(env, scheduler)
    editor = edit_tool(env, scheduler)
    original = "alpha beta alpha\nlast line\n"
    await writer.execute(WriteArgs(path="note.txt", content=original), tool_runtime())
    ambiguous = await editor.execute(
        EditArgs(
            path="note.txt",
            edits=[EditOperation(old_text="alpha", new_text="A")],
        ),
        tool_runtime(),
    )
    assert ambiguous.is_error is True
    assert "matches 2 times" in ambiguous.text_content
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == original

    rejected = await editor.execute(
        EditArgs(
            path="note.txt",
            edits=[
                EditOperation(old_text="alpha beta", new_text="first"),
                EditOperation(old_text="missing", new_text="x"),
            ],
        ),
        tool_runtime(),
    )
    assert rejected.is_error is True
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == original

    done = await editor.execute(
        EditArgs(
            path="note.txt",
            edits=[
                EditOperation(old_text="alpha beta", new_text="first"),
                EditOperation(old_text="last line", new_text="final"),
            ],
        ),
        tool_runtime(),
    )
    assert done.is_error is False
    assert "first change line 1" in done.text_content
    assert "--- note.txt" in done.text_content
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "first alpha\nfinal\n"


@pytest.mark.asyncio
async def test_directory_read_is_sorted(tmp_path: Path) -> None:
    env, scheduler = _env(tmp_path)
    (tmp_path / "z").mkdir()
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    reader = read_tool(env, scheduler)
    result = await reader.execute(ReadArgs(path="."), tool_runtime())
    assert result.text_content.index("a.txt") < result.text_content.index("z")


async def test_directory_read_cursor_continues_and_rejects_mutation(tmp_path: Path) -> None:
    env, scheduler = _env(tmp_path)
    for name in ("a", "b", "c"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    reader = read_tool(env, scheduler)

    first = await reader.execute(ReadArgs(path=".", limit=2), tool_runtime())
    cursor_match = re.search(r"cursor='([^']+)'", first.protected_text)
    assert cursor_match is not None
    cursor = cursor_match.group(1)
    second = await reader.execute(ReadArgs(path=".", cursor=cursor, limit=2), tool_runtime())
    assert second.text_content.endswith("\tc")

    (tmp_path / "d").write_text("d", encoding="utf-8")
    stale = await reader.execute(ReadArgs(path=".", cursor=cursor, limit=2), tool_runtime())
    assert stale.is_error is True
    assert "directory changed" in stale.text_content


async def test_directory_continuation_is_canonical_and_fully_bounded(tmp_path: Path) -> None:
    env, scheduler = _env(tmp_path)
    for index in range(600):
        (tmp_path / f"entry-{index:04}.txt").write_text("x", encoding="utf-8")
    redundant = "./" * 1000 + "."

    result = await read_tool(env, scheduler).execute(
        ReadArgs(path=redundant, limit=500),
        tool_runtime(),
    )

    assert result.is_error is False
    assert len(result.text_content.encode("utf-8")) <= TOOL_RESULT_MAX_BYTES
    assert "read(path='.', cursor=" in result.protected_text
    assert redundant not in result.text_content
    with pytest.raises(ValidationError):
        ReadArgs(path="./" * 30000 + ".")


async def test_read_file_continuation_uses_canonical_bounded_path(tmp_path: Path) -> None:
    env, scheduler = _env(tmp_path)
    (tmp_path / "big.txt").write_text("\n".join(f"line-{i}" for i in range(3000)))
    redundant = "./" * 1000 + "big.txt"

    result = await read_tool(env, scheduler).execute(
        ReadArgs(path=redundant),
        tool_runtime(),
    )

    assert len(result.text_content.encode("utf-8")) <= TOOL_RESULT_MAX_BYTES
    assert result.protected_text == "read(path='big.txt', offset=2000)"
    assert redundant not in result.text_content


async def test_read_continuation_round_trips_backslash_path(tmp_path: Path) -> None:
    env, scheduler = _env(tmp_path)
    name = "back\\slash.txt"
    (tmp_path / name).write_text("\n".join(f"line-{index}" for index in range(3000)))

    result = await read_tool(env, scheduler).execute(ReadArgs(path=name), tool_runtime())

    assert result.protected_text == f"read(path={name!r}, offset=2000)"
    assert len(result.text_content.encode("utf-8")) <= TOOL_RESULT_MAX_BYTES


async def test_read_large_limit_is_schema_bounded_and_page_work_is_line_bounded(
    tmp_path: Path,
) -> None:
    env, scheduler = _env(tmp_path)
    (tmp_path / "many.txt").write_text("\n".join("x" for _ in range(10_000)))

    result = await read_tool(env, scheduler).execute(
        ReadArgs(path="many.txt", limit=100_000),
        tool_runtime(),
    )

    assert len(result.text_content.splitlines()) <= 2000
    assert result.protected_text == "read(path='many.txt', offset=2000)"
    with pytest.raises(ValidationError):
        ReadArgs(path="many.txt", limit=100_001)


async def test_read_paginates_at_2000_lines_with_continuation(tmp_path: Path) -> None:
    env, scheduler = _env(tmp_path)
    text = "".join(f"line {i}\n" for i in range(1, 2005))
    (tmp_path / "big.txt").write_text(text, encoding="utf-8")
    reader = read_tool(env, scheduler)

    first = await reader.execute(ReadArgs(path="big.txt"), tool_runtime())

    assert first.text_content.startswith("line 1\n")
    assert "line 1999" in first.text_content
    assert "line 2000\n" not in first.text_content
    assert "more lines; read(path='big.txt', offset=2000)" in first.text_content
    assert first.protected_text == "read(path='big.txt', offset=2000)"

    second = await reader.execute(ReadArgs(path="big.txt", offset=2000), tool_runtime())
    assert second.text_content.startswith("line 2000")
    assert "line 2004" in second.text_content
    assert second.protected_text == ""


async def test_read_missing_file_is_a_true_error(tmp_path: Path) -> None:
    env, scheduler = _env(tmp_path)
    result = await read_tool(env, scheduler).execute(ReadArgs(path="gone.txt"), tool_runtime())
    assert result.is_error is True
    assert "file not found" in result.text_content


async def test_read_image_path_attaches_the_original_snapshot(tmp_path: Path) -> None:
    import io

    from PIL import Image

    env, scheduler = _env(tmp_path)
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (30, 90, 200)).save(buffer, "PNG")
    png = buffer.getvalue()
    (tmp_path / "chart.png").write_bytes(png)

    result = await read_tool(env, scheduler).execute(ReadArgs(path="chart.png"), tool_runtime())

    assert result.is_error is False
    from dlightrag.engine.agent.tool_content import tool_content_attachments

    (attachment,) = tool_content_attachments(result.parts)
    assert attachment.media_type == "image/png"
    assert attachment.data == png
    assert attachment.size_bytes == len(png)
    assert "image attachment" in result.text_content
    (attached,) = result.effects.attached_resources
    assert attached.resource_id == attachment.resource_id
    assert attached.content == png
    assert attached.mime_type == "image/png"


async def test_read_corrupt_image_falls_back_to_text_decoding(tmp_path: Path) -> None:
    env, scheduler = _env(tmp_path)
    (tmp_path / "fake.png").write_bytes(b"\x89PNG\r\n\x1a\nnot really a png")

    result = await read_tool(env, scheduler).execute(ReadArgs(path="fake.png"), tool_runtime())

    from dlightrag.engine.agent.tool_content import tool_content_attachments

    assert tool_content_attachments(result.parts) == ()


async def test_read_large_image_keeps_source_and_attaches_bounded_derivative(
    tmp_path: Path,
) -> None:
    import io

    from PIL import Image

    env, scheduler = _env(tmp_path)
    buffer = io.BytesIO()
    Image.new("RGB", (3000, 2000), (20, 80, 160)).save(buffer, "PNG")
    source = buffer.getvalue()
    (tmp_path / "large.png").write_bytes(source)

    result = await read_tool(env, scheduler).execute(ReadArgs(path="large.png"), tool_runtime())

    from dlightrag.engine.agent.tool_content import tool_content_attachments

    (attachment,) = tool_content_attachments(result.parts)
    assert attachment.data != source
    assert len(attachment.data) <= 3 * 1024 * 1024
    assert attachment.size_bytes == len(attachment.data)
    assert len(result.effects.attached_resources) == 2
    assert result.effects.attached_resources[0].content == source
    assert result.effects.attached_resources[1].content == attachment.data


@pytest.mark.asyncio
async def test_oversized_result_without_spill_raises() -> None:
    with pytest.raises(FullOutputUnavailable):
        await preview_or_spill("x" * (50 * 1024 + 1), spill=None, tool="read")


@pytest.mark.asyncio
async def test_oversized_result_with_spill_returns_receipt() -> None:
    async def spill(text: str) -> CommittedOutput:
        return CommittedOutput(
            resource_id="spill_1",
            content_digest="a" * 64,
            size_bytes=len(text.encode("utf-8")),
        )

    body, extra = await preview_or_spill("x" * (50 * 1024 + 1), spill=spill, tool="grep")
    assert "spill_1" in body
    assert extra is not None
    assert extra.resource_id == "spill_1"


@pytest.mark.asyncio
async def test_write_attaches_inventory_details(tmp_path: Path) -> None:
    env, scheduler = _env(tmp_path)
    writer = write_tool(env, scheduler)
    result = await writer.execute(WriteArgs(path="notes.md", content="hello"), tool_runtime())
    inventory = result.effects.workspace_inventory
    assert inventory is not None
    assert inventory.replace_all is False
    assert inventory.upserts[0].relative_path == "notes.md"
    assert inventory.upserts[0].size_bytes == 5


@pytest.mark.asyncio
async def test_path_tool_order_and_hardened_contract_versions(tmp_path: Path) -> None:
    env, scheduler = _env(tmp_path)
    tools = path_tools(env, scheduler=scheduler)

    assert [tool.name for tool in tools] == [
        "read",
        "bash",
        "edit",
        "write",
        "grep",
        "find",
        "ls",
    ]
    versions = {tool.name: tool.contract_version for tool in tools}
    assert versions == {
        "read": 3,
        "bash": 3,
        "edit": 3,
        "write": 3,
        "grep": 3,
        "find": 2,
        "ls": 2,
    }


async def test_find_is_stable_hidden_ignore_aware_and_never_follows_symlinks(
    tmp_path: Path,
) -> None:
    env, scheduler = _env(tmp_path)
    (tmp_path / ".hidden.py").write_text("hidden", encoding="utf-8")
    (tmp_path / "B.py").write_text("b", encoding="utf-8")
    (tmp_path / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("ignored", encoding="utf-8")
    (tmp_path / "ignored-by-dot-ignore.py").write_text("ignored", encoding="utf-8")
    (tmp_path / "ignored-by-fd-ignore.py").write_text("ignored", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (tmp_path / ".ignore").write_text("ignored-by-dot-ignore.py\n", encoding="utf-8")
    (tmp_path / ".fdignore").write_text("ignored-by-fd-ignore.py\n", encoding="utf-8")
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "nested.py").write_text("nested", encoding="utf-8")
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)

    toolchain = SearchToolchain(fd=_compatible_binary(tmp_path, "fd", "fd 10.5.0"))
    result = await find_tool(env, scheduler, search_toolchain=toolchain).execute(
        FindArgs(pattern="*.py"), tool_runtime(tool_name="find")
    )

    assert result.text_content.splitlines() == [
        ".hidden.py",
        "a.py",
        "B.py",
        "real/nested.py",
    ]
    assert "link/nested.py" not in result.text_content

    rooted = await find_tool(env, scheduler, search_toolchain=toolchain).execute(
        FindArgs(pattern="real/**/*.py"), tool_runtime(tool_name="find")
    )
    assert rooted.text_content == "real/nested.py"


async def test_find_sorts_the_complete_match_set_before_applying_limit(tmp_path: Path) -> None:
    env, scheduler = _env(tmp_path)
    for name in ("z.py", "b.py", "a.py", "m.py"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    toolchain = SearchToolchain(
        fd=_compatible_binary(tmp_path, "fd", "fd 10.5.0"),
        cache_root=tmp_path / "cache",
    )

    result = await find_tool(env, scheduler, search_toolchain=toolchain).execute(
        FindArgs(pattern="*.py", limit=2),
        tool_runtime(tool_name="find"),
    )

    assert result.text_content.splitlines()[:2] == ["a.py", "b.py"]
    assert "more matches exist" in result.text_content


async def test_find_fails_closed_without_a_compatible_fd(tmp_path: Path) -> None:
    env, scheduler = _env(tmp_path)
    toolchain = SearchToolchain(
        fd=str(tmp_path / "missing-fd"),
        cache_root=tmp_path / "empty-cache",
    )

    result = await find_tool(env, scheduler, search_toolchain=toolchain).execute(
        FindArgs(pattern="*"), tool_runtime(tool_name="find")
    )

    assert result.is_error is True
    assert "fd>=10.5.0 is required" in result.text_content


async def test_ls_lists_symlinks_without_following_and_applies_limit_after_sort(
    tmp_path: Path,
) -> None:
    env, scheduler = _env(tmp_path)
    (tmp_path / "b").write_text("b", encoding="utf-8")
    (tmp_path / "A").write_text("a", encoding="utf-8")
    (tmp_path / "link").symlink_to(tmp_path / "A")

    result = await ls_tool(env, scheduler).execute(LsArgs(limit=2), tool_runtime(tool_name="ls"))

    assert result.text_content.splitlines()[0].endswith("\tA")
    assert result.text_content.splitlines()[1].endswith("\tb")
    assert "1 more entries" in result.text_content
    assert "continue with ls(" in result.text_content
    all_entries = await ls_tool(env, scheduler).execute(LsArgs(), tool_runtime(tool_name="ls"))
    assert "symlink\t0\tlink" in all_entries.text_content


async def test_bash_final_control_framing_stays_within_result_bounds(
    tmp_path: Path,
) -> None:
    class Stage:
        def __init__(self) -> None:
            self.data = bytearray()

        def append(self, data: bytes) -> None:
            self.data.extend(data)

        async def commit(self) -> CommittedOutput:
            return CommittedOutput(
                resource_id="spill_bash_bounds",
                content_digest="a" * 64,
                size_bytes=len(self.data),
            )

        def discard(self) -> None:
            return None

    env, scheduler = _env(tmp_path)
    stages: list[Stage] = []

    def stage_factory(_tool: str) -> Stage:
        stage = Stage()
        stages.append(stage)
        return stage

    result = await bash_tool(
        env,
        scheduler,
        output_stage_factory=stage_factory,
    ).execute(
        BashArgs(
            command=("for ((i=0; i<2001; i++)); do printf 'line\\n'; done; ln -s /etc/passwd link")
        ),
        tool_runtime(tool_name="bash"),
    )

    assert result.is_error is True
    assert len(result.text_content.encode("utf-8")) <= TOOL_RESULT_MAX_BYTES
    assert len(result.text_content.splitlines()) <= 2000
    assert "Full output: read(resource_id='spill_bash_bounds', cursor=...)" in result.text_content
    assert "workspace is latched" in result.text_content
    assert result.effects.committed_outputs
    assert stages and stages[0].data.startswith(b"line\nline\n")
    assert b"workspace is latched" not in stages[0].data


async def test_bash_integrity_policy_blocks_other_tools_until_cleanup(
    tmp_path: Path,
) -> None:
    env, scheduler = _env(tmp_path)
    tools = {tool.name: tool for tool in path_tools(env, scheduler=scheduler)}

    violation = await tools["bash"].execute(
        BashArgs(command="ln -s /etc/passwd link"), tool_runtime(tool_name="bash")
    )

    assert violation.is_error is True
    assert "link" in violation.text_content
    assert "unsafe or unreadable workspace entries" in violation.text_content
    assert violation.effects.workspace_inventory is None

    blocked = await tools["read"].execute(ReadArgs(path="link"), tool_runtime())
    assert blocked.is_error is True
    assert "integrity latched" in blocked.text_content

    cleanup_blocked = await tools["bash"].execute(
        BashArgs(command="rm link"), tool_runtime(tool_name="bash")
    )
    assert cleanup_blocked.is_error is True
    assert (tmp_path / "link").is_symlink()

    (tmp_path / "link").unlink()
    assert env.refresh_integrity() == ()
    restored = await tools["read"].execute(ReadArgs(path="note.txt"), tool_runtime())
    assert restored.is_error is True  # missing file, but integrity gate passed
    assert "file not found" in restored.text_content


async def test_bash_quota_overflow_latches_all_path_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.engine.agent.environment import local

    monkeypatch.setattr(local, "WORKSPACE_MAX_BYTES", 8)
    env, scheduler = _env(tmp_path)
    tools = {tool.name: tool for tool in path_tools(env, scheduler=scheduler)}

    overflow = await tools["bash"].execute(
        BashArgs(command="printf 123456789 > large"), tool_runtime(tool_name="bash")
    )
    assert overflow.is_error is True
    assert "workspace quota exceeded" in overflow.text_content
    assert overflow.effects.workspace_inventory is None
    blocked = await tools["write"].execute(
        WriteArgs(path="small", content="x"), tool_runtime(tool_name="write")
    )
    assert blocked.is_error is True
    assert "integrity latched" in blocked.text_content

    (tmp_path / "large").unlink()
    env.refresh_integrity()
    restored = await tools["write"].execute(
        WriteArgs(path="small", content="x"), tool_runtime(tool_name="write")
    )
    assert restored.is_error is False


async def test_grep_limits_json_matches_for_single_and_multiple_files(
    tmp_path: Path,
) -> None:
    rg = _compatible_binary(tmp_path, "rg", "ripgrep 15.2.0")
    env, scheduler = _env(tmp_path)
    (tmp_path / "single.txt").write_text(
        "".join(f"hit {i}\n" for i in range(1, 12)), encoding="utf-8"
    )
    tool = grep_tool(env, scheduler, ripgrep=rg)
    # JSON events are normalized to workspace-relative path:line:text records.
    single = await tool.execute(GrepArgs(pattern="hit", path="single.txt", limit=3), tool_runtime())

    assert single.is_error is False
    assert "limited to 3 matching lines" in single.text_content
    assert "hit 4\n" not in single.text_content

    # Multi-file search terminated at the limit: matches beyond the limit are
    # never emitted even though the tree contains far more of them.
    for i in range(1, 6):
        (tmp_path / f"m{i}.txt").write_text(
            "".join(f"needle {j}\n" for j in range(1, 8)), encoding="utf-8"
        )
    multi = await tool.execute(GrepArgs(pattern="needle", limit=4), tool_runtime())

    assert multi.is_error is False
    assert "limited to 4 matching lines" in multi.text_content
    assert multi.text_content.count(":1:needle") + multi.text_content.count(":2:needle") <= 4


async def test_grep_uses_argv_not_a_shell(tmp_path: Path) -> None:
    fake = tmp_path / "rg"
    fake.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = --version ]; then echo 'ripgrep 15.2.0'; exit 0; fi\n"
        "printf '%s\\n' \"$@\" > argv.txt\n"
        'printf \'%s\\n\' \'{"type":"summary","data":{}}\'\n'
        "exit 1\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    env, scheduler = _env(tmp_path)
    (tmp_path / "hit.txt").write_text("needle", encoding="utf-8")

    tool = grep_tool(env, scheduler, ripgrep=str(fake))
    result = await tool.execute(GrepArgs(pattern="$(touch injected)"), tool_runtime())
    argv = (tmp_path / "argv.txt").read_text(encoding="utf-8").splitlines()
    assert result.is_error is False
    assert "--json" in argv
    assert "$(touch injected)" in argv
    assert not (tmp_path / "injected").exists()


async def test_grep_searches_hidden_ignored_and_limits_matching_lines(
    tmp_path: Path,
) -> None:
    rg = _compatible_binary(tmp_path, "rg", "ripgrep 15.2.0")
    env, scheduler = _env(tmp_path)
    (tmp_path / "visible.txt").write_text(
        "needle one\nplain\nneedle two\nneedle three\n", encoding="utf-8"
    )
    (tmp_path / ".hidden.txt").write_text("hidden needle\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("ignored needle\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    tool = grep_tool(env, scheduler, ripgrep=rg)
    result = await tool.execute(
        GrepArgs(pattern="NEEDLE", ignore_case=True, limit=2), tool_runtime()
    )

    lines = result.text_content.splitlines()
    assert any(".hidden.txt:1:hidden needle" in line for line in lines)
    assert any("visible.txt:1:needle one" in line for line in lines)
    assert "needle two" not in result.text_content
    assert "needle three" not in result.text_content
    assert "ignored needle" not in result.text_content
    assert "limited to 2 matching lines" in result.text_content


async def test_grep_literal_flag_disables_regex_and_relative_paths_are_posix(
    tmp_path: Path,
) -> None:
    rg = _compatible_binary(tmp_path, "rg", "ripgrep 15.2.0")
    env, scheduler = _env(tmp_path)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "dots.txt").write_text("a.b literal\naxb\n", encoding="utf-8")
    tool = grep_tool(env, scheduler, ripgrep=rg)
    regex = await tool.execute(GrepArgs(pattern="a.b", path="sub"), tool_runtime())
    assert "axb" in regex.text_content
    literal = await tool.execute(GrepArgs(pattern="a.b", path="sub", literal=True), tool_runtime())
    assert "literal" in literal.text_content
    assert "axb" not in literal.text_content
    assert any(line.startswith("sub/dots.txt:") for line in literal.text_content.splitlines())


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan"), 2_147_484.0])
def test_bash_timeout_must_be_positive_finite_and_process_safe(timeout: float) -> None:
    with pytest.raises(ValueError):
        BashArgs(command="true", timeout_seconds=timeout)


@pytest.mark.asyncio
async def test_bash_does_not_inherit_seeded_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("DLIGHTRAG_STORAGE__POSTGRES__PASSWORD", "pw")
    env, scheduler = _env(tmp_path)
    tool = bash_tool(env, scheduler)
    result = await tool.execute(
        BashArgs(command="printf '%s' \"$OPENAI_API_KEY$DLIGHTRAG_STORAGE__POSTGRES__PASSWORD\""),
        tool_runtime(),
    )
    assert "sk-secret" not in result.text_content
    assert "pw" not in result.text_content
