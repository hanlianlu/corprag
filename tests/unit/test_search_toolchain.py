# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Versioned fd/ripgrep discovery and verified managed installation."""

import gzip
import hashlib
import inspect
import io
import tarfile
from pathlib import Path

import pytest

from dlightrag.engine.agent.environment import toolchain
from dlightrag.engine.agent.environment.toolchain import (
    SearchToolchain,
    SearchToolUnavailable,
)


def _binary(path: Path, version_output: str, *, managed: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\nprintf '%s\\n' '{version_output}'\n", encoding="utf-8")
    path.chmod(0o755)
    if managed:
        path.with_suffix(".sha256").write_text(
            hashlib.sha256(path.read_bytes()).hexdigest() + "\n",
            encoding="ascii",
        )
    return path


async def test_configured_binary_wins_and_records_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = _binary(tmp_path / "configured-fd", "fd 10.5.0")
    path_binary = _binary(tmp_path / "path" / "fd", "fd 99.0.0")
    monkeypatch.setenv("PATH", str(path_binary.parent))
    resolver = SearchToolchain(fd=str(configured), cache_root=tmp_path / "cache")

    resolved = await resolver.path("fd")

    assert resolved == str(configured.resolve())
    assert resolver.provenance["fd"] == {
        "path": str(configured.resolve()),
        "version": "10.5.0",
        "sha256": hashlib.sha256(configured.read_bytes()).hexdigest(),
    }


async def test_path_then_managed_cache_and_minimum_version(tmp_path: Path) -> None:
    old = _binary(tmp_path / "path" / "fd", "fd 10.4.2")
    cached = _binary(
        tmp_path / "cache" / "fd" / "10.5.1" / "fd",
        "fd 10.5.1",
        managed=True,
    )
    resolver = SearchToolchain(
        fd=str(old),
        cache_root=tmp_path / "cache",
    )

    assert await resolver.path("fd") == str(cached.resolve())

    unavailable = SearchToolchain(fd=str(old), cache_root=tmp_path / "empty")
    with pytest.raises(SearchToolUnavailable, match=r"fd>=10\.5\.0"):
        await unavailable.path("fd")


@pytest.mark.parametrize(
    ("system", "machine", "target"),
    [
        ("Linux", "x86_64", "x86_64-unknown-linux-gnu"),
        ("Linux", "aarch64", "aarch64-unknown-linux-gnu"),
        ("Darwin", "x86_64", "x86_64-apple-darwin"),
        ("Darwin", "arm64", "aarch64-apple-darwin"),
    ],
)
def test_release_asset_names_cover_supported_platforms(
    system: str,
    machine: str,
    target: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(toolchain.platform, "system", lambda: system)
    monkeypatch.setattr(toolchain.platform, "machine", lambda: machine)

    assert toolchain._release_asset_name("fd", (10, 5, 0)) == f"fd-v10.5.0-{target}.tar.gz"
    assert toolchain._release_asset_name("rg", (15, 2, 0)) == f"ripgrep-15.2.0-{target}.tar.gz"


def test_managed_cache_digest_is_checked_before_execution(tmp_path: Path) -> None:
    cached = _binary(
        tmp_path / "cache" / "fd" / "10.5.1" / "fd",
        "fd 10.5.1",
        managed=True,
    )
    marker = tmp_path / "executed"
    cached.write_text(
        f"#!/bin/sh\ntouch {marker!s}\nprintf '%s\\n' 'fd 10.5.1'\n",
        encoding="utf-8",
    )
    cached.chmod(0o755)

    assert toolchain._best_cached("fd", tmp_path / "cache", (10, 5, 0)) is None
    assert not marker.exists()


def test_archive_digest_mismatch_fails_before_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = {
        "tag_name": "v10.5.0",
        "assets": [
            {
                "name": toolchain._release_asset_name("fd", (10, 5, 0)),
                "browser_download_url": "https://example.invalid/fd.tar.gz",
                "digest": "sha256:" + "0" * 64,
            }
        ],
    }
    monkeypatch.setattr(toolchain, "_github_json", lambda _url: release)
    monkeypatch.setattr(toolchain, "_download", lambda _url: b"not-the-declared-archive")

    with pytest.raises(SearchToolUnavailable, match="SHA-256"):
        toolchain._install_latest("fd", tmp_path, (10, 5, 0))
    assert list(tmp_path.rglob("fd")) == []


def test_install_publishes_binary_and_digest_as_one_cache_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = b"verified archive"
    release = {
        "tag_name": "v10.5.0",
        "assets": [
            {
                "name": toolchain._release_asset_name("fd", (10, 5, 0)),
                "browser_download_url": "https://github.com/fd.tar.gz",
                "digest": "sha256:" + hashlib.sha256(archive).hexdigest(),
            }
        ],
    }
    executable = b"#!/bin/sh\nprintf '%s\\n' 'fd 10.5.0'\n"
    monkeypatch.setattr(toolchain, "_github_json", lambda _url: release)
    monkeypatch.setattr(toolchain, "_download", lambda _url: archive)
    monkeypatch.setattr(toolchain, "_extract_binary", lambda _archive, _name: executable)

    installed = toolchain._install_latest("fd", tmp_path, (10, 5, 0))

    digest_path = installed.path.with_suffix(".sha256")
    digest_text = digest_path.read_text(encoding="ascii")
    assert digest_text.strip() == hashlib.sha256(executable).hexdigest()
    assert toolchain._best_cached("fd", tmp_path, (10, 5, 0)) == installed


def test_install_validates_temporary_binary_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = b"verified archive"
    release = {
        "tag_name": "v10.5.0",
        "assets": [
            {
                "name": toolchain._release_asset_name("fd", (10, 5, 0)),
                "browser_download_url": "https://github.com/fd.tar.gz",
                "digest": "sha256:" + hashlib.sha256(archive).hexdigest(),
            }
        ],
    }
    monkeypatch.setattr(toolchain, "_github_json", lambda _url: release)
    monkeypatch.setattr(toolchain, "_download", lambda _url: archive)
    monkeypatch.setattr(toolchain, "_extract_binary", lambda _archive, _name: b"not executable")

    with pytest.raises(SearchToolUnavailable, match="could not execute"):
        toolchain._install_latest("fd", tmp_path, (10, 5, 0))

    assert not (tmp_path / "fd" / "10.5.0" / "fd").exists()


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/release",
        "https://github.com.evil.invalid/release",
        "https://user@github.com/release",
        "https://github.com:8443/release",
    ],
)
def test_download_origins_are_https_and_allowlisted(url: str) -> None:
    with pytest.raises(SearchToolUnavailable, match="approved HTTPS origin"):
        toolchain._require_https_url(url, hosts={"github.com"})


def test_truncated_gzip_is_reported_as_unavailable() -> None:
    truncated = gzip.compress(b"not a tar stream")[:-4]

    with pytest.raises(SearchToolUnavailable, match="could not extract"):
        toolchain._extract_binary(truncated, "fd")


def test_extract_rejects_excessive_total_expanded_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(toolchain, "_MAX_EXPANDED_ARCHIVE_BYTES", 10)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
        info = tarfile.TarInfo("large-unrelated-file")
        info.size = 11
        bundle.addfile(info, io.BytesIO(b"x" * 11))

    with pytest.raises(SearchToolUnavailable, match="expanded-size"):
        toolchain._extract_binary(archive.getvalue(), "fd")


def test_extracts_only_the_unique_bounded_executable() -> None:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
        payload = b"#!/bin/sh\n"
        info = tarfile.TarInfo("fd-v10.5.0-target/fd")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))

    assert toolchain._extract_binary(archive.getvalue(), "fd") == b"#!/bin/sh\n"


def test_cli_requires_explicit_download_consent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["dlight-tools", "install"])
    with pytest.raises(SystemExit, match="2"):
        toolchain.main()


def test_sync_install_function_is_not_accidentally_async() -> None:
    assert not inspect.iscoroutinefunction(toolchain.install_latest_search_tools)
