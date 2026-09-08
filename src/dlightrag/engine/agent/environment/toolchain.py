# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Validated fd/ripgrep discovery with optional managed installation."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import io
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import urlsplit

SearchToolName = Literal["fd", "rg"]
Version = tuple[int, int, int]

FD_MIN_VERSION: Version = (10, 5, 0)
RG_MIN_VERSION: Version = (15, 2, 0)

_REPOSITORIES: dict[SearchToolName, str] = {
    "fd": "sharkdp/fd",
    "rg": "BurntSushi/ripgrep",
}
_MINIMUMS: dict[SearchToolName, Version] = {
    "fd": FD_MIN_VERSION,
    "rg": RG_MIN_VERSION,
}
_TOOL_NAMES: tuple[SearchToolName, ...] = ("fd", "rg")
_VERSION_RE = re.compile(r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?!\d)")
_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
_MAX_BINARY_BYTES = 128 * 1024 * 1024
_MAX_EXPANDED_ARCHIVE_BYTES = 256 * 1024 * 1024
logger = logging.getLogger(__name__)


class SearchToolUnavailable(RuntimeError):
    """A required search binary is absent, obsolete, or unverifiable."""


@dataclass(frozen=True, slots=True)
class SearchToolBinary:
    name: SearchToolName
    path: Path
    version: Version
    digest: str

    @property
    def version_text(self) -> str:
        return ".".join(str(part) for part in self.version)


class SearchToolchain:
    """Resolve and cache the two process-wide search executables."""

    def __init__(
        self,
        *,
        fd: str = "fd",
        ripgrep: str = "rg",
        cache_root: Path | None = None,
        auto_install: bool = False,
    ) -> None:
        self._commands: dict[SearchToolName, str] = {"fd": fd, "rg": ripgrep}
        self._cache_root = (cache_root or Path.home() / ".dlightrag" / "tools").expanduser()
        if not self._cache_root.is_absolute():
            raise ValueError("search tool cache root must be absolute")
        self._auto_install = auto_install
        self._resolved: dict[SearchToolName, SearchToolBinary] = {}
        self._lock = asyncio.Lock()

    async def ensure(self) -> dict[SearchToolName, SearchToolBinary]:
        async with self._lock:
            if len(self._resolved) != len(_MINIMUMS):
                self._resolved = await asyncio.to_thread(self._resolve_all)
            return dict(self._resolved)

    async def path(self, name: SearchToolName) -> str:
        async with self._lock:
            binary = self._resolved.get(name)
            if binary is None:
                binary = await asyncio.to_thread(self._resolve_one, name)
                self._resolved[name] = binary
            return str(binary.path)

    @property
    def provenance(self) -> dict[str, dict[str, str]]:
        return {
            name: {
                "path": str(binary.path),
                "version": binary.version_text,
                "sha256": binary.digest,
            }
            for name, binary in self._resolved.items()
        }

    def _resolve_all(self) -> dict[SearchToolName, SearchToolBinary]:
        resolved: dict[SearchToolName, SearchToolBinary] = {}
        for name in _TOOL_NAMES:
            resolved[name] = self._resolve_one(name)
        logger.info(
            "Agent search toolchain ready",
            extra={
                "search_toolchain": {
                    name: {
                        "path": str(item.path),
                        "version": item.version_text,
                        "sha256": item.digest,
                    }
                    for name, item in resolved.items()
                }
            },
        )
        return resolved

    def _resolve_one(self, name: SearchToolName) -> SearchToolBinary:
        minimum = _MINIMUMS[name]
        configured = _command_path(self._commands[name])
        configured_binary: SearchToolBinary | None = None
        if configured is not None:
            try:
                configured_binary = _inspect_binary(name, configured)
            except SearchToolUnavailable:
                configured_binary = None
            if configured_binary is not None and configured_binary.version >= minimum:
                return configured_binary

        cached = _best_cached(name, self._cache_root, minimum)
        if cached is not None:
            return cached

        if not self._auto_install:
            found = (
                f"{configured} is too old"
                if configured is not None
                else f"{self._commands[name]!r} was not found"
            )
            raise SearchToolUnavailable(
                f"{found}; {name}>={_format_version(minimum)} is required. "
                "Install a current binary or enable agent.search_tool_auto_install."
            )
        return _install_latest(name, self._cache_root, minimum)


def install_latest_search_tools(
    *,
    cache_root: Path,
    bin_dir: Path | None = None,
) -> dict[SearchToolName, SearchToolBinary]:
    """Install current verified releases; callers must enforce download consent."""
    cache_root = cache_root.expanduser()
    bin_dir = bin_dir.expanduser() if bin_dir is not None else None
    if not cache_root.is_absolute() or (bin_dir is not None and not bin_dir.is_absolute()):
        raise SearchToolUnavailable("search tool installation roots must be absolute")
    installed: dict[SearchToolName, SearchToolBinary] = {}
    for name in _TOOL_NAMES:
        installed[name] = _install_latest(name, cache_root, _MINIMUMS[name])
    if bin_dir is not None:
        missing_directories = _missing_directories(bin_dir)
        bin_dir.mkdir(parents=True, exist_ok=True)
        for name, binary in installed.items():
            destination = bin_dir / name
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{name}-", dir=bin_dir)
            temporary = Path(temporary_name)
            try:
                with binary.path.open("rb") as source, os.fdopen(descriptor, "wb") as target:
                    shutil.copyfileobj(source, target)
                    os.fchmod(target.fileno(), 0o755)
                    target.flush()
                    os.fsync(target.fileno())
                temporary.replace(destination)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        _fsync_directory(bin_dir)
        for directory in reversed(missing_directories):
            _fsync_directory(directory.parent)
    return installed


def _command_path(command: str) -> Path | None:
    candidate = Path(command).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        return candidate.resolve() if candidate.is_file() else None
    discovered = shutil.which(command)
    return Path(discovered).resolve() if discovered else None


def _inspect_binary(
    name: SearchToolName,
    path: Path,
    *,
    expected_digest: str | None = None,
) -> SearchToolBinary:
    try:
        if path.stat().st_size > _MAX_BINARY_BYTES:
            raise SearchToolUnavailable(f"{name} executable exceeded its size limit")
        digest = _sha256_file(path)
    except OSError as exc:
        raise SearchToolUnavailable(f"could not hash {name} at {path}") from exc
    if expected_digest is not None and digest != expected_digest:
        raise SearchToolUnavailable(f"cached {name} failed executable SHA-256 verification")
    try:
        # The executable path is operator-configured/PATH-resolved and argv is fixed.
        completed = subprocess.run(  # noqa: S603
            [str(path), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            env={"PATH": os.environ.get("PATH", ""), "LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SearchToolUnavailable(f"could not execute {name} at {path}") from exc
    first_line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    expected_name = "fd" if name == "fd" else "ripgrep"
    match = _VERSION_RE.search(first_line)
    if not first_line.startswith(f"{expected_name} ") or match is None:
        raise SearchToolUnavailable(f"could not parse {name} version from {path}")
    return SearchToolBinary(
        name=name,
        path=path,
        version=tuple(int(match.group(index)) for index in range(1, 4)),  # type: ignore[arg-type]
        digest=digest,
    )


def _best_cached(
    name: SearchToolName,
    cache_root: Path,
    minimum: Version,
) -> SearchToolBinary | None:
    root = cache_root / name
    if not root.is_dir():
        return None
    candidates: list[SearchToolBinary] = []
    for path in root.glob(f"*/{name}"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            directory_version = _parse_version(path.parent.name)
            expected_digest = _read_cached_digest(path)
            binary = _inspect_binary(name, path, expected_digest=expected_digest)
        except SearchToolUnavailable:
            continue
        if (
            path.parent.name == _format_version(directory_version)
            and binary.version == directory_version
            and binary.version >= minimum
        ):
            candidates.append(binary)
    return max(candidates, key=lambda item: item.version, default=None)


def _install_latest(
    name: SearchToolName,
    cache_root: Path,
    minimum: Version,
) -> SearchToolBinary:
    release = _github_json(f"https://api.github.com/repos/{_REPOSITORIES[name]}/releases/latest")
    tag = str(release.get("tag_name") or "")
    version = _parse_version(tag)
    if version < minimum:
        raise SearchToolUnavailable(
            f"latest {name} release {_format_version(version)} is below required "
            f"{_format_version(minimum)}"
        )
    asset_name = _release_asset_name(name, version)
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise SearchToolUnavailable(f"latest {name} release has no assets")
    asset = next(
        (item for item in assets if isinstance(item, dict) and item.get("name") == asset_name),
        None,
    )
    if asset is None:
        raise SearchToolUnavailable(f"latest {name} release has no {asset_name} asset")
    url = str(asset.get("browser_download_url") or "")
    digest = str(asset.get("digest") or "")
    if not url or not digest.startswith("sha256:"):
        raise SearchToolUnavailable(f"latest {name} asset lacks URL or SHA-256 provenance")
    archive = _download(url)
    actual = hashlib.sha256(archive).hexdigest()
    expected = digest.removeprefix("sha256:").lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise SearchToolUnavailable(f"latest {name} asset has an invalid SHA-256 digest")
    if actual != expected:
        raise SearchToolUnavailable(f"downloaded {name} archive failed SHA-256 verification")
    executable = _extract_binary(archive, name)

    destination_parent = cache_root / name
    destination_dir = destination_parent / _format_version(version)
    missing_directories = _missing_directories(destination_parent)
    destination_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{_format_version(version)}-",
            dir=destination_parent,
        )
    )
    temporary = staging / name
    digest_temporary = _cached_digest_path(temporary)
    try:
        with temporary.open("xb") as handle:
            handle.write(executable)
            os.fchmod(handle.fileno(), 0o755)
            handle.flush()
            os.fsync(handle.fileno())
        binary = _inspect_binary(name, temporary)
        if binary.version != version:
            raise SearchToolUnavailable(f"installed {name} version does not match its release")
        with digest_temporary.open("x", encoding="ascii", newline="\n") as handle:
            handle.write(f"{binary.digest}\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(staging)
        _remove_cache_entry(destination_dir)
        staging.replace(destination_dir)
        _fsync_directory(destination_parent)
        for directory in reversed(missing_directories):
            _fsync_directory(directory.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    destination = destination_dir / name
    return SearchToolBinary(
        name=name,
        path=destination,
        version=binary.version,
        digest=binary.digest,
    )


def _github_json(url: str) -> dict[str, object]:
    _require_https_url(url, hosts={"api.github.com"})
    request = urllib.request.Request(  # noqa: S310 - HTTPS/host constrained above
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "DlightRAG"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            payload = response.read(4 * 1024 * 1024 + 1)
    except OSError as exc:
        raise SearchToolUnavailable("could not query GitHub releases") from exc
    if len(payload) > 4 * 1024 * 1024:
        raise SearchToolUnavailable("GitHub release metadata exceeded its size limit")
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SearchToolUnavailable("GitHub release metadata is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise SearchToolUnavailable("GitHub release metadata is not an object")
    return decoded


def _download(url: str) -> bytes:
    _require_https_url(url, hosts={"github.com"})
    request = urllib.request.Request(  # noqa: S310 - HTTPS/host constrained above
        url, headers={"User-Agent": "DlightRAG"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            payload = response.read(_MAX_ARCHIVE_BYTES + 1)
    except OSError as exc:
        raise SearchToolUnavailable("could not download search tool release") from exc
    if len(payload) > _MAX_ARCHIVE_BYTES:
        raise SearchToolUnavailable("search tool archive exceeded its size limit")
    return payload


def _require_https_url(url: str, *, hosts: set[str]) -> None:
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError:
        port = -1
    if (
        parsed.scheme != "https"
        or parsed.hostname not in hosts
        or parsed.username is not None
        or port not in {None, 443}
    ):
        raise SearchToolUnavailable("search tool download URL is not an approved HTTPS origin")


def _extract_binary(archive: bytes, name: SearchToolName) -> bytes:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(archive), mode="rb") as compressed:
            expanded = compressed.read(_MAX_EXPANDED_ARCHIVE_BYTES + 1)
        if len(expanded) > _MAX_EXPANDED_ARCHIVE_BYTES:
            raise SearchToolUnavailable("search tool archive exceeded its expanded-size limit")
        # Parse the already bounded stream so GNU long-name/link and PAX metadata
        # cannot make tarfile decompress beyond the global ceiling internally.
        with tarfile.open(fileobj=io.BytesIO(expanded), mode="r:") as bundle:
            matches: list[tarfile.TarInfo] = []
            expanded_bytes = 0
            for index, member in enumerate(bundle):
                if index >= 10_000:
                    raise SearchToolUnavailable("search tool archive has too many members")
                expanded_bytes += member.size
                if expanded_bytes > _MAX_EXPANDED_ARCHIVE_BYTES:
                    raise SearchToolUnavailable(
                        "search tool archive exceeded its expanded-size limit"
                    )
                if member.isfile() and PurePosixPath(member.name).name == name:
                    matches.append(member)
            if len(matches) != 1 or matches[0].size > _MAX_BINARY_BYTES:
                raise SearchToolUnavailable(f"{name} archive has no unique bounded executable")
            source = bundle.extractfile(matches[0])
            if source is None:
                raise SearchToolUnavailable(f"could not read {name} from release archive")
            payload = source.read(_MAX_BINARY_BYTES + 1)
    except (EOFError, tarfile.TarError, OSError) as exc:
        raise SearchToolUnavailable(f"could not extract {name} release archive") from exc
    if len(payload) > _MAX_BINARY_BYTES:
        raise SearchToolUnavailable(f"{name} executable exceeded its size limit")
    return payload


def _release_asset_name(name: SearchToolName, version: Version) -> str:
    machine = platform.machine().lower()
    architecture = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "arm64": "aarch64",
        "aarch64": "aarch64",
    }.get(machine)
    system = platform.system().lower()
    target = {
        "linux": f"{architecture}-unknown-linux-gnu" if architecture else None,
        "darwin": f"{architecture}-apple-darwin" if architecture else None,
    }.get(system)
    if target is None:
        raise SearchToolUnavailable(
            f"automatic {name} installation does not support {system}/{machine}"
        )
    text = _format_version(version)
    prefix = "fd-v" if name == "fd" else "ripgrep-"
    return f"{prefix}{text}-{target}.tar.gz"


def _parse_version(value: str) -> Version:
    match = _VERSION_RE.search(value)
    if match is None:
        raise SearchToolUnavailable(f"could not parse release version {value!r}")
    return tuple(int(match.group(index)) for index in range(1, 4))  # type: ignore[return-value]


def _format_version(version: Version) -> str:
    return ".".join(str(part) for part in version)


def _cached_digest_path(binary: Path) -> Path:
    return binary.with_name(f"{binary.name}.sha256")


def _read_cached_digest(binary: Path) -> str:
    try:
        with _cached_digest_path(binary).open("r", encoding="ascii") as source:
            raw = source.read(66)
        if len(raw) > 65 or not raw.endswith("\n"):
            raise SearchToolUnavailable(f"cached {binary.name} has invalid SHA-256 provenance")
        value = raw[:-1]
    except (OSError, UnicodeError) as exc:
        raise SearchToolUnavailable(f"cached {binary.name} lacks SHA-256 provenance") from exc
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SearchToolUnavailable(f"cached {binary.name} has invalid SHA-256 provenance")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_cache_entry(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _missing_directories(path: Path) -> list[Path]:
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        if current == current.parent:
            break
        current = current.parent
    missing.reverse()
    return missing


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(prog="dlight-tools")
    parser.add_argument("command", choices=("install",))
    parser.add_argument(
        "--allow-runtime-download",
        action="store_true",
        help="explicitly permit downloading GitHub release assets",
    )
    parser.add_argument("--cache-root", type=Path, default=Path.home() / ".dlightrag" / "tools")
    parser.add_argument("--bin-dir", type=Path)
    args = parser.parse_args()
    if not args.allow_runtime_download:
        parser.error("install requires --allow-runtime-download")
    installed = install_latest_search_tools(cache_root=args.cache_root, bin_dir=args.bin_dir)
    print(
        json.dumps(
            {
                name: {
                    "path": str(binary.path),
                    "version": binary.version_text,
                    "sha256": binary.digest,
                }
                for name, binary in installed.items()
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "FD_MIN_VERSION",
    "RG_MIN_VERSION",
    "SearchToolBinary",
    "SearchToolUnavailable",
    "SearchToolchain",
    "install_latest_search_tools",
    "main",
]
