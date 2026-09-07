# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral checks for root plus independently installable Memory artifacts."""

import importlib
import io
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_VERSION = "2.0.0"
_BATTERIES = (
    "aiofiles>=24.1.0",
    "aiobotocore>=3.9.0",
    "anthropic>=1.0.0",
    "azure-storage-blob>=12.28.0",
    "botocore>=1.43.3",
    "google-genai>=2.19.0",
    "json-repair>=0.62.0",
    "lightrag-hku>=1.5.7",
    "lingua-language-detector>=2.2.0",
    "openai>=2.54.0",
)
_ROOT_REQUIRES = (f"dlightrag-memory=={_VERSION}", *_BATTERIES)


@pytest.mark.parametrize("package_name", ["dlightrag", "dlightrag_memory"])
def test_workspace_package_root_imports(package_name: str) -> None:
    importlib.import_module(package_name)


def _write_wheel(
    dist_dir: Path,
    *,
    distribution: str,
    package: str,
    requires: tuple[str, ...] = (),
    provides_extras: tuple[str, ...] = (),
    source: str = "",
    version: str = _VERSION,
    include_legal: bool = True,
    include_frontend: bool = False,
    additional_sources: dict[str, str] | None = None,
    sdist_source: str | None = None,
) -> None:
    wheel_name = distribution.replace("-", "_")
    dist_info = f"{wheel_name}-{version}.dist-info"
    metadata = [
        "Metadata-Version: 2.3",
        f"Name: {distribution}",
        f"Version: {version}",
        *(("License-File: LICENSE", "License-File: NOTICE") if include_legal else ()),
        *(f"Requires-Dist: {requirement}" for requirement in requires),
        *(f"Provides-Extra: {extra}" for extra in provides_extras),
        "",
    ]
    sources = additional_sources or {}
    with zipfile.ZipFile(dist_dir / f"{wheel_name}-{version}-py3-none-any.whl", "w") as wheel:
        wheel.writestr(f"{package}/__init__.py", source)
        for relative_path, content in sources.items():
            wheel.writestr(f"{package}/{relative_path}", content)
        wheel.writestr(f"{package}/py.typed", "")
        wheel.writestr(f"{dist_info}/METADATA", "\n".join(metadata))
        wheel.writestr(f"{dist_info}/RECORD", "")
        if include_legal:
            wheel.write(_REPO / "LICENSE", f"{dist_info}/licenses/LICENSE")
            wheel.write(_REPO / "NOTICE", f"{dist_info}/licenses/NOTICE")
        if include_frontend:
            wheel.writestr(
                f"{package}/adapters/http/browser/static/app/index.html", "<dl-app></dl-app>"
            )
            wheel.writestr(
                f"{package}/adapters/http/browser/static/app/login.html", "<form></form>"
            )
            wheel.writestr(
                f"{package}/adapters/http/browser/static/app/assets/style-test.css", "body {}"
            )
            wheel.writestr(
                f"{package}/adapters/http/browser/static/app/assets/app-test.js", "export {}"
            )
            wheel.writestr(
                f"{package}/adapters/http/browser/static/app/assets/theme-init-test.js", ""
            )

    sdist_root = f"{wheel_name}-{version}"
    packaged_source = source if sdist_source is None else sdist_source
    members = {
        f"{sdist_root}/PKG-INFO": "\n".join(metadata),
        f"{sdist_root}/src/{package}/__init__.py": packaged_source,
        **{
            f"{sdist_root}/src/{package}/{relative_path}": content
            for relative_path, content in sources.items()
        },
        f"{sdist_root}/src/{package}/py.typed": "",
    }
    if include_legal:
        members.update(
            {
                f"{sdist_root}/LICENSE": (_REPO / "LICENSE").read_text(encoding="utf-8"),
                f"{sdist_root}/NOTICE": (_REPO / "NOTICE").read_text(encoding="utf-8"),
            }
        )
    if include_frontend:
        members.update(
            {
                f"{sdist_root}/src/{package}/adapters/http/browser/static/app/index.html": "<dl-app></dl-app>",
                f"{sdist_root}/src/{package}/adapters/http/browser/static/app/login.html": "<form></form>",
                f"{sdist_root}/src/{package}/adapters/http/browser/static/app/assets/style-test.css": "body {}",
                f"{sdist_root}/src/{package}/adapters/http/browser/static/app/assets/app-test.js": "export {}",
                f"{sdist_root}/src/{package}/adapters/http/browser/static/app/assets/theme-init-test.js": "",
            }
        )
    with tarfile.open(dist_dir / f"{wheel_name}-{version}.tar.gz", "w:gz") as sdist:
        for name, content in members.items():
            payload = content.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            sdist.addfile(info, io.BytesIO(payload))


def _write_workspace_artifacts(
    tmp_path: Path,
    *,
    root_source: str = "",
    root_requires: tuple[str, ...] = _ROOT_REQUIRES,
    root_version: str = _VERSION,
    root_package: str = "dlightrag",
    root_additional_sources: dict[str, str] | None = None,
    root_sdist_source: str | None = None,
    root_include_legal: bool = True,
    root_include_model_catalog: bool = True,
    root_include_frontend: bool = True,
    root_extras: tuple[str, ...] = ("milvus",),
    memory_source: str = "",
    memory_sdist_source: str | None = None,
    memory_version: str = _VERSION,
) -> None:
    root_sources = dict(root_additional_sources or {})
    if root_include_model_catalog:
        root_sources["engine/ai/model_catalog.json"] = '{"revision":"test","models":[]}'
    _write_wheel(
        tmp_path,
        distribution="dlightrag",
        package=root_package,
        requires=root_requires,
        provides_extras=root_extras,
        source=root_source,
        sdist_source=root_sdist_source,
        version=root_version,
        include_legal=root_include_legal,
        include_frontend=root_include_frontend,
        additional_sources=root_sources,
    )
    _write_wheel(
        tmp_path,
        distribution="dlightrag-memory",
        package="dlightrag_memory",
        requires=("asyncpg>=0.31.0", "mcp>=2.0.0"),
        source=memory_source,
        sdist_source=memory_sdist_source,
        version=memory_version,
    )


def _verify_wheels(
    dist_dir: Path,
    *,
    config_path: Path = _REPO / "pyproject.toml",
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(_REPO / "scripts" / "verify_workspace_wheels.py"),
            "--config",
            str(config_path),
            "--workspace-root",
            str(_REPO),
            "--dist",
            str(dist_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_workspace_wheel_verifier_accepts_root_and_memory(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path)

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "memory_source",
    ["import fastapi\n", "import dlightrag\n", "from dlightrag.engine import ai\n"],
)
def test_workspace_wheel_verifier_rejects_memory_host_imports(
    tmp_path: Path, memory_source: str
) -> None:
    _write_workspace_artifacts(tmp_path, memory_source=memory_source)

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "forbidden import" in completed.stderr


def test_workspace_wheel_verifier_scans_sdist_sources(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path, memory_sdist_source="import fastapi\n")

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "forbidden import fastapi" in completed.stderr


@pytest.mark.parametrize(
    "rag_source",
    [
        "import lightrag.kg.postgres_impl\n",
        "from lightrag.kg import postgres_impl\n",
        "import importlib\nimportlib.import_module('lightrag.kg.postgres_impl')\n",
    ],
)
def test_workspace_wheel_verifier_rejects_concrete_rag_backend_imports(
    tmp_path: Path, rag_source: str
) -> None:
    _write_workspace_artifacts(
        tmp_path,
        root_additional_sources={"engine/rag/example.py": rag_source},
    )

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "forbidden import lightrag.kg.postgres_impl" in completed.stderr


def test_workspace_wheel_verifier_allows_offline_rebuild_composition(tmp_path: Path) -> None:
    _write_workspace_artifacts(
        tmp_path,
        root_additional_sources={
            "engine/rag/corpus/rebuild_bm25.py": (
                "from dlightrag.adapters.postgres.core._pool import pg_pool\n"
            )
        },
    )

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 0, completed.stderr


def test_workspace_wheel_verifier_rejects_answer_import_of_postgres_adapter(
    tmp_path: Path,
) -> None:
    _write_workspace_artifacts(
        tmp_path,
        root_additional_sources={
            "engine/answer/example.py": "from dlightrag.adapters.postgres import answer_runs\n"
        },
    )

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "forbidden import dlightrag.adapters.postgres" in completed.stderr


@pytest.mark.parametrize(
    "config_source",
    ["import dlightrag.engine.answer\n", "from dlightrag.engine import answer\n"],
)
def test_workspace_wheel_verifier_enforces_root_source_contract(
    tmp_path: Path, config_source: str
) -> None:
    _write_workspace_artifacts(
        tmp_path,
        root_additional_sources={"application/config/probe.py": config_source},
    )

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert (
        "forbidden import dlightrag.engine.answer in dlightrag/application/config/probe.py"
        in completed.stderr
    )


def test_workspace_wheel_verifier_rejects_mismatched_sdist_set(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path)
    (tmp_path / f"dlightrag_memory-{_VERSION}.tar.gz").rename(
        tmp_path / f"unexpected-{_VERSION}.tar.gz"
    )

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "sdist set does not match" in completed.stderr


def test_workspace_wheel_verifier_rejects_version_drift(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path, memory_version="2.0.1")

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "versions are not lockstep" in completed.stderr


def test_workspace_wheel_verifier_rejects_wrong_top_level_package(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path, root_package="wrong_package")

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "expected top-level package" in completed.stderr


@pytest.mark.parametrize("junk_path", [".DS_Store", "__pycache__/probe.pyc", "probe.pyo"])
def test_workspace_wheel_verifier_rejects_generated_junk(tmp_path: Path, junk_path: str) -> None:
    _write_workspace_artifacts(tmp_path, root_additional_sources={junk_path: "junk"})

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "contains forbidden generated files" in completed.stderr


def test_workspace_wheel_verifier_requires_memory_dependency(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path, root_requires=_BATTERIES)

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "expected DlightRAG dependencies" in completed.stderr


def test_workspace_wheel_verifier_requires_lockstep_memory_pin(tmp_path: Path) -> None:
    _write_workspace_artifacts(
        tmp_path,
        root_requires=("dlightrag-memory>=2.0.0", *_BATTERIES),
    )

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "dependency must be pinned as dlightrag-memory==2.0.0" in completed.stderr


def test_workspace_wheel_verifier_requires_batteries_in_wheel_metadata(tmp_path: Path) -> None:
    _write_workspace_artifacts(
        tmp_path,
        root_requires=tuple(item for item in _ROOT_REQUIRES if not item.startswith("anthropic")),
    )

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "missing batteries-included dependencies ['anthropic']" in completed.stderr


def test_workspace_wheel_verifier_rejects_corrupt_sdist(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path)
    (tmp_path / f"dlightrag_memory-{_VERSION}.tar.gz").write_bytes(b"not a tar archive")

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "workspace wheel verification failed" in completed.stderr


def test_workspace_wheel_verifier_requires_legal_files(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path, root_include_legal=False)

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "LICENSE and NOTICE" in completed.stderr


def test_workspace_wheel_verifier_requires_root_model_catalog(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path, root_include_model_catalog=False)

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "model_catalog.json" in completed.stderr


def test_workspace_wheel_verifier_rejects_unexpected_root_extras(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path, root_extras=("all",))

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "expected extras" in completed.stderr


def test_workspace_wheel_verifier_requires_root_frontend(tmp_path: Path) -> None:
    _write_workspace_artifacts(tmp_path, root_include_frontend=False)

    completed = _verify_wheels(tmp_path)

    assert completed.returncode == 1
    assert "generated frontend" in completed.stderr
