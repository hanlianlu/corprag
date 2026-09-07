# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Verify DlightRAG workspace wheel ownership and dependency direction."""

from __future__ import annotations

import argparse
import ast
import email.parser
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

_EXPECTED_PACKAGES = {
    "dlightrag": "dlightrag",
    "dlightrag-memory": "dlightrag_memory",
}
_EXPECTED_DLIGHTRAG_DEPENDENCIES = {
    "dlightrag": {"dlightrag-memory"},
    "dlightrag-memory": set(),
}
_EXPECTED_EXTRAS = {
    "dlightrag": {"milvus"},
    "dlightrag-memory": set(),
}
_REQUIRED_ROOT_DEPENDENCIES = frozenset(
    {
        "aiofiles",
        "aiobotocore",
        "anthropic",
        "azure-storage-blob",
        "botocore",
        "google-genai",
        "json-repair",
        "lightrag-hku",
        "lingua-language-detector",
        "openai",
    }
)
_WORKSPACE_MANIFESTS = {
    "dlightrag": (Path("pyproject.toml"), ".", "src/dlightrag"),
    "dlightrag-memory": (
        Path("packages/memory/pyproject.toml"),
        "packages/memory",
        "src/dlightrag_memory",
    ),
}
_EXPECTED_WORKSPACE_MEMBERS = ["packages/memory"]
_EXPECTED_WORKSPACE_SOURCES = {"dlightrag-memory": {"workspace": True}}
_ROOT_CONSOLE_SCRIPTS = (
    "dlightrag-api",
    "dlightrag-mcp",
    "dlightrag-rebuild-bm25",
    "dlightrag-rebuild-vdb",
)
_CONCRETE_LIGHTRAG_BACKEND = "lightrag.kg.postgres_impl"
# import-linter rejects external submodules as contract targets, so the built
# artifact gate owns this one exact LightRAG implementation prohibition.
_SPECIFIC_SOURCE_PROHIBITIONS = {
    "dlightrag": {"dlightrag.engine.rag": (_CONCRETE_LIGHTRAG_BACKEND,)},
}
_REQUIRED_EXTERNAL_PROHIBITIONS = {
    "dlightrag": set(),
    "dlightrag-memory": {"lightrag", "fastapi"},
}
_DLIGHTRAG_DISTRIBUTIONS = frozenset(_EXPECTED_PACKAGES)
_NORMALIZE_RE = re.compile(r"[-_.]+")

_ABSENT_HELPER = """
def absent(name):
    try:
        return importlib.util.find_spec(name) is None
    except ModuleNotFoundError:
        return True
"""

_MEMORY_SMOKE = (
    """
import asyncio
import importlib
import pkgutil
import dlightrag_memory
from dlightrag_memory import Memory, MemoryProvenance
from dlightrag_memory.store import InMemoryMemoryStore

for module in pkgutil.walk_packages(dlightrag_memory.__path__, prefix='dlightrag_memory.'):
    importlib.import_module(module.name)

async def main():
    store = InMemoryMemoryStore()
    provenance = MemoryProvenance(
        origin_kind='answer_run',
        origin_id='11111111-1111-1111-1111-111111111111',
        run_id='11111111-1111-1111-1111-111111111111',
        session_id='11111111-1111-1111-1111-111111111111',
    )
    memory = Memory(store)
    receipt = await memory.remember(
        owner_id='owner-1',
        kind='preference',
        body='Installed memory works.',
        provenance=provenance,
        idempotency_key='installed-wheel-proposal',
    )
    replay = await memory.remember(
        owner_id='owner-1',
        kind='preference',
        body='Installed memory works.',
        provenance=provenance,
        idempotency_key='installed-wheel-proposal',
    )
    assert receipt.outcome == 'changed' and replay == receipt
    records, _ = await Memory(store).browse(owner_id='owner-1', limit=100)
    assert [item.body for item in records] == ['Installed memory works.']
    recalled = await memory.recall(owner_id='owner-1', query='memory works')
    assert recalled.records
    forgotten = await memory.forget(
        owner_id='owner-1',
        memory_id=receipt.memory_id,
        provenance=provenance,
        idempotency_key='installed-wheel-forget',
    )
    assert forgotten.outcome == 'changed'
    records, _ = await memory.browse(owner_id='owner-1', limit=100)
    assert records == ()
    server = dlightrag_memory.mcp_server.build_memory_server(memory, subject='owner-1')
    assert {tool.name for tool in await server.list_tools()} == {
        'memory_recall', 'memory_remember', 'memory_forget', 'memory_undo'
    }

asyncio.run(main())
"""
    + _ABSENT_HELPER
    + """
assert all(absent(name) for name in (
    'dlightrag', 'dlightrag.engine.agent', 'dlightrag.engine.ai', 'dlightrag.engine.rag',
    'lightrag', 'fastapi', 'openai', 'anthropic', 'google.genai'
))
"""
)


@dataclass(frozen=True, slots=True)
class WheelFacts:
    distribution: str
    version: str
    dependencies: frozenset[str]
    requirements: tuple[str, ...]
    extras: frozenset[str]
    top_level_packages: frozenset[str]
    license_files: frozenset[str]
    legal_hashes: tuple[str, str]
    has_py_typed: bool
    has_frontend: bool
    has_model_catalog: bool


@dataclass(frozen=True, slots=True)
class SdistFacts:
    distribution: str
    version: str
    dependencies: frozenset[str]
    requirements: tuple[str, ...]
    extras: frozenset[str]
    top_level_packages: frozenset[str]
    license_files: frozenset[str]
    legal_hashes: tuple[str, str]
    has_py_typed: bool
    has_frontend: bool
    has_model_catalog: bool


@dataclass(frozen=True, slots=True)
class ImportRule:
    source: str
    forbidden: tuple[str, ...]
    ignored: tuple[tuple[str, str], ...] = ()


def _normalize_distribution(value: str) -> str:
    return _NORMALIZE_RE.sub("-", value).lower()


def _requirement_name(value: str) -> str:
    name = re.split(r"[\s<>=!~;\[(]", value, maxsplit=1)[0]
    return _normalize_distribution(name)


def _metadata_facts(
    raw: bytes,
    *,
    artifact: str,
) -> tuple[str, str, tuple[str, ...], frozenset[str], frozenset[str]]:
    metadata = email.parser.Parser().parsestr(raw.decode("utf-8"))
    distribution = _normalize_distribution(str(metadata.get("Name") or ""))
    version = str(metadata.get("Version") or "")
    requirements = tuple(metadata.get_all("Requires-Dist", []))
    extras = frozenset(metadata.get_all("Provides-Extra", []))
    license_files = frozenset(metadata.get_all("License-File", []))
    if not distribution or not version:
        raise ValueError(f"{artifact}: missing distribution name or version")
    return distribution, version, requirements, extras, license_files


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _required_member_hashes(
    members: dict[str, bytes],
    *,
    artifact: str,
) -> tuple[str, str]:
    try:
        return _sha256(members["LICENSE"]), _sha256(members["NOTICE"])
    except KeyError as exc:
        raise ValueError(f"{artifact}: must contain LICENSE and NOTICE") from exc


def _reject_forbidden_members(names: Iterable[str], *, artifact: str) -> None:
    forbidden = sorted(
        name
        for name in names
        if ".DS_Store" in Path(name).parts
        or "__pycache__" in Path(name).parts
        or Path(name).suffix in {".pyc", ".pyo"}
    )
    if forbidden:
        raise ValueError(f"{artifact}: contains forbidden generated files {forbidden}")


def _top_level_from_wheel(names: list[str]) -> frozenset[str]:
    top_level: set[str] = set()
    for name in names:
        if name.startswith((".", "/")) or ".dist-info/" in name or ".data/" in name:
            continue
        if "/" in name:
            top_level.add(name.split("/", 1)[0])
        elif name.endswith(".py"):
            top_level.add(Path(name).stem)
    return frozenset(top_level)


def _has_vite_frontend(members: set[str], *, prefix: str) -> bool:
    assets = f"{prefix}/assets/"
    return (
        {f"{prefix}/index.html", f"{prefix}/login.html"}.issubset(members)
        and any(name.startswith(assets + "app-") and name.endswith(".js") for name in members)
        and any(
            name.startswith(assets + "theme-init-") and name.endswith(".js") for name in members
        )
        and any(name.startswith(assets + "style-") and name.endswith(".css") for name in members)
    )


def _wheel_facts(
    path: Path,
    *,
    rules_by_distribution: dict[str, tuple[ImportRule, ...]],
) -> WheelFacts:
    with zipfile.ZipFile(path) as wheel:
        member_names = wheel.namelist()
        _reject_forbidden_members(member_names, artifact=path.name)
        metadata_paths = [name for name in member_names if name.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError(f"{path.name}: expected one METADATA file")
        distribution, version, requirements, extras, license_files = _metadata_facts(
            wheel.read(metadata_paths[0]), artifact=path.name
        )
        dependencies = frozenset(_requirement_name(requirement) for requirement in requirements)
        top_level = _top_level_from_wheel(wheel.namelist())
        legal_members = {
            Path(name).name: wheel.read(name)
            for name in wheel.namelist()
            if ".dist-info/licenses/" in name and Path(name).name in {"LICENSE", "NOTICE"}
        }
        legal_hashes = _required_member_hashes(legal_members, artifact=path.name)
        expected_package = _EXPECTED_PACKAGES.get(distribution, "")
        has_py_typed = f"{expected_package}/py.typed" in wheel.namelist()
        has_frontend = _has_vite_frontend(
            set(wheel.namelist()),
            prefix="dlightrag/adapters/http/browser/static/app",
        )
        has_model_catalog = "dlightrag/engine/ai/model_catalog.json" in wheel.namelist()
        sources = (
            (name, wheel.read(name))
            for name in wheel.namelist()
            if name.endswith(".py") and ".dist-info/" not in name and ".data/" not in name
        )
        _validate_source_imports(
            sources,
            path=path,
            distribution=distribution,
            rules=rules_by_distribution.get(distribution, ()),
        )
    return WheelFacts(
        distribution,
        version,
        dependencies,
        requirements,
        extras,
        top_level,
        license_files,
        legal_hashes,
        has_py_typed,
        has_frontend,
        has_model_catalog,
    )


def _source_module(name: str) -> tuple[str, bool]:
    module_parts = list(Path(name).with_suffix("").parts)
    is_package = module_parts[-1] == "__init__"
    if is_package:
        module_parts.pop()
    return ".".join(module_parts), is_package


def _import_candidates(
    node: ast.Import | ast.ImportFrom,
    *,
    module: str,
    is_package: bool,
) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)

    if node.level == 0:
        base = node.module or ""
    else:
        package = module if is_package else module.rpartition(".")[0]
        try:
            base = importlib.util.resolve_name(f"{'.' * node.level}{node.module or ''}", package)
        except ImportError:
            return ()
    if not base:
        return ()
    imported_members = tuple(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
    return (base, *imported_members)


def _dynamic_import_candidates(
    node: ast.AST,
    *,
    static_strings: tuple[str, ...],
) -> tuple[str, ...]:
    if not isinstance(node, ast.Call) or not node.args:
        return ()
    function_name = (
        node.func.attr
        if isinstance(node.func, ast.Attribute)
        else node.func.id
        if isinstance(node.func, ast.Name)
        else ""
    )
    is_import_module = function_name == "import_module" or function_name == "__import__"
    if not is_import_module:
        return ()

    first_arg = node.args[0]
    values = (
        (first_arg.value,)
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str)
        else static_strings
    )
    candidates = set(values)
    if function_name == "import_module" and any(value.startswith(".") for value in values):
        package_arg = (
            node.args[1]
            if len(node.args) > 1
            else next(
                (keyword.value for keyword in node.keywords if keyword.arg == "package"),
                None,
            )
        )
        packages = (
            (package_arg.value,)
            if isinstance(package_arg, ast.Constant) and isinstance(package_arg.value, str)
            else static_strings
        )
        for value in values:
            if not value.startswith("."):
                continue
            for package in packages:
                if package.startswith("."):
                    continue
                try:
                    candidates.add(importlib.util.resolve_name(value, package))
                except ImportError:
                    continue
    return tuple(sorted(candidates))


def _forbidden_import(
    imported: str,
    rules: tuple[ImportRule, ...],
    *,
    source_module: str,
) -> bool:
    return any(
        (imported == prefix or imported.startswith(f"{prefix}."))
        and not any(
            source_module == ignored_source
            and (imported == ignored_target or imported.startswith(f"{ignored_target}."))
            for ignored_source, ignored_target in rule.ignored
        )
        for rule in rules
        for prefix in rule.forbidden
    )


def _validate_source_imports(
    sources: Iterable[tuple[str, bytes]],
    *,
    path: Path,
    distribution: str,
    rules: tuple[ImportRule, ...],
) -> None:
    for name, raw in sources:
        module, is_package = _source_module(name)
        applicable_rules = tuple(
            rule for rule in rules if module == rule.source or module.startswith(f"{rule.source}.")
        )
        tree = ast.parse(raw, filename=f"{path.name}:{name}")
        static_strings = tuple(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
        for node in ast.walk(tree):
            imports = (
                _import_candidates(node, module=module, is_package=is_package)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                else ()
            )
            for imported in imports:
                if _forbidden_import(imported, applicable_rules, source_module=module):
                    raise ValueError(f"{distribution}: forbidden import {imported} in {name}")
            for imported in _dynamic_import_candidates(node, static_strings=static_strings):
                if _forbidden_import(imported, applicable_rules, source_module=module):
                    raise ValueError(f"{distribution}: forbidden import {imported} in {name}")


def _import_rules(config_path: Path) -> dict[str, tuple[ImportRule, ...]]:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    contracts = config["tool"]["importlinter"]["contracts"]
    rules_by_root: dict[str, list[ImportRule]] = {root: [] for root in _EXPECTED_PACKAGES.values()}
    for contract in contracts:
        if contract.get("type") != "forbidden":
            continue
        forbidden = tuple(contract.get("forbidden_modules") or ())
        ignored = tuple(
            tuple(part.strip() for part in value.split("->", maxsplit=1))
            for value in contract.get("ignore_imports") or ()
        )
        sources = contract.get("source_modules") or ()
        for source in sources:
            for root in rules_by_root:
                if source == root or source.startswith(f"{root}."):
                    rules_by_root[root].append(ImportRule(source, forbidden, ignored))
                    break

    for distribution, source_rules in _SPECIFIC_SOURCE_PROHIBITIONS.items():
        root = _EXPECTED_PACKAGES[distribution]
        rules_by_root[root].extend(
            ImportRule(source, forbidden) for source, forbidden in source_rules.items()
        )

    rules_by_distribution = {
        distribution: tuple(rules_by_root[root])
        for distribution, root in _EXPECTED_PACKAGES.items()
    }
    all_roots = set(_EXPECTED_PACKAGES.values())
    for distribution, root in _EXPECTED_PACKAGES.items():
        allowed_roots = {
            _EXPECTED_PACKAGES[dependency]
            for dependency in _EXPECTED_DLIGHTRAG_DEPENDENCIES[distribution]
        }
        required = (
            all_roots - {root} - allowed_roots | _REQUIRED_EXTERNAL_PROHIBITIONS[distribution]
        )
        root_forbidden = {
            forbidden
            for rule in rules_by_distribution[distribution]
            if rule.source == root
            for forbidden in rule.forbidden
        }
        if missing := required - root_forbidden:
            raise ValueError(
                f"{distribution}: import policy is missing prohibitions {sorted(missing)}"
            )
    return rules_by_distribution


def _sdist_facts(
    path: Path,
    *,
    rules_by_distribution: dict[str, tuple[ImportRule, ...]],
) -> SdistFacts:
    with tarfile.open(path, "r:gz") as sdist:
        members = [member for member in sdist.getmembers() if member.isfile()]
        _reject_forbidden_members((member.name for member in members), artifact=path.name)
        metadata_members = [member for member in members if member.name.endswith("/PKG-INFO")]
        if len(metadata_members) != 1:
            raise ValueError(f"{path.name}: expected one PKG-INFO file")
        metadata_file = sdist.extractfile(metadata_members[0])
        if metadata_file is None:
            raise ValueError(f"{path.name}: could not read PKG-INFO")
        distribution, version, requirements, extras, license_files = _metadata_facts(
            metadata_file.read(), artifact=path.name
        )
        sdist_root = Path(metadata_members[0].name).parts[0]
        expected_py_typed = f"{sdist_root}/src/{_EXPECTED_PACKAGES.get(distribution, '')}/py.typed"

        top_level: set[str] = set()
        legal_members: dict[str, bytes] = {}
        has_py_typed = False
        frontend_members: set[str] = set()
        has_model_catalog = False
        sources: list[tuple[str, bytes]] = []
        for member in members:
            parts = Path(member.name).parts
            if len(parts) >= 4 and parts[1] == "src" and member.name.endswith(".py"):
                top_level.add(parts[2])
            elif len(parts) >= 3 and member.name.endswith(".py"):
                top_level.add(parts[1])
            if Path(member.name).name in {"LICENSE", "NOTICE"} and len(parts) == 2:
                legal_file = sdist.extractfile(member)
                if legal_file is not None:
                    legal_members[Path(member.name).name] = legal_file.read()
            if member.name == expected_py_typed:
                has_py_typed = True
            if member.name == f"{sdist_root}/src/dlightrag/engine/ai/model_catalog.json":
                has_model_catalog = True
            if len(parts) > 1:
                frontend_members.add("/".join(parts[1:]))
            relative_parts = parts[1:]
            if relative_parts[:1] == ("src",):
                relative_parts = relative_parts[1:]
            if relative_parts[:1] == (
                _EXPECTED_PACKAGES.get(distribution, ""),
            ) and member.name.endswith(".py"):
                source_file = sdist.extractfile(member)
                if source_file is not None:
                    sources.append(("/".join(relative_parts), source_file.read()))
        legal_hashes = _required_member_hashes(legal_members, artifact=path.name)
        _validate_source_imports(
            sources,
            path=path,
            distribution=distribution,
            rules=rules_by_distribution.get(distribution, ()),
        )
    return SdistFacts(
        distribution,
        version,
        frozenset(_requirement_name(item) for item in requirements),
        requirements,
        extras,
        frozenset(top_level),
        license_files,
        legal_hashes,
        has_py_typed,
        _has_vite_frontend(
            frontend_members,
            prefix="src/dlightrag/adapters/http/browser/static/app",
        ),
        has_model_catalog,
    )


def verify_workspace_definition(workspace_root: Path) -> None:
    """Verify manifests and uv.lock describe the root-plus-Memory workspace."""
    workspace_root = workspace_root.resolve()
    uv_executable = shutil.which("uv")
    if uv_executable is None:
        raise ValueError("uv is required to verify uv.lock")
    lock_check = subprocess.run(  # noqa: S603 - fixed uv command against caller-selected root
        [uv_executable, "lock", "--check"],
        cwd=workspace_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if lock_check.returncode != 0:
        detail = lock_check.stderr.strip() or lock_check.stdout.strip()
        raise ValueError(f"uv.lock is stale: {detail}")
    configs: dict[str, dict[str, object]] = {}
    versions: set[str] = set()
    direct_dependencies: dict[str, set[str]] = {}

    for distribution, (relative_path, _, _) in _WORKSPACE_MANIFESTS.items():
        manifest_path = workspace_root / relative_path
        try:
            config = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
            project = config["project"]
            build_system = config["build-system"]
            uv_build = config["tool"]["uv"]["build-backend"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{relative_path}: incomplete workspace manifest") from exc
        if _normalize_distribution(str(project.get("name") or "")) != distribution:
            raise ValueError(f"{relative_path}: unexpected project name")
        version = str(project.get("version") or "")
        versions.add(version)
        if project.get("requires-python") != ">=3.14":
            raise ValueError(f"{distribution}: Python requirement must be >=3.14")
        if project.get("license") != "Apache-2.0":
            raise ValueError(f"{distribution}: license must be Apache-2.0")
        if build_system.get("build-backend") != "uv_build":
            raise ValueError(f"{distribution}: build backend must be uv_build")
        if build_system.get("requires") != ["uv_build>=0.11.21"]:
            raise ValueError(f"{distribution}: uv_build requirement must be >=0.11.21")
        if uv_build.get("source-exclude") != ["**/.DS_Store"]:
            raise ValueError(f"{distribution}: source build must exclude .DS_Store")
        expected_source_include = (
            ["compose.yaml", "examples/**"] if distribution == "dlightrag-memory" else []
        )
        if uv_build.get("source-include", []) != expected_source_include:
            raise ValueError(f"{distribution}: unexpected source distribution includes")
        dependencies = project.get("dependencies")
        if not isinstance(dependencies, list) or not all(
            isinstance(dependency, str) for dependency in dependencies
        ):
            raise ValueError(f"{distribution}: project dependencies must be strings")
        direct_dependencies[distribution] = {
            _requirement_name(dependency) for dependency in dependencies
        }
        configs[distribution] = config

    if len(versions) != 1 or not next(iter(versions), ""):
        raise ValueError(f"workspace manifest versions are not lockstep: {sorted(versions)}")
    (version,) = versions

    try:
        root_uv = configs["dlightrag"]["tool"]["uv"]  # type: ignore[index]
    except (KeyError, TypeError) as exc:
        raise ValueError("root manifest is missing [tool.uv]") from exc
    if root_uv.get("workspace", {}).get("members") != _EXPECTED_WORKSPACE_MEMBERS:
        raise ValueError("root workspace members differ from the root-plus-Memory contract")
    if root_uv.get("sources") != _EXPECTED_WORKSPACE_SOURCES:
        raise ValueError("root workspace sources differ from the root-plus-Memory contract")

    lock_path = workspace_root / "uv.lock"
    try:
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
        lock_packages = lock["package"]
    except (KeyError, TypeError) as exc:
        raise ValueError("uv.lock is missing package records") from exc
    if lock.get("requires-python") != ">=3.14":
        raise ValueError("uv.lock Python requirement differs from workspace manifests")
    if lock.get("manifest", {}).get("members") != list(_EXPECTED_PACKAGES):
        raise ValueError("uv.lock manifest members differ from the root-plus-Memory contract")
    for distribution, (_, editable_path, _) in _WORKSPACE_MANIFESTS.items():
        matches = [package for package in lock_packages if package.get("name") == distribution]
        if len(matches) != 1:
            raise ValueError(f"uv.lock must contain one workspace record for {distribution}")
        package = matches[0]
        if package.get("version") != version:
            raise ValueError(f"uv.lock version drift for {distribution}")
        if package.get("source") != {"editable": editable_path}:
            raise ValueError(f"uv.lock editable source drift for {distribution}")
        locked_dependencies = {
            str(dependency.get("name")) for dependency in package.get("dependencies", [])
        }
        if locked_dependencies != direct_dependencies[distribution]:
            raise ValueError(f"uv.lock direct dependencies drift for {distribution}")
        expected_extras = set(
            configs[distribution]["project"].get("optional-dependencies", {})  # type: ignore[index,union-attr]
        )
        if set(package.get("optional-dependencies", {})) != expected_extras:
            raise ValueError(f"uv.lock optional dependency drift for {distribution}")


def verify_dist(dist_dir: Path, *, config_path: Path) -> None:
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 2 or len(sdists) != 2:
        raise ValueError(
            f"expected two wheels and two sdists, found {len(wheels)} wheels and {len(sdists)} sdists"
        )

    facts_by_distribution: dict[str, WheelFacts] = {}
    legal_hashes = (
        _sha256((config_path.parent / "LICENSE").read_bytes()),
        _sha256((config_path.parent / "NOTICE").read_bytes()),
    )
    rules_by_distribution = _import_rules(config_path)
    for wheel in wheels:
        facts = _wheel_facts(
            wheel,
            rules_by_distribution=rules_by_distribution,
        )
        if facts.distribution in facts_by_distribution:
            raise ValueError(f"duplicate wheel for {facts.distribution}")
        facts_by_distribution[facts.distribution] = facts

    if set(facts_by_distribution) != set(_EXPECTED_PACKAGES):
        raise ValueError(
            "workspace wheel set differs from expected distributions: "
            f"{sorted(facts_by_distribution)}"
        )

    versions = {facts.version for facts in facts_by_distribution.values()}
    if len(versions) != 1:
        raise ValueError(f"workspace versions are not lockstep: {sorted(versions)}")
    (version,) = versions
    expected_sdists = {
        f"{distribution.replace('-', '_')}-{version}.tar.gz" for distribution in _EXPECTED_PACKAGES
    }
    actual_sdists = {path.name for path in sdists}
    if actual_sdists != expected_sdists:
        raise ValueError(
            f"sdist set does not match workspace wheels: expected {sorted(expected_sdists)}, "
            f"found {sorted(actual_sdists)}"
        )

    for distribution, facts in facts_by_distribution.items():
        expected_top_level = {_EXPECTED_PACKAGES[distribution]}
        if set(facts.top_level_packages) != expected_top_level:
            raise ValueError(
                f"{distribution}: expected top-level package {sorted(expected_top_level)}, "
                f"found {sorted(facts.top_level_packages)}"
            )
        actual_dlightrag_dependencies = set(facts.dependencies) & _DLIGHTRAG_DISTRIBUTIONS
        expected_dependencies = _EXPECTED_DLIGHTRAG_DEPENDENCIES[distribution]
        if actual_dlightrag_dependencies != expected_dependencies:
            raise ValueError(
                f"{distribution}: expected DlightRAG dependencies {sorted(expected_dependencies)}, "
                f"found {sorted(actual_dlightrag_dependencies)}"
            )
        if set(facts.extras) != _EXPECTED_EXTRAS[distribution]:
            raise ValueError(
                f"{distribution}: expected extras {sorted(_EXPECTED_EXTRAS[distribution])}, "
                f"found {sorted(facts.extras)}"
            )
        if distribution == "dlightrag":
            missing = _REQUIRED_ROOT_DEPENDENCIES - set(facts.dependencies)
            if missing:
                raise ValueError(
                    f"dlightrag: wheel is missing batteries-included dependencies {sorted(missing)}"
                )
        compact_requirements = {
            requirement.replace(" ", "").lower() for requirement in facts.requirements
        }
        for dependency in expected_dependencies:
            expected_requirement = f"{dependency}=={version}"
            if expected_requirement not in compact_requirements:
                raise ValueError(
                    f"{distribution}: dependency must be pinned as {expected_requirement}"
                )
        if facts.license_files != {"LICENSE", "NOTICE"} or facts.legal_hashes != legal_hashes:
            raise ValueError(f"{distribution}: wheel must contain repository LICENSE and NOTICE")
        if not facts.has_py_typed:
            raise ValueError(f"{distribution}: wheel must contain py.typed")
        if distribution == "dlightrag" and not facts.has_model_catalog:
            raise ValueError("dlightrag: wheel must contain engine/ai/model_catalog.json")
        if distribution == "dlightrag" and not facts.has_frontend:
            raise ValueError("dlightrag: wheel must contain generated frontend assets")

    parsed_sdists = [
        _sdist_facts(path, rules_by_distribution=rules_by_distribution) for path in sdists
    ]
    sdist_facts = {facts.distribution: facts for facts in parsed_sdists}
    for distribution, facts in sdist_facts.items():
        if facts.version != version or facts.top_level_packages != {
            _EXPECTED_PACKAGES[distribution]
        }:
            raise ValueError("sdist metadata or top-level packages do not match workspace wheels")
        if set(facts.extras) != _EXPECTED_EXTRAS[distribution]:
            raise ValueError(f"{distribution}: sdist extras do not match workspace wheel metadata")
        expected_dependencies = _EXPECTED_DLIGHTRAG_DEPENDENCIES[distribution]
        actual_dlightrag_dependencies = set(facts.dependencies) & _DLIGHTRAG_DISTRIBUTIONS
        if actual_dlightrag_dependencies != expected_dependencies:
            raise ValueError(f"{distribution}: sdist dependencies do not match wheel contract")
        compact_requirements = {
            requirement.replace(" ", "").lower() for requirement in facts.requirements
        }
        for dependency in expected_dependencies:
            expected_requirement = f"{dependency}=={version}"
            if expected_requirement not in compact_requirements:
                raise ValueError(
                    f"{distribution}: sdist dependency must be pinned as {expected_requirement}"
                )
        if distribution == "dlightrag":
            missing = _REQUIRED_ROOT_DEPENDENCIES - set(facts.dependencies)
            if missing:
                raise ValueError(
                    f"dlightrag: sdist is missing batteries-included dependencies {sorted(missing)}"
                )
        if facts.license_files != {"LICENSE", "NOTICE"} or facts.legal_hashes != legal_hashes:
            raise ValueError(f"{distribution}: sdist must contain repository LICENSE and NOTICE")
        if not facts.has_py_typed:
            raise ValueError(f"{distribution}: sdist must contain py.typed")
        if distribution == "dlightrag" and not facts.has_model_catalog:
            raise ValueError("dlightrag: sdist must contain engine/ai/model_catalog.json")
        if distribution == "dlightrag" and not facts.has_frontend:
            raise ValueError("dlightrag: sdist must contain generated frontend assets")


def _run_checked(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    completed = subprocess.run(  # noqa: S603 - verifier builds fixed argv from validated artifacts
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"installed wheel smoke failed: {' '.join(command)}\n{detail}")


def _wheel_path(dist_dir: Path, distribution: str) -> Path:
    matches = tuple(dist_dir.glob(f"{distribution.replace('-', '_')}-*.whl"))
    if len(matches) != 1:
        raise ValueError(f"expected one wheel for {distribution}, found {len(matches)}")
    return matches[0]


def _direct_url_path(value: str) -> Path | None:
    parsed = urlparse(value)
    if parsed.scheme != "file":
        return None
    return Path(url2pathname(unquote(parsed.path))).resolve()


def _direct_url_sha256(value: dict[str, object]) -> str | None:
    archive_info = value.get("archive_info")
    if not isinstance(archive_info, dict):
        return None
    hashes = archive_info.get("hashes")
    if isinstance(hashes, dict) and isinstance(hashes.get("sha256"), str):
        return hashes["sha256"]
    legacy_hash = archive_info.get("hash")
    if isinstance(legacy_hash, str) and legacy_hash.startswith("sha256="):
        return legacy_hash.removeprefix("sha256=")
    return None


def _wheel_installation_members(path: Path) -> tuple[str, str, dict[str, bytes]]:
    with zipfile.ZipFile(path) as wheel:
        metadata_paths = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError(f"{path.name}: expected one METADATA file")
        distribution, version, _, _, _ = _metadata_facts(
            wheel.read(metadata_paths[0]), artifact=path.name
        )
        members = {
            name: wheel.read(name)
            for name in wheel.namelist()
            if not name.endswith("/") and not name.endswith(".dist-info/RECORD")
        }
    return distribution, version, members


def _smoke_root_interfaces() -> None:
    import asyncio
    from types import SimpleNamespace
    from typing import Any, cast

    import dlightrag
    from dlightrag import Application, create_application
    from dlightrag.adapters.http.client import AnswerRunClient
    from dlightrag.application.access import DEPLOYMENT_OWNER_ID
    from dlightrag.application.config import (
        DlightragConfig,
        QueryLaneRuntimeConfig,
        RuntimeConfig,
    )
    from dlightrag.application.corpus_admin import CorpusAdmin, CorpusAdminSettings
    from dlightrag.application.retrieval import (
        ProjectedRetrieval,
        RetrievalOptions,
        RetrievalService,
        RetrievalSettings,
    )
    from dlightrag.application.settings import rag_settings
    from dlightrag.engine.agent import AgentSessionRuntime, ContextContribution, ToolRegistry
    from dlightrag.engine.ai.settings import ModelsSettings
    from dlightrag.engine.ai.telemetry import NoopTelemetry
    from dlightrag.engine.answer.execution import AnswerExecutor
    from dlightrag.engine.answer.fast import FastRunBoundaries
    from dlightrag.engine.answer.orchestration import AnswerOrchestrator
    from dlightrag.engine.answer.research.context import ContextAssembler
    from dlightrag.engine.rag.retrieval import RetrievalResult
    from dlightrag.engine.rag.workspace.settings import (
        CorpusSettings,
        IngestionSettings,
        PipelineSettings,
    )
    from dlightrag.engine.runtime import run_request_fingerprint

    class Planner:
        async def plan(self, query, **_kwargs):
            return SimpleNamespace(
                standalone_query=query,
                metadata_filter=None,
                metadata_filter_source=None,
                bm25_query=None,
                outcome="planned",
            )

    class Planners:
        def __init__(self):
            self.planner = Planner()

        def planner_for(self, _model_profile=None):
            return self.planner

        async def aclose(self):
            return None

    class Pool:
        def __init__(self):
            self.query = None
            self.kwargs = None

        async def warm(self, _workspaces):
            return None

        async def acquire(self, _workspace):
            return self

        async def aretrieve(self, query, **kwargs):
            self.query = query
            self.kwargs = kwargs
            return RetrievalResult(contexts={"chunks": [{"chunk_id": "installed"}]})

    def projector(result, _projection):
        return ProjectedRetrieval(contexts=result.contexts, sources=())

    async def empty_schema(_workspaces):
        return {}

    async def no_images(_images):
        return []

    async def retrieval_smoke():
        pool = Pool()
        service = RetrievalService(
            pool=cast(Any, pool),
            planners=cast(Any, Planners()),
            schema_lookup=empty_schema,
            image_preparer=no_images,
            projector=projector,
            settings=RetrievalSettings(
                default_top_k=8,
                default_chunk_top_k=5,
                timeout_seconds=5,
                query_image_limit=4,
            ),
            telemetry=NoopTelemetry(),
        )
        service.warm(("default",))
        response = await service.retrieve_result(
            "installed retrieval",
            workspaces=("default",),
            retrieval=RetrievalOptions(),
        )
        if response.contexts["chunks"][0]["chunk_id"] != "installed":
            raise ValueError("installed Retrieval service did not project RAG contexts")
        if pool.query != "installed retrieval" or pool.kwargs is None or pool.kwargs["top_k"] != 8:
            raise ValueError("installed Retrieval service did not apply request defaults")
        await service.aclose()
        if not service.closed:
            raise ValueError("installed Retrieval service did not close")

    async def corpus_smoke():
        runtime = SimpleNamespace(aregister_workspace=lambda **_kwargs: None)

        async def acquire(_workspace):
            return runtime

        async def list_workspace_records():
            return [
                {
                    "workspace": "installed",
                    "display_name": "Installed",
                    "embedding_model": "embed-model",
                }
            ]

        pool = SimpleNamespace(acquire=acquire)
        maintenance = SimpleNamespace(list_workspace_records=list_workspace_records)
        service = CorpusAdmin(
            settings=CorpusAdminSettings(
                default_workspace_id="default",
                default_display_name="Default",
                default_embedding_model="embed-model",
                input_root=Path(tempfile.gettempdir()) / "installed-corpus",
                read_only=False,
            ),
            pool=cast(Any, pool),
            maintenance=cast(Any, maintenance),
            file_panel=cast(Any, SimpleNamespace()),
            metadata_search=cast(Any, SimpleNamespace()),
            source_download_for=cast(Any, lambda _workspace: SimpleNamespace()),
            file_panel_cursor_secret=b"wheel-file-panel-cursor",
            metadata_search_cursor_secret=b"wheel-metadata-search-cursor",
            workspace_catalog_cursor_secret=b"wheel-workspace-catalog-cursor",
        )
        if await service.list_workspaces() != ["installed", "default"]:
            raise ValueError("installed CorpusAdmin did not expose its workspace catalog")

    config = DlightragConfig(
        models=ModelsSettings(max_concurrency=2),
        runtime=RuntimeConfig(query=QueryLaneRuntimeConfig(worker_concurrency=3)),
        corpus=CorpusSettings(
            ingestion=IngestionSettings(pipeline=PipelineSettings(max_concurrency=5))
        ),
    )
    settings = rag_settings(config)
    if len(DEPLOYMENT_OWNER_ID) != 64:
        raise ValueError("installed Access package did not expose a SHA-256 owner id")
    if AnswerRunClient.__module__ != "dlightrag.adapters.http.client.client":
        raise ValueError("installed HTTP client did not expose the durable Answer client")
    if not all(
        hasattr(AnswerRunClient, name)
        for name in (
            "create",
            "status",
            "events",
            "cancel",
            "list_runs",
            "steer",
            "follow_up",
            "fork",
            "children",
        )
    ):
        raise ValueError("installed HTTP client is missing the Agent 3.0 Answer controls")
    if AgentSessionRuntime.__module__ != "dlightrag.engine.agent.session.runtime":
        raise ValueError("installed Agent kernel did not expose AgentSessionRuntime")
    answer_identities = {
        AnswerExecutor: "dlightrag.engine.answer.execution.executor",
        AnswerOrchestrator: "dlightrag.engine.answer.orchestration.orchestrator",
        FastRunBoundaries: "dlightrag.engine.answer.fast.boundaries",
        ContextAssembler: "dlightrag.engine.answer.research.context",
    }
    for answer_type, expected_module in answer_identities.items():
        if answer_type.__module__ != expected_module:
            raise ValueError(
                f"installed Answer type {answer_type.__name__} came from {answer_type.__module__}"
            )
    ToolRegistry()
    ContextContribution(
        source="wheel.smoke",
        authority="reference",
        messages=({"role": "user", "content": "installed"},),
    )
    if dlightrag.DlightragConfig is not DlightragConfig:
        raise ValueError("installed root package did not expose its config owner")
    if dlightrag.Application is not Application:
        raise ValueError("installed root package did not expose Application")
    if dlightrag.create_application is not create_application:
        raise ValueError("installed root package did not expose create_application")
    if set(dlightrag.__all__) != {
        "Application",
        "DlightragConfig",
        "create_application",
        "__version__",
    }:
        raise ValueError("installed root facade exports unexpected names")
    for retired_module in (
        "dlightrag.access",
        "dlightrag.agent",
        "dlightrag.ai",
        "dlightrag.config",
        "dlightrag.health",
        "dlightrag.model_settings",
        "dlightrag.runtime",
        "dlightrag.rag",
        "dlightrag." + "answer",
        "dlightrag.maintenance",
        "dlightrag.observability",
        "dlightrag.adapters.mcp_tools",
        "dlightrag.api",
        "dlightrag.mcp",
        "dlightrag.sdk",
        "dlightrag.web",
        "dlightrag.services",
        "dlightrag.app_state",
        "dlightrag.contracts",
        "dlightrag.core.client_attachments",
        "dlightrag.core.client_contracts",
        "dlightrag.core.client_payloads",
        "dlightrag.core.client_requests",
        "dlightrag.core.servicemanager",
        "dlightrag.utils",
    ):
        try:
            retired_spec = importlib.util.find_spec(retired_module)
        except ModuleNotFoundError:
            retired_spec = None
        if retired_spec is not None:
            raise ValueError(f"installed root package still contains {retired_module}")
    if config.models.max_concurrency != 2:
        raise ValueError("installed root config did not preserve AI concurrency")
    if config.runtime.query.worker_concurrency != 3:
        raise ValueError("installed root config did not preserve Runtime concurrency")
    if settings.rag_pipeline_max_async != 5:
        raise ValueError("installed root mapping did not preserve RAG concurrency")
    fingerprint = run_request_fingerprint({"query": "installed wheel", "workspaces": ["default"]})
    if len(fingerprint) != 64:
        raise ValueError("installed Runtime did not produce a SHA-256 request fingerprint")
    asyncio.run(retrieval_smoke())
    asyncio.run(corpus_smoke())


def verify_installed(dist_dir: Path) -> None:
    """Prove this interpreter loaded root and Memory from the current wheels."""
    dist_dir = dist_dir.resolve()
    repository = dist_dir.parent.resolve()
    versions: set[str] = set()
    for expected_distribution, package in _EXPECTED_PACKAGES.items():
        wheel_path = _wheel_path(dist_dir, expected_distribution).resolve()
        distribution, version, members = _wheel_installation_members(wheel_path)
        if distribution != expected_distribution:
            raise ValueError(f"{wheel_path.name}: unexpected distribution {distribution}")
        installed = importlib.metadata.distribution(expected_distribution)
        if installed.version != version:
            raise ValueError(
                f"{expected_distribution}: installed version {installed.version} != {version}"
            )
        versions.add(version)

        direct_url_raw = installed.read_text("direct_url.json")
        if direct_url_raw is None:
            raise ValueError(f"{expected_distribution}: missing direct_url.json")
        direct_url = json.loads(direct_url_raw)
        if not isinstance(direct_url, dict):
            raise ValueError(f"{expected_distribution}: invalid direct_url.json")
        if _direct_url_path(str(direct_url.get("url") or "")) != wheel_path:
            raise ValueError(
                f"{expected_distribution}: installed artifact is not {wheel_path.name}"
            )
        direct_url_hash = _direct_url_sha256(direct_url)
        if direct_url_hash is not None and direct_url_hash != _sha256(wheel_path.read_bytes()):
            raise ValueError(f"{expected_distribution}: installed artifact hash differs")

        installed_files = {str(path).replace("\\", "/"): path for path in installed.files or ()}
        for member, raw in members.items():
            if member not in installed_files:
                raise ValueError(f"{expected_distribution}: installed file missing: {member}")
            installed_path = Path(str(installed.locate_file(installed_files[member]))).resolve()
            if not installed_path.is_file() or _sha256(installed_path.read_bytes()) != _sha256(raw):
                raise ValueError(f"{expected_distribution}: installed file differs: {member}")
            if installed_path.is_relative_to(repository):
                raise ValueError(
                    f"{expected_distribution}: imported from checkout: {installed_path}"
                )

        module = importlib.import_module(package)
        module_path = Path(str(module.__file__)).resolve()
        if "site-packages" not in module_path.parts or module_path.is_relative_to(repository):
            raise ValueError(f"{package}: not imported from isolated site-packages: {module_path}")

    if len(versions) != 1:
        raise ValueError(f"installed workspace versions are not lockstep: {sorted(versions)}")
    _smoke_root_interfaces()


def _venv_executable(venv: Path, name: str) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / f"{name}.exe"
    return venv / "bin" / name


def smoke_installed(dist_dir: Path, *, config_path: Path) -> None:
    dist_dir = dist_dir.resolve()
    config_path = config_path.resolve()
    uv_executable = shutil.which("uv")
    if uv_executable is None:
        raise ValueError("uv is required to smoke installed wheels")
    smoke_cases = (("memory", ("dlightrag-memory",), _MEMORY_SMOKE),)
    required_distributions = {
        requirement.partition("[")[0]
        for _, requirements, _ in smoke_cases
        for requirement in requirements
    }
    wheel_paths = {
        distribution: _wheel_path(dist_dir, distribution) for distribution in required_distributions
    }
    base_env = dict(os.environ)
    base_env.pop("PYTHONPATH", None)
    for name in tuple(base_env):
        if name.startswith("DLIGHTRAG_"):
            base_env.pop(name)
    base_env["PYTHONSAFEPATH"] = "1"

    for label, distributions, code in smoke_cases:
        with tempfile.TemporaryDirectory(prefix=f"dlightrag-{label}-wheel-") as raw_temp:
            temp = Path(raw_temp)
            venv = temp / ".venv"
            _run_checked(
                [uv_executable, "venv", "--python", "3.14", str(venv)],
                cwd=temp,
                env=base_env,
            )
            python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            _run_checked(
                [
                    uv_executable,
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    *(
                        f"{requirement} @ {wheel_paths[requirement.partition('[')[0]].as_uri()}"
                        if "[" in requirement
                        else str(wheel_paths[requirement])
                        for requirement in distributions
                    ),
                ],
                cwd=temp,
                env=base_env,
            )
            _run_checked([str(python), "-I", "-c", code], cwd=temp, env=base_env)

    with tempfile.TemporaryDirectory(prefix="dlightrag-root-wheel-") as raw_temp:
        temp = Path(raw_temp)
        venv = temp / ".venv"
        _run_checked(
            [uv_executable, "venv", "--python", "3.14", str(venv)],
            cwd=temp,
            env=base_env,
        )
        python = _venv_executable(venv, "python")
        _run_checked(
            [
                uv_executable,
                "pip",
                "install",
                "--python",
                str(python),
                *(str(_wheel_path(dist_dir, distribution)) for distribution in _EXPECTED_PACKAGES),
            ],
            cwd=temp,
            env=base_env,
        )
        _run_checked(
            [
                str(python),
                "-I",
                str(Path(__file__).resolve()),
                "--installed",
                "--dist",
                str(dist_dir),
                "--config",
                str(config_path),
            ],
            cwd=temp,
            env=base_env,
        )
        for script in _ROOT_CONSOLE_SCRIPTS:
            _run_checked(
                [str(_venv_executable(venv, script)), "--help"],
                cwd=temp,
                env=base_env,
            )
        _run_checked(
            [uv_executable, "pip", "install", "--python", str(python), "import-linter"],
            cwd=temp,
            env=base_env,
        )
        _run_checked(
            [
                str(_venv_executable(venv, "lint-imports")),
                "--config",
                str(config_path),
            ],
            cwd=temp,
            env=base_env,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist", type=Path, required=True, help="Directory containing built artifacts"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pyproject.toml"),
        help="Import-linter configuration used as the source policy",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        help="Workspace containing root/Memory manifests and uv.lock (defaults to config parent)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--smoke-installed",
        action="store_true",
        help="Install all wheels into isolated Python 3.14 environments and run interface smokes",
    )
    mode.add_argument(
        "--installed",
        action="store_true",
        help="Verify this interpreter is using the current built wheel set",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        workspace_root = args.workspace_root or args.config.resolve().parent
        verify_workspace_definition(workspace_root)
        verify_dist(args.dist, config_path=args.config)
        if args.installed:
            verify_installed(args.dist)
        elif args.smoke_installed:
            smoke_installed(args.dist, config_path=args.config)
    except (
        ImportError,
        importlib.metadata.PackageNotFoundError,
        json.JSONDecodeError,
        OSError,
        SyntaxError,
        ValueError,
        tarfile.TarError,
        zipfile.BadZipFile,
    ) as exc:
        print(f"workspace wheel verification failed: {exc}", file=sys.stderr)
        return 1
    print("workspace wheel verification passed")
    if args.installed:
        # This is a disposable isolated smoke process. Some optional native
        # dependencies race during interpreter teardown on macOS and can turn
        # an already-passed verification into a spurious libc++ abort.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
