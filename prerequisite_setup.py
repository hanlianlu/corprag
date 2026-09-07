#!/usr/bin/env python3
# /// script
# requires-python = ">=3.14"
# dependencies = ["dlightrag", "dlightrag-memory", "questionary>=2", "rich>=13", "ruamel.yaml>=0.18"]
# [tool.uv.sources]
# dlightrag = { path = ".", editable = true }
# dlightrag-memory = { path = "packages/memory", editable = true }
# ///
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""DlightRAG one-command onboarding wizard.

Run with:  uv run prerequisite_setup.py

Module-level imports stay limited to the stdlib so the pure logic is importable
under pytest; ruamel.yaml is imported lazily in the config writer, and
questionary/rich are imported lazily inside the interactive functions.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import posixpath
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.yaml"
ENV_PATH = REPO_ROOT / ".env"
ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"
MODEL_CATALOG_PATH = REPO_ROOT / "src" / "dlightrag" / "engine" / "ai" / "model_catalog.json"
MINERU_ENV_PATH = REPO_ROOT / ".env.mineru"
MINERU_ENV_EXAMPLE_PATH = REPO_ROOT / ".env.mineru.example"
API_READY_URL = "http://localhost:8100/ready"
WEB_URL = "http://localhost:8100/web/"

# Mirror of the shipped AnswerConfig attachment/image defaults, used only to
# display sensible values when a hand-edited config.yaml omits an Answer block.
DEFAULT_MAX_ATTACHMENTS = 6
DEFAULT_MAX_ATTACHMENT_BYTES = 100 * 1024 * 1024
DEFAULT_MAX_TOTAL_ATTACHMENT_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_ANSWER_IMAGES = 12

DOCLING_EXTERNAL_CHOICE = "Docling external endpoint"
DOCLING_MPS_DEVICE_CHOICE = "Apple Silicon MPS (granite_docling)"
DOCLING_SERVICE_DEFAULT_DEVICE_CHOICE = "CUDA, XPU, or CPU (service default)"


# ---------------------------------------------------------------------------
# Provider registry and role resolvers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProviderSpec:
    """Maps a novice-friendly provider name to DlightRAG config."""

    provider: str  # DlightRAG provider enum value
    base_url: str | None  # canonical default; None => native (no base_url)
    requires_url: bool = False  # user MUST supply (Azure / Other, tenant-specific)
    requires_key: bool = True
    default_model: str = ""  # "" => no safe default; the user must name the model
    gui_binding: str | None = None
    gui_host: str | None = None


# LLM roles: openai-compatible => provider "openai" + base_url; else native.
# Azure and "Other" get no default model: the name is the caller's own deployment.
PROVIDERS_LLM: dict[str, ProviderSpec] = {
    "OpenAI": ProviderSpec("openai", "https://api.openai.com/v1", default_model="gpt-5.6-terra"),
    "DeepSeek": ProviderSpec(
        "openai", "https://api.deepseek.com", default_model="deepseek-v4-flash"
    ),
    "OpenRouter": ProviderSpec(
        "openai", "https://openrouter.ai/api/v1", default_model="google/gemini-3.7-flash"
    ),
    "Anthropic": ProviderSpec("anthropic", None, default_model="claude-sonnet-5"),
    "Gemini": ProviderSpec("gemini", None, default_model="gemini-3.7-flash"),
    "Azure OpenAI": ProviderSpec("openai", None, requires_url=True),
    "Other (OpenAI-compatible)": ProviderSpec(
        "openai", None, requires_url=True, requires_key=False
    ),
}

# Embedding providers. Defaults are the current first-class model for each
# wire protocol; Jina v4 remains intentional because v5 does not expose the
# native single-vector text+image fusion DlightRAG's canonical chunk needs.
PROVIDERS_EMBED: dict[str, ProviderSpec] = {
    "Voyage": ProviderSpec(
        "voyage",
        "https://api.voyageai.com/v1",
        default_model="voyage-multimodal-3.5",
        gui_binding="voyageai",
        gui_host="https://api.voyageai.com/v1",
    ),
    "OpenAI": ProviderSpec(
        "openai",
        "https://api.openai.com/v1",
        default_model="text-embedding-3-large",
        gui_binding="openai",
        gui_host="https://api.openai.com/v1",
    ),
    "Gemini": ProviderSpec(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta",
        default_model="gemini-embedding-2",
        gui_binding="gemini",
        gui_host="DEFAULT_GEMINI_ENDPOINT",
    ),
    "Jina": ProviderSpec(
        "jina",
        "https://api.jina.ai/v1",
        default_model="jina-embeddings-v4",
        gui_binding="jina",
        gui_host="https://api.jina.ai/v1",
    ),
    "Cohere": ProviderSpec(
        "cohere",
        "https://api.cohere.com",
        default_model="embed-v4.0",
        gui_binding="openai",
        gui_host="https://api.cohere.ai/compatibility/v1",
    ),
    "Azure OpenAI": ProviderSpec(
        "openai",
        None,
        requires_url=True,
        gui_binding="openai",
        gui_host="https://api.openai.com/v1",
    ),
    "Azure Cohere": ProviderSpec(
        "azure_cohere",
        None,
        requires_url=True,
        default_model="Cohere-embed-v4",
        gui_binding="openai",
        gui_host="https://api.openai.com/v1",
    ),
    "Other (OpenAI-compatible)": ProviderSpec(
        "openai_compatible",
        None,
        requires_url=True,
        requires_key=False,
        gui_binding="openai",
        gui_host="https://api.openai.com/v1",
    ),
}

# Known first-class embedding model -> native output dimension.
EMBED_DIMS: dict[str, int] = {
    "voyage-multimodal-3.5": 1024,
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "gemini-embedding-2": 3072,
    "jina-embeddings-v4": 2048,
    "embed-v4.0": 1536,
    "Cohere-embed-v4": 1536,
}

# Rerank menu label -> (strategy, needs its own API key, default model).
RERANK_CHOICES: dict[str, tuple[str, bool, str]] = {
    "Reuse my LLM": ("chat_llm_reranker", False, ""),
    "Voyage": ("voyage_reranker", True, "rerank-2.5-lite"),
    "Jina": ("jina_reranker", True, ""),
    "Cohere": ("cohere_reranker", True, ""),
    "Azure Cohere": ("azure_cohere", True, ""),
}

LLM_ROLE_ENV_KEYS: dict[str, str] = {
    "extract": "DLIGHTRAG_MODELS__CHAT__ROLES__EXTRACT__API_KEY",
    "keyword": "DLIGHTRAG_MODELS__CHAT__ROLES__KEYWORD__API_KEY",
    "query": "DLIGHTRAG_MODELS__CHAT__ROLES__QUERY__API_KEY",
    "vlm": "DLIGHTRAG_MODELS__CHAT__ROLES__VLM__API_KEY",
}


def _model_block(spec: ProviderSpec, model: str, base_url: str | None) -> dict:
    block: dict = {"provider": spec.provider, "model": model}
    resolved = base_url or spec.base_url
    if resolved is not None:
        block["base_url"] = resolved
    return block


def resolve_llm_choice(provider_name: str, *, model: str, base_url: str | None) -> tuple[dict, str]:
    spec = PROVIDERS_LLM[provider_name]
    return _model_block(spec, model, base_url), "DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY"


def _normalized_endpoint(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = urlsplit(value)
        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").rstrip(".").lower()
        if scheme not in {"http", "https"} or not hostname:
            return None
        port = parsed.port
    except ValueError:
        return None
    if port == {"http": 80, "https": 443}[scheme]:
        port = None
    authority = f"[{hostname}]" if ":" in hostname else hostname
    if port is not None:
        authority = f"{authority}:{port}"
    path = posixpath.normpath(parsed.path or "/")
    if not path.startswith("/"):
        path = f"/{path}"
    return urlunsplit((scheme, authority, path, "", ""))


def _model_catalog() -> dict:
    return json.loads(MODEL_CATALOG_PATH.read_text(encoding="utf-8"))


def catalog_model_profile(block: dict) -> dict | None:
    """Return shared catalog facts for one exact model endpoint."""
    identity = (
        str(block.get("provider") or ""),
        str(block.get("model") or ""),
        _normalized_endpoint(block.get("base_url")),
    )
    for item in _model_catalog().get("models", []):
        candidate = (
            str(item.get("provider") or ""),
            str(item.get("model") or ""),
            _normalized_endpoint(item.get("base_url")),
        )
        if candidate == identity:
            return dict(item["profile"])
    return None


def resolve_embedding_choice(
    provider_name: str, *, model: str, base_url: str | None
) -> tuple[dict, str]:
    spec = PROVIDERS_EMBED[provider_name]
    block = _model_block(spec, model, base_url)
    block["dim"] = EMBED_DIMS.get(model, 0)  # 0 => caller must prompt
    return block, "DLIGHTRAG_MODELS__EMBEDDING__API_KEY"


def resolve_rerank_choice(
    choice: str,
    *,
    base_url: str | None = None,
) -> tuple[dict, str | None]:
    strategy, needs_key, model = RERANK_CHOICES[choice]
    block: dict = {"strategy": strategy}
    if model:
        block["model"] = model
    if base_url:
        block["base_url"] = base_url
    return block, ("DLIGHTRAG_MODELS__RERANK__API_KEY" if needs_key else None)


# ---------------------------------------------------------------------------
# config.yaml writer (comment-preserving) and .env upsert
# ---------------------------------------------------------------------------
def _yaml():
    from ruamel.yaml import YAML  # lazy: PEP 723 runtime dep / dev-test dep

    y = YAML()  # round-trip mode by default: preserves comments
    y.preserve_quotes = True
    y.indent(mapping=2, sequence=4, offset=2)
    return y


def _apply_model_block(node, block: dict) -> None:
    """Replace one model block without retaining provider-specific settings."""
    node.clear()
    node.update(block)


def _configured_embedding_dim(path: Path, env_path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(_load_effective_config(path, env_path).models.embedding.dim)
    except OSError, TypeError, ValueError:
        data = _yaml().load(path) or {}
        value = ((data.get("models") or {}).get("embedding") or {}).get("dim")
        return int(value) if value is not None else 1024


def write_config_yaml(
    path: Path,
    *,
    llm_default: dict | None = None,
    llm_roles: dict[str, dict] | None = None,
    embedding: dict | None = None,
    rerank: dict | None = None,
    parser_kind: str | None = None,
    mineru_api_mode: str | None = None,
    docling_endpoint: str | None = None,
    docling_code_formula_preset: str | None = None,
    web_sources: dict | None = None,
) -> None:
    yaml = _yaml()
    data = yaml.load(path)
    canonical_sections = {
        "deployment",
        "storage",
        "models",
        "corpus",
        "answer",
        "access",
        "interfaces",
        "observability",
    }
    unknown = set(data) - canonical_sections
    if unknown:
        raise ValueError(
            "Legacy or unknown DlightRAG config fields were found; replace them with the "
            f"3.0 nine-section schema before running setup (unknown: {sorted(unknown)})"
        )
    answer = data.setdefault("answer", {})
    agent = answer.setdefault("agent", {})
    agent.setdefault("execution_environment", "trust")
    agent.setdefault("workspace_root", None)
    agent.setdefault("outbound_mcp", [])
    if web_sources is not None:
        node = answer.setdefault("web_sources", {})
        node.clear()
        node.update(web_sources)
        answer.pop("web_search", None)
    models = data.setdefault("models", {})
    chat = models.setdefault("chat", {})
    if llm_default is not None:
        _apply_model_block(chat.setdefault("default", {}), llm_default)
    if llm_roles is not None:
        if not llm_roles:
            chat.pop("roles", None)
        else:
            roles = chat.setdefault("roles", {})
            roles.clear()
            for role, block in llm_roles.items():
                roles.setdefault(role, {})
                _apply_model_block(roles[role], block)
    models.pop("capacity_overrides", None)
    if embedding is not None:
        _apply_model_block(models.setdefault("embedding", {}), embedding)
    if rerank is not None:
        rerank_node = models.setdefault("rerank", {})
        rerank_node.clear()
        rerank_node.update(rerank)
    if parser_kind is not None:
        if parser_kind not in {"mineru", "docling"}:
            raise ValueError(f"Unsupported parser kind: {parser_kind}")
        corpus = data.setdefault("corpus", {})
        parser = corpus.get("parser")
        if isinstance(parser, dict):
            parser.pop("rules", None)
            if not parser:
                corpus.pop("parser", None)
        sidecars = corpus.setdefault("sidecars", {})
        if parser_kind == "mineru":
            sidecars.pop("docling", None)
            mineru = sidecars.setdefault("mineru", {})
            mineru["api_mode"] = mineru_api_mode or "local"
            if mineru["api_mode"] == "local":
                mineru["local_endpoint"] = "http://host.docker.internal:8210"
            mineru.setdefault("language", "ch")
        else:
            sidecars.pop("mineru", None)
            docling = sidecars.setdefault("docling", {})
            docling["endpoint"] = docling_endpoint or "http://docling:5001"
            docling["code_formula_preset"] = docling_code_formula_preset
    yaml.dump(data, path)


def upsert_env(path: Path, values: dict[str, str], *, remove_keys: tuple[str, ...] = ()) -> None:
    """Insert/replace KEY=value lines; remove requested active keys."""
    remaining = dict(values)
    keys_to_remove = set(remove_keys) - set(remaining)
    lines: list[str] = []
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            key = raw.split("=", 1)[0].strip() if "=" in raw else ""
            if key in keys_to_remove:
                continue
            if key in remaining:
                lines.append(f"{key}={remaining.pop(key)}")
            elif key in values:
                continue
            else:
                lines.append(raw)
    for key, value in remaining.items():
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Backup + validation
# ---------------------------------------------------------------------------
MAX_BACKUPS = 1  # keep only the most recent <file>.bak-<timestamp> per file


def _prune_backups(path: Path, *, keep: int) -> None:
    """Delete all but the newest ``keep`` ``<name>.bak-*`` siblings of ``path``."""
    if keep < 0:
        return
    backups = sorted(path.parent.glob(f"{path.name}.bak-*"), key=lambda p: p.name)
    for old in backups[: len(backups) - keep]:
        old.unlink(missing_ok=True)


def backup_file(path: Path, *, keep: int = MAX_BACKUPS) -> Path | None:
    if not path.exists():
        return None
    stamp = _dt.datetime.now().strftime("%Y%m%d%H%M%S")
    backup = path.with_name(f"{path.name}.bak-{stamp}")
    backup.write_bytes(path.read_bytes())
    _prune_backups(path, keep=keep)
    return backup


def _load_effective_config(config_path: Path, env_path: Path):
    """Load one explicit repository config with normal env-over-dotenv precedence."""
    if config_path.name != "config.yaml":
        raise ValueError("canonical configuration file must be named config.yaml")
    from dlightrag.application.config import load_config, reset_config

    previous = Path.cwd()
    try:
        os.chdir(config_path.parent)
        reset_config()
        return load_config(env_path)
    finally:
        os.chdir(previous)


def validate_config() -> None:
    """Load and validate the exact files written by this wizard."""
    _load_effective_config(CONFIG_PATH, ENV_PATH)


def _restore_file(path: Path, existed: bool, backup: Path | None) -> None:
    if existed:
        if backup is None:
            raise RuntimeError(f"missing rollback backup for {path}")
        path.write_bytes(backup.read_bytes())
    else:
        path.unlink(missing_ok=True)


def _apply_parser_change(
    configure: Callable[[], None],
    *,
    after_validation: Callable[[], None] | None = None,
) -> None:
    """Back up, apply, validate, and fully roll back one parser reconfiguration."""
    snapshots = [
        (path, path.exists(), backup_file(path))
        for path in (CONFIG_PATH, ENV_PATH, MINERU_ENV_PATH)
    ]
    try:
        configure()
        validate_config()
        if after_validation is not None:
            after_validation()
    except Exception:
        for path, existed, backup in snapshots:
            _restore_file(path, existed, backup)
        raise


# ---------------------------------------------------------------------------
# Platform / GPU / WSL2 detection and preflight
# ---------------------------------------------------------------------------
_PROC_VERSION = Path("/proc/version")


@dataclass(frozen=True)
class PlatformInfo:
    os: str  # "macos" | "linux" | "windows"
    arch: str  # "arm64" | "x86_64" | ...
    is_wsl: bool


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    hint: str = ""


def detect_platform() -> PlatformInfo:
    system = platform.system()
    os_name = {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}.get(
        system, system.lower()
    )
    is_wsl = False
    if os_name == "linux" and _PROC_VERSION.exists():
        is_wsl = "microsoft" in _PROC_VERSION.read_text(encoding="utf-8", errors="ignore").lower()
    return PlatformInfo(os=os_name, arch=platform.machine().lower(), is_wsl=is_wsl)


def has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


def run_preflight() -> list[Check]:
    info = detect_platform()
    checks: list[Check] = []
    for tool, hint in (
        ("uv", "https://docs.astral.sh/uv/"),
        ("docker", "https://docs.docker.com/get-docker/"),
        ("make", "macOS: xcode-select --install | Debian: apt-get install make"),
    ):
        checks.append(Check(tool, shutil.which(tool) is not None, hint))
    if info.os == "windows" and not info.is_wsl:
        checks.append(
            Check(
                "wsl2",
                False,
                "Run this wizard inside WSL2 (Docker Desktop WSL2 backend).",
            )
        )
    return checks


# ---------------------------------------------------------------------------
# MinerU extras + hybrid service-model resolution
# ---------------------------------------------------------------------------
def select_mineru_extras(info: PlatformInfo, *, has_gpu: bool) -> str:
    if info.os == "macos" and info.arch in ("arm64", "aarch64"):
        return "core,mlx"
    if has_gpu:
        return "core,vllm"
    return "core"


def resolve_service_model(info: PlatformInfo, *, systemd_available: bool) -> str:
    """Hybrid: background where a first-class mechanism exists, else foreground."""
    if info.os == "macos":
        return "launchd"
    if info.os == "linux" and systemd_available:
        return "systemd-user"
    return "foreground"


def systemd_user_available() -> bool:
    if shutil.which("systemctl") is None:
        return False
    result = subprocess.run(
        ["systemctl", "--user", "is-system-running"],  # noqa: S607 - fixed argv on PATH
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 or "running" in result.stdout or "degraded" in result.stdout


# ---------------------------------------------------------------------------
# MinerU parser: official cloud vs local install + hybrid background service
# ---------------------------------------------------------------------------
def configure_mineru_official(token: str) -> None:
    write_config_yaml(CONFIG_PATH, parser_kind="mineru", mineru_api_mode="official")
    upsert_env(
        ENV_PATH,
        {"DLIGHTRAG_CORPUS__SIDECARS__MINERU__API_TOKEN": token},
        remove_keys=("COMPOSE_PROFILES",),
    )


def configure_mineru_local_env(extras: str, *, title_aided: dict | None = None) -> None:
    if not MINERU_ENV_PATH.exists() and MINERU_ENV_EXAMPLE_PATH.exists():
        MINERU_ENV_PATH.write_bytes(MINERU_ENV_EXAMPLE_PATH.read_bytes())
    values = {"MINERU_INSTALL_EXTRAS": extras}
    remove_keys = ()
    if title_aided:
        values["MINERU_TITLE_AIDED_ENABLE"] = "true"
        values["MINERU_TITLE_AIDED_API_KEY"] = title_aided["api_key"]
        values["MINERU_TITLE_AIDED_BASE_URL"] = title_aided["base_url"]
        values["MINERU_TITLE_AIDED_MODEL"] = title_aided["model"]
    else:
        values["MINERU_TITLE_AIDED_ENABLE"] = "false"
        remove_keys = (
            "MINERU_TITLE_AIDED_API_KEY",
            "MINERU_TITLE_AIDED_BASE_URL",
            "MINERU_TITLE_AIDED_MODEL",
            "MINERU_TITLE_AIDED_ENABLE_THINKING",
        )
    upsert_env(MINERU_ENV_PATH, values, remove_keys=remove_keys)
    write_config_yaml(CONFIG_PATH, parser_kind="mineru", mineru_api_mode="local")
    upsert_env(
        ENV_PATH,
        {},
        remove_keys=(
            "COMPOSE_PROFILES",
            "DLIGHTRAG_CORPUS__SIDECARS__MINERU__API_TOKEN",
        ),
    )


def configure_docling(
    endpoint: str,
    *,
    bundled: bool,
    code_formula_preset: str | None = "granite_docling",
) -> None:
    write_config_yaml(
        CONFIG_PATH,
        parser_kind="docling",
        docling_endpoint=endpoint,
        docling_code_formula_preset=None if bundled else code_formula_preset,
    )
    values = {"COMPOSE_PROFILES": "docling"} if bundled else {}
    remove_keys = (
        ("DLIGHTRAG_CORPUS__SIDECARS__MINERU__API_TOKEN",)
        if bundled
        else (
            "COMPOSE_PROFILES",
            "DLIGHTRAG_CORPUS__SIDECARS__MINERU__API_TOKEN",
        )
    )
    upsert_env(ENV_PATH, values, remove_keys=remove_keys)


def build_mineru_local_commands(service_model: str) -> list[list[str]]:
    cmds: list[list[str]] = [["make", "mineru-install"], ["make", "mineru-title-aided"]]
    if service_model in ("launchd", "systemd-user"):
        cmds.append(["make", "mineru-service-install"])
    return cmds


def _default_runner(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)  # noqa: S603 - fixed make/docker/uv argv


def _note_foreground_mineru() -> None:
    from rich.console import Console

    Console().print(
        "[yellow]No background service available.[/yellow] Start MinerU in another "
        "terminal with:  make mineru-api"
    )


def run_parser_step(
    prompter: Prompter,
    info: PlatformInfo,
    *,
    has_gpu: bool,
    llm_title_aided: dict | None = None,
    runner=_default_runner,
    require_confirm: bool = False,
) -> str | None:
    choice = prompter.select(
        "Document parser",
        [
            "MinerU local",
            "MinerU official cloud API",
            DOCLING_EXTERNAL_CHOICE,
            "Docling bundled (Compose)",
        ],
    )
    if choice == "MinerU official cloud API":
        token = _ask_required(lambda: prompter.password("MinerU API token (required)"))
        if require_confirm and not prompter.confirm(PARSER_OVERWRITE_CONFIRM, default=False):
            return None
        _apply_parser_change(lambda: configure_mineru_official(token))
        return "mineru"

    if choice in {"Docling bundled (Compose)", DOCLING_EXTERNAL_CHOICE}:
        endpoint = "http://docling:5001"
        code_formula_preset = None
        if choice == DOCLING_EXTERNAL_CHOICE:
            endpoint = _ask_required(
                lambda: prompter.text(
                    "Docling endpoint (required)",
                    default="http://host.docker.internal:5001",
                )
            )
            device = prompter.select(
                "Docling service device",
                [DOCLING_MPS_DEVICE_CHOICE, DOCLING_SERVICE_DEFAULT_DEVICE_CHOICE],
            )
            if device == DOCLING_MPS_DEVICE_CHOICE:
                code_formula_preset = "granite_docling"
        if require_confirm and not prompter.confirm(PARSER_OVERWRITE_CONFIRM, default=False):
            return None
        bundled = choice == "Docling bundled (Compose)"
        _apply_parser_change(
            lambda: configure_docling(
                endpoint,
                bundled=bundled,
                code_formula_preset=code_formula_preset,
            )
        )
        return "docling" if bundled else "external"

    extras = select_mineru_extras(info, has_gpu=has_gpu)
    title_aided = None
    if (
        llm_title_aided
        and llm_title_aided.get("base_url")
        and prompter.confirm("Improve heading detection with your LLM (title-aided)?", default=True)
    ):
        title_aided = llm_title_aided
    # Confirm AFTER collecting choices, right before overwriting existing settings.
    if require_confirm and not prompter.confirm(PARSER_OVERWRITE_CONFIRM, default=False):
        return None
    service_model = resolve_service_model(info, systemd_available=systemd_user_available())

    def configure_and_install() -> None:
        configure_mineru_local_env(extras, title_aided=title_aided)

    def install() -> None:
        for cmd in build_mineru_local_commands(service_model):
            runner(cmd)

    _apply_parser_change(configure_and_install, after_validation=install)
    if service_model == "foreground":
        _note_foreground_mineru()
    return "mineru"


# ---------------------------------------------------------------------------
# Interactive Models step (thin shell over the pure logic above)
# ---------------------------------------------------------------------------
class Prompter:
    """Minimal prompt surface; the questionary impl is created lazily in main()."""

    def select(self, message: str, choices: list[str]) -> str:
        raise NotImplementedError

    def text(self, message: str, default: str = "") -> str:
        raise NotImplementedError

    def password(self, message: str) -> str:
        raise NotImplementedError

    def confirm(self, message: str, default: bool = False) -> bool:
        raise NotImplementedError


def _ask_required(ask: Callable[[], str]) -> str:
    while True:
        value = ask().strip()
        if value:
            return value


def _ask_model(
    prompter: Prompter, providers: dict[str, ProviderSpec], role_label: str
) -> tuple[str, str, str | None, str | None]:
    name = prompter.select(f"{role_label} provider", list(providers))
    spec = providers[name]
    model = _ask_required(
        lambda: prompter.text(f"{role_label} model name (required)", default=spec.default_model)
    )
    if spec.requires_url:
        base_url = _ask_required(
            lambda: prompter.text(f"{role_label} base URL (required for {name})")
        )
    else:
        base_url = prompter.text(f"{role_label} base URL", default=spec.base_url or "")
    if spec.requires_key:
        key = _ask_required(lambda: prompter.password(f"{role_label} API key (required)"))
    else:
        key = prompter.password(f"{role_label} API key (optional)").strip() or None
    return name, model, (base_url or None), key


def run_models_step(prompter: Prompter, *, require_confirm: bool = False) -> dict | None:
    env_values: dict[str, str] = {}
    remove_env_keys: list[str] = []

    from rich.console import Console

    console = Console()
    console.print(
        "[dim]Provider = API protocol (openai / anthropic / gemini). Pick your vendor below — "
        "DeepSeek, OpenRouter, Azure, etc. map to the OpenAI-compatible protocol automatically.[/dim]"
    )
    console.print(
        "[dim]Minimum replaces old role-specific LLMs. Custom replaces old roles with "
        "extract/keyword choices.[/dim]"
    )

    mode = prompter.select(MODEL_MODE_PROMPT, MODEL_MODE_CHOICES)

    name, model, base_url, key = _ask_model(prompter, PROVIDERS_LLM, "LLM")
    llm_block, llm_env = resolve_llm_choice(name, model=model, base_url=base_url)
    if key is None:
        llm_block["api_key"] = None
        remove_env_keys.append(llm_env)
    else:
        env_values[llm_env] = key

    llm_roles: dict[str, dict] = {}
    if mode == MODEL_MODE_CUSTOM:
        for role in ("extract", "keyword"):
            rn, rm, rurl, rk = _ask_model(prompter, PROVIDERS_LLM, f"{role} LLM")
            block, _ = resolve_llm_choice(rn, model=rm, base_url=rurl)
            if rk is None:
                block["api_key"] = None
                remove_env_keys.append(LLM_ROLE_ENV_KEYS[role])
            else:
                env_values[LLM_ROLE_ENV_KEYS[role]] = rk
            llm_roles[role] = block

    console.print(EMBEDDING_MODALITY_NOTE)
    ename, emodel, ebase, ekey = _ask_model(prompter, PROVIDERS_EMBED, "Embedding")
    embed_block, embed_env = resolve_embedding_choice(ename, model=emodel, base_url=ebase)
    if embed_block["dim"] == 0:
        embed_block["dim"] = int(prompter.text("Embedding dimension (dim)", default="1024"))
    if ekey is None:
        embed_block["api_key"] = None
        remove_env_keys.append(embed_env)
    else:
        env_values[embed_env] = ekey
    embed_spec = PROVIDERS_EMBED[ename]
    if embed_spec.gui_binding is None or embed_spec.gui_host is None:
        raise ValueError(f"{ename} is missing LightRAG GUI embedding metadata")
    env_values.update(
        {
            "LIGHTRAG_GUI_EMBEDDING_BINDING": embed_spec.gui_binding,
            "LIGHTRAG_GUI_EMBEDDING_HOST": embed_block.get("base_url") or embed_spec.gui_host,
            "LIGHTRAG_GUI_EMBEDDING_MODEL": embed_block["model"],
            "LIGHTRAG_GUI_EMBEDDING_DIM": str(embed_block["dim"]),
        }
    )

    rerank_choice = "Reuse my LLM"
    if mode == MODEL_MODE_CUSTOM:
        rerank_choice = prompter.select("Reranker", list(RERANK_CHOICES))
    rerank_base_url = None
    if rerank_choice == "Azure Cohere":
        rerank_base_url = _ask_required(lambda: prompter.text("Azure Cohere endpoint (required)"))
    rerank_block, rerank_env = resolve_rerank_choice(
        rerank_choice,
        base_url=rerank_base_url,
    )
    if rerank_env is not None:
        env_values[rerank_env] = _ask_required(
            lambda: prompter.password("Reranker API key (required)")
        )

    previous_embedding_dim = _configured_embedding_dim(CONFIG_PATH, ENV_PATH)
    embedding_dim_changed = previous_embedding_dim not in (None, embed_block["dim"])
    if embedding_dim_changed and not prompter.confirm(
        EMBEDDING_DIM_RESET_CONFIRM,
        default=False,
    ):
        return None

    # Confirm AFTER collecting answers, right before overwriting existing settings.
    if require_confirm and not prompter.confirm(MODELS_OVERWRITE_CONFIRM, default=False):
        return None

    # Back up config and any pre-existing .env only now — right before writing —
    # so aborted or declined runs never leave a stray .bak behind.
    config_backup = backup_file(CONFIG_PATH)
    env_existed = ENV_PATH.exists()
    env_backup = backup_file(ENV_PATH) if env_existed else None
    if not env_existed and ENV_EXAMPLE_PATH.exists():
        ENV_PATH.write_bytes(ENV_EXAMPLE_PATH.read_bytes())

    write_config_yaml(
        CONFIG_PATH,
        llm_default=llm_block,
        llm_roles=llm_roles,
        embedding=embed_block,
        rerank=rerank_block,
    )
    selected_role_keys = {
        LLM_ROLE_ENV_KEYS[role] for role in llm_roles if LLM_ROLE_ENV_KEYS[role] in env_values
    }
    stale_role_keys = tuple(
        key for key in LLM_ROLE_ENV_KEYS.values() if key not in selected_role_keys
    )
    remove_env_keys.extend(stale_role_keys)
    if rerank_env is None:
        remove_env_keys.append("DLIGHTRAG_MODELS__RERANK__API_KEY")
    upsert_env(ENV_PATH, env_values, remove_keys=tuple(dict.fromkeys(remove_env_keys)))
    return {
        "llm": {"api_key": key, "base_url": llm_block.get("base_url"), "model": model},
        "config_backup": config_backup,
        "env_backup": env_backup,
        "env_existed": env_existed,
        "embedding_dim_changed": embedding_dim_changed,
    }


def _dotenv_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.lstrip().startswith("#") or "=" not in raw:
            continue
        candidate, value = raw.split("=", 1)
        if candidate.strip() == key:
            return value
    return None


def legacy_web_settings_present(
    *,
    config_path: Path = CONFIG_PATH,
    env_path: Path = ENV_PATH,
) -> bool:
    """Return whether setup can offer the explicit one-time Web migration."""
    if _dotenv_value(env_path, "DLIGHTRAG_ANSWER__WEB_SEARCH__API_KEY") is not None:
        return True
    if not config_path.exists():
        return False
    data = _yaml().load(config_path) or {}
    answer = data.get("answer") or {}
    return isinstance(answer, dict) and "web_search" in answer


def migrate_legacy_web_settings(
    *,
    config_path: Path = CONFIG_PATH,
    env_path: Path = ENV_PATH,
) -> bool:
    """Transactionally migrate the legacy Exa setting and keep secrets in ``.env``."""
    old_env = "DLIGHTRAG_ANSWER__WEB_SEARCH__API_KEY"
    new_env = "DLIGHTRAG_ANSWER__WEB_SOURCES__EXA__API_KEY"
    old_value = _dotenv_value(env_path, old_env)
    yaml = _yaml()
    data = yaml.load(config_path) if config_path.exists() else {}
    data = data or {}
    answer = data.get("answer") or {}
    had_legacy = isinstance(answer, dict) and "web_search" in answer
    legacy = answer.pop("web_search", None) if had_legacy else None
    if old_value is None and not had_legacy:
        return False

    inline_value = legacy.get("api_key") if isinstance(legacy, dict) else None
    migrated_value = old_value
    if migrated_value is None and isinstance(inline_value, str) and inline_value.strip():
        migrated_value = inline_value.strip()
    env_values = (
        {new_env: migrated_value}
        if migrated_value is not None and _dotenv_value(env_path, new_env) is None
        else {}
    )
    if had_legacy:
        web = answer.setdefault("web_sources", {})
        exa = web.setdefault("exa", {})
        exa["api_key"] = None

    config_existed = config_path.exists()
    env_existed = env_path.exists()
    config_backup = backup_file(config_path) if config_existed else None
    env_backup = backup_file(env_path) if env_existed else None
    try:
        if old_value is not None or env_values:
            upsert_env(env_path, env_values, remove_keys=(old_env,))
        if had_legacy:
            yaml.dump(data, config_path)
        _load_effective_config(config_path, env_path)
    except BaseException:
        _restore_file(config_path, config_existed, config_backup)
        _restore_file(env_path, env_existed, env_backup)
        raise
    return True


def run_web_research_step(
    prompter: Prompter,
    *,
    optional: bool,
    require_confirm: bool = False,
) -> bool:
    """Configure first-class REST Web providers without remote key verification."""
    choices = ([WEB_SKIP] if optional else []) + [WEB_EXA, WEB_TAVILY, WEB_BOTH, WEB_DISABLE]
    choice = prompter.select("Web research provider · 网络研究服务", choices)
    if choice == WEB_SKIP:
        return False
    if require_confirm and not prompter.confirm(
        "Overwrite Web research settings and keys? · 覆盖网络研究设置与密钥？",
        default=False,
    ):
        return False

    providers: list[str]
    if choice == WEB_DISABLE:
        providers = []
    elif choice == WEB_EXA:
        providers = ["exa"]
    elif choice == WEB_TAVILY:
        providers = ["tavily"]
    else:
        providers = ["exa", "tavily"]

    search_order = list(providers)
    extract_order = list(providers)
    if len(providers) == 2:
        primary = prompter.select(
            "Provider order · 服务顺序",
            [WEB_COMMON_EXA, WEB_COMMON_TAVILY, WEB_INDEPENDENT],
        )
        if primary == WEB_COMMON_TAVILY:
            search_order = extract_order = ["tavily", "exa"]
        elif primary == WEB_INDEPENDENT:
            search_first = prompter.select("Search first · 搜索优先", [WEB_EXA, WEB_TAVILY])
            extract_first = prompter.select("Extract first · 网页提取优先", [WEB_EXA, WEB_TAVILY])
            search_order = [
                search_first.lower(),
                WEB_TAVILY.lower() if search_first == WEB_EXA else WEB_EXA.lower(),
            ]
            extract_order = [
                extract_first.lower(),
                WEB_TAVILY.lower() if extract_first == WEB_EXA else WEB_EXA.lower(),
            ]

    env_values: dict[str, str] = {}
    remove = [
        "DLIGHTRAG_ANSWER__WEB_SEARCH__API_KEY",
        "DLIGHTRAG_ANSWER__WEB_SOURCES__EXA__API_KEY",
        "DLIGHTRAG_ANSWER__WEB_SOURCES__TAVILY__API_KEY",
    ]
    if "exa" in providers:
        env_values["DLIGHTRAG_ANSWER__WEB_SOURCES__EXA__API_KEY"] = _ask_required(
            lambda: prompter.password("Exa API key (required)")
        )
    if "tavily" in providers:
        env_values["DLIGHTRAG_ANSWER__WEB_SOURCES__TAVILY__API_KEY"] = _ask_required(
            lambda: prompter.password("Tavily API key (required)")
        )
    env_existed = ENV_PATH.exists()
    config_existed = CONFIG_PATH.exists()
    env_backup = backup_file(ENV_PATH) if env_existed else None
    config_backup = backup_file(CONFIG_PATH) if config_existed else None
    try:
        if not ENV_PATH.exists() and ENV_EXAMPLE_PATH.exists():
            ENV_PATH.write_bytes(ENV_EXAMPLE_PATH.read_bytes())
        upsert_env(
            ENV_PATH,
            env_values,
            remove_keys=tuple(key for key in remove if key not in env_values),
        )
        write_config_yaml(
            CONFIG_PATH,
            web_sources={
                "search_providers": search_order,
                "extract_providers": extract_order,
                "exa": {"api_key": None},
                "tavily": {"api_key": None},
            },
        )
        validate_config()
    except BaseException:
        _restore_file(ENV_PATH, env_existed, env_backup)
        _restore_file(CONFIG_PATH, config_existed, config_backup)
        raise
    return True


class SetupCancelled(Exception):
    """Raised when the user picks the in-menu Quit option (a clean, non-error exit)."""


QUIT_CHOICE = "✕ Quit · 退出"


# Home menu — shown only when DlightRAG is already configured (see is_configured).
MENU_START = "Start DlightRAG · 启动"
MENU_CHANGE = "Change settings · 修改设置"
MENU_SHOW = "Show settings · 查看设置"
MENU_RESET = "Start over · 重新配置"
HOME_CHOICES = [MENU_SHOW, MENU_START, MENU_CHANGE, MENU_RESET]
HOME_PROMPT = "DlightRAG is already set up — what next? · DlightRAG 已配置，接下来做什么？"

# "Change settings" sub-menu (section-level, per the design).
SEC_MODELS = "Models & API keys · 模型与密钥"
SEC_PARSER = "Document parser · 文档解析器"
SEC_WEB = "Web research · 网络研究"
SEC_ALL = "Everything · 全部"
SEC_BACK = "← Back · 返回"
CHANGE_CHOICES = [SEC_MODELS, SEC_PARSER, SEC_WEB, SEC_ALL, SEC_BACK]
CHANGE_PROMPT = "What do you want to change? · 你想修改什么？"

MODEL_MODE_MINIMUM = "Minimum · one LLM + one embedding"
MODEL_MODE_CUSTOM = "Custom · separate extraction/keyword models"
MODEL_MODE_CHOICES = [MODEL_MODE_MINIMUM, MODEL_MODE_CUSTOM]
MODEL_MODE_PROMPT = "Model setup mode · 模型配置模式"

# Shown before the embedding provider list: the choice silently decides whether
# fused visual retrieval is available at all.
EMBEDDING_MODALITY_NOTE = (
    "[dim]A native-fusion model (Voyage multimodal 3.5, Gemini Embedding 2, "
    "Jina v4, or Cohere Embed v4) upgrades each visual chunk to one canonical "
    "text+image vector. A text-only model still works, but visual evidence is then "
    "retrieved through its VLM description alone.[/dim]"
)

MODELS_OVERWRITE_CONFIRM = (
    "Overwrite your current model settings and API keys with these answers? · "
    "用这些答案覆盖当前的模型设置与密钥？"
)
WEB_SKIP = "Skip Web research · 暂不配置"
WEB_EXA = "Exa"
WEB_TAVILY = "Tavily"
WEB_BOTH = "Exa + Tavily"
WEB_DISABLE = "Disable Web research · 停用网络研究"
WEB_COMMON_EXA = "Exa first for Search and Extract"
WEB_COMMON_TAVILY = "Tavily first for Search and Extract"
WEB_INDEPENDENT = "Choose Search and Extract order independently"

PARSER_OVERWRITE_CONFIRM = (
    "Overwrite your current document-parser settings? · 覆盖当前的文档解析器设置？"
)
EMBEDDING_DIM_RESET_CONFIRM = (
    "The embedding dimension changed. Apply it and permanently reset all Corpus data? · "
    "嵌入维度已变化。应用并永久清空所有语料数据？"
)
RESET_WIPE_CONFIRM = (
    "Delete ALL documents you've already added (vectors, graph)? This cannot be undone. · "
    "删除所有已导入的文档（向量、图谱）？此操作不可恢复"
)
START_OVER_APPLY_CONFIRM = (
    "Replace model and document-parsing settings from scratch? · 从头替换模型与文档解析设置？"
)


def with_quit(choices: list[str]) -> list[str]:
    """Append the Quit sentinel so every menu offers a no-Ctrl+C way out."""
    return [*choices, QUIT_CHOICE]


def check_quit(answer: str) -> str:
    """Turn a Quit selection into a clean SetupCancelled; pass everything else through."""
    if answer == QUIT_CHOICE:
        raise SetupCancelled
    return answer


def _questionary_prompter() -> Prompter:
    import questionary

    # unsafe_ask(): let Ctrl+C / Ctrl+D propagate as KeyboardInterrupt / EOFError
    # (caught once in main) instead of silently returning None and crashing later.
    class _Q(Prompter):
        def select(self, message: str, choices: list[str]) -> str:
            return check_quit(questionary.select(message, choices=with_quit(choices)).unsafe_ask())

        def text(self, message: str, default: str = "") -> str:
            return questionary.text(message, default=default).unsafe_ask()

        def password(self, message: str) -> str:
            return questionary.password(message).unsafe_ask()

        def confirm(self, message: str, default: bool = False) -> bool:
            return questionary.confirm(message, default=default).unsafe_ask()

    return _Q()


# ---------------------------------------------------------------------------
# Docker bring-up + readiness poll
# ---------------------------------------------------------------------------
def docker_up_command(*, profile: str | None = None) -> list[str]:
    command = ["docker", "compose"]
    if profile is not None:
        command.extend(["--profile", profile])
    return [*command, "up", "-d"]


def probe_readiness(url: str, *, opener=None) -> bool:
    import urllib.request

    opener = opener or urllib.request.urlopen
    try:
        with opener(url, timeout=5) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:
        return False


def wait_for_readiness(
    url: str, *, attempts=60, delay=2.0, probe=probe_readiness, sleep=None
) -> bool:
    sleep = sleep or time.sleep
    for i in range(attempts):
        if probe(url):
            return True
        if i < attempts - 1:
            sleep(delay)
    return False


# ---------------------------------------------------------------------------
# Re-run menu: view / change / start over (shown only when already configured)
# ---------------------------------------------------------------------------
def is_configured(
    env_path: Path | None = None,
    *,
    config_path: Path | None = None,
) -> bool:
    """Return whether one valid config has explicit keyed or keyless model auth."""
    try:
        config = _load_effective_config(config_path or CONFIG_PATH, env_path or ENV_PATH)
    except Exception:
        return False
    default_ready = config.models.chat.default.has_explicit_auth
    embedding = config.models.embedding
    embedding_ready = "api_key" in embedding.model_fields_set and (
        embedding.api_key is None or bool(embedding.api_key.strip())
    )
    return default_ready and embedding_ready


def _capacity_summary(block: dict) -> dict:
    profile = catalog_model_profile(block)
    if profile is not None:
        return {"source": "catalog", **profile}
    return {
        "source": "fallback",
        "context_window_tokens": 1_048_576,
        "max_input_tokens": None,
        "max_output_tokens": 262_144,
    }


def _public_model_block(settings) -> dict:
    return {
        "provider": settings.provider,
        "model": settings.model,
        "base_url": settings.base_url,
    }


def read_config_summary(config_path: Path, env_path: Path) -> dict:
    """Build a display-ready, secret-free summary from effective canonical settings."""
    config = _load_effective_config(config_path, env_path)
    default_settings = config.models.chat.default
    role_settings = dict(config.models.chat.overrides)
    default = _public_model_block(default_settings)
    roles = {role: _public_model_block(settings) for role, settings in role_settings.items()}
    embedding = config.models.embedding
    rerank = config.models.rerank
    answer = config.answer.generation
    sidecars = config.corpus.sidecars
    if sidecars.active_parser == "mineru":
        parser = {
            "name": "MinerU",
            "detail": sidecars.mineru.api_mode if sidecars.mineru is not None else "local",
        }
    else:
        parser = {
            "name": "Docling",
            "detail": sidecars.docling.endpoint if sidecars.docling is not None else "?",
        }
    inspection_settings = role_settings.get("vlm", default_settings)
    return {
        "llm_default": default,
        "llm_roles": roles,
        "model_capacities": {
            "default": _capacity_summary(default),
            **{role: _capacity_summary(block) for role, block in roles.items()},
        },
        "embedding": {
            "provider": embedding.provider,
            "model": embedding.model,
            "dim": embedding.dim,
            "base_url": embedding.base_url,
        },
        "rerank": {
            "strategy": rerank.strategy,
            "enabled": rerank.enabled,
            "model": rerank.model,
            "base_url": rerank.base_url,
        },
        "answer": {
            "max_attachments": answer.max_attachments,
            "max_attachment_bytes": answer.max_attachment_bytes,
            "max_total_attachment_bytes": answer.max_total_attachment_bytes,
            "max_images": answer.max_images,
        },
        "visual_inspection": {
            "role": "vlm" if "vlm" in role_settings else "default",
            "provider": inspection_settings.provider,
            "model": inspection_settings.model,
        },
        "parser": parser,
        "web_research": {
            "direct_url_read": True,
            "search": list(config.answer.web_sources.search_order()),
            "extract": list(config.answer.web_sources.extract_order()),
        },
        "workspace": config.deployment.workspace,
        "keys_set": {
            "LLM": bool(default_settings.api_key),
            "Embedding": bool(embedding.api_key),
            "Rerank": bool(rerank.api_key),
            "Exa": bool(config.answer.web_sources.exa.api_key),
            "Tavily": bool(config.answer.web_sources.tavily.api_key),
        },
    }


def render_summary(console, summary: dict) -> None:
    from rich.table import Table

    table = Table(title="Current settings · 当前配置", show_header=False)
    default = summary["llm_default"]
    table.add_row("LLM", f"{default['provider']} · {default['model']}")
    if default.get("base_url"):
        table.add_row("", f"[dim]{default['base_url']}[/dim]")
    table.add_row("  capacity", _capacity_label(summary["model_capacities"]["default"]))
    for role, block in summary["llm_roles"].items():
        table.add_row(f"  • {role}", f"{block['provider']} · {block['model']}")
        if block.get("base_url"):
            table.add_row("", f"[dim]{block['base_url']}[/dim]")
        table.add_row("    capacity", _capacity_label(summary["model_capacities"][role]))
    embedding = summary["embedding"]
    table.add_row(
        "Embedding", f"{embedding['provider']} · {embedding['model']} (dim {embedding['dim']})"
    )
    if embedding.get("base_url"):
        table.add_row("", f"[dim]{embedding['base_url']}[/dim]")
    rerank = summary["rerank"]
    rerank_model = f" · {rerank['model']}" if rerank.get("model") else ""
    rerank_state = "on" if rerank["enabled"] else "off"
    table.add_row("Rerank", f"{rerank['strategy']}{rerank_model} ({rerank_state})")
    if rerank.get("base_url"):
        table.add_row("", f"[dim]{rerank['base_url']}[/dim]")
    parser = summary["parser"]
    table.add_row("Parser", f"{parser['name']} · {parser['detail']}")
    web = summary["web_research"]
    table.add_row(
        "Web research",
        f"Direct URL read: available · Search: {', '.join(web['search']) or 'off'} · "
        f"Extract: {', '.join(web['extract']) or 'off'}",
    )
    answer = summary["answer"]
    per_mib = answer["max_attachment_bytes"] // (1024 * 1024)
    total_mib = answer["max_total_attachment_bytes"] // (1024 * 1024)
    table.add_row(
        "Answer attachments",
        f"{answer['max_attachments']} max · ≤ {per_mib} MiB each · ≤ {total_mib} MiB total",
    )
    table.add_row(
        "Answer images",
        f"{answer['max_images']} max",
    )
    inspection = summary["visual_inspection"]
    table.add_row(
        "Visual inspection",
        f"{inspection['role']} · {inspection['provider']} · {inspection['model']}",
    )
    table.add_row("Workspace", summary["workspace"])
    table.add_row(
        "API keys",
        "   ".join(
            f"{name}: {'set ✓' if ok else 'missing ✗'}" for name, ok in summary["keys_set"].items()
        ),
    )
    console.print(table)


def _capacity_label(capacity: dict) -> str:
    context = f"C {int(capacity['context_window_tokens']):,}"
    max_input = capacity.get("max_input_tokens")
    max_output = capacity.get("max_output_tokens")
    input_label = f"I {int(max_input):,}" if max_input is not None else "I = C"
    output_label = f"O {int(max_output):,}" if max_output is not None else "O omitted"
    return f"{capacity['source']} · {context} · {input_label} · {output_label}"


def _prepare_compose_bind_sources(
    *,
    env_path: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Create operator-owned bind sources before Compose can run as root."""
    env_path = ENV_PATH if env_path is None else env_path
    environment = os.environ if environment is None else environment
    configured = environment.get("COMPOSE_GLOBAL_SKILLS_DIR") or _dotenv_value(
        env_path, "COMPOSE_GLOBAL_SKILLS_DIR"
    )
    raw_path = configured.strip() if configured else ""
    if len(raw_path) >= 2 and raw_path[0] == raw_path[-1] and raw_path[0] in {'"', "'"}:
        raw_path = raw_path[1:-1]
    if raw_path:
        path = Path(os.path.expandvars(raw_path)).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
    else:
        path = Path.home() / ".dlightrag" / "skills"
    path.mkdir(parents=True, exist_ok=True)
    return path.absolute()


def _bring_up_stack(console, *, profile: str | None = None) -> int:
    console.print("Starting DlightRAG + PostgreSQL… · 正在启动…")
    try:
        _prepare_compose_bind_sources()
        _default_runner(docker_up_command(profile=profile))
    except Exception as exc:
        console.print(f"[red]docker compose up failed:[/red] {exc}")
        return 1
    if wait_for_readiness(API_READY_URL):
        console.print(f"[green]Ready![/green] Open [link={WEB_URL}]{WEB_URL}[/link] · 已就绪")
    else:
        console.print(
            f"[yellow]Not ready yet — check `docker compose ps`, then open[/yellow] {WEB_URL}"
        )
    return 0


def _restore_model_changes(result: dict) -> None:
    backup = result.get("config_backup")
    if backup is not None:
        CONFIG_PATH.write_bytes(backup.read_bytes())
    env_backup = result.get("env_backup")
    if env_backup is not None:
        ENV_PATH.write_bytes(env_backup.read_bytes())
    elif result.get("env_existed") is False:
        ENV_PATH.unlink(missing_ok=True)


def _apply_and_validate(console, result: dict) -> bool:
    """Validate model settings and enforce the required reset after dimension changes."""
    try:
        validate_config()
    except Exception as exc:
        _restore_model_changes(result)
        console.print(f"[red]Config invalid; restored backup:[/red] {exc}")
        return False
    if result.get("embedding_dim_changed"):
        if not _wipe_data(console):
            _restore_model_changes(result)
            console.print("[red]Corpus reset failed; restored model settings.[/red]")
            return False
        result["data_wiped"] = True
    return True


def run_first_time_setup(
    console,
    prompter: Prompter,
    info: PlatformInfo,
    *,
    require_confirm: bool = False,
    launch: bool = True,
    outcome: dict | None = None,
) -> int | None:
    result = run_models_step(prompter, require_confirm=require_confirm)
    if result is None:
        console.print("No changes made. · 未做任何更改")
        return None
    if not _apply_and_validate(console, result):
        return 1
    if outcome is not None:
        outcome["data_wiped"] = bool(result.get("data_wiped"))
    parser_mode = run_parser_step(
        prompter,
        info,
        has_gpu=has_nvidia_gpu(),
        llm_title_aided=result["llm"],
        require_confirm=require_confirm,
    )
    run_web_research_step(
        prompter,
        optional=True,
        require_confirm=require_confirm,
    )
    return (
        _bring_up_stack(console, profile="docling" if parser_mode == "docling" else None)
        if launch
        else 0
    )


def run_change_settings(console, prompter: Prompter, info: PlatformInfo) -> None:
    section = prompter.select(CHANGE_PROMPT, CHANGE_CHOICES)
    if section == SEC_BACK:
        return
    changed = False
    result = None
    if section in (SEC_MODELS, SEC_ALL):
        result = run_models_step(prompter, require_confirm=True)
        if result is None:
            console.print("No changes made. · 未做任何更改")
            return
        if not _apply_and_validate(console, result):
            return
        changed = True
    if section in (SEC_PARSER, SEC_ALL):
        if run_parser_step(
            prompter,
            info,
            has_gpu=has_nvidia_gpu(),
            llm_title_aided=result["llm"] if result else None,
            require_confirm=True,
        ):
            changed = True
    if section in (SEC_WEB, SEC_ALL):
        if run_web_research_step(prompter, optional=False, require_confirm=True):
            try:
                validate_config()
            except Exception as exc:
                console.print(f"[red]Web research config is invalid:[/red] {exc}")
                return
            changed = True
    if changed:
        console.print(
            "[green]Saved.[/green] Pick 'Start DlightRAG' to (re)launch. · 已保存，选择“启动”重新运行"
        )
    else:
        console.print("No changes made. · 未做任何更改")


def _wipe_data(console, *, runner=_default_runner) -> bool:
    console.print("Erasing ingested data… · 正在清除已导入数据…")
    try:
        runner(["uv", "run", "scripts/reset_workspace.py", "--all", "-y"])
        runner(
            [
                "docker",
                "compose",
                "run",
                "--rm",
                "--no-deps",
                "--entrypoint",
                "sh",
                "dlightrag-api",
                "-c",
                "find /app/dlightrag_storage -mindepth 1 -delete",
            ]
        )
    except Exception as exc:
        console.print(
            f"[yellow]Couldn't erase data automatically ({exc}); "
            f"run `uv run scripts/reset_workspace.py --all` yourself.[/yellow]"
        )
        return False
    return True


def run_start_over(console, prompter: Prompter, info: PlatformInfo) -> int | None:
    console.print(
        "[bold]Start over[/bold] — re-enter settings; nothing changes until you confirm. · "
        "[bold]重新配置[/bold]：重新输入设置，确认前不会改动"
    )
    if not prompter.confirm(START_OVER_APPLY_CONFIRM, default=False):
        console.print("No changes made. · 未做任何更改")
        return None
    outcome: dict = {}
    rc = run_first_time_setup(
        console,
        prompter,
        info,
        require_confirm=False,
        launch=False,
        outcome=outcome,
    )
    if rc != 0:  # None (declined the overwrite) or 1 (invalid config)
        return rc
    # Confirm the optional wipe unless an embedding-dimension change already required it.
    if not outcome.get("data_wiped") and prompter.confirm(RESET_WIPE_CONFIRM, default=False):
        _wipe_data(console)
    return _bring_up_stack(console)


def run_home(console, prompter: Prompter, info: PlatformInfo) -> int:
    while True:
        choice = prompter.select(HOME_PROMPT, HOME_CHOICES)
        if choice == MENU_START:
            return _bring_up_stack(console)
        if choice == MENU_RESET:
            rc = run_start_over(console, prompter, info)
            if rc is not None:
                return rc
        elif choice == MENU_CHANGE:
            run_change_settings(console, prompter, info)
        elif choice == MENU_SHOW:
            render_summary(console, read_config_summary(CONFIG_PATH, ENV_PATH))


def main(prompter: Prompter | None = None) -> int:
    from rich.console import Console

    console = Console()
    console.rule("[bold]DlightRAG setup")
    console.print("[dim]Pick '✕ Quit · 退出' in any menu, or press Ctrl+C, to cancel.[/dim]")
    if prompter is None and not sys.stdin.isatty():
        console.print(
            "[red]Interactive terminal required.[/red] Run "
            "[bold]uv run prerequisite_setup.py[/bold] from a terminal."
        )
        return 2
    failed = [c for c in run_preflight() if not c.ok]
    for c in failed:
        console.print(f"[red]missing[/red] {c.name} — {c.hint}")
    if failed:
        return 1

    prompter = prompter or _questionary_prompter()
    info = detect_platform()
    try:
        if legacy_web_settings_present():
            if not prompter.confirm(
                "Migrate the legacy Exa Web setting now? · 迁移旧版 Exa 网络设置？",
                default=True,
            ):
                raise SetupCancelled
            if migrate_legacy_web_settings():
                console.print("[green]Migrated legacy Exa Web setting.[/green]")
        if is_configured():
            return run_home(console, prompter, info)
        rc = run_first_time_setup(console, prompter, info)
        return 0 if rc is None else rc
    except KeyboardInterrupt, EOFError, SetupCancelled:
        console.print(
            "\n[yellow]Setup cancelled.[/yellow] Re-run any time with "
            "[bold]uv run prerequisite_setup.py[/bold]"
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
