# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit tests for the prerequisite_setup.py onboarding wizard (Plan 1)."""

import importlib.util
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def wiz():
    spec = importlib.util.spec_from_file_location(
        "prerequisite_setup", _ROOT / "prerequisite_setup.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass/typing annotation resolution can find it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _repository_mineru_backup_names() -> set[str]:
    return {path.name for path in _ROOT.glob(".env.mineru.bak-*")}


@pytest.fixture(autouse=True)
def _isolate_wizard_mutable_paths(wiz, tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Keep every setup-managed repository file inside the test sandbox."""
    repository_backups_before = _repository_mineru_backup_names()
    isolated_repo = tmp_path / "setup-repository"
    isolated_repo.mkdir()
    monkeypatch.setattr(wiz, "REPO_ROOT", isolated_repo)
    for name, filename in {
        "CONFIG_PATH": "config.yaml",
        "ENV_PATH": ".env",
        "ENV_EXAMPLE_PATH": ".env.example",
        "MINERU_ENV_PATH": ".env.mineru",
        "MINERU_ENV_EXAMPLE_PATH": ".env.mineru.example",
    }.items():
        monkeypatch.setattr(wiz, name, isolated_repo / filename)

    yield

    assert _repository_mineru_backup_names() == repository_backups_before


class _ScriptedPrompter:
    """Feeds pre-scripted answers to the Models step without a TTY."""

    def __init__(self, answers):
        self._a = list(answers)
        self.select_choices = []

    def select(self, message, choices):
        self.select_choices.append(list(choices))
        return self._a.pop(0)

    def text(self, message, default=""):
        value = self._a.pop(0)
        return value if value != "" else default

    def password(self, message):
        return self._a.pop(0)

    def confirm(self, message, default=False):
        return self._a.pop(0)


def test_module_imports(wiz):
    assert wiz.CONFIG_PATH.name == "config.yaml"
    assert wiz.ENV_PATH.name == ".env"


def test_parser_backups_stay_in_test_sandbox(wiz, monkeypatch: pytest.MonkeyPatch):
    for path in (wiz.CONFIG_PATH, wiz.ENV_PATH, wiz.MINERU_ENV_PATH):
        path.write_text("test-only\n", encoding="utf-8")
    repository_backups_before = _repository_mineru_backup_names()
    monkeypatch.setattr(wiz, "validate_config", lambda: None)

    wiz._apply_parser_change(lambda: None)

    assert _repository_mineru_backup_names() == repository_backups_before
    assert len(list(wiz.MINERU_ENV_PATH.parent.glob(".env.mineru.bak-*"))) == 1


def test_inline_script_environment_installs_the_local_product() -> None:
    source = (_ROOT / "prerequisite_setup.py").read_text(encoding="utf-8")
    assert '# dependencies = ["dlightrag", "dlightrag-memory",' in source
    assert '# dlightrag = { path = ".", editable = true }' in source
    assert '# dlightrag-memory = { path = "packages/memory", editable = true }' in source


def test_memory_compose_healthcheck_uses_configured_database_identity() -> None:
    compose = (_ROOT / "packages/memory/compose.yaml").read_text(encoding="utf-8")
    assert 'pg_isready -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"' in compose


def test_mineru_template_defaults_to_cross_platform_core() -> None:
    template = (_ROOT / ".env.mineru.example").read_text(encoding="utf-8")
    active = [line for line in template.splitlines() if line.startswith("MINERU_INSTALL_EXTRAS=")]
    assert active == ["MINERU_INSTALL_EXTRAS=core"]


def test_local_ci_installs_playwright_chromium() -> None:
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "frontend-browser-install: frontend-install" in makefile
    assert "npx playwright install chromium" in makefile
    assert "frontend-browser-test: frontend-browser-install" in makefile


def test_validation_reads_the_written_repo_config_from_any_cwd(
    wiz, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "config.yaml").write_text("llm:\n  invalid: true\n", encoding="utf-8")
    (repo / ".env").write_text("", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(wiz, "REPO_ROOT", repo)
    monkeypatch.setattr(wiz, "CONFIG_PATH", repo / "config.yaml")
    monkeypatch.setattr(wiz, "ENV_PATH", repo / ".env")
    monkeypatch.chdir(elsewhere)

    with pytest.raises(ValueError, match="Invalid dlightrag configuration"):
        wiz.validate_config()


# --- Task 2: provider registry / resolvers --------------------------------
def test_llm_openai_compatible_mapping(wiz):
    block, env_key = wiz.resolve_llm_choice("DeepSeek", model="deepseek-v4-flash", base_url=None)
    assert block["provider"] == "openai"
    assert block["base_url"] == "https://api.deepseek.com"
    assert block["model"] == "deepseek-v4-flash"
    assert env_key == "DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY"


def test_llm_native_provider_has_no_base_url(wiz):
    block, _ = wiz.resolve_llm_choice("Anthropic", model="claude-4", base_url=None)
    assert block["provider"] == "anthropic"
    assert "base_url" not in block


def test_llm_azure_requires_user_base_url(wiz):
    assert wiz.PROVIDERS_LLM["Azure OpenAI"].requires_url is True
    block, _ = wiz.resolve_llm_choice(
        "Azure OpenAI", model="gpt-4o", base_url="https://x.openai.azure.com/v1"
    )
    assert block["base_url"] == "https://x.openai.azure.com/v1"


def test_known_llm_resolves_capacity_from_shared_catalog(wiz):
    block, _ = wiz.resolve_llm_choice(
        "OpenRouter",
        model="google/gemini-3.7-flash",
        base_url=None,
    )

    profile = wiz.catalog_model_profile(block)

    assert profile["context_window_tokens"] == 1_048_576
    assert profile["max_output_tokens"] == 65_536


def test_malformed_endpoint_is_unknown_instead_of_crashing_catalog_resolution(wiz):
    block = {
        "provider": "openai",
        "model": "private-model",
        "base_url": "https://example.com:not-a-port/v1",
    }

    assert wiz.catalog_model_profile(block) is None


def test_unknown_llm_uses_permissive_fallback_without_config_override(wiz):
    block, _ = wiz.resolve_llm_choice(
        "Other (OpenAI-compatible)",
        model="private-model",
        base_url="http://localhost:8888/v1",
    )

    assert wiz._capacity_summary(block) == {
        "source": "fallback",
        "context_window_tokens": 1_048_576,
        "max_input_tokens": None,
        "max_output_tokens": 262_144,
    }


def test_embedding_mapping_prefills_dim(wiz):
    block, env_key = wiz.resolve_embedding_choice(
        "Voyage", model="voyage-multimodal-3.5", base_url=None
    )
    assert block["provider"] == "voyage"
    assert block["dim"] == 1024
    assert env_key == "DLIGHTRAG_MODELS__EMBEDDING__API_KEY"


def test_azure_cohere_rerank_records_its_required_endpoint(wiz):
    block, env_key = wiz.resolve_rerank_choice(
        "Azure Cohere",
        base_url="https://example.services.ai.azure.com/models",
    )
    assert block == {
        "strategy": "azure_cohere",
        "base_url": "https://example.services.ai.azure.com/models",
    }
    assert env_key == "DLIGHTRAG_MODELS__RERANK__API_KEY"


def test_rerank_reuse_llm_needs_no_key(wiz):
    block, env_key = wiz.resolve_rerank_choice("Reuse my LLM")
    assert block["strategy"] == "chat_llm_reranker"
    assert env_key is None


def test_ask_model_reprompts_for_required_values(wiz):
    # Azure has no default model and demands its own URL, so every answer is required.
    prompter = _ScriptedPrompter(
        [
            "Azure OpenAI",
            "",
            "gpt-4o",
            "",
            "https://x.openai.azure.com/v1",
            "",
            "sk-valid",
        ]
    )

    assert wiz._ask_model(prompter, wiz.PROVIDERS_LLM, "LLM") == (
        "Azure OpenAI",
        "gpt-4o",
        "https://x.openai.azure.com/v1",
        "sk-valid",
    )


def test_ask_model_accepts_provider_default_model_and_url(wiz):
    # Empty answers mean "keep what you were shown" for a known vendor.
    prompter = _ScriptedPrompter(["DeepSeek", "", "", "sk-valid"])

    assert wiz._ask_model(prompter, wiz.PROVIDERS_LLM, "LLM") == (
        "DeepSeek",
        "deepseek-v4-flash",
        "https://api.deepseek.com",
        "sk-valid",
    )


def test_embedding_defaults_are_first_class_models_with_known_dimensions(wiz):
    defaults = {
        name: spec.default_model for name, spec in wiz.PROVIDERS_EMBED.items() if spec.default_model
    }
    assert defaults == {
        "Voyage": "voyage-multimodal-3.5",
        "OpenAI": "text-embedding-3-large",
        "Gemini": "gemini-embedding-2",
        "Jina": "jina-embeddings-v4",
        "Cohere": "embed-v4.0",
        "Azure Cohere": "Cohere-embed-v4",
    }
    assert all(wiz.EMBED_DIMS.get(model) for model in defaults.values())


def test_tenant_specific_models_are_explicit_except_fixed_azure_cohere_offer(wiz):
    for providers in (wiz.PROVIDERS_LLM, wiz.PROVIDERS_EMBED):
        for name, spec in providers.items():
            if spec.requires_url and name != "Azure Cohere":
                assert spec.default_model == "", name
    assert wiz.PROVIDERS_EMBED["Azure Cohere"].default_model == "Cohere-embed-v4"


def test_ask_model_reprompts_for_required_custom_url(wiz):
    prompter = _ScriptedPrompter(
        ["Azure OpenAI", "gpt-4o", "", "https://example.openai.azure.com/v1", "sk-azure"]
    )

    assert wiz._ask_model(prompter, wiz.PROVIDERS_LLM, "LLM") == (
        "Azure OpenAI",
        "gpt-4o",
        "https://example.openai.azure.com/v1",
        "sk-azure",
    )


def test_ask_model_returns_none_for_unauthenticated_local_provider(wiz):
    prompter = _ScriptedPrompter(
        ["Other (OpenAI-compatible)", "local-model", "http://localhost:8000/v1", ""]
    )

    assert wiz._ask_model(prompter, wiz.PROVIDERS_LLM, "LLM") == (
        "Other (OpenAI-compatible)",
        "local-model",
        "http://localhost:8000/v1",
        None,
    )


# --- Task 3: config.yaml writer -------------------------------------------
def test_write_config_rejects_legacy_schema_with_actionable_message(wiz, tmp_path):
    src = tmp_path / "config.yaml"
    src.write_text("llm:\n  default:\n    model: old\n", encoding="utf-8")

    with pytest.raises(ValueError, match="3.0 nine-section schema"):
        wiz.write_config_yaml(src, llm_default={"model": "new"})


def test_write_config_preserves_comments_and_updates(wiz, tmp_path):
    src = tmp_path / "config.yaml"
    src.write_text(
        "# curated header comment\n"
        "models:\n"
        "  chat:\n"
        "    default:\n"
        "      provider: openai  # inline note\n"
        "      model: old-model\n"
        "      base_url: https://old\n"
        "  embedding:\n"
        "    provider: voyage\n"
        "    model: old-embed\n"
        "    dim: 1024\n",
        encoding="utf-8",
    )
    wiz.write_config_yaml(
        src,
        llm_default={
            "provider": "openai",
            "model": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com",
        },
        embedding={
            "provider": "voyage",
            "model": "voyage-multimodal-3.5",
            "base_url": "https://api.voyageai.com/v1",
            "dim": 1024,
        },
    )
    text = src.read_text(encoding="utf-8")
    assert "# curated header comment" in text
    assert "# inline note" in text
    assert "deepseek-v4-flash" in text
    assert "old-model" not in text
    data = wiz._yaml().load(text)
    assert data["answer"]["agent"] == {
        "execution_environment": "trust",
        "workspace_root": None,
        "outbound_mcp": [],
    }


def test_write_config_replaces_stale_role_blocks_when_roles_are_explicit(wiz, tmp_path):
    src = tmp_path / "config.yaml"
    src.write_text(
        "models:\n"
        "  chat:\n"
        "    default:\n      provider: openai\n      model: default\n"
        "    roles:\n"
        "      extract:\n        provider: openai\n        model: stale-extract\n"
        "      vlm:\n        provider: openai\n        model: stale-vlm\n"
        "  embedding:\n    provider: voyage\n    model: embed\n    dim: 1024\n",
        encoding="utf-8",
    )

    wiz.write_config_yaml(
        src,
        llm_roles={
            "keyword": {
                "provider": "openai",
                "model": "cheap-keyword",
                "base_url": "https://api.deepseek.com",
            }
        },
    )

    text = src.read_text(encoding="utf-8")
    assert "keyword:" in text
    assert "cheap-keyword" in text
    assert "stale-extract" not in text
    assert "stale-vlm" not in text


def test_write_config_keyed_role_removes_stale_keyless_yaml(wiz, tmp_path):
    src = tmp_path / "config.yaml"
    src.write_text(
        "models:\n"
        "  chat:\n"
        "    default:\n      provider: openai\n      model: default\n"
        "    roles:\n"
        "      extract:\n        provider: openai\n        model: local\n        api_key: null\n"
        "  embedding:\n    provider: voyage\n    model: embed\n    dim: 1024\n",
        encoding="utf-8",
    )

    wiz.write_config_yaml(
        src,
        llm_roles={
            "extract": {
                "provider": "openai",
                "model": "keyed-extract",
                "base_url": "https://api.deepseek.com",
            }
        },
    )

    role = wiz._yaml().load(src)["models"]["chat"]["roles"]["extract"]
    assert role["model"] == "keyed-extract"
    assert "api_key" not in role


def test_write_config_selects_docling_and_removes_mineru(wiz, tmp_path):
    src = tmp_path / "config.yaml"
    src.write_text(
        "# parser comment\n"
        "corpus:\n"
        '  parser:\n    rules: "*:mineru-iteP"\n'
        "  sidecars:\n"
        "    mineru:\n      api_mode: local\n      language: ch\n",
        encoding="utf-8",
    )

    wiz.write_config_yaml(
        src,
        parser_kind="docling",
        docling_endpoint="https://docling.example.com",
    )

    data = wiz._yaml().load(src)
    assert "parser" not in data["corpus"]
    assert data["corpus"]["sidecars"] == {
        "docling": {
            "endpoint": "https://docling.example.com",
            "code_formula_preset": None,
        }
    }
    assert "# parser comment" in src.read_text(encoding="utf-8")


def test_write_config_clears_stale_mps_preset_for_cpu_service(wiz, tmp_path):
    src = tmp_path / "config.yaml"
    src.write_text(
        "corpus:\n"
        "  sidecars:\n"
        "    docling:\n"
        "      endpoint: http://host.docker.internal:5001\n"
        "      code_formula_preset: granite_docling\n",
        encoding="utf-8",
    )

    wiz.write_config_yaml(
        src,
        parser_kind="docling",
        docling_endpoint="https://docling.example.com",
    )

    assert wiz._yaml().load(src)["corpus"]["sidecars"]["docling"] == {
        "endpoint": "https://docling.example.com",
        "code_formula_preset": None,
    }


def test_write_config_selects_mineru_and_removes_docling(wiz, tmp_path):
    src = tmp_path / "config.yaml"
    src.write_text(
        "corpus:\n"
        '  parser:\n    rules: "*:docling-iteP"\n'
        "  sidecars:\n    docling:\n      endpoint: http://docling:5001\n",
        encoding="utf-8",
    )

    wiz.write_config_yaml(src, parser_kind="mineru", mineru_api_mode="local")

    data = wiz._yaml().load(src)
    assert "parser" not in data["corpus"]
    assert data["corpus"]["sidecars"] == {
        "mineru": {
            "api_mode": "local",
            "local_endpoint": "http://host.docker.internal:8210",
            "language": "ch",
        }
    }


def test_write_config_replaces_native_mineru_endpoint_for_compose(wiz, tmp_path):
    src = tmp_path / "config.yaml"
    src.write_text(
        "corpus:\n"
        "  sidecars:\n"
        "    mineru:\n"
        "      api_mode: local\n"
        "      local_endpoint: http://127.0.0.1:8210\n",
        encoding="utf-8",
    )

    wiz.write_config_yaml(src, parser_kind="mineru", mineru_api_mode="local")

    data = wiz._yaml().load(src)
    assert (
        data["corpus"]["sidecars"]["mineru"]["local_endpoint"] == "http://host.docker.internal:8210"
    )


# --- Task 4: .env upsert ---------------------------------------------------
def test_upsert_env_preserves_and_updates(wiz, tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "EXISTING=keep\nDLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY=old\n", encoding="utf-8"
    )
    wiz.upsert_env(
        env,
        {
            "DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY": "new",
            "DLIGHTRAG_MODELS__EMBEDDING__API_KEY": "e",
        },
    )
    lines = env.read_text(encoding="utf-8").splitlines()
    assert "EXISTING=keep" in lines
    assert "DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY=new" in lines
    assert "DLIGHTRAG_MODELS__EMBEDDING__API_KEY=e" in lines
    assert sum(line.startswith("DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY=") for line in lines) == 1


def test_upsert_env_collapses_duplicate_keys(wiz, tmp_path):
    env = tmp_path / ".env"
    env.write_text("K=stale-first\nK=stale-last\n", encoding="utf-8")

    wiz.upsert_env(env, {"K": "current"})

    assert env.read_text(encoding="utf-8").splitlines() == ["K=current"]


def test_upsert_env_creates_from_missing(wiz, tmp_path):
    env = tmp_path / ".env"
    wiz.upsert_env(env, {"K": "v"})
    assert env.read_text(encoding="utf-8").strip() == "K=v"


# --- Task 5: backup --------------------------------------------------------
def test_backup_file_creates_timestamped_copy(wiz, tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("x: 1\n", encoding="utf-8")
    backup = wiz.backup_file(f)
    assert backup is not None
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == "x: 1\n"
    assert backup.name.startswith("config.yaml.bak-")


def test_backup_file_missing_returns_none(wiz, tmp_path):
    assert wiz.backup_file(tmp_path / "nope.yaml") is None


def test_backup_file_keeps_only_latest(wiz, tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("current\n", encoding="utf-8")
    # Two pre-existing older backups (older timestamps sort first).
    (tmp_path / "config.yaml.bak-20200101000000").write_text("old0\n", encoding="utf-8")
    (tmp_path / "config.yaml.bak-20200101000001").write_text("old1\n", encoding="utf-8")
    backup = wiz.backup_file(f)
    assert backup is not None
    remaining = sorted(p.name for p in tmp_path.glob("config.yaml.bak-*"))
    assert remaining == [backup.name]  # only the freshly-created backup survives
    assert backup.read_text(encoding="utf-8") == "current\n"


# --- Task 6: detection + preflight ----------------------------------------
def test_detect_platform_apple_silicon(wiz, monkeypatch):
    monkeypatch.setattr(wiz.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(wiz.platform, "machine", lambda: "arm64")
    info = wiz.detect_platform()
    assert info.os == "macos"
    assert info.arch == "arm64"
    assert info.is_wsl is False


def test_detect_platform_linux_wsl(wiz, monkeypatch, tmp_path):
    monkeypatch.setattr(wiz.platform, "system", lambda: "Linux")
    monkeypatch.setattr(wiz.platform, "machine", lambda: "x86_64")
    proc = tmp_path / "version"
    proc.write_text("Linux version 5.x microsoft-standard-WSL2", encoding="utf-8")
    monkeypatch.setattr(wiz, "_PROC_VERSION", proc)
    info = wiz.detect_platform()
    assert info.os == "linux"
    assert info.is_wsl is True


def test_has_nvidia_gpu(wiz, monkeypatch):
    monkeypatch.setattr(
        wiz.shutil,
        "which",
        lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None,
    )
    assert wiz.has_nvidia_gpu() is True


def test_preflight_flags_missing_tool(wiz, monkeypatch):
    monkeypatch.setattr(
        wiz.shutil, "which", lambda name: None if name == "docker" else "/usr/bin/" + name
    )
    checks = wiz.run_preflight()
    docker = next(c for c in checks if c.name == "docker")
    assert docker.ok is False


# --- Task 7: MinerU extras + service model --------------------------------
@pytest.mark.parametrize(
    "os_name,arch,gpu,expected",
    [
        ("macos", "arm64", False, "core,mlx"),
        ("linux", "x86_64", True, "core,vllm"),
        ("linux", "x86_64", False, "core"),
    ],
)
def test_select_mineru_extras(wiz, os_name, arch, gpu, expected):
    info = wiz.PlatformInfo(os=os_name, arch=arch, is_wsl=False)
    assert wiz.select_mineru_extras(info, has_gpu=gpu) == expected


@pytest.mark.parametrize(
    "os_name,is_wsl,systemd,expected",
    [
        ("macos", False, False, "launchd"),
        ("linux", False, True, "systemd-user"),
        ("linux", True, True, "systemd-user"),
        ("linux", False, False, "foreground"),
    ],
)
def test_resolve_service_model(wiz, os_name, is_wsl, systemd, expected):
    info = wiz.PlatformInfo(os=os_name, arch="x86_64", is_wsl=is_wsl)
    assert wiz.resolve_service_model(info, systemd_available=systemd) == expected


# --- Task 8: interactive Models step --------------------------------------
def test_models_step_writes_config_and_env(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n"
        "  chat:\n    default:\n      provider: openai\n      model: x\n      base_url: https://x\n"
        "    roles:\n      extract:\n        provider: openai\n        model: stale-extract\n"
        "      vlm:\n        provider: openai\n        model: stale-vlm\n"
        "  embedding:\n    provider: voyage\n    model: x\n    dim: 1024\n"
        "  rerank:\n    strategy: voyage_reranker\n    model: stale-rerank\n    base_url: https://stale\n",
        encoding="utf-8",
    )
    env = tmp_path / ".env"
    env.write_text(
        "DLIGHTRAG_MODELS__CHAT__ROLES__EXTRACT__API_KEY=old-extract\n"
        "DLIGHTRAG_MODELS__CHAT__ROLES__KEYWORD__API_KEY=old-keyword\n"
        "DLIGHTRAG_MODELS__CHAT__ROLES__QUERY__API_KEY=old-query\n"
        "DLIGHTRAG_MODELS__CHAT__ROLES__VLM__API_KEY=old-vlm\n"
        "DLIGHTRAG_MODELS__RERANK__API_KEY=old-rerank\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", env)
    monkeypatch.setattr(wiz, "ENV_EXAMPLE_PATH", tmp_path / "missing.env.example")
    prompter = _ScriptedPrompter(
        [
            "Minimum · one LLM + one embedding",
            "DeepSeek",
            "deepseek-v4-flash",
            "",
            "sk-llm",  # LLM: provider, model, base_url(default), key
            "Voyage",
            "voyage-multimodal-3.5",
            "",
            "sk-embed",  # embedding: provider, model, base_url(default), key
        ]
    )
    wiz.run_models_step(prompter)
    text = cfg.read_text(encoding="utf-8")
    assert "deepseek-v4-flash" in text
    assert "api.deepseek.com" in text
    assert "chat_llm_reranker" in text
    assert "stale-rerank" not in text
    assert "https://stale" not in text
    assert "roles:" not in text
    assert "stale-extract" not in text
    assert "stale-vlm" not in text
    env_text = env.read_text(encoding="utf-8")
    assert "DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY=sk-llm" in env_text
    assert "DLIGHTRAG_MODELS__EMBEDDING__API_KEY=sk-embed" in env_text
    assert "DLIGHTRAG_MODELS__CHAT__ROLES__EXTRACT__API_KEY" not in env_text
    assert "DLIGHTRAG_MODELS__CHAT__ROLES__KEYWORD__API_KEY" not in env_text
    assert "DLIGHTRAG_MODELS__CHAT__ROLES__QUERY__API_KEY" not in env_text
    assert "DLIGHTRAG_MODELS__CHAT__ROLES__VLM__API_KEY" not in env_text
    assert "DLIGHTRAG_MODELS__RERANK__API_KEY" not in env_text
    assert "LIGHTRAG_GUI_EMBEDDING_BINDING=voyageai" in env_text
    assert "LIGHTRAG_GUI_EMBEDDING_MODEL=voyage-multimodal-3.5" in env_text
    assert "LIGHTRAG_GUI_EMBEDDING_DIM=1024" in env_text


def test_configured_embedding_dimension_honors_env_override(wiz, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n  embedding:\n    provider: openai_compatible\n    model: embed\n    dim: 1024\n",
        encoding="utf-8",
    )
    env = tmp_path / ".env"
    env.write_text("DLIGHTRAG_MODELS__EMBEDDING__DIM=768\n", encoding="utf-8")

    assert wiz._configured_embedding_dim(cfg, env) == 768


def test_models_step_marks_embedding_dimension_change_for_required_reset(
    wiz, tmp_path, monkeypatch
):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n"
        "  chat:\n    default:\n      provider: openai\n      model: old\n"
        "  embedding:\n    provider: openai_compatible\n    model: old\n    dim: 768\n",
        encoding="utf-8",
    )
    env = tmp_path / ".env"
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", env)
    monkeypatch.setattr(wiz, "ENV_EXAMPLE_PATH", tmp_path / "missing")
    prompter = _ScriptedPrompter(
        [
            wiz.MODEL_MODE_MINIMUM,
            "DeepSeek",
            "deepseek-v4-flash",
            "",
            "sk-llm",
            "Voyage",
            "voyage-multimodal-3.5",
            "",
            "sk-embed",
            True,
        ]
    )

    result = wiz.run_models_step(prompter)

    assert result is not None
    assert result["embedding_dim_changed"] is True


def test_models_step_custom_replaces_roles_and_writes_role_env(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n"
        "  chat:\n    default:\n      provider: openai\n      model: x\n      base_url: https://x\n"
        "    roles:\n      vlm:\n        provider: openai\n        model: stale-vlm\n"
        "  embedding:\n    provider: voyage\n    model: x\n    dim: 1024\n"
        "  rerank:\n    strategy: voyage_reranker\n",
        encoding="utf-8",
    )
    env = tmp_path / ".env"
    env.write_text(
        "DLIGHTRAG_MODELS__CHAT__ROLES__EXTRACT__API_KEY=old-extract\n"
        "DLIGHTRAG_MODELS__CHAT__ROLES__KEYWORD__API_KEY=old-keyword\n"
        "DLIGHTRAG_MODELS__CHAT__ROLES__QUERY__API_KEY=old-query\n"
        "DLIGHTRAG_MODELS__CHAT__ROLES__VLM__API_KEY=old-vlm\n"
        "DLIGHTRAG_MODELS__RERANK__API_KEY=old-rerank\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", env)
    monkeypatch.setattr(wiz, "ENV_EXAMPLE_PATH", tmp_path / "missing.env.example")
    prompter = _ScriptedPrompter(
        [
            "Custom · separate extraction/keyword models",
            "OpenRouter",
            "google/gemini-3.7-flash",
            "",
            "sk-llm",
            "DeepSeek",
            "deepseek-v4-flash",
            "",
            "sk-extract",
            "DeepSeek",
            "deepseek-v4-flash",
            "",
            "sk-keyword",
            "Voyage",
            "voyage-multimodal-3.5",
            "",
            "sk-embed",
            "Reuse my LLM",
        ]
    )

    wiz.run_models_step(prompter)

    text = cfg.read_text(encoding="utf-8")
    assert "roles:" in text
    assert "extract:" in text
    assert "keyword:" in text
    assert "stale-vlm" not in text
    env_text = env.read_text(encoding="utf-8")
    assert "DLIGHTRAG_MODELS__CHAT__ROLES__EXTRACT__API_KEY=sk-extract" in env_text
    assert "DLIGHTRAG_MODELS__CHAT__ROLES__KEYWORD__API_KEY=sk-keyword" in env_text
    assert "old-extract" not in env_text
    assert "old-keyword" not in env_text
    assert "DLIGHTRAG_MODELS__CHAT__ROLES__QUERY__API_KEY" not in env_text
    assert "DLIGHTRAG_MODELS__CHAT__ROLES__VLM__API_KEY" not in env_text
    assert "DLIGHTRAG_MODELS__RERANK__API_KEY" not in env_text


def test_models_step_writes_keyless_role_to_yaml_and_removes_stale_env(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n"
        "  chat:\n    default:\n      provider: openai\n      model: old\n"
        "    roles:\n      extract:\n        provider: openai\n        model: old\n"
        "  embedding:\n    provider: voyage\n    model: old\n    dim: 1024\n"
        "  rerank:\n    strategy: chat_llm_reranker\n",
        encoding="utf-8",
    )
    env = tmp_path / ".env"
    env.write_text(
        "DLIGHTRAG_MODELS__CHAT__ROLES__EXTRACT__API_KEY=stale-key\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", env)
    monkeypatch.setattr(wiz, "ENV_EXAMPLE_PATH", tmp_path / "missing.env.example")
    monkeypatch.setattr(wiz, "catalog_model_profile", lambda _block: {"context_window_tokens": 1})
    prompter = _ScriptedPrompter(
        [
            "Custom · separate extraction/keyword models",
            "DeepSeek",
            "default-model",
            "",
            "sk-default",
            "Other (OpenAI-compatible)",
            "local-extract",
            "http://localhost:8000/v1",
            "",
            "DeepSeek",
            "keyword-model",
            "",
            "sk-keyword",
            "Voyage",
            "voyage-multimodal-3.5",
            "",
            "sk-embed",
            "Reuse my LLM",
        ]
    )

    wiz.run_models_step(prompter)

    config = wiz._yaml().load(cfg)
    assert config["models"]["chat"]["roles"]["extract"]["api_key"] is None
    env_text = env.read_text(encoding="utf-8")
    assert "DLIGHTRAG_MODELS__CHAT__ROLES__EXTRACT__API_KEY" not in env_text
    assert "DLIGHTRAG_MODELS__CHAT__ROLES__KEYWORD__API_KEY=sk-keyword" in env_text


# --- Plan 2 Task 1: MinerU config helpers ---------------------------------
def test_configure_mineru_official(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("corpus:\n  sidecars:\n    mineru:\n      api_mode: local\n", encoding="utf-8")
    env = tmp_path / ".env"
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", env)
    wiz.configure_mineru_official("tok-123")
    assert "api_mode: official" in cfg.read_text(encoding="utf-8")
    assert "DLIGHTRAG_CORPUS__SIDECARS__MINERU__API_TOKEN=tok-123" in env.read_text(
        encoding="utf-8"
    )


def test_configure_external_docling_default_uses_host_mps_service(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "corpus:\n  sidecars:\n    mineru:\n      api_mode: local\n",
        encoding="utf-8",
    )
    env = tmp_path / ".env"
    env.write_text("COMPOSE_PROFILES=docling\n", encoding="utf-8")
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", env)

    wiz.configure_docling("http://host.docker.internal:5001", bundled=False)

    assert wiz._yaml().load(cfg)["corpus"]["sidecars"] == {
        "docling": {
            "endpoint": "http://host.docker.internal:5001",
            "code_formula_preset": "granite_docling",
        }
    }
    assert "COMPOSE_PROFILES" not in env.read_text(encoding="utf-8")


def test_configure_mineru_local_env_writes_extras_and_title_aided(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "corpus:\n  sidecars:\n    mineru:\n      api_mode: official\n", encoding="utf-8"
    )
    mineru_env = tmp_path / ".env.mineru"
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(wiz, "MINERU_ENV_PATH", mineru_env)
    monkeypatch.setattr(wiz, "MINERU_ENV_EXAMPLE_PATH", tmp_path / "missing")
    wiz.configure_mineru_local_env(
        "core,mlx",
        title_aided={
            "api_key": "sk",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
        },
    )
    text = mineru_env.read_text(encoding="utf-8")
    assert "MINERU_INSTALL_EXTRAS=core,mlx" in text
    assert "MINERU_TITLE_AIDED_ENABLE=true" in text
    assert "MINERU_TITLE_AIDED_MODEL=deepseek-v4-flash" in text
    assert "api_mode: local" in cfg.read_text(encoding="utf-8")


def test_configure_mineru_local_env_disables_and_scrubs_stale_title_aided(
    wiz, tmp_path, monkeypatch
):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("corpus:\n  sidecars:\n    mineru:\n      api_mode: local\n", encoding="utf-8")
    mineru_env = tmp_path / ".env.mineru"
    mineru_env.write_text(
        "MINERU_INSTALL_EXTRAS=core\n"
        "MINERU_TITLE_AIDED_ENABLE=true\n"
        "MINERU_TITLE_AIDED_API_KEY=stale-secret\n"
        "MINERU_TITLE_AIDED_BASE_URL=https://stale.example\n"
        "MINERU_TITLE_AIDED_MODEL=stale-model\n"
        "MINERU_TITLE_AIDED_ENABLE_THINKING=true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(wiz, "MINERU_ENV_PATH", mineru_env)
    monkeypatch.setattr(wiz, "MINERU_ENV_EXAMPLE_PATH", tmp_path / "missing")

    wiz.configure_mineru_local_env("core,mlx")

    text = mineru_env.read_text(encoding="utf-8")
    assert "MINERU_TITLE_AIDED_ENABLE=false" in text
    assert "stale-secret" not in text
    assert "stale.example" not in text
    assert "stale-model" not in text
    assert "MINERU_TITLE_AIDED_ENABLE_THINKING" not in text


@pytest.mark.parametrize(
    "service_model,expected",
    [
        (
            "launchd",
            [
                ["make", "mineru-install"],
                ["make", "mineru-title-aided"],
                ["make", "mineru-service-install"],
            ],
        ),
        (
            "systemd-user",
            [
                ["make", "mineru-install"],
                ["make", "mineru-title-aided"],
                ["make", "mineru-service-install"],
            ],
        ),
        ("foreground", [["make", "mineru-install"], ["make", "mineru-title-aided"]]),
    ],
)
def test_build_mineru_local_commands(wiz, service_model, expected):
    assert wiz.build_mineru_local_commands(service_model) == expected


# --- Plan 2 Task 2: parser step --------------------------------------------
def test_run_parser_step_mineru_official(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("corpus:\n  sidecars:\n    mineru:\n      api_mode: local\n", encoding="utf-8")
    env = tmp_path / ".env"
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(wiz, "ENV_PATH", env)
    ran: list = []
    info = wiz.PlatformInfo(os="linux", arch="x86_64", is_wsl=False)
    prompter = _ScriptedPrompter(["MinerU official cloud API", "", "tok-xyz"])
    wiz.run_parser_step(prompter, info, has_gpu=False, runner=lambda cmd: ran.append(cmd))
    assert "api_mode: official" in cfg.read_text(encoding="utf-8")
    assert ran == []


def test_run_parser_step_mineru_local_runs_commands(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "corpus:\n  sidecars:\n    mineru:\n      api_mode: official\n", encoding="utf-8"
    )
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(wiz, "MINERU_ENV_PATH", tmp_path / ".env.mineru")
    monkeypatch.setattr(wiz, "MINERU_ENV_EXAMPLE_PATH", tmp_path / "missing")
    monkeypatch.setattr(wiz, "systemd_user_available", lambda: False)
    ran: list = []
    info = wiz.PlatformInfo(os="macos", arch="arm64", is_wsl=False)
    prompter = _ScriptedPrompter(["MinerU local", False])
    wiz.run_parser_step(prompter, info, has_gpu=False, runner=lambda cmd: ran.append(cmd))
    assert ["make", "mineru-install"] in ran
    assert ["make", "mineru-title-aided"] in ran
    assert ["make", "mineru-service-install"] in ran
    assert "MINERU_INSTALL_EXTRAS=core,mlx" in (tmp_path / ".env.mineru").read_text(
        encoding="utf-8"
    )


def test_run_parser_step_configures_bundled_docling(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        'corpus:\n  parser:\n    rules: "*:mineru-iteP"\n  sidecars:\n    mineru:\n      api_mode: local\n',
        encoding="utf-8",
    )
    env = tmp_path / ".env"
    env.write_text("DLIGHTRAG_CORPUS__SIDECARS__MINERU__API_TOKEN=stale\n", encoding="utf-8")
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(wiz, "ENV_PATH", env)
    info = wiz.PlatformInfo(os="linux", arch="x86_64", is_wsl=False)

    mode = wiz.run_parser_step(
        _ScriptedPrompter(["Docling bundled (Compose)"]),
        info,
        has_gpu=False,
    )

    assert mode == "docling"
    assert wiz._yaml().load(cfg)["corpus"]["sidecars"] == {
        "docling": {
            "endpoint": "http://docling:5001",
            "code_formula_preset": None,
        }
    }
    env_text = env.read_text(encoding="utf-8")
    assert "MINERU__API_TOKEN" not in env_text
    assert "COMPOSE_PROFILES=docling" in env_text


def test_run_parser_step_configures_external_docling(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        'corpus:\n  parser:\n    rules: "*:mineru-iteP"\n  sidecars:\n    mineru:\n      api_mode: local\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", tmp_path / ".env")
    env = tmp_path / ".env"
    env.write_text("COMPOSE_PROFILES=docling\n", encoding="utf-8")
    monkeypatch.setattr(wiz, "ENV_PATH", env)
    info = wiz.PlatformInfo(os="linux", arch="x86_64", is_wsl=False)

    prompter = _ScriptedPrompter(
        [
            wiz.DOCLING_EXTERNAL_CHOICE,
            "https://docling.example.com",
            wiz.DOCLING_SERVICE_DEFAULT_DEVICE_CHOICE,
        ]
    )
    mode = wiz.run_parser_step(prompter, info, has_gpu=False)

    assert prompter.select_choices[1][0] == wiz.DOCLING_MPS_DEVICE_CHOICE
    assert mode == "external"
    docling = wiz._yaml().load(cfg)["corpus"]["sidecars"]["docling"]
    assert docling["endpoint"] == "https://docling.example.com"
    assert docling["code_formula_preset"] is None
    assert "COMPOSE_PROFILES" not in env.read_text(encoding="utf-8")


def test_parser_change_rolls_back_all_files_when_validation_fails(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    env = tmp_path / ".env"
    mineru_env = tmp_path / ".env.mineru"
    originals = {
        cfg: "corpus:\n  sidecars:\n    mineru:\n      api_mode: local\n",
        env: "KEEP=env\n",
        mineru_env: "KEEP=mineru\n",
    }
    for path, content in originals.items():
        path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", env)
    monkeypatch.setattr(wiz, "MINERU_ENV_PATH", mineru_env)
    monkeypatch.setattr(
        wiz,
        "validate_config",
        lambda: (_ for _ in ()).throw(ValueError("invalid parser config")),
    )

    with pytest.raises(ValueError, match="invalid parser config"):
        wiz.run_parser_step(
            _ScriptedPrompter(["Docling bundled (Compose)"]),
            wiz.PlatformInfo(os="linux", arch="x86_64", is_wsl=False),
            has_gpu=False,
        )

    for path, content in originals.items():
        assert path.read_text(encoding="utf-8") == content


# --- Plan 3: creds return, title-aided reuse, docker bring-up --------------
def test_models_step_returns_llm_creds(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n"
        "  chat:\n    default:\n      provider: openai\n      model: x\n      base_url: https://x\n"
        "  embedding:\n    provider: voyage\n    model: x\n    dim: 1024\n"
        "  rerank:\n    strategy: voyage_reranker\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(wiz, "ENV_EXAMPLE_PATH", tmp_path / "missing")
    prompter = _ScriptedPrompter(
        [
            "Minimum · one LLM + one embedding",
            "DeepSeek",
            "deepseek-v4-flash",
            "",
            "sk-llm",
            "Voyage",
            "voyage-multimodal-3.5",
            "",
            "sk-embed",
            "Reuse my LLM",
        ]
    )
    result = wiz.run_models_step(prompter)
    assert result["llm"] == {
        "api_key": "sk-llm",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
    }


def test_models_step_reprompts_for_reranker_key(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n"
        "  chat:\n    default:\n      provider: openai\n      model: x\n"
        "  embedding:\n    provider: voyage\n    model: x\n    dim: 1024\n"
        "  rerank:\n    strategy: chat_llm_reranker\n",
        encoding="utf-8",
    )
    env = tmp_path / ".env"
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", env)
    monkeypatch.setattr(wiz, "ENV_EXAMPLE_PATH", tmp_path / "missing")
    monkeypatch.setattr(wiz, "catalog_model_profile", lambda _block: {"context_window_tokens": 1})
    prompter = _ScriptedPrompter(
        [
            wiz.MODEL_MODE_CUSTOM,
            "DeepSeek",
            "default-model",
            "",
            "sk-default",
            "DeepSeek",
            "extract-model",
            "",
            "sk-extract",
            "DeepSeek",
            "keyword-model",
            "",
            "sk-keyword",
            "Voyage",
            "voyage-multimodal-3.5",
            "",
            "sk-embed",
            "Voyage",
            "",
            "sk-rerank",
        ]
    )

    wiz.run_models_step(prompter)

    assert "DLIGHTRAG_MODELS__RERANK__API_KEY=sk-rerank" in env.read_text(encoding="utf-8")


def test_run_parser_step_mineru_local_title_aided(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "corpus:\n  sidecars:\n    mineru:\n      api_mode: official\n", encoding="utf-8"
    )
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(wiz, "MINERU_ENV_PATH", tmp_path / ".env.mineru")
    monkeypatch.setattr(wiz, "MINERU_ENV_EXAMPLE_PATH", tmp_path / "missing")
    monkeypatch.setattr(wiz, "systemd_user_available", lambda: True)
    ran: list = []
    info = wiz.PlatformInfo(os="linux", arch="x86_64", is_wsl=False)
    prompter = _ScriptedPrompter(["MinerU local", True])
    creds = {
        "api_key": "sk",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
    }
    wiz.run_parser_step(
        prompter, info, has_gpu=False, llm_title_aided=creds, runner=lambda cmd: ran.append(cmd)
    )
    assert ["make", "mineru-title-aided"] in ran
    assert "MINERU_TITLE_AIDED_MODEL=deepseek-v4-flash" in (tmp_path / ".env.mineru").read_text(
        encoding="utf-8"
    )


def test_prepare_compose_bind_sources_creates_configured_skills_directory(wiz, tmp_path):
    skills_dir = tmp_path / "operator-skills"
    env_path = tmp_path / ".env"
    env_path.write_text(f"COMPOSE_GLOBAL_SKILLS_DIR={skills_dir}\n", encoding="utf-8")

    resolved = wiz._prepare_compose_bind_sources(env_path=env_path, environment={})

    assert resolved == skills_dir
    assert skills_dir.is_dir()


def test_docker_up_command(wiz):
    assert wiz.docker_up_command() == ["docker", "compose", "up", "-d"]
    assert wiz.docker_up_command(profile="docling") == [
        "docker",
        "compose",
        "--profile",
        "docling",
        "up",
        "-d",
    ]


def test_wait_for_readiness_success(wiz):
    calls = {"n": 0}

    def probe(url):
        calls["n"] += 1
        return calls["n"] >= 2

    assert (
        wiz.wait_for_readiness("u", attempts=5, delay=0, probe=probe, sleep=lambda _: None) is True
    )


def test_wait_for_readiness_gives_up(wiz):
    assert (
        wiz.wait_for_readiness(
            "u", attempts=3, delay=0, probe=lambda url: False, sleep=lambda _: None
        )
        is False
    )


def test_bring_up_stack_waits_for_strict_readiness(wiz, monkeypatch):
    events: list[str] = []
    urls: list[str] = []
    monkeypatch.setattr(
        wiz,
        "_prepare_compose_bind_sources",
        lambda: events.append("prepare"),
    )
    monkeypatch.setattr(wiz, "_default_runner", lambda command: events.append("run"))
    monkeypatch.setattr(wiz, "wait_for_readiness", lambda url: urls.append(url) or True)

    assert wiz._bring_up_stack(_NullConsole()) == 0
    assert events == ["prepare", "run"]
    assert urls == ["http://localhost:8100/ready"]


def test_with_quit_appends_sentinel(wiz):
    assert wiz.with_quit(["A", "B"]) == ["A", "B", wiz.QUIT_CHOICE]


def test_check_quit_passes_through_normal_answer(wiz):
    assert wiz.check_quit("A") == "A"


def test_check_quit_raises_setup_cancelled(wiz):
    with pytest.raises(wiz.SetupCancelled):
        wiz.check_quit(wiz.QUIT_CHOICE)


@pytest.mark.parametrize("kind", ["ctrl_c", "ctrl_d", "menu_quit"])
def test_main_cancel_exits_cleanly(wiz, tmp_path, monkeypatch, kind):
    """Ctrl+C, Ctrl+D, or the in-menu Quit all exit 130 with no traceback."""
    monkeypatch.setattr(wiz, "run_preflight", lambda: [])  # pretend all tools present
    cfg = tmp_path / "config.yaml"
    cfg.write_text("llm:\n  default:\n    provider: openai\n", encoding="utf-8")
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(wiz, "ENV_EXAMPLE_PATH", tmp_path / "missing")
    exc = {"ctrl_c": KeyboardInterrupt, "ctrl_d": EOFError, "menu_quit": wiz.SetupCancelled}[kind]

    class _Cancel(wiz.Prompter):
        def select(self, message, choices):
            raise exc

        def text(self, message, default=""):
            raise exc

        def password(self, message):
            raise exc

        def confirm(self, message, default=False):
            raise exc

    assert wiz.main(prompter=_Cancel()) == 130


def test_main_without_tty_exits_without_opening_questionary(wiz, monkeypatch):
    monkeypatch.setattr(wiz, "run_preflight", lambda: [])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        wiz,
        "_questionary_prompter",
        lambda: pytest.fail("questionary must not open without an interactive terminal"),
    )

    assert wiz.main() == 2


# --- §11 re-run menu: detection / summary / dispatch ----------------------
class _NullConsole:
    def print(self, *args, **kwargs):
        pass

    def rule(self, *args, **kwargs):
        pass


def _info(wiz):
    return wiz.PlatformInfo(os="linux", arch="x86_64", is_wsl=False)


def test_apply_and_validate_restores_config_and_env_on_failure(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    env = tmp_path / ".env"
    cfg.write_text("old-config\n", encoding="utf-8")
    env.write_text("OLD_ENV=1\n", encoding="utf-8")
    config_backup = wiz.backup_file(cfg)
    env_backup = wiz.backup_file(env)
    cfg.write_text("new-config\n", encoding="utf-8")
    env.write_text("NEW_ENV=1\n", encoding="utf-8")
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", env)
    monkeypatch.setattr(wiz, "validate_config", lambda: (_ for _ in ()).throw(ValueError("bad")))

    assert (
        wiz._apply_and_validate(
            _NullConsole(),
            {"config_backup": config_backup, "env_backup": env_backup, "env_existed": True},
        )
        is False
    )

    assert cfg.read_text(encoding="utf-8") == "old-config\n"
    assert env.read_text(encoding="utf-8") == "OLD_ENV=1\n"


def test_apply_and_validate_removes_new_env_on_failure(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    env = tmp_path / ".env"
    cfg.write_text("old-config\n", encoding="utf-8")
    config_backup = wiz.backup_file(cfg)
    cfg.write_text("new-config\n", encoding="utf-8")
    env.write_text("NEW_ENV=1\n", encoding="utf-8")
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", env)
    monkeypatch.setattr(wiz, "validate_config", lambda: (_ for _ in ()).throw(ValueError("bad")))

    assert (
        wiz._apply_and_validate(
            _NullConsole(),
            {"config_backup": config_backup, "env_backup": None, "env_existed": False},
        )
        is False
    )

    assert cfg.read_text(encoding="utf-8") == "old-config\n"
    assert not env.exists()


def test_is_configured_true_when_keys_present(wiz, tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY=sk-a\nDLIGHTRAG_MODELS__EMBEDDING__API_KEY=sk-b\n",
        encoding="utf-8",
    )
    assert wiz.is_configured(env) is True


def test_is_configured_false_when_missing_key(wiz, tmp_path):
    env = tmp_path / ".env"
    env.write_text("DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY=sk-a\n", encoding="utf-8")
    assert wiz.is_configured(env) is False


def test_is_configured_false_when_no_env(wiz, tmp_path):
    assert wiz.is_configured(tmp_path / "missing.env") is False


def test_is_configured_accepts_explicit_keyless_local_models(wiz, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n"
        "  chat:\n"
        "    default:\n"
        "      provider: openai\n"
        "      model: local-chat\n"
        "      base_url: http://127.0.0.1:8888/v1\n"
        "      api_key: null\n"
        "  embedding:\n"
        "    provider: openai_compatible\n"
        "    model: local-embed\n"
        "    base_url: http://127.0.0.1:1234/v1\n"
        "    dim: 768\n"
        "    api_key: null\n",
        encoding="utf-8",
    )
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")

    assert wiz.is_configured(env, config_path=cfg) is True


def test_read_config_summary_uses_effective_defaults_and_role_fallback(wiz, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n"
        "  chat:\n"
        "    default:\n"
        "      provider: openai\n"
        "      model: local-chat\n"
        "      api_key: null\n"
        "    roles:\n"
        "      query:\n"
        "        provider: openai\n"
        "        model: incomplete-query\n"
        "  embedding:\n"
        "    provider: openai_compatible\n"
        "    model: local-embed\n"
        "    dim: 768\n"
        "    api_key: null\n",
        encoding="utf-8",
    )
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")

    summary = wiz.read_config_summary(cfg, env)

    assert summary["llm_roles"] == {}
    assert summary["rerank"]["enabled"] is True


def test_read_config_summary_masks_secrets_and_extracts(wiz, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n"
        "  chat:\n"
        "    default:\n      provider: openai\n      model: gpt-x\n      base_url: https://api.x\n"
        "    roles:\n      extract:\n        provider: openai\n        model: cheap\n"
        "        base_url: https://api.deepseek.com\n"
        "  embedding:\n    provider: voyage\n    model: voyage-x\n    dim: 1024\n"
        "    base_url: https://api.voyageai.com/v1\n"
        "  rerank:\n    enabled: true\n    strategy: voyage_reranker\n"
        "    model: rerank-2.5-lite\n"
        "answer:\n  generation:\n    max_attachments: 6\n"
        "    max_attachment_bytes: 104857600\n"
        "    max_total_attachment_bytes: 134217728\n"
        "    max_images: 12\n"
        "corpus:\n  sidecars:\n    mineru:\n      api_mode: local\n"
        "    docling:\n      endpoint: http://docling:5001\n"
        "deployment:\n  workspace: default\n",
        encoding="utf-8",
    )
    env = tmp_path / ".env"
    env.write_text(
        "DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY=sk-a\n"
        "DLIGHTRAG_MODELS__CHAT__ROLES__EXTRACT__API_KEY=sk-role\n"
        "DLIGHTRAG_MODELS__EMBEDDING__API_KEY=sk-b\n",
        encoding="utf-8",
    )
    s = wiz.read_config_summary(cfg, env)
    assert s["llm_default"] == {"provider": "openai", "model": "gpt-x", "base_url": "https://api.x"}
    assert s["llm_roles"]["extract"] == {
        "provider": "openai",
        "model": "cheap",
        "base_url": "https://api.deepseek.com",
    }
    assert s["embedding"]["dim"] == 1024
    assert s["embedding"]["base_url"] == "https://api.voyageai.com/v1"
    assert s["rerank"] == {
        "strategy": "voyage_reranker",
        "enabled": True,
        "model": "rerank-2.5-lite",
        "base_url": None,
    }
    assert s["parser"] == {"name": "MinerU", "detail": "local"}
    assert s["workspace"] == "default"
    assert s["keys_set"] == {
        "LLM": True,
        "Embedding": True,
        "Rerank": False,
        "Exa": False,
        "Tavily": False,
    }
    assert s["web_research"] == {
        "direct_url_read": True,
        "search": [],
        "extract": [],
    }
    assert "sk-a" not in repr(s) and "sk-b" not in repr(s)
    assert s["model_capacities"]["default"]["source"] == "fallback"
    assert s["model_capacities"]["default"]["context_window_tokens"] == 1_048_576
    assert s["model_capacities"]["extract"]["source"] == "fallback"
    # Attachment settings are surfaced independently from model capacity.
    assert s["answer"] == {
        "max_attachments": 6,
        "max_attachment_bytes": 104857600,
        "max_total_attachment_bytes": 134217728,
        "max_images": 12,
    }
    # No dedicated VLM role: the default LLM performs answer visual inspection.
    assert s["visual_inspection"] == {
        "role": "default",
        "provider": "openai",
        "model": "gpt-x",
    }


def test_read_config_summary_reports_vlm_role_visual_inspection(wiz, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n"
        "  chat:\n"
        "    default:\n      provider: openai\n      model: gpt-x\n"
        "    roles:\n      vlm:\n        provider: gemini\n        model: gemini-vision\n"
        "  embedding:\n    provider: voyage\n    model: voyage-x\n    dim: 1024\n"
        "  rerank:\n    enabled: false\n    strategy: chat_llm_reranker\n"
        "corpus:\n  sidecars:\n    mineru:\n      api_mode: local\n"
        "deployment:\n  workspace: default\n",
        encoding="utf-8",
    )
    env = tmp_path / ".env"
    env.write_text(
        "DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY=sk-a\n"
        "DLIGHTRAG_MODELS__CHAT__ROLES__VLM__API_KEY=sk-vlm\n",
        encoding="utf-8",
    )
    s = wiz.read_config_summary(cfg, env)
    # A complete explicit vlm role owns answer visual inspection.
    assert s["visual_inspection"] == {
        "role": "vlm",
        "provider": "gemini",
        "model": "gemini-vision",
    }


def test_read_config_summary_uses_answer_defaults_when_absent(wiz, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n"
        "  chat:\n    default:\n      provider: openai\n      model: gpt-x\n"
        "  embedding:\n    provider: voyage\n    model: voyage-x\n    dim: 1024\n"
        "  rerank:\n    strategy: chat_llm_reranker\n"
        "corpus:\n  sidecars:\n    mineru:\n      api_mode: local\n"
        "deployment:\n  workspace: default\n",
        encoding="utf-8",
    )
    env = tmp_path / ".env"
    env.write_text("DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY=sk-a\n", encoding="utf-8")
    s = wiz.read_config_summary(cfg, env)
    # Absent answer block falls back to attachment/image product defaults only.
    assert "context_window_tokens" not in s["answer"]
    assert s["answer"]["max_attachments"] == 6
    assert s["answer"]["max_attachment_bytes"] == 100 * 1024 * 1024
    assert s["answer"]["max_total_attachment_bytes"] == 128 * 1024 * 1024


def test_render_summary_shows_context_and_attachment_settings(wiz, tmp_path):
    from rich.console import Console

    # Distinctive values, so the assertions cannot be satisfied by the shipped config.yaml.
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n"
        "  chat:\n    default:\n      provider: openai\n      model: gpt-x\n"
        "      base_url: https://api.x/v1\n"
        "  embedding:\n    provider: voyage\n    model: voyage-x\n    dim: 1024\n"
        "  rerank:\n    strategy: chat_llm_reranker\n"
        "corpus:\n  sidecars:\n    mineru:\n      api_mode: local\n"
        "deployment:\n  workspace: default\n"
        "answer:\n  generation:\n    max_attachments: 7\n"
        "    max_attachment_bytes: 3145728\n",
        encoding="utf-8",
    )
    env = tmp_path / ".env"
    env.write_text("DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY=sk-a\n", encoding="utf-8")

    console = Console(record=True, width=100)
    wiz.render_summary(console, wiz.read_config_summary(cfg, env))
    text = console.export_text()
    assert "fallback · C 1,048,576 · I = C · O 262,144" in text
    assert "7 max" in text
    assert "3 MiB each" in text
    assert "visual inspection" in text.lower()


def test_home_start_brings_up_stack(wiz, monkeypatch):
    ups: list = []
    monkeypatch.setattr(wiz, "_default_runner", lambda cmd: ups.append(cmd))
    monkeypatch.setattr(wiz, "wait_for_readiness", lambda url, **k: True)
    prompter = _ScriptedPrompter([wiz.MENU_START])
    rc = wiz.run_home(_NullConsole(), prompter, _info(wiz))
    assert rc == 0
    assert ups == [["docker", "compose", "up", "-d"]]


def test_home_choices_prioritize_show_before_start(wiz):
    assert wiz.HOME_CHOICES == [wiz.MENU_SHOW, wiz.MENU_START, wiz.MENU_CHANGE, wiz.MENU_RESET]


def test_home_show_then_start(wiz, monkeypatch):
    shown: list = []
    monkeypatch.setattr(wiz, "read_config_summary", lambda c, e: {"ok": True})
    monkeypatch.setattr(wiz, "render_summary", lambda console, s: shown.append(s))
    monkeypatch.setattr(wiz, "_bring_up_stack", lambda console: 0)
    prompter = _ScriptedPrompter([wiz.MENU_SHOW, wiz.MENU_START])
    rc = wiz.run_home(_NullConsole(), prompter, _info(wiz))
    assert rc == 0
    assert shown == [{"ok": True}]


def test_home_change_then_start(wiz, monkeypatch):
    changed: list = []
    monkeypatch.setattr(wiz, "run_change_settings", lambda console, p, info: changed.append(True))
    monkeypatch.setattr(wiz, "_bring_up_stack", lambda console: 0)
    prompter = _ScriptedPrompter([wiz.MENU_CHANGE, wiz.MENU_START])
    rc = wiz.run_home(_NullConsole(), prompter, _info(wiz))
    assert rc == 0
    assert changed == [True]


def test_home_reset_without_wipe(wiz, monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(wiz, "run_first_time_setup", lambda console, p, info, **k: 0)
    monkeypatch.setattr(wiz, "_wipe_data", lambda console, **k: events.append("wipe"))
    monkeypatch.setattr(wiz, "_bring_up_stack", lambda console: events.append("start") or 0)
    prompter = _ScriptedPrompter([wiz.MENU_RESET, True, False])
    rc = wiz.run_home(_NullConsole(), prompter, _info(wiz))
    assert rc == 0
    assert events == ["start"]


def test_home_reset_with_wipe(wiz, monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(wiz, "run_first_time_setup", lambda console, p, info, **k: 0)
    monkeypatch.setattr(wiz, "_wipe_data", lambda console, **k: events.append("wipe"))
    monkeypatch.setattr(wiz, "_bring_up_stack", lambda console: events.append("start") or 0)
    prompter = _ScriptedPrompter([wiz.MENU_RESET, True, True])
    rc = wiz.run_home(_NullConsole(), prompter, _info(wiz))
    assert rc == 0
    assert events == ["wipe", "start"]


def test_wipe_data_invokes_workspace_reset_and_clears_compose_volume(wiz):
    calls: list[list[str]] = []

    assert wiz._wipe_data(_NullConsole(), runner=lambda args: calls.append(args)) is True

    assert calls == [
        ["uv", "run", "scripts/reset_workspace.py", "--all", "-y"],
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
        ],
    ]


def test_home_reset_declined_returns_to_menu(wiz, monkeypatch):
    monkeypatch.setattr(wiz, "run_first_time_setup", lambda console, p, info, **k: None)
    monkeypatch.setattr(wiz, "_bring_up_stack", lambda console: 0)
    wiped: list = []
    monkeypatch.setattr(wiz, "_wipe_data", lambda console, **k: wiped.append(True))
    prompter = _ScriptedPrompter([wiz.MENU_RESET, False, wiz.MENU_START])
    rc = wiz.run_home(_NullConsole(), prompter, _info(wiz))
    assert rc == 0
    assert wiped == []


def test_change_models_only(wiz, monkeypatch):
    monkeypatch.setattr(
        wiz, "run_models_step", lambda p, **k: {"llm": {"base_url": "u"}, "config_backup": None}
    )
    monkeypatch.setattr(wiz, "validate_config", lambda: None)
    parser: list = []
    monkeypatch.setattr(wiz, "run_parser_step", lambda *a, **k: parser.append(True) or "mineru")
    wiz.run_change_settings(_NullConsole(), _ScriptedPrompter([wiz.SEC_MODELS]), _info(wiz))
    assert parser == []


def test_legacy_web_settings_migrate_to_exa(wiz, tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "DLIGHTRAG_ANSWER__WEB_SEARCH__API_KEY=legacy-key\nOTHER=value\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("answer:\n  web_search:\n    api_key: null\n", encoding="utf-8")

    assert wiz.legacy_web_settings_present(config_path=config_path, env_path=env_path)
    assert wiz.migrate_legacy_web_settings(config_path=config_path, env_path=env_path)
    assert not wiz.legacy_web_settings_present(config_path=config_path, env_path=env_path)
    env_text = env_path.read_text(encoding="utf-8")
    assert "DLIGHTRAG_ANSWER__WEB_SOURCES__EXA__API_KEY=legacy-key" in env_text
    assert "WEB_SEARCH" not in env_text
    data = wiz._yaml().load(config_path)
    assert data["answer"]["web_sources"]["exa"]["api_key"] is None
    assert "web_search" not in data["answer"]


def test_legacy_inline_web_secret_moves_to_env_with_backups(wiz, tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("OTHER=value\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "answer:\n  web_search:\n    api_key: inline-secret\n",
        encoding="utf-8",
    )

    assert wiz.migrate_legacy_web_settings(config_path=config_path, env_path=env_path)

    assert "DLIGHTRAG_ANSWER__WEB_SOURCES__EXA__API_KEY=inline-secret" in (
        env_path.read_text(encoding="utf-8")
    )
    data = wiz._yaml().load(config_path)
    assert data["answer"]["web_sources"]["exa"]["api_key"] is None
    assert list(tmp_path.glob("config.yaml.bak-*"))
    assert list(tmp_path.glob(".env.bak-*"))


def test_commented_legacy_web_key_does_not_trigger_migration(wiz, tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# DLIGHTRAG_ANSWER__WEB_SEARCH__API_KEY=example-only\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("answer: {}\n", encoding="utf-8")

    assert not wiz.legacy_web_settings_present(config_path=config_path, env_path=env_path)
    assert not wiz.migrate_legacy_web_settings(config_path=config_path, env_path=env_path)
    assert "WEB_SEARCH" in env_path.read_text(encoding="utf-8")


def test_web_setup_writes_independent_provider_orders(wiz, tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("answer: {}\n", encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(wiz, "CONFIG_PATH", config_path)
    monkeypatch.setattr(wiz, "ENV_PATH", env_path)
    monkeypatch.setattr(wiz, "ENV_EXAMPLE_PATH", tmp_path / "missing")
    monkeypatch.setattr(wiz, "validate_config", lambda: None)
    prompter = _ScriptedPrompter(
        [
            wiz.WEB_BOTH,
            wiz.WEB_INDEPENDENT,
            wiz.WEB_TAVILY,
            wiz.WEB_EXA,
            "exa-key",
            "tavily-key",
        ]
    )

    assert wiz.run_web_research_step(prompter, optional=False)
    data = wiz._yaml().load(config_path)["answer"]["web_sources"]
    assert data["search_providers"] == ["tavily", "exa"]
    assert data["extract_providers"] == ["exa", "tavily"]
    assert data["exa"]["api_key"] is None
    assert data["tavily"]["api_key"] is None
    env_text = env_path.read_text(encoding="utf-8")
    assert "DLIGHTRAG_ANSWER__WEB_SOURCES__EXA__API_KEY=exa-key" in env_text
    assert "DLIGHTRAG_ANSWER__WEB_SOURCES__TAVILY__API_KEY=tavily-key" in env_text
    assert list(tmp_path.glob("config.yaml.bak-*"))
    assert list(tmp_path.glob(".env.bak-*"))


def test_web_setup_rolls_back_config_and_env_when_validation_fails(
    wiz, tmp_path, monkeypatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_before = b"answer: {}\n"
    config_path.write_bytes(config_before)
    env_path = tmp_path / ".env"
    env_before = b"OTHER=value\n"
    env_path.write_bytes(env_before)
    monkeypatch.setattr(wiz, "CONFIG_PATH", config_path)
    monkeypatch.setattr(wiz, "ENV_PATH", env_path)
    monkeypatch.setattr(wiz, "ENV_EXAMPLE_PATH", tmp_path / "missing")
    monkeypatch.setattr(
        wiz,
        "validate_config",
        lambda: (_ for _ in ()).throw(ValueError("invalid")),
    )

    with pytest.raises(ValueError, match="invalid"):
        wiz.run_web_research_step(
            _ScriptedPrompter([wiz.WEB_EXA, "exa-key"]),
            optional=False,
        )

    assert config_path.read_bytes() == config_before
    assert env_path.read_bytes() == env_before


def test_change_parser_only(wiz, monkeypatch):
    models: list = []
    monkeypatch.setattr(wiz, "run_models_step", lambda p, **k: models.append(True))
    parser: list = []
    monkeypatch.setattr(wiz, "run_parser_step", lambda *a, **k: parser.append(True) or "mineru")
    wiz.run_change_settings(_NullConsole(), _ScriptedPrompter([wiz.SEC_PARSER]), _info(wiz))
    assert models == []
    assert parser == [True]


def test_change_everything_runs_both(wiz, monkeypatch):
    monkeypatch.setattr(
        wiz, "run_models_step", lambda p, **k: {"llm": {"base_url": "u"}, "config_backup": None}
    )
    monkeypatch.setattr(wiz, "validate_config", lambda: None)
    parser: list = []
    monkeypatch.setattr(wiz, "run_parser_step", lambda *a, **k: parser.append(True) or "mineru")
    web: list = []
    monkeypatch.setattr(
        wiz,
        "run_web_research_step",
        lambda *a, **k: web.append(True) or True,
    )
    wiz.run_change_settings(_NullConsole(), _ScriptedPrompter([wiz.SEC_ALL]), _info(wiz))
    assert parser == [True]
    assert web == [True]


def test_change_back_does_nothing(wiz, monkeypatch):
    touched: list = []
    monkeypatch.setattr(wiz, "run_models_step", lambda p, **k: touched.append("models"))
    monkeypatch.setattr(wiz, "run_parser_step", lambda *a, **k: touched.append("parser"))
    wiz.run_change_settings(_NullConsole(), _ScriptedPrompter([wiz.SEC_BACK]), _info(wiz))
    assert touched == []


def test_change_models_declined_makes_no_change(wiz, monkeypatch):
    monkeypatch.setattr(wiz, "run_models_step", lambda p, **k: None)
    parser: list = []
    monkeypatch.setattr(wiz, "run_parser_step", lambda *a, **k: parser.append(True) or "mineru")
    wiz.run_change_settings(_NullConsole(), _ScriptedPrompter([wiz.SEC_MODELS]), _info(wiz))
    assert parser == []


_MODELS_ANSWERS = [
    "Minimum · one LLM + one embedding",
    "DeepSeek",
    "deepseek-v4-flash",
    "",
    "sk-llm",
    "Voyage",
    "voyage-multimodal-3.5",
    "",
    "sk-embed",
]


def _models_cfg(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n"
        "  chat:\n    default:\n      provider: openai\n      model: old\n"
        "      base_url: https://old\n"
        "  embedding:\n    provider: voyage\n    model: old\n    dim: 1024\n"
        "  rerank:\n    strategy: voyage_reranker\n",
        encoding="utf-8",
    )
    return cfg


def test_models_step_confirm_declined_leaves_config(wiz, tmp_path, monkeypatch):
    cfg = _models_cfg(tmp_path)
    original = cfg.read_text(encoding="utf-8")
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(wiz, "ENV_EXAMPLE_PATH", tmp_path / "missing")
    prompter = _ScriptedPrompter([*_MODELS_ANSWERS, False])  # decline the overwrite
    assert wiz.run_models_step(prompter, require_confirm=True) is None
    assert cfg.read_text(encoding="utf-8") == original
    assert not list(cfg.parent.glob(f"{cfg.name}.bak-*"))


def test_models_step_confirm_accepted_writes(wiz, tmp_path, monkeypatch):
    cfg = _models_cfg(tmp_path)
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(wiz, "ENV_EXAMPLE_PATH", tmp_path / "missing")
    prompter = _ScriptedPrompter([*_MODELS_ANSWERS, True])  # accept the overwrite
    assert wiz.run_models_step(prompter, require_confirm=True) is not None
    assert "deepseek-v4-flash" in cfg.read_text(encoding="utf-8")
    assert list(cfg.parent.glob(f"{cfg.name}.bak-*"))


def test_parser_step_confirm_declined_skips_write(wiz, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "corpus:\n  sidecars:\n    mineru:\n      api_mode: official\n", encoding="utf-8"
    )
    monkeypatch.setattr(wiz, "CONFIG_PATH", cfg)
    monkeypatch.setattr(wiz, "MINERU_ENV_PATH", tmp_path / ".env.mineru")
    monkeypatch.setattr(wiz, "MINERU_ENV_EXAMPLE_PATH", tmp_path / "missing")
    ran: list = []
    info = wiz.PlatformInfo(os="macos", arch="arm64", is_wsl=False)
    prompter = _ScriptedPrompter(["MinerU local", False])
    applied = wiz.run_parser_step(
        prompter, info, has_gpu=False, runner=lambda c: ran.append(c), require_confirm=True
    )
    assert applied is None
    assert ran == []
    assert "api_mode: official" in cfg.read_text(encoding="utf-8")
