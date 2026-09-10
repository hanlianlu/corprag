# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""YAML precedence for the canonical nested configuration."""

from pathlib import Path

import pytest

from dlightrag.application.config import DlightragConfig, LaneRuntimeConfig, _find_yaml_config


def test_nested_yaml_loads_and_environment_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.yaml").write_text(
        """
deployment:
  workspace: yaml-space
storage:
  postgres:
    host: yaml-db
models:
  embedding:
    dim: 768
corpus:
  retrieval:
    top_k: 41
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)
    monkeypatch.setenv("DLIGHTRAG_CORPUS__RETRIEVAL__TOP_K", "43")

    config = DlightragConfig()

    assert config.deployment.workspace == "yaml-space"
    assert config.storage.postgres.host == "yaml-db"
    assert config.models.embedding.dim == 768
    assert config.corpus.retrieval.top_k == 43


def test_incomplete_yaml_role_falls_back_but_explicit_null_is_keyless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.yaml").write_text(
        """
models:
  chat:
    default:
      model: default-model
    roles:
      query:
        model: incomplete-query
      keyword:
        model: local-keyword
        api_key: null
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)

    roles = DlightragConfig().models.chat

    assert roles.resolve("query").model == "default-model"
    assert roles.resolve("keyword").model == "local-keyword"


def test_constructor_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.yaml").write_text(
        "deployment:\n  workspace: yaml-space\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)

    config = DlightragConfig(deployment={"workspace": "constructor-space"})  # type: ignore[arg-type]

    assert config.deployment.workspace == "constructor-space"


def test_nested_runtime_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)
    monkeypatch.setenv("DLIGHTRAG_RUNTIME__QUERY__WORKER_CONCURRENCY", "7")

    runtime = DlightragConfig().runtime

    assert runtime.query.worker_concurrency == 7
    assert runtime.query.max_nonterminal_runs == 30_000


def test_nested_runtime_yaml_retains_the_other_lane_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.yaml").write_text(
        "runtime:\n  corpus_mutation:\n    max_nonterminal_runs: 57\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)

    runtime = DlightragConfig().runtime

    assert runtime.corpus_mutation.worker_concurrency == 2
    assert runtime.corpus_mutation.max_nonterminal_runs == 57


def test_runtime_lanes_share_one_required_config_type_with_lane_defaults() -> None:
    runtime = DlightragConfig(_env_file=None).runtime

    assert type(runtime.query) is LaneRuntimeConfig
    assert type(runtime.corpus_mutation) is LaneRuntimeConfig
    assert set(LaneRuntimeConfig.model_fields) == {
        "worker_concurrency",
        "max_nonterminal_runs",
    }
    assert all(field.is_required() for field in LaneRuntimeConfig.model_fields.values())
    assert runtime.query == LaneRuntimeConfig(
        worker_concurrency=16,
        max_nonterminal_runs=30_000,
    )
    assert runtime.corpus_mutation == LaneRuntimeConfig(
        worker_concurrency=2,
        max_nonterminal_runs=1_000,
    )


def test_flat_runtime_environment_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)
    monkeypatch.setenv("DLIGHTRAG_ANSWER_WORKER_CONCURRENCY", "7")

    with pytest.raises(ValueError, match="Unknown DlightRAG environment variables"):
        DlightragConfig()


def test_yaml_discovery_and_no_yaml_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert _find_yaml_config() is None
    config = DlightragConfig(_env_file=None)
    assert config.deployment.workspace == "default"

    path = tmp_path / "config.yaml"
    path.write_text("deployment:\n  workspace: found\n", encoding="utf-8")
    assert _find_yaml_config() == Path("config.yaml")
    assert DlightragConfig(_env_file=None).deployment.workspace == "found"


def test_shipped_config_and_env_example_use_canonical_sections() -> None:
    root = Path(__file__).resolve().parents[2]
    config_text = (root / "config.yaml").read_text(encoding="utf-8")
    env_text = (root / ".env.example").read_text(encoding="utf-8")

    assert "models:\n" in config_text
    assert "  embedding:\n" in config_text
    assert "    input_modality: auto\n" in config_text
    assert config_text.count("model: deepseek-flash\n") == 4
    assert "model: deepseek-v4.1-flash\n" not in config_text
    assert "DLIGHTRAG_ANSWER__WEB_SOURCES__EXA__API_KEY" in env_text
    assert "DLIGHTRAG_ANSWER__WEB_SOURCES__TAVILY__API_KEY" in env_text
    assert "DLIGHTRAG_ANSWER__WEB_SEARCH__API_KEY" not in env_text
    assert "DLIGHTRAG_WEB_SEARCH__API_KEY" not in env_text


def test_old_yaml_root_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.yaml").write_text("postgres_host: old-db\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)

    with pytest.raises(Exception, match="Extra inputs"):
        DlightragConfig()
