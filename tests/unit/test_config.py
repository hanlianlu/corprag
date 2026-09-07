# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical nine-section configuration contracts."""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from dlightrag.application.config import (
    AccessSectionSettings,
    DeploymentSettings,
    DlightragConfig,
    PostgresSettings,
    StorageSettings,
    load_config,
)
from dlightrag.engine.ai.settings import EmbeddingSettings, ModelSettings, ModelsSettings
from dlightrag.engine.rag.workspace.settings import (
    CorpusSettings,
    MinerUSidecarSettings,
    ParserSidecarsSettings,
)


@pytest.fixture(autouse=True)
def _clean_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("DLIGHTRAG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(DlightragConfig.model_config, "env_file", None)


def test_root_has_exactly_nine_sections() -> None:
    assert tuple(DlightragConfig.model_fields) == (
        "deployment",
        "storage",
        "models",
        "corpus",
        "answer",
        "runtime",
        "access",
        "interfaces",
        "observability",
    )


def test_defaults_preserve_runtime_contract(tmp_path: Path) -> None:
    config = DlightragConfig(
        deployment=DeploymentSettings(working_dir=str(tmp_path)),
        models=ModelsSettings(embedding=EmbeddingSettings(startup_probe=False)),
    )
    assert config.deployment.service_role == "writer"
    assert config.deployment.workspace == "default"
    assert config.storage.postgres.port == 5432
    assert config.models.chat.default.model == "google/gemini-3.7-flash"
    assert config.models.chat.default.structured_output == "auto"
    assert config.models.embedding.dim == 1024
    assert config.models.embedding.batch_size == 64
    assert config.corpus.retrieval.top_k == 40
    assert config.corpus.retrieval.chunk_top_k == 20
    assert config.corpus.retrieval.federation_min_chunks_per_workspace == 7
    assert config.corpus.promotion.doc_threshold is None
    assert config.corpus.promotion.chunk_threshold is None
    assert config.runtime.run_retention_days == 365
    assert config.input_dir_path == tmp_path / "inputs"


def test_nested_environment_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DLIGHTRAG_STORAGE__POSTGRES__HOST", "db.internal")
    monkeypatch.setenv("DLIGHTRAG_MODELS__EMBEDDING__DIM", "768")
    monkeypatch.setenv("DLIGHTRAG_CORPUS__RETRIEVAL__TOP_K", "42")
    monkeypatch.setenv("DLIGHTRAG_CORPUS__PROMOTION__DOC_THRESHOLD", "3")
    config = DlightragConfig()
    assert config.storage.postgres.host == "db.internal"
    assert config.models.embedding.dim == 768
    assert config.corpus.retrieval.top_k == 42
    assert config.corpus.promotion.doc_threshold == 3


def test_old_flat_constructor_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        DlightragConfig(postgres_host="old")  # type: ignore[call-arg]


def test_old_flat_environment_field_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DLIGHTRAG_POSTGRES_HOST", "old")
    with pytest.raises(ValueError, match="Unknown DlightRAG environment variables"):
        DlightragConfig()


def test_auxiliary_environment_is_not_misread_as_server_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DLIGHTRAG_API_URL", "https://client.example")
    monkeypatch.setenv("DLIGHTRAG_API_TOKEN", "client-token")
    monkeypatch.setenv("DLIGHTRAG_POSTGRES_SHARED_BUFFERS", "8GB")
    monkeypatch.setenv("COMPOSE_GLOBAL_SKILLS_DIR", "/srv/operator-skills")

    assert DlightragConfig().interfaces.api.host == "127.0.0.1"


def test_settings_are_deeply_immutable() -> None:
    config = DlightragConfig(
        models=ModelsSettings(
            chat={  # pyright: ignore[reportArgumentType]
                "default": {"model": "x", "model_kwargs": {"nested": {"enabled": True}}}
            }
        )
    )
    with pytest.raises(ValidationError, match="frozen"):
        config.models.embedding.dim = 2
    with pytest.raises(TypeError):
        config.models.chat.default.model_kwargs["x"] = 1  # type: ignore[index]
    nested = config.models.chat.default.model_kwargs["nested"]
    with pytest.raises(TypeError):
        nested["enabled"] = False  # type: ignore[index]


def test_model_provider_validation_and_mutable_provider_copy() -> None:
    settings = ModelSettings(
        provider="OpenAI",  # type: ignore[arg-type]
        model="x",
        model_kwargs={"reasoning": {"enabled": True}},
    )
    assert settings.provider == "openai"
    copied = settings.model_kwargs_copy()
    copied["reasoning"]["enabled"] = False
    assert settings.model_kwargs["reasoning"]["enabled"] is True


def test_auth_validation_uses_nested_access() -> None:
    with pytest.raises(ValidationError, match="requires api_token"):
        DlightragConfig(access=AccessSectionSettings(auth_mode="simple"))
    config = DlightragConfig(access=AccessSectionSettings(auth_mode="simple", api_token="secret"))
    assert config.access.auth_mode == "simple"


def test_models_accept_complete_startup_catalogue_entries() -> None:
    row = {
        "provider": " OpenAI ",
        "model": "vendor/new-model",
        "base_url": "https://api.vendor.example/v1",
        "profile": {
            "context_window_tokens": 262_144,
            "max_input_tokens": None,
            "max_output_tokens": 32_768,
            "supports_images": True,
            "reasoning": {
                "format": "openai",
                "levels": {
                    "off": "none",
                    "minimal": None,
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                    "xhigh": None,
                    "max": None,
                },
            },
        },
    }

    settings = ModelsSettings.model_validate({"catalogue": [row]})

    assert settings.catalogue_data() == [{**row, "provider": "openai"}]


def test_postgres_projection_and_reader_settings() -> None:
    config = DlightragConfig(
        deployment=DeploymentSettings(service_role="reader"),
        storage=StorageSettings(
            postgres=PostgresSettings(host="db", port=5433, user="u", password="p", database="d")
        ),
    )
    assert config.pg_connection_kwargs() == {
        "host": "db",
        "port": 5433,
        "user": "u",
        "password": "p",
        "database": "d",
    }
    assert config.lightrag_pool_server_settings()["default_transaction_read_only"] == "on"


def test_dump_redacts_nested_secrets() -> None:
    config = DlightragConfig(
        storage=StorageSettings(postgres=PostgresSettings(password="long-secret-value")),
        access=AccessSectionSettings(auth_mode="simple", api_token="bearer-secret-value"),
        corpus=CorpusSettings(
            sidecars=ParserSidecarsSettings(
                mineru=MinerUSidecarSettings(api_mode="official", api_token="mineru-secret-value")
            )
        ),
    )
    rendered = f"{config!r}\n{config.model_dump_json()}"
    assert "long-secret-value" not in rendered
    assert "bearer-secret-value" not in rendered
    assert "mineru-secret-value" not in rendered


def test_load_config_redacts_invalid_secret_input() -> None:
    with pytest.raises(ValueError) as caught:
        load_config(access={"unknown_secret": "do-not-echo"})
    assert "do-not-echo" not in str(caught.value)
