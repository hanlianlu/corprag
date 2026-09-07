# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Helpers for the opt-in PostgreSQL 18 + LightRAG smoke tests."""

import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dlightrag.application.config import DlightragConfig
from dlightrag.engine.ai.settings import EmbeddingSettings, ModelRoleSettings, ModelSettings

RUN_E2E_ENV = "DLIGHTRAG_RUN_E2E_PG18"
REQUIRED_EXTENSIONS = ("vector", "pg_textsearch", "pg_jieba", "pg_trgm")
REQUIRED_PRELOAD_LIBRARIES = ("pg_textsearch", "pg_jieba")


def e2e_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether PG18 E2E tests were explicitly enabled."""
    value = (env or os.environ).get(RUN_E2E_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def pg_conn_kwargs_from_env(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Build asyncpg kwargs, preferring E2E-specific env over app env."""
    source = env or os.environ

    def get(name: str, default: str) -> str:
        return (
            source.get(f"DLIGHTRAG_E2E_POSTGRES_{name}")
            or source.get(f"DLIGHTRAG_STORAGE__POSTGRES__{name}")
            or default
        )

    return {
        "host": get("HOST", "localhost"),
        "port": int(get("PORT", "5432")),
        "user": get("USER", "dlightrag"),
        "password": get("PASSWORD", "dlightrag"),
        "database": get("DATABASE", "dlightrag"),
    }


def missing_preload_libraries(setting: str) -> list[str]:
    """Return required libraries missing from shared_preload_libraries."""
    loaded = {item.strip().strip('"').strip("'") for item in setting.split(",") if item.strip()}
    return [name for name in REQUIRED_PRELOAD_LIBRARIES if name not in loaded]


@dataclass(frozen=True)
class PgPrereqReport:
    server_version: str
    server_major: int
    installed_extensions: tuple[str, ...]
    shared_preload_libraries: str

    @property
    def missing_extensions(self) -> list[str]:
        installed = set(self.installed_extensions)
        return [name for name in REQUIRED_EXTENSIONS if name not in installed]

    @property
    def missing_preload_libraries(self) -> list[str]:
        return missing_preload_libraries(self.shared_preload_libraries)


async def fetch_pg_prereq_report(conn: Any) -> PgPrereqReport:
    """Read PG major version, installed extensions, and preload settings."""
    version_num = int(await conn.fetchval("SHOW server_version_num"))
    version = str(await conn.fetchval("SHOW server_version"))
    preload = str(await conn.fetchval("SHOW shared_preload_libraries"))
    rows = await conn.fetch(
        "SELECT extname FROM pg_extension WHERE extname = ANY($1::text[]) ORDER BY extname",
        list(REQUIRED_EXTENSIONS),
    )
    return PgPrereqReport(
        server_version=version,
        server_major=version_num // 10000,
        installed_extensions=tuple(row["extname"] for row in rows),
        shared_preload_libraries=preload,
    )


def make_workspace_name(prefix: str = "e2e_pg18") -> str:
    """Build a PostgreSQL-safe workspace identifier."""
    token = hashlib.sha1(os.urandom(16)).hexdigest()[:10]
    return f"{prefix}_{token}"


def make_e2e_config(
    *,
    working_dir: Path,
    workspace: str,
    conn_kwargs: Mapping[str, Any],
) -> DlightragConfig:
    """Create a compact config for the local fake-model E2E smoke."""
    return DlightragConfig(
        _env_file=None,
        deployment={"workspace": workspace, "working_dir": str(working_dir)},
        storage={
            "postgres": {
                "host": str(conn_kwargs["host"]),
                "port": int(conn_kwargs["port"]),
                "user": str(conn_kwargs["user"]),
                "password": str(conn_kwargs["password"]),
                "database": str(conn_kwargs["database"]),
                "pool_min_size": 1,
                "pool_max_size": 2,
            }
        },
        models={
            "max_concurrency": 1,
            "chat": ModelRoleSettings(
                default=ModelSettings(
                    provider="openai",
                    model="e2e-fake-llm",
                    api_key="e2e-fake-key",
                    timeout=30,
                )
            ),
            "embedding": EmbeddingSettings(
                provider="voyage",
                model="e2e-fake-multimodal",
                api_key="e2e-fake-key",
                dim=8,
                max_token_size=1024,
                max_concurrency=1,
                batch_size=2,
                startup_probe=False,
            ),
            "rerank": {"enabled": False},
        },
        corpus={
            "ingestion": {
                "chunk_token_size": 128,
                "pipeline": {"max_concurrency": 1},
            },
            "retrieval": {"bm25_enabled": True},
        },
    )


def stable_vector(seed: str | bytes, *, dim: int = 8) -> list[float]:
    """Return a deterministic non-zero embedding vector."""
    raw = seed if isinstance(seed, bytes) else seed.encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    values = [((digest[i] / 255.0) * 2.0) - 1.0 for i in range(dim)]
    norm = sum(v * v for v in values) ** 0.5 or 1.0
    return [v / norm for v in values]


def image_seed(image: Image.Image) -> bytes:
    """Build a deterministic seed for a PIL image."""
    normalized = image.convert("RGB")
    return b"|".join(
        [
            str(normalized.size).encode("ascii"),
            normalized.mode.encode("ascii"),
            normalized.tobytes(),
        ]
    )


class FakeMultimodalEmbedder:
    """Small deterministic multimodal embedder for local E2E runs."""

    supports_asymmetric = True
    supports_images = True
    batch_size = 4

    def __init__(self, *, dim: int = 8) -> None:
        self.dim = dim
        self.model = "e2e-fake-multimodal"

    async def aclose(self) -> None:
        return None

    async def probe_image_embedding(self) -> None:
        return None

    async def embed_texts(
        self, texts: list[str], *, context: str = "document"
    ) -> list[list[float]]:
        return [stable_vector(f"{context}:{text}", dim=self.dim) for text in texts]

    async def embed_index_fused(self, items: list[tuple[str, Image.Image]]) -> list[list[float]]:
        return [
            stable_vector(f"{description}:{image_seed(image)}", dim=self.dim)
            for description, image in items
        ]

    async def embed_query_images(self, images: list[Image.Image]) -> list[list[float]]:
        return [stable_vector(f"query:{image_seed(image)}", dim=self.dim) for image in images]


def fake_embedding_func(*, dim: int = 8) -> Any:
    """Build a LightRAG EmbeddingFunc backed by deterministic local vectors."""
    from lightrag.utils import EmbeddingFunc

    async def embed(texts: list[str], *, context: str = "document") -> np.ndarray:
        return np.array([stable_vector(f"{context}:{text}", dim=dim) for text in texts])

    return EmbeddingFunc(
        embedding_dim=dim,
        max_token_size=1024,
        func=embed,
        model_name="e2e-fake-multimodal",
        supports_asymmetric=True,
    )


async def fake_lightrag_llm(prompt: str, **_: Any) -> str:
    """Return deterministic local extraction and query-keyword payloads."""
    if re.search(r"keyword", prompt, re.IGNORECASE):
        return '{"high_level_keywords": ["image"], "low_level_keywords": ["native image"]}'
    return (
        '{"entities":['
        '{"name":"LightRAG","type":"Artifact",'
        '"description":"LightRAG provides the document graph path."},'
        '{"name":"PostgreSQL","type":"Artifact",'
        '"description":"PostgreSQL stores the document graph."}'
        '],"relationships":['
        '{"source":"LightRAG","target":"PostgreSQL","keywords":"storage",'
        '"description":"LightRAG uses PostgreSQL storage."}'
        "]}"
    )


def install_fake_model_functions(monkeypatch: Any, *, dim: int = 8) -> FakeMultimodalEmbedder:
    """Patch WorkspaceRag model factories to avoid external network calls."""
    import dlightrag.engine.rag.workspace.workspace_rag as service_module

    multimodal_embedder = FakeMultimodalEmbedder(dim=dim)

    class FakeLightRagChatModels:
        @classmethod
        async def acreate(cls, *_args: Any, **_kwargs: Any) -> FakeLightRagChatModels:
            return cls()

        def __init__(self) -> None:
            self.default_func = fake_lightrag_llm
            self.role_configs = None

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(
        service_module,
        "LightRagChatModels",
        FakeLightRagChatModels,
    )
    monkeypatch.setattr(
        service_module,
        "build_product_reranker",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        service_module,
        "build_lightrag_embedding",
        lambda _settings, _embedder: fake_embedding_func(dim=dim),
    )
    monkeypatch.setattr(
        service_module,
        "create_embedding_model",
        lambda *_args, **_kwargs: multimodal_embedder,
    )
    return multimodal_embedder
