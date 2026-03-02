# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Self-contained configuration for corprag.

Uses pydantic-settings with CORPRAG_ env prefix. Works standalone (MCP/API server)
and as library (imported by other projects which construct CorpragConfig explicitly).

LightRAG reads POSTGRES_* env vars directly — model_post_init bridges
CORPRAG_POSTGRES_* → POSTGRES_* so both modes work seamlessly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Supported LLM providers
LLMProvider = Literal[
    "openai",
    "azure_openai",
    "anthropic",
    "google_gemini",
    "qwen",
    "minimax",
]


class CorpragConfig(BaseSettings):
    """Corporate RAG configuration.

    Configuration sources (in order of precedence):
        1. Constructor arguments (when used as library)
        2. Environment variables (CORPRAG_ prefix)
        3. .env file (development)
        4. Default values
    """

    model_config = SettingsConfigDict(
        env_prefix="CORPRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== PostgreSQL (Default Storage Backend) =====
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_user: str = Field(default="corprag")
    postgres_password: str = Field(default="corprag")
    postgres_database: str = Field(default="corprag")
    postgres_workspace: str = Field(default="default")

    # pgvector HNSW index configuration
    pg_vector_index_type: str = Field(
        default="HNSW",
        description="pgvector index type (HNSW, IVFFlat, VCHORDRQ)",
    )
    pg_hnsw_m: int = Field(default=32, description="HNSW M parameter (connections per node)")
    pg_hnsw_ef_construction: int = Field(
        default=300, description="HNSW ef_construction (index build quality)"
    )
    pg_hnsw_ef_search: int = Field(
        default=300, description="HNSW ef_search (query exploration, pgvector default is 40)"
    )

    # ===== Storage Backends (configurable, default PostgreSQL) =====
    vector_storage: str = Field(
        default="PGVectorStorage",
        description="LightRAG vector storage backend. Options: PGVectorStorage, "
        "MilvusVectorDBStorage, NanoVectorDBStorage, ChromaVectorDBStorage, "
        "FaissVectorDBStorage, QdrantVectorDBStorage",
    )
    graph_storage: str = Field(
        default="PGGraphStorage",
        description="LightRAG graph storage backend. Options: PGGraphStorage, "
        "Neo4JStorage, NetworkXStorage, MemgraphStorage",
    )
    kv_storage: str = Field(
        default="PGKVStorage",
        description="LightRAG KV storage backend. Options: PGKVStorage, "
        "JsonKVStorage, RedisKVStorage, MongoKVStorage",
    )
    doc_status_storage: str = Field(
        default="PGDocStatusStorage",
        description="LightRAG doc status backend. Options: PGDocStatusStorage, "
        "JsonDocStatusStorage, RedisDocStatusStorage, MongoDocStatusStorage",
    )

    # ===== Optional: Neo4j (if graph_storage=Neo4JStorage) =====
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_username: str = Field(default="neo4j")
    neo4j_password: str = Field(default="neo4j")

    # ===== Optional: Milvus (if vector_storage=MilvusVectorDBStorage) =====
    milvus_uri: str = Field(default="http://localhost:19530")
    milvus_user: str = Field(default="")
    milvus_password: str = Field(default="")
    milvus_token: str | None = Field(default=None)
    milvus_db_name: str = Field(default="default")
    milvus_hnsw_m: int = Field(default=32)
    milvus_hnsw_ef: int = Field(default=256)
    milvus_hnsw_ef_construction: int = Field(default=300)

    # ===== LLM Provider Configuration =====
    llm_provider: LLMProvider = Field(default="openai")

    # Optional: separate providers for embedding and vision (default to llm_provider)
    embedding_provider: LLMProvider | None = Field(
        default=None,
        description="Embedding provider. Defaults to llm_provider.",
    )
    vision_provider: LLMProvider | None = Field(
        default=None,
        description="Vision provider. Defaults to llm_provider.",
    )

    # ----- Unified Model Fields -----
    chat_model: str = Field(default="gpt-4.1-mini")
    ingestion_model: str = Field(default="gpt-4.1-mini")
    vision_model: str | None = Field(default="gpt-4.1-mini")

    # ----- Provider API Keys & Endpoints -----
    openai_api_key: str | None = Field(default=None)
    openai_base_url: str | None = Field(default=None)
    azure_openai_base_url: str | None = Field(default=None)
    azure_openai_api_key: str | None = Field(default=None)
    anthropic_api_key: str | None = Field(default=None)
    google_gemini_api_key: str | None = Field(default=None)
    qwen_api_key: str | None = Field(default=None)
    qwen_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    minimax_api_key: str | None = Field(default=None)
    minimax_base_url: str = Field(default="https://api.minimax.chat/v1")

    # ===== Embedding =====
    embedding_model: str = Field(default="text-embedding-3-large")
    embedding_dim: int = Field(default=1024)

    # ===== Temperatures =====
    llm_temperature: float = Field(default=0.5)
    vision_temperature: float = Field(default=0.1)
    ingestion_temperature: float = Field(default=0.1)
    rerank_temperature: float = Field(default=0.1)

    # ===== LLM Request =====
    llm_request_timeout: int = Field(default=120)
    llm_max_retries: int = Field(default=2)

    # ===== Reranking =====
    enable_rerank: bool = Field(default=True)
    rerank_backend: Literal["llm", "cohere", "azure_cohere"] = Field(default="llm")
    rerank_llm_provider: LLMProvider | None = Field(
        default=None,
        description="LLM provider for reranking when rerank_backend='llm'. "
        "Defaults to llm_provider.",
    )
    rerank_model: str | None = Field(
        default=None,
        description="Model for LLM reranking. Defaults to ingestion_model.",
    )
    cohere_api_key: str | None = Field(default=None)
    cohere_rerank_model: str = Field(default="rerank-v4.0-pro")
    azure_cohere_endpoint: str | None = Field(default=None)
    azure_cohere_api_key: str | None = Field(default=None)
    azure_cohere_deployment: str = Field(default="Cohere-rerank-v4.0-pro")

    # ===== RAG Processing =====
    working_dir: str = Field(default="./corprag_storage")
    parser: Literal["mineru", "docling"] = Field(default="mineru")
    parse_method: Literal["auto", "ocr", "txt"] = Field(default="auto")
    mineru_backend: str | None = Field(
        default=None,
        description="MinerU parsing backend. If None, auto-detects: CUDA GPU → hybrid-auto-engine, "
        "otherwise → pipeline. "
        "Valid values: pipeline, vlm-auto-engine, vlm-http-client, hybrid-auto-engine, hybrid-http-client.",
    )
    enable_image_processing: bool = Field(default=True)
    enable_table_processing: bool = Field(default=True)
    enable_equation_processing: bool = Field(default=False)
    display_content_stats: bool = Field(default=False)
    use_full_path: bool = Field(default=True)
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=52)
    context_window: int = Field(default=2)
    context_filter_types: str = Field(default="text,table")
    max_context_tokens: int = Field(default=3000)

    # ===== Ingestion Performance =====
    max_concurrent_ingestion: int = Field(default=4)
    max_parallel_insert: int = Field(default=6)
    max_async: int = Field(default=16)
    embedding_func_max_async: int = Field(default=8)
    embedding_batch_num: int = Field(default=5)
    ingestion_replace_default: bool = Field(default=False)

    # ===== Query Configuration =====
    top_k: int = Field(default=60)
    chunk_top_k: int = Field(default=30)
    max_entity_tokens: int = Field(default=8000)
    max_relation_tokens: int = Field(default=10000)
    max_total_tokens: int = Field(default=40000)
    default_mode: Literal["local", "global", "hybrid", "naive", "mix"] = Field(default="mix")
    max_conversation_turns: int = Field(default=20)
    max_conversation_tokens: int = Field(default=50000)

    # ===== Knowledge Graph =====
    kg_entity_types: list[str] = Field(
        default=[
            "Product",
            "Component",
            "Technology",
            "Design",
            "Genre",
            "Organization",
            "Standard",
            "Biography",
            "Location",
            "Event",
        ],
    )

    # ===== LibreOffice Conversion =====
    excel_auto_convert_to_pdf: bool = Field(default=True)
    excel_pdf_delete_original: bool = Field(default=True)
    libreoffice_timeout: int = Field(default=120)
    libreoffice_pdf_quality: int = Field(default=90)

    # ===== Sourcing (Optional) =====
    blob_connection_string: str | None = Field(default=None)
    snowflake_account: str | None = Field(default=None)
    snowflake_user: str | None = Field(default=None)
    snowflake_password: str | None = Field(default=None)
    snowflake_database: str | None = Field(default=None)
    snowflake_schema: str | None = Field(default=None)
    snowflake_warehouse: str | None = Field(default=None)

    # ===== Domain Knowledge (injected by caller) =====
    domain_knowledge_hints: str = Field(
        default="",
        description="Domain-specific hints for reranking. Injected by the calling application.",
    )

    # ===== MCP Server =====
    mcp_transport: Literal["stdio", "streamable-http"] = Field(default="stdio")
    mcp_host: str = Field(default="0.0.0.0")
    mcp_port: int = Field(default=8100)

    # ===== REST API Server =====
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8100)
    api_auth_token: str | None = Field(
        default=None,
        description="Bearer token for REST API authentication. If not set, no auth required.",
    )

    # ===== Computed Properties =====

    @property
    def working_dir_path(self) -> Path:
        return Path(self.working_dir).resolve()

    @property
    def sources_dir(self) -> Path:
        return self.working_dir_path / "sources"

    @property
    def artifacts_dir(self) -> Path:
        return self.working_dir_path / "artifacts"

    @property
    def chat_model_name(self) -> str:
        return self.chat_model

    @property
    def ingestion_model_name(self) -> str:
        return self.ingestion_model

    @property
    def vision_model_name(self) -> str | None:
        return self.vision_model

    @property
    def effective_embedding_provider(self) -> str:
        """Resolve embedding provider. Defaults to llm_provider."""
        return self.embedding_provider or self.llm_provider

    @property
    def effective_vision_provider(self) -> str:
        """Resolve vision provider. Defaults to llm_provider (all providers support vision)."""
        return self.vision_provider or self.llm_provider

    @property
    def effective_rerank_llm_provider(self) -> str:
        """Resolve LLM provider for reranking. Falls back to llm_provider."""
        return self.rerank_llm_provider or self.llm_provider

    @property
    def effective_rerank_model(self) -> str:
        """Resolve rerank model. Falls back to ingestion_model."""
        return self.rerank_model or self.ingestion_model

    def _get_provider_api_key(self, provider: str) -> str:
        """Get API key for a specific provider."""
        return getattr(self, f"{provider}_api_key", "") or ""

    def _get_provider_base_url(self, provider: str) -> str | None:
        """Get base URL for a specific provider."""
        return getattr(self, f"{provider}_base_url", None)

    def model_post_init(self, __context) -> None:
        """Bridge CORPRAG_POSTGRES_* to POSTGRES_* env vars for LightRAG.

        Also resolve relative working_dir to absolute path.
        """
        # Bridge PostgreSQL env vars for LightRAG
        pg_env_map = {
            "POSTGRES_HOST": self.postgres_host,
            "POSTGRES_PORT": str(self.postgres_port),
            "POSTGRES_USER": self.postgres_user,
            "POSTGRES_PASSWORD": self.postgres_password,
            "POSTGRES_DATABASE": self.postgres_database,
            "POSTGRES_WORKSPACE": self.postgres_workspace,
            "POSTGRES_VECTOR_INDEX_TYPE": self.pg_vector_index_type,
            "POSTGRES_HNSW_M": str(self.pg_hnsw_m),
            "POSTGRES_HNSW_EF": str(self.pg_hnsw_ef_construction),
        }
        for key, value in pg_env_map.items():
            if key not in os.environ:
                os.environ[key] = value

        # Inject hnsw.ef_search via POSTGRES_SERVER_SETTINGS
        # (LightRAG doesn't expose this; pgvector default is 40 which is too low)
        if "POSTGRES_SERVER_SETTINGS" not in os.environ:
            os.environ["POSTGRES_SERVER_SETTINGS"] = f"hnsw.ef_search={self.pg_hnsw_ef_search}"

        # Bridge Neo4j env vars if using Neo4JStorage
        if self.graph_storage == "Neo4JStorage":
            neo4j_env_map = {
                "NEO4J_URI": self.neo4j_uri,
                "NEO4J_USERNAME": self.neo4j_username,
                "NEO4J_PASSWORD": self.neo4j_password,
            }
            for key, value in neo4j_env_map.items():
                if key not in os.environ:
                    os.environ[key] = value

        # Bridge Milvus env vars if using MilvusVectorDBStorage
        if self.vector_storage == "MilvusVectorDBStorage":
            if "MILVUS_DB_NAME" not in os.environ:
                os.environ["MILVUS_DB_NAME"] = self.milvus_db_name

        # Resolve working_dir to absolute path
        path = Path(self.working_dir)
        if not path.is_absolute():
            self.working_dir = str(path.resolve())

    def _check_provider_key(self, provider: str, context: str) -> None:
        """Raise ValueError if the provider's API key is missing."""
        field = f"{provider}_api_key"
        if hasattr(self, field) and not getattr(self, field):
            raise ValueError(f"{field} is required for {context}")

    @model_validator(mode="after")
    def _validate_provider_fields(self) -> CorpragConfig:
        """Ensure provider-specific required fields are present."""
        # Validate primary LLM provider
        self._check_provider_key(self.llm_provider, f"{self.llm_provider} provider")

        # Azure OpenAI also requires endpoint
        if self.llm_provider == "azure_openai" and not self.azure_openai_base_url:
            raise ValueError("azure_openai_base_url is required for azure_openai provider")

        # Validate embedding provider if it differs from llm_provider
        eff_emb = self.effective_embedding_provider
        if eff_emb != self.llm_provider:
            self._check_provider_key(eff_emb, f"embedding_provider={eff_emb}")

        # Validate vision provider if it differs from llm_provider
        eff_vis = self.effective_vision_provider
        if eff_vis != self.llm_provider:
            self._check_provider_key(eff_vis, f"vision_provider={eff_vis}")

        # Validate rerank LLM provider if it differs from llm_provider
        if self.rerank_backend == "llm":
            eff_rerank = self.effective_rerank_llm_provider
            if eff_rerank != self.llm_provider:
                self._check_provider_key(eff_rerank, f"rerank_llm_provider={eff_rerank}")

        return self


# Singleton for standalone mode (MCP/API server)
_config: CorpragConfig | None = None


def get_config() -> CorpragConfig:
    """Get global corprag configuration (singleton).

    For standalone use (MCP/API server). When used as a library,
    construct CorpragConfig directly and pass it to RAGService.
    """
    global _config
    if _config is None:
        _config = CorpragConfig()  # type: ignore[call-arg]
    return _config


def set_config(config: CorpragConfig) -> None:
    """Set the global config singleton. Useful for testing."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global config singleton. Useful for testing."""
    global _config
    _config = None
