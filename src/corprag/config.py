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
    postgres_user: str = Field(default="rag")
    postgres_password: str = Field(default="rag")
    postgres_database: str = Field(default="corprag")
    postgres_workspace: str = Field(default="default")

    # pgvector HNSW index configuration
    pg_vector_index_type: str = Field(
        default="HNSW",
        description="pgvector index type (HNSW, IVFFlat, VCHORDRQ)",
    )
    pg_hnsw_m: int = Field(default=32, description="HNSW M parameter (connections per node)")
    pg_hnsw_ef: int = Field(default=300, description="HNSW ef parameter (search exploration)")

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
    llm_provider: Literal["azure_openai", "openai"] = Field(default="openai")

    # OpenAI
    openai_api_key: str | None = Field(default=None)
    openai_base_url: str | None = Field(default=None)
    openai_chat_model: str = Field(default="gpt-4.1-mini")
    openai_ingestion_model: str = Field(default="gpt-4.1-mini")
    openai_vision_model: str | None = Field(default="gpt-4.1-mini")

    # Azure OpenAI
    azure_openai_endpoint: str | None = Field(default=None)
    azure_openai_api_key: str | None = Field(default=None)
    azure_openai_deployment_name: str = Field(default="gpt-4.1-mini")
    azure_openai_ingestion_deployment_name: str = Field(default="gpt-4.1-mini")
    azure_openai_vision_deployment_name: str | None = Field(default="gpt-4.1-mini")

    # ===== Embedding =====
    embedding_model: str = Field(default="text-embedding-3-large")
    embedding_dim: int = Field(default=3072)

    # ===== Temperatures =====
    llm_temperature: float = Field(default=0.4)
    vision_temperature: float = Field(default=0.1)
    ingestion_temperature: float = Field(default=0.1)
    rerank_temperature: float = Field(default=0.0)

    # ===== LLM Request =====
    llm_request_timeout: int = Field(default=120)
    llm_max_retries: int = Field(default=2)

    # ===== Reranking =====
    enable_rerank: bool = Field(default=True)
    rerank_provider: Literal["llm", "cohere", "azure_cohere"] = Field(default="llm")
    rerank_model: str = Field(default="gpt-4.1-mini")
    cohere_api_key: str | None = Field(default=None)
    cohere_rerank_model: str = Field(default="rerank-v4.0-pro")
    azure_cohere_endpoint: str | None = Field(default=None)
    azure_cohere_api_key: str | None = Field(default=None)
    azure_cohere_deployment: str = Field(default="Cohere-rerank-v4.0-pro")

    # ===== RAG Processing =====
    working_dir: str = Field(default="./corprag_storage")
    parser: Literal["mineru", "docling"] = Field(default="mineru")
    parse_method: Literal["auto", "ocr", "txt"] = Field(default="auto")
    mineru_backend: str | None = Field(default=None)
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

    # ===== Knowledge Graph =====
    kg_entity_types: list[str] = Field(
        default=[
            "Product",
            "Component",
            "Technology",
            "Organization",
            "Standard",
            "Metric",
            "Design",
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
        description="Domain-specific hints for reranking. "
        "Injected by the calling application.",
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
        if self.llm_provider == "openai":
            return self.openai_chat_model
        return self.azure_openai_deployment_name

    @property
    def ingestion_model_name(self) -> str:
        if self.llm_provider == "openai":
            return self.openai_ingestion_model
        return self.azure_openai_ingestion_deployment_name

    @property
    def vision_model_name(self) -> str | None:
        if self.llm_provider == "openai":
            return self.openai_vision_model
        return self.azure_openai_vision_deployment_name

    @property
    def unified_api_key(self) -> str:
        if self.llm_provider == "openai":
            return self.openai_api_key or ""
        return self.azure_openai_api_key or ""

    @property
    def unified_base_url(self) -> str | None:
        if self.llm_provider == "openai":
            return self.openai_base_url
        return self.azure_openai_endpoint

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
            "POSTGRES_HNSW_EF": str(self.pg_hnsw_ef),
        }
        for key, value in pg_env_map.items():
            if key not in os.environ:
                os.environ[key] = value

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

    @model_validator(mode="after")
    def _validate_provider_fields(self) -> CorpragConfig:
        """Ensure provider-specific required fields are present."""
        if self.llm_provider == "azure_openai":
            if not self.azure_openai_endpoint or not self.azure_openai_api_key:
                raise ValueError(
                    "azure_openai_endpoint and azure_openai_api_key are required "
                    "for azure_openai provider"
                )
        elif self.llm_provider == "openai":
            if not self.openai_api_key:
                raise ValueError("openai_api_key is required for openai provider")
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


def reset_config() -> None:
    """Reset the global config singleton. Useful for testing."""
    global _config
    _config = None
