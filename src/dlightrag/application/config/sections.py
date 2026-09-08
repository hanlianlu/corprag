# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Self-contained configuration for dlightrag.

Configuration sources (highest → lowest precedence):
    1. Constructor arguments (when used as library)
    2. Environment variables (DLIGHTRAG_ prefix)
    3. .env file (secrets + deployment-only overrides)
    4. config.yaml (structured app settings)
    5. Default values

LightRAG reads backend-specific env vars directly — model_post_init bridges
DLIGHTRAG_* → backend env vars so both modes work seamlessly.
"""

import json
import math
import os
import re
import ssl
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlencode, urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from dlightrag.engine.ai.settings import (
    FrozenSettings,
    ModelsSettings,
    freeze_settings_value,
    thaw_settings_value,
)
from dlightrag.engine.rag.workspace.settings import (
    CorpusSettings,
    DoclingSidecarSettings,
    MinerUSidecarSettings,
    VLMSidecarSettings,
)

type ServiceRole = Literal["writer", "reader"]

_YAML_FILE = "config.yaml"
_ENV_FILE = ".env"
_PG_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PG_QUALIFIED_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")
_LOCAL_MCP_ALLOWED_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
_LOCAL_MCP_ALLOWED_ORIGINS = [
    "http://127.0.0.1:*",
    "http://localhost:*",
    "http://[::1]:*",
]
_LOCAL_API_HOSTS = {"127.0.0.1", "localhost", "::1"}
_AUXILIARY_ENV_NAMES = {
    "DLIGHTRAG_API_TOKEN",
    "DLIGHTRAG_API_URL",
    "DLIGHTRAG_CLIENT_TIMEOUT",
    "DLIGHTRAG_OPENAI_API_KEY",
    # Compose-only PostgreSQL container tuning; these are interpolated by
    # docker-compose and intentionally are not application config fields.
    "DLIGHTRAG_POSTGRES_EFFECTIVE_CACHE_SIZE",
    "DLIGHTRAG_POSTGRES_MAINTENANCE_WORK_MEM",
    "DLIGHTRAG_POSTGRES_MAX_CONNECTIONS",
    "DLIGHTRAG_POSTGRES_SHARED_BUFFERS",
    "DLIGHTRAG_POSTGRES_SHM_SIZE",
    "DLIGHTRAG_POSTGRES_WORK_MEM",
    "DLIGHTRAG_RUN_E2E_PG18",
}
_AUXILIARY_ENV_PREFIXES = ("DLIGHTRAG_E2E_",)
PostgresSSLMode = Literal["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]


def _validate_oauth_endpoint_url(value: str, field_name: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError(f"{field_name} must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError(f"{field_name} must not include credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{field_name} must not include query or fragment components")
    if parsed.scheme != "https" and parsed.hostname not in _LOCAL_API_HOSTS:
        raise ValueError(f"{field_name} must use HTTPS except on loopback")


MinerULanguage = Literal[
    "ch",
    "ch_server",
    "korean",
    "ta",
    "te",
    "ka",
    "th",
    "el",
    "arabic",
    "east_slavic",
    "cyrillic",
    "devanagari",
]
MinerULocalBackend = Literal[
    "pipeline",
    "vlm-engine",
    "hybrid-engine",
]


# Auto-derived from the typed sidecar models below.
_LIGHTRAG_SIDECAR_ENV_KEYS: frozenset[str] = frozenset()  # populated after class definitions


def _find_env_file() -> Path | None:
    """Locate .env in the current working directory only."""
    candidate = Path(_ENV_FILE)
    return candidate if candidate.is_file() else None


def _find_yaml_config() -> Path | None:
    """Locate config.yaml in the current working directory."""
    cwd = Path(_YAML_FILE)
    if cwd.is_file():
        return cwd
    return None


class CitationHighlightConfig(BaseModel):
    """Optional semantic highlighting for cited source snippets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    timeout: float = Field(default=10.0, gt=0)
    max_concurrency: int = Field(default=8, ge=1)
    batch_size: int = Field(default=8, ge=1)
    max_input_chars: int = Field(default=4096, ge=1)
    cache_size: int = Field(default=500, ge=1)


class CitationsConfig(BaseModel):
    """Citation validation and UI enrichment configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    highlights: CitationHighlightConfig = Field(default_factory=CitationHighlightConfig)


class AnswerConfig(BaseModel):
    """Final answer generation controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_attachments: int = Field(
        default=6,
        ge=0,
        description="Maximum answer attachments admitted per request.",
    )
    max_attachment_bytes: int = Field(
        default=100 * 1024 * 1024,
        ge=1,
        description="Maximum bytes accepted for a single answer attachment (100 MiB).",
    )
    max_total_attachment_bytes: int = Field(
        default=128 * 1024 * 1024,
        ge=1,
        description="Maximum total bytes accepted across all answer attachments (128 MiB).",
    )
    max_images: int = Field(
        default=12,
        ge=0,
        description="Maximum current and retrieved image blocks sent to the answer LLM.",
    )

    # Vision support is runtime Answer state, not config. Users do not set it
    # in config.yaml; the startup probe records it on Application health.
    image_max_bytes: int = Field(
        default=3_000_000,
        ge=1,
        description="Maximum compressed binary bytes per answer image.",
    )
    image_max_total_bytes: int = Field(
        default=24_000_000,
        ge=1,
        description="Maximum total compressed binary image bytes per answer request.",
    )
    image_max_px: int = Field(
        default=1536,
        ge=1,
        description="Maximum image long edge sent to the answer LLM.",
    )
    image_max_pixels: int = Field(
        default=40_000_000,
        ge=1,
        description="Maximum decoded source pixels accepted for answer and Web images.",
    )
    image_min_px: int = Field(
        default=1024,
        ge=1,
        description="Minimum long edge preserved before skipping oversized answer images.",
    )
    image_quality: int = Field(
        default=89,
        ge=1,
        le=95,
        description="Initial JPEG quality for answer LLM image previews.",
    )
    image_min_quality: int = Field(
        default=79,
        ge=1,
        le=95,
        description="Minimum JPEG quality before skipping oversized answer images.",
    )


class LaneRuntimeConfig(BaseModel):
    """Per-lane local workers and deployment-wide nonterminal admission limit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    worker_concurrency: int = Field(ge=1)
    max_nonterminal_runs: int = Field(ge=1)


class RuntimeConfig(BaseModel):
    """Operation-neutral durable RunRuntime admission and retention."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: LaneRuntimeConfig = LaneRuntimeConfig(
        worker_concurrency=16,
        max_nonterminal_runs=30_000,
    )
    corpus_mutation: LaneRuntimeConfig = LaneRuntimeConfig(
        worker_concurrency=2,
        max_nonterminal_runs=1_000,
    )
    run_retention_days: int = Field(
        default=365,
        ge=1,
        description=(
            "Retention floor for terminal Answer Runs and their event logs, "
            "counted from finished_at; top-level Retrieval uses seven days. "
            "Also the retention clock for memory: superseded profile history "
            "is purged after the same span. A conversation whose last turn's "
            "Run ages out becomes empty and is then reclaimed."
        ),
    )


class OutboundMcpServerConfig(BaseModel):
    """One thin deployment-configured outbound MCP endpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    transport: Literal["stdio", "streamable-http"]
    tools: tuple[str, ...] = Field(min_length=1)
    command: str | None = None
    args: tuple[str, ...] = ()
    url: str | None = None

    @model_validator(mode="after")
    def _validate_endpoint(self) -> OutboundMcpServerConfig:
        if self.transport == "stdio" and (not self.command or self.url):
            raise ValueError("stdio outbound MCP requires command and forbids url")
        if self.transport == "streamable-http" and (not self.url or self.command):
            raise ValueError("streamable-http outbound MCP requires url and forbids command")
        if len(set(self.tools)) != len(self.tools):
            raise ValueError("outbound MCP tool names must be unique")
        return self


class ArtifactPublicationConfig(BaseModel):
    """Independent Agent workspace, publication, and browser-preview budgets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_artifacts: int = Field(default=20, ge=1)
    max_file_bytes: int = Field(default=30 * 1024 * 1024, ge=1)
    max_total_bytes: int = Field(default=100 * 1024 * 1024, ge=1)
    workspace_max_bytes: int = Field(
        default=1024 * 1024 * 1024,
        ge=1,
        le=5 * 1024 * 1024 * 1024,
    )
    preview_image_max_pixels: int = Field(default=16_000_000, ge=1)
    preview_image_max_edge: int = Field(default=4096, ge=1)
    original_image_max_pixels: int = Field(default=64_000_000, ge=1)
    original_image_max_edge: int = Field(default=8000, ge=1)
    active_html_max_bytes: int = Field(default=20 * 1024 * 1024, ge=1)


class AgentExecutionConfig(BaseModel):
    """Optional Agent execution; sandbox requires a trusted adapter extension."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("skills_root", "owner_skills_root")
    @classmethod
    def _validate_skills_roots(cls, value: str | None) -> str | None:
        if value is not None and not Path(value).expanduser().is_absolute():
            raise ValueError("skill roots must be absolute paths when set")
        return value

    execution_environment: Literal["disabled", "trust", "sandbox"] = Field(
        default="trust",
        description=(
            "disabled | trust | sandbox. Default trust runs the rooted local "
            "adapter; set disabled to expose no path or Bash tools. sandbox "
            "requires an installed adapter and never downgrades."
        ),
    )
    workspace_root: str | None = Field(
        default=None,
        description=(
            "Absolute Agent Workspace root. When trust or sandbox and unset, "
            "defaults to ~/.dlightrag/agent_workspaces. Multi-host deployments "
            "must set the same absolute path on every worker."
        ),
    )
    skills_root: str | None = Field(
        default=None,
        description=(
            "Absolute global Agent Skills root. When unset, defaults to "
            "~/.dlightrag/skills. Multi-host deployments must set the same "
            "shared absolute path on every worker."
        ),
    )
    owner_skills_root: str | None = Field(
        default=None,
        description=(
            "Absolute per-owner Agent Skills root. When unset, defaults to "
            "~/.dlightrag/owner_skills. Users publish their own skills here "
            "through the validated publish_skill tool; the global root stays "
            "operator-provisioned and read-only."
        ),
    )
    publication: ArtifactPublicationConfig = Field(default_factory=ArtifactPublicationConfig)
    outbound_mcp: tuple[OutboundMcpServerConfig, ...] = ()


class WebConversationsConfig(BaseModel):
    """Browser conversation surface; retention follows RuntimeConfig."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    active_html_preview_enabled: bool = Field(
        default=True,
        description="Allow explicit opaque-origin execution of self-contained HTML Artifacts.",
    )


class WebSourceProviderConfig(BaseModel):
    """Credential for one first-class Web source provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_key: str | None = None

    @field_validator("api_key", mode="before")
    @classmethod
    def _normalize_api_key(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


class WebSourcesConfig(BaseModel):
    """Independent ordered Search and Extract provider chains.

    ``None`` derives an order from configured keys (Exa, then Tavily). An
    explicit empty tuple disables that operation while retaining credentials.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    search_providers: tuple[Literal["exa", "tavily"], ...] | None = None
    extract_providers: tuple[Literal["exa", "tavily"], ...] | None = None
    exa: WebSourceProviderConfig = Field(default_factory=WebSourceProviderConfig)
    tavily: WebSourceProviderConfig = Field(default_factory=WebSourceProviderConfig)

    @field_validator("search_providers", "extract_providers")
    @classmethod
    def _unique_provider_order(
        cls, value: tuple[Literal["exa", "tavily"], ...] | None
    ) -> tuple[Literal["exa", "tavily"], ...] | None:
        if value is None:
            return None
        if len(set(value)) != len(value):
            raise ValueError("Web provider order cannot contain duplicates")
        return value

    def configured_providers(self) -> tuple[Literal["exa", "tavily"], ...]:
        return tuple(
            name
            for name, configured in (("exa", self.exa.api_key), ("tavily", self.tavily.api_key))
            if configured
        )  # type: ignore[return-value]

    def search_order(self) -> tuple[Literal["exa", "tavily"], ...]:
        return (
            self.search_providers
            if self.search_providers is not None
            else self.configured_providers()
        )

    def extract_order(self) -> tuple[Literal["exa", "tavily"], ...]:
        return (
            self.extract_providers
            if self.extract_providers is not None
            else self.configured_providers()
        )

    @model_validator(mode="after")
    def _orders_have_credentials(self) -> WebSourcesConfig:
        keys = {"exa": self.exa.api_key, "tavily": self.tavily.api_key}
        for operation, order in (
            ("search", self.search_order()),
            ("extract", self.extract_order()),
        ):
            missing = [name for name in order if not keys[name]]
            if missing:
                raise ValueError(f"Web {operation} provider(s) lack api_key: {', '.join(missing)}")
        return self


class AccessControlRuleConfig(BaseModel):
    """Map one verified JWT claim value to DlightRAG actions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim: str
    value: str
    workspaces: list[str] = Field(default_factory=lambda: ["*"])
    actions: list[str]


class AccessControlConfig(BaseModel):
    """DlightRAG resource authorization settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["allow_all", "jwt_claims"] = "allow_all"
    rules: list[AccessControlRuleConfig] = Field(default_factory=list)


def _redact_dict(data: dict[str, Any], patterns: tuple[str, ...]) -> dict[str, Any]:
    """Recursively redact values whose keys match sensitive patterns."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if any(pattern in key.lower() for pattern in patterns):
            if isinstance(value, str) and len(value) > 8:
                result[key] = value[:4] + "***" + value[-4:]
            elif isinstance(value, str):
                result[key] = "***"
            else:
                result[key] = value
        elif isinstance(value, dict):
            result[key] = _redact_dict(value, patterns)
        elif isinstance(value, list):
            result[key] = [
                _redact_dict(item, patterns) if isinstance(item, dict) else item for item in value
            ]
        else:
            result[key] = value
    return result


class WebIdentitySettings(BaseModel):
    """Edge-asserted identity source for the Web surface.

    The browser front door already authenticated the human (Cloudflare Access,
    Azure Easy Auth, or AWS Amplify/CloudFront); the Web surface verifies the
    edge credential per request and never renders a login page or issues a
    token of its own. JWKS and issuer values are public material and may live
    in ``config.yaml``.
    """

    edge: Literal["cloudflare", "azure", "aws"] | None = Field(
        default=None,
        description=(
            "Edge identity provider. None keeps the existing pasted-token Web "
            "login (development/operator hatch)."
        ),
    )
    issuer: str | None = Field(
        default=None,
        description=(
            "Expected edge-token issuer: https://<team>.cloudflareaccess.com for "
            "Cloudflare, https://login.microsoftonline.com/<tenant>/v2.0 for Azure, "
            "the IdP issuer for AWS."
        ),
    )
    audience: Annotated[str | list[str] | None, NoDecode] = Field(
        default=None,
        description="Expected edge-token audience (Cloudflare AUD tag; AAD client id).",
    )
    jwks_url: str | None = Field(
        default=None,
        description=(
            "JWKS endpoint for edge-token signing keys. Optional: Cloudflare "
            "derives its team certs endpoint from the issuer."
        ),
    )

    @field_validator("audience", mode="before")
    @classmethod
    def _normalize_audience(cls, value: Any) -> str | list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if not text.startswith("["):
                return text
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "web_identity.audience string must be a plain audience or a JSON array"
                ) from exc
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return value
        raise ValueError("web_identity.audience must be a string or a list of strings")

    @model_validator(mode="after")
    def _validate_edge(self) -> Self:
        if self.edge is None:
            return self
        if not self.issuer:
            raise ValueError("web_identity.edge requires web_identity.issuer")
        if not self.audience:
            raise ValueError("web_identity.edge requires web_identity.audience")
        if self.edge == "aws" and not self.jwks_url:
            raise ValueError("web_identity.edge='aws' requires web_identity.jwks_url")
        return self


class DeploymentSettings(FrozenSettings):
    service_role: ServiceRole = "writer"
    workspace: str = "default"
    working_dir: str = "./dlightrag_storage"

    @field_validator("working_dir")
    @classmethod
    def _absolute_working_dir(cls, value: str) -> str:
        return str(Path(value).resolve())

    @property
    def working_dir_path(self) -> Path:
        return Path(self.working_dir)


class PostgresSettings(FrozenSettings):
    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    user: str = "dlightrag"
    password: str = "dlightrag"  # noqa: S105 - local development default
    database: str = "dlightrag"
    ssl_mode: PostgresSSLMode | None = None
    ssl_cert: str | None = None
    ssl_key: str | None = None
    ssl_root_cert: str | None = None
    ssl_crl: str | None = None
    pool_min_size: int = 2
    pool_max_size: int = 16
    command_timeout: float | None = Field(default=60.0, gt=0)
    acquire_timeout: float = Field(default=30.0, gt=0)
    lightrag_pool_max_size: int = Field(default=16, ge=1)
    session_settings: Mapping[str, str | int | float | bool] = Field(default_factory=dict)
    statement_cache_size: int | None = None
    connection_retries: int = Field(default=10, ge=1, le=100)
    connection_retry_backoff: float = Field(default=3.0, ge=0, le=300)
    connection_retry_backoff_max: float = Field(default=30.0, ge=0, le=600)
    pool_close_timeout: float = Field(default=5.0, ge=0, le=30)

    @field_validator("session_settings", mode="after")
    @classmethod
    def _freeze_session_settings(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze_settings_value(value)

    @field_serializer("session_settings")
    def _serialize_session_settings(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return thaw_settings_value(value)


class LightRAGStorageSettings(FrozenSettings):
    """Deployment-static LightRAG storage names and narrow vector bindings."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
        hide_input_in_errors=True,
    )

    vector_index_type: Literal["HNSW", "HNSW_HALFVEC", "IVFFLAT", "VCHORDRQ"] = "HNSW_HALFVEC"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 256
    hnsw_ef_search: int = 256
    vector_storage: Literal["PGVectorStorage", "MilvusVectorDBStorage"] = "PGVectorStorage"
    graph_storage: Literal["PGTableGraphStorage"] = "PGTableGraphStorage"
    kv_storage: Literal["PGKVStorage"] = "PGKVStorage"
    doc_status_storage: Literal["PGDocStatusStorage"] = "PGDocStatusStorage"
    milvus_uri: str | None = None
    milvus_token: str | None = None
    milvus_db_name: str | None = None
    vector_db_kwargs: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("milvus_uri", "milvus_token", "milvus_db_name", mode="before")
    @classmethod
    def _normalize_optional_milvus_binding(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        return value.strip() or None

    @field_validator("vector_db_kwargs", mode="after")
    @classmethod
    def _freeze_vector_kwargs(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        if len(value) > 16:
            raise ValueError("vector_db_kwargs accepts at most 16 entries")
        normalized: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key).strip()
            if not key or len(key) > 64:
                raise ValueError("vector_db_kwargs keys must contain 1 to 64 characters")
            if any(marker in key.lower() for marker in ("password", "secret", "token", "uri")):
                raise ValueError("vector_db_kwargs must not contain credentials or endpoint URIs")
            if item is not None and not isinstance(item, str | int | float | bool):
                raise ValueError("vector_db_kwargs values must be scalar")
            if isinstance(item, str) and len(item) > 128:
                raise ValueError("vector_db_kwargs string values must not exceed 128 characters")
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("vector_db_kwargs numeric values must be finite")
            normalized[key] = item
        if len(json.dumps(normalized, separators=(",", ":"), default=str).encode()) > 4096:
            raise ValueError("vector_db_kwargs exceeds the 4096-byte configuration bound")
        return freeze_settings_value(normalized)

    @model_validator(mode="after")
    def _validate_vector_kwargs(self) -> Self:
        if self.vector_storage != "MilvusVectorDBStorage":
            if any((self.milvus_uri, self.milvus_token, self.milvus_db_name)):
                raise ValueError(
                    "milvus_uri, milvus_token, and milvus_db_name require "
                    "vector_storage='MilvusVectorDBStorage'"
                )
            return self
        supported = {
            "cosine_better_than_threshold",
            "index_type",
            "metric_type",
            "hnsw_m",
            "hnsw_ef_construction",
            "hnsw_ef",
            "sq_type",
            "sq_refine",
            "sq_refine_type",
            "sq_refine_k",
            "ivf_nlist",
            "ivf_nprobe",
        }
        unknown = sorted(set(self.vector_db_kwargs).difference(supported))
        if unknown:
            raise ValueError(
                "Milvus vector_db_kwargs contains unsupported keys: " + ", ".join(unknown)
            )
        return self

    @field_serializer("vector_db_kwargs")
    def _serialize_vector_kwargs(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return thaw_settings_value(value)


class StorageSettings(FrozenSettings):
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    lightrag: LightRAGStorageSettings = Field(default_factory=LightRAGStorageSettings)


class AnswerSectionSettings(FrozenSettings):
    generation: AnswerConfig = Field(default_factory=AnswerConfig)
    agent: AgentExecutionConfig = Field(default_factory=AgentExecutionConfig)
    citations: CitationsConfig = Field(default_factory=CitationsConfig)
    conversations: WebConversationsConfig = Field(default_factory=WebConversationsConfig)
    web_sources: WebSourcesConfig = Field(default_factory=WebSourcesConfig)


class AccessSectionSettings(FrozenSettings):
    auth_mode: Literal["none", "simple", "jwt"] = "none"
    api_token: str | None = None
    allow_insecure_no_auth: bool = False
    jwt_verification_key: str | None = None
    jwt_jwks_url: str | None = None
    jwt_issuer: str | None = None
    jwt_audience: Annotated[str | tuple[str, ...] | None, NoDecode] = None
    jwt_algorithm: Literal["HS256", "HS384", "HS512", "RS256", "RS384", "RS512", "ES256"] = "HS256"
    cors_allow_origins: tuple[str, ...] = ("*",)
    web_identity: WebIdentitySettings = Field(default_factory=WebIdentitySettings)
    control: AccessControlConfig = Field(default_factory=AccessControlConfig)

    @field_validator("jwt_audience", mode="before")
    @classmethod
    def _normalize_audience(cls, value: Any) -> str | tuple[str, ...] | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if not text.startswith("["):
                return text
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("jwt_audience must be a plain audience or JSON array") from exc
        if isinstance(value, (list, tuple)):
            items = tuple(str(item).strip() for item in value if str(item).strip())
            return items or None
        raise ValueError("jwt_audience must be a string or sequence of strings")


class ApiInterfaceSettings(FrozenSettings):
    host: str = "127.0.0.1"
    port: int = Field(default=8100, ge=1, le=65535)


class McpInterfaceSettings(FrozenSettings):
    transport: Literal["stdio", "streamable-http"] = "stdio"
    host: str = "127.0.0.1"
    port: int = Field(default=8101, ge=1, le=65535)
    allowed_hosts: tuple[str, ...] = tuple(_LOCAL_MCP_ALLOWED_HOSTS)
    allowed_origins: tuple[str, ...] = tuple(_LOCAL_MCP_ALLOWED_ORIGINS)
    resource_server_url: str | None = None


class InterfacesSettings(FrozenSettings):
    api: ApiInterfaceSettings = Field(default_factory=ApiInterfaceSettings)
    mcp: McpInterfaceSettings = Field(default_factory=McpInterfaceSettings)
    max_upload_size_mb: int = Field(default=512, ge=1)


class ObservabilitySettings(FrozenSettings):
    log_level: str = "info"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_export_external_spans: bool = False
    langfuse_trace_sensitive_data: bool = True
    langfuse_environment: str | None = None
    langfuse_release: str | None = None
    langfuse_sample_rate: float = Field(default=1.0, ge=0, le=1)
    langfuse_timeout: int | None = Field(default=None, ge=1, le=300)
    langfuse_flush_at: int | None = Field(default=None, ge=1)
    langfuse_flush_interval: float | None = Field(default=None, ge=0.1, le=300)


class DlightragConfig(BaseSettings):
    """The nine-section immutable DlightRAG configuration."""

    _SECRET_PATTERNS: tuple[str, ...] = (
        "api_key",
        "api_secret",
        "api_token",
        "secret",
        "verification_key",
        "password",
        "connection_string",
        "milvus_uri",
        "account_key",
        "sas_token",
        "token",
    )
    model_config = SettingsConfigDict(
        env_prefix="DLIGHTRAG_",
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
        env_file=_find_env_file(),
        env_file_encoding="utf-8",
        dotenv_filtering="match_prefix",
        case_sensitive=False,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )

    def __init__(self, **values: Any) -> None:
        allowed = {name.upper() for name in self.__class__.model_fields}
        unknown = sorted(
            key
            for key in os.environ
            if key.startswith("DLIGHTRAG_")
            and key not in _AUXILIARY_ENV_NAMES
            and not key.startswith(_AUXILIARY_ENV_PREFIXES)
            and key.removeprefix("DLIGHTRAG_").split("__", 1)[0].upper() not in allowed
        )
        if unknown:
            raise ValueError(f"Unknown DlightRAG environment variables: {unknown}")
        super().__init__(**values)
        # BaseSettings serializes constructor-supplied nested models through its
        # source pipeline. Restore the already-validated canonical instances so
        # explicit-field provenance (notably keyless vs incomplete model roles)
        # and object identity survive composition.
        for field_name in self.__class__.model_fields:
            supplied = values.get(field_name)
            if isinstance(supplied, BaseModel):
                object.__setattr__(self, field_name, supplied)

    deployment: DeploymentSettings = Field(default_factory=DeploymentSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    models: ModelsSettings = Field(default_factory=ModelsSettings)
    corpus: CorpusSettings = Field(default_factory=CorpusSettings)
    answer: AnswerSectionSettings = Field(default_factory=AnswerSectionSettings)
    runtime: RuntimeConfig = RuntimeConfig()
    access: AccessSectionSettings = Field(default_factory=AccessSectionSettings)
    interfaces: InterfacesSettings = Field(default_factory=InterfacesSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        from pydantic_settings import YamlConfigSettingsSource

        sources = [init_settings, env_settings, dotenv_settings]
        if (yaml_path := _find_yaml_config()) is not None:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path))
        sources.append(file_secret_settings)
        return tuple(sources)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return _redact_dict(super().model_dump(**kwargs), self._SECRET_PATTERNS)

    def model_dump_json(self, **kwargs: Any) -> str:
        indent = kwargs.pop("indent", None)
        return json.dumps(self.model_dump(**kwargs), default=str, indent=indent)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(f'{k}={v!r}' for k, v in self.model_dump().items())})"

    @model_validator(mode="after")
    def _validate_config(self) -> Self:
        profiles = self.corpus.retrieval.bm25_profiles
        if len({profile.name for profile in profiles}) != len(profiles):
            raise ValueError("bm25_profiles names must be unique")
        if self.corpus.retrieval.bm25_enabled and not any(p.fallback for p in profiles):
            raise ValueError("bm25_profiles must include at least one fallback profile")
        self._validate_auth()
        self._validate_storage_composition()
        return self

    def _validate_storage_composition(self) -> None:
        lightrag = self.storage.lightrag
        if lightrag.vector_storage != "MilvusVectorDBStorage":
            return
        if self.is_reader:
            raise ValueError(
                "service_role='reader' does not support MilvusVectorDBStorage; "
                "use the writer role or PGVectorStorage"
            )
        promotion = self.corpus.promotion
        if promotion.doc_threshold is not None or promotion.chunk_threshold is not None:
            raise ValueError(
                "workspace promotion thresholds require PGVectorStorage and cannot be used "
                "with MilvusVectorDBStorage"
            )

    def _validate_auth(self) -> None:
        access, api, mcp = self.access, self.interfaces.api, self.interfaces.mcp
        if access.auth_mode == "none" and access.api_token:
            raise ValueError("api_token is set; configure auth_mode='simple' explicitly")
        if access.auth_mode == "simple" and not access.api_token:
            raise ValueError("auth_mode='simple' requires api_token")
        if access.auth_mode == "jwt" and not (access.jwt_verification_key or access.jwt_jwks_url):
            raise ValueError("auth_mode='jwt' requires jwt_verification_key or jwt_jwks_url")
        if access.jwt_jwks_url and not (access.jwt_issuer and access.jwt_audience):
            raise ValueError("jwt_jwks_url requires jwt_issuer and jwt_audience")
        if access.web_identity.edge and access.auth_mode != "jwt":
            raise ValueError("web_identity.edge requires auth_mode='jwt'")
        if mcp.resource_server_url:
            if access.auth_mode != "jwt" or mcp.transport != "streamable-http":
                raise ValueError("mcp.resource_server_url requires JWT and streamable-http")
            if not access.jwt_issuer:
                raise ValueError("mcp.resource_server_url requires jwt_issuer")
            _validate_oauth_endpoint_url(mcp.resource_server_url, "mcp.resource_server_url")
            _validate_oauth_endpoint_url(access.jwt_issuer, "jwt_issuer")
        insecure = []
        if api.host not in _LOCAL_API_HOSTS:
            insecure.append(f"REST host={api.host}")
        if mcp.transport == "streamable-http" and mcp.host not in _LOCAL_API_HOSTS:
            insecure.append(f"MCP host={mcp.host}")
        if access.auth_mode == "none" and insecure:
            if not access.allow_insecure_no_auth:
                raise ValueError("auth_mode='none' with non-loopback listeners is refused")
            warnings.warn(
                "auth_mode='none' on non-loopback listeners (allow_insecure_no_auth=true)",
                stacklevel=2,
            )
        if access.auth_mode != "none" and access.cors_allow_origins == ("*",):
            warnings.warn(
                "auth_mode is enabled but wildcard CORS rejects credentials", stacklevel=2
            )
        if access.control.mode != "allow_all":
            if access.auth_mode != "jwt" or not access.control.rules:
                raise ValueError("jwt_claims access control requires JWT and rules")

    @property
    def working_dir_path(self) -> Path:
        return Path(self.deployment.working_dir)

    @property
    def temp_dir(self) -> Path:
        return self.working_dir_path / ".tmp"

    @property
    def input_dir_path(self) -> Path:
        return self.working_dir_path / "inputs"

    @property
    def is_reader(self) -> bool:
        return self.deployment.service_role == "reader"

    @property
    def max_upload_batch_bytes(self) -> int:
        return self.interfaces.max_upload_size_mb * 1024 * 1024

    @property
    def parser_rules(self) -> str:
        return self.corpus.parser_rules

    def _pg_ssl_value(self) -> ssl.SSLContext | bool | None:
        pg = self.storage.postgres
        if pg.ssl_mode is None:
            return None
        if pg.ssl_mode in {"require", "prefer"}:
            return True
        if pg.ssl_mode == "disable":
            return False
        if pg.ssl_mode == "allow":
            return None
        try:
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            context.check_hostname = pg.ssl_mode == "verify-full"
            if pg.ssl_root_cert and Path(pg.ssl_root_cert).exists():
                context.load_verify_locations(cafile=pg.ssl_root_cert)
            if (
                pg.ssl_cert
                and pg.ssl_key
                and Path(pg.ssl_cert).exists()
                and Path(pg.ssl_key).exists()
            ):
                context.load_cert_chain(pg.ssl_cert, pg.ssl_key)
            if pg.ssl_crl and Path(pg.ssl_crl).exists():
                context.verify_flags |= ssl.VERIFY_CRL_CHECK_LEAF
                context.load_verify_locations(cafile=pg.ssl_crl)
            return context
        except Exception as exc:
            raise ValueError(f"PostgreSQL SSL configuration error: {exc}") from exc

    def pg_connection_kwargs(self) -> dict[str, Any]:
        pg = self.storage.postgres
        kwargs: dict[str, Any] = {
            "host": pg.host,
            "port": pg.port,
            "user": pg.user,
            "password": pg.password,
            "database": pg.database,
        }
        if (ssl_value := self._pg_ssl_value()) is not None:
            kwargs["ssl"] = ssl_value
        return kwargs

    @staticmethod
    def _env_value(value: str | int | float | bool | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value).strip() or None

    def domain_pool_server_settings(self) -> dict[str, str]:
        pg, vector = self.storage.postgres, self.storage.lightrag
        settings = {"hnsw.ef_search": str(vector.hnsw_ef_search)}
        for key, value in pg.session_settings.items():
            if (rendered := self._env_value(value)) is not None:
                settings[str(key)] = rendered
        return settings

    def lightrag_pool_server_settings(self) -> dict[str, str]:
        settings = self.domain_pool_server_settings()
        if self.is_reader:
            settings["default_transaction_read_only"] = "on"
        return settings

    def postgres_server_settings_env_value(self) -> str:
        return urlencode(self.lightrag_pool_server_settings())

    def _lightrag_sidecar_env_map(self) -> dict[str, str]:
        sidecars = self.corpus.sidecars
        objects: list[VLMSidecarSettings | MinerUSidecarSettings | DoclingSidecarSettings] = [
            sidecars.vlm
        ]
        objects.append(sidecars.mineru if sidecars.mineru is not None else sidecars.docling)  # type: ignore[arg-type]
        raw = {env: getattr(obj, field) for obj in objects for field, env in obj._ENV_MAP.items()}
        return {
            key: text for key, value in raw.items() if (text := self._env_value(value)) is not None
        }

    def apply_lightrag_sidecar_env(self) -> None:
        # Resolved bindings win. Optional fields that are absent deliberately do
        # not erase inherited LightRAG behavior.
        os.environ.update(self._lightrag_sidecar_env_map())

    def apply_lightrag_backend_env(self, *, force: bool = False) -> None:
        pg, vector = self.storage.postgres, self.storage.lightrag
        active = self.pg_connection_kwargs()
        # DlightRAG multiplexes WorkspaceRag instances in one process. Disable
        # LightRAG's process-global PostgreSQL workspace override so each
        # storage instance keeps the workspace supplied by WorkspaceRag.
        os.environ["POSTGRES_WORKSPACE"] = ""
        values: dict[str, Any] = {
            "POSTGRES_HOST": active["host"],
            "POSTGRES_PORT": active["port"],
            "POSTGRES_USER": active["user"],
            "POSTGRES_PASSWORD": active["password"],
            "POSTGRES_DATABASE": active["database"],
            "POSTGRES_VECTOR_INDEX_TYPE": vector.vector_index_type,
            "POSTGRES_HNSW_M": vector.hnsw_m,
            "POSTGRES_HNSW_EF": vector.hnsw_ef_construction,
            "POSTGRES_MAX_CONNECTIONS": pg.lightrag_pool_max_size,
            "POSTGRES_CONNECTION_RETRIES": pg.connection_retries,
            "POSTGRES_CONNECTION_RETRY_BACKOFF": pg.connection_retry_backoff,
            "POSTGRES_CONNECTION_RETRY_BACKOFF_MAX": pg.connection_retry_backoff_max,
            "POSTGRES_POOL_CLOSE_TIMEOUT": pg.pool_close_timeout,
        }
        if pg.statement_cache_size is not None:
            values["POSTGRES_STATEMENT_CACHE_SIZE"] = pg.statement_cache_size
        for key, value in {
            "POSTGRES_SSL_MODE": pg.ssl_mode,
            "POSTGRES_SSL_CERT": pg.ssl_cert,
            "POSTGRES_SSL_KEY": pg.ssl_key,
            "POSTGRES_SSL_ROOT_CERT": pg.ssl_root_cert,
            "POSTGRES_SSL_CRL": pg.ssl_crl,
            "MILVUS_URI": vector.milvus_uri,
            "MILVUS_TOKEN": vector.milvus_token,
            "MILVUS_DB_NAME": vector.milvus_db_name,
        }.items():
            if (rendered := self._env_value(value)) is not None:
                values[key] = rendered
        for key, value in values.items():
            if force or key not in os.environ:
                os.environ[key] = str(value)
        if force or "POSTGRES_SERVER_SETTINGS" not in os.environ:
            os.environ["POSTGRES_SERVER_SETTINGS"] = self.postgres_server_settings_env_value()

    def apply_lightrag_runtime_env(self, *, force: bool = False) -> None:
        if force or "LIGHTRAG_PARSER" not in os.environ:
            os.environ["LIGHTRAG_PARSER"] = self.parser_rules
        if force or "INPUT_DIR" not in os.environ:
            os.environ["INPUT_DIR"] = str(self.input_dir_path)
