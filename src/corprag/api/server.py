# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""FastAPI REST server for bulk ingestion and queries.

Entry point: corprag-api
Primary interface for offline/batch data ingestion operations.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from corprag.config import CorpragConfig, get_config
from corprag.pool import (
    RAGServiceUnavailableError,
    close_shared_rag_service,
    get_shared_rag_service,
    is_rag_service_initialized,
)
from corprag.service import RAGService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield
    await close_shared_rag_service()


app = FastAPI(
    title="corprag",
    description="Corporate RAG - multimodal document ingestion & retrieval service",
    version="0.1.0",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════════
# Auth middleware
# ═══════════════════════════════════════════════════════════════════


def _get_config() -> CorpragConfig:
    return get_config()


async def _verify_auth(request: Request, config: CorpragConfig = Depends(_get_config)) -> None:
    """Verify bearer token if CORPRAG_API_AUTH_TOKEN is set."""
    token = config.api_auth_token
    if not token:
        return  # No auth required

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    provided_token = auth_header[7:]
    if provided_token != token:
        raise HTTPException(status_code=403, detail="Invalid token")


async def _get_rag_service() -> RAGService:
    """Get the shared RAG service."""
    try:
        return await get_shared_rag_service()
    except RAGServiceUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e.detail)) from e


# ═══════════════════════════════════════════════════════════════════
# Request/Response Models
# ═══════════════════════════════════════════════════════════════════


class IngestRequest(BaseModel):
    source_type: Literal["local", "azure_blob", "snowflake"]
    path: str | None = None
    container_name: str | None = None
    blob_path: str | None = None
    prefix: str | None = None
    query: str | None = None
    table: str | None = None
    replace: bool | None = None
    sync_hashes: bool = False


class RetrieveRequest(BaseModel):
    query: str
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "mix"
    top_k: int | None = None
    chunk_top_k: int | None = None


class AnswerRequest(BaseModel):
    query: str
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "mix"
    top_k: int | None = None
    chunk_top_k: int | None = None
    conversation_history: list[dict[str, str]] | None = None


class DeleteRequest(BaseModel):
    file_paths: list[str] | None = None
    filenames: list[str] | None = None
    delete_source: bool = True


# ═══════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════


@app.post("/ingest", dependencies=[Depends(_verify_auth)])
async def ingest(body: IngestRequest) -> dict[str, Any]:
    """Bulk document ingestion."""
    service = await _get_rag_service()

    kwargs: dict[str, Any] = {}
    if body.replace is not None:
        kwargs["replace"] = body.replace
    kwargs["sync_hashes"] = body.sync_hashes

    if body.source_type == "local":
        if not body.path:
            raise HTTPException(status_code=400, detail="'path' is required for local ingestion")
        kwargs["path"] = body.path

    elif body.source_type == "azure_blob":
        if not body.container_name:
            raise HTTPException(
                status_code=400, detail="'container_name' is required for azure_blob"
            )
        if body.blob_path and body.prefix is not None:
            raise HTTPException(
                status_code=400, detail="'blob_path' and 'prefix' are mutually exclusive"
            )
        kwargs["container_name"] = body.container_name
        if body.blob_path:
            kwargs["blob_path"] = body.blob_path
        if body.prefix is not None:
            kwargs["prefix"] = body.prefix

    elif body.source_type == "snowflake":
        if not body.query:
            raise HTTPException(status_code=400, detail="'query' is required for snowflake")
        kwargs["query"] = body.query
        if body.table:
            kwargs["table"] = body.table

    result = await service.aingest(source_type=body.source_type, **kwargs)
    return result


@app.post("/retrieve", dependencies=[Depends(_verify_auth)])
async def retrieve(body: RetrieveRequest) -> dict[str, Any]:
    """Retrieve contexts and sources without LLM answer generation."""
    service = await _get_rag_service()

    result = await service.aretrieve(
        query=body.query,
        mode=body.mode,
        top_k=body.top_k,
        chunk_top_k=body.chunk_top_k,
    )

    return {
        "answer": result.answer,
        "contexts": result.contexts,
        "raw": result.raw,
    }


@app.post("/answer", dependencies=[Depends(_verify_auth)])
async def answer(body: AnswerRequest) -> dict[str, Any]:
    """RAG query with LLM-generated answer and structured results."""
    service = await _get_rag_service()

    kwargs: dict[str, Any] = {}
    if body.conversation_history:
        kwargs["conversation_history"] = body.conversation_history

    result = await service.aanswer(
        query=body.query,
        mode=body.mode,
        top_k=body.top_k,
        chunk_top_k=body.chunk_top_k,
        **kwargs,
    )

    return {
        "answer": result.answer,
        "contexts": result.contexts,
        "raw": result.raw,
    }


@app.get("/files", dependencies=[Depends(_verify_auth)])
async def list_files() -> dict[str, Any]:
    """List all ingested documents."""
    service = await _get_rag_service()
    files = service.list_ingested_files()
    return {"files": files, "count": len(files)}


@app.delete("/files", dependencies=[Depends(_verify_auth)])
async def delete_files(body: DeleteRequest) -> dict[str, Any]:
    """Delete documents from knowledge base."""
    service = await _get_rag_service()
    results = await service.adelete_files(
        file_paths=body.file_paths,
        filenames=body.filenames,
        delete_source=body.delete_source,
    )
    return {"results": results}


@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check including RAG service status."""
    config = get_config()
    status: dict[str, Any] = {
        "status": "healthy",
        "rag_initialized": is_rag_service_initialized(),
        "crafted_by": "hllyu",
        "maintained_by": "HanlianLyu",
        "storage": {
            "vector": config.vector_storage,
            "graph": config.graph_storage,
            "kv": config.kv_storage,
        },
    }

    # Check PostgreSQL connectivity if using PG backends
    if config.kv_storage.startswith("PG"):
        try:
            import asyncpg

            conn = await asyncpg.connect(
                host=config.postgres_host,
                port=config.postgres_port,
                user=config.postgres_user,
                password=config.postgres_password,
                database=config.postgres_database,
            )
            await conn.fetchval("SELECT 1")
            await conn.close()
            status["postgres"] = "connected"
        except Exception as e:
            status["postgres"] = f"error: {e}"
            status["status"] = "degraded"

    return status


@app.exception_handler(RAGServiceUnavailableError)
async def rag_unavailable_handler(
    request: Request,  # noqa: ARG001
    exc: RAGServiceUnavailableError,
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": exc.detail},
    )


def main() -> None:
    """Entry point for corprag-api."""
    import uvicorn

    config = get_config()
    uvicorn.run(
        "corprag.api.server:app",
        host=config.api_host,
        port=config.api_port,
        log_level="info",
    )


__all__ = ["app", "main"]
