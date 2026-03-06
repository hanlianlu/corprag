# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""FastAPI REST server for bulk ingestion and queries.

Entry point: dlightrag-api
Primary interface for offline/batch data ingestion operations.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from dlightrag.config import DlightragConfig, get_config
from dlightrag.core.servicemanager import RAGServiceManager, RAGServiceUnavailableError

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    try:
        _app.state.manager = await RAGServiceManager.create()
    except Exception:
        logger.exception("Failed to initialize RAG service manager")
        raise
    yield
    await _app.state.manager.close()


app = FastAPI(
    title="dlightrag",
    description="DlightRAG - Dual-mode (Caption based & Unified representation based) multi-modal RAG service",
    version=__import__("dlightrag").__version__,
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════════
# Auth middleware
# ═══════════════════════════════════════════════════════════════════


def _get_config() -> DlightragConfig:
    return get_config()


async def _verify_auth(request: Request, config: DlightragConfig = Depends(_get_config)) -> None:
    """Verify bearer token if DLIGHTRAG_API_AUTH_TOKEN is set."""
    token = config.api_auth_token
    if not token:
        return  # No auth required

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    provided_token = auth_header[7:]
    if provided_token != token:
        raise HTTPException(status_code=403, detail="Invalid token")


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
    workspace: str | None = None


class RetrieveRequest(BaseModel):
    query: str
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "mix"
    top_k: int | None = None
    chunk_top_k: int | None = None
    workspaces: list[str] | None = None


class AnswerRequest(BaseModel):
    query: str
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "mix"
    stream: bool = False
    top_k: int | None = None
    chunk_top_k: int | None = None
    conversation_history: list[dict[str, str]] | None = None
    workspaces: list[str] | None = None


class DeleteRequest(BaseModel):
    file_paths: list[str] | None = None
    filenames: list[str] | None = None
    delete_source: bool = True
    workspace: str | None = None


# ═══════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════


def _get_manager(request: Request) -> RAGServiceManager:
    return request.app.state.manager


@app.post("/ingest", dependencies=[Depends(_verify_auth)])
async def ingest(body: IngestRequest, request: Request) -> dict[str, Any]:
    """Bulk document ingestion."""
    manager = _get_manager(request)
    ws = body.workspace or get_config().workspace

    kwargs: dict[str, Any] = {}
    if body.replace is not None:
        kwargs["replace"] = body.replace

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

    result = await manager.aingest(ws, source_type=body.source_type, **kwargs)
    return result


@app.post("/retrieve", dependencies=[Depends(_verify_auth)])
async def retrieve(body: RetrieveRequest, request: Request) -> dict[str, Any]:
    """Retrieve contexts and sources without LLM answer generation."""
    manager = _get_manager(request)
    result = await manager.aretrieve(
        body.query,
        workspaces=body.workspaces,
        mode=body.mode,
        top_k=body.top_k,
        chunk_top_k=body.chunk_top_k,
    )
    return {
        "answer": result.answer,
        "contexts": result.contexts,
        "raw": result.raw,
    }


@app.post("/answer", dependencies=[Depends(_verify_auth)], response_model=None)
async def answer(body: AnswerRequest, request: Request):
    """RAG query with LLM-generated answer. Set stream=true for SSE."""
    import json

    manager = _get_manager(request)
    kwargs: dict[str, Any] = {}
    if body.conversation_history:
        kwargs["conversation_history"] = body.conversation_history

    if not body.stream:
        result = await manager.aanswer(
            body.query,
            workspaces=body.workspaces,
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

    # Streaming mode
    contexts, raw, token_iter = await manager.aanswer_stream(
        body.query,
        workspaces=body.workspaces,
        mode=body.mode,
        top_k=body.top_k,
        chunk_top_k=body.chunk_top_k,
        **kwargs,
    )

    async def event_generator() -> AsyncIterator[str]:
        yield f"data: {json.dumps({'type': 'context', 'data': contexts, 'raw': raw}, ensure_ascii=False)}\n\n"
        try:
            async for chunk in token_iter:
                yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as exc:
            logger.exception("Error during SSE streaming")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/files", dependencies=[Depends(_verify_auth)])
async def list_files(
    request: Request, workspace: str | None = Query(default=None)
) -> dict[str, Any]:
    """List all ingested documents."""
    manager = _get_manager(request)
    ws = workspace or get_config().workspace
    try:
        files = await manager.list_ingested_files(ws)
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=400,
            detail="File listing is not supported in unified RAG mode",
        ) from exc
    return {"files": files, "count": len(files), "workspace": ws}


@app.delete("/files", dependencies=[Depends(_verify_auth)])
async def delete_files(body: DeleteRequest, request: Request) -> dict[str, Any]:
    """Delete documents from knowledge base."""
    manager = _get_manager(request)
    ws = body.workspace or get_config().workspace
    try:
        results = await manager.delete_files(
            ws,
            file_paths=body.file_paths,
            filenames=body.filenames,
            delete_source=body.delete_source,
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=400,
            detail="File deletion is not supported in unified RAG mode",
        ) from exc
    return {"results": results, "workspace": ws}


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Health check including RAG service status."""
    config = get_config()
    manager = _get_manager(request)

    is_degraded = manager.is_degraded()
    warnings = manager.get_warnings()

    status: dict[str, Any] = {
        "status": "degraded" if is_degraded else "healthy",
        "rag_initialized": manager.is_ready(),
        "rag_mode": config.rag_mode,
        "crafted_by": "hllyu",
        "maintained_by": "HanlianLyu",
        "storage": {
            "vector": config.vector_storage,
            "graph": config.graph_storage,
            "kv": config.kv_storage,
        },
    }
    if warnings:
        status["warnings"] = warnings

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


@app.get("/workspaces", dependencies=[Depends(_verify_auth)])
async def workspaces(request: Request) -> dict[str, Any]:
    """List all available workspaces."""
    manager = _get_manager(request)
    ws_list = await manager.list_workspaces()
    return {"workspaces": ws_list}


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
    """Entry point for dlightrag-api."""
    import argparse

    import uvicorn
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="dlightrag REST API server")
    parser.add_argument("--env-file", help="Path to .env configuration file")
    args = parser.parse_args()

    if args.env_file:
        load_dotenv(args.env_file, override=True)

    config = get_config()
    uvicorn.run(
        "dlightrag.api.server:app",
        host=config.api_host,
        port=config.api_port,
        log_level=config.log_level,
    )


__all__ = ["app", "main"]
