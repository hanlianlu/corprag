# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""MCP server for agent integration (stdio + streamable-http).

Entry point: dlightrag-mcp
Primarily used by DeerFlow and other MCP-compatible agents for
retrieve() + lightweight ingest().
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from dlightrag.config import DlightragConfig, get_config
from dlightrag.core.servicemanager import RAGServiceManager

logger = logging.getLogger(__name__)

server = Server(
    "dlightrag",
    version=__import__("dlightrag").__version__,
)


def _get_config() -> DlightragConfig:
    return get_config()


_manager: RAGServiceManager | None = None


async def _ensure_manager() -> RAGServiceManager:
    global _manager
    if _manager is None:
        _manager = await RAGServiceManager.create()
    return _manager


# ═══════════════════════════════════════════════════════════════════
# Tool definitions
# ═══════════════════════════════════════════════════════════════════


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="retrieve",
            description="Query the RAG knowledge base for relevant information.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["local", "global", "hybrid", "naive", "mix"],
                        "default": "mix",
                        "description": "Retrieval mode",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results to return",
                    },
                    "workspaces": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Workspace names to search. Omit to search default workspace.",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="ingest",
            description="Ingest document(s) into the RAG knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_type": {
                        "type": "string",
                        "enum": ["local", "azure_blob", "snowflake"],
                        "description": "Type of data source",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory path (for local source)",
                    },
                    "container_name": {
                        "type": "string",
                        "description": "Azure Blob container name",
                    },
                    "blob_path": {
                        "type": "string",
                        "description": "Specific blob path",
                    },
                    "prefix": {
                        "type": "string",
                        "description": "Blob prefix filter",
                    },
                    "replace": {
                        "type": "boolean",
                        "default": False,
                        "description": "Replace existing documents",
                    },
                    "workspace": {
                        "type": "string",
                        "description": "Target workspace. Omit for default workspace.",
                    },
                },
                "required": ["source_type"],
            },
        ),
        Tool(
            name="answer",
            description="Ask a question and get an LLM-generated answer backed by retrieved context from the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question to answer",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["local", "global", "hybrid", "naive", "mix"],
                        "default": "mix",
                        "description": "Retrieval mode",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results to retrieve",
                    },
                    "conversation_history": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": ["user", "assistant"]},
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                        "description": "Previous conversation turns for multi-turn context",
                    },
                    "workspaces": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Workspace names to search. Omit to search default workspace.",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_workspaces",
            description="List all available workspaces with ingested data.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_files",
            description="List all documents ingested in the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workspace": {
                        "type": "string",
                        "description": "Workspace to list files from. Omit for default workspace.",
                    },
                },
            },
        ),
        Tool(
            name="delete_files",
            description="Delete documents from the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filenames": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of filenames to delete",
                    },
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths to delete",
                    },
                    "workspace": {
                        "type": "string",
                        "description": "Workspace to delete from. Omit for default workspace.",
                    },
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle MCP tool calls."""
    try:
        if name == "retrieve":
            manager = await _ensure_manager()
            result = await manager.aretrieve(
                arguments["query"],
                workspaces=arguments.get("workspaces"),
                mode=arguments.get("mode", "mix"),
                top_k=arguments.get("top_k"),
            )
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "answer": result.answer,
                            "contexts": result.contexts,
                            "sources": result.raw.get("sources", []),
                        },
                        default=str,
                        indent=2,
                    ),
                )
            ]

        if name == "answer":
            manager = await _ensure_manager()
            kwargs: dict[str, Any] = {}
            if arguments.get("conversation_history"):
                kwargs["conversation_history"] = arguments["conversation_history"]
            result = await manager.aanswer(
                arguments["query"],
                workspaces=arguments.get("workspaces"),
                mode=arguments.get("mode", "mix"),
                top_k=arguments.get("top_k"),
                **kwargs,
            )
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "answer": result.answer,
                            "contexts": result.contexts,
                            "sources": result.raw.get("sources", []),
                        },
                        default=str,
                        indent=2,
                    ),
                )
            ]

        if name == "list_workspaces":
            manager = await _ensure_manager()
            ws_list = await manager.list_workspaces()
            return [TextContent(type="text", text=json.dumps({"workspaces": ws_list}, indent=2))]

        if name == "ingest":
            manager = await _ensure_manager()
            ws = arguments.get("workspace") or _get_config().workspace
            source_type = arguments["source_type"]
            kwargs: dict[str, Any] = {}
            if source_type == "local":
                kwargs["path"] = arguments.get("path", ".")
            elif source_type == "azure_blob":
                kwargs["container_name"] = arguments.get("container_name", "")
                if arguments.get("blob_path"):
                    kwargs["blob_path"] = arguments["blob_path"]
                if arguments.get("prefix") is not None:
                    kwargs["prefix"] = arguments["prefix"]
            if arguments.get("replace") is not None:
                kwargs["replace"] = arguments["replace"]
            result = await manager.aingest(ws, source_type=source_type, **kwargs)
            return [TextContent(type="text", text=json.dumps(result, default=str))]

        if name == "list_files":
            manager = await _ensure_manager()
            ws = arguments.get("workspace") or _get_config().workspace
            try:
                files = await manager.list_ingested_files(ws)
            except NotImplementedError:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": "File listing is not supported in unified RAG mode"},
                        ),
                    )
                ]
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"files": files, "count": len(files), "workspace": ws}, default=str
                    ),
                )
            ]

        if name == "delete_files":
            manager = await _ensure_manager()
            ws = arguments.get("workspace") or _get_config().workspace
            try:
                results = await manager.delete_files(
                    ws, filenames=arguments.get("filenames"), file_paths=arguments.get("file_paths")
                )
            except NotImplementedError:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": "File deletion is not supported in unified RAG mode"},
                        ),
                    )
                ]
            return [
                TextContent(
                    type="text", text=json.dumps({"results": results, "workspace": ws}, default=str)
                )
            ]

        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.exception(f"MCP tool '{name}' failed: {e}")
        return [TextContent(type="text", text=f"Error: {e}")]


# ═══════════════════════════════════════════════════════════════════
# Server startup
# ═══════════════════════════════════════════════════════════════════


async def run_stdio() -> None:
    """Run MCP server over stdio transport."""
    await _ensure_manager()
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        if _manager is not None:
            await _manager.close()


async def run_streamable_http(host: str, port: int) -> None:
    """Run MCP server over streamable-http transport."""
    await _ensure_manager()
    import uvicorn
    from mcp.server.streamable_http import StreamableHTTPServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Mount

    transport = StreamableHTTPServerTransport(mcp_session_id=None)

    starlette_app = Starlette(
        routes=[Mount("/mcp", app=transport.handle_request)],
    )

    config = uvicorn.Config(
        starlette_app,
        host=host,
        port=port,
        log_level="info",
    )
    uv_server = uvicorn.Server(config)

    try:
        async with transport.connect() as (read_stream, write_stream):
            server_task = asyncio.create_task(
                server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options(),
                )
            )
            await uv_server.serve()
            server_task.cancel()
    finally:
        if _manager is not None:
            await _manager.close()


def main() -> None:
    """Entry point for dlightrag-mcp."""
    import argparse

    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="dlightrag MCP server")
    parser.add_argument("--env-file", help="Path to .env configuration file")
    args = parser.parse_args()

    if args.env_file:
        load_dotenv(args.env_file, override=True)

    config = _get_config()

    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))

    if config.mcp_transport == "streamable-http":
        logger.info(f"Starting MCP server (streamable-http) on {config.mcp_host}:{config.mcp_port}")
        asyncio.run(run_streamable_http(config.mcp_host, config.mcp_port))
    else:
        logger.info("Starting MCP server (stdio)")
        asyncio.run(run_stdio())


__all__ = ["main", "server"]
