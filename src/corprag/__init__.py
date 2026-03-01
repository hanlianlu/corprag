# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""corprag - Corporate RAG: multimodal document ingestion & retrieval service.

Built on RAGAnything and LightRAG with PostgreSQL as the default unified backend.
Exposable as both a REST API (bulk ingestion) and MCP server (agent integration).
"""

__version__ = "0.1.0"
__maintainer__ = "HanlianLyu"
__credits__ = ["hllyu"]

from corprag.config import CorpragConfig

__all__ = [
    "CorpragConfig",
    "__version__",
]


def _lazy_imports():
    """Lazy imports for heavy modules — only loaded when accessed."""
    from corprag.retrieval.engine import RetrievalResult
    from corprag.service import RAGService

    return RAGService, RetrievalResult


# Re-export for convenience (lazy to avoid heavy import on package load)
def __getattr__(name: str):
    if name in ("RAGService", "RetrievalResult"):
        RAGService, RetrievalResult = _lazy_imports()
        if name == "RAGService":
            return RAGService
        return RetrievalResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
