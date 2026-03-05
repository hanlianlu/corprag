# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""DlightRAG - Dual-mode (Caption based & Unified representation based) multi-modal RAG service.

Built on RAGAnything and LightRAG with PostgreSQL as the default unified backend.
Exposable as both a REST API (bulk ingestion) and MCP server (agent integration).
"""

try:
    from importlib.metadata import version as _version

    __version__ = _version("dlightrag")
except Exception:
    __version__ = "0.0.0"
__maintainer__ = "HanlianLyu"
__credits__ = ["hllyu"]

# Fix macOS SSL: use homebrew CA bundle (includes corporate CAs from System Keychain)
import os as _os
import platform as _platform

if _platform.system() == "Darwin" and "SSL_CERT_FILE" not in _os.environ:
    _homebrew_ca = "/opt/homebrew/etc/ca-certificates/cert.pem"
    if _os.path.exists(_homebrew_ca):
        _os.environ["SSL_CERT_FILE"] = _homebrew_ca
        _os.environ["REQUESTS_CA_BUNDLE"] = _homebrew_ca

from dlightrag.config import DlightragConfig

__all__ = [
    "DlightragConfig",
    "__version__",
]


def _lazy_imports():
    """Lazy imports for heavy modules — only loaded when accessed."""
    from dlightrag.core.retrieval.engine import RetrievalResult
    from dlightrag.core.service import RAGService

    return RAGService, RetrievalResult


# Re-export for convenience (lazy to avoid heavy import on package load)
def __getattr__(name: str):
    if name in ("RAGService", "RetrievalResult"):
        RAGService, RetrievalResult = _lazy_imports()
        if name == "RAGService":
            return RAGService
        return RetrievalResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
