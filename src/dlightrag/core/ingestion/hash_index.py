# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Hash-based content deduplication for RAG ingestion.

Provides content-addressable storage tracking via SHA256 hashes.
Supports PostgreSQL (PGHashIndex), Redis (RedisHashIndex),
MongoDB (MongoHashIndex), and JSON file (HashIndex) backends.

Each distributed backend reuses LightRAG's shared connection manager.
HashIndex (JSON) is the fallback for local/single-process deployments.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class HashIndexProtocol(Protocol):
    """Common interface for all hash index backends.

    Implemented by HashIndex (JSON), PGHashIndex, RedisHashIndex, MongoHashIndex.
    """

    async def check_exists(self, content_hash: str) -> tuple[bool, str | None]: ...
    async def register(self, content_hash: str, doc_id: str, file_path: str) -> None: ...
    async def remove(self, content_hash: str) -> bool: ...
    async def should_skip_file(
        self, file_path: Path, replace: bool
    ) -> tuple[bool, str | None, str | None]: ...
    async def clear(self) -> None: ...
    async def list_all(self) -> list[dict[str, Any]]: ...
    def invalidate(self) -> None: ...
    def find_by_name(self, filename: str) -> tuple[str | None, str | None, str | None]: ...
    def find_by_path(self, file_path: str) -> tuple[str | None, str | None, str | None]: ...

    @staticmethod
    def generate_doc_id_from_path(file_path: Path) -> str: ...


def derive_source_type(file_path: str) -> str:
    """Derive source type from a file path or URI.

    Returns "azure_blobs", "snowflake", "local", or "unknown".
    Handles both new-style URIs (azure://...) and legacy sources/ paths.
    """
    if not file_path:
        return "unknown"
    if file_path.startswith("azure://"):
        return "azure_blobs"
    if file_path.startswith("snowflake://"):
        return "snowflake"
    # Legacy: /abs/path/sources/{source_type}/file.pdf
    parts = Path(file_path).parts
    idx = next((i for i, p in enumerate(parts) if p == "sources"), -1)
    if idx >= 0 and len(parts) > idx + 1:
        return parts[idx + 1]
    # Default for local absolute/relative paths
    if file_path.startswith("/") or file_path.startswith(".") or "://" not in file_path:
        return "local"
    return "unknown"


# --- Content-aware hash extensions ---
_PDF_EXTENSIONS = {".pdf"}
_OFFICE_EXTENSIONS = {".docx", ".pptx", ".xlsx"}

_OFFICE_CONTENT_PREFIXES: dict[str, tuple[str, ...]] = {
    ".docx": ("word/document.xml", "word/media/"),
    ".pptx": ("ppt/slides/", "ppt/media/"),
    ".xlsx": ("xl/worksheets/", "xl/sharedStrings.xml"),
}


def _hash_pdf_content(file_path: Path) -> str | None:
    """Extract text from PDF pages via pypdfium2 and hash it.

    Returns sha256 hash string, or None to signal fallback to byte hash
    (when no text is extractable or pypdfium2 is unavailable).
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        logger.debug("pypdfium2 not available, falling back to byte hash for %s", file_path.name)
        return None

    try:
        doc = pdfium.PdfDocument(str(file_path))
        sha256_hash = hashlib.sha256()
        has_text = False

        for i in range(len(doc)):
            page = doc[i]
            textpage = page.get_textpage()
            text = textpage.get_text_bounded()
            if text.strip():
                has_text = True
            sha256_hash.update(text.encode("utf-8"))

        if not has_text:
            logger.debug("No text extracted from %s, falling back to byte hash", file_path.name)
            return None

        return f"sha256:{sha256_hash.hexdigest()}"
    except Exception as exc:
        logger.debug("PDF content extraction failed for %s: %s", file_path.name, exc)
        return None


def _hash_office_content(file_path: Path) -> str | None:
    """Hash only content entries from Office ZIP archive, skipping metadata.

    Returns sha256 hash string, or None to signal fallback to byte hash.
    """
    import zipfile

    ext = file_path.suffix.lower()
    prefixes = _OFFICE_CONTENT_PREFIXES.get(ext)
    if not prefixes:
        return None

    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            sha256_hash = hashlib.sha256()
            has_content = False

            for name in sorted(zf.namelist()):
                if any(name.startswith(p) or name == p for p in prefixes):
                    sha256_hash.update(name.encode("utf-8"))
                    sha256_hash.update(zf.read(name))
                    has_content = True

            if not has_content:
                logger.debug(
                    "No content entries found in %s, falling back to byte hash", file_path.name
                )
                return None

            return f"sha256:{sha256_hash.hexdigest()}"
    except Exception as exc:
        logger.debug("Office content extraction failed for %s: %s", file_path.name, exc)
        return None


def compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA256 hash of file content.

    For PDF and Office formats, hashes only the actual content (text, media)
    and skips volatile metadata (timestamps, IDs) that change on every
    download/export. Falls back to raw byte hash if content extraction fails.

    Returns:
        Hash string in format "sha256:<hex_digest>"
    """
    suffix = file_path.suffix.lower()

    # PDF: extract text via pypdfium2
    if suffix in _PDF_EXTENSIONS:
        result = _hash_pdf_content(file_path)
        if result is not None:
            return result

    # Office: hash ZIP content entries only
    if suffix in _OFFICE_EXTENSIONS:
        result = _hash_office_content(file_path)
        if result is not None:
            return result

    # Fallback: raw byte hash
    return _hash_file_bytes(file_path, chunk_size)


def _hash_file_bytes(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA256 hash of raw file bytes (original behavior)."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return f"sha256:{sha256_hash.hexdigest()}"


class HashIndex:
    """JSON file-based content hash index for deduplication.

    Used as fallback when PostgreSQL is not the storage backend.
    For PostgreSQL, use PGHashIndex instead.
    """

    def __init__(self, working_dir: Path, workspace: str = "default") -> None:
        self._workspace = workspace
        if workspace:
            self._working_dir = working_dir / workspace
        else:
            self._working_dir = working_dir
        self._working_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict[str, Any]] | None = None

    def _get_index_path(self) -> Path:
        return self._working_dir / "file_content_hashes.json"

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._cache is not None:
            return self._cache

        index_path = self._get_index_path()
        if not index_path.exists():
            self._cache = {}
            return self._cache

        try:
            loaded: dict[str, dict[str, Any]] = json.loads(index_path.read_text())
            self._cache = loaded
        except Exception as exc:
            logger.warning(f"Failed to load hash index: {exc}")
            self._cache = {}

        return self._cache or {}

    def _save(self) -> None:
        if self._cache is None:
            return
        index_path = self._get_index_path()
        temp_path = index_path.with_suffix(".tmp")
        try:
            temp_path.write_text(json.dumps(self._cache, indent=2))
            temp_path.replace(index_path)
        except Exception as exc:
            logger.warning(f"Failed to save hash index: {exc}")
            if temp_path.exists():
                temp_path.unlink()

    def invalidate(self) -> None:
        self._cache = None

    async def clear(self) -> None:
        """Remove all hash entries (used by reset)."""
        self._cache = {}
        index_path = self._get_index_path()
        if index_path.exists():
            index_path.unlink()
        logger.info("HashIndex cleared")

    async def check_exists(self, content_hash: str) -> tuple[bool, str | None]:
        index = self._load()
        entry = index.get(content_hash)
        if entry:
            return (True, entry.get("doc_id"))
        return (False, None)

    def lookup(self, content_hash: str) -> dict[str, Any] | None:
        """Return the full entry dict for a content hash, or None."""
        index = self._load()
        return index.get(content_hash)

    async def register(self, content_hash: str, doc_id: str, file_path: str) -> None:
        index = self._load()
        index[content_hash] = {
            "doc_id": doc_id,
            "file_path": file_path,
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._save()

    async def remove(self, content_hash: str) -> bool:
        index = self._load()
        if content_hash in index:
            del index[content_hash]
            self._save()
            return True
        return False

    async def should_skip_file(
        self,
        file_path: Path,
        replace: bool,
    ) -> tuple[bool, str | None, str | None]:
        content_hash = await asyncio.to_thread(compute_file_hash, file_path)
        if replace:
            return (False, content_hash, None)
        exists, doc_id = await self.check_exists(content_hash)
        if exists:
            logger.info(f"Skipping duplicate: {file_path.name} (hash matches doc_id={doc_id})")
            return (True, content_hash, f"Duplicate of {doc_id}")
        return (False, content_hash, None)

    @staticmethod
    def generate_doc_id_from_path(file_path: Path) -> str:
        return file_path.stem

    def find_by_path(self, file_path: str) -> tuple[str | None, str | None, str | None]:
        index = self._load()
        for h, info in index.items():
            if info.get("file_path") == file_path:
                return (info.get("doc_id"), h, info.get("file_path"))
        return (None, None, None)

    def find_by_name(self, filename: str) -> tuple[str | None, str | None, str | None]:
        index = self._load()
        for h, info in index.items():
            stored_path = info.get("file_path", "")
            if Path(stored_path).name == filename:
                return (info.get("doc_id"), h, stored_path)
        return (None, None, None)

    async def list_all(self) -> list[dict[str, Any]]:
        self.invalidate()
        index = self._load()
        results = []
        for content_hash, info in index.items():
            file_path = info.get("file_path", "")
            source_type = derive_source_type(file_path)
            results.append(
                {
                    "file_path": file_path,
                    "doc_id": info.get("doc_id", ""),
                    "source_type": source_type,
                    "file_name": Path(file_path).name,
                    "content_hash": content_hash,
                    "created_at": info.get("created_at", ""),
                }
            )
        return results


class PGHashIndex:
    """PostgreSQL-backed content hash index for deduplication.

    Multi-worker safe — uses PostgreSQL transactions instead of file locks.
    Table: dlightrag_file_hashes (created by init.sql or auto-created).
    Uses LightRAG's shared ClientManager for connection pooling.
    """

    TABLE = "dlightrag_file_hashes"

    def __init__(
        self,
        workspace: str = "default",
        pool: Any = None,
    ) -> None:
        self._pool = pool  # asyncpg.Pool — set via initialize() or passed directly
        self._workspace = workspace

    def _get_pool(self) -> Any:
        if self._pool is None:
            raise RuntimeError("PGHashIndex not initialized — call initialize() first")
        return self._pool

    async def initialize(self) -> None:
        """Get shared pool from LightRAG ClientManager and create table."""
        if self._pool is None:
            from lightrag.kg.postgres_impl import ClientManager

            db = await ClientManager.get_client()
            self._pool = db.pool

        async with self._get_pool().acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    content_hash TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    workspace TEXT NOT NULL DEFAULT 'default',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_file_hashes_doc_id
                ON {self.TABLE}(doc_id)
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_file_hashes_workspace
                ON {self.TABLE}(workspace)
            """)

    async def clear(self) -> None:
        """Remove all hash entries for this workspace (used by reset)."""
        async with self._get_pool().acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {self.TABLE} WHERE workspace = $1",
                self._workspace,
            )
            logger.info("PGHashIndex cleared for workspace %s: %s", self._workspace, result)

    async def check_exists(self, content_hash: str) -> tuple[bool, str | None]:
        async with self._get_pool().acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT doc_id FROM {self.TABLE} WHERE content_hash = $1 AND workspace = $2",
                content_hash,
                self._workspace,
            )
            return (True, row["doc_id"]) if row else (False, None)

    async def register(self, content_hash: str, doc_id: str, file_path: str) -> None:
        async with self._get_pool().acquire() as conn:
            await conn.execute(
                f"""INSERT INTO {self.TABLE} (content_hash, doc_id, file_path, workspace)
                   VALUES ($1, $2, $3, $4)
                   ON CONFLICT (content_hash) DO UPDATE SET
                     doc_id = EXCLUDED.doc_id,
                     file_path = EXCLUDED.file_path""",
                content_hash,
                doc_id,
                file_path,
                self._workspace,
            )

    async def remove(self, content_hash: str) -> bool:
        async with self._get_pool().acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM {self.TABLE} WHERE content_hash = $1 AND workspace = $2",
                content_hash,
                self._workspace,
            )
            return result != "DELETE 0"

    async def should_skip_file(
        self,
        file_path: Path,
        replace: bool,
    ) -> tuple[bool, str | None, str | None]:
        content_hash = await asyncio.to_thread(compute_file_hash, file_path)
        if replace:
            return (False, content_hash, None)
        exists, doc_id = await self.check_exists(content_hash)
        if exists:
            logger.info(f"Skipping duplicate: {file_path.name} (hash matches doc_id={doc_id})")
            return (True, content_hash, f"Duplicate of {doc_id}")
        return (False, content_hash, None)

    @staticmethod
    def generate_doc_id_from_path(file_path: Path) -> str:
        return file_path.stem

    def find_by_name(self, filename: str) -> tuple[str | None, str | None, str | None]:
        """Synchronous find — delegates to async in current event loop.

        For deletion context compatibility with HashIndex interface.
        """
        # Fallback: run async version synchronously if no event loop
        import asyncio as _asyncio

        try:
            loop = _asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Can't await from sync context inside running loop — use blocking query
            # This is acceptable for deletion (rare operation)

            return self._sync_find_by_name(filename)
        return _asyncio.run(self._async_find_by_name(filename))

    def _sync_find_by_name(self, filename: str) -> tuple[str | None, str | None, str | None]:
        """Blocking find for use from sync deletion context."""
        # Return None — deletion will use other strategies (doc_status JSON fallback)
        return (None, None, None)

    async def _async_find_by_name(self, filename: str) -> tuple[str | None, str | None, str | None]:
        async with self._get_pool().acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT content_hash, doc_id, file_path FROM {self.TABLE} "
                f"WHERE file_path LIKE $1 AND workspace = $2 LIMIT 1",
                f"%/{filename}",
                self._workspace,
            )
            if row:
                return (row["doc_id"], row["content_hash"], row["file_path"])
            return (None, None, None)

    def find_by_path(self, file_path: str) -> tuple[str | None, str | None, str | None]:
        """Synchronous find by path — similar to find_by_name."""
        return self._sync_find_by_name(Path(file_path).name)

    async def list_all(self) -> list[dict[str, Any]]:
        async with self._get_pool().acquire() as conn:
            rows = await conn.fetch(
                f"SELECT content_hash, doc_id, file_path, created_at "
                f"FROM {self.TABLE} WHERE workspace = $1",
                self._workspace,
            )
            results = []
            for row in rows:
                file_path = row["file_path"]
                source_type = derive_source_type(file_path)
                results.append(
                    {
                        "file_path": file_path,
                        "doc_id": row["doc_id"],
                        "source_type": source_type,
                        "file_name": Path(file_path).name,
                        "content_hash": row["content_hash"],
                        "created_at": row["created_at"].isoformat() if row["created_at"] else "",
                    }
                )
            return results

    def invalidate(self) -> None:
        """No-op for PG backend (no cache to invalidate)."""


class RedisHashIndex:
    """Redis-backed content hash index for deduplication.

    Multi-worker safe — uses Redis Hash for atomic operations.
    Requires: redis package (installed with LightRAG RedisKVStorage).
    """

    KEY_PREFIX = "dlightrag:file_hashes"

    def __init__(self, workspace: str = "default") -> None:
        self._workspace = workspace
        self._redis: Any = None
        self._redis_url: str | None = None

    def _key(self) -> str:
        return f"{self.KEY_PREFIX}:{self._workspace}"

    def _get_redis(self) -> Any:
        if self._redis is None:
            raise RuntimeError("RedisHashIndex not initialized — call initialize() first")
        return self._redis

    async def initialize(self) -> None:
        """Connect to Redis using LightRAG's shared connection pool."""
        import os

        from lightrag.kg.redis_impl import RedisConnectionManager
        from redis.asyncio import Redis  # type: ignore[import-not-found]

        config_uri = "redis://localhost:6379"
        try:
            import configparser

            cfg = configparser.ConfigParser()
            cfg.read("config.ini", "utf-8")
            config_uri = cfg.get("redis", "uri", fallback=config_uri)
        except Exception:
            pass
        self._redis_url = os.environ.get("REDIS_URI", config_uri)

        pool = RedisConnectionManager.get_pool(self._redis_url)
        self._redis = Redis(connection_pool=pool)

    async def check_exists(self, content_hash: str) -> tuple[bool, str | None]:
        data = await self._get_redis().hget(self._key(), content_hash)
        if data:
            entry = json.loads(data)
            return (True, entry.get("doc_id"))
        return (False, None)

    async def register(self, content_hash: str, doc_id: str, file_path: str) -> None:
        entry = json.dumps(
            {
                "doc_id": doc_id,
                "file_path": file_path,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        await self._get_redis().hset(self._key(), content_hash, entry)

    async def remove(self, content_hash: str) -> bool:
        removed = await self._get_redis().hdel(self._key(), content_hash)
        return removed > 0

    async def clear(self) -> None:
        """Remove all hash entries for this workspace."""
        await self._get_redis().delete(self._key())
        logger.info("RedisHashIndex cleared for workspace %s", self._workspace)

    async def should_skip_file(
        self, file_path: Path, replace: bool
    ) -> tuple[bool, str | None, str | None]:
        content_hash = await asyncio.to_thread(compute_file_hash, file_path)
        if replace:
            return (False, content_hash, None)
        exists, doc_id = await self.check_exists(content_hash)
        if exists:
            logger.info(f"Skipping duplicate: {file_path.name} (hash matches doc_id={doc_id})")
            return (True, content_hash, f"Duplicate of {doc_id}")
        return (False, content_hash, None)

    @staticmethod
    def generate_doc_id_from_path(file_path: Path) -> str:
        return file_path.stem

    def invalidate(self) -> None:
        pass  # No local cache to invalidate

    def find_by_name(self, filename: str) -> tuple[str | None, str | None, str | None]:
        return (None, None, None)  # Sync context fallback

    def find_by_path(self, file_path: str) -> tuple[str | None, str | None, str | None]:
        return (None, None, None)  # Sync context fallback

    async def list_all(self) -> list[dict[str, Any]]:
        all_entries = await self._get_redis().hgetall(self._key())
        results = []
        for content_hash, data in all_entries.items():
            info = json.loads(data)
            fp = info.get("file_path", "")
            source_type = derive_source_type(fp)
            results.append(
                {
                    "file_path": fp,
                    "doc_id": info.get("doc_id", ""),
                    "source_type": source_type,
                    "file_name": Path(fp).name,
                    "content_hash": content_hash,
                    "created_at": info.get("created_at", ""),
                }
            )
        return results


class MongoHashIndex:
    """MongoDB-backed content hash index for deduplication.

    Multi-worker safe — uses MongoDB atomic operations.
    Requires: motor package (installed with LightRAG MongoKVStorage).
    """

    COLLECTION = "dlightrag_file_hashes"

    def __init__(self, workspace: str = "default") -> None:
        self._workspace = workspace
        self._collection: Any = None
        self._db: Any = None

    def _get_collection(self) -> Any:
        if self._collection is None:
            raise RuntimeError("MongoHashIndex not initialized — call initialize() first")
        return self._collection

    async def initialize(self) -> None:
        """Connect to MongoDB using LightRAG's shared client."""
        from lightrag.kg.mongo_impl import ClientManager

        self._db = await ClientManager.get_client()
        self._collection = self._db[self.COLLECTION]
        # Ensure index on workspace for efficient queries
        await self._get_collection().create_index("workspace")

    async def check_exists(self, content_hash: str) -> tuple[bool, str | None]:
        doc = await self._get_collection().find_one(
            {"_id": content_hash, "workspace": self._workspace}
        )
        if doc:
            return (True, doc.get("doc_id"))
        return (False, None)

    async def register(self, content_hash: str, doc_id: str, file_path: str) -> None:
        await self._get_collection().update_one(
            {"_id": content_hash},
            {
                "$set": {
                    "doc_id": doc_id,
                    "file_path": file_path,
                    "workspace": self._workspace,
                    "created_at": datetime.now(UTC).isoformat(),
                }
            },
            upsert=True,
        )

    async def remove(self, content_hash: str) -> bool:
        result = await self._get_collection().delete_one(
            {"_id": content_hash, "workspace": self._workspace}
        )
        return result.deleted_count > 0

    async def clear(self) -> None:
        """Remove all hash entries for this workspace."""
        await self._get_collection().delete_many({"workspace": self._workspace})
        logger.info("MongoHashIndex cleared for workspace %s", self._workspace)

    async def should_skip_file(
        self, file_path: Path, replace: bool
    ) -> tuple[bool, str | None, str | None]:
        content_hash = await asyncio.to_thread(compute_file_hash, file_path)
        if replace:
            return (False, content_hash, None)
        exists, doc_id = await self.check_exists(content_hash)
        if exists:
            logger.info(f"Skipping duplicate: {file_path.name} (hash matches doc_id={doc_id})")
            return (True, content_hash, f"Duplicate of {doc_id}")
        return (False, content_hash, None)

    @staticmethod
    def generate_doc_id_from_path(file_path: Path) -> str:
        return file_path.stem

    def invalidate(self) -> None:
        pass

    def find_by_name(self, filename: str) -> tuple[str | None, str | None, str | None]:
        return (None, None, None)

    def find_by_path(self, file_path: str) -> tuple[str | None, str | None, str | None]:
        return (None, None, None)

    async def list_all(self) -> list[dict[str, Any]]:
        cursor = self._get_collection().find({"workspace": self._workspace})
        results = []
        async for doc in cursor:
            fp = doc.get("file_path", "")
            source_type = derive_source_type(fp)
            results.append(
                {
                    "file_path": fp,
                    "doc_id": doc.get("doc_id", ""),
                    "source_type": source_type,
                    "file_name": Path(fp).name,
                    "content_hash": doc["_id"],
                    "created_at": doc.get("created_at", ""),
                }
            )
        return results


__all__ = [
    "HashIndexProtocol",
    "HashIndex",
    "PGHashIndex",
    "RedisHashIndex",
    "MongoHashIndex",
    "compute_file_hash",
    "derive_source_type",
]
