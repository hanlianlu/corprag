# Unified HashIndex Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify HashIndex implementations across all KV backends (PG, Redis, Mongo, JSON), reusing LightRAG's shared connection managers.

**Architecture:** Add RedisHashIndex and MongoHashIndex classes to hash_index.py, refactor PGHashIndex to use LightRAG's ClientManager instead of a separate pool, and update the factory in service.py to route based on kv_storage config.

**Tech Stack:** Python, asyncio, asyncpg (PG), redis.asyncio (Redis), motor (Mongo), LightRAG ClientManagers

---

### Task 1: Add RedisHashIndex

**Files:**
- Modify: `src/dlightrag/ingestion/hash_index.py`
- Modify: `tests/unit/test_hash_index.py`

**Step 1: Add RedisHashIndex class at the end of hash_index.py**

Append after the PGHashIndex class (currently ends around line 530):

```python
class RedisHashIndex:
    """Redis-backed content hash index for deduplication.

    Multi-worker safe — uses Redis Hash for atomic operations.
    Requires: redis package (installed with LightRAG RedisKVStorage).
    """

    KEY_PREFIX = "dlightrag:file_hashes"

    def __init__(self, workspace: str = "default", sources_dir: Path | None = None) -> None:
        self._workspace = workspace
        self._sources_dir = sources_dir
        self._redis: Any = None
        self._redis_url: str | None = None

    def _key(self) -> str:
        return f"{self.KEY_PREFIX}:{self._workspace}"

    async def initialize(self) -> None:
        """Connect to Redis using LightRAG's shared connection pool."""
        import os

        from redis.asyncio import Redis

        config_uri = "redis://localhost:6379"
        try:
            import configparser

            cfg = configparser.ConfigParser()
            cfg.read("config.ini", "utf-8")
            config_uri = cfg.get("redis", "uri", fallback=config_uri)
        except Exception:
            pass
        self._redis_url = os.environ.get("REDIS_URI", config_uri)

        from lightrag.kg.redis_impl import RedisConnectionManager

        pool = RedisConnectionManager.get_pool(self._redis_url)
        self._redis = Redis(connection_pool=pool)

    def check_exists(self, content_hash: str) -> tuple[bool, str | None]:
        # Sync wrapper — run async version if loop available
        import asyncio as _aio

        try:
            loop = _aio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # In sync deletion context, return not-found (matches PGHashIndex pattern)
            return (False, None)
        return _aio.run(self._async_check_exists(content_hash))

    async def _async_check_exists(self, content_hash: str) -> tuple[bool, str | None]:
        data = await self._redis.hget(self._key(), content_hash)
        if data:
            entry = json.loads(data)
            return (True, entry.get("doc_id"))
        return (False, None)

    async def register(self, content_hash: str, doc_id: str, file_path: str) -> None:
        entry = json.dumps({
            "doc_id": doc_id,
            "file_path": file_path,
            "created_at": datetime.now(UTC).isoformat(),
        })
        await self._redis.hset(self._key(), content_hash, entry)

    async def remove(self, content_hash: str) -> bool:
        removed = await self._redis.hdel(self._key(), content_hash)
        return removed > 0

    async def clear(self) -> None:
        """Remove all hash entries for this workspace."""
        await self._redis.delete(self._key())
        logger.info("RedisHashIndex cleared for workspace %s", self._workspace)

    async def should_skip_file(
        self, file_path: Path, replace: bool
    ) -> tuple[bool, str | None, str | None]:
        content_hash = await asyncio.to_thread(compute_file_hash, file_path)
        if replace:
            return (False, content_hash, None)
        exists, doc_id = await self._async_check_exists(content_hash)
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
        all_entries = await self._redis.hgetall(self._key())
        results = []
        for content_hash, data in all_entries.items():
            info = json.loads(data)
            fp = info.get("file_path", "")
            source_type = "unknown"
            path_parts = Path(fp).parts
            sources_idx = next((i for i, p in enumerate(path_parts) if p == "sources"), -1)
            if sources_idx >= 0 and len(path_parts) > sources_idx + 1:
                source_type = path_parts[sources_idx + 1]
            results.append({
                "file_path": fp,
                "doc_id": info.get("doc_id", ""),
                "source_type": source_type,
                "file_name": Path(fp).name,
                "content_hash": content_hash,
                "created_at": info.get("created_at", ""),
            })
        return results

    async def sync_existing(self) -> int:
        if not self._sources_dir or not self._sources_dir.exists():
            return 0
        existing = await self._redis.hkeys(self._key())
        existing_hashes = set(existing)
        # Build file_path lookup
        all_entries = await self._redis.hgetall(self._key())
        existing_paths = set()
        for data in all_entries.values():
            info = json.loads(data)
            existing_paths.add(info.get("file_path", ""))
        synced = 0
        for fp in self._sources_dir.rglob("*"):
            if fp.is_dir() or fp.name.startswith("."):
                continue
            if fp.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if str(fp) in existing_paths:
                continue
            try:
                h = await asyncio.to_thread(compute_file_hash, fp)
            except Exception as exc:
                logger.warning(f"Failed to hash {fp}: {exc}")
                continue
            if h in existing_hashes:
                continue
            doc_id = self.generate_doc_id_from_path(fp)
            await self.register(h, doc_id, str(fp))
            existing_hashes.add(h)
            existing_paths.add(str(fp))
            synced += 1
        if synced:
            logger.info(f"Synced {synced} hashes from sources directory")
        return synced
```

**Step 2: Add unit test with mocked Redis**

Add to `tests/unit/test_hash_index.py`:

```python
class TestRedisHashIndex:
    """Tests for RedisHashIndex using a mock Redis client."""

    @pytest.fixture
    def redis_index(self):
        from dlightrag.ingestion.hash_index import RedisHashIndex

        idx = RedisHashIndex(workspace="test", sources_dir=None)
        # Mock the Redis client with a dict-based fake
        idx._redis = FakeRedis()
        return idx

    async def test_register_and_check_exists(self, redis_index):
        await redis_index.register("sha256:aaa", "doc-001", "/path/a.pdf")
        exists, doc_id = await redis_index._async_check_exists("sha256:aaa")
        assert exists is True
        assert doc_id == "doc-001"

    async def test_check_exists_missing(self, redis_index):
        exists, doc_id = await redis_index._async_check_exists("sha256:missing")
        assert exists is False
        assert doc_id is None

    async def test_remove(self, redis_index):
        await redis_index.register("sha256:aaa", "doc-001", "/path/a.pdf")
        removed = await redis_index.remove("sha256:aaa")
        assert removed is True
        exists, _ = await redis_index._async_check_exists("sha256:aaa")
        assert exists is False

    async def test_clear(self, redis_index):
        await redis_index.register("sha256:aaa", "doc-001", "/a.pdf")
        await redis_index.register("sha256:bbb", "doc-002", "/b.pdf")
        await redis_index.clear()
        entries = await redis_index.list_all()
        assert len(entries) == 0

    async def test_list_all(self, redis_index):
        await redis_index.register("sha256:aaa", "doc-001", "/a.pdf")
        entries = await redis_index.list_all()
        assert len(entries) == 1
        assert entries[0]["doc_id"] == "doc-001"


class FakeRedis:
    """Minimal async Redis mock using a dict."""

    def __init__(self):
        self._data: dict[str, dict[str, str]] = {}

    async def hset(self, key: str, field: str, value: str) -> int:
        self._data.setdefault(key, {})[field] = value
        return 1

    async def hget(self, key: str, field: str) -> str | None:
        return self._data.get(key, {}).get(field)

    async def hdel(self, key: str, *fields: str) -> int:
        bucket = self._data.get(key, {})
        count = 0
        for f in fields:
            if f in bucket:
                del bucket[f]
                count += 1
        return count

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self._data.get(key, {}))

    async def hkeys(self, key: str) -> list[str]:
        return list(self._data.get(key, {}).keys())

    async def delete(self, *keys: str) -> int:
        count = 0
        for k in keys:
            if k in self._data:
                del self._data[k]
                count += 1
        return count
```

**Step 3: Run tests**

```bash
uv run pytest tests/unit/test_hash_index.py -v
```

**Step 4: Lint and commit**

```bash
uv run ruff check --fix src/dlightrag/ingestion/hash_index.py tests/unit/test_hash_index.py
uv run ruff format src/dlightrag/ingestion/hash_index.py tests/unit/test_hash_index.py
git add src/dlightrag/ingestion/hash_index.py tests/unit/test_hash_index.py
git commit -m "feat: add RedisHashIndex for multi-worker hash deduplication"
```

---

### Task 2: Add MongoHashIndex

**Files:**
- Modify: `src/dlightrag/ingestion/hash_index.py`
- Modify: `tests/unit/test_hash_index.py`

**Step 1: Add MongoHashIndex class after RedisHashIndex**

```python
class MongoHashIndex:
    """MongoDB-backed content hash index for deduplication.

    Multi-worker safe — uses MongoDB atomic operations.
    Requires: motor package (installed with LightRAG MongoKVStorage).
    """

    COLLECTION = "dlightrag_file_hashes"

    def __init__(self, workspace: str = "default", sources_dir: Path | None = None) -> None:
        self._workspace = workspace
        self._sources_dir = sources_dir
        self._collection: Any = None
        self._db: Any = None

    async def initialize(self) -> None:
        """Connect to MongoDB using LightRAG's shared client."""
        from lightrag.kg.mongo_impl import ClientManager

        self._db = await ClientManager.get_client()
        self._collection = self._db[self.COLLECTION]
        # Ensure index on workspace for efficient queries
        await self._collection.create_index("workspace")

    def check_exists(self, content_hash: str) -> tuple[bool, str | None]:
        return (False, None)  # Sync context fallback

    async def _async_check_exists(self, content_hash: str) -> tuple[bool, str | None]:
        doc = await self._collection.find_one(
            {"_id": content_hash, "workspace": self._workspace}
        )
        if doc:
            return (True, doc.get("doc_id"))
        return (False, None)

    async def register(self, content_hash: str, doc_id: str, file_path: str) -> None:
        await self._collection.update_one(
            {"_id": content_hash},
            {"$set": {
                "doc_id": doc_id,
                "file_path": file_path,
                "workspace": self._workspace,
                "created_at": datetime.now(UTC).isoformat(),
            }},
            upsert=True,
        )

    async def remove(self, content_hash: str) -> bool:
        result = await self._collection.delete_one(
            {"_id": content_hash, "workspace": self._workspace}
        )
        return result.deleted_count > 0

    async def clear(self) -> None:
        """Remove all hash entries for this workspace."""
        await self._collection.delete_many({"workspace": self._workspace})
        logger.info("MongoHashIndex cleared for workspace %s", self._workspace)

    async def should_skip_file(
        self, file_path: Path, replace: bool
    ) -> tuple[bool, str | None, str | None]:
        content_hash = await asyncio.to_thread(compute_file_hash, file_path)
        if replace:
            return (False, content_hash, None)
        exists, doc_id = await self._async_check_exists(content_hash)
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
        cursor = self._collection.find({"workspace": self._workspace})
        results = []
        async for doc in cursor:
            fp = doc.get("file_path", "")
            source_type = "unknown"
            path_parts = Path(fp).parts
            sources_idx = next((i for i, p in enumerate(path_parts) if p == "sources"), -1)
            if sources_idx >= 0 and len(path_parts) > sources_idx + 1:
                source_type = path_parts[sources_idx + 1]
            results.append({
                "file_path": fp,
                "doc_id": doc.get("doc_id", ""),
                "source_type": source_type,
                "file_name": Path(fp).name,
                "content_hash": doc["_id"],
                "created_at": doc.get("created_at", ""),
            })
        return results

    async def sync_existing(self) -> int:
        if not self._sources_dir or not self._sources_dir.exists():
            return 0
        existing_docs = await self.list_all()
        existing_hashes = {d["content_hash"] for d in existing_docs}
        existing_paths = {d["file_path"] for d in existing_docs}
        synced = 0
        for fp in self._sources_dir.rglob("*"):
            if fp.is_dir() or fp.name.startswith("."):
                continue
            if fp.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if str(fp) in existing_paths:
                continue
            try:
                h = await asyncio.to_thread(compute_file_hash, fp)
            except Exception as exc:
                logger.warning(f"Failed to hash {fp}: {exc}")
                continue
            if h in existing_hashes:
                continue
            doc_id = self.generate_doc_id_from_path(fp)
            await self.register(h, doc_id, str(fp))
            existing_hashes.add(h)
            existing_paths.add(str(fp))
            synced += 1
        if synced:
            logger.info(f"Synced {synced} hashes from sources directory")
        return synced
```

**Step 2: Add unit test with mocked Mongo collection**

Add to `tests/unit/test_hash_index.py`:

```python
class TestMongoHashIndex:
    """Tests for MongoHashIndex using a mock collection."""

    @pytest.fixture
    def mongo_index(self):
        from dlightrag.ingestion.hash_index import MongoHashIndex

        idx = MongoHashIndex(workspace="test", sources_dir=None)
        idx._collection = FakeMongoCollection()
        return idx

    async def test_register_and_check_exists(self, mongo_index):
        await mongo_index.register("sha256:aaa", "doc-001", "/path/a.pdf")
        exists, doc_id = await mongo_index._async_check_exists("sha256:aaa")
        assert exists is True
        assert doc_id == "doc-001"

    async def test_check_exists_missing(self, mongo_index):
        exists, doc_id = await mongo_index._async_check_exists("sha256:missing")
        assert exists is False

    async def test_remove(self, mongo_index):
        await mongo_index.register("sha256:aaa", "doc-001", "/path/a.pdf")
        removed = await mongo_index.remove("sha256:aaa")
        assert removed is True

    async def test_clear(self, mongo_index):
        await mongo_index.register("sha256:aaa", "doc-001", "/a.pdf")
        await mongo_index.register("sha256:bbb", "doc-002", "/b.pdf")
        await mongo_index.clear()
        entries = await mongo_index.list_all()
        assert len(entries) == 0

    async def test_list_all(self, mongo_index):
        await mongo_index.register("sha256:aaa", "doc-001", "/a.pdf")
        entries = await mongo_index.list_all()
        assert len(entries) == 1
        assert entries[0]["doc_id"] == "doc-001"


class FakeDeleteResult:
    def __init__(self, count: int):
        self.deleted_count = count


class FakeMongoCollection:
    """Minimal async MongoDB collection mock."""

    def __init__(self):
        self._docs: dict[str, dict] = {}

    async def find_one(self, filter: dict) -> dict | None:
        _id = filter.get("_id")
        workspace = filter.get("workspace")
        doc = self._docs.get(_id)
        if doc and doc.get("workspace") == workspace:
            return doc
        return None

    async def update_one(self, filter: dict, update: dict, upsert: bool = False):
        _id = filter["_id"]
        if _id in self._docs:
            self._docs[_id].update(update.get("$set", {}))
        elif upsert:
            self._docs[_id] = {"_id": _id, **update.get("$set", {})}

    async def delete_one(self, filter: dict) -> FakeDeleteResult:
        _id = filter.get("_id")
        workspace = filter.get("workspace")
        doc = self._docs.get(_id)
        if doc and doc.get("workspace") == workspace:
            del self._docs[_id]
            return FakeDeleteResult(1)
        return FakeDeleteResult(0)

    async def delete_many(self, filter: dict) -> FakeDeleteResult:
        workspace = filter.get("workspace")
        to_del = [k for k, v in self._docs.items() if v.get("workspace") == workspace]
        for k in to_del:
            del self._docs[k]
        return FakeDeleteResult(len(to_del))

    def find(self, filter: dict):
        workspace = filter.get("workspace")
        docs = [v for v in self._docs.values() if v.get("workspace") == workspace]
        return FakeAsyncCursor(docs)

    async def create_index(self, field: str):
        pass


class FakeAsyncCursor:
    def __init__(self, docs: list):
        self._docs = docs
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._docs):
            raise StopAsyncIteration
        doc = self._docs[self._idx]
        self._idx += 1
        return doc
```

**Step 3: Run tests**

```bash
uv run pytest tests/unit/test_hash_index.py -v
```

**Step 4: Lint and commit**

```bash
uv run ruff check --fix src/dlightrag/ingestion/hash_index.py tests/unit/test_hash_index.py
uv run ruff format src/dlightrag/ingestion/hash_index.py tests/unit/test_hash_index.py
git add src/dlightrag/ingestion/hash_index.py tests/unit/test_hash_index.py
git commit -m "feat: add MongoHashIndex for multi-worker hash deduplication"
```

---

### Task 3: Refactor PGHashIndex to use LightRAG ClientManager

**Files:**
- Modify: `src/dlightrag/ingestion/hash_index.py` (PGHashIndex class, lines ~342-530)

**Step 1: Change PGHashIndex constructor and add initialize()**

Replace the `__init__` and `initialize` methods of PGHashIndex:

```python
class PGHashIndex:
    """PostgreSQL-backed content hash index for deduplication.

    Multi-worker safe — uses PostgreSQL transactions instead of file locks.
    Table: dlightrag_file_hashes (created by init.sql or auto-created).
    Uses LightRAG's shared ClientManager for connection pooling.
    """

    TABLE = "dlightrag_file_hashes"

    def __init__(
        self, workspace: str = "default", sources_dir: Path | None = None,
        pool: Any = None,
    ) -> None:
        self._pool = pool  # asyncpg.Pool — set via initialize() or passed directly
        self._workspace = workspace
        self._sources_dir = sources_dir

    async def initialize(self) -> None:
        """Get shared pool from LightRAG ClientManager and create table."""
        if self._pool is None:
            from lightrag.kg.postgres_impl import ClientManager

            db = await ClientManager.get_client()
            self._pool = db.pool

        async with self._pool.acquire() as conn:
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
```

Note: `pool` param kept for backward compatibility but defaults to None (auto-detect).

**Step 2: Run existing tests to verify nothing breaks**

```bash
uv run pytest tests/unit/test_hash_index.py -v
```

**Step 3: Lint and commit**

```bash
uv run ruff check --fix src/dlightrag/ingestion/hash_index.py
uv run ruff format src/dlightrag/ingestion/hash_index.py
git add src/dlightrag/ingestion/hash_index.py
git commit -m "refactor: PGHashIndex uses LightRAG ClientManager by default"
```

---

### Task 4: Update service.py factory and remove _hash_pool

**Files:**
- Modify: `src/dlightrag/service.py` (lines 153, 392-432, 458-464)

**Step 1: Rewrite `_create_hash_index` and clean up `_hash_pool`**

In `__init__` (line 153): remove `self._hash_pool: Any | None = None`

Replace `_create_hash_index` (lines 392-432):

```python
    async def _create_hash_index(self, config: DlightragConfig) -> Any:
        """Create the appropriate hash index backend based on KV storage config.

        Uses the same backend as the configured KV storage for consistency.
        Falls back to JSON file-based HashIndex if the backend package is unavailable.
        """
        kv = config.kv_storage

        if kv.startswith("PG"):
            try:
                from dlightrag.ingestion.hash_index import PGHashIndex

                idx = PGHashIndex(workspace=config.workspace, sources_dir=config.sources_dir)
                await idx.initialize()
                logger.info("Hash index: PGHashIndex (PostgreSQL via shared pool)")
                return idx
            except ImportError:
                logger.warning("asyncpg not available, falling back to JSON HashIndex")
            except Exception as e:
                logger.warning(f"PGHashIndex creation failed, falling back to JSON: {e}")

        elif kv.startswith("Redis"):
            try:
                from dlightrag.ingestion.hash_index import RedisHashIndex

                idx = RedisHashIndex(workspace=config.workspace, sources_dir=config.sources_dir)
                await idx.initialize()
                logger.info("Hash index: RedisHashIndex (Redis via shared pool)")
                return idx
            except ImportError:
                logger.warning("redis not available, falling back to JSON HashIndex")
            except Exception as e:
                logger.warning(f"RedisHashIndex creation failed, falling back to JSON: {e}")

        elif kv.startswith("Mongo"):
            try:
                from dlightrag.ingestion.hash_index import MongoHashIndex

                idx = MongoHashIndex(workspace=config.workspace, sources_dir=config.sources_dir)
                await idx.initialize()
                logger.info("Hash index: MongoHashIndex (MongoDB via shared client)")
                return idx
            except ImportError:
                logger.warning("motor not available, falling back to JSON HashIndex")
            except Exception as e:
                logger.warning(f"MongoHashIndex creation failed, falling back to JSON: {e}")

        from dlightrag.ingestion.hash_index import HashIndex

        logger.info("Hash index: HashIndex (JSON file)")
        return HashIndex(config.working_dir_path, config.sources_dir, workspace=config.workspace)
```

In `close()` (lines 458-464): remove the `_hash_pool` cleanup block:

```python
        # DELETE THIS BLOCK:
        # Close PGHashIndex connection pool
        if self._hash_pool is not None:
            try:
                await self._hash_pool.close()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to close hash index pool", exc_info=True)
            self._hash_pool = None
```

**Step 2: Update the module docstring of hash_index.py**

Replace lines 1-9 docstring:

```python
"""Hash-based content deduplication for RAG ingestion.

Provides content-addressable storage tracking via SHA256 hashes.
Supports PostgreSQL (PGHashIndex), Redis (RedisHashIndex),
MongoDB (MongoHashIndex), and JSON file (HashIndex) backends.

Each distributed backend reuses LightRAG's shared connection manager.
HashIndex (JSON) is the fallback for local/single-process deployments.
"""
```

**Step 3: Run all tests**

```bash
uv run pytest tests/unit/ -x -q
```

Expected: all tests pass (294+), no regressions.

**Step 4: Lint and commit**

```bash
uv run ruff check --fix src/dlightrag/service.py src/dlightrag/ingestion/hash_index.py
uv run ruff format src/dlightrag/service.py src/dlightrag/ingestion/hash_index.py
git add src/dlightrag/service.py src/dlightrag/ingestion/hash_index.py
git commit -m "refactor: unified hash index factory, remove separate PG pool"
```

---

### Task 5: Add factory selection test

**Files:**
- Modify: `tests/unit/test_hash_index.py`

**Step 1: Add test verifying factory selects correct backend**

```python
class TestHashIndexFactory:
    """Test that _create_hash_index selects the right backend."""

    async def test_pg_backend_selection(self, monkeypatch):
        """PG config should attempt PGHashIndex (falls back to JSON without PG)."""
        from unittest.mock import AsyncMock, MagicMock

        from dlightrag.service import RAGService

        config = MagicMock()
        config.kv_storage = "PGKVStorage"
        config.workspace = "test"
        config.sources_dir = None
        config.working_dir_path = "/tmp/test"

        service = RAGService.__new__(RAGService)
        # PGHashIndex will fail (no PG), should fall back to JSON
        result = await service._create_hash_index(config)
        assert type(result).__name__ == "HashIndex"

    async def test_json_backend_selection(self, tmp_path):
        from unittest.mock import MagicMock

        from dlightrag.service import RAGService

        config = MagicMock()
        config.kv_storage = "JsonKVStorage"
        config.workspace = "test"
        config.sources_dir = None
        config.working_dir_path = tmp_path

        service = RAGService.__new__(RAGService)
        result = await service._create_hash_index(config)
        assert type(result).__name__ == "HashIndex"

    async def test_redis_fallback_to_json(self):
        """Redis config without redis package should fall back to JSON."""
        from unittest.mock import MagicMock

        from dlightrag.service import RAGService

        config = MagicMock()
        config.kv_storage = "RedisKVStorage"
        config.workspace = "test"
        config.sources_dir = None
        config.working_dir_path = "/tmp/test"

        service = RAGService.__new__(RAGService)
        result = await service._create_hash_index(config)
        # Will be HashIndex if redis not installed, RedisHashIndex if installed
        assert type(result).__name__ in ("HashIndex", "RedisHashIndex")

    async def test_mongo_fallback_to_json(self):
        """Mongo config without motor package should fall back to JSON."""
        from unittest.mock import MagicMock

        from dlightrag.service import RAGService

        config = MagicMock()
        config.kv_storage = "MongoKVStorage"
        config.workspace = "test"
        config.sources_dir = None
        config.working_dir_path = "/tmp/test"

        service = RAGService.__new__(RAGService)
        result = await service._create_hash_index(config)
        assert type(result).__name__ in ("HashIndex", "MongoHashIndex")
```

**Step 2: Run tests**

```bash
uv run pytest tests/unit/test_hash_index.py -v
```

**Step 3: Lint and commit**

```bash
uv run ruff check --fix tests/unit/test_hash_index.py
uv run ruff format tests/unit/test_hash_index.py
git add tests/unit/test_hash_index.py
git commit -m "test: add factory selection tests for unified hash index"
```
