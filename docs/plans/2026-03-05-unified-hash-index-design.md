# Unified HashIndex Design

## Problem

`HashIndex` (JSON file) and `PGHashIndex` (PostgreSQL) are two independent classes
with no shared interface. The factory in `service.py` only handles PG vs JSON —
Redis and Mongo users silently fall back to a local JSON file, which breaks
multi-worker deduplication in distributed deployments.

Additionally, `PGHashIndex` creates its own `asyncpg.Pool` instead of reusing
LightRAG's shared connection pool, wasting resources.

## Goal

Unify all HashIndex implementations behind a consistent interface, cover all
KV storage backends (PG, Redis, Mongo, JSON), and reuse LightRAG's existing
connection managers instead of creating separate connections.

## Approach: Reuse LightRAG ClientManagers

LightRAG already provides shared, reference-counted connection managers for each
backend:

| Backend | Manager | Attribute |
|---------|---------|-----------|
| PostgreSQL | `lightrag.kg.postgres_impl.ClientManager` | `db.pool` (asyncpg Pool) |
| Redis | `lightrag.kg.redis_impl.RedisConnectionManager` | shared pool |
| MongoDB | `lightrag.kg.mongo_impl.ClientManager` | shared AsyncDatabase |

Each HashIndex implementation calls the appropriate manager in `initialize()` to
obtain a shared connection, rather than creating its own.

## Design

### Implementations

```
hash_index.py
├── compute_file_hash()          — unchanged (SHA256 hashing)
├── HashIndex                    — JSON file (local/single-process fallback)
├── PGHashIndex                  — PostgreSQL via LightRAG ClientManager
├── RedisHashIndex               — Redis via RedisConnectionManager (NEW)
└── MongoHashIndex               — MongoDB via Mongo ClientManager (NEW)
```

### Interface (duck typing)

All implementations expose the same methods:

- `async initialize() -> None` — obtain shared connection
- `check_exists(hash) -> (bool, doc_id?)` — check if content hash exists
- `async register(hash, doc_id, file_path)` — register a new hash
- `async remove(hash) -> bool` — remove a hash entry
- `async clear() -> None` — remove all entries (for reset)
- `async should_skip_file(path, replace) -> (skip, hash, reason)` — dedup check
- `async list_all() -> list[dict]` — list all entries
- `async sync_existing() -> int` — sync hashes from sources dir
- `find_by_name(filename) -> (doc_id?, hash?, path?)` — lookup by filename
- `find_by_path(path) -> (doc_id?, hash?, path?)` — lookup by file path
- `invalidate() -> None` — clear in-memory cache
- `generate_doc_id_from_path(path) -> str` — static, returns stem

### PGHashIndex Refactor

Current: receives `asyncpg.Pool` from `service.py` which creates a separate pool.

New: receives `workspace` and `sources_dir`. In `initialize()`, imports
`lightrag.kg.postgres_impl.ClientManager`, calls `get_client()` to get the
shared `PostgreSQLDB` instance, and uses `db.pool` for queries.

This eliminates `self._hash_pool` from `RAGService` and the separate pool
creation in `_create_hash_index`.

### RedisHashIndex (new, ~60 lines)

- `initialize()`: import `lightrag.kg.redis_impl.RedisConnectionManager`,
  get shared pool, create `Redis` client
- Storage: Redis Hash key `dlightrag:file_hashes:{workspace}`,
  field = content_hash, value = JSON `{doc_id, file_path, created_at}`
- `clear()`: `DEL dlightrag:file_hashes:{workspace}`
- `sync_existing()`: scan sources dir, register missing hashes

### MongoHashIndex (new, ~60 lines)

- `initialize()`: import `lightrag.kg.mongo_impl.ClientManager`,
  get shared database
- Storage: collection `dlightrag_file_hashes`,
  document = `{_id: content_hash, workspace, doc_id, file_path, created_at}`
- `clear()`: `delete_many({"workspace": self._workspace})`
- `sync_existing()`: scan sources dir, register missing hashes

### Factory Method (service.py)

```python
async def _create_hash_index(self, config):
    kv = config.kv_storage
    if kv.startswith("PG"):
        try:
            from dlightrag.ingestion.hash_index import PGHashIndex
            idx = PGHashIndex(workspace=..., sources_dir=...)
            await idx.initialize()
            return idx
        except ImportError:
            logger.warning("asyncpg not available, falling back to JSON HashIndex")
    elif kv.startswith("Redis"):
        try:
            from dlightrag.ingestion.hash_index import RedisHashIndex
            idx = RedisHashIndex(workspace=..., sources_dir=...)
            await idx.initialize()
            return idx
        except ImportError:
            logger.warning("redis not available, falling back to JSON HashIndex")
    elif kv.startswith("Mongo"):
        try:
            from dlightrag.ingestion.hash_index import MongoHashIndex
            idx = MongoHashIndex(workspace=..., sources_dir=...)
            await idx.initialize()
            return idx
        except ImportError:
            logger.warning("motor not available, falling back to JSON HashIndex")

    return HashIndex(working_dir, sources_dir, workspace=...)
```

### Error Handling

Each distributed backend import is wrapped in try/except. If the required
package is not installed (e.g. `redis` for Redis users), it falls back to
JSON HashIndex with a warning log. This matches the existing PG fallback pattern.

## Files Changed

- `src/dlightrag/ingestion/hash_index.py` — add RedisHashIndex, MongoHashIndex;
  refactor PGHashIndex to use ClientManager
- `src/dlightrag/service.py` — update `_create_hash_index` factory, remove
  `self._hash_pool` and separate pool creation
- `tests/unit/test_hash_index.py` — add tests for new implementations

## Testing

- HashIndex (JSON): existing unit tests (no external deps)
- PGHashIndex: integration tests (requires PostgreSQL)
- RedisHashIndex: unit tests with mock redis client
- MongoHashIndex: unit tests with mock mongo client
- Factory: unit test verifying correct backend selection based on config
