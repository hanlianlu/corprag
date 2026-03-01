-- Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
-- PostgreSQL initialization for corprag
-- Extensions: pgvector (vector search), age (graph storage)

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- Custom hash index table for content deduplication
CREATE TABLE IF NOT EXISTS corprag_file_hashes (
    content_hash TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    workspace TEXT NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_file_hashes_doc_id ON corprag_file_hashes(doc_id);
CREATE INDEX IF NOT EXISTS idx_file_hashes_workspace ON corprag_file_hashes(workspace);
