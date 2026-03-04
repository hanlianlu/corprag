-- Auto-create required extensions for dlightrag
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;

-- AGE requires the ag_catalog schema on search_path
SET search_path = ag_catalog, "$user", public;
