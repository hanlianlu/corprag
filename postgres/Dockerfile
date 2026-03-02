# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
# PostgreSQL with pgvector + Apache AGE for corprag

FROM pgvector/pgvector:pg17

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates postgresql-server-dev-17 libreadline-dev \
    flex bison \
    && update-ca-certificates \
    && git clone --branch PG17 --depth 1 https://github.com/apache/age.git /tmp/age \
    && cd /tmp/age \
    && make install \
    && rm -rf /tmp/age \
    && apt-get purge -y build-essential git postgresql-server-dev-17 libreadline-dev flex bison \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Auto-create extensions on database init
COPY init.sql /docker-entrypoint-initdb.d/
