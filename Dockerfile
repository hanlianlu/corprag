# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
# DlightRAG - multimodal RAG

ARG UV_VERSION=0.11.21

FROM python:3.14.7-slim-bookworm AS uv-bin
ARG UV_VERSION
RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

# Resolve current upstream releases at build time. The installer rejects assets
# without GitHub-published SHA-256 provenance and enforces DlightRAG's static minima.
FROM python:3.14.7-slim-bookworm AS search-tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY src/dlightrag/engine/agent/environment/toolchain.py /tmp/toolchain.py
RUN python /tmp/toolchain.py install --allow-runtime-download \
    --cache-root /tmp/search-tool-cache --bin-dir /search-tools

# Match GitHub Actions so one npm version produces the same cross-platform lock behavior.
FROM node:26-slim AS frontend
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json frontend/
RUN --mount=type=cache,target=/root/.npm npm --prefix frontend ci
COPY frontend/ frontend/
# Vite writes HTML and hashed assets into ../src/dlightrag/adapters/http/browser/static/app, which the wheel picks up.
RUN npm --prefix frontend run build

FROM python:3.14.7-slim-bookworm AS builder

WORKDIR /app
ENV UV_LINK_MODE=copy
COPY --from=uv-bin /usr/local/bin/uv /bin/

COPY pyproject.toml uv.lock ./
COPY packages/memory/pyproject.toml packages/memory/pyproject.toml
# Deps only — binary-only (UV_NO_BUILD): never compile an sdist; the slim base has
# no toolchain, so a missing wheel fails fast. Keep it off the project build below.
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=300 UV_NO_BUILD=1 uv sync --frozen --no-dev --no-install-workspace

COPY LICENSE NOTICE README.md ./
COPY packages/ packages/
COPY src/ src/
COPY --from=frontend /app/src/dlightrag/adapters/http/browser/static/app/ src/dlightrag/adapters/http/browser/static/app/
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=300 uv sync --frozen --no-dev --no-editable

FROM python:3.14.7-slim-bookworm
LABEL maintainer="HanlianLyu"

WORKDIR /app

# Create non-root user BEFORE copying files to avoid chown layer duplication
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=frontend /usr/local/bin/node /usr/local/bin/node
COPY --from=search-tools /search-tools/fd /search-tools/rg /usr/local/bin/
RUN groupadd --gid 1000 app && useradd --uid 1000 --gid app --create-home app \
    && mkdir -p /app/dlightrag_storage /home/app/.dlightrag/agent_workspaces \
    /home/app/.dlightrag/skills /home/app/.dlightrag/owner_skills \
    && chown app:app /app/dlightrag_storage /home/app/.dlightrag/agent_workspaces \
    /home/app/.dlightrag/skills /home/app/.dlightrag/owner_skills

COPY --from=builder --chown=app:app /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8100 8101

USER app

# Default image role; deployments can override it for MCP or maintenance commands.
CMD ["dlightrag-api"]
