# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
# Corporate RAG service - multimodal document ingestion & retrieval

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
LABEL maintainer="HanlianLyu"

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=300 uv sync --frozen --no-dev --no-install-project 2>/dev/null || \
    UV_HTTP_TIMEOUT=300 uv sync --no-dev --no-install-project

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
LABEL maintainer="hllyu"

WORKDIR /app

# LibreOffice + MinerU system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates \
    libreoffice-writer libreoffice-calc libreoffice-impress \
    fonts-dejavu-core \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    fonts-noto-core fonts-noto-cjk fontconfig \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY pyproject.toml README.md ./
COPY src/ src/

RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=300 uv sync --frozen --no-dev 2>/dev/null || \
    UV_HTTP_TIMEOUT=300 uv sync --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Default storage directory
RUN mkdir -p /app/corprag_storage
VOLUME /app/corprag_storage

EXPOSE 8100 8101

# Default: start the REST API server
CMD ["corprag-api"]
