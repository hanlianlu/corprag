# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""API routers package."""

from fastapi import APIRouter

from .answer_runs import router as answer_runs_router
from .corpus_mutations import router as corpus_mutations_router
from .files import router as files_router
from .files import serve_file
from .images import router as images_router
from .memory import router as memory_router
from .metadata import router as metadata_router
from .model_catalogue import router as model_catalogue_router
from .rag import router as rag_router
from .runs import router as runs_router
from .status import router as status_router
from .workspaces import router as workspaces_router

router = APIRouter()
router.include_router(status_router)
router.include_router(workspaces_router)
router.include_router(rag_router)
router.include_router(corpus_mutations_router)
router.include_router(answer_runs_router)
router.include_router(runs_router)
router.include_router(images_router)
router.include_router(files_router)
router.include_router(metadata_router)
router.include_router(memory_router)
router.include_router(model_catalogue_router)

__all__ = ["router", "serve_file"]
