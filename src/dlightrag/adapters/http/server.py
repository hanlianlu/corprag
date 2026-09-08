# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""FastAPI app factory for DlightRAG REST server.

Entry point: dlightrag-api
All endpoint logic lives in routes.py; this module handles app lifecycle,
middleware, exception handlers, and router mounting.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dlightrag import create_application
from dlightrag.adapters.http.rest.middleware import (
    RequestBodyLimitMiddleware,
    RequestIdMiddleware,
    install_request_id_log_record_factory,
)
from dlightrag.adapters.http.rest.models import ANSWER_REQUEST_PART_MAX_BYTES, ErrorDetail
from dlightrag.adapters.http.rest.routes import router
from dlightrag.application import ApplicationClosedError
from dlightrag.application.answer_runs import AnswerRuntimeUnavailableError
from dlightrag.application.corpus_admin import MetadataValidationError
from dlightrag.application.errors import RunSchemaError, StorageSchemaError
from dlightrag.application.model_catalogue import (
    ModelCatalogueSchemaError,
    ModelCatalogueUnavailableError,
)
from dlightrag.application.retrieval import CorpusUnavailableError, RetrievalInputError
from dlightrag.application.runs import RunRuntimeUnavailableError
from dlightrag.application.web_conversations import WebConversationSchemaError
from dlightrag.engine.answer.client_contracts import MAX_QUERY_IMAGES
from dlightrag.engine.answer.errors import AnswerInputError, InvalidToolConfigurationError

if TYPE_CHECKING:
    from dlightrag.application.config import DlightragConfig

logger = logging.getLogger(__name__)

# Room for the query, workspace list and identifiers that travel beside the images.
_JSON_BODY_OVERHEAD_BYTES = 64 * 1024
# Multipart framing: boundaries, part headers, and the small text fields beside files.
_MULTIPART_FRAMING_BYTES = 64 * 1024
# Ingest and Web workspace uploads additionally carry a per-request metadata envelope.
_MULTIPART_ENVELOPE_BYTES = 2 * 1024 * 1024 + _MULTIPART_FRAMING_BYTES


def _request_body_limits(cfg: DlightragConfig) -> tuple[int, dict[str, int]]:
    """Derive the receive-layer caps once, before any body is parsed.

    These are transport ceilings that keep an oversized or chunked body out of
    memory and temporary storage. They deliberately sit above each route's
    semantic limits — attachment count, per-item bytes, total bytes, pixels —
    which stay in the routes because only the route knows what the parts mean.
    """
    # Base64 inflates an image by 4/3.
    encoded_image_bytes = ((cfg.answer.generation.image_max_bytes + 2) // 3) * 4
    answer_multipart_max = (
        cfg.answer.generation.max_total_attachment_bytes
        + ANSWER_REQUEST_PART_MAX_BYTES
        + _MULTIPART_FRAMING_BYTES
    )
    return (
        max(
            _JSON_BODY_OVERHEAD_BYTES + MAX_QUERY_IMAGES * encoded_image_bytes,
            ANSWER_REQUEST_PART_MAX_BYTES,
        ),
        {
            "/answer": answer_multipart_max,
            "/web/api/answer": answer_multipart_max,
            "/runs/corpus/ingest/upload": cfg.corpus.ingestion.max_upload_bytes
            + _MULTIPART_ENVELOPE_BYTES,
            "/runs/corpus/replace/upload": cfg.corpus.ingestion.max_upload_bytes
            + _MULTIPART_ENVELOPE_BYTES,
            "/runs/corpus/ingest/uploads": cfg.max_upload_batch_bytes + _MULTIPART_ENVELOPE_BYTES,
            "/runs/corpus/replace/uploads": cfg.max_upload_batch_bytes + _MULTIPART_ENVELOPE_BYTES,
            "/web/api/files/upload": cfg.max_upload_batch_bytes + _MULTIPART_ENVELOPE_BYTES,
        },
    )


# ═══════════════════════════════════════════════════════════════════
# Lifespan
# ═══════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    try:
        web_enabled = bool(getattr(_app.state, "web_enabled", False))
        application = await create_application(
            config=_app.state.config,
            web_enabled=web_enabled,
        )
    except Exception:
        logger.exception("Failed to initialize DlightRAG application")
        raise
    _app.state.application = application
    try:
        yield
    finally:
        await application.aclose()


# ═══════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════


def create_app(*, include_web_app: bool = True) -> FastAPI:
    """Create the REST API and optionally mount the bundled browser app."""
    from dlightrag.application.config import get_config

    cfg = get_config()

    application = FastAPI(
        title="dlightrag",
        description="DlightRAG - LightRAG-main unified multimodal RAG service",
        version=__import__("dlightrag").__version__,
        lifespan=lifespan,
    )
    application.state.config = cfg

    # Install the body limiter before response middleware so its 413 replies still
    # receive CORS and request-ID headers.
    max_body_bytes, multipart_path_max_bytes = _request_body_limits(cfg)
    application.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=max_body_bytes,
        multipart_path_max_bytes=multipart_path_max_bytes,
    )

    # -- Request ID middleware --
    application.add_middleware(RequestIdMiddleware)

    # -- CORS middleware (config-driven; see DlightragConfig.cors_allow_origins) --
    # allow_credentials toggles based on origin list: browsers refuse '*' +
    # credentials, so we only enable credentials when origins are explicit.
    allow_credentials = cfg.access.cors_allow_origins != ("*",)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.access.cors_allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Exception handlers --

    @application.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,  # noqa: ARG001
        exc: HTTPException,
    ) -> JSONResponse:
        """Wrap HTTP errors, preserving explicit typed browser command envelopes."""
        status = exc.status_code
        if (
            isinstance(exc.detail, dict)
            and isinstance(exc.detail.get("kind"), str)
            and isinstance(exc.detail.get("message"), str)
        ):
            return JSONResponse(
                status_code=status,
                content=exc.detail,
                headers=exc.headers,
            )
        if status == 503:
            error_type = "unavailable"
        elif 400 <= status < 500:
            error_type = "validation" if status in {400, 413, 422} else "auth"
        else:
            error_type = "internal"
        body = ErrorDetail(detail=str(exc.detail), error_type=error_type)
        return JSONResponse(
            status_code=status,
            content=body.model_dump(),
            headers=exc.headers,
        )

    @application.exception_handler(ApplicationClosedError)
    @application.exception_handler(CorpusUnavailableError)
    @application.exception_handler(AnswerRuntimeUnavailableError)
    @application.exception_handler(RunRuntimeUnavailableError)
    @application.exception_handler(ModelCatalogueUnavailableError)
    async def rag_unavailable_handler(
        request: Request,  # noqa: ARG001
        exc: (
            ApplicationClosedError
            | CorpusUnavailableError
            | AnswerRuntimeUnavailableError
            | RunRuntimeUnavailableError
            | ModelCatalogueUnavailableError
        ),
    ) -> JSONResponse:
        body = ErrorDetail(detail=str(exc), error_type="unavailable")
        return JSONResponse(status_code=503, content=body.model_dump())

    @application.exception_handler(RetrievalInputError)
    async def retrieval_input_handler(
        request: Request,  # noqa: ARG001
        exc: RetrievalInputError,
    ) -> JSONResponse:
        body = ErrorDetail(detail=str(exc), error_type="validation")
        return JSONResponse(status_code=422, content=body.model_dump())

    @application.exception_handler(PermissionError)
    async def permission_error_handler(
        request: Request,  # noqa: ARG001
        exc: PermissionError,
    ) -> JSONResponse:
        body = ErrorDetail(detail=str(exc), error_type="auth")
        return JSONResponse(status_code=403, content=body.model_dump())

    @application.exception_handler(AnswerInputError)
    async def answer_input_error_handler(
        request: Request,  # noqa: ARG001
        exc: AnswerInputError,
    ) -> JSONResponse:
        """Answer input rejection -> 422 with a stable error kind."""
        body = ErrorDetail(detail=str(exc), error_type="validation", error_kind=exc.error_kind)
        return JSONResponse(status_code=422, content=body.model_dump())

    @application.exception_handler(InvalidToolConfigurationError)
    async def invalid_tool_configuration_handler(
        request: Request,  # noqa: ARG001
        exc: InvalidToolConfigurationError,
    ) -> JSONResponse:
        """Server tool-composition failure -> 500; the colliding names stay in the log."""
        logger.error("Answer tool composition is invalid", exc_info=exc)
        body = ErrorDetail(
            detail=exc.public_message,
            error_type="configuration",
            error_kind=exc.error_kind,
        )
        return JSONResponse(status_code=500, content=body.model_dump())

    @application.exception_handler(MetadataValidationError)
    async def metadata_validation_error_handler(
        request: Request,  # noqa: ARG001
        exc: MetadataValidationError,
    ) -> JSONResponse:
        """Metadata is validated below the request model, so it needs its own mapping."""
        body = ErrorDetail(detail=str(exc), error_type="validation")
        return JSONResponse(status_code=400, content=body.model_dump())

    async def schema_validation_error_handler(
        request: Request,  # noqa: ARG001
        exc: Exception,
    ) -> JSONResponse:
        """An incompatible schema is an operator fault; callers see no schema detail."""
        logger.error("Durable schema is incompatible with this revision", exc_info=exc)
        body = ErrorDetail(
            detail="Durable storage is unavailable on this deployment",
            error_type="unavailable",
        )
        return JSONResponse(status_code=503, content=body.model_dump())

    for schema_error in (
        StorageSchemaError,
        RunSchemaError,
        WebConversationSchemaError,
        ModelCatalogueSchemaError,
    ):
        application.add_exception_handler(schema_error, schema_validation_error_handler)

    # -- API routes --
    application.include_router(router)

    # -- Web frontend --
    if include_web_app:
        from dlightrag.adapters.http.browser.auth import WebAuthMiddleware
        from dlightrag.adapters.http.browser.routes import router as web_router
        from dlightrag.adapters.http.browser.static_files import STATIC_DIR, WebStaticFiles
        from dlightrag.application.web_conversations import WebConversationUnavailableError

        application.state.web_enabled = True

        @application.exception_handler(WebConversationUnavailableError)
        async def web_conversation_unavailable_handler(
            request: Request,  # noqa: ARG001
            exc: WebConversationUnavailableError,
        ) -> JSONResponse:
            body = ErrorDetail(detail=exc.detail, error_type="unavailable")
            return JSONResponse(status_code=503, content=body.model_dump())

        application.add_middleware(WebAuthMiddleware, config_getter=lambda cfg=cfg: cfg)
        application.include_router(web_router)
        if STATIC_DIR.exists():
            application.mount(
                "/static",
                WebStaticFiles(directory=str(STATIC_DIR)),
                name="static",
            )

    return application


def get_app() -> FastAPI:
    """ASGI factory entry point (e.g. uvicorn dlightrag.adapters.http.server:get_app --factory)."""
    return create_app()


def main() -> None:
    """Entry point for dlightrag-api."""
    import argparse

    import uvicorn

    from dlightrag.application.config import get_config, load_config, set_config

    parser = argparse.ArgumentParser(
        description="dlightrag REST API server",
        suggest_on_error=True,
    )
    parser.add_argument("--env-file", help="Path to .env configuration file")
    args = parser.parse_args()

    if args.env_file:
        config = load_config(args.env_file)
        set_config(config)
    else:
        config = get_config()
    install_request_id_log_record_factory()
    logging.basicConfig(
        level=getattr(logging, config.observability.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s",
    )

    uvicorn.run(
        "dlightrag.adapters.http.server:get_app",
        host=config.interfaces.api.host,
        port=config.interfaces.api.port,
        log_level=config.observability.log_level,
        factory=True,
    )


__all__ = ["create_app", "get_app", "main"]
