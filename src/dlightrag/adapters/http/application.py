# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Access to the one Application instance bound by the HTTP lifespan."""

from __future__ import annotations

from typing import Any

from fastapi import Request

from dlightrag.application.errors import ApplicationClosedError


def get_application(request: Request) -> Any:
    """Return the lifespan-bound Application; never compose a fallback service."""
    application = getattr(request.app.state, "application", None)
    if application is None:
        raise ApplicationClosedError("Application is not bound to this HTTP transport")
    return application


__all__ = ["get_application"]
