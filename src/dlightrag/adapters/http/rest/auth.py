# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""FastAPI bearer extraction over transport-neutral Access authentication.

Routes receive UserContext via FastAPI dependency injection.
They never know which auth strategy is active.
"""

from fastapi import HTTPException, Request

from dlightrag.application.access import (
    AuthenticationError,
    UserContext,
    authenticate_bearer_token,
)
from dlightrag.application.settings import authentication_settings


def _extract_bearer_token(request: Request) -> str:
    """Extract Bearer token from Authorization header. Raises 401 if missing/malformed."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    return auth_header[7:]


def authentication_http_error(error: AuthenticationError) -> HTTPException:
    """Map one Access authentication failure onto the REST/Web HTTP contract."""
    status_code = 500 if error.kind == "verifier_misconfigured" else 401
    return HTTPException(status_code=status_code, detail=str(error))


async def get_current_user(request: Request) -> UserContext:
    """Extract one bearer token and authenticate it through Access."""
    from dlightrag.adapters.http.application import get_application

    cfg = get_application(request).config
    if cfg.access.auth_mode == "none":
        return UserContext(user_id="anonymous", auth_mode="none")

    raw_token = _extract_bearer_token(request)
    user_id = request.headers.get("X-User-Id", "anonymous")
    try:
        return authenticate_bearer_token(
            raw_token,
            authentication_settings(cfg),
            default_user_id=user_id,
        )
    except AuthenticationError as exc:
        raise authentication_http_error(exc) from None
