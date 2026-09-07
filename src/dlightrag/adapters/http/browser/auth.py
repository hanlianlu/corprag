# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web UI authentication using the global DlightRAG auth mode."""

import base64
import binascii
import secrets
from collections.abc import Callable
from urllib.parse import quote, urlencode, urlsplit

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from dlightrag.adapters.http.browser.app_shell import app_html_response
from dlightrag.adapters.http.browser.edge_identity import (
    EdgeIdentityError,
    edge_identity_provider,
)
from dlightrag.application.access import (
    AuthenticationError,
    UserContext,
    authenticate_bearer_token,
)
from dlightrag.application.config import DlightragConfig, get_config
from dlightrag.application.settings import authentication_settings

WEB_AUTH_COOKIE = "dlightrag_web_auth"
WEB_CSRF_COOKIE = "dlightrag_web_csrf"
CSRF_HEADER = "X-CSRF-Token"
_PUBLIC_WEB_PATHS = {"/web/login", "/web/logout"}
_WEB_COOKIE_PATH = "/web"
_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

router = APIRouter()


def _authenticate_bearer(
    raw_token: str,
    config: DlightragConfig,
    *,
    default_user_id: str = "anonymous",
) -> UserContext:
    try:
        return authenticate_bearer_token(
            raw_token,
            authentication_settings(config),
            default_user_id=default_user_id,
        )
    except AuthenticationError as exc:
        status_code = 500 if exc.kind == "verifier_misconfigured" else 401
        raise HTTPException(status_code=status_code, detail=str(exc)) from None


def _safe_next_path(value: str | None) -> str:
    """Return a same-origin web path for post-login redirects."""
    if not value:
        return "/web/"
    cleaned = value.replace("\r", "").replace("\n", "").strip()
    if cleaned == "/web":
        return "/web"
    if cleaned.startswith("/web/"):
        return "/web/" + cleaned.removeprefix("/web/")
    return "/web/"


def _login_url(next_path: str) -> str:
    return f"/web/login?next={quote(_safe_next_path(next_path), safe='')}"


def _request_next_path(request: Request) -> str:
    path = str(request.url.path)
    query = str(request.url.query)
    return f"{path}?{query}" if query else path


def _origin_tuple(value: str) -> tuple[str, str, int] | None:
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            return None
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return None
    return parsed.scheme, parsed.hostname.lower(), port


def _request_origin(request: Request) -> tuple[str, str, int] | None:
    try:
        hostname = request.url.hostname
        if hostname is None or request.url.scheme not in {"http", "https"}:
            return None
        port = request.url.port or (443 if request.url.scheme == "https" else 80)
    except ValueError:
        return None
    return request.url.scheme, hostname.lower(), port


def _has_exact_same_origin(request: Request) -> bool:
    origin = request.headers.get("Origin")
    return origin is not None and _origin_tuple(origin) == _request_origin(request)


def _bearer_from_header(request: Request) -> str | None:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    if auth_header:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    return None


def _token_from_request(request: Request) -> tuple[str | None, str | None]:
    raw = _bearer_from_header(request)
    if raw is not None:
        return raw, "header"
    raw = request.cookies.get(WEB_AUTH_COOKIE)
    if raw:
        return _decode_cookie_token(raw), "cookie"
    return None, None


def _encode_cookie_token(token: str) -> str:
    return base64.urlsafe_b64encode(token.encode()).decode().rstrip("=")


def _decode_cookie_token(value: str) -> str | None:
    try:
        padded = value + ("=" * (-len(value) % 4))
        return base64.b64decode(padded, altchars=b"-_", validate=True).decode()
    except binascii.Error, UnicodeDecodeError, ValueError:
        return None


def _set_auth_cookie(response: Response, request: Request, token: str) -> None:
    response.set_cookie(
        key=WEB_AUTH_COOKIE,
        value=_encode_cookie_token(token),
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        path=_WEB_COOKIE_PATH,
    )


def _clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(key=WEB_AUTH_COOKIE, path=_WEB_COOKIE_PATH)


def _ensure_csrf_cookie(request: Request, response: Response) -> None:
    """Issue the JS-readable double-submit token once per browser."""
    if request.cookies.get(WEB_CSRF_COOKIE):
        return
    response.set_cookie(
        key=WEB_CSRF_COOKIE,
        value=secrets.token_urlsafe(32),
        httponly=False,
        samesite="strict",
        secure=request.url.scheme == "https",
        path=_WEB_COOKIE_PATH,
    )


def _csrf_header_matches(request: Request) -> bool:
    cookie_token = request.cookies.get(WEB_CSRF_COOKIE)
    if not cookie_token:
        return True
    header_token = request.headers.get(CSRF_HEADER, "")
    return secrets.compare_digest(cookie_token, header_token)


def _reject_web_mutation(request: Request) -> bool:
    """Return True when one unsafe /web request must be rejected.

    Browsers that carry the double-submit cookie must echo it in the header;
    cookie-authenticated (paste) mutations must additionally prove exact
    same-origin via an Origin header, and any Origin header must match.
    Scripted clients without any web cookie are left to their bearer
    credentials.
    """
    if request.method.upper() not in _UNSAFE_METHODS:
        return False
    if not request.url.path.startswith("/web"):
        return False
    if not _csrf_header_matches(request):
        return True
    bearer_present = request.headers.get("Authorization", "").startswith("Bearer ")
    if (
        WEB_AUTH_COOKIE in request.cookies
        and not bearer_present
        and request.headers.get("Origin") is None
    ):
        return True
    origin = request.headers.get("Origin")
    return origin is not None and not _has_exact_same_origin(request)


def _browser_missing_auth_response(request: Request) -> Response:
    if request.method.upper() == "GET":
        return RedirectResponse(_login_url(_request_next_path(request)), status_code=303)
    return PlainTextResponse("Authentication required", status_code=401)


class WebAuthMiddleware(BaseHTTPMiddleware):
    """Protect `/web/*` with the same auth mode used by REST/MCP."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        config_getter: Callable[[], DlightragConfig] = get_config,
    ) -> None:
        super().__init__(app)
        self._config_getter = config_getter

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not path.startswith("/web"):
            return await call_next(request)
        if path in _PUBLIC_WEB_PATHS:
            # Login/logout CSRF: browser-driven POSTs must be exact same-origin.
            if path in {"/web/login", "/web/logout"} and request.method.upper() == "POST":
                origin = request.headers.get("Origin")
                if origin is not None and not _has_exact_same_origin(request):
                    return PlainTextResponse("Cross-origin request rejected", status_code=403)
            return await call_next(request)

        cfg = self._config_getter()
        if cfg.access.auth_mode == "none":
            request.state.user_context = UserContext(user_id="anonymous", auth_mode="none")
            return await call_next(request)

        if cfg.access.web_identity.edge is not None:
            return await self._dispatch_edge_identity(cfg, request, call_next)

        source: str | None = None
        try:
            raw_token, source = _token_from_request(request)
            if not raw_token:
                if source == "cookie" and request.method.upper() == "GET":
                    response = RedirectResponse(
                        _login_url(_request_next_path(request)), status_code=303
                    )
                    _clear_auth_cookie(response)
                    return response
                return _browser_missing_auth_response(request)
            request.state.user_context = _authenticate_bearer(
                raw_token,
                cfg,
                default_user_id=request.headers.get("X-User-Id", "anonymous"),
            )
        except HTTPException as exc:
            if source == "cookie" and request.method.upper() == "GET":
                response = RedirectResponse(
                    _login_url(_request_next_path(request)), status_code=303
                )
                _clear_auth_cookie(response)
                return response
            return PlainTextResponse(str(exc.detail), status_code=exc.status_code)

        if _reject_web_mutation(request):
            return PlainTextResponse("Cross-origin request rejected", status_code=403)

        return await self._finish_web_response(request, call_next)

    async def _dispatch_edge_identity(self, cfg, request: Request, call_next) -> Response:
        """Resolve the Web caller from the configured edge credential only."""
        try:
            provider = edge_identity_provider(cfg.access.web_identity)
            identity = provider.authenticate(request)
        except EdgeIdentityError as exc:
            status = 500 if exc.kind == "misconfigured" else 401
            return PlainTextResponse(
                "Authentication required" if status == 401 else str(exc),
                status_code=status,
            )
        request.state.user_context = UserContext(
            user_id=identity.subject,
            auth_mode="jwt",
            claims=identity.claims,
        )
        if _reject_web_mutation(request):
            return PlainTextResponse("Cross-origin request rejected", status_code=403)
        return await self._finish_web_response(request, call_next)

    async def _finish_web_response(self, request: Request, call_next) -> Response:
        """Run the request and issue the double-submit cookie on GET responses."""
        response = await call_next(request)
        if request.method.upper() == "GET":
            _ensure_csrf_cookie(request, response)
        return response


@router.get("/login", response_class=FileResponse)
async def login_page(request: Request, next: str = "/web/"):
    """Serve the static paste-token form when global auth is enabled."""
    from dlightrag.adapters.http.application import get_application

    cfg = get_application(request).config
    target = _safe_next_path(next)
    if cfg.access.auth_mode == "none" or cfg.access.web_identity.edge is not None:
        # The edge owns login; the paste form is the no-edge development hatch.
        return RedirectResponse(target, status_code=303)
    return app_html_response("login.html")


@router.post("/login")
async def login(
    request: Request,
    token: str = Form(default=""),
    next: str = Form(default="/web/"),
):
    """Validate a bearer token and store it in an HttpOnly web cookie."""
    from dlightrag.adapters.http.application import get_application

    cfg = get_application(request).config
    target = _safe_next_path(next)
    if cfg.access.auth_mode == "none" or cfg.access.web_identity.edge is not None:
        return RedirectResponse(target, status_code=303)
    try:
        _authenticate_bearer(token, cfg)
    except HTTPException:
        query = urlencode({"next": target, "error": "Authentication failed"})
        return RedirectResponse(f"/web/login?{query}", status_code=303)

    response = RedirectResponse(target, status_code=303)
    _set_auth_cookie(response, request, token)
    return response


@router.post("/logout")
async def logout() -> RedirectResponse:
    """Clear the web auth cookie."""
    response = RedirectResponse("/web/login", status_code=303)
    _clear_auth_cookie(response)
    return response


__all__ = ["WEB_AUTH_COOKIE", "WEB_CSRF_COOKIE", "WebAuthMiddleware", "router"]
