"""
=============================================================================
HUMMINGBIRD-LEA - Auth Required Middleware
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Default-deny middleware:
- Allows frontend (/) and static assets (/static/*) so the login UI loads
- Allows /api/auth/* endpoints
- Optionally allows /api/health/* (adjustable)
- Requires Authorization: Bearer <access_token> for all other /api/* routes
=============================================================================
"""

from datetime import datetime
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from core.utils.auth import decode_token


PUBLIC_PATH_PREFIXES = (
    "/static",
    "/robots.txt",
)

PUBLIC_API_PREFIXES = (
    "/api/microsoft",
    "/api/auth",
    "/api/health",  # Optional: remove if you want health locked down too
    "/api/version",
)


class AuthRequiredMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Allow login page + static assets
        if path == "/" or any(path.startswith(p) for p in PUBLIC_PATH_PREFIXES):
            return await call_next(request)

        # Only enforce on API routes
        if not path.startswith("/api/"):
            return await call_next(request)

        # Allow auth (and optionally health)
        if any(path.startswith(p) for p in PUBLIC_API_PREFIXES):
            return await call_next(request)

        # Require bearer token
        auth = request.headers.get("Authorization", "")
        if not auth.lower().startswith("bearer "):
            return JSONResponse(
                {"detail": "Authentication required"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth.split(" ", 1)[1].strip()
        token_data = decode_token(token)
        if not token_data:
            return JSONResponse(
                {"detail": "Invalid or expired token"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Expiry check (decode_token currently does not enforce type)
        if datetime.utcnow() > token_data.exp:
            return JSONResponse(
                {"detail": "Token has expired"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Attach username for downstream handlers if useful
        request.state.username = token_data.username

        return await call_next(request)
