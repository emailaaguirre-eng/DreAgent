"""
=============================================================================
HUMMINGBIRD-LEA - Security Middleware
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Security middleware for request validation, rate limiting, and protection.

Features:
- Rate limiting per IP/user
- Input validation and sanitization
- Request size limits
- Security headers
- SQL injection prevention
- XSS protection
=============================================================================
"""

import logging
import time
import re
import hashlib
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# =============================================================================
# Rate Limiter
# =============================================================================

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int = 240
    requests_per_hour: int = 1000
    burst_limit: int = 25          # Max requests in 1 second
    block_duration_minutes: int = 15
    whitelist_ips: List[str] = field(default_factory=lambda: ["127.0.0.1", "::1"])


class RateLimiter:
    """
    In-memory rate limiter with sliding window.

    Tracks requests per IP address and enforces limits.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._blocked: Dict[str, datetime] = {}
        self._cleanup_interval = 60  # seconds
        self._last_cleanup = time.time()

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check for forwarded headers (behind proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _cleanup_old_requests(self, ip: str):
        """Remove requests older than 1 hour"""
        cutoff = time.time() - 3600
        self._requests[ip] = [t for t in self._requests[ip] if t > cutoff]

    def _periodic_cleanup(self):
        """Periodically clean up all old data"""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now

        # Clean up old requests
        cutoff = now - 3600
        for ip in list(self._requests.keys()):
            self._requests[ip] = [t for t in self._requests[ip] if t > cutoff]
            if not self._requests[ip]:
                del self._requests[ip]

        # Clean up expired blocks
        now_dt = datetime.utcnow()
        for ip in list(self._blocked.keys()):
            if self._blocked[ip] < now_dt:
                del self._blocked[ip]

    def is_blocked(self, request: Request) -> bool:
        """Check if IP is currently blocked"""
        ip = self._get_client_ip(request)

        if ip in self.config.whitelist_ips:
            return False

        if ip in self._blocked:
            if self._blocked[ip] > datetime.utcnow():
                return True
            else:
                del self._blocked[ip]

        return False

    def check_rate_limit(self, request: Request) -> tuple[bool, Optional[str]]:
        """
        Check if request is within rate limits.

        Returns:
            Tuple of (is_allowed, error_message)
        """
        ip = self._get_client_ip(request)

        # Whitelist check
        if ip in self.config.whitelist_ips:
            return True, None

        # Block check
        if self.is_blocked(request):
            return False, "Too many requests. Please try again later."

        self._periodic_cleanup()
        self._cleanup_old_requests(ip)

        now = time.time()
        requests = self._requests[ip]

        # Check burst limit (last 1 second)
        recent = [t for t in requests if t > now - 1]
        if len(recent) >= self.config.burst_limit:
            self._block_ip(ip)
            return False, "Rate limit exceeded (burst). Please slow down."

        # Check per-minute limit
        last_minute = [t for t in requests if t > now - 60]
        if len(last_minute) >= self.config.requests_per_minute:
            return False, f"Rate limit exceeded ({self.config.requests_per_minute}/min)."

        # Check per-hour limit
        if len(requests) >= self.config.requests_per_hour:
            self._block_ip(ip)
            return False, f"Rate limit exceeded ({self.config.requests_per_hour}/hour)."

        # Record request
        self._requests[ip].append(now)
        return True, None

    def _block_ip(self, ip: str):
        """Block an IP address"""
        block_until = datetime.utcnow() + timedelta(minutes=self.config.block_duration_minutes)
        self._blocked[ip] = block_until
        logger.warning(f"Blocked IP {ip} until {block_until}")

    def get_remaining(self, request: Request) -> Dict[str, int]:
        """Get remaining rate limit allowance"""
        ip = self._get_client_ip(request)
        self._cleanup_old_requests(ip)

        now = time.time()
        requests = self._requests[ip]

        last_minute = len([t for t in requests if t > now - 60])
        last_hour = len(requests)

        return {
            "remaining_per_minute": max(0, self.config.requests_per_minute - last_minute),
            "remaining_per_hour": max(0, self.config.requests_per_hour - last_hour),
        }


# =============================================================================
# Input Validator
# =============================================================================

class InputValidator:
    """
    Input validation and sanitization.

    Protects against:
    - SQL injection
    - XSS attacks
    - Path traversal
    - Command injection
    """

    # Dangerous patterns
    SQL_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|TRUNCATE)\b)",
        r"(--|\#|\/\*|\*\/)",
        r"(\bOR\b\s+\d+\s*=\s*\d+)",
        r"(\bAND\b\s+\d+\s*=\s*\d+)",
        r"(;.*\b(SELECT|INSERT|UPDATE|DELETE|DROP)\b)",
    ]

    XSS_PATTERNS = [
        r"<script[^>]*>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
    ]

    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e%2f",
        r"%2e%2e/",
        r"\.%2e/",
    ]

    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$]",
        r"\$\(",
        r"`.*`",
    ]

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self._compiled_sql = [re.compile(p, re.IGNORECASE) for p in self.SQL_PATTERNS]
        self._compiled_xss = [re.compile(p, re.IGNORECASE) for p in self.XSS_PATTERNS]
        self._compiled_path = [re.compile(p, re.IGNORECASE) for p in self.PATH_TRAVERSAL_PATTERNS]
        self._compiled_cmd = [re.compile(p) for p in self.COMMAND_INJECTION_PATTERNS]

    def check_sql_injection(self, value: str) -> bool:
        """Check for SQL injection patterns"""
        for pattern in self._compiled_sql:
            if pattern.search(value):
                return True
        return False

    def check_xss(self, value: str) -> bool:
        """Check for XSS patterns"""
        for pattern in self._compiled_xss:
            if pattern.search(value):
                return True
        return False

    def check_path_traversal(self, value: str) -> bool:
        """Check for path traversal patterns"""
        for pattern in self._compiled_path:
            if pattern.search(value):
                return True
        return False

    def check_command_injection(self, value: str) -> bool:
        """Check for command injection patterns"""
        if not self.strict_mode:
            return False
        for pattern in self._compiled_cmd:
            if pattern.search(value):
                return True
        return False

    def validate(self, value: str) -> tuple[bool, Optional[str]]:
        """
        Validate input string for security issues.

        Returns:
            Tuple of (is_safe, threat_type)
        """
        if not isinstance(value, str):
            return True, None

        if self.check_sql_injection(value):
            return False, "sql_injection"

        if self.check_xss(value):
            return False, "xss"

        if self.check_path_traversal(value):
            return False, "path_traversal"

        if self.check_command_injection(value):
            return False, "command_injection"

        return True, None

    def sanitize(self, value: str) -> str:
        """Sanitize input string by escaping dangerous characters"""
        if not isinstance(value, str):
            return value

        # HTML escape
        value = value.replace("&", "&amp;")
        value = value.replace("<", "&lt;")
        value = value.replace(">", "&gt;")
        value = value.replace('"', "&quot;")
        value = value.replace("'", "&#x27;")

        return value

    def validate_json_body(self, data: Dict[str, Any]) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Recursively validate JSON body.

        Returns:
            Tuple of (is_safe, field_name, threat_type)
        """
        def check_value(val, path=""):
            if isinstance(val, str):
                is_safe, threat = self.validate(val)
                if not is_safe:
                    return False, path, threat
            elif isinstance(val, dict):
                for k, v in val.items():
                    is_safe, field, threat = check_value(v, f"{path}.{k}" if path else k)
                    if not is_safe:
                        return False, field, threat
            elif isinstance(val, list):
                for i, item in enumerate(val):
                    is_safe, field, threat = check_value(item, f"{path}[{i}]")
                    if not is_safe:
                        return False, field, threat
            return True, None, None

        return check_value(data)


# =============================================================================
# Security Headers Middleware
# =============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses.

    Headers added:
    - Content-Security-Policy
    - X-Content-Type-Options
    - X-Frame-Options
    - X-XSS-Protection
    - Strict-Transport-Security
    - Referrer-Policy
    - Permissions-Policy
    """

    def __init__(self, app, csp_policy: Optional[str] = None):
        super().__init__(app)
        self.csp_policy = csp_policy or self._default_csp()

    def _default_csp(self) -> str:
        """Generate default Content-Security-Policy"""
        return "; ".join([
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline'",  # Allow inline scripts for simple frontend
            "style-src 'self' 'unsafe-inline'",   # Allow inline styles
            "img-src 'self' data: blob:",
            "font-src 'self'",
            "connect-src 'self'",
            "frame-ancestors 'none'",
            "form-action 'self'",
            "base-uri 'self'",
        ])

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(self), camera=()"

        # CSP header
        response.headers["Content-Security-Policy"] = self.csp_policy

        # HSTS for HTTPS
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Prevent caching of sensitive data
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
            response.headers["Pragma"] = "no-cache"

        return response


# =============================================================================
# Rate Limit Middleware
# =============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware.

    Enforces rate limits on API endpoints.
    """

    def __init__(self, app, config: Optional[RateLimitConfig] = None):
        super().__init__(app)
        self.limiter = RateLimiter(config)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for static files
        if request.url.path.startswith("/static"):
            return await call_next(request)

        # Check if blocked
        if self.limiter.is_blocked(request):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Too many requests",
                    "message": "You have been temporarily blocked. Please try again later.",
                }
            )

        # Check rate limit
        is_allowed, error = self.limiter.check_rate_limit(request)
        if not is_allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": error,
                }
            )

        response = await call_next(request)

        # Add rate limit headers
        remaining = self.limiter.get_remaining(request)
        response.headers["X-RateLimit-Limit"] = str(self.limiter.config.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining["remaining_per_minute"])

        return response


# =============================================================================
# Input Validation Middleware
# =============================================================================

class InputValidationMiddleware(BaseHTTPMiddleware):
    """
    Input validation middleware.

    Validates request bodies for security threats.
    """

    def __init__(self, app, strict_mode: bool = False):
        super().__init__(app)
        self.validator = InputValidator(strict_mode)
        self.max_body_size = 10 * 1024 * 1024  # 10MB

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_body_size:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "error": "Request too large",
                    "message": f"Maximum request size is {self.max_body_size // 1024 // 1024}MB",
                }
            )

        # Validate JSON bodies for POST/PUT/PATCH
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")

            if "application/json" in content_type:
                try:
                    body = await request.json()

                    is_safe, field, threat = self.validator.validate_json_body(body)
                    if not is_safe:
                        logger.warning(f"Security threat detected: {threat} in field {field}")
                        return JSONResponse(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            content={
                                "error": "Invalid input",
                                "message": "Request contains potentially harmful content",
                                "field": field,
                            }
                        )
                except Exception:
                    pass  # Let the actual endpoint handle JSON parsing errors

        return await call_next(request)


# =============================================================================
# Request ID Middleware
# =============================================================================

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Add unique request ID to each request for tracing.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        import uuid

        # Generate or get request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Store in request state
        request.state.request_id = request_id

        response = await call_next(request)

        # Add to response
        response.headers["X-Request-ID"] = request_id

        return response


# =============================================================================
# Factory Functions
# =============================================================================

def create_rate_limiter(
    requests_per_minute: int = 240,
    requests_per_hour: int = 1000,
) -> RateLimiter:
    """Create a configured rate limiter"""
    config = RateLimitConfig(
        requests_per_minute=requests_per_minute,
        requests_per_hour=requests_per_hour,
    )
    return RateLimiter(config)


def create_input_validator(strict_mode: bool = False) -> InputValidator:
    """Create an input validator"""
    return InputValidator(strict_mode)
