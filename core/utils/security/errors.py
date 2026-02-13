"""
=============================================================================
HUMMINGBIRD-LEA - Error Handling System
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Global error handling with structured responses and logging.

Features:
- Custom exception classes
- Global exception handler
- Structured error responses
- Error logging with context
- User-friendly messages
- Graceful degradation
=============================================================================
"""

import logging
import traceback
import sys
from datetime import datetime
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

logger = logging.getLogger(__name__)


# =============================================================================
# Error Categories
# =============================================================================

class ErrorCategory(Enum):
    """Categories of errors for classification"""
    VALIDATION = "validation"       # Input validation errors
    AUTHENTICATION = "auth"         # Auth/permission errors
    NOT_FOUND = "not_found"         # Resource not found
    RATE_LIMIT = "rate_limit"       # Rate limiting
    SERVER = "server"               # Internal server errors
    AI_SERVICE = "ai_service"       # AI/Ollama errors
    STORAGE = "storage"             # File/database errors
    NETWORK = "network"             # Network/connectivity errors
    CONFIGURATION = "config"        # Configuration errors
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Severity levels for errors"""
    LOW = "low"           # Minor issues, no action needed
    MEDIUM = "medium"     # Should be investigated
    HIGH = "high"         # Needs attention
    CRITICAL = "critical" # Immediate action required


# =============================================================================
# Custom Exceptions
# =============================================================================

class HummingbirdError(Exception):
    """Base exception for Hummingbird-LEA"""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.status_code = status_code
        self.details = details or {}
        self.user_message = user_message or self._default_user_message()
        self.timestamp = datetime.utcnow()

    def _default_user_message(self) -> str:
        """Generate user-friendly message based on category"""
        messages = {
            ErrorCategory.VALIDATION: "The provided data is invalid. Please check your input.",
            ErrorCategory.AUTHENTICATION: "Authentication required or access denied.",
            ErrorCategory.NOT_FOUND: "The requested resource was not found.",
            ErrorCategory.RATE_LIMIT: "Too many requests. Please slow down.",
            ErrorCategory.SERVER: "An internal error occurred. Please try again later.",
            ErrorCategory.AI_SERVICE: "The AI service is temporarily unavailable.",
            ErrorCategory.STORAGE: "A storage error occurred. Please try again.",
            ErrorCategory.NETWORK: "A network error occurred. Please check your connection.",
            ErrorCategory.CONFIGURATION: "A configuration error occurred.",
            ErrorCategory.UNKNOWN: "An unexpected error occurred.",
        }
        return messages.get(self.category, messages[ErrorCategory.UNKNOWN])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "error": True,
            "category": self.category.value,
            "message": self.user_message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class ValidationError_(HummingbirdError):
    """Validation error"""
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        details = {"field": field} if field else {}
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            status_code=400,
            details=details,
            **kwargs,
        )


class AuthenticationError(HummingbirdError):
    """Authentication/authorization error"""
    def __init__(self, message: str = "Authentication required", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.MEDIUM,
            status_code=401,
            **kwargs,
        )


class NotFoundError(HummingbirdError):
    """Resource not found error"""
    def __init__(self, resource: str = "Resource", **kwargs):
        super().__init__(
            f"{resource} not found",
            category=ErrorCategory.NOT_FOUND,
            severity=ErrorSeverity.LOW,
            status_code=404,
            user_message=f"The requested {resource.lower()} was not found.",
            **kwargs,
        )


class RateLimitError(HummingbirdError):
    """Rate limit exceeded error"""
    def __init__(self, retry_after: Optional[int] = None, **kwargs):
        details = {"retry_after_seconds": retry_after} if retry_after else {}
        super().__init__(
            "Rate limit exceeded",
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.LOW,
            status_code=429,
            details=details,
            **kwargs,
        )


class AIServiceError(HummingbirdError):
    """AI service (Ollama) error"""
    def __init__(self, message: str = "AI service unavailable", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AI_SERVICE,
            severity=ErrorSeverity.HIGH,
            status_code=503,
            user_message="The AI assistant is temporarily unavailable. Please try again in a moment.",
            **kwargs,
        )


class StorageError(HummingbirdError):
    """Storage/database error"""
    def __init__(self, message: str = "Storage error", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.STORAGE,
            severity=ErrorSeverity.HIGH,
            status_code=500,
            **kwargs,
        )


class ConfigurationError(HummingbirdError):
    """Configuration error"""
    def __init__(self, message: str = "Configuration error", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            status_code=500,
            **kwargs,
        )


# =============================================================================
# Error Response Builder
# =============================================================================

@dataclass
class ErrorResponse:
    """Structured error response"""
    error: bool = True
    status_code: int = 500
    category: str = "unknown"
    message: str = "An error occurred"
    user_message: str = "An unexpected error occurred. Please try again."
    details: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "error": self.error,
            "category": self.category,
            "message": self.user_message,
            "timestamp": self.timestamp,
        }

        if self.request_id:
            result["request_id"] = self.request_id

        if self.details:
            result["details"] = self.details

        return result


def build_error_response(
    exception: Exception,
    request: Optional[Request] = None,
    include_traceback: bool = False,
) -> tuple[int, Dict[str, Any]]:
    """
    Build a structured error response from an exception.

    Args:
        exception: The exception that occurred
        request: The request context
        include_traceback: Whether to include traceback (dev only)

    Returns:
        Tuple of (status_code, response_dict)
    """
    request_id = None
    if request and hasattr(request.state, "request_id"):
        request_id = request.state.request_id

    # Handle our custom exceptions
    if isinstance(exception, HummingbirdError):
        response = ErrorResponse(
            status_code=exception.status_code,
            category=exception.category.value,
            message=str(exception),
            user_message=exception.user_message,
            details=exception.details,
            request_id=request_id,
        )
        return exception.status_code, response.to_dict()

    # Handle FastAPI HTTPException
    if isinstance(exception, HTTPException):
        category = ErrorCategory.SERVER
        if exception.status_code == 401:
            category = ErrorCategory.AUTHENTICATION
        elif exception.status_code == 403:
            category = ErrorCategory.AUTHENTICATION
        elif exception.status_code == 404:
            category = ErrorCategory.NOT_FOUND
        elif exception.status_code == 429:
            category = ErrorCategory.RATE_LIMIT
        elif exception.status_code < 500:
            category = ErrorCategory.VALIDATION

        response = ErrorResponse(
            status_code=exception.status_code,
            category=category.value,
            message=str(exception.detail),
            user_message=str(exception.detail),
            request_id=request_id,
        )
        return exception.status_code, response.to_dict()

    # Handle Pydantic validation errors
    if isinstance(exception, (RequestValidationError, ValidationError)):
        errors = []
        if hasattr(exception, "errors"):
            for error in exception.errors():
                loc = ".".join(str(l) for l in error.get("loc", []))
                errors.append({
                    "field": loc,
                    "message": error.get("msg", "Invalid value"),
                })

        response = ErrorResponse(
            status_code=422,
            category=ErrorCategory.VALIDATION.value,
            message="Validation error",
            user_message="The provided data is invalid. Please check your input.",
            details={"validation_errors": errors},
            request_id=request_id,
        )
        return 422, response.to_dict()

    # Handle generic exceptions
    response = ErrorResponse(
        status_code=500,
        category=ErrorCategory.SERVER.value,
        message=str(exception) if include_traceback else "Internal server error",
        user_message="An unexpected error occurred. Please try again later.",
        request_id=request_id,
    )

    if include_traceback:
        response.details["traceback"] = traceback.format_exc()

    return 500, response.to_dict()


# =============================================================================
# Error Logging
# =============================================================================

def log_error(
    exception: Exception,
    request: Optional[Request] = None,
    additional_context: Optional[Dict[str, Any]] = None,
):
    """
    Log an error with context.

    Args:
        exception: The exception to log
        request: The request context
        additional_context: Additional context to log
    """
    context = additional_context or {}

    # Add request context
    if request:
        context.update({
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host if request.client else "unknown",
        })

        if hasattr(request.state, "request_id"):
            context["request_id"] = request.state.request_id

    # Determine severity
    severity = ErrorSeverity.MEDIUM
    if isinstance(exception, HummingbirdError):
        severity = exception.severity

    # Log based on severity
    log_message = f"{type(exception).__name__}: {str(exception)}"
    if context:
        log_message += f" | Context: {context}"

    if severity == ErrorSeverity.CRITICAL:
        logger.critical(log_message, exc_info=True)
    elif severity == ErrorSeverity.HIGH:
        logger.error(log_message, exc_info=True)
    elif severity == ErrorSeverity.MEDIUM:
        logger.warning(log_message)
    else:
        logger.info(log_message)


# =============================================================================
# Exception Handlers
# =============================================================================

def setup_exception_handlers(app: FastAPI, debug: bool = False):
    """
    Set up global exception handlers for FastAPI app.

    Args:
        app: FastAPI application
        debug: Whether to include debug info in responses
    """

    @app.exception_handler(HummingbirdError)
    async def hummingbird_error_handler(request: Request, exc: HummingbirdError):
        """Handle custom Hummingbird exceptions"""
        log_error(exc, request)
        status_code, response = build_error_response(exc, request, debug)
        return JSONResponse(status_code=status_code, content=response)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle FastAPI HTTP exceptions"""
        if exc.status_code >= 500:
            log_error(exc, request)
        status_code, response = build_error_response(exc, request, debug)
        return JSONResponse(status_code=status_code, content=response)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors"""
        status_code, response = build_error_response(exc, request, debug)
        return JSONResponse(status_code=status_code, content=response)

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all unhandled exceptions"""
        log_error(exc, request)
        status_code, response = build_error_response(exc, request, debug)
        return JSONResponse(status_code=status_code, content=response)


# =============================================================================
# Error Recovery
# =============================================================================

class ErrorRecovery:
    """
    Utilities for graceful error recovery and degradation.
    """

    @staticmethod
    async def with_fallback(
        primary_func,
        fallback_func,
        *args,
        **kwargs,
    ):
        """
        Execute primary function with fallback on error.

        Args:
            primary_func: Primary async function to execute
            fallback_func: Fallback async function if primary fails
            *args, **kwargs: Arguments to pass to functions

        Returns:
            Result from primary or fallback function
        """
        try:
            return await primary_func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary function failed, using fallback: {e}")
            return await fallback_func(*args, **kwargs)

    @staticmethod
    async def with_retry(
        func,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: tuple = (Exception,),
        *args,
        **kwargs,
    ):
        """
        Execute function with retries on failure.

        Args:
            func: Async function to execute
            max_retries: Maximum number of retries
            delay: Initial delay between retries
            backoff: Backoff multiplier for delay
            exceptions: Exception types to retry on
            *args, **kwargs: Arguments to pass to function

        Returns:
            Result from function
        """
        import asyncio

        last_exception = None
        current_delay = delay

        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {current_delay}s: {e}")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

        raise last_exception

    @staticmethod
    def safe_execute(func, default=None, *args, **kwargs):
        """
        Execute function safely, returning default on error.

        Args:
            func: Function to execute
            default: Default value on error
            *args, **kwargs: Arguments to pass to function

        Returns:
            Result or default value
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Safe execute caught error: {e}")
            return default
