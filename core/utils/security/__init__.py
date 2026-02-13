"""
=============================================================================
HUMMINGBIRD-LEA - Security Module
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Security utilities for the application.

Components:
- Middleware: Rate limiting, input validation, security headers
- Errors: Custom exceptions, error handling, recovery
- Cache: LRU caching with TTL support
=============================================================================
"""

# Middleware components
from .middleware import (
    # Rate limiting
    RateLimiter,
    RateLimitConfig,
    RateLimitMiddleware,
    create_rate_limiter,
    # Input validation
    InputValidator,
    InputValidationMiddleware,
    create_input_validator,
    # Security headers
    SecurityHeadersMiddleware,
    # Request ID
    RequestIDMiddleware,
)

# Error handling
from .errors import (
    # Categories and severity
    ErrorCategory,
    ErrorSeverity,
    # Custom exceptions
    HummingbirdError,
    ValidationError_,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    AIServiceError,
    StorageError,
    ConfigurationError,
    # Response builder
    ErrorResponse,
    build_error_response,
    # Logging
    log_error,
    # Exception handlers
    setup_exception_handlers,
    # Recovery
    ErrorRecovery,
)

# Caching
from .cache import (
    # Cache classes
    LRUCache,
    CacheEntry,
    CacheStats,
    ResponseCache,
    EmbeddingCache,
    # Decorators
    cached,
    async_cached,
    # Singletons
    get_response_cache,
    get_embedding_cache,
    get_general_cache,
    reset_caches,
    # Utilities
    get_cache_stats,
    clear_all_caches,
)

__all__ = [
    # Middleware
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitMiddleware",
    "create_rate_limiter",
    "InputValidator",
    "InputValidationMiddleware",
    "create_input_validator",
    "SecurityHeadersMiddleware",
    "RequestIDMiddleware",
    # Errors
    "ErrorCategory",
    "ErrorSeverity",
    "HummingbirdError",
    "ValidationError_",
    "AuthenticationError",
    "NotFoundError",
    "RateLimitError",
    "AIServiceError",
    "StorageError",
    "ConfigurationError",
    "ErrorResponse",
    "build_error_response",
    "log_error",
    "setup_exception_handlers",
    "ErrorRecovery",
    # Caching
    "LRUCache",
    "CacheEntry",
    "CacheStats",
    "ResponseCache",
    "EmbeddingCache",
    "cached",
    "async_cached",
    "get_response_cache",
    "get_embedding_cache",
    "get_general_cache",
    "reset_caches",
    "get_cache_stats",
    "clear_all_caches",
]
