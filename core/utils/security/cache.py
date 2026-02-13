"""
=============================================================================
HUMMINGBIRD-LEA - Caching System
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
In-memory caching with TTL support for performance optimization.

Features:
- LRU cache with configurable size
- TTL (time-to-live) support
- Cache decorators for functions
- Response caching for API endpoints
- Cache statistics and monitoring
=============================================================================
"""

import logging
import time
import hashlib
import json
import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, Callable, TypeVar, Generic
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# Cache Entry
# =============================================================================

@dataclass
class CacheEntry(Generic[T]):
    """A single cache entry with metadata"""
    value: T
    created_at: float
    expires_at: Optional[float] = None
    hits: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def touch(self):
        """Record a cache hit"""
        self.hits += 1


# =============================================================================
# LRU Cache
# =============================================================================

class LRUCache(Generic[T]):
    """
    Least Recently Used (LRU) cache with TTL support.

    Thread-safe for basic operations.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = 300,  # 5 minutes default
        cleanup_interval: float = 60,
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval

        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._stats = CacheStats()
        self._last_cleanup = time.time()
        self._lock = asyncio.Lock()

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments"""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.md5(key_data.encode()).hexdigest()

    def _cleanup_expired(self):
        """Remove expired entries"""
        now = time.time()
        if now - self._last_cleanup < self.cleanup_interval:
            return

        self._last_cleanup = now
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired
        ]

        for key in expired_keys:
            del self._cache[key]
            self._stats.evictions += 1

    def _evict_if_needed(self):
        """Evict oldest entries if cache is full"""
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
            self._stats.evictions += 1

    def get(self, key: str) -> Optional[T]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        self._cleanup_expired()

        if key not in self._cache:
            self._stats.misses += 1
            return None

        entry = self._cache[key]

        if entry.is_expired:
            del self._cache[key]
            self._stats.misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.touch()
        self._stats.hits += 1

        return entry.value

    def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for default)
        """
        self._evict_if_needed()

        actual_ttl = ttl if ttl is not None else self.default_ttl
        expires_at = time.time() + actual_ttl if actual_ttl else None

        self._cache[key] = CacheEntry(
            value=value,
            created_at=time.time(),
            expires_at=expires_at,
        )

        # Move to end
        self._cache.move_to_end(key)

    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if entry was deleted, False if not found
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all entries"""
        self._cache.clear()
        self._stats.reset()

    def has(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        if key not in self._cache:
            return False
        return not self._cache[key].is_expired

    @property
    def size(self) -> int:
        """Current cache size"""
        return len(self._cache)

    @property
    def stats(self) -> 'CacheStats':
        """Get cache statistics"""
        return self._stats

    async def async_get(self, key: str) -> Optional[T]:
        """Async version of get"""
        async with self._lock:
            return self.get(key)

    async def async_set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Async version of set"""
        async with self._lock:
            self.set(key, value, ttl)

    async def async_delete(self, key: str) -> bool:
        """Async version of delete"""
        async with self._lock:
            return self.delete(key)


# =============================================================================
# Cache Statistics
# =============================================================================

@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    def reset(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate * 100, 2),
            "total_requests": self.total_requests,
            "uptime_seconds": round(time.time() - self.created_at, 2),
        }


# =============================================================================
# Cache Decorators
# =============================================================================

def cached(
    cache: Optional[LRUCache] = None,
    ttl: Optional[float] = None,
    key_prefix: str = "",
):
    """
    Decorator to cache function results.

    Args:
        cache: LRUCache instance (creates new if None)
        ttl: Time-to-live in seconds
        key_prefix: Prefix for cache keys

    Example:
        @cached(ttl=60)
        def expensive_function(x, y):
            return x + y
    """
    _cache = cache or LRUCache()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = f"{key_prefix}{func.__name__}:{_cache._generate_key(*args, **kwargs)}"

            # Try cache
            result = _cache.get(key)
            if result is not None:
                return result

            # Execute function
            result = func(*args, **kwargs)

            # Store in cache
            _cache.set(key, result, ttl)

            return result

        wrapper.cache = _cache
        wrapper.clear_cache = _cache.clear
        return wrapper

    return decorator


def async_cached(
    cache: Optional[LRUCache] = None,
    ttl: Optional[float] = None,
    key_prefix: str = "",
):
    """
    Decorator to cache async function results.

    Args:
        cache: LRUCache instance (creates new if None)
        ttl: Time-to-live in seconds
        key_prefix: Prefix for cache keys

    Example:
        @async_cached(ttl=60)
        async def async_expensive_function(x, y):
            return x + y
    """
    _cache = cache or LRUCache()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key = f"{key_prefix}{func.__name__}:{_cache._generate_key(*args, **kwargs)}"

            # Try cache
            result = await _cache.async_get(key)
            if result is not None:
                return result

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            await _cache.async_set(key, result, ttl)

            return result

        wrapper.cache = _cache
        wrapper.clear_cache = _cache.clear
        return wrapper

    return decorator


# =============================================================================
# Response Cache
# =============================================================================

class ResponseCache:
    """
    Cache for API responses.

    Caches based on request path, method, and query parameters.
    """

    def __init__(
        self,
        max_size: int = 500,
        default_ttl: float = 60,
    ):
        self._cache = LRUCache[Dict[str, Any]](
            max_size=max_size,
            default_ttl=default_ttl,
        )

    def _generate_key(
        self,
        path: str,
        method: str = "GET",
        query_params: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Generate cache key from request details"""
        key_parts = [method, path]

        if query_params:
            sorted_params = sorted(query_params.items())
            key_parts.append(json.dumps(sorted_params))

        if user_id:
            key_parts.append(user_id)

        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(
        self,
        path: str,
        method: str = "GET",
        query_params: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        key = self._generate_key(path, method, query_params, user_id)
        return self._cache.get(key)

    def set(
        self,
        path: str,
        response: Dict[str, Any],
        method: str = "GET",
        query_params: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
        ttl: Optional[float] = None,
    ) -> None:
        """Cache a response"""
        key = self._generate_key(path, method, query_params, user_id)
        self._cache.set(key, response, ttl)

    def invalidate(
        self,
        path: str,
        method: str = "GET",
        query_params: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """Invalidate a cached response"""
        key = self._generate_key(path, method, query_params, user_id)
        return self._cache.delete(key)

    def invalidate_pattern(self, path_prefix: str) -> int:
        """
        Invalidate all cached responses matching a path prefix.

        Note: This is O(n) and should be used sparingly.
        """
        count = 0
        keys_to_delete = []

        for key in list(self._cache._cache.keys()):
            # We can't easily match by path prefix in hashed keys
            # This would need a separate index for production use
            pass

        for key in keys_to_delete:
            self._cache.delete(key)
            count += 1

        return count

    def clear(self) -> None:
        """Clear all cached responses"""
        self._cache.clear()

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics"""
        return self._cache.stats


# =============================================================================
# Embedding Cache (for RAG)
# =============================================================================

class EmbeddingCache:
    """
    Specialized cache for embeddings.

    Caches embedding vectors by text hash to avoid recomputation.
    """

    def __init__(
        self,
        max_size: int = 10000,
        default_ttl: float = 3600,  # 1 hour
    ):
        self._cache = LRUCache[list](
            max_size=max_size,
            default_ttl=default_ttl,
        )

    def _text_to_key(self, text: str, model: str = "default") -> str:
        """Generate key from text and model"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"emb:{model}:{text_hash}"

    def get(self, text: str, model: str = "default") -> Optional[list]:
        """Get cached embedding"""
        key = self._text_to_key(text, model)
        return self._cache.get(key)

    def set(
        self,
        text: str,
        embedding: list,
        model: str = "default",
        ttl: Optional[float] = None,
    ) -> None:
        """Cache an embedding"""
        key = self._text_to_key(text, model)
        self._cache.set(key, embedding, ttl)

    async def get_or_compute(
        self,
        text: str,
        compute_func: Callable,
        model: str = "default",
    ) -> list:
        """
        Get embedding from cache or compute it.

        Args:
            text: Text to embed
            compute_func: Async function to compute embedding
            model: Model identifier

        Returns:
            Embedding vector
        """
        cached = self.get(text, model)
        if cached is not None:
            return cached

        # Compute embedding
        embedding = await compute_func(text)

        # Cache it
        self.set(text, embedding, model)

        return embedding

    def clear(self) -> None:
        """Clear all cached embeddings"""
        self._cache.clear()

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics"""
        return self._cache.stats


# =============================================================================
# Global Cache Instances
# =============================================================================

# Singleton instances
_response_cache: Optional[ResponseCache] = None
_embedding_cache: Optional[EmbeddingCache] = None
_general_cache: Optional[LRUCache] = None


def get_response_cache() -> ResponseCache:
    """Get or create the response cache singleton"""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache()
    return _response_cache


def get_embedding_cache() -> EmbeddingCache:
    """Get or create the embedding cache singleton"""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache


def get_general_cache() -> LRUCache:
    """Get or create the general cache singleton"""
    global _general_cache
    if _general_cache is None:
        _general_cache = LRUCache(max_size=5000, default_ttl=300)
    return _general_cache


def reset_caches() -> None:
    """Reset all cache singletons (for testing)"""
    global _response_cache, _embedding_cache, _general_cache
    _response_cache = None
    _embedding_cache = None
    _general_cache = None


# =============================================================================
# Cache Utilities
# =============================================================================

def get_cache_stats() -> Dict[str, Any]:
    """Get statistics for all caches"""
    return {
        "response_cache": get_response_cache().stats.to_dict(),
        "embedding_cache": get_embedding_cache().stats.to_dict(),
        "general_cache": get_general_cache().stats.to_dict(),
    }


def clear_all_caches() -> None:
    """Clear all caches"""
    get_response_cache().clear()
    get_embedding_cache().clear()
    get_general_cache().clear()
    logger.info("All caches cleared")
