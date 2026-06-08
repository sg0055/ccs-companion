"""Cache manager for stored embeddings and retrieval state."""

import os
from typing import Any


class CacheManager:
    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get(self, key: str) -> Any:
        """Fetch an item from cache."""
        return None

    def set(self, key: str, value: Any) -> None:
        """Store an item in cache."""
        pass
# src/caching/cache_manager.py
"""
Multi-level caching for performance optimization.
"""

from dataclasses import dataclass
from typing import Optional, Any
import hashlib
import json
import time
from functools import wraps
import redis
from loguru import logger

from src.config import settings


@dataclass
class CacheEntry:
    """Cached item with metadata."""
    key: str
    value: Any
    created_at: float
    ttl: int
    hits: int = 0


class CacheManager:
    """
    Multi-level cache with LRU eviction.
    
    Levels:
    1. L1: In-memory LRU cache (fastest)
    2. L2: Redis cache (shared across instances)
    
    Cached items:
    - Query embeddings
    - Retrieval results
    - Generated responses (careful with staleness)
    """
    
    def __init__(
        self,
        redis_url: str = None,
        max_memory_items: int = 1000,
        default_ttl: int = 3600
    ):
        self.redis_url = redis_url or settings.cache.redis_url
        self.max_memory_items = max_memory_items
        self.default_ttl = default_ttl
        
        # L1: Memory cache
        self._memory_cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []  # For LRU
        
        # L2: Redis cache
        self._redis: Optional[redis.Redis] = None
        if settings.cache.enabled and self.redis_url:
            try:
                self._redis = redis.from_url(self.redis_url)
                self._redis.ping()
                logger.info("Redis cache connected")
            except redis.ConnectionError:
                logger.warning("Redis not available, using memory-only cache")
                self._redis = None
                
    def _compute_key(self, prefix: str, *args) -> str:
        """Compute cache key from inputs."""
        key_data = json.dumps(args, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache (tries L1, then L2)."""
        # L1: Memory
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            # Check TTL
            if time.time() - entry.created_at < entry.ttl:
                entry.hits += 1
                self._touch(key)
                return entry.value
            else:
                # Expired
                del self._memory_cache[key]
                
        # L2: Redis
        if self._redis:
            data = self._redis.get(key)
            if data:
                value = json.loads(data)
                # Populate L1
                self._set_memory(key, value, self.default_ttl)
                return value
                
        return None
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: int = None
    ) -> None:
        """Set item in both cache levels."""
        ttl = ttl or self.default_ttl
        
        # L1
        self._set_memory(key, value, ttl)
        
        # L2
        if self._redis:
            try:
                self._redis.setex(key, ttl, json.dumps(value, default=str))
            except redis.RedisError as e:
                logger.warning(f"Redis set failed: {e}")
                
    def _set_memory(self, key: str, value: Any, ttl: int):
        """Set in memory cache with LRU eviction."""
        # Evict if necessary
        while len(self._memory_cache) >= self.max_memory_items:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._memory_cache:
                del self._memory_cache[oldest_key]
                
        self._memory_cache[key] = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            ttl=ttl
        )
        self._access_order.append(key)
        
    def _touch(self, key: str):
        """Update access order for LRU."""
        if key in self._access_order:
            self._access_order.remove(key)
            self._access_order.append(key)
            
    def invalidate(self, pattern: str = None):
        """Invalidate cache entries."""
        if pattern:
            # Invalidate matching keys
            keys_to_remove = [
                k for k in self._memory_cache 
                if pattern in k
            ]
            for key in keys_to_remove:
                del self._memory_cache[key]
                
            if self._redis:
                for key in self._redis.scan_iter(f"*{pattern}*"):
                    self._redis.delete(key)
        else:
            # Clear all
            self._memory_cache.clear()
            self._access_order.clear()
            if self._redis:
                self._redis.flushdb()
                
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_hits = sum(e.hits for e in self._memory_cache.values())
        return {
            'memory_items': len(self._memory_cache),
            'total_hits': total_hits,
            'redis_connected': self._redis is not None
        }


def cached(prefix: str, ttl: int = 3600):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, 'cache') or self.cache is None:
                return func(self, *args, **kwargs)
                
            # Compute cache key
            key = self.cache._compute_key(prefix, args, kwargs)
            
            # Try cache
            cached_result = self.cache.get(key)
            if cached_result is not None:
                return cached_result
                
            # Execute and cache
            result = func(self, *args, **kwargs)
            self.cache.set(key, result, ttl)
            return result
            
        return wrapper
    return decorator
