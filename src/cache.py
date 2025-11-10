"""
Query cache with TTL and LRU eviction
Optimized configuration from Lab 3.4+
"""
import hashlib
import time
from typing import Optional, List, Any
from dataclasses import dataclass
from datetime import datetime

from .config import CACHE_TTL_SECONDS, MAX_CACHE_SIZE


@dataclass
class CacheEntry:
    """Cache entry with results and metadata"""
    results: List[Any]
    timestamp: float
    mode: str = "full"

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if entry is expired"""
        age = time.time() - self.timestamp
        return age > ttl_seconds


class QueryCache:
    """
    In-memory query cache with TTL and LRU eviction
    
    Features:
    - Time-To-Live (TTL) expiration
    - LRU eviction when size limit reached
    - Cache statistics tracking
    """

    def __init__(self, ttl_seconds: int = CACHE_TTL_SECONDS,
                 max_size: int = MAX_CACHE_SIZE):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache = {}

        # Statistics
        self.hits = 0
        self.misses = 0
        self.total_queries = 0

    def _generate_key(self, query: str, filters: dict = None) -> str:
        """Generate cache key from query and filters"""
        key_str = query.lower().strip()
        if filters:
            key_str += str(sorted(filters.items()))
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query: str, filters: dict = None) -> Optional[tuple]:
        """
        Retrieve cached results
        
        Returns:
            (results, mode, None) if cache hit
            None if cache miss
        """
        self.total_queries += 1

        key = self._generate_key(query, filters)

        if key in self.cache:
            entry = self.cache[key]

            # Check if expired
            if entry.is_expired(self.ttl_seconds):
                del self.cache[key]
                self.misses += 1
                return None

            # Cache hit
            self.hits += 1
            return (entry.results, entry.mode, None)

        # Cache miss
        self.misses += 1
        return None

    def set(self, query: str, results: List[Any], mode: str = "full",
            filters: dict = None):
        """Cache query results"""
        key = self._generate_key(query, filters)

        # Evict oldest entry if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.items(),
                           key=lambda x: x[1].timestamp)[0]
            del self.cache[oldest_key]

        # Store new entry
        self.cache[key] = CacheEntry(
            results=results,
            timestamp=time.time(),
            mode=mode
        )

    def clear(self):
        """Clear all cached entries"""
        self.cache.clear()

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate percentage"""
        if self.total_queries == 0:
            return 0.0
        return (self.hits / self.total_queries) * 100

    def __len__(self):
        return len(self.cache)

    def __str__(self):
        return f"QueryCache(size={len(self)}, hits={self.hits}, misses={self.misses}, hit_rate={self.hit_rate:.1f}%)"
