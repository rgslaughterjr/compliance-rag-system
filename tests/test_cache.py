"""
Tests for QueryCache

Tests cache hit/miss logic, TTL expiration, LRU eviction,
and cache key generation for the query caching system.
"""
import pytest
import time
import hashlib

from src.cache import QueryCache, CacheEntry


class TestCacheKeyGeneration:
    """Test cache key generation from queries and filters"""

    def test_generate_key_basic(self, test_cache):
        """
        Test basic cache key generation

        Verifies that cache keys are generated consistently
        from query strings using MD5 hashing.
        """
        query = "What is GDPR Article 17?"
        key1 = test_cache._generate_key(query)
        key2 = test_cache._generate_key(query)

        # Same query should produce same key
        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) == 32  # MD5 hash length

    def test_generate_key_case_insensitive(self, test_cache):
        """
        Test that key generation is case insensitive

        Verifies that queries with different cases produce
        the same cache key.
        """
        key1 = test_cache._generate_key("What is GDPR?")
        key2 = test_cache._generate_key("what is gdpr?")
        key3 = test_cache._generate_key("WHAT IS GDPR?")

        assert key1 == key2 == key3

    def test_generate_key_strips_whitespace(self, test_cache):
        """
        Test that key generation strips whitespace

        Verifies that leading/trailing whitespace doesn't
        affect cache key generation.
        """
        key1 = test_cache._generate_key("  What is NIST?  ")
        key2 = test_cache._generate_key("What is NIST?")

        assert key1 == key2

    def test_generate_key_with_filters(self, test_cache):
        """
        Test cache key generation with metadata filters

        Verifies that filters are incorporated into cache keys
        so filtered queries are cached separately.
        """
        query = "data protection"
        filters1 = {"source": "gdpr.pdf"}
        filters2 = {"source": "nist.pdf"}

        key1 = test_cache._generate_key(query, filters1)
        key2 = test_cache._generate_key(query, filters2)
        key3 = test_cache._generate_key(query, None)

        # Different filters should produce different keys
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_generate_key_filter_order_invariant(self, test_cache):
        """
        Test that filter order doesn't affect cache key

        Verifies that filters with same content but different
        order produce the same cache key.
        """
        query = "test"
        filters1 = {"source": "a.pdf", "page": 1}
        filters2 = {"page": 1, "source": "a.pdf"}

        key1 = test_cache._generate_key(query, filters1)
        key2 = test_cache._generate_key(query, filters2)

        assert key1 == key2


class TestCacheHitMiss:
    """Test cache hit and miss logic"""

    def test_cache_miss_on_first_query(self, test_cache):
        """
        Test that first query results in cache miss

        Verifies that a new query returns None and increments
        miss counter.
        """
        initial_misses = test_cache.misses

        result = test_cache.get("What is GDPR?")

        assert result is None
        assert test_cache.misses == initial_misses + 1
        assert test_cache.hits == 0

    def test_cache_hit_on_repeated_query(self, test_cache):
        """
        Test that repeated query results in cache hit

        Verifies that after caching a query, subsequent identical
        queries return cached results.
        """
        query = "What is NIST framework?"
        results = ["doc1", "doc2", "doc3"]

        # Store in cache
        test_cache.set(query, results)

        # Retrieve from cache
        cached = test_cache.get(query)

        assert cached is not None
        assert cached[0] == results  # Results match
        assert cached[1] == "full"   # Mode is "full"
        assert cached[2] is None     # No error
        assert test_cache.hits == 1

    def test_cache_statistics_tracking(self, test_cache):
        """
        Test that cache statistics are accurately tracked

        Verifies that hits, misses, and total queries are
        correctly counted.
        """
        assert test_cache.total_queries == 0
        assert test_cache.hits == 0
        assert test_cache.misses == 0

        # First query - miss
        test_cache.get("query1")
        assert test_cache.total_queries == 1
        assert test_cache.misses == 1

        # Cache and retrieve - hit
        test_cache.set("query1", ["result"])
        test_cache.get("query1")
        assert test_cache.total_queries == 2
        assert test_cache.hits == 1
        assert test_cache.misses == 1

        # Another miss
        test_cache.get("query2")
        assert test_cache.total_queries == 3
        assert test_cache.misses == 2

    def test_hit_rate_calculation(self, test_cache):
        """
        Test cache hit rate percentage calculation

        Verifies that hit_rate property correctly calculates
        percentage of cache hits.
        """
        # Empty cache - 0% hit rate
        assert test_cache.hit_rate == 0.0

        # 1 miss, 0 hits - 0%
        test_cache.get("query1")
        assert test_cache.hit_rate == 0.0

        # Cache the query
        test_cache.set("query1", ["result"])

        # 1 hit, 1 miss - 50%
        test_cache.get("query1")
        assert test_cache.hit_rate == 50.0

        # 2 hits, 1 miss - 66.67%
        test_cache.get("query1")
        assert abs(test_cache.hit_rate - 66.67) < 0.1

    def test_different_queries_cache_separately(self, test_cache):
        """
        Test that different queries are cached separately

        Verifies that each unique query has its own cache entry.
        """
        test_cache.set("query1", ["result1"], mode="full")
        test_cache.set("query2", ["result2"], mode="cache")

        cached1 = test_cache.get("query1")
        cached2 = test_cache.get("query2")

        assert cached1[0] == ["result1"]
        assert cached1[1] == "full"

        assert cached2[0] == ["result2"]
        assert cached2[1] == "cache"


class TestTTLExpiration:
    """Test TTL (Time To Live) expiration logic"""

    def test_cache_entry_not_expired_immediately(self):
        """
        Test that cache entries are not expired immediately

        Verifies that newly created cache entries are valid.
        """
        entry = CacheEntry(results=["doc1"], timestamp=time.time(), mode="full")

        assert not entry.is_expired(ttl_seconds=3600)

    def test_cache_entry_expires_after_ttl(self, test_cache_short_ttl):
        """
        Test that cache entries expire after TTL

        Verifies that cache entries are considered expired
        after the configured TTL period.
        """
        query = "test query"
        test_cache_short_ttl.set(query, ["result"])

        # Should be cached initially
        cached = test_cache_short_ttl.get(query)
        assert cached is not None

        # Wait for TTL to expire
        time.sleep(1.5)

        # Should be expired now
        cached = test_cache_short_ttl.get(query)
        assert cached is None
        assert test_cache_short_ttl.misses == 1  # Expired get counts as miss

    def test_expired_entries_are_deleted(self, test_cache_short_ttl):
        """
        Test that expired entries are removed from cache

        Verifies that when an expired entry is accessed,
        it is deleted from the cache.
        """
        query = "test query"
        test_cache_short_ttl.set(query, ["result"])

        initial_size = len(test_cache_short_ttl)
        assert initial_size == 1

        # Wait for expiration
        time.sleep(1.5)

        # Access expired entry
        test_cache_short_ttl.get(query)

        # Should be removed
        assert len(test_cache_short_ttl) == 0

    def test_ttl_is_configurable(self):
        """
        Test that TTL is configurable on cache creation

        Verifies that different TTL values can be set for
        different cache instances.
        """
        cache_short = QueryCache(ttl_seconds=1)
        cache_long = QueryCache(ttl_seconds=3600)

        assert cache_short.ttl_seconds == 1
        assert cache_long.ttl_seconds == 3600


class TestLRUEviction:
    """Test LRU (Least Recently Used) eviction logic"""

    def test_eviction_when_at_capacity(self):
        """
        Test that oldest entry is evicted when cache is full

        Verifies LRU eviction policy removes the oldest entry
        when max_size is reached.
        """
        cache = QueryCache(max_size=3)

        # Fill cache to capacity
        cache.set("query1", ["result1"])
        time.sleep(0.01)  # Ensure different timestamps
        cache.set("query2", ["result2"])
        time.sleep(0.01)
        cache.set("query3", ["result3"])

        assert len(cache) == 3

        # Add one more - should evict oldest (query1)
        cache.set("query4", ["result4"])

        assert len(cache) == 3
        assert cache.get("query1") is None  # Evicted
        assert cache.get("query2") is not None
        assert cache.get("query3") is not None
        assert cache.get("query4") is not None

    def test_evicts_by_timestamp(self):
        """
        Test that eviction is based on entry timestamp

        Verifies that the entry with the oldest timestamp
        is selected for eviction.
        """
        cache = QueryCache(max_size=2)

        # Add entries with clear time separation
        cache.set("old", ["result_old"])
        time.sleep(0.05)
        cache.set("new", ["result_new"])

        assert len(cache) == 2

        # Add another - "old" should be evicted
        cache.set("newest", ["result_newest"])

        assert len(cache) == 2
        assert cache.get("old") is None
        assert cache.get("new") is not None
        assert cache.get("newest") is not None

    def test_max_size_is_enforced(self):
        """
        Test that cache never exceeds max_size

        Verifies that regardless of insertions, cache size
        stays at or below max_size.
        """
        max_size = 5
        cache = QueryCache(max_size=max_size)

        # Add more entries than max_size
        for i in range(10):
            cache.set(f"query{i}", [f"result{i}"])
            time.sleep(0.01)

        assert len(cache) == max_size


class TestCacheMaintenance:
    """Test cache maintenance operations"""

    def test_clear_removes_all_entries(self, test_cache):
        """
        Test that clear() removes all cache entries

        Verifies that the clear method empties the cache
        completely.
        """
        # Add multiple entries
        test_cache.set("query1", ["result1"])
        test_cache.set("query2", ["result2"])
        test_cache.set("query3", ["result3"])

        assert len(test_cache) > 0

        # Clear cache
        test_cache.clear()

        assert len(test_cache) == 0
        assert test_cache.get("query1") is None
        assert test_cache.get("query2") is None
        assert test_cache.get("query3") is None

    def test_clear_preserves_statistics(self, test_cache):
        """
        Test that clear() preserves hit/miss statistics

        Verifies that clearing cache doesn't reset statistics
        counters.
        """
        # Generate some statistics
        test_cache.get("query1")  # Miss
        test_cache.set("query1", ["result"])
        test_cache.get("query1")  # Hit

        hits = test_cache.hits
        misses = test_cache.misses
        total = test_cache.total_queries

        # Clear cache
        test_cache.clear()

        # Statistics should be preserved
        assert test_cache.hits == hits
        assert test_cache.misses == misses
        assert test_cache.total_queries == total

    def test_cache_str_representation(self, test_cache):
        """
        Test cache string representation

        Verifies that __str__ method returns useful information
        about cache state.
        """
        test_cache.set("query1", ["result1"])
        test_cache.get("query1")  # Hit

        cache_str = str(test_cache)

        assert "QueryCache" in cache_str
        assert "size=1" in cache_str
        assert "hits=1" in cache_str
        assert "hit_rate" in cache_str

    def test_cache_len(self, test_cache):
        """
        Test cache length reporting

        Verifies that len() returns the number of cached entries.
        """
        assert len(test_cache) == 0

        test_cache.set("query1", ["result1"])
        assert len(test_cache) == 1

        test_cache.set("query2", ["result2"])
        assert len(test_cache) == 2

        test_cache.clear()
        assert len(test_cache) == 0


class TestCacheEntry:
    """Test CacheEntry dataclass"""

    def test_cache_entry_creation(self):
        """
        Test CacheEntry creation with all fields

        Verifies that CacheEntry objects store results,
        timestamp, and mode correctly.
        """
        results = ["doc1", "doc2"]
        timestamp = time.time()
        mode = "full"

        entry = CacheEntry(results=results, timestamp=timestamp, mode=mode)

        assert entry.results == results
        assert entry.timestamp == timestamp
        assert entry.mode == mode

    def test_cache_entry_default_mode(self):
        """
        Test CacheEntry default mode is 'full'

        Verifies that mode defaults to 'full' if not specified.
        """
        entry = CacheEntry(results=[], timestamp=time.time())
        assert entry.mode == "full"

    def test_cache_entry_is_expired(self):
        """
        Test CacheEntry expiration checking

        Verifies that is_expired correctly determines if entry
        has exceeded TTL.
        """
        # Recent entry - not expired
        entry_new = CacheEntry(results=[], timestamp=time.time())
        assert not entry_new.is_expired(ttl_seconds=10)

        # Old entry - expired
        entry_old = CacheEntry(results=[], timestamp=time.time() - 20)
        assert entry_old.is_expired(ttl_seconds=10)
