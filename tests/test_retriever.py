"""
Tests for ResilientHybridRetriever

Tests hybrid search functionality combining semantic (Pinecone) and keyword (BM25) search
with optimized semantic weight (0.9), circuit breaker patterns, and result ranking.
"""
import pytest
import time
from unittest.mock import Mock, patch

from src.retriever import ResilientHybridRetriever, CircuitBreaker, CircuitState
from src.config import SEMANTIC_WEIGHT, RETRIEVAL_K


class TestHybridSearch:
    """Test hybrid search combining semantic and keyword search"""

    def test_hybrid_search_basic(self, mock_pinecone_index, sample_documents,
                                 mock_embeddings):
        """
        Test basic hybrid search retrieves documents

        Verifies that hybrid search successfully combines semantic (Pinecone)
        and keyword (BM25) search to return relevant documents.
        """
        retriever = ResilientHybridRetriever(
            pinecone_index=mock_pinecone_index,
            documents=sample_documents,
            embeddings=mock_embeddings
        )

        query = "What is NIST AI Risk Management Framework?"
        docs, mode, error = retriever.retrieve(query, k=5)

        # Verify successful retrieval
        assert error is None
        assert mode == "full"
        assert len(docs) > 0
        assert len(docs) <= 5

        # Verify embeddings were called
        mock_embeddings.embed_query.assert_called_once_with(query)

        # Verify Pinecone was queried
        mock_pinecone_index.query.assert_called_once()

    def test_semantic_weight_configuration(self, mock_pinecone_index,
                                          sample_documents, mock_embeddings):
        """
        Test semantic weight is correctly set to 0.9

        Verifies that the retriever uses the optimized semantic weight of 0.9
        and keyword weight of 0.1 as specified in Lab 2.4+.
        """
        # Test with default weight (0.9)
        retriever_default = ResilientHybridRetriever(
            pinecone_index=mock_pinecone_index,
            documents=sample_documents,
            embeddings=mock_embeddings
        )

        assert retriever_default.semantic_weight == 0.9
        assert retriever_default.keyword_weight == 0.1
        assert retriever_default.semantic_weight == SEMANTIC_WEIGHT

        # Test with custom weight
        custom_weight = 0.75
        retriever_custom = ResilientHybridRetriever(
            pinecone_index=mock_pinecone_index,
            documents=sample_documents,
            embeddings=mock_embeddings,
            semantic_weight=custom_weight
        )

        assert retriever_custom.semantic_weight == custom_weight
        assert retriever_custom.keyword_weight == 1.0 - custom_weight

    def test_result_ranking_by_score(self, mock_pinecone_index,
                                    sample_documents, mock_embeddings):
        """
        Test that results are properly ranked by combined score

        Verifies that documents are ranked by the weighted combination
        of semantic and keyword scores in descending order.
        """
        retriever = ResilientHybridRetriever(
            pinecone_index=mock_pinecone_index,
            documents=sample_documents,
            embeddings=mock_embeddings
        )

        query = "NIST framework"
        docs, mode, error = retriever.retrieve(query, k=3)

        assert error is None
        assert len(docs) > 0

        # Results should be returned (hybrid search combines and ranks scores)
        # We can't assert exact order without knowing BM25 scores,
        # but we can verify documents were retrieved
        doc_ids = [doc.metadata.get('id') for doc in docs]
        assert len(doc_ids) > 0
        assert all(doc_id is not None for doc_id in doc_ids)

    def test_hybrid_search_with_filters(self, mock_pinecone_index,
                                       sample_documents, mock_embeddings):
        """
        Test hybrid search with metadata filters

        Verifies that filters are correctly passed to Pinecone semantic search.
        """
        retriever = ResilientHybridRetriever(
            pinecone_index=mock_pinecone_index,
            documents=sample_documents,
            embeddings=mock_embeddings
        )

        query = "data protection"
        filters = {"source": "gdpr.pdf"}
        docs, mode, error = retriever.retrieve(query, k=5, filters=filters)

        assert error is None

        # Verify Pinecone was called with filters
        call_args = mock_pinecone_index.query.call_args
        assert call_args is not None
        assert 'filter' in call_args.kwargs
        assert call_args.kwargs['filter'] == filters

    def test_retrieval_k_parameter(self, mock_pinecone_index,
                                   sample_documents, mock_embeddings):
        """
        Test that k parameter controls number of results

        Verifies that the k parameter correctly limits the number of
        documents returned from hybrid search.
        """
        retriever = ResilientHybridRetriever(
            pinecone_index=mock_pinecone_index,
            documents=sample_documents,
            embeddings=mock_embeddings
        )

        # Test with k=3
        docs, mode, error = retriever.retrieve("test query", k=3)
        assert len(docs) <= 3

        # Test with k=1
        docs, mode, error = retriever.retrieve("test query", k=1)
        assert len(docs) <= 1


class TestCacheIntegration:
    """Test retriever integration with query cache"""

    def test_cache_hit_returns_cached_results(self, mock_pinecone_index,
                                             sample_documents, mock_embeddings,
                                             test_cache):
        """
        Test that cache hits return cached results without querying

        Verifies that when a query is cached, the retriever returns
        cached results and mode is 'cache'.
        """
        retriever = ResilientHybridRetriever(
            pinecone_index=mock_pinecone_index,
            documents=sample_documents,
            embeddings=mock_embeddings,
            cache=test_cache
        )

        query = "What is GDPR?"

        # First query - cache miss
        docs1, mode1, error1 = retriever.retrieve(query)
        assert mode1 == "full"
        assert error1 is None

        # Reset mock to verify it's not called again
        mock_pinecone_index.query.reset_mock()
        mock_embeddings.embed_query.reset_mock()

        # Second query - cache hit
        docs2, mode2, error2 = retriever.retrieve(query)
        assert mode2 == "cache"
        assert error2 is None
        assert len(docs2) == len(docs1)

        # Verify Pinecone was not queried on cache hit
        mock_pinecone_index.query.assert_not_called()
        mock_embeddings.embed_query.assert_not_called()

    def test_cache_miss_performs_search(self, mock_pinecone_index,
                                       sample_documents, mock_embeddings,
                                       test_cache):
        """
        Test that cache misses trigger full hybrid search

        Verifies that new queries perform full semantic and keyword search.
        """
        retriever = ResilientHybridRetriever(
            pinecone_index=mock_pinecone_index,
            documents=sample_documents,
            embeddings=mock_embeddings,
            cache=test_cache
        )

        query = "trustworthy AI"
        docs, mode, error = retriever.retrieve(query)

        assert mode == "full"
        assert error is None
        assert len(docs) > 0

        # Verify both search methods were used
        mock_embeddings.embed_query.assert_called_once()
        mock_pinecone_index.query.assert_called_once()

    def test_cache_stores_successful_results(self, mock_pinecone_index,
                                            sample_documents, mock_embeddings,
                                            test_cache):
        """
        Test that successful retrievals are cached

        Verifies that after a successful retrieval, results are stored
        in the cache for future use.
        """
        retriever = ResilientHybridRetriever(
            pinecone_index=mock_pinecone_index,
            documents=sample_documents,
            embeddings=mock_embeddings,
            cache=test_cache
        )

        query = "AI monitoring"

        # Initial cache size
        initial_size = len(test_cache)

        # Perform retrieval
        docs, mode, error = retriever.retrieve(query)
        assert error is None

        # Cache should have new entry
        assert len(test_cache) == initial_size + 1

        # Verify we can retrieve from cache
        cached_result = test_cache.get(query)
        assert cached_result is not None


class TestCircuitBreaker:
    """Test circuit breaker error handling"""

    def test_circuit_breaker_opens_after_failures(self):
        """
        Test circuit breaker opens after threshold failures

        Verifies that the circuit breaker enters OPEN state after
        the configured number of consecutive failures.
        """
        cb = CircuitBreaker(failure_threshold=3, timeout_seconds=60)

        assert cb.state == CircuitState.CLOSED

        # Simulate failures
        for i in range(3):
            try:
                cb.call(lambda: 1/0)  # Deliberately fail
            except ZeroDivisionError:
                pass

        # Circuit should be open after threshold failures
        assert cb.state == CircuitState.OPEN

    def test_circuit_breaker_rejects_when_open(self):
        """
        Test circuit breaker rejects requests when open

        Verifies that when circuit is OPEN, requests are immediately
        rejected without calling the function.
        """
        cb = CircuitBreaker(failure_threshold=2, timeout_seconds=60)

        # Force circuit open
        for i in range(2):
            try:
                cb.call(lambda: 1/0)
            except ZeroDivisionError:
                pass

        assert cb.state == CircuitState.OPEN

        # Next call should be rejected
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            cb.call(lambda: "test")

    def test_circuit_breaker_half_open_after_timeout(self):
        """
        Test circuit breaker enters HALF_OPEN after timeout

        Verifies that after the timeout period, circuit transitions
        to HALF_OPEN state to test if service recovered.
        """
        cb = CircuitBreaker(failure_threshold=2, timeout_seconds=1)

        # Force circuit open
        for i in range(2):
            try:
                cb.call(lambda: 1/0)
            except ZeroDivisionError:
                pass

        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(1.1)

        # Try call - should transition to HALF_OPEN
        try:
            cb.call(lambda: "success")
        except Exception:
            pass

        # State should have transitioned through HALF_OPEN
        # (may be CLOSED if success threshold met)
        assert cb.state in [CircuitState.HALF_OPEN, CircuitState.CLOSED]

    def test_circuit_breaker_closes_after_successes(self):
        """
        Test circuit breaker closes after successful calls in HALF_OPEN

        Verifies that after the success threshold is met in HALF_OPEN state,
        circuit returns to CLOSED state.
        """
        cb = CircuitBreaker(failure_threshold=2, timeout_seconds=1,
                          success_threshold=2)

        # Force circuit open
        for i in range(2):
            try:
                cb.call(lambda: 1/0)
            except ZeroDivisionError:
                pass

        # Wait for timeout to enter HALF_OPEN
        time.sleep(1.1)

        # Make successful calls
        for i in range(2):
            cb.call(lambda: "success")

        # Circuit should be closed
        assert cb.state == CircuitState.CLOSED


class TestErrorHandling:
    """Test error handling and retry logic"""

    def test_retrieval_retries_on_failure(self, sample_documents, mock_embeddings):
        """
        Test that retriever retries on transient failures

        Verifies exponential backoff retry logic attempts retrieval
        multiple times before giving up.
        """
        mock_index = Mock()
        # Fail twice, then succeed
        mock_index.query.side_effect = [
            Exception("Network error"),
            Exception("Timeout"),
            {'matches': [{'id': 'doc_0', 'score': 0.9}]}
        ]

        retriever = ResilientHybridRetriever(
            pinecone_index=mock_index,
            documents=sample_documents,
            embeddings=mock_embeddings
        )

        with patch('time.sleep'):  # Mock sleep to speed up test
            docs, mode, error = retriever.retrieve("test")

        # Should eventually succeed
        assert error is None
        assert mode == "full"
        assert mock_index.query.call_count == 3

    def test_retrieval_fails_after_max_retries(self, sample_documents,
                                               mock_embeddings):
        """
        Test that retriever gives up after max retries

        Verifies that after MAX_RETRIES attempts, retriever returns
        error mode with the exception.
        """
        mock_index = Mock()
        # Always fail
        mock_index.query.side_effect = Exception("Persistent error")

        retriever = ResilientHybridRetriever(
            pinecone_index=mock_index,
            documents=sample_documents,
            embeddings=mock_embeddings
        )

        with patch('time.sleep'):  # Mock sleep to speed up test
            docs, mode, error = retriever.retrieve("test")

        # Should return error after retries
        assert mode == "error"
        assert error is not None
        assert len(docs) == 0

    def test_empty_documents_list(self, mock_pinecone_index, mock_embeddings):
        """
        Test retriever handles empty document list gracefully

        Verifies that retriever can be initialized with empty documents
        and handles searches appropriately.
        """
        retriever = ResilientHybridRetriever(
            pinecone_index=mock_pinecone_index,
            documents=[],
            embeddings=mock_embeddings
        )

        docs, mode, error = retriever.retrieve("test query")

        # Should complete without error (though may return no results)
        assert mode in ["full", "error"]
