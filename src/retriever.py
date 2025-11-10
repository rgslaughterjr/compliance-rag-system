"""
Resilient hybrid retriever with circuit breaker and retry logic
Combines semantic search (Pinecone) with keyword search (BM25)
Implemented in Lab 2.1, hardened in Lab 3.3
"""
import time
from typing import List, Optional, Tuple
from enum import Enum

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

from .config import (
    SEMANTIC_WEIGHT,
    RETRIEVAL_K,
    CIRCUIT_BREAKER_THRESHOLD,
    CIRCUIT_BREAKER_TIMEOUT,
    CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
    MAX_RETRIES,
    INITIAL_BACKOFF,
    MAX_BACKOFF
)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance
    
    Prevents cascading failures by:
    - Opening circuit after threshold failures
    - Rejecting requests while open
    - Testing recovery after timeout
    """

    def __init__(self, failure_threshold: int = CIRCUIT_BREAKER_THRESHOLD,
                 timeout_seconds: int = CIRCUIT_BREAKER_TIMEOUT,
                 success_threshold: int = CIRCUIT_BREAKER_SUCCESS_THRESHOLD):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.success_threshold = success_threshold

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker"""
        if self.state == CircuitState.OPEN:
            # Check if timeout expired
            if time.time() - self.last_failure_time > self.timeout_seconds:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN


class ResilientHybridRetriever:
    """
    Hybrid retriever combining semantic and keyword search with error handling
    
    Features:
    - Semantic search via Pinecone vector database
    - Keyword search via BM25
    - Weighted fusion (0.9/0.1 optimized in Lab 2.4+)
    - Circuit breaker for fault tolerance
    - Exponential backoff retry logic
    - Query caching integration
    """

    def __init__(self, pinecone_index, documents: List[Document],
                 embeddings: OpenAIEmbeddings, cache=None,
                 semantic_weight: float = SEMANTIC_WEIGHT):
        self.index = pinecone_index
        self.documents = documents
        self.embeddings = embeddings
        self.cache = cache
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1.0 - semantic_weight

        # Initialize BM25
        self.tokenized_docs = [doc.page_content.lower().split()
                               for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

        # Error handling
        self.circuit_breaker = CircuitBreaker()

    def retrieve(self, query: str, k: int = RETRIEVAL_K,
                filters: dict = None) -> Tuple[List[Document], str, Optional[Exception]]:
        """
        Retrieve documents with error handling
        
        Returns:
            (documents, mode, error)
            mode: "full", "cache", or "error"
        """
        # Try cache first
        if self.cache:
            cached = self.cache.get(query, filters)
            if cached:
                return cached

        # Try retrieval with retry logic
        for attempt in range(MAX_RETRIES):
            try:
                docs = self.circuit_breaker.call(
                    self._hybrid_search,
                    query, k, filters
                )

                # Cache successful result
                if self.cache:
                    self.cache.set(query, docs, mode="full", filters=filters)

                return (docs, "full", None)

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    # Exponential backoff
                    wait_time = min(INITIAL_BACKOFF * (2 ** attempt), MAX_BACKOFF)
                    time.sleep(wait_time)
                    continue
                else:
                    # All retries failed
                    return ([], "error", e)

        return ([], "error", Exception("Max retries exceeded"))

    def _hybrid_search(self, query: str, k: int,
                      filters: dict = None) -> List[Document]:
        """
        Hybrid search combining semantic and keyword search
        
        Optimized weights: 0.9 semantic / 0.1 keyword (Lab 2.4+)
        """
        # Semantic search (Pinecone)
        query_vector = self.embeddings.embed_query(query)

        if filters:
            semantic_results = self.index.query(
                vector=query_vector,
                top_k=k,
                include_metadata=True,
                filter=filters
            )
        else:
            semantic_results = self.index.query(
                vector=query_vector,
                top_k=k,
                include_metadata=True
            )

        # Keyword search (BM25)
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)

        # Fusion: weighted combination
        doc_scores = {}

        # Add semantic scores
        for match in semantic_results['matches']:
            doc_id = match['id']
            doc_scores[doc_id] = match['score'] * self.semantic_weight

        # Add BM25 scores
        for idx, score in enumerate(bm25_scores):
            doc_id = self.documents[idx].metadata.get('id', str(idx))
            if doc_id in doc_scores:
                doc_scores[doc_id] += score * self.keyword_weight
            else:
                doc_scores[doc_id] = score * self.keyword_weight

        # Sort by combined score
        sorted_ids = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top k documents
        result_docs = []
        for doc_id, score in sorted_ids[:k]:
            # Find document by ID
            for doc in self.documents:
                if doc.metadata.get('id', '') == doc_id:
                    result_docs.append(doc)
                    break

        return result_docs
