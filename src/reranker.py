"""
Cross-encoder re-ranking for improved relevance
Implemented in Lab 2.2
"""
from typing import List
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from .config import RERANKER_MODEL, FINAL_K


class CrossEncoderReranker:
    """
    Re-rank documents using cross-encoder for deep semantic analysis
    
    Cross-encoders provide superior relevance scoring by processing
    query+document pairs jointly, but are too slow for initial retrieval.
    Use for re-ranking top candidates from hybrid search.
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(self, query: str, documents: List[Document],
               top_k: int = FINAL_K) -> List[Document]:
        """
        Re-rank documents by relevance using cross-encoder
        
        Args:
            query: Search query
            documents: Documents to re-rank
            top_k: Number of top documents to return
        
        Returns:
            Re-ranked documents (top_k most relevant)
        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Sort documents by score (descending)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top_k documents
        reranked = [doc for doc, score in scored_docs[:top_k]]

        return reranked

    def rerank_with_scores(self, query: str, documents: List[Document],
                          top_k: int = FINAL_K) -> List[tuple]:
        """
        Re-rank documents and return with confidence scores
        
        Returns:
            List of (document, confidence_score) tuples
        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Sort documents by score (descending)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:top_k]
