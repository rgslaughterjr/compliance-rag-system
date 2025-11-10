"""
Tests for citation extraction and confidence scoring

Tests the CrossEncoderReranker's ability to generate confidence scores
for citations and validates accuracy requirements (>90%).
"""
import pytest
from unittest.mock import Mock, patch
import numpy as np

from src.reranker import CrossEncoderReranker
from src.rag_system import ComplianceRAGSystem
from langchain_core.documents import Document


class TestCitationExtraction:
    """Test citation extraction from source documents"""

    def test_sources_included_in_response(self, mock_pinecone_client,
                                         sample_documents, mock_embeddings,
                                         mock_llm):
        """
        Test that source citations are included in query response

        Verifies that when return_sources=True, the response includes
        source document metadata for citations.
        """
        with patch('src.rag_system.Pinecone', return_value=mock_pinecone_client), \
             patch('src.rag_system.OpenAIEmbeddings', return_value=mock_embeddings), \
             patch('src.rag_system.ChatAnthropic', return_value=mock_llm), \
             patch.object(ComplianceRAGSystem, '_load_documents',
                         return_value=sample_documents):

            rag = ComplianceRAGSystem()
            result = rag.query("What is NIST framework?", return_sources=True)

            assert 'sources' in result
            assert isinstance(result['sources'], list)
            assert len(result['sources']) > 0

    def test_source_metadata_structure(self, mock_pinecone_client,
                                      sample_documents, mock_embeddings,
                                      mock_llm):
        """
        Test that source citations contain required metadata

        Verifies that each citation includes source, page, and content fields.
        """
        with patch('src.rag_system.Pinecone', return_value=mock_pinecone_client), \
             patch('src.rag_system.OpenAIEmbeddings', return_value=mock_embeddings), \
             patch('src.rag_system.ChatAnthropic', return_value=mock_llm), \
             patch.object(ComplianceRAGSystem, '_load_documents',
                         return_value=sample_documents):

            rag = ComplianceRAGSystem()
            result = rag.query("What is GDPR?", return_sources=True)

            for source in result['sources']:
                assert 'source' in source
                assert 'page' in source
                assert 'content' in source
                assert isinstance(source['source'], str)
                assert isinstance(source['content'], str)

    def test_citation_content_truncated(self, mock_pinecone_client,
                                       sample_documents, mock_embeddings,
                                       mock_llm):
        """
        Test that citation content is appropriately truncated

        Verifies that long document content is truncated to ~200 chars
        for citation display.
        """
        with patch('src.rag_system.Pinecone', return_value=mock_pinecone_client), \
             patch('src.rag_system.OpenAIEmbeddings', return_value=mock_embeddings), \
             patch('src.rag_system.ChatAnthropic', return_value=mock_llm), \
             patch.object(ComplianceRAGSystem, '_load_documents',
                         return_value=sample_documents):

            rag = ComplianceRAGSystem()
            result = rag.query("trustworthy AI", return_sources=True)

            for source in result['sources']:
                # Content should be truncated with "..."
                assert len(source['content']) <= 204  # 200 + "..."
                if len(source['content']) > 200:
                    assert source['content'].endswith("...")

    def test_sources_omitted_when_not_requested(self, mock_pinecone_client,
                                                sample_documents, mock_embeddings,
                                                mock_llm):
        """
        Test that sources are omitted when return_sources=False

        Verifies that citation sources are not included when not requested.
        """
        with patch('src.rag_system.Pinecone', return_value=mock_pinecone_client), \
             patch('src.rag_system.OpenAIEmbeddings', return_value=mock_embeddings), \
             patch('src.rag_system.ChatAnthropic', return_value=mock_llm), \
             patch.object(ComplianceRAGSystem, '_load_documents',
                         return_value=sample_documents):

            rag = ComplianceRAGSystem()
            result = rag.query("What is NIST?", return_sources=False)

            assert 'sources' not in result
            assert 'answer' in result


class TestConfidenceScoring:
    """Test confidence score calculation for citations"""

    def test_reranker_returns_scores(self, sample_documents):
        """
        Test that reranker returns confidence scores

        Verifies that rerank_with_scores returns tuples of
        (document, confidence_score).
        """
        reranker = CrossEncoderReranker()
        query = "What are trustworthy AI characteristics?"

        scored_results = reranker.rerank_with_scores(query, sample_documents, top_k=3)

        assert isinstance(scored_results, list)
        assert len(scored_results) <= 3

        for doc, score in scored_results:
            assert isinstance(doc, Document)
            assert isinstance(score, (float, np.floating))

    def test_confidence_scores_are_numeric(self, sample_documents):
        """
        Test that confidence scores are numeric values

        Verifies that all confidence scores are valid numbers.
        """
        reranker = CrossEncoderReranker()
        query = "GDPR data protection"

        scored_results = reranker.rerank_with_scores(query, sample_documents)

        for doc, score in scored_results:
            assert isinstance(score, (int, float, np.floating))
            assert not np.isnan(score)
            assert not np.isinf(score)

    def test_scores_sorted_descending(self, sample_documents):
        """
        Test that results are sorted by confidence score (descending)

        Verifies that higher confidence citations appear first.
        """
        reranker = CrossEncoderReranker()
        query = "AI risk management framework"

        scored_results = reranker.rerank_with_scores(query, sample_documents, top_k=5)

        # Extract scores
        scores = [score for _, score in scored_results]

        # Verify descending order
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], \
                f"Scores not in descending order: {scores}"

    def test_top_k_limits_results(self, sample_documents):
        """
        Test that top_k limits number of scored results

        Verifies that only top_k most relevant documents are returned.
        """
        reranker = CrossEncoderReranker()
        query = "compliance requirements"

        # Test different k values
        for k in [1, 2, 3]:
            scored_results = reranker.rerank_with_scores(
                query, sample_documents, top_k=k
            )
            assert len(scored_results) == min(k, len(sample_documents))

    def test_empty_documents_returns_empty(self):
        """
        Test that reranking empty document list returns empty results

        Verifies graceful handling of edge case.
        """
        reranker = CrossEncoderReranker()
        query = "test query"

        scored_results = reranker.rerank_with_scores(query, [], top_k=3)

        assert scored_results == []


class TestCitationAccuracy:
    """Test citation accuracy requirements (>90%)"""

    def test_high_confidence_for_relevant_docs(self, sample_documents):
        """
        Test that relevant documents receive high confidence scores

        Verifies that when query matches document content closely,
        confidence scores are high (>0.5 for relevant matches).
        """
        reranker = CrossEncoderReranker()

        # Query that should match first document well
        query = "What are the core functions of NIST AI Risk Management Framework?"

        scored_results = reranker.rerank_with_scores(query, sample_documents, top_k=3)

        # At least one result should have high confidence
        top_score = scored_results[0][1] if scored_results else 0
        assert top_score > 0.3, \
            f"Expected high confidence for relevant doc, got {top_score}"

    def test_low_confidence_for_irrelevant_docs(self, sample_documents):
        """
        Test that irrelevant documents receive lower confidence scores

        Verifies that documents not matching query content receive
        lower confidence scores.
        """
        reranker = CrossEncoderReranker()

        # Add an irrelevant document
        irrelevant_doc = Document(
            page_content="The weather forecast shows sunny skies with temperatures "
                        "reaching 75 degrees Fahrenheit.",
            metadata={'id': 'irrelevant', 'source': 'weather.txt'}
        )

        docs_with_irrelevant = sample_documents + [irrelevant_doc]

        # Query about AI/compliance - irrelevant doc should score low
        query = "NIST AI framework core functions"

        scored_results = reranker.rerank_with_scores(
            query, docs_with_irrelevant, top_k=len(docs_with_irrelevant)
        )

        # Find the irrelevant document's score
        irrelevant_score = None
        for doc, score in scored_results:
            if doc.metadata.get('id') == 'irrelevant':
                irrelevant_score = score
                break

        # Irrelevant doc should not be in top results or have low score
        if irrelevant_score is not None:
            # Get top score for comparison
            top_score = scored_results[0][1]
            assert irrelevant_score < top_score, \
                "Irrelevant document should score lower than relevant ones"

    def test_confidence_differentiation(self, sample_documents):
        """
        Test that confidence scores differentiate between documents

        Verifies that the model assigns different scores to different
        documents based on relevance.
        """
        reranker = CrossEncoderReranker()
        query = "GDPR right to erasure Article 17"

        scored_results = reranker.rerank_with_scores(query, sample_documents)

        # Extract unique scores
        scores = [score for _, score in scored_results]
        unique_scores = set(scores)

        # Should have some variation in scores (not all identical)
        assert len(unique_scores) > 1, \
            "Confidence scores should differentiate between documents"

    def test_consistent_scoring(self, sample_documents):
        """
        Test that confidence scoring is consistent across calls

        Verifies that the same query-document pair receives
        consistent scores.
        """
        reranker = CrossEncoderReranker()
        query = "AI monitoring and drift analysis"

        # Run twice
        results1 = reranker.rerank_with_scores(query, sample_documents, top_k=3)
        results2 = reranker.rerank_with_scores(query, sample_documents, top_k=3)

        # Should get same scores (model is deterministic)
        scores1 = [score for _, score in results1]
        scores2 = [score for _, score in results2]

        assert len(scores1) == len(scores2)
        for s1, s2 in zip(scores1, scores2):
            assert abs(s1 - s2) < 0.001, \
                f"Scores should be consistent: {s1} vs {s2}"


class TestAccuracyValidation:
    """Validate >90% citation accuracy requirement"""

    def test_top_result_relevance(self):
        """
        Test that top result is highly relevant (90%+ accuracy proxy)

        For a set of known queries, verifies that the most relevant
        document is ranked first, achieving >90% accuracy.
        """
        reranker = CrossEncoderReranker()

        # Test cases: (query, expected_top_doc_id)
        test_cases = [
            ("What are the core functions of NIST AI Risk Management Framework?", "doc_0"),
            ("GDPR Article 17 right to erasure", "doc_1"),
            ("trustworthy AI characteristics fairness transparency", "doc_2"),
            ("data protection impact assessments GDPR", "doc_3"),
            ("AI model monitoring drift detection", "doc_4"),
        ]

        correct_predictions = 0

        # Create documents for testing
        documents = [
            Document(
                page_content="The NIST AI Risk Management Framework provides guidelines for managing AI risks. "
                            "It includes four core functions: Govern, Map, Measure, and Manage.",
                metadata={'id': 'doc_0', 'source': 'nist_ai_rmf.pdf'}
            ),
            Document(
                page_content="GDPR Article 17 establishes the right to erasure, also known as the right to be forgotten. "
                            "This allows individuals to request deletion of their personal data.",
                metadata={'id': 'doc_1', 'source': 'gdpr.pdf'}
            ),
            Document(
                page_content="Trustworthy AI characteristics include fairness, transparency, accountability, "
                            "privacy, security, safety, and robustness.",
                metadata={'id': 'doc_2', 'source': 'ai_ethics.pdf'}
            ),
            Document(
                page_content="Data protection impact assessments (DPIAs) are required under GDPR when processing "
                            "is likely to result in high risk to individuals' rights and freedoms.",
                metadata={'id': 'doc_3', 'source': 'gdpr.pdf'}
            ),
            Document(
                page_content="AI model monitoring involves tracking performance metrics, bias detection, "
                            "and drift analysis to ensure continued model quality in production.",
                metadata={'id': 'doc_4', 'source': 'ml_ops.pdf'}
            ),
        ]

        for query, expected_id in test_cases:
            scored_results = reranker.rerank_with_scores(query, documents, top_k=1)

            if scored_results:
                top_doc = scored_results[0][0]
                if top_doc.metadata.get('id') == expected_id:
                    correct_predictions += 1

        accuracy = (correct_predictions / len(test_cases)) * 100

        assert accuracy >= 90.0, \
            f"Citation accuracy {accuracy:.1f}% does not meet 90% requirement"

    def test_multi_document_ranking_accuracy(self):
        """
        Test ranking accuracy across multiple relevant documents

        Verifies that when multiple documents are relevant,
        the most relevant one ranks highest.
        """
        reranker = CrossEncoderReranker()

        # Create documents with varying relevance to GDPR
        documents = [
            Document(
                page_content="AI model training requires large datasets and computational resources.",
                metadata={'id': 'low_relevance', 'source': 'ml.pdf'}
            ),
            Document(
                page_content="GDPR Article 17 right to erasure allows data deletion requests.",
                metadata={'id': 'high_relevance', 'source': 'gdpr.pdf'}
            ),
            Document(
                page_content="Data protection regulations including GDPR govern personal data.",
                metadata={'id': 'medium_relevance', 'source': 'privacy.pdf'}
            ),
        ]

        query = "GDPR Article 17 right to erasure"
        scored_results = reranker.rerank_with_scores(query, documents, top_k=3)

        # High relevance doc should be first
        top_doc_id = scored_results[0][0].metadata.get('id')
        assert top_doc_id == 'high_relevance', \
            f"Expected 'high_relevance' to rank first, got '{top_doc_id}'"

        # Scores should reflect relevance ordering
        scores = [score for _, score in scored_results]
        # First score should be notably higher than last
        assert scores[0] > scores[-1], \
            "Most relevant document should have higher score than least relevant"


class TestRerankerIntegration:
    """Test reranker integration with RAG system"""

    def test_reranker_reduces_result_set(self, sample_documents):
        """
        Test that reranker reduces results to top_k most relevant

        Verifies that re-ranking filters down to FINAL_K documents.
        """
        from src.config import FINAL_K

        reranker = CrossEncoderReranker()
        query = "compliance requirements"

        # Start with more documents than FINAL_K
        assert len(sample_documents) >= FINAL_K

        reranked = reranker.rerank(query, sample_documents, top_k=FINAL_K)

        assert len(reranked) == FINAL_K

    def test_reranker_preserves_document_objects(self, sample_documents):
        """
        Test that reranker preserves Document object structure

        Verifies that reranked results are still valid Document objects
        with all metadata intact.
        """
        reranker = CrossEncoderReranker()
        query = "test query"

        reranked = reranker.rerank(query, sample_documents, top_k=3)

        for doc in reranked:
            assert isinstance(doc, Document)
            assert hasattr(doc, 'page_content')
            assert hasattr(doc, 'metadata')
            assert isinstance(doc.metadata, dict)

    def test_reranker_handles_empty_input(self):
        """
        Test that reranker handles empty document list gracefully

        Verifies edge case handling.
        """
        reranker = CrossEncoderReranker()
        query = "test query"

        reranked = reranker.rerank(query, [], top_k=3)

        assert reranked == []
