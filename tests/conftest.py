"""
Shared pytest fixtures for compliance-rag-system tests

Provides mock objects and test data for:
- Pinecone vector database connections
- OpenAI embeddings
- Sample compliance documents
- Cache instances
"""
import pytest
from unittest.mock import Mock, MagicMock
from langchain_core.documents import Document
from typing import List

from src.cache import QueryCache


@pytest.fixture
def sample_documents() -> List[Document]:
    """
    Sample compliance documents for testing

    Returns:
        List of Document objects with compliance-related content
    """
    return [
        Document(
            page_content="The NIST AI Risk Management Framework provides guidelines for managing AI risks. "
                        "It includes four core functions: Govern, Map, Measure, and Manage.",
            metadata={
                'id': 'doc_0',
                'source': 'nist_ai_rmf.pdf',
                'page': 1,
                'section': 'Core Functions'
            }
        ),
        Document(
            page_content="GDPR Article 17 establishes the right to erasure, also known as the right to be forgotten. "
                        "This allows individuals to request deletion of their personal data.",
            metadata={
                'id': 'doc_1',
                'source': 'gdpr.pdf',
                'page': 17,
                'section': 'Rights of Data Subjects'
            }
        ),
        Document(
            page_content="Trustworthy AI characteristics include fairness, transparency, accountability, "
                        "privacy, security, safety, and robustness. These principles ensure AI systems "
                        "are developed responsibly.",
            metadata={
                'id': 'doc_2',
                'source': 'ai_ethics.pdf',
                'page': 5,
                'section': 'Principles'
            }
        ),
        Document(
            page_content="Data protection impact assessments (DPIAs) are required under GDPR when processing "
                        "is likely to result in high risk to individuals' rights and freedoms.",
            metadata={
                'id': 'doc_3',
                'source': 'gdpr.pdf',
                'page': 35,
                'section': 'Data Protection'
            }
        ),
        Document(
            page_content="AI model monitoring involves tracking performance metrics, bias detection, "
                        "and drift analysis to ensure continued model quality in production.",
            metadata={
                'id': 'doc_4',
                'source': 'ml_ops.pdf',
                'page': 12,
                'section': 'Monitoring'
            }
        )
    ]


@pytest.fixture
def mock_embeddings():
    """
    Mock OpenAI embeddings model

    Returns:
        Mock embeddings object with embed_query method
    """
    mock = Mock()
    # Return consistent embedding vectors for testing
    mock.embed_query = Mock(return_value=[0.1] * 1536)
    return mock


@pytest.fixture
def mock_pinecone_index():
    """
    Mock Pinecone index with query functionality

    Returns:
        Mock Pinecone index that returns realistic query results
    """
    mock_index = Mock()

    def mock_query(vector=None, top_k=20, include_metadata=True, filter=None):
        """Mock Pinecone query response"""
        # Return results with decreasing similarity scores
        matches = [
            {
                'id': 'doc_0',
                'score': 0.95,
                'metadata': {
                    'source': 'nist_ai_rmf.pdf',
                    'page': 1,
                    'section': 'Core Functions'
                }
            },
            {
                'id': 'doc_2',
                'score': 0.88,
                'metadata': {
                    'source': 'ai_ethics.pdf',
                    'page': 5,
                    'section': 'Principles'
                }
            },
            {
                'id': 'doc_1',
                'score': 0.82,
                'metadata': {
                    'source': 'gdpr.pdf',
                    'page': 17,
                    'section': 'Rights of Data Subjects'
                }
            },
            {
                'id': 'doc_4',
                'score': 0.75,
                'metadata': {
                    'source': 'ml_ops.pdf',
                    'page': 12,
                    'section': 'Monitoring'
                }
            },
            {
                'id': 'doc_3',
                'score': 0.70,
                'metadata': {
                    'source': 'gdpr.pdf',
                    'page': 35,
                    'section': 'Data Protection'
                }
            }
        ]

        return {'matches': matches[:top_k]}

    mock_index.query = Mock(side_effect=mock_query)
    return mock_index


@pytest.fixture
def mock_pinecone_client(mock_pinecone_index):
    """
    Mock Pinecone client

    Returns:
        Mock Pinecone client that returns mock index
    """
    mock_client = Mock()
    mock_client.Index = Mock(return_value=mock_pinecone_index)
    return mock_client


@pytest.fixture
def test_cache():
    """
    Fresh QueryCache instance for testing

    Returns:
        QueryCache with default settings (24hr TTL, 5000 max size)
    """
    return QueryCache()


@pytest.fixture
def test_cache_short_ttl():
    """
    QueryCache with short TTL for testing expiration

    Returns:
        QueryCache with 1 second TTL
    """
    return QueryCache(ttl_seconds=1, max_size=10)


@pytest.fixture
def mock_cross_encoder():
    """
    Mock CrossEncoder model for re-ranking

    Returns:
        Mock CrossEncoder that returns predictable scores
    """
    mock = Mock()

    def mock_predict(pairs):
        """Return descending scores based on pair order"""
        return [1.0 - (i * 0.1) for i in range(len(pairs))]

    mock.predict = Mock(side_effect=mock_predict)
    return mock


@pytest.fixture
def mock_llm():
    """
    Mock Claude LLM for answer generation

    Returns:
        Mock LLM that returns structured responses
    """
    mock = Mock()

    def mock_invoke(prompt):
        """Return mock response with content attribute"""
        response = Mock()
        response.content = "This is a test answer based on the provided context. " \
                          "According to NIST AI RMF, the core functions are Govern, Map, Measure, and Manage."
        return response

    mock.invoke = Mock(side_effect=mock_invoke)
    return mock


@pytest.fixture
def sample_query():
    """
    Sample query for testing

    Returns:
        String query about AI risk management
    """
    return "What are the core functions of the NIST AI Risk Management Framework?"


@pytest.fixture
def sample_query_gdpr():
    """
    Sample GDPR-related query for testing

    Returns:
        String query about GDPR Article 17
    """
    return "What is GDPR Article 17?"
