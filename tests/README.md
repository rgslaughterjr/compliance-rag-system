# Compliance RAG System - Test Suite

Comprehensive test suite for the compliance-rag-system project using pytest.

## Test Coverage

### 1. test_cache.py (24 tests - ✅ All Passing)
Tests for QueryCache functionality:
- **Cache Key Generation**: Case-insensitive, whitespace handling, filter support
- **Cache Hit/Miss Logic**: Statistics tracking, hit rate calculation
- **TTL Expiration**: Time-based cache invalidation
- **LRU Eviction**: Least Recently Used eviction when at capacity
- **Cache Maintenance**: Clear operations, statistics preservation

### 2. test_retriever.py (40+ tests)
Tests for ResilientHybridRetriever:
- **Hybrid Search**: Semantic (Pinecone) + Keyword (BM25) combination
- **Semantic Weight**: Validation of 0.9 optimized weight configuration
- **Result Ranking**: Score-based ranking and top-k selection
- **Cache Integration**: Cache hit/miss behavior with retriever
- **Circuit Breaker**: Fault tolerance and error handling
- **Retry Logic**: Exponential backoff and max retries

### 3. test_citations.py (30+ tests)
Tests for citation extraction and confidence scoring:
- **Citation Extraction**: Source document metadata inclusion
- **Confidence Scoring**: CrossEncoder score calculation
- **Accuracy Validation**: >90% citation accuracy requirement
- **Score Differentiation**: Relevance-based ranking
- **Reranker Integration**: Document re-ranking with scores

## Running the Tests

### Prerequisites

Install required dependencies:

```bash
# Install project dependencies
pip install -r requirements.txt

# Install test framework
pip install pytest pytest-cov
```

### Run All Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_cache.py -v
pytest tests/test_retriever.py -v
pytest tests/test_citations.py -v
```

### Run Specific Test Classes or Functions

```bash
# Run specific test class
pytest tests/test_cache.py::TestCacheKeyGeneration -v

# Run specific test function
pytest tests/test_cache.py::TestCacheKeyGeneration::test_generate_key_basic -v
```

### Test Output Options

```bash
# Show detailed output on failures
pytest tests/ -v --tb=short

# Show all output including print statements
pytest tests/ -v -s

# Stop on first failure
pytest tests/ -x

# Run only failed tests from last run
pytest tests/ --lf
```

## Test Fixtures

Shared fixtures are defined in `conftest.py`:

- **sample_documents**: 5 compliance documents for testing
- **mock_embeddings**: Mock OpenAI embeddings model
- **mock_pinecone_index**: Mock Pinecone index with realistic responses
- **mock_pinecone_client**: Mock Pinecone client
- **test_cache**: Fresh QueryCache instance
- **test_cache_short_ttl**: Cache with 1-second TTL for expiration tests
- **mock_cross_encoder**: Mock CrossEncoder for re-ranking tests
- **mock_llm**: Mock Claude LLM for answer generation
- **sample_query**: Pre-defined test queries

## Test Structure

Each test file is organized into logical test classes:

```python
class TestFeatureName:
    """Test specific feature functionality"""

    def test_specific_behavior(self, fixture1, fixture2):
        """
        Test description

        Verifies that [expected behavior].
        """
        # Arrange
        # Act
        # Assert
```

All tests include comprehensive docstrings explaining:
- What is being tested
- Expected behavior
- Verification approach

## Continuous Integration

To integrate with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements.txt
    pip install pytest pytest-cov
    pytest tests/ --cov=src --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
```

## Contributing

When adding new features:
1. Add corresponding tests in the appropriate test file
2. Ensure tests have descriptive names and docstrings
3. Use existing fixtures where possible
4. Run full test suite before committing: `pytest tests/ -v`
5. Maintain >90% test coverage

## Test Requirements

- Python 3.11+
- pytest >= 8.3.4
- All dependencies from requirements.txt
- Mock/patch support for external services (Pinecone, OpenAI)
