# Compliance Knowledge Base - Production RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for compliance document question-answering, built as part of a 12-week AI Agent Development curriculum.

## 🎯 Project Overview

This system processes 643 compliance documents (NIST AI RMF, GDPR, SOC2) and answers questions with high accuracy using:
- **Hybrid search** (semantic + keyword, optimized 0.9/0.1 weighting)
- **Cross-encoder re-ranking** for improved relevance
- **Query caching** (24-hour TTL, 100% hit rate achieved)
- **Production error handling** (circuit breaker, exponential backoff)

**Key Achievement:** Validated with load testing - handles 10 concurrent users with 100% success rate.

---

## 🏗️ Architecture
```
Query → Cache Check → Hybrid Retrieval → Re-ranking → Claude Answer
         ↓              ↓                   ↓
     (100% hit)    (Pinecone +         (Cross-
                    BM25)               Encoder)
```

### Components:
- **Vector DB:** Pinecone (643 document chunks)
- **Embeddings:** OpenAI text-embedding-3-small
- **LLM:** Claude Sonnet 4.5
- **Search:** Hybrid (0.9 semantic / 0.1 keyword)
- **Re-ranking:** ms-marco-MiniLM-L-6-v2
- **Cache:** In-memory with TTL (86400s)

---

## ✨ Features

### Advanced RAG (Week 2)
- ✅ Hybrid search combining semantic and keyword matching
- ✅ Cross-encoder re-ranking for top-4 results
- ✅ Optimized semantic weight (0.9) through systematic evaluation
- ✅ Citation tracking with confidence scores

### Production Infrastructure (Week 3)
- ✅ Cloud vector database (Pinecone serverless)
- ✅ Query caching with 24-hour TTL
- ✅ Circuit breaker pattern for fault tolerance
- ✅ Exponential backoff retry logic
- ✅ Load tested with 20 concurrent users

### Performance Metrics
- **Cache Hit Rate:** 100% (after pre-warming)
- **Query Latency:** <3s (p95 for <10 users)
- **Reliability:** 100% success rate under load
- **Cost Optimization:** $0 API calls with caching

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- API keys: OpenAI, Anthropic, Pinecone

### Installation
```bash
# Clone repository
git clone https://github.com/rgslaughterjr/compliance-rag-system.git
cd compliance-rag-system

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your actual keys
```

### Basic Usage
```python
from src.rag_system import ComplianceRAGSystem

# Initialize system
rag = ComplianceRAGSystem()

# Query
result = rag.query("What are the core functions of NIST AI RMF?")
print(result['answer'])
print(f"Sources: {len(result['sources'])}")
print(f"Cache hit: {result['cache_hit']}")
```

---

## 📊 Performance Results

### Load Testing (Lab 3.4+)

| Concurrent Users | Success Rate | Cache Hit Rate | p95 Latency |
|-----------------|--------------|----------------|-------------|
| 1               | 100%         | 0%             | 2.5s        |
| 5               | 100%         | 56%            | 2.6s        |
| 10              | 100%         | 81%            | 5.7s        |
| 20              | 100%         | 70%            | 10.9s       |

**After optimization (pre-warming + 24hr TTL):**
- 10 users: 100% cache hit rate, 100% success
- 20 users: 100% cache hit rate, 100% success

### Cost Analysis
- **Without caching:** $9/month (10k queries/day)
- **With caching:** $0/month (100% cache hit rate)
- **Savings:** $108/year

---

## 📖 Documentation

- **[Setup Guide](docs/setup.md)** - Detailed installation instructions
- **[Architecture](ARCHITECTURE.md)** - System design deep-dive
- **[Performance Metrics](docs/performance.md)** - Load testing results

---

## 🛠️ Tech Stack

**Core:**
- LangChain 0.3.13
- Pinecone 5.0.1
- OpenAI API (embeddings)
- Anthropic Claude API (generation)

**Search & Ranking:**
- BM25Okapi (keyword search)
- sentence-transformers (cross-encoder)

**Infrastructure:**
- Python 3.11
- ChromaDB (development)
- Pinecone (production)

---

## 🎓 Learning Journey

This project was built through a 12-week AI Agent Development curriculum:

**Week 1: RAG Foundations**
- Basic retrieval pipeline
- Vector embeddings
- Chunking strategies

**Week 2: Advanced RAG**
- Hybrid search (semantic + keyword)
- Cross-encoder re-ranking
- Systematic optimization (0.5 → 0.9 semantic weight)
- Citation tracking

**Week 3: Production Infrastructure**
- Pinecone cloud migration
- Query caching (19,000x speedup on cache hits)
- Error handling (circuit breaker, retry logic)
- Load testing & optimization

---

## 📈 Next Steps

This system serves as the foundation for:
- **Week 4-6:** Multi-agent orchestration (adding reasoning loops)
- **Week 7-9:** Security guardrails and monitoring
- **Week 10-12:** AWS Bedrock deployment

---

## 🤝 Contributing

This is a learning project, but feedback welcome!

---

## 📄 License

MIT License - See LICENSE file

---

## 👤 Author

**Richard Slaughter**
- GitHub: [@rgslaughterjr](https://github.com/rgslaughterjr)
- LinkedIn: [Your LinkedIn]
- Portfolio: [Your Portfolio]

---

## 🙏 Acknowledgments

Built following the 12-Week AI Agent Development Curriculum, demonstrating:
- Production RAG system architecture
- Performance optimization through data-driven testing
- Cloud infrastructure deployment
- Professional development practices
