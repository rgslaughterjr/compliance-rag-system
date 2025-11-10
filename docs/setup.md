# Setup Guide - Compliance RAG System

Complete installation and configuration instructions.

---

## Prerequisites

### 1. System Requirements
- **Python:** 3.11 or higher
- **OS:** Windows, macOS, or Linux
- **RAM:** 8GB minimum (16GB recommended)
- **Disk:** 2GB free space

### 2. API Keys Required
- **OpenAI API key** (for embeddings)
- **Anthropic API key** (for Claude)
- **Pinecone API key** (for vector database)

---

## Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/rgslaughterjr/compliance-rag-system.git
cd compliance-rag-system
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**This installs:**
- LangChain packages
- Pinecone client
- OpenAI and Anthropic SDKs
- Search and ranking libraries
- Utilities

---

## Configuration

### Step 1: Create .env File
```bash
cp .env.example .env
```

### Step 2: Add Your API Keys

Edit `.env` file:
```bash
# OpenAI (for embeddings)
OPENAI_API_KEY=sk-proj-your_actual_key_here

# Anthropic Claude (for generation)
ANTHROPIC_API_KEY=sk-ant-your_actual_key_here

# Pinecone (for vector database)
PINECONE_API_KEY=pcsk-your_actual_key_here
```

**⚠️ Important:**
- Never commit `.env` to Git
- Keep your keys secret
- Rotate keys if exposed

---

## Getting API Keys

### OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Sign up or log in
3. Click "Create new secret key"
4. Copy key (starts with `sk-proj-`)
5. Add to `.env` file

**Cost:** $0.0002 per 1K tokens for embeddings

### Anthropic API Key

1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Navigate to "API Keys"
4. Click "Create Key"
5. Copy key (starts with `sk-ant-`)
6. Add to `.env` file

**Cost:** $3 per 1M input tokens, $15 per 1M output tokens

### Pinecone API Key

1. Go to https://www.pinecone.io/
2. Sign up (free tier available)
3. Create a new project
4. Copy API key from dashboard
5. Add to `.env` file

**Cost:** Free tier includes 1 index, 100K vectors

---

## Verification

### Test Installation
```bash
python -c "import src.rag_system; print('✓ Installation successful')"
```

**Expected:** `✓ Installation successful`

### Test API Keys
```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('OpenAI:', 'OK' if os.getenv('OPENAI_API_KEY') else 'MISSING'); print('Anthropic:', 'OK' if os.getenv('ANTHROPIC_API_KEY') else 'MISSING'); print('Pinecone:', 'OK' if os.getenv('PINECONE_API_KEY') else 'MISSING')"
```

**Expected:**
```
OpenAI: OK
Anthropic: OK
Pinecone: OK
```

---

## Quick Start

### Basic Usage
```python
from src.rag_system import ComplianceRAGSystem

# Initialize (takes ~10 seconds)
rag = ComplianceRAGSystem()

# Query
result = rag.query("What are the NIST AI RMF core functions?")

# View answer
print(result['answer'])

# View sources
for source in result['sources']:
    print(f"- {source['source']}, page {source['page']}")

# Check cache stats
stats = rag.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']}")
```

---

## Common Issues

### Issue 1: ModuleNotFoundError

**Error:** `ModuleNotFoundError: No module named 'langchain_community'`

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue 2: API Key Not Found

**Error:** `Error: OPENAI_API_KEY not found`

**Solution:**
1. Check `.env` file exists
2. Verify key is on correct line
3. No quotes around key value
4. Restart Python after editing `.env`

### Issue 3: Pinecone Connection Error

**Error:** `PineconeException: Index not found`

**Solution:**
1. Check index name in `src/config.py`
2. Verify index exists in Pinecone dashboard
3. Ensure API key has access to index

### Issue 4: Rate Limit Errors

**Error:** `RateLimitError: You exceeded your current quota`

**Solution:**
1. Check API usage on provider dashboard
2. Add payment method if needed
3. Wait for rate limit to reset
4. Cache helps reduce API calls!

---

## Development Setup

### For Local Development

**Install development tools:**
```bash
pip install pytest black flake8
```

**Run tests:**
```bash
pytest tests/
```

**Format code:**
```bash
black src/
```

**Lint code:**
```bash
flake8 src/
```

---

## Production Deployment

### Checklist

- [ ] Set secure API keys
- [ ] Configure production Pinecone index
- [ ] Set up monitoring
- [ ] Configure rate limiting
- [ ] Enable logging
- [ ] Set up backup/recovery

### Environment Variables

**Production .env:**
```bash
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
PINECONE_API_KEY=your_key

# Production settings
CACHE_TTL_SECONDS=86400
MAX_CACHE_SIZE=5000
CIRCUIT_BREAKER_THRESHOLD=5
```

---

## Next Steps

1. **Try examples:**
```bash
   python examples/demo_basic.py
```

2. **Read architecture:**
   See [ARCHITECTURE.md](../ARCHITECTURE.md)

3. **Check performance:**
   See [performance.md](performance.md)

4. **Build your own:**
   Adapt code for your documents!

---

## Support

- **Issues:** https://github.com/rgslaughterjr/compliance-rag-system/issues
- **Documentation:** This folder
- **Author:** Richard Slaughter

---

**Last Updated:** November 9, 2025
