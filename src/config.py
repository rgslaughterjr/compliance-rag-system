"""
Configuration settings for the RAG system
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone Configuration
PINECONE_INDEX_NAME = "compliance-kb-prod"
PINECONE_ENVIRONMENT = "us-east-1"

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# Retrieval Configuration
SEMANTIC_WEIGHT = 0.9  # Optimized in Lab 2.4+
KEYWORD_WEIGHT = 0.1
RETRIEVAL_K = 20  # Initial candidates
FINAL_K = 4       # After re-ranking

# Cross-Encoder Configuration
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Cache Configuration (Optimized in Lab 3.4+)
CACHE_TTL_SECONDS = 86400  # 24 hours
MAX_CACHE_SIZE = 5000

# Circuit Breaker Configuration
CIRCUIT_BREAKER_THRESHOLD = 5  # Failures before opening
CIRCUIT_BREAKER_TIMEOUT = 60   # Seconds before retry
CIRCUIT_BREAKER_SUCCESS_THRESHOLD = 2

# Retry Configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # Seconds
MAX_BACKOFF = 16.0     # Seconds

# LLM Configuration
LLM_MODEL = "claude-sonnet-4-5-20250929"
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 2000
