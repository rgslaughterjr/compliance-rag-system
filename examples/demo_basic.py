#!/usr/bin/env python3
"""
Basic Demo - Compliance RAG System
Simple demonstration of system capabilities
"""
import sys
sys.path.insert(0, '..')

from src.rag_system import ComplianceRAGSystem


def main():
    """Run basic demo"""

    print("=" * 70)
    print("COMPLIANCE RAG SYSTEM - BASIC DEMO")
    print("=" * 70)
    print()

    # Initialize system
    print("Initializing system...\n")
    rag = ComplianceRAGSystem()

    # Demo queries
    queries = [
        "What are the core functions of the NIST AI Risk Management Framework?",
        "What are trustworthy AI characteristics?",
        "What is GDPR Article 17?",
    ]

    print("=" * 70)
    print("RUNNING DEMO QUERIES")
    print("=" * 70)

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {query}")
        print('='*70)

        # Query system
        result = rag.query(query)

        # Display answer
        print(f"\n📝 Answer:")
        print(result['answer'])

        # Display sources
        if result.get('sources'):
            print(f"\n📚 Sources:")
            for j, source in enumerate(result['sources'], 1):
                print(f"  {j}. {source['source']}, page {source['page']}")

        # Display metadata
        print(f"\n⚙️ Metadata:")
        print(f"  Cache hit: {result['cache_hit']}")
        print(f"  Mode: {result['mode']}")

    # Show cache statistics
    print(f"\n{'='*70}")
    print("CACHE STATISTICS")
    print('='*70)

    stats = rag.get_cache_stats()
    print(f"  Cache size: {stats['size']} entries")
    print(f"  Total queries: {stats['hits'] + stats['misses']}")
    print(f"  Cache hits: {stats['hits']}")
    print(f"  Cache misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']}")

    print(f"\n{'='*70}")
    print("✅ DEMO COMPLETE!")
    print('='*70)
    print("\nTry running again to see 100% cache hit rate!")


if __name__ == "__main__":
    main()
