"""
Test script for the reranker implementation.

This script tests the reranker step with sample data to verify it works correctly.
"""

import logging
import sys
from query.steps.b_reranker import RerankerStep, RerankConfig
from query.steps.a_hybrid_search import VectorSearchResults
from query.steps.searchers.base_searcher import SearchResult

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_sample_results():
    """Create sample search results for testing."""
    results = [
        SearchResult(
            node_id="result_1",
            similarity_score=0.85,
            document_text="Python is a high-level programming language used for web development, data science, and machine learning.",
            node_type="entity",
            chat_name="Tech Discussion",
            metadata={"source": "test"},
            search_method="hybrid"
        ),
        SearchResult(
            node_id="result_2",
            similarity_score=0.75,
            document_text="Java is an object-oriented programming language commonly used for enterprise applications.",
            node_type="entity",
            chat_name="Tech Discussion",
            metadata={"source": "test"},
            search_method="hybrid"
        ),
        SearchResult(
            node_id="result_3",
            similarity_score=0.70,
            document_text="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            node_type="claim",
            chat_name="AI Discussion",
            metadata={"source": "test"},
            search_method="hybrid"
        ),
        SearchResult(
            node_id="result_4",
            similarity_score=0.65,
            document_text="Data science involves extracting insights from structured and unstructured data using Python and R.",
            node_type="entity",
            chat_name="Data Science",
            metadata={"source": "test"},
            search_method="hybrid"
        ),
        SearchResult(
            node_id="result_5",
            similarity_score=0.60,
            document_text="Web development frameworks like Django and Flask make it easy to build Python web applications.",
            node_type="full_text",
            chat_name="Web Dev",
            metadata={"source": "test"},
            search_method="hybrid"
        )
    ]

    return VectorSearchResults(
        entity_ids={"result_1", "result_2", "result_4"},
        person_ids=set(),
        claim_ids={"result_3"},
        full_text_ids={"result_5"},
        results=results,
        original_query="Tell me about Python programming and machine learning",
        extracted_entities=["Python", "machine learning"],
        extracted_keywords=["programming", "Python", "machine", "learning"],
        extracted_variations=[]
    )


def test_cross_encoder_reranker():
    """Test the cross-encoder reranker."""
    print("\n" + "="*80)
    print("Testing Cross-Encoder Reranker")
    print("="*80)

    try:
        # Create reranker with cross-encoder method
        config = {
            "rerank_method": "cross-encoder",
            "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "rerank_top_k_before": 5,
            "rerank_top_k_after": 3,
            "rerank_boost_exact_matches": True,
            "rerank_use_query_expansion": True
        }

        reranker = RerankerStep(config)

        # Create sample data
        sample_data = create_sample_results()

        print(f"\nInput: {len(sample_data.results)} results")
        print(f"Query: {sample_data.original_query}")
        print(f"Keywords: {sample_data.extracted_keywords}")
        print(f"Entities: {sample_data.extracted_entities}")

        # Process
        result = reranker.process(sample_data)

        print(f"\nOutput: {len(result.results)} results")
        print("\nReranked results:")
        for i, r in enumerate(result.results):
            print(f"  {i+1}. [{r.node_id}] Score: {r.similarity_score:.3f}")
            print(f"     Text: {r.document_text[:80]}...")
            print(f"     Method: {r.search_method}")

        print("\n✓ Cross-encoder test passed!")
        return True

    except ImportError as e:
        print(f"\n⚠ Cross-encoder test skipped: Missing dependency ({e})")
        print("  Install with: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"\n✗ Cross-encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_zerank_reranker():
    """Test the zerank reranker."""
    print("\n" + "="*80)
    print("Testing ZeRank Reranker")
    print("="*80)

    try:
        # Create reranker with zerank method
        config = {
            "rerank_method": "zerank",
            "rerank_model": "all-MiniLM-L6-v2",
            "rerank_top_k_before": 5,
            "rerank_top_k_after": 3,
            "rerank_boost_exact_matches": True
        }

        reranker = RerankerStep(config)

        # Create sample data
        sample_data = create_sample_results()

        print(f"\nInput: {len(sample_data.results)} results")
        print(f"Query: {sample_data.original_query}")

        # Process
        result = reranker.process(sample_data)

        print(f"\nOutput: {len(result.results)} results")
        print("\nReranked results:")
        for i, r in enumerate(result.results):
            print(f"  {i+1}. [{r.node_id}] Score: {r.similarity_score:.3f}")
            print(f"     Text: {r.document_text[:80]}...")

        print("\n✓ ZeRank test passed!")
        return True

    except ImportError as e:
        print(f"\n⚠ ZeRank test skipped: Missing dependency ({e})")
        print("  Install with: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"\n✗ ZeRank test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_boosting():
    """Test boosting functionality."""
    print("\n" + "="*80)
    print("Testing Boosting Functionality")
    print("="*80)

    try:
        # Create reranker with aggressive boosting
        config = {
            "rerank_method": "cross-encoder",
            "rerank_top_k_before": 5,
            "rerank_top_k_after": 5,
            "rerank_boost_exact_matches": True,
            "rerank_use_query_expansion": True,
            "rerank_exact_match_boost": 2.0,  # Aggressive boosting
            "rerank_entity_boost": 1.5
        }

        reranker = RerankerStep(config)
        sample_data = create_sample_results()

        print(f"\nQuery: {sample_data.original_query}")
        print(f"Keywords for boosting: {sample_data.extracted_keywords}")
        print(f"Entities for boosting: {sample_data.extracted_entities}")

        # Process
        result = reranker.process(sample_data)

        print(f"\nResults with boosting applied:")
        for i, r in enumerate(result.results):
            has_keyword = any(kw.lower() in r.document_text.lower()
                            for kw in sample_data.extracted_keywords)
            has_entity = any(ent.lower() in r.document_text.lower()
                           for ent in sample_data.extracted_entities)

            boost_info = []
            if has_keyword:
                boost_info.append("keyword✓")
            if has_entity:
                boost_info.append("entity✓")

            boost_str = f" [{', '.join(boost_info)}]" if boost_info else ""

            print(f"  {i+1}. Score: {r.similarity_score:.3f}{boost_str}")
            print(f"     {r.document_text[:80]}...")

        print("\n✓ Boosting test passed!")
        return True

    except Exception as e:
        print(f"\n✗ Boosting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading from different sources."""
    print("\n" + "="*80)
    print("Testing Configuration Loading")
    print("="*80)

    try:
        # Test with minimal config
        config1 = {}
        reranker1 = RerankerStep(config1)
        print(f"✓ Default config loaded: method={reranker1.rerank_config.method}")

        # Test with custom config
        config2 = {
            "rerank_method": "zerank",
            "rerank_top_k_after": 15,
            "rerank_batch_size": 64
        }
        reranker2 = RerankerStep(config2)
        print(f"✓ Custom config loaded: method={reranker2.rerank_config.method}, "
              f"top_k_after={reranker2.rerank_config.top_k_after}")

        print("\n✓ Configuration test passed!")
        return True

    except Exception as e:
        print(f"\n✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("RERANKER IMPLEMENTATION TEST SUITE")
    print("="*80)

    tests = [
        ("Configuration Loading", test_config_loading),
        ("Cross-Encoder Reranker", test_cross_encoder_reranker),
        ("ZeRank Reranker", test_zerank_reranker),
        ("Boosting Functionality", test_boosting),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} failed with exception: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED/SKIPPED"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed successfully!")
        return 0
    elif passed > 0:
        print(f"\n⚠ Some tests failed or were skipped")
        return 1
    else:
        print(f"\n❌ All tests failed")
        return 2


if __name__ == "__main__":
    sys.exit(main())
