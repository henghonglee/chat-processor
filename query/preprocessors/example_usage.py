#!/usr/bin/env python3
"""
Example usage of the modular query preprocessing system.

This script demonstrates how to use individual preprocessors and the preprocessor
pipeline with various configurations.
"""

from . import (
    EntityExtractor,
    IntentClassifier,
    KeywordExtractor,
    PreprocessorPipeline,
    TextCleaner,
)


def demonstrate_text_cleaner():
    """Demonstrate the TextCleaner preprocessor."""
    print("=" * 60)
    print("TEXT CLEANER DEMONSTRATION")
    print("=" * 60)

    # Example text with various cleaning challenges
    example_text = "Check out https://example.com for BTC info @crypto_guru #bitcoin! Extra   spaces   everywhere"

    print(f"Original text: {example_text}")
    print()

    # Default configuration
    cleaner = TextCleaner()
    result = cleaner.process(example_text)

    if result.success:
        print("Default cleaning:")
        print(f"  Cleaned: {result.data['cleaned_text']}")
        print(f"  Reduction: {result.data['reduction_ratio']:.2%}")
        print(f"  Operations: {result.metadata['applied_operations']}")

    print()

    # Custom configuration - more aggressive
    aggressive_config = {
        "remove_urls": True,
        "remove_mentions": True,
        "remove_hashtags": True,
        "remove_special_chars": True,
    }

    aggressive_cleaner = TextCleaner(aggressive_config)
    result = aggressive_cleaner.process(example_text)

    if result.success:
        print("Aggressive cleaning:")
        print(f"  Cleaned: {result.data['cleaned_text']}")
        print(f"  Reduction: {result.data['reduction_ratio']:.2%}")


def demonstrate_entity_extractor():
    """Demonstrate the EntityExtractor preprocessor."""
    print("\n" + "=" * 60)
    print("ENTITY EXTRACTOR DEMONSTRATION")
    print("=" * 60)

    example_queries = [
        "What did Shaun Lim say about BTC last week?",
        "Compare ETH and SOL performance at $2,500 price level",
        "Find conversations about Coinbase and Binance from yesterday",
        "Show me DeFi protocols mentioned by John on 2024-01-15",
    ]

    extractor = EntityExtractor()

    for query in example_queries:
        print(f"\nQuery: {query}")
        result = extractor.process(query)

        if result.success:
            entities = result.data["entities"]
            print(f"  Found {len(entities)} entities:")
            for entity in entities:
                print(
                    f"    - {entity['text']} ({entity['type']}) [confidence: {entity['confidence']:.2f}]"
                )

            breakdown = result.metadata["entity_breakdown"]
            if breakdown:
                print(f"  Breakdown: {breakdown}")


def demonstrate_intent_classifier():
    """Demonstrate the IntentClassifier preprocessor."""
    print("\n" + "=" * 60)
    print("INTENT CLASSIFIER DEMONSTRATION")
    print("=" * 60)

    example_queries = [
        "What is Bitcoin?",
        "Compare BTC and ETH prices",
        "How do people feel about the current market?",
        "Show me conversations between Alice and Bob",
        "When did the price start rising?",
        "Should I buy more SOL?",
        "Find all mentions of DeFi protocols",
        "What's the difference between Uniswap and SushiSwap?",
    ]

    classifier = IntentClassifier()

    for query in example_queries:
        print(f"\nQuery: {query}")
        result = classifier.process(query)

        if result.success:
            intent = result.data["primary_intent"]
            confidence = result.data["confidence"]
            all_scores = result.data["all_scores"]

            print(f"  Primary intent: {intent} (confidence: {confidence:.2f})")

            # Show top 3 intent scores
            sorted_scores = sorted(
                all_scores.items(), key=lambda x: x[1], reverse=True
            )[:3]
            print("  Top intents:")
            for intent_name, score in sorted_scores:
                if score > 0:
                    print(f"    - {intent_name}: {score:.2f}")


def demonstrate_keyword_extractor():
    """Demonstrate the KeywordExtractor preprocessor."""
    print("\n" + "=" * 60)
    print("KEYWORD EXTRACTOR DEMONSTRATION")
    print("=" * 60)

    example_queries = [
        "What are the latest trends in cryptocurrency trading?",
        "How does Bitcoin mining affect the environment?",
        "Find conversations about DeFi yield farming strategies",
        "Compare performance of different blockchain protocols",
    ]

    extractor = KeywordExtractor()

    for query in example_queries:
        print(f"\nQuery: {query}")
        result = extractor.process(query)

        if result.success:
            keywords = result.data["keywords"]
            categorized = result.data["categorized_keywords"]

            print(f"  Keywords ({len(keywords)}): {', '.join(keywords)}")

            # Show categorized keywords
            for category, kw_list in categorized.items():
                if kw_list:
                    kw_texts = [kw for kw, _ in kw_list]
                    print(f"  {category.title()}: {', '.join(kw_texts)}")


def demonstrate_preprocessor_pipeline():
    """Demonstrate the PreprocessorPipeline with different configurations."""
    print("\n" + "=" * 60)
    print("PREPROCESSOR PIPELINE DEMONSTRATION")
    print("=" * 60)

    example_query = "What did Shaun say about BTC trading on Coinbase last week?"
    print(f"Query: {example_query}")
    print()

    # Default configuration
    print("1. Default Configuration:")
    default_processor = PreprocessorPipeline()
    result = default_processor.process(example_query)

    if result.success:
        data = result.data
        print(
            f"   Intent: {data['intent']} (confidence: {data['intent_confidence']:.2f})"
        )
        print(f"   Query Type: {data['query_type']}")
        print(f"   Entities: {len(data['entities'])}")
        print(f"   Keywords: {len(data['keywords'])}")
        print(f"   Processing Time: {data['total_processing_time']:.3f}s")

    print()

    # Custom configuration - crypto-focused
    print("2. Crypto-Focused Configuration:")
    crypto_config = {
        "stage_configs": {
            "entity_extractor": {
                "extract_organizations": True,
                "confidence_threshold": 0.6,
                "crypto_symbols": ["BTC", "ETH", "SOL", "ADA", "DOT", "LINK"],
            },
            "keyword_extractor": {"max_keywords": 15, "boost_domain_terms": True},
            "intent_classifier": {"confidence_threshold": 0.4},
        }
    }

    crypto_processor = PreprocessorPipeline(crypto_config)
    result = crypto_processor.process(example_query)

    if result.success:
        data = result.data
        print(
            f"   Intent: {data['intent']} (confidence: {data['intent_confidence']:.2f})"
        )
        print(f"   Query Type: {data['query_type']}")
        print(f"   Entities: {[e['text'] for e in data['entities']]}")
        print(f"   Keywords: {data['keywords'][:5]}...")  # First 5 keywords

        # Show categorized keywords
        categorized = data["categorized_keywords"]
        for category in ["crypto", "finance"]:
            if category in categorized and categorized[category]:
                kw_texts = [kw for kw, _ in categorized[category]]
                print(f"   {category.title()} keywords: {kw_texts}")

    print()

    # Minimal configuration - only essential stages
    print("3. Minimal Configuration:")
    minimal_config = {
        "pipeline_stages": ["text_cleaner", "entity_extractor"],
        "stage_configs": {
            "entity_extractor": {
                "extract_crypto": True,
                "extract_people": True,
                "extract_dates": False,
                "extract_organizations": False,
                "extract_prices": False,
            }
        },
    }

    minimal_processor = PreprocessorPipeline(minimal_config)
    result = minimal_processor.process(example_query)

    if result.success:
        data = result.data
        print(f"   Stages Run: {data['pipeline_stages_run']}")
        print(f"   Entities: {[e['text'] for e in data['entities']]}")
        print(f"   Processing Time: {data['total_processing_time']:.3f}s")


def demonstrate_advanced_features():
    """Demonstrate advanced features like context passing and error handling."""
    print("\n" + "=" * 60)
    print("ADVANCED FEATURES DEMONSTRATION")
    print("=" * 60)

    # Show how individual processors can use context from previous stages
    print("1. Context Passing Between Stages:")

    text = "Check out https://example.com - Shaun thinks BTC will moon! #crypto"

    # First clean the text
    cleaner = TextCleaner()
    clean_result = cleaner.process(text)

    if clean_result.success:
        cleaned_text = clean_result.data["cleaned_text"]
        print(f"   Original: {text}")
        print(f"   Cleaned:  {cleaned_text}")

        # Pass cleaned text as context to entity extractor
        context = {"cleaned_text": cleaned_text}
        extractor = EntityExtractor()
        entity_result = extractor.process(text, context)

        if entity_result.success:
            entities = entity_result.data["entities"]
            print(f"   Entities from cleaned text: {[e['text'] for e in entities]}")

            # Pass entities to keyword extractor to avoid duplication
            keyword_context = {"cleaned_text": cleaned_text, "entities": entities}
            keyword_extractor = KeywordExtractor()
            keyword_result = keyword_extractor.process(text, keyword_context)

            if keyword_result.success:
                keywords = keyword_result.data["keywords"]
                print(f"   Keywords (filtered): {keywords}")
                print(
                    f"   Entity overlap filtered: {keyword_result.metadata['entity_overlap_filtered']}"
                )

    print()

    # Show error handling
    print("2. Error Handling:")

    # Example with empty text
    error_processor = PreprocessorPipeline()
    error_result = error_processor.process("")

    print(f"   Empty text processing: success={error_result.success}")
    if not error_result.success:
        print(f"   Error: {error_result.error}")

    # Example with fail_on_stage_error configuration
    fail_fast_config = {
        "fail_on_stage_error": True,
        "stage_configs": {
            "entity_extractor": {
                "confidence_threshold": 2.0  # Invalid threshold to cause error
            }
        },
    }

    fail_processor = PreprocessorPipeline(fail_fast_config)
    fail_result = fail_processor.process("example query")

    print(f"   Fail-fast mode: success={fail_result.success}")
    if not fail_result.success:
        print(f"   Error: {fail_result.error}")


def main():
    """Main demonstration function."""
    print("Query Preprocessors Demonstration")
    print("=" * 80)

    try:
        demonstrate_text_cleaner()
        demonstrate_entity_extractor()
        demonstrate_intent_classifier()
        demonstrate_keyword_extractor()
        demonstrate_preprocessor_pipeline()
        demonstrate_advanced_features()

        print("\n" + "=" * 80)
        print("Demonstration completed successfully!")
        print("Check the individual preprocessor files for more configuration options.")

    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
