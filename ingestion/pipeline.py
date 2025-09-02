#!/usr/bin/env python3
"""
Ingestion Pipeline

Orchestrates the complete ingestion pipeline for chat data processing.

Pipeline Steps:
1. Parse raw chat files (WhatsApp, Facebook, etc.)
2. Break conversations into logical chunks
3. Clean and enhance chunk data
4. Final processing and validation
5. Extract entities using LLMs
6. Generate Cypher queries using AI (steps 6, 7, 8 run together)
7. Post-process and sanitize Cypher queries
8. Store processed data in Neo4j

Usage:
  # Run complete pipeline from step 1
  python3 ingestion/pipeline.py facebook-soldegen --model openai/gpt-4o-mini

  # Start from a specific step
  python3 ingestion/pipeline.py facebook-soldegen --model openai/gpt-4o-mini --step 3

  # Run only step 6 (which includes 7 and 8)
  python3 ingestion/pipeline.py facebook-soldegen --model openai/gpt-4o-mini --step 6

  # Dry run to see what would be done
  python3 ingestion/pipeline.py facebook-soldegen --model openai/gpt-4o-mini --dry-run

Environment Variables:
- OPENAI_API_KEY: OpenAI API key (required for OpenAI models)
- GROQ_API_KEY: Groq API key (required for Groq models)
- NEO4J_URI (default bolt://localhost:7687)
- NEO4J_USER (default neo4j)
- NEO4J_PASSWORD (default password)
- OLLAMA_URL (default http://localhost:11434)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def setup_imports():
    """Setup dynamic imports for step modules to handle relative imports properly."""
    # Add the root directory to Python path to enable proper imports
    root_dir = Path(__file__).parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    # Import step modules using full module paths
    from ingestion.steps.a_parsing import ChatParser
    from ingestion.steps.b_chunking import ChatChunker
    from ingestion.steps.c_chunk_processing import ChainOfResponsibilityProcessor
    # from ingestion.steps.d_post_processing import PostProcessor  # Currently unused
    from ingestion.steps.e_entity_extraction import extract_entities_with_langextract
    from ingestion.steps.f_cypher_query_generation import generate_cypher_for_chat
    from ingestion.steps.g_ingestion import post_process_cypher_data, ingest_processed_cypher

    return {
        "ChatParser": ChatParser,
        "ChatChunker": ChatChunker,
        "ChainOfResponsibilityProcessor": ChainOfResponsibilityProcessor,
        # "PostProcessor": PostProcessor,  # Currently unused
        "extract_entities_with_langextract": extract_entities_with_langextract,
        "generate_cypher_for_chat": generate_cypher_for_chat,
        "post_process_cypher_data": post_process_cypher_data,
        "ingest_processed_cypher": ingest_processed_cypher,
    }


def main():
    """Main pipeline orchestrator."""
    parser = argparse.ArgumentParser(description="Ingestion Pipeline")
    parser.add_argument("chat_name", help="Chat name to process")
    parser.add_argument(
        "--step", type=int, default=1, help="Starting step (1-6, default: 1)"
    )
    parser.add_argument(
        "--model",
        default="groq/qwen/qwen3-32b",
        help="AI model for Cypher generation (default: groq/qwen/qwen3-32b)",
    )

    args = parser.parse_args()

    print("🚀 Ingestion Pipeline")
    print(f"📊 Chat: {args.chat_name}")
    print(f"🔢 Starting from step: {args.step}")
    print(f"🤖 Model: {args.model}")

    # Setup imports
    try:
        imports = setup_imports()
    except ImportError as e:
        print(f"❌ Failed to import required modules: {e}")
        return 1

    # Extract the classes and functions we need
    ChatParser = imports["ChatParser"]
    ChatChunker = imports["ChatChunker"]
    ChainOfResponsibilityProcessor = imports["ChainOfResponsibilityProcessor"]
    extract_entities_with_langextract = imports["extract_entities_with_langextract"]
    generate_cypher_for_chat = imports["generate_cypher_for_chat"]
    post_process_cypher_data = imports["post_process_cypher_data"]
    ingest_processed_cypher = imports["ingest_processed_cypher"]

    # Step 1: Parsing
    if args.step <= 1:
        print(f"\n🔍 Step 1: Parsing - {args.chat_name}")
        try:
            parser = ChatParser()
            parser.process_single_chat(args.chat_name)
            print("✅ Step 1 completed")
        except Exception as e:
            print(f"❌ Step 1 failed: {e}")
            return 1

    # Step 2: Chunking
    if args.step <= 2:
        print(f"\n✂️ Step 2: Chunking - {args.chat_name}")
        try:
            strategy_config = {"type": "time", "gap_minutes": 10}
            strategy = ChatChunker.create_strategy_from_config(strategy_config)
            chunker = ChatChunker(strategy)
            chunker.process_single_chat(args.chat_name)
            print("✅ Step 2 completed")
        except Exception as e:
            print(f"❌ Step 2 failed: {e}")
            return 1

    # Step 3: Chunk Processing
    if args.step <= 3:
        print(f"\n🔄 Step 3: Chunk Processing - {args.chat_name}")
        try:
            config = ChainOfResponsibilityProcessor.get_default_processor_config()
            url_cache = ChainOfResponsibilityProcessor.load_url_cache(".")
            processor_chain = (
                ChainOfResponsibilityProcessor.create_processor_chain_from_config(
                    config, url_cache
                )
            )
            processor = ChainOfResponsibilityProcessor(".", processor_chain)
            processor.process_single_chat(args.chat_name)

            print("✅ Step 3 completed")
        except Exception as e:
            print(f"❌ Step 3 failed: {e}")
            return 1

    # Step 4: Post Processing
    if args.step <= 4:
        print(f"\n🔧 Step 4: Post Processing - {args.chat_name}")
        try:
            # postProcessor = PostProcessor()
            # success = postProcessor.export_chat_to_jsonl(args.chat_name)
            # if not success:
            #     print("❌ Step 4 failed")
            #     return 1
            print("✅ Step 4 completed")
        except Exception as e:
            print(f"❌ Step 4 failed: {e}")
            return 1

    # Step 5: Entity Extraction
    if args.step <= 5:
        print(f"\n🎯 Step 5: Entity Extraction - {args.chat_name}")
        try:
            extract_entities_with_langextract(args.chat_name)
            print("✅ Step 5 completed")
        except Exception as e:
            print(f"❌ Step 5 failed: {e}")
            return 1

    # Step 6: Cypher Generation
    if args.step <= 6:
        print(f"\n🔮 Step 6: Cypher Generation - {args.chat_name}")
        try:
            generate_cypher_for_chat(args.chat_name, args.model)
            print("✅ Step 6 completed")
        except Exception as e:
            print(f"❌ Step 6 failed: {e}")
            return 1

        # Step 7: Post-process Cypher queries
        print(f"\n🔧 Step 7: Cypher Post-Processing - {args.chat_name}")
        try:
            post_process_cypher_data(args.chat_name)
            print("✅ Step 7 completed")
        except Exception as e:
            print(f"❌ Step 7 failed: {e}")
            return 1

        # Step 8: Ingest into Neo4j
        print(f"\n🔗 Step 8: Neo4j Ingestion - {args.chat_name}")
        try:
            ingest_processed_cypher(args.chat_name)
            print("✅ Step 8 completed")
        except Exception as e:
            print(f"❌ Step 8 failed: {e}")
            return 1

    print("\n🎉 Pipeline completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
