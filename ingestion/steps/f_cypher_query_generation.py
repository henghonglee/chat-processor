#!/usr/bin/env python3
"""
Step 6: Cypher Query Generation

Generate Cypher queries from enriched JSONL using AI strategies (OpenAI, Ollama, or Groq).

This step takes enriched chat chunks and generates Cypher queries that can later be
post-processed and executed against a Neo4j database.

Inputs:
- Chat name (auto-finds enriched JSONL in chats-enriched/CHAT_NAME/)
- Model specification (provider/model format)

Outputs:
- Raw Cypher queries saved to cypher-queries/CHAT_NAME/

Environment Variables:
- OPENAI_API_KEY: OpenAI API key (required for OpenAI models)
- GROQ_API_KEY: Groq API key (required for Groq models)
- OLLAMA_URL: Ollama server URL (default http://localhost:11434)

Usage examples:
  # Generate Cypher using OpenAI
  python3 steps/6_cypher_query_generation.py facebook-soldegen openai/gpt-4o-mini

  # Generate using Ollama
  python3 steps/6_cypher_query_generation.py facebook-soldegen ollama/llama2

  # Process specific chunk
  python3 steps/6_cypher_query_generation.py facebook-soldegen openai/gpt-4o-mini --chunk-id chunk_001.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

# Load environment variables from .env file
from dotenv import load_dotenv

# Import Cypher Query Generation strategies
from ingestion.strategies.cypher_query_generation import (
    BaseCypherQueryAIStrategy,
    GroqCypherQueryAIStrategy,
    OllamaCypherQueryAIStrategy,
    OpenAICypherQueryAIStrategy,
)

load_dotenv()

# Add the parent directory to sys.path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


def load_jsonl(path: Path) -> Iterable[dict]:
    """Load JSONL file line by line."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_system_prompt() -> str:
    """Load the system prompt for Cypher generation."""
    prompts_dir = Path(__file__).parent.parent / "prompts"
    cypher_prompt_path = prompts_dir / "cypher_prompt.json"

    try:
        with open(cypher_prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    except FileNotFoundError:
        print(f"⚠️ System prompt file not found: {cypher_prompt_path}")
        return "You are a Neo4j Cypher expert. Generate Cypher queries from the provided chat data."


def build_user_prompt(content: str, metadata: dict) -> str:
    """Build user prompt from chunk content and metadata."""
    # Extract key metadata for the prompt
    meta_snippet = {
        k: v
        for k, v in metadata.items()
        if k in ["chat_name", "chunk_id", "timestamp_range", "source_file"]
    }

    return (
        "Metadata: " + json.dumps(meta_snippet, ensure_ascii=False) + "\n\n"
        "Chunk content follows (full):\n" + content
    )


def create_cypher_query_ai_strategy(model: str) -> BaseCypherQueryAIStrategy:
    """
    Create an appropriate Cypher Query AI strategy based on the model name.

    Args:
        model: Model name in format "provider/model_name" (e.g., "openai/gpt-4", "ollama/llama2")

    Returns:
        Configured Cypher Query AI strategy instance

    Raises:
        ValueError: If the model format is unsupported
    """
    if model.startswith("openai/"):
        model_name = model[7:]  # Strip "openai/" prefix
        return OpenAICypherQueryAIStrategy(model=model_name)
    elif model.startswith("ollama/"):
        model_name = model[7:]  # Strip "ollama/" prefix
        return OllamaCypherQueryAIStrategy(model=model_name)
    elif model.startswith("groq/"):
        model_name = model[5:]  # Strip "groq/" prefix
        return GroqCypherQueryAIStrategy(model=model_name)
    else:
        raise ValueError(
            f"Unsupported model format: {model}. Use 'provider/model' format (e.g., 'openai/gpt-4')"
        )


def generate_cypher_for_chunk(
    strategy: BaseCypherQueryAIStrategy, chunk: dict, system_prompt: str
) -> str:
    """
    Generate Cypher query for a single chunk using the AI strategy.

    Args:
        strategy: The AI strategy to use for generation
        chunk: The chat chunk data
        system_prompt: System prompt for the AI

    Returns:
        Generated Cypher query
    """
    content = chunk.get("page_content", "")
    metadata = chunk.get("metadata", {})

    if not content.strip():
        raise ValueError("Empty chunk content")

    user_prompt = build_user_prompt(content, metadata)

    return strategy.generate_cypher(system_prompt, user_prompt)


def save_cypher_query(cypher_query: str, output_path: Path) -> None:
    """Save generated Cypher query to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cypher_query)


def generate_cypher_for_chat(chat_name: str, model: str) -> dict:
    """
    Generate Cypher queries for a chat using AI strategies.

    Args:
        chat_name: Chat name to process
        model: AI model to use (format: provider/model)

    Returns:
        Dictionary mapping chunk_id to generated Cypher query string
    """
    print(f"🔍 Cypher Query Generation for: {chat_name}")
    print(f"🤖 Using model: {model}")

    # Find enriched JSONL file
    enriched_dir = Path("chats-enriched") / chat_name
    if not enriched_dir.exists():
        print(f"❌ Enriched directory not found: {enriched_dir}")
        return {}

    jsonl_files = list(enriched_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"❌ No JSONL files found in: {enriched_dir}")
        return {}

    if len(jsonl_files) > 1:
        print(f"⚠️ Multiple JSONL files found, using: {jsonl_files[0]}")

    jsonl_path = jsonl_files[0]
    print(f"📄 Loading chunks from: {jsonl_path}")

    # Create output directory
    output_dir = Path("cypher-queries") / chat_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load system prompt
    system_prompt = load_system_prompt()
    print(f"📝 Loaded system prompt ({len(system_prompt)} characters)")

    # Create AI strategy
    try:
        strategy = create_cypher_query_ai_strategy(model)
        provider_name = strategy.get_endpoint_info()["provider"].upper()
        print(f"🧠 Created {provider_name} strategy: {strategy}")
    except Exception as e:
        print(f"❌ Failed to create AI strategy: {e}")
        return {}

    # Process chunks
    cypher_results = {}
    chunks_processed = 0

    try:
        for chunk in load_jsonl(jsonl_path):
            metadata = chunk.get("metadata", {})
            chunk_id_current = metadata.get("chunk_id", f"chunk_{chunks_processed}")

            print(f"\n📦 Processing chunk: {chunk_id_current}")

            # Check if file already exists
            output_file = output_dir / f"{chunk_id_current}.cypher"
            if output_file.exists():
                print(f"⏭️ File already exists, skipping: {output_file}")
                # Still load the existing file content for the return dictionary
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        cypher_results[chunk_id_current] = f.read()
                    chunks_processed += 1
                except Exception as e:
                    print(f"⚠️ Could not read existing file {output_file}: {e}")
                continue

            try:
                # Generate Cypher query
                print(f"🧠 Generating Cypher using {provider_name}...")
                cypher_query = generate_cypher_for_chunk(strategy, chunk, system_prompt)
                print(f"✓ Generated Cypher ({len(cypher_query)} characters)")

                # Save to individual text file per chunk
                save_cypher_query(cypher_query, output_file)
                print(f"💾 Saved to: {output_file}")

                # Store in memory for pipeline processing
                cypher_results[chunk_id_current] = cypher_query
                chunks_processed += 1

            except Exception as e:
                print(f"❌ Failed to generate Cypher for {chunk_id_current}: {e}")
                # Print the full exception details for debugging
                import traceback

                print(f"🔍 Full error details: {traceback.format_exc()}")
                continue

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return {}
    except Exception as e:
        print(f"❌ Error processing chunks: {e}")
        return {}

    print("\n✅ Cypher generation complete!")
    print(f"📊 Processed: {chunks_processed} chunks")

    return cypher_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Cypher queries from enriched chat data"
    )
    parser.add_argument("chat_name", help="Chat name to process")
    parser.add_argument(
        "--model",
        default="groq/qwen2.5-72b-instruct",
        help="AI model for Cypher generation (default: groq/qwen2.5-72b-instruct)",
    )

    args = parser.parse_args()

    try:
        cypher_results = generate_cypher_for_chat(args.chat_name, args.model)
        if cypher_results:
            print(
                f"\n🎉 Successfully generated Cypher queries for {len(cypher_results)} chunks"
            )
        else:
            print("❌ No Cypher queries were generated")
            exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
