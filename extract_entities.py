#!/usr/bin/env python3
"""
Standalone entity extraction script.

This script processes JSONL files to extract entities using Gemini LangExtract
and outputs enriched JSONL files.

Environment
- GOOGLE_API_KEY: Google Gemini API key (required for entity extraction)

Usage examples
  # Extract entities from JSONL
  python3 extract_entities.py --jsonl exports/facebook-soldegen.jsonl \
    --output exports/facebook-soldegen-enriched.jsonl

  # Extract entities with filtering
  python3 extract_entities.py --jsonl exports/facebook-soldegen.jsonl \
    --output exports/facebook-soldegen-enriched.jsonl \
    --chat-name facebook-soldegen --chunk-id chunk_001.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from entity_extraction import enrich_jsonl_with_entities


def main() -> None:
    # Load .env if present
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        print("✓ Loaded environment variables from .env file")
    except Exception as e:
        print(f"⚠ Could not load .env file: {e}")
        pass

    parser = argparse.ArgumentParser(description="Extract entities from JSONL chat documents")
    parser.add_argument("--jsonl", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", required=True, help="Path to output enriched JSONL file")
    parser.add_argument("--chat-name", help="Filter by chat name (folder name)")
    parser.add_argument("--chunk-id", help="Filter by chunk id (e.g., chunk_001.txt)")

    args = parser.parse_args()

    print(f"📁 Input JSONL: {args.jsonl}")
    print(f"📄 Output JSONL: {args.output}")
    
    input_path = Path(args.jsonl).resolve()
    output_path = Path(args.output).resolve()
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        raise FileNotFoundError(input_path)
    
    print(f"✓ Found input JSONL file: {input_path}")

    # Run entity extraction
    enrich_jsonl_with_entities(
        input_jsonl=input_path,
        output_jsonl=output_path,
        chat_name=args.chat_name,
        chunk_id=args.chunk_id
    )

    print("\n🎉 Entity extraction completed!")
    print(f"📄 Enriched JSONL saved to: {output_path}")


if __name__ == "__main__":
    main()
