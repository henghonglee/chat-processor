#!/usr/bin/env python3
"""
Entity extraction module using Gemini LangExtract.

This module provides functionality to extract entities from chat content
and enrich JSONL files with extracted entities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def extract_entities_with_langextract(chat_name: str) -> None:
    """
    Extract entities from chat content using Gemini LangExtract.
    Processes chunks for the specified chat and enriches them with extracted entities.
    """
    print("🧠 Starting entity extraction pass...")

    base_dir = Path(".")
    input_file = base_dir / "chats-processed" / chat_name / f"{chat_name}_chunks.jsonl"

    # Output to chats-enriched directory
    output_dir = base_dir / "chats-enriched" / chat_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{chat_name}_chunks_enriched.jsonl"

    print(f"📂 Input: {input_file}")
    print(f"📂 Output: {output_file}")

    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return

    try:
        import langextract as lx
    except ImportError:
        print("❌ langextract not installed. Install with: pip install langextract")
        return

    # Load prompt configuration from prompts folder
    try:
        from ingestion.prompts.langextract_entities import get_langextract_config

        prompt, examples, model_id = get_langextract_config()
    except ImportError:
        print("⚠️ Could not load prompt config, using fallback")
        # Fallback prompt
        prompt = "Extract entities from this text."
        examples = []
        model_id = "gemini-2.5-flash"

    # Load all documents from the input chunks file
    all_docs = load_jsonl(input_file)

    if not all_docs:
        print("❌ No documents found in input file")
        return

    # Create output file if it doesn't exist
    if not output_file.exists():
        output_file.touch()

    # Load existing documents from output file to check what's already processed
    existing_docs = load_jsonl(output_file)
    existing_docs_dict = {doc.get("chunk_id"): doc for doc in existing_docs}

    # Process documents and track what was processed
    processed_chunk_ids = set()

    for doc in all_docs:
        current_chunk_id = doc.get("chunk_id")

        # Check if this chunk already has extracted_entities in existing output
        existing_doc = existing_docs_dict.get(current_chunk_id)
        if (
            existing_doc
            and "extracted_entities" in existing_doc
            and existing_doc["extracted_entities"] is not None
        ):
            print(
                f"✓ Chunk {current_chunk_id} already has extracted entities, skipping"
            )
            processed_chunk_ids.add(current_chunk_id)
        else:
            print(f"\n--- Processing chunk {current_chunk_id} ---")

            page_content = doc.get("page_content", "")
            print(
                f"📄 Extracting entities from chunk: {current_chunk_id} from chat '{chat_name}'"
            )

            # Extract entities
            metadata = doc.get("metadata", {})
            entities = _extract_entities_from_text(
                page_content, metadata, prompt, examples, model_id
            )
            print(f"✓ Extracted {len(entities)} entities")

            # Enrich the document with extracted entities
            doc["extracted_entities"] = entities

            # Save enriched document to output file by appending
            with open(output_file, "a", encoding="utf-8") as outfile:
                outfile.write(json.dumps(doc, ensure_ascii=False) + "\n")

            processed_chunk_ids.add(current_chunk_id)

    print(
        f"\n🎉 Entity extraction completed! Processed {len(processed_chunk_ids)} chunks"
    )
    print(f"📄 Enriched JSONL saved to: {output_file}")


def _extract_entities_from_text(
    page_content: str,
    metadata: dict = None,
    prompt: str = None,
    examples: list = None,
    model_id: str = None,
) -> List[Dict[str, Any]]:
    """
    Extract entities from text using Gemini LangExtract.
    Returns a list of extracted entities with their attributes.
    """
    if metadata is None:
        metadata = {}

    try:
        import langextract as lx

        # Run entity extraction
        result = lx.extract(
            text_or_documents=page_content,
            prompt_description=prompt,
            examples=examples,
            model_id=model_id,
            debug=False,
        )
        # Print extracted entities for debugging
        print("📍 Extracted entities for this chunk:")
        if hasattr(result, "extractions") and result.extractions:
            for i, extraction in enumerate(result.extractions, 1):
                print(
                    f"  {i}. {extraction.extraction_class}: '{extraction.extraction_text}'"
                )
        else:
            print("  No entities extracted (this is fine)")

        # Convert to our format
        entities = []
        if hasattr(result, "extractions") and result.extractions:
            for extraction in result.extractions:
                entity = {
                    "type": extraction.extraction_class,
                    "text": extraction.extraction_text,
                    "attributes": (
                        extraction.attributes
                        if hasattr(extraction, "attributes")
                        else {}
                    ),
                }
                entities.append(entity)

        return entities

    except Exception as e:
        print(f"⚠️ Entity extraction failed: {e}")
        import traceback

        traceback.print_exc()
        return []


def load_jsonl(path: Path) -> List[dict]:
    """Load JSONL file and return all documents."""
    docs = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                docs.append(json.loads(line))
    return docs


def enrich_jsonl_with_entities(
    input_jsonl: Path,
    output_jsonl: Path,
    chat_name: str,
    chunk_id: Optional[str] = None,
) -> None:
    """
    Load chunks file, extract entities for specified chunk_id or all chunks, and save enriched version.
    """
    print("🧠 Starting entity extraction pass...")
    print(f"📂 Input: {input_jsonl}")
    print(f"📂 Output: {output_jsonl}")

    # Load all documents from the input chunks file
    all_docs = load_jsonl(input_jsonl)

    if not all_docs:
        print("❌ No documents found in input file")
        return

    # Ensure output directory exists
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Create output file if it doesn't exist
    if not output_jsonl.exists():
        output_jsonl.touch()

    # Load existing documents from output file to check what's already processed
    existing_docs = load_jsonl(output_jsonl)
    existing_docs_dict = {doc.get("chunk_id"): doc for doc in existing_docs}

    # Process documents and track what was processed
    processed_chunk_ids = set()

    for doc in all_docs:
        current_chunk_id = doc.get("chunk_id")

        # Check if this document should be processed
        should_process = False
        if chunk_id:
            should_process = str(current_chunk_id) == str(chunk_id)
        else:
            should_process = True

        if should_process:
            # Check if this chunk already has extracted_entities in existing output
            existing_doc = existing_docs_dict.get(current_chunk_id)
            if (
                existing_doc
                and "extracted_entities" in existing_doc
                and existing_doc["extracted_entities"] is not None
            ):
                print(
                    f"✓ Chunk {current_chunk_id} already has extracted entities, skipping"
                )
                processed_chunk_ids.add(current_chunk_id)
            else:
                print(f"\n--- Processing chunk {current_chunk_id} ---")

                page_content = doc.get("page_content", "")
                print(
                    f"📄 Extracting entities from chunk: {current_chunk_id} from chat '{chat_name}'"
                )

                # Extract entities and enrich the document
                metadata = doc.get("metadata", {})
                entities = _extract_entities_from_text(page_content, metadata)
                print(f"✓ Extracted {len(entities)} entities")

                # Enrich the document with extracted entities
                doc["extracted_entities"] = entities

                # Save enriched document to output file by appending
                with open(output_jsonl, "a", encoding="utf-8") as outfile:
                    outfile.write(json.dumps(doc, ensure_ascii=False) + "\n")

                processed_chunk_ids.add(current_chunk_id)

    print(
        f"\n🎉 Entity extraction completed! Processed {len(processed_chunk_ids)} chunks"
    )
    print(f"📄 Enriched JSONL saved to: {output_jsonl}")


def main():
    """Main function to run entity extraction on processed chunks."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Extract entities from processed chat chunks using Gemini LangExtract"
    )
    parser.add_argument(
        "chat-name",
        help="Chat name to process (reads from chats-processed/CHAT_NAME/)",
    )
    parser.add_argument(
        "--chunk-id", help="Process only a specific chunk ID (e.g., chunk_001.txt)"
    )

    args = parser.parse_args()

    # Convert kebab-case to underscore for variable access
    chat_name = getattr(args, "chat-name")

    base_dir = Path(".")
    input_file = base_dir / "chats-processed" / chat_name / f"{chat_name}_chunks.jsonl"

    # Output to chats-processed (same directory)
    output_dir = base_dir / "chats-enriched" / chat_name
    output_file = output_dir / f"{chat_name}_chunks_enriched.jsonl"

    # Include chunk_id in processing message if specified
    if args.chunk_id:
        print(f"🧠 Starting entity extraction for {chat_name} - chunk {args.chunk_id}")
    else:
        print(f"🧠 Starting entity extraction for {chat_name}")

    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return

    enrich_jsonl_with_entities(input_file, output_file, chat_name, args.chunk_id)


if __name__ == "__main__":
    main()
