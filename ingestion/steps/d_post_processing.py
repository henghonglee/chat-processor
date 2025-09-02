#!/usr/bin/env python3
"""
Post-Processing Module

Handles final processing steps after chunk processing is complete.
Includes operations like aggregation, statistics, and final output formatting.
"""

import argparse
import json
import logging
from pathlib import Path


class PostProcessor:
    """Post-processing engine for final operations on processed chunks."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.input_dir = self.base_dir / "chats-processed"
        self.output_dir = self.base_dir / "chats-enriched"

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)

    def export_chat_to_jsonl(
        self, chat_name: str, include_context: bool = True
    ) -> None:
        """Export a single chat to a consolidated JSONL file."""

        chat_dir = self.input_dir / chat_name

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        output_file = self.output_dir / f"{chat_name}.jsonl"
        chunk_files = sorted(chat_dir.glob("*_chunks.jsonl"))

        if not chunk_files:
            self.logger.warning(f"⚠️ No chunk files found in {chat_dir}")
            return

        self.logger.info(f"📦 Exporting {chat_name} to {output_file}")

        messages_written = 0
        context_written = False

        with open(output_file, "w", encoding="utf-8") as outf:
            for chunk_file in chunk_files:
                try:
                    with open(chunk_file, "r", encoding="utf-8") as inf:
                        for line in inf:
                            if not line.strip():
                                continue

                            try:
                                chunk_record = json.loads(line)

                                # Write context only once from metadata
                                if include_context and not context_written:
                                    context_data = chunk_record.get("metadata", {}).get(
                                        "context"
                                    )
                                    if context_data:
                                        context_record = {
                                            "type": "context",
                                            "data": context_data,
                                        }
                                        outf.write(
                                            json.dumps(
                                                context_record, ensure_ascii=False
                                            )
                                            + "\n"
                                        )
                                        context_written = True

                                # Write the chunk as a content record
                                if "page_content" in chunk_record:
                                    content_record = {
                                        "type": "chunk",
                                        "chunk_id": chunk_record.get("chunk_id"),
                                        "page_content": chunk_record.get(
                                            "page_content"
                                        ),
                                        "metadata": chunk_record.get("metadata", {}),
                                    }
                                    outf.write(
                                        json.dumps(content_record, ensure_ascii=False)
                                        + "\n"
                                    )
                                    messages_written += 1

                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    self.logger.warning(f"⚠️ Error reading {chunk_file}: {e}")
                    continue

        self.logger.info(f"✅ Exported {messages_written} messages to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Post-processing operations for processed chunks"
    )
    parser.add_argument("chat_name", help="Name of the chat to process")

    parser.add_argument(
        "--base-dir", default=".", help="Base directory containing chats-processed/"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    processor = PostProcessor(args.base_dir)

    # Run specific operations
    processor.export_chat_to_jsonl(args.chat_name)


if __name__ == "__main__":
    main()
