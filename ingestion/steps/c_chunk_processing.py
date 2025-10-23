#!/usr/bin/env python3
"""
Chunk Processing Module with Chain of Responsibility Pattern

Processes all chunks in chats-parsed-chunked/ using a configurable chain of processors
and saves results to chats-processed/ in JSONL format.
"""

import csv
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

# Import chunk processing strategies from the same ingestion package
from ingestion.strategies.chunk_processing.base import BaseProcessor
from ingestion.strategies.chunk_processing.image_processor import ImageProcessor
from ingestion.strategies.chunk_processing.link_cleaner import LinkCleanerProcessor
from ingestion.strategies.chunk_processing.twitter_processor import TwitterProcessor
from ingestion.strategies.chunk_processing.url_expander import URLExpanderProcessor


class ChainOfResponsibilityProcessor:
    """Processes chunks with a configurable chain of processors."""

    def __init__(
        self, base_dir: str = ".", processor_chain: Optional[List[BaseProcessor]] = None
    ):
        self.base_dir = Path(base_dir)
        self.input_dir = self.base_dir / "chats-parsed-chunked"
        self.output_dir = self.base_dir / "chats-processed"

        # Setup logging first
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [Thread-%(thread)d] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)

        # Suppress trafilatura warnings that generate "WARNING: discarding data: None"
        logging.getLogger('trafilatura').setLevel(logging.ERROR)
        logging.getLogger('trafilatura.core').setLevel(logging.ERROR)
        logging.getLogger('trafilatura.utils').setLevel(logging.ERROR)
        logging.getLogger('trafilatura.htmlprocessing').setLevel(logging.ERROR)

        # Setup default processor chain if not provided
        if processor_chain is None:
            self.processor_chain = self._create_default_chain()
        else:
            self.processor_chain = self._build_chain(processor_chain)

    @staticmethod
    def get_default_processor_config() -> Dict:
        """Get the default processor configuration (inlined from processor_config.json)."""
        return {
            "processors": [
                {"type": "link_cleaner", "params": {}},
                {"type": "url_expander", "params": {"timeout": 15}},
                {"type": "twitter", "params": {"timeout": 15}},
                {"type": "image", "params": {}},
            ]
        }

    @staticmethod
    def create_processor_chain_from_config(
        config: Dict, url_cache: Optional[dict] = None
    ) -> List[BaseProcessor]:
        """Create processor chain from configuration."""
        processors = []

        for processor_config in config.get("processors", []):
            processor_type = processor_config.get("type")
            params = processor_config.get("params", {})

            if processor_type == "link_cleaner":
                processors.append(LinkCleanerProcessor())
            elif processor_type == "url_expander":
                timeout = params.get("timeout", 15)
                processors.append(
                    URLExpanderProcessor(timeout=timeout, cache=url_cache)
                )
            elif processor_type == "twitter":
                timeout = params.get("timeout", 15)
                processors.append(TwitterProcessor(timeout=timeout, cache=url_cache))
            elif processor_type == "image":
                processors.append(ImageProcessor())
            else:
                logging.warning(f"Unknown processor type: {processor_type}")

        return processors

    def process_single_chat(self, chat_name: str, max_workers: int = 4) -> None:
        """Process chunks for a single chat directory."""
        self.logger.info(f"🎯 Processing single chat: {chat_name}")

        input_chat_dir = self.input_dir / chat_name
        output_chat_dir = self.output_dir / chat_name

        if not input_chat_dir.exists():
            self.logger.error(f"❌ Chat directory not found: {input_chat_dir}")
            return

        # Create output directory
        output_chat_dir.mkdir(parents=True, exist_ok=True)

        # Get chunk files (new format: {chat_name}_chunks.jsonl)
        chunk_files = list(input_chat_dir.glob("*_chunks.jsonl"))

        if not chunk_files:
            self.logger.warning(f"⚠️ No chunk files found in {input_chat_dir}")
            return

        self.logger.info(f"📊 Found {len(chunk_files)} chunk files in {chat_name}")

        # Process chunks in parallel
        processed_chunks = self._process_chunks_parallel(
            chunk_files, output_chat_dir, max_workers, chat_name
        )

        self.logger.info(
            f"✅ Completed {chat_name}: {processed_chunks} chunks processed"
        )

    @staticmethod
    def load_url_cache(base_dir: str = ".") -> Dict[str, str]:
        """Load URL cache from CSV file."""
        cache = {}
        cache_file = Path(base_dir) / "url_cache.csv"

        if not cache_file.exists():
            logging.warning(f"⚠️ URL cache file not found: {cache_file}")
            return cache

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "url" in row and "content" in row:
                        cache[row["url"]] = row["content"]

            logging.info(f"✅ Loaded {len(cache)} entries from URL cache")
            return cache

        except Exception as e:
            logging.error(f"❌ Failed to load URL cache: {e}")
            return cache

    def _load_url_cache(self) -> Dict[str, str]:
        """Load URL cache from CSV file (instance method wrapper)."""
        return self.load_url_cache(str(self.base_dir))

    def _create_default_chain(self) -> Optional[BaseProcessor]:
        """Create the default processing chain."""
        # Load URL cache for processors that support it
        url_cache = self._load_url_cache()

        processors = [
            LinkCleanerProcessor(),
            URLExpanderProcessor(timeout=15, cache=url_cache),
            TwitterProcessor(timeout=15, cache=url_cache),
            ImageProcessor(),
        ]
        return self._build_chain(processors)

    def _build_chain(self, processors: List[BaseProcessor]) -> Optional[BaseProcessor]:
        """Build a chain of responsibility from list of processors."""
        if not processors:
            return None

        # Link processors together
        for i in range(len(processors) - 1):
            processors[i].set_next(processors[i + 1])

        return processors[0]  # Return the first processor in the chain

    def _process_chunks_parallel(
        self,
        chunk_files: List[Path],
        output_dir: Path,
        max_workers: int,
        chat_name: str,
    ) -> int:
        """Process chunk files in parallel."""

        processed_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunk processing tasks
            future_to_chunk = {
                executor.submit(
                    self._process_single_chunk, chunk_file, output_dir, chat_name
                ): chunk_file
                for chunk_file in chunk_files
            }

            # Collect results
            for future in as_completed(future_to_chunk):
                chunk_file = future_to_chunk[future]
                chunk_name = chunk_file.name

                try:
                    success = future.result()
                    if success:
                        processed_count += 1
                        if processed_count % 50 == 0:  # Progress every 50 chunks
                            self.logger.info(
                                f"📈 {chat_name}: {processed_count}/{len(chunk_files)} chunks processed"
                            )
                except Exception as e:
                    self.logger.error(f"❌ Error processing {chunk_name}: {e}")

        return processed_count

    def _process_single_chunk(
        self, chunk_file: Path, output_dir: Path, chat_name: str
    ) -> bool:
        """Process a single chunk file with new JSONL format."""

        try:
            # Read chunk JSONL file - new format has one chunk record per line
            processed_chunks = []

            with open(chunk_file, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        chunk_record = json.loads(line)

                        # Process the page_content text through the processor chain
                        if "page_content" in chunk_record:
                            original_content = chunk_record["page_content"]
                            processed_content = original_content

                            # Apply the processor chain to the text content
                            if self.processor_chain:
                                processed_content = self.processor_chain.process(
                                    original_content
                                )

                            # Update the chunk record with processed content
                            chunk_record["page_content"] = processed_content

                        processed_chunks.append(chunk_record)

                    except json.JSONDecodeError:
                        continue

            if not processed_chunks:
                # Empty chunk file, create empty output file
                output_file = output_dir / chunk_file.name
                with open(output_file, "w", encoding="utf-8") as f:
                    pass
                return True

            # Write processed chunks
            output_file = output_dir / chunk_file.name
            with open(output_file, "w", encoding="utf-8") as f:
                for chunk_record in processed_chunks:
                    f.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to process {chunk_file.name}: {e}")
            return False
