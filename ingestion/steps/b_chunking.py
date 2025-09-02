#!/usr/bin/env python3
"""
Chunking Module with Strategy Pattern

Splits messages when certain criteria are met using pluggable strategies.
Supports time-based chunking, message count chunking, and custom strategies.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Import chunking strategies
from ingestion.strategies.chunking import (
    ChunkingStrategy,
    HybridChunkingStrategy,
    LineCountChunkingStrategy,
    MessageCountChunkingStrategy,
    TimeBasedChunkingStrategy,
)

# WhatsApp timestamp pattern: [DD/MM/YY, HH:MM:SS AM/PM]
WHATSAPP_TIMESTAMP_RE = re.compile(
    r"^\[(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{1,2}:\d{1,2}) (AM|PM)\]"
)

# Unix timestamp pattern: [1234567890]
UNIX_TIMESTAMP_RE = re.compile(r"^\[(\d{10,})\]")


def parse_timestamp(line: str) -> Optional[datetime]:
    """
    Parse timestamp from a message line.

    Supports both Unix timestamp format and WhatsApp format.

    Args:
        line: Message line that may contain a timestamp

    Returns:
        datetime object if timestamp found, None otherwise
    """
    # Try Unix timestamp first (most common in processed data)
    unix_result = _parse_unix_timestamp(line)
    if unix_result:
        return unix_result

    # Try WhatsApp format (legacy)
    whatsapp_result = _parse_whatsapp_timestamp(line)
    if whatsapp_result:
        return whatsapp_result

    return None


def _parse_unix_timestamp(line: str) -> Optional[datetime]:
    """Parse Unix timestamp from a message line."""
    clean_line = line.strip()
    match = UNIX_TIMESTAMP_RE.match(clean_line)
    if not match:
        return None

    try:
        unix_timestamp = int(match.group(1))
        return datetime.fromtimestamp(unix_timestamp)
    except (ValueError, OverflowError, OSError):
        return None


def _parse_whatsapp_timestamp(line: str) -> Optional[datetime]:
    """Parse WhatsApp timestamp from a message line (legacy format)."""
    # Clean invisible characters like right-to-left marks and narrow no-break spaces
    clean_line = line.strip().lstrip("\u200e\u200f\u202d\u202e").replace("\u202f", " ")
    match = WHATSAPP_TIMESTAMP_RE.match(clean_line)
    if not match:
        return None

    date_str, time_str, am_pm = match.groups()
    try:
        # Parse date components
        day, month, year = date_str.split("/")
        year = int(year) + 2000 if int(year) < 50 else int(year) + 1900

        # Parse time components
        hour, minute, second = time_str.split(":")
        hour = int(hour)

        # Convert to 24-hour format
        if am_pm == "PM" and hour != 12:
            hour += 12
        elif am_pm == "AM" and hour == 12:
            hour = 0

        return datetime(year, int(month), int(day), hour, int(minute), int(second))
    except (ValueError, IndexError):
        return None


class ChatChunker:
    """Lightweight chunking engine that uses pluggable strategies."""

    def __init__(self, strategy: ChunkingStrategy):
        self.strategy = strategy

    def chunk_messages(self, lines: List[str]) -> List[Tuple[int, List[str]]]:
        """
        Create chunks using the configured strategy.

        Args:
            lines: List of message lines

        Returns:
            List of (start_index, chunk_lines) tuples
        """
        if not lines:
            return []

        chunks = []
        current_chunk = []
        chunk_context = {}

        for line_idx, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                continue

            # Check if we should start a new chunk
            if current_chunk and self.strategy.should_start_new_chunk(
                line, line_idx, chunk_context
            ):

                # Finalize current chunk
                chunks.append((chunk_context["start_index"], current_chunk))

                # Start new chunk
                current_chunk = [line]
                chunk_context = self.strategy.initialize_chunk_context(line, line_idx)
            else:
                # Add to current chunk
                current_chunk.append(line)

                # Initialize context if this is the first line
                if not chunk_context:
                    chunk_context = self.strategy.initialize_chunk_context(
                        line, line_idx
                    )
                else:
                    self.strategy.update_chunk_context(chunk_context, line, line_idx)

        # Add final chunk if it has content
        if current_chunk:
            chunks.append((chunk_context["start_index"], current_chunk))

        return chunks

    def chunk_jsonl_file(self, jsonl_path: str, output_dir: str) -> None:
        """
        Chunk a JSONL file containing chat messages.

        Args:
            jsonl_path: Path to input JSONL file
            output_dir: Directory to save chunks
        """
        input_path = Path(jsonl_path)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Read JSONL file
        messages = []
        context_data = None

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                if record.get("type") == "context":
                    context_data = record.get("data", {})
                elif record.get("type") == "message":
                    # Convert back to text format for chunking
                    timestamp = record.get("timestamp")
                    sender = record.get("sender", "Unknown")
                    content = record.get("content", "")
                    message_type = record.get("message_type", "text")

                    if message_type == "text":
                        text_line = f"[{timestamp}] {sender}: {content}"
                    elif message_type == "image":
                        text_line = f"[{timestamp}] {sender}: <image: {content}>"
                    elif message_type == "video":
                        text_line = f"[{timestamp}] {sender}: <video: {content}>"
                    elif message_type == "audio":
                        text_line = f"[{timestamp}] {sender}: <audio: {content}>"
                    elif message_type == "file":
                        text_line = f"[{timestamp}] {sender}: <file: {content}>"
                    elif message_type == "reaction":
                        text_line = f"[{timestamp}] {content}"
                    else:
                        text_line = (
                            f"[{timestamp}] {sender}: <{message_type}: {content}>"
                        )

                    messages.append(text_line)

        # Chunk the messages
        chunks = self.chunk_messages(messages)

        print(f"📊 Created {len(chunks)} chunks from {len(messages)} messages")

        # Save chunks as JSONL file in chat subfolder
        chat_name = input_path.stem
        chunks_filename = f"{chat_name}_chunks.jsonl"
        chunks_path = output_path / chunks_filename

        with open(chunks_path, "w", encoding="utf-8") as f:
            # Write each chunk as a separate record using the new schema
            for chunk_idx, (start_index, chunk_lines) in enumerate(chunks):
                # Join all chunk lines into page_content
                page_content = "\n".join(chunk_lines)

                chunk_record = {
                    "chunk_id": chunk_idx,
                    "page_content": page_content,
                    "metadata": {
                        "chat_name": chat_name,
                        "message_count": len(chunk_lines),
                        "context": context_data if context_data else {},
                    },
                }
                f.write(json.dumps(chunk_record) + "\n")

        print(f"💾 Saved chunks to: {chunks_path}")

    @staticmethod
    def create_strategy_from_config(config: dict) -> ChunkingStrategy:
        """Create a chunking strategy from configuration dictionary."""
        strategy_type = config.get("type", "time")

        if strategy_type == "time":
            gap_minutes = config.get("gap_minutes", 40)
            return TimeBasedChunkingStrategy(gap_minutes)

        elif strategy_type == "message_count":
            max_messages = config.get("max_messages", 100)
            return MessageCountChunkingStrategy(max_messages)

        elif strategy_type == "line_count":
            max_lines = config.get("max_lines", 200)
            return LineCountChunkingStrategy(max_lines)

        elif strategy_type == "hybrid":
            strategy_configs = config.get("strategies", [])
            strategies = [
                ChatChunker.create_strategy_from_config(sc) for sc in strategy_configs
            ]
            return HybridChunkingStrategy(strategies)

        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    def process_single_chat(self, chat_name: str, base_dir: str = ".") -> None:
        """Process a single chat by name."""
        base_path = Path(base_dir)
        input_dir = base_path / "chats-parsed"
        output_dir = base_path / "chats-parsed-chunked"

        # Find the chat file
        jsonl_file = input_dir / f"{chat_name}.jsonl"

        if not jsonl_file.exists():
            print(f"❌ Chat file not found: {jsonl_file}")
            print("💡 Available chats:")
            available = list(input_dir.glob("*.jsonl"))
            for f in available:
                print(f"  - {f.stem}")
            return

        print(f"🎯 Processing chat: {chat_name}")

        # Create output directory
        output_dir.mkdir(exist_ok=True)
        chat_output_dir = output_dir / chat_name

        try:
            # Chunk the file
            self.chunk_jsonl_file(str(jsonl_file), str(chat_output_dir))
            print(f"✅ Chunked successfully: {chat_output_dir}")

        except Exception as e:
            print(f"❌ Error processing {chat_name}: {e}")
