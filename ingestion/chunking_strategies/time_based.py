"""
Time-based chunking strategy.
"""

import importlib.util
from datetime import timedelta
from pathlib import Path

from .base import ChunkingStrategy

# Import parse_timestamp from the chunking module in steps
try:
    steps_dir = Path(__file__).parent.parent / "steps"
    chunking_path = steps_dir / "b_chunking.py"
    spec = importlib.util.spec_from_file_location("chunking", chunking_path)
    chunking_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chunking_module)
    parse_timestamp = chunking_module.parse_timestamp
except (ImportError, FileNotFoundError):
    # Fallback implementation if module not available
    def parse_timestamp(timestamp_str: str):
        """Fallback timestamp parser."""
        from datetime import datetime

        try:
            return datetime.fromisoformat(timestamp_str)
        except ValueError:
            return None


class TimeBasedChunkingStrategy(ChunkingStrategy):
    """Chunking strategy based on time gaps between messages."""

    def __init__(self, gap_minutes: int = 40):
        self.gap_threshold = timedelta(minutes=gap_minutes)

    def should_start_new_chunk(
        self, current_line: str, current_index: int, chunk_context: dict
    ) -> bool:
        """Start new chunk if time gap exceeds threshold."""
        if not current_line.strip():
            return False

        current_timestamp = parse_timestamp(current_line)
        if current_timestamp is None:
            return False

        last_timestamp = chunk_context.get("last_timestamp")
        if last_timestamp is None:
            return False

        time_gap = current_timestamp - last_timestamp
        return time_gap > self.gap_threshold
