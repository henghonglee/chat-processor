"""
Time-based chunking strategy.

Starts new chunks when time gaps between messages exceed a threshold.
"""

from datetime import timedelta

from .base import ChunkingStrategy


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

        current_timestamp = self._parse_timestamp(current_line)
        if current_timestamp is None:
            return False

        last_timestamp = chunk_context.get("last_timestamp")
        if last_timestamp is None:
            return False

        time_gap = current_timestamp - last_timestamp
        return time_gap > self.gap_threshold
