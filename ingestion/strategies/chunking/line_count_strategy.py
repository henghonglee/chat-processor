"""
Line count-based chunking strategy.

Starts new chunks when the number of lines exceeds a threshold.
"""

from .base import ChunkingStrategy


class LineCountChunkingStrategy(ChunkingStrategy):
    """Chunking strategy based on number of lines."""

    def __init__(self, max_lines: int = 200):
        self.max_lines = max_lines

    def should_start_new_chunk(
        self, current_line: str, current_index: int, chunk_context: dict
    ) -> bool:
        """Start new chunk if line count exceeds threshold."""
        start_index = chunk_context.get("start_index", 0)
        current_lines = current_index - start_index + 1
        return current_lines >= self.max_lines
