"""
Message count-based chunking strategy.
"""

from .base import ChunkingStrategy


class MessageCountChunkingStrategy(ChunkingStrategy):
    """Chunking strategy based on number of messages."""

    def __init__(self, max_messages: int = 100):
        self.max_messages = max_messages

    def should_start_new_chunk(
        self, current_line: str, current_index: int, chunk_context: dict
    ) -> bool:
        """Start new chunk if message count exceeds threshold."""
        if not current_line.strip() or not current_line.startswith("["):
            return False

        return chunk_context.get("message_count", 0) >= self.max_messages
