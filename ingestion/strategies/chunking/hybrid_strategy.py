"""
Hybrid chunking strategy.

Combines multiple chunking strategies and starts new chunks when any strategy indicates to do so.
"""

from typing import List

from .base import ChunkingStrategy


class HybridChunkingStrategy(ChunkingStrategy):
    """Hybrid strategy that combines multiple chunking strategies."""

    def __init__(self, strategies: List[ChunkingStrategy]):
        self.strategies = strategies

    def should_start_new_chunk(
        self, current_line: str, current_index: int, chunk_context: dict
    ) -> bool:
        """Start new chunk if any of the strategies says to do so."""
        return any(
            strategy.should_start_new_chunk(current_line, current_index, chunk_context)
            for strategy in self.strategies
        )
