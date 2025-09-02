"""
Base processor class for Chain of Responsibility pattern.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseProcessor(ABC):
    """Abstract base class for processors in the chain of responsibility."""

    def __init__(self):
        self._next_processor: Optional["BaseProcessor"] = None

    def set_next(self, processor: "BaseProcessor") -> "BaseProcessor":
        """Set the next processor in the chain."""
        self._next_processor = processor
        return processor

    def process(self, content: str, context: dict = None) -> str:
        """
        Process the content and pass to next processor if needed.

        Args:
            content: Content to process
            context: Optional context dictionary with metadata

        Returns:
            Processed content
        """
        if context is None:
            context = {}

        # Process current content
        processed_content = self._process_content(content, context)

        # Pass to next processor if exists
        if self._next_processor:
            return self._next_processor.process(processed_content, context)

        return processed_content

    @abstractmethod
    def _process_content(self, content: str, context: dict) -> str:
        """
        Implement the specific processing logic.

        Args:
            content: Content to process
            context: Context dictionary

        Returns:
            Processed content
        """
        pass
