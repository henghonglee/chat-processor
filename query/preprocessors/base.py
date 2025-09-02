"""
Base Preprocessor Classes

Defines the abstract base class and common data structures for query preprocessors.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class PreprocessorResult:
    """Container for preprocessor results with metadata."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the result after initialization."""
        if self.success and self.data is None:
            raise ValueError("Successful result must have data")
        if not self.success and self.error is None:
            raise ValueError("Failed result must have error message")


class BasePreprocessor(ABC):
    """Abstract base class for all query preprocessors."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the preprocessor.

        Args:
            config: Configuration dictionary for the preprocessor
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def process(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> PreprocessorResult:
        """
        Process the input text.

        Args:
            text: Input text to process
            context: Optional context information from previous processors

        Returns:
            PreprocessorResult containing the processed data
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this preprocessor.

        Returns:
            String name of the preprocessor
        """
        pass

    def validate_input(self, text: str) -> bool:
        """
        Validate input text.

        Args:
            text: Input text to validate

        Returns:
            True if input is valid, False otherwise
        """
        if not isinstance(text, str):
            self.logger.error(f"Input must be string, got {type(text)}")
            return False

        if not text.strip():
            self.logger.warning("Input text is empty or only whitespace")
            return False

        return True

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with default fallback.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)

    def log_processing_start(self, text: str) -> None:
        """Log the start of processing."""
        self.logger.debug(f"{self.get_name()} processing text: {text[:100]}...")

    def log_processing_end(self, result: PreprocessorResult) -> None:
        """Log the end of processing."""
        if result.success:
            self.logger.debug(
                f"{self.get_name()} completed successfully in {result.processing_time:.3f}s"
            )
        else:
            self.logger.warning(
                f"{self.get_name()} failed after {result.processing_time:.3f}s: {result.error}"
            )
