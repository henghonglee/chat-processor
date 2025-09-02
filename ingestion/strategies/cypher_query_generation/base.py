"""
Base Cypher Query AI strategy interface.

Defines the abstract base class for all Cypher Query AI strategies that generate
Cypher queries from processed chat data using different LLM providers.
"""

import re
from abc import ABC, abstractmethod


class BaseCypherQueryAIStrategy(ABC):
    """Abstract base class for all Cypher Query AI strategies."""

    def __init__(self, model: str, **kwargs):
        """
        Initialize the Cypher Query AI strategy.

        Args:
            model: The model identifier to use for this strategy
            **kwargs: Additional strategy-specific configuration
        """
        self.model = model
        self.config = kwargs

    @abstractmethod
    def generate_cypher(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate Cypher queries using the specific LLM provider.

        Args:
            system_prompt: The system prompt providing context and instructions
            user_prompt: The user prompt containing the data to process

        Returns:
            Generated Cypher query string

        Raises:
            RuntimeError: If the API call fails or returns invalid data
        """
        pass

    def clean_cypher_response(self, text: str) -> str:
        """
        Clean and normalize the LLM response to extract pure Cypher.

        This method removes code blocks, thinking sections, and other
        non-Cypher content that LLMs might include in their responses.

        Args:
            text: Raw text response from the LLM

        Returns:
            Cleaned Cypher query string
        """
        text = text.strip()

        # Remove code blocks if present
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)

        # Remove thinking sections (used by some models)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        return text.strip()

    def get_config_value(self, key: str, default=None):
        """Get a configuration value with optional default."""
        return self.config.get(key, default)

    def validate_response(self, response: str) -> str:
        """
        Validate and clean the LLM response.

        Args:
            response: Raw response from the LLM

        Returns:
            Validated and cleaned response

        Raises:
            RuntimeError: If the response is empty or invalid
        """
        if not response or not response.strip():
            raise RuntimeError("Empty response from LLM")

        cleaned = self.clean_cypher_response(response)

        if not cleaned:
            raise RuntimeError("No valid Cypher content found in response")

        return cleaned

    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"{self.__class__.__name__}(model={self.model})"

    def __repr__(self) -> str:
        """Developer representation of the strategy."""
        return f"{self.__class__.__name__}(model='{self.model}', config={self.config})"
