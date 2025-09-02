"""
OpenAI Cypher Query AI strategy.

Uses the OpenAI Chat Completions API to generate Cypher queries from processed chat data.
"""

import os
from typing import Optional

from .base import BaseCypherQueryAIStrategy


class OpenAICypherQueryAIStrategy(BaseCypherQueryAIStrategy):
    """Cypher Query AI strategy using OpenAI API for Cypher generation."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_completion_tokens: int = 6000,
        temperature: float = 0.0,
        **kwargs,
    ):
        """
        Initialize the OpenAI Cypher Query AI strategy.

        Args:
            model: The OpenAI model to use (e.g., 'gpt-4', 'gpt-3.5-turbo')
            api_key: The OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            max_completion_tokens: Maximum tokens in the response
            temperature: Sampling temperature (0.0 to 2.0)
            **kwargs: Additional configuration options
        """
        super().__init__(model, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment")

    def generate_cypher(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate Cypher queries using the OpenAI Chat Completions API.

        Args:
            system_prompt: The system prompt providing context and instructions
            user_prompt: The user prompt containing the data to process

        Returns:
            Generated Cypher query string

        Raises:
            RuntimeError: If the OpenAI API call fails
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "OpenAI library not installed. Install with: pip install openai"
            )

        client = OpenAI(api_key=self.api_key)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=self.max_completion_tokens,
                temperature=self.temperature,
            )

            text = response.choices[0].message.content or ""

            if not text:
                raise RuntimeError("Empty response from OpenAI API")

            return self.validate_response(text)

        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}") from e

    def get_endpoint_info(self) -> dict:
        """Get information about the OpenAI configuration."""
        return {
            "provider": "openai",
            "model": self.model,
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
            "api_key_set": bool(self.api_key),
        }

    def test_connection(self) -> bool:
        """
        Test if the OpenAI API is accessible.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)

            # Make a minimal test call
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_completion_tokens=1,
            )
            return bool(response.choices[0].message.content)
        except Exception:
            return False
