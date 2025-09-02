"""
Groq Cypher Query AI strategy.

Uses the Groq API to generate Cypher queries from processed chat data.
"""

import logging
import os
from typing import Optional

from .base import BaseCypherQueryAIStrategy

logger = logging.getLogger(__name__)


class GroqCypherQueryAIStrategy(BaseCypherQueryAIStrategy):
    """Cypher Query AI strategy using Groq API for Cypher generation."""

    def __init__(
        self,
        model: str = "qwen/qwen3-32b",
        api_key: Optional[str] = None,
        max_completion_tokens: int = 8000,
        temperature: float = 0.0,
        top_p: float = 0.0,
        **kwargs,
    ):
        """
        Initialize the Groq Cypher Query AI strategy.

        Args:
            model: The Groq model to use (e.g., 'llama-3.1-70b-versatile')
            api_key: The Groq API key (uses GROQ_API_KEY env var if not provided)
            max_completion_tokens: Maximum tokens in the response
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Top-p sampling parameter
            **kwargs: Additional configuration options
        """
        super().__init__(model, **kwargs)
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.top_p = top_p

        if not self.api_key:
            logger.error("GROQ_API_KEY is not set in environment")
            raise RuntimeError("GROQ_API_KEY is not set in environment")

        logger.info(f"Initialized Groq strategy with model: {self.model}")
        logger.debug(
            f"Configuration: max_tokens={self.max_completion_tokens}, temperature={self.temperature}, top_p={self.top_p}"
        )

    def generate_cypher(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate Cypher queries using the Groq API.

        Args:
            system_prompt: The system prompt providing context and instructions
            user_prompt: The user prompt containing the data to process

        Returns:
            Generated Cypher query string

        Raises:
            RuntimeError: If the Groq API call fails
        """
        logger.info(f"Generating Cypher query using Groq model: {self.model}")
        logger.debug(f"System prompt length: {len(system_prompt)} chars")
        logger.debug(f"User prompt length: {len(user_prompt)} chars")

        try:
            from groq import Groq
        except ImportError:
            logger.error("Groq library not installed")
            raise RuntimeError(
                "Groq library not installed. Install with: pip install groq"
            ) from None

        client = Groq(api_key=self.api_key)

        try:
            logger.debug("Making API call to Groq")
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
                top_p=self.top_p,
                stream=False,
                stop=None,
            )

            text = completion.choices[0].message.content or ""
            logger.debug(f"Received response from Groq API: {len(text)} chars")

            if not text:
                logger.error("Empty response from Groq API")
                raise RuntimeError("Empty response from Groq API")

            validated_response = self.validate_response(text)
            logger.info("Successfully generated and validated Cypher query")
            logger.debug(f"Final query length: {len(validated_response)} chars")
            return validated_response

        except Exception as e:
            logger.error(f"Groq API call failed: {str(e)}")
            raise RuntimeError(f"Groq API call failed: {str(e)}") from e

    def clean_cypher_response(self, text: str) -> str:
        """
        Clean Groq response with additional thinking tag removal.

        Groq models sometimes use <think></think> tags, so we ensure
        those are properly removed in addition to the base cleaning.
        """
        logger.debug("Cleaning Groq response")
        original_length = len(text)

        # First apply base cleaning
        text = super().clean_cypher_response(text)

        # Additional Groq-specific cleaning could go here
        # (already handled in base class, but can be extended)

        logger.debug(f"Cleaned response: {original_length} -> {len(text)} chars")
        return text

    def get_endpoint_info(self) -> dict:
        """Get information about the Groq configuration."""
        info = {
            "provider": "groq",
            "model": self.model,
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "api_key_set": bool(self.api_key),
        }
        logger.debug(f"Endpoint info: {info}")
        return info

    def test_connection(self) -> bool:
        """
        Test if the Groq API is accessible.

        Returns:
            True if connection successful, False otherwise
        """
        logger.info("Testing Groq API connection")
        try:
            from groq import Groq

            client = Groq(api_key=self.api_key)

            # Make a minimal test call
            completion = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_completion_tokens=1,
                temperature=0.0,
            )
            result = bool(completion.choices[0].message.content)
            logger.info(
                f"Groq API connection test {'successful' if result else 'failed'}"
            )
            return result
        except Exception as e:
            logger.error(f"Groq API connection test failed: {str(e)}")
            return False
