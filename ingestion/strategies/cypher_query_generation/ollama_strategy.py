"""
Ollama Cypher Query AI strategy.

Uses the Ollama API to generate Cypher queries from processed chat data.
"""

import os
from typing import Optional

import requests

from .base import BaseCypherQueryAIStrategy


class OllamaCypherQueryAIStrategy(BaseCypherQueryAIStrategy):
    """Cypher Query AI strategy using Ollama API for Cypher generation."""

    def __init__(
        self,
        model: str,
        ollama_url: Optional[str] = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        num_predict: int = 6000,
        timeout: int = 300,
        **kwargs,
    ):
        """
        Initialize the Ollama Cypher Query AI strategy.

        Args:
            model: The Ollama model to use (e.g., 'llama2', 'codellama')
            ollama_url: The Ollama server URL (defaults to localhost:11434)
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Top-p sampling parameter
            num_predict: Maximum number of tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        super().__init__(model, **kwargs)
        self.ollama_url = ollama_url or os.getenv(
            "OLLAMA_URL", "http://localhost:11434"
        )
        self.temperature = temperature
        self.top_p = top_p
        self.num_predict = num_predict
        self.timeout = timeout

    def generate_cypher(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate Cypher queries using the Ollama API.

        Args:
            system_prompt: The system prompt providing context and instructions
            user_prompt: The user prompt containing the data to process

        Returns:
            Generated Cypher query string

        Raises:
            RuntimeError: If the Ollama API call fails
        """
        # Combine system and user prompts for Ollama
        combined_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"

        payload = {
            "model": self.model,
            "prompt": combined_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.num_predict,
            },
        }

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate", json=payload, timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            text = result.get("response", "")

            if not text:
                raise RuntimeError("Empty response from Ollama API")

            return self.validate_response(text)

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API call failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Error processing Ollama response: {e}") from e

    def get_endpoint_info(self) -> dict:
        """Get information about the Ollama endpoint."""
        return {
            "provider": "ollama",
            "url": self.ollama_url,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.num_predict,
            "timeout": self.timeout,
        }

    def test_connection(self) -> bool:
        """
        Test if the Ollama server is accessible.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException:
            return False
