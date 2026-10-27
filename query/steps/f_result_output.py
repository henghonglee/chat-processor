"""
Step 6: Result Output

Sends the coalesced system prompt together with the user's query to the
configured LLM, then formats and prints the final answer.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseQueryStep

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Final result of the query pipeline."""

    query: str
    answer: str
    model: str
    context_summary: Dict[str, Any] = field(default_factory=dict)
    token_usage: Dict[str, int] = field(default_factory=dict)
    preprocessing_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ResultOutputter(BaseQueryStep):
    """Generates the final LLM answer from the system prompt and formats/prints it."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        super().__init__(config)

    def setup(self):
        """Setup configuration for result output."""
        self.model = self.config.get("model", "gpt-4o-mini")
        self.max_tokens = self.config.get("max_tokens", 1000)
        self.temperature = self.config.get("temperature", 0.0)
        self.include_token_usage = self.config.get("include_token_usage", True)
        self.include_timing = self.config.get("include_timing", True)
        self.verbose = self.config.get("verbose", False)

        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for ResultOutputter")

    def _call_llm(self, system_prompt: str, user_query: str) -> tuple:
        """Send the system prompt and user query to the configured LLM."""
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "OpenAI library not installed. Install with: pip install openai"
            )

        client = OpenAI(api_key=self.openai_api_key)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                max_completion_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}") from e

        answer = response.choices[0].message.content or ""

        usage = {}
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return answer, usage

    def process(self, input_data: tuple) -> QueryResult:
        """
        Main processing method: generate the final answer and print it.

        Args:
            input_data: Tuple of (system_prompt, user_query, preprocessing_time,
                output_format, graph_context)

        Returns:
            QueryResult with the generated answer and metadata
        """
        system_prompt, user_query, preprocessing_time, output_format, graph_context = (
            input_data
        )

        self.logger.info(f"Sending query to model: {self.model}")
        start_time = time.time()
        answer, usage = self._call_llm(system_prompt.prompt, user_query)
        generation_time = time.time() - start_time

        result = QueryResult(
            query=user_query,
            answer=answer,
            model=self.model,
            context_summary=system_prompt.context_summary,
            token_usage=usage,
            preprocessing_time=preprocessing_time,
            generation_time=generation_time,
            total_time=preprocessing_time + generation_time,
        )

        self._print_result(result, output_format)

        return result

    def _result_to_dict(self, result: QueryResult) -> Dict[str, Any]:
        return {
            "query": result.query,
            "answer": result.answer,
            "model": result.model,
            "context_summary": result.context_summary,
            "token_usage": result.token_usage,
            "preprocessing_time": result.preprocessing_time,
            "generation_time": result.generation_time,
            "total_time": result.total_time,
            "timestamp": result.timestamp,
        }

    def _print_result(self, result: QueryResult, output_format: str):
        """Format and print the final result to stdout."""
        if output_format == "json":
            print(json.dumps(self._result_to_dict(result), indent=2))
            return

        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(result.answer)
        print("=" * 80)

        if self.include_token_usage and result.token_usage:
            print(f"Tokens: {result.token_usage}")

        if self.include_timing:
            print(
                f"Timing: preprocessing={result.preprocessing_time:.2f}s "
                f"generation={result.generation_time:.2f}s "
                f"total={result.total_time:.2f}s"
            )
        print()

    def save_result(
        self, result: QueryResult, path: str, output_format: str = "text"
    ):
        """Save the query result to a file."""
        output_path = Path(path)

        if output_format == "json":
            output_path.write_text(
                json.dumps(self._result_to_dict(result), indent=2), encoding="utf-8"
            )
        else:
            output_path.write_text(result.answer, encoding="utf-8")

        self.logger.info(f"Result saved to {output_path}")

    def _log_step_result(self, result: QueryResult):
        """Log the result output details."""
        self.logger.debug(f"Answer length: {len(result.answer)} characters")
        self.logger.debug(f"Token usage: {result.token_usage}")
