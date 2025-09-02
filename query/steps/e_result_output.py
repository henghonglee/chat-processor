"""
Step 5: Result Output

This module handles the final step of sending the system prompt and user query
to chat completions, and formatting/printing the final results.
"""

import importlib.util
import json
import os
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import openai

from .base import BaseQueryStep


def _import_context_coalescence_module():
    """Import the context coalescence module dynamically."""
    current_dir = Path(__file__).parent
    step_path = current_dir / "d_context_coalescence.py"

    project_root = str(current_dir.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        spec = importlib.util.spec_from_file_location("context_coalescence", step_path)
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "query.steps"
        spec.loader.exec_module(module)
        return module
    finally:
        if project_root in sys.path:
            sys.path.remove(project_root)


# Import the required classes
_coalescence_module = _import_context_coalescence_module()
SystemPrompt = _coalescence_module.SystemPrompt

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Container for LLM response data."""

    response_text: str
    model_used: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    response_time: float
    success: bool
    error: Optional[str] = None


@dataclass
class QueryResult:
    """Container for the complete query result."""

    user_query: str
    system_prompt: SystemPrompt
    llm_response: LLMResponse
    processing_time: float
    success: bool
    graph_context: Optional[Any] = None  # GraphContext from query expansion


class ResultOutputter(BaseQueryStep):
    """Handles sending queries to LLM and formatting output."""

    def __init__(self, openai_api_key: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the result outputter.

        Args:
            openai_api_key: OpenAI API key
            config: Configuration dictionary
        """
        self.openai_api_key = openai_api_key
        super().__init__(config)

    def setup(self):
        """Setup OpenAI client and configuration."""
        self.openai_client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

        # Configuration
        self.model = self.config.get("model", "gpt-4o")
        self.max_tokens = self.config.get("max_tokens", 1000)
        self.temperature = self.config.get("temperature", 0.7)
        self.timeout = self.config.get("timeout", 30)

        # Output formatting
        self.include_context_summary = self.config.get("include_context_summary", True)
        self.include_token_usage = self.config.get("include_token_usage", True)
        self.include_timing = self.config.get("include_timing", True)
        self.verbose = self.config.get("verbose", False)

    def send_to_llm(self, system_prompt: SystemPrompt, user_query: str) -> LLMResponse:
        """
        Send the system prompt and user query to the LLM.

        Args:
            system_prompt: The SystemPrompt object with context
            user_query: The original user query

        Returns:
            LLMResponse with the result
        """
        self.logger.info(f"Sending query to LLM: {self.model}")

        # Log the complete input being sent to LLM
        self._log_llm_input(system_prompt, user_query)

        start_time = time.time()

        try:
            # Check token limits
            if system_prompt.total_tokens_estimate > 100000:  # Rough safety limit
                self.logger.warning(
                    f"System prompt very large: {system_prompt.total_tokens_estimate} tokens"
                )

            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt.prompt},
                {"role": "user", "content": user_query},
            ]

            # Make API call
            response = self.openai_client.chat.completions.create(
                model="openai/gpt-4o",
                messages=messages,
                # max_tokens=self.max_tokens,
                # temperature=self.temperature,
                # timeout=self.timeout,
            )

            response_time = time.time() - start_time

            # Extract response data
            response_text = response.choices[0].message.content
            usage = response.usage

            llm_response = LLMResponse(
                response_text=response_text,
                model_used=self.model,
                total_tokens=usage.total_tokens,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                response_time=response_time,
                success=True,
            )

            # Log the complete LLM response
            self._log_llm_output(llm_response)

            return llm_response

        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(f"LLM request failed: {str(e)}")

            return LLMResponse(
                response_text="",
                model_used=self.model,
                total_tokens=0,
                prompt_tokens=0,
                completion_tokens=0,
                response_time=response_time,
                success=False,
                error=str(e),
            )

    def format_response(self, query_result: QueryResult) -> str:
        """
        Format the complete query result for display.

        Args:
            query_result: The QueryResult to format

        Returns:
            Formatted response string
        """
        output_lines = []

        # Header
        output_lines.append("=" * 80)
        output_lines.append("CHAT KNOWLEDGE BASE QUERY RESULT")
        output_lines.append("=" * 80)
        output_lines.append("")

        # Query
        output_lines.append(f"Query: {query_result.user_query}")
        output_lines.append("")

        # Context Summary (if enabled)
        if self.include_context_summary:
            summary = query_result.system_prompt.context_summary
            output_lines.append("Context Retrieved:")
            output_lines.append(f"  - {summary.get('entities_count', 0)} entities")
            output_lines.append(f"  - {summary.get('people_count', 0)} people")
            output_lines.append(f"  - {summary.get('claims_count', 0)} claims")
            output_lines.append(
                f"  - {summary.get('said_relationships_count', 0)} statements"
            )
            output_lines.append(
                f"  - {summary.get('mention_relationships_count', 0)} mentions"
            )
            output_lines.append(
                f"  - {summary.get('reaction_relationships_count', 0)} reactions"
            )
            output_lines.append("")

            # Add detailed context properties if available
            self._add_detailed_context_summary(output_lines, query_result)
            output_lines.append("")

        # Main Response
        if query_result.llm_response.success:
            output_lines.append("Answer:")
            output_lines.append("-" * 40)
            output_lines.append(query_result.llm_response.response_text)
            output_lines.append("-" * 40)
        else:
            output_lines.append("❌ Error generating response:")
            output_lines.append(f"   {query_result.llm_response.error}")

        output_lines.append("")

        # Token Usage (if enabled)
        if self.include_token_usage and query_result.llm_response.success:
            output_lines.append("Token Usage:")
            output_lines.append(
                f"  - Prompt tokens: {query_result.llm_response.prompt_tokens:,}"
            )
            output_lines.append(
                f"  - Completion tokens: {query_result.llm_response.completion_tokens:,}"
            )
            output_lines.append(
                f"  - Total tokens: {query_result.llm_response.total_tokens:,}"
            )
            output_lines.append(f"  - Model: {query_result.llm_response.model_used}")
            output_lines.append("")

        # Timing (if enabled)
        if self.include_timing:
            output_lines.append("Performance:")
            output_lines.append(
                f"  - Total processing time: {query_result.processing_time:.2f}s"
            )
            output_lines.append(
                f"  - LLM response time: {query_result.llm_response.response_time:.2f}s"
            )
            summary = query_result.system_prompt.context_summary
            output_lines.append(
                f"  - Database queries executed: {summary.get('total_queries_executed', 0)}"
            )
            output_lines.append(
                f"  - Successful queries: {summary.get('successful_queries', 0)}"
            )
            output_lines.append("")

        return "\n".join(output_lines)

    def _add_detailed_context_summary(
        self, output_lines: list, query_result: QueryResult
    ):
        """Add detailed properties of each item in the context summary."""
        from datetime import datetime

        if not query_result.graph_context:
            return

        graph_context = query_result.graph_context

        # Detailed entities
        if graph_context.entities:
            output_lines.append("🏷️ DETAILED ENTITIES:")
            entities_items = list(graph_context.entities.items())[
                :5
            ]  # Limit for readability
            for i, (entity_id, entity_data) in enumerate(entities_items, 1):
                output_lines.append(f"  {i}. Entity: {entity_id}")
                for key, value in entity_data.items():
                    if value is not None and value != "":
                        output_lines.append(f"     {key}: {value}")
                output_lines.append("")

        # Detailed people
        if graph_context.people:
            output_lines.append("👤 DETAILED PEOPLE:")
            people_items = list(graph_context.people.items())[
                :5
            ]  # Limit for readability
            for i, (person_id, person_data) in enumerate(people_items, 1):
                output_lines.append(f"  {i}. Person: {person_id}")
                for key, value in person_data.items():
                    if value is not None and value != "":
                        output_lines.append(f"     {key}: {value}")
                output_lines.append("")

        # Detailed claims
        if graph_context.claims:
            output_lines.append("💬 DETAILED CLAIMS:")
            claims_items = list(graph_context.claims.items())[
                :3
            ]  # Limit for readability
            for i, (claim_id, claim_data) in enumerate(claims_items, 1):
                output_lines.append(f"  {i}. Claim: {claim_id}")
                for key, value in claim_data.items():
                    if value is not None and value != "":
                        # Format timestamp if it's valid_at
                        if key == "valid_at" and isinstance(value, (int, float)):
                            try:
                                formatted_time = datetime.fromtimestamp(value).strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                )
                                output_lines.append(
                                    f"     {key}: {formatted_time} ({value})"
                                )
                            except:
                                output_lines.append(f"     {key}: {value}")
                        else:
                            # Truncate long text fields
                            if (
                                key in ["text", "summary_text"]
                                and len(str(value)) > 200
                            ):
                                truncated_value = str(value)[:197] + "..."
                                output_lines.append(f"     {key}: {truncated_value}")
                            else:
                                output_lines.append(f"     {key}: {value}")
                output_lines.append("")

        # Detailed relationships
        if graph_context.said_relationships:
            output_lines.append("💬 DETAILED SAID RELATIONSHIPS:")
            for i, relationship in enumerate(graph_context.said_relationships[:3], 1):
                output_lines.append(f"  {i}. Said Relationship:")
                for key, value in relationship.items():
                    if value is not None and value != "":
                        output_lines.append(f"     {key}: {value}")
                output_lines.append("")

        if graph_context.mention_relationships:
            output_lines.append("🏷️ DETAILED MENTION RELATIONSHIPS:")
            for i, relationship in enumerate(
                graph_context.mention_relationships[:3], 1
            ):
                output_lines.append(f"  {i}. Mention Relationship:")
                for key, value in relationship.items():
                    if value is not None and value != "":
                        output_lines.append(f"     {key}: {value}")
                output_lines.append("")

        if graph_context.reaction_relationships:
            output_lines.append("⚡ DETAILED REACTION RELATIONSHIPS:")
            for i, relationship in enumerate(
                graph_context.reaction_relationships[:3], 1
            ):
                output_lines.append(f"  {i}. Reaction Relationship:")
                for key, value in relationship.items():
                    if value is not None and value != "":
                        output_lines.append(f"     {key}: {value}")
                output_lines.append("")

    def format_json_response(self, query_result: QueryResult) -> str:
        """
        Format the query result as JSON.

        Args:
            query_result: The QueryResult to format

        Returns:
            JSON formatted string
        """
        result_dict = {
            "query": query_result.user_query,
            "success": query_result.success,
            "processing_time": query_result.processing_time,
            "response": {
                "text": (
                    query_result.llm_response.response_text
                    if query_result.llm_response.success
                    else None
                ),
                "success": query_result.llm_response.success,
                "error": query_result.llm_response.error,
                "model": query_result.llm_response.model_used,
                "response_time": query_result.llm_response.response_time,
            },
            "context": query_result.system_prompt.context_summary,
            "tokens": (
                {
                    "prompt_tokens": query_result.llm_response.prompt_tokens,
                    "completion_tokens": query_result.llm_response.completion_tokens,
                    "total_tokens": query_result.llm_response.total_tokens,
                    "estimated_prompt_tokens": query_result.system_prompt.total_tokens_estimate,
                }
                if query_result.llm_response.success
                else None
            ),
        }

        return json.dumps(result_dict, indent=2, ensure_ascii=False)

    def print_result(
        self, query_result: QueryResult, format_type: str = "text"
    ) -> None:
        """
        Print the query result in the specified format.

        Args:
            query_result: The QueryResult to print
            format_type: Output format ("text" or "json")
        """
        if format_type.lower() == "json":
            print(self.format_json_response(query_result))
        else:
            print(self.format_response(query_result))

    def process(self, input_data: tuple) -> QueryResult:
        """
        Main processing method to send query to LLM and output results.

        Args:
            input_data: Tuple of (system_prompt, user_query, processing_time, format_type, graph_context)

        Returns:
            QueryResult object
        """
        if len(input_data) == 5:
            system_prompt, user_query, processing_time, format_type, graph_context = (
                input_data
            )
        else:
            # Backward compatibility
            system_prompt, user_query, processing_time, format_type = input_data
            graph_context = None
        self.logger.info("Processing final output")

        # Send to LLM
        llm_response = self.send_to_llm(system_prompt, user_query)

        # Create result object
        query_result = QueryResult(
            user_query=user_query,
            system_prompt=system_prompt,
            llm_response=llm_response,
            processing_time=processing_time,
            success=llm_response.success,
            graph_context=graph_context,
        )

        # Print results
        self.print_result(query_result, format_type)

        self.logger.info(f"Output complete. Success: {query_result.success}")

        return query_result

    def process_and_output(
        self,
        system_prompt: SystemPrompt,
        user_query: str,
        processing_time: float,
        format_type: str = "text",
        graph_context: Optional[Any] = None,
    ) -> QueryResult:
        """
        Legacy method for backward compatibility.
        Delegates to the process method.
        """
        if graph_context is not None:
            return self.process(
                (system_prompt, user_query, processing_time, format_type, graph_context)
            )
        else:
            return self.process(
                (system_prompt, user_query, processing_time, format_type)
            )

    def _log_llm_input(self, system_prompt: SystemPrompt, user_query: str):
        """Log the complete input being sent to the LLM."""
        self.logger.info("=== LLM INPUT ===")
        self.logger.info(f"Model: {self.model}")
        self.logger.info(f"User Query: {user_query}")
        self.logger.info(
            f"System Prompt Token Estimate: {system_prompt.total_tokens_estimate}"
        )
        self.logger.info("System Prompt Content:")
        self.logger.info("-" * 60)
        self.logger.info(system_prompt.prompt)
        self.logger.info("-" * 60)

    def _log_llm_output(self, llm_response: LLMResponse):
        """Log the complete response from the LLM."""
        self.logger.info("=== LLM OUTPUT ===")
        self.logger.info(f"Model Used: {llm_response.model_used}")
        self.logger.info(f"Response Time: {llm_response.response_time:.2f}s")
        self.logger.info(f"Success: {llm_response.success}")
        self.logger.info(f"Total Tokens: {llm_response.total_tokens}")
        self.logger.info(f"Prompt Tokens: {llm_response.prompt_tokens}")
        self.logger.info(f"Completion Tokens: {llm_response.completion_tokens}")
        self.logger.info("LLM Response Content:")
        self.logger.info("-" * 60)
        self.logger.info(llm_response.response_text)
        self.logger.info("-" * 60)
        self.logger.info("=== END LLM OUTPUT ===")

    def save_result(
        self, query_result: QueryResult, file_path: str, format_type: str = "json"
    ) -> None:
        """
        Save the query result to a file.

        Args:
            query_result: The QueryResult to save
            file_path: Path to save the result
            format_type: Format to save ("json" or "text")
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                if format_type.lower() == "json":
                    f.write(self.format_json_response(query_result))
                else:
                    f.write(self.format_response(query_result))

            self.logger.info(f"Result saved to: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save result: {str(e)}")

    def _log_step_result(self, result: QueryResult):
        """Log the result output details."""
        self.logger.debug(f"Query processed: {result.user_query}")
        self.logger.debug(f"LLM success: {result.llm_response.success}")
        if result.llm_response.success:
            self.logger.debug(f"Model used: {result.llm_response.model_used}")
            self.logger.debug(f"Total tokens: {result.llm_response.total_tokens}")
            self.logger.debug(
                f"Response time: {result.llm_response.response_time:.2f}s"
            )
        else:
            self.logger.debug(f"LLM error: {result.llm_response.error}")


if __name__ == "__main__":
    # Example usage
    import os

    # SystemPrompt is already imported at module level
    # Mock system prompt for testing
    mock_prompt = SystemPrompt(
        prompt="# Test System Prompt\n\nThis is a test context.",
        context_summary={
            "entities_count": 1,
            "people_count": 1,
            "claims_count": 1,
            "said_relationships_count": 1,
            "mention_relationships_count": 0,
            "reaction_relationships_count": 0,
            "paths_count": 0,
            "total_queries_executed": 5,
            "successful_queries": 5,
        },
        total_tokens_estimate=100,
    )
    # Initialize outputter (you'll need a valid API key)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        outputter = ResultOutputter(api_key)

        # Test query
        user_query = "What did Shaun say about BTC?"
        processing_time = 1.5

        # Process and output
        result = outputter.process_and_output(mock_prompt, user_query, processing_time)

        print("\n" + "=" * 50)
        print("Example completed successfully!")
    else:
        print("OPENAI_API_KEY not found in environment variables")
