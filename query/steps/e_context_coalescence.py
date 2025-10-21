"""
Step 4: Context Coalescence

This module takes the expanded graph context and coalesces all information into
a structured system prompt that will be sent to chat completions along with the user query.
"""

import importlib.util
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .base import BaseQueryStep


def _import_cypher_generation_module():
    """Import the cypher generation module dynamically."""
    current_dir = Path(__file__).parent
    step_path = current_dir / "c_cypher_generation.py"

    project_root = str(current_dir.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        spec = importlib.util.spec_from_file_location("cypher_generation", step_path)
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "query.steps"
        spec.loader.exec_module(module)
        return module
    finally:
        if project_root in sys.path:
            sys.path.remove(project_root)


def _import_expansion_module():
    """Import the expansion module to get GraphContext."""
    current_dir = Path(__file__).parent
    step_path = current_dir / "d_query_expansion.py"

    project_root = str(current_dir.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        spec = importlib.util.spec_from_file_location("query_expansion", step_path)
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "query.steps"
        spec.loader.exec_module(module)
        return module
    finally:
        if project_root in sys.path:
            sys.path.remove(project_root)


# Import the CypherQuerySet class
_cypher_module = _import_cypher_generation_module()
CypherQuerySet = _cypher_module.CypherQuerySet

# Import GraphContext from expansion module
_expansion_module = _import_expansion_module()
GraphContext = _expansion_module.GraphContext

logger = logging.getLogger(__name__)


@dataclass
class SystemPrompt:
    """Container for the final system prompt sent to LLM."""

    prompt: str
    context_summary: Dict[str, Any]
    total_tokens_estimate: int
    original_query_set: CypherQuerySet = None


class ContextCoalescer(BaseQueryStep):
    """Coalesces graph context into structured system prompts for LLM consumption."""

    def setup(self):
        """Setup configuration for context coalescence."""
        self.include_timestamps = self.config.get("include_timestamps", True)
        self.include_full_text = self.config.get("include_full_text", True)
        self.include_query_metadata = self.config.get("include_query_metadata", True)
        self.max_entities = self.config.get("max_entities", 20)
        self.max_claims = self.config.get("max_claims", 15)
        self.max_relationships = self.config.get("max_relationships", 20)

    def format_timestamp(self, timestamp: Any) -> str:
        """
        Format a timestamp for display in the system prompt.

        Args:
            timestamp: The timestamp to format (could be int, float, or string)

        Returns:
            Formatted timestamp string
        """
        try:
            if timestamp is None:
                return "unknown time"

            # Handle string timestamps
            if isinstance(timestamp, str):
                # Try to parse as timestamp
                try:
                    ts = float(timestamp)
                    dt = datetime.fromtimestamp(ts)
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    return timestamp
            else:
                return str(timestamp)
        except (ValueError, OverflowError, OSError):
            return str(timestamp)

    def format_graph_context(self, graph_context: GraphContext) -> str:
        """
        Format GraphContext for the system prompt.

        Args:
            graph_context: The GraphContext to format

        Returns:
            Formatted context string
        """
        # print(f"DEBUG: graph_context = {graph_context}")

        sections = []
        sections.append(f"### {graph_context}")
        return "\n".join(sections)

    def format_search_results(self, search_results) -> str:
        """
        Format search results for the system prompt when graph context is not available.

        Args:
            search_results: The search results object (VectorSearchResults)

        Returns:
            Formatted context string
        """
        sections = []
        sections.append("## Search Results")
        sections.append(f"Found {len(search_results.results)} relevant documents:")

        for i, result in enumerate(search_results.results[:20], 1):  # Limit to top 20
            sections.append(f"\n### Result {i}")
            sections.append(f"**Source:** {result.chat_name}")
            sections.append(f"**Type:** {result.node_type}")
            sections.append(f"**Relevance Score:** {result.similarity_score:.4f}")

            # Truncate text if too long
            text = result.document_text
            if len(text) > 1000:
                text = text[:1000] + "..."
            sections.append(f"**Content:** {text}")

        if len(search_results.results) > 20:
            sections.append(f"\n... and {len(search_results.results) - 20} more results")

        return "\n".join(sections)

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the given text.
        Very rough approximation: ~4 characters per token for English.

        Args:
            text: The text to estimate tokens for

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def load_prompt_template(self) -> str:
        """
        Load the system prompt template from the markdown file.

        Returns:
            The prompt template string
        """
        template_path = Path(__file__).parent / "prompts" / "system_prompt.md"
        return template_path.read_text(encoding="utf-8")

    def build_system_prompt(
        self, graph_context: GraphContext, user_query: str, search_results: any = None
    ) -> SystemPrompt:
        """
        Build the complete system prompt from the GraphContext or search results.

        Args:
            graph_context: The GraphContext with query results
            user_query: The original user query
            search_results: Search results to use when graph context is empty

        Returns:
            SystemPrompt object with the complete prompt
        """
        logger.info("Building system prompt from query results")

        # Load prompt template
        template = self.load_prompt_template()

        # Check if we have meaningful graph context or need to use search results
        has_graph_data = (
            len(graph_context.entities) > 0 or
            len(graph_context.people) > 0 or
            len(graph_context.claims) > 0 or
            len(graph_context.said_relationships) > 0 or
            len(graph_context.mention_relationships) > 0 or
            len(graph_context.reaction_relationships) > 0
        )

        if has_graph_data:
            # Use graph context (normal case when graph search is enabled)
            query_results = getattr(graph_context, "query_results", [])
            num_results = len(query_results)
            total_records = sum(len(result.get("records", [])) for result in query_results)
            formatted_results = self.format_graph_context(graph_context)
        elif search_results and hasattr(search_results, 'results') and len(search_results.results) > 0:
            # Use search results when graph context is empty (graph search disabled)
            num_results = len(search_results.results)
            total_records = len(search_results.results)
            formatted_results = self.format_search_results(search_results)
        else:
            # No context available
            num_results = 0
            total_records = 0
            formatted_results = "No relevant information found in the knowledge base."

        # Fill in template variables
        full_prompt = template.format(
            user_query=user_query,
            graph_context=formatted_results
        )

        # Estimate tokens
        token_estimate = self.estimate_tokens(full_prompt)

        # Build context summary
        context_summary = {
            "total_results": num_results,
            "total_records": total_records,
        }

        logger.info(f"System prompt built: {token_estimate} estimated tokens")
        # logger.info(f"Full system prompt:\n{full_prompt}")

        return SystemPrompt(
            prompt=full_prompt,
            context_summary=context_summary,
            total_tokens_estimate=token_estimate,
        )

    def process(self, context_data: tuple) -> SystemPrompt:
        """
        Main processing method to coalesce GraphContext and search results into a system prompt.

        Args:
            context_data: Tuple of (graph_context, user_query, search_results)

        Returns:
            SystemPrompt ready for LLM consumption
        """
        if len(context_data) == 3:
            graph_context, user_query, search_results = context_data
        else:
            # Backward compatibility
            graph_context, user_query = context_data
            search_results = None

        self.logger.info("Starting context coalescence")

        system_prompt = self.build_system_prompt(graph_context, user_query, search_results)

        self.logger.info(
            f"Context coalescence complete. "
            f"Prompt length: {len(system_prompt.prompt)} characters, "
            f"Estimated tokens: {system_prompt.total_tokens_estimate}"
        )

        return system_prompt

    def _log_step_result(self, result: SystemPrompt):
        """Log the context coalescence result details."""
        self.logger.debug(f"System prompt length: {len(result.prompt)} characters")
        self.logger.debug(f"Estimated tokens: {result.total_tokens_estimate}")
        self.logger.debug(f"Context summary: {result.context_summary}")
