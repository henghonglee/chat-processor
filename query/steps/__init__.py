"""
Query processing pipeline for chat text processor.

This package provides a multi-step pipeline for processing user queries:
1. Hybrid search - find relevant entities and claims using semantic similarity and full-text search
2. Cypher generation - generate Neo4j Cypher queries for specific entity/claim IDs
3. Query expansion - expand queries to get all neighbors
4. Context coalescence - combine all information into system prompt
5. Result output - format and print final results

The hybrid search step performs both semantic search against ChromaDB and
full-text search to find relevant entity and claim IDs that match the user query.
"""

# Dynamic imports for numbered modules
import importlib.util
import sys
from pathlib import Path

# Import preprocessor components for advanced usage
from ..preprocessors import (
    BasePreprocessor,
    EntityExtractor,
    IntentClassifier,
    KeywordExtractor,
    PreprocessorPipeline,
    PreprocessorResult,
    TextCleaner,
)


def _import_query_step(step_number: int, class_name: str):
    """Dynamically import a query step module and return the specified class."""
    step_files = {
        1: "a_hybrid_search.py",
        2: "b_cypher_generation.py",
        3: "c_query_expansion.py",
        4: "d_context_coalescence.py",
        5: "e_result_output.py",
    }

    step_file = step_files[step_number]

    # Get the current file's directory and build the path to the step file
    current_dir = Path(__file__).parent
    step_path = current_dir / step_file

    # Add project root to sys.path temporarily
    project_root = str(current_dir.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        # Import using importlib
        module_name = f"query_step_{step_number}"
        spec = importlib.util.spec_from_file_location(module_name, step_path)
        module = importlib.util.module_from_spec(spec)

        # Set the module's package context for relative imports
        module.__package__ = "query.steps"

        spec.loader.exec_module(module)

        return getattr(module, class_name)
    finally:
        # Clean up sys.path
        if project_root in sys.path:
            sys.path.remove(project_root)


# Import query step classes
HybridSearcher = _import_query_step(1, "HybridSearcher")
CypherGenerator = _import_query_step(2, "CypherGenerator")
QueryExpander = _import_query_step(3, "QueryExpander")
ContextCoalescer = _import_query_step(4, "ContextCoalescer")
ResultOutputter = _import_query_step(5, "ResultOutputter")

__all__ = [
    # Main pipeline components
    "HybridSearcher",
    "CypherGenerator",
    "QueryExpander",
    "ContextCoalescer",
    "ResultOutputter",
    # Preprocessor components
    "BasePreprocessor",
    "PreprocessorResult",
    "TextCleaner",
    "EntityExtractor",
    "IntentClassifier",
    "KeywordExtractor",
    "PreprocessorPipeline",
]
