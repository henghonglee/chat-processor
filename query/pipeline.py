"""
Query Pipeline

Main orchestrator that runs the complete query processing pipeline:
1. Cypher generation
2. Context coalescence
3. Result output
"""

import importlib.util
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()  # Automatically load environment variables from .env file
except ImportError:
    # dotenv is optional, continue without it
    pass

try:
    from neo4j import GraphDatabase

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

# Import preprocessor pipeline
try:
    from .preprocessors import PreprocessorPipeline
    PREPROCESSOR_AVAILABLE = True
except ImportError:
    PREPROCESSOR_AVAILABLE = False


def _import_query_step(step_number, class_name: str):
    """Dynamically import a query step module and return the specified class."""
    if step_number == 1:
        step_file = "a_hybrid_search.py"
    elif step_number == 2:
        step_file = "b_reranker.py"
    elif step_number == 3:
        step_file = "c_cypher_generation.py"
    elif step_number == 4:
        step_file = "d_query_expansion.py"
    elif step_number == 5:
        step_file = "e_context_coalescence.py"
    elif step_number == 6:
        step_file = "f_result_output.py"
    else:
        raise ValueError(f"Unknown step number: {step_number}")

    # Get the current file's directory and build the path to the step file
    current_dir = Path(__file__).parent
    step_path = current_dir / "steps" / step_file

    # Add project root to sys.path temporarily
    project_root = str(current_dir.parent)
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
SearcherClass = _import_query_step(1, "HybridSearcher")
RerankerStep = _import_query_step(2, "RerankerStep")
CypherGenerator = _import_query_step(3, "CypherGenerator")
QueryExpander = _import_query_step(4, "QueryExpander")
ContextCoalescer = _import_query_step(5, "ContextCoalescer")
ResultOutputter = _import_query_step(6, "ResultOutputter")


# Import GraphContext from the query expansion step
def _import_graph_context():
    """Import GraphContext from step 3."""
    current_dir = Path(__file__).parent
    step_path = current_dir / "steps" / "d_query_expansion.py"
    project_root = str(current_dir.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        spec = importlib.util.spec_from_file_location("query_expansion", step_path)
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "query.steps"
        spec.loader.exec_module(module)
        return module.GraphContext
    finally:
        if project_root in sys.path:
            sys.path.remove(project_root)


GraphContext = _import_graph_context()

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the query pipeline."""

    # OpenAI API
    openai_api_key: str

    # Neo4j configuration
    neo4j_uri: str = None
    neo4j_user: str = None
    neo4j_password: str = None

    # Processing options
    max_entities: int = 20
    max_claims: int = 30
    max_relationships: int = 50

    # Search options
    enable_fulltext_search: bool = True
    enable_vector_search: bool = True
    enable_graph_search: bool = True

    # Reranker options
    enable_reranking: bool = True
    rerank_method: str = "cross-encoder"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k_before: int = 50
    rerank_top_k_after: int = 20

    # LLM options
    model: str = "gpt-4o-mini"
    max_tokens: int = 1000
    temperature: float = 0.0

    # Output options
    output_format: str = "text"  # "text" or "json"
    include_token_usage: bool = True
    include_timing: bool = True
    verbose: bool = False

    # Preprocessing options
    enable_query_preprocessing: bool = True
    replace_personal_pronouns: bool = True


class QueryPipeline:
    """Main pipeline orchestrator for query processing."""

    def __init__(self, config: PipelineConfig):
        """
        Initialize the query pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.setup_logging()
        self.initialize_components()

    def setup_logging(self):
        """Setup logging configuration."""
        level = logging.DEBUG if self.config.verbose else logging.INFO
        logging.basicConfig(
            level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def initialize_components(self):
        """Initialize all pipeline components."""
        logger.info("Initializing pipeline components")

        # Step 0: Query Preprocessor (optional)
        self.query_preprocessor = None
        if PREPROCESSOR_AVAILABLE and self.config.enable_query_preprocessing:
            preprocessor_config = {
                "pipeline_stages": ["text_cleaner"],
                "stage_configs": {
                    "text_cleaner": {
                        "replace_personal_pronouns": self.config.replace_personal_pronouns,
                        "remove_urls": True,
                        "normalize_whitespace": True,
                    }
                }
            }
            self.query_preprocessor = PreprocessorPipeline(preprocessor_config)
            logger.info("Query preprocessor initialized")

        # Step 1: Hybrid Searcher
        searcher_config = {
            "enable_fulltext_search": self.config.enable_fulltext_search,
            "enable_vector_search": self.config.enable_vector_search,
            "enable_graph_search": self.config.enable_graph_search
        }
        self.searcher = SearcherClass(searcher_config)

        # Step 2: Reranker (optional)
        self.reranker = None
        if self.config.enable_reranking:
            reranker_config = {
                "rerank_method": self.config.rerank_method,
                "rerank_model": self.config.rerank_model,
                "rerank_top_k_before": self.config.rerank_top_k_before,
                "rerank_top_k_after": self.config.rerank_top_k_after,
            }
            self.reranker = RerankerStep(reranker_config)
            logger.info("Reranker initialized")

        # Step 3: Cypher Generator
        self.cypher_generator = CypherGenerator()

        # Step 4: Query Expander (Neo4j)
        QueryExpander = _import_query_step(4, "QueryExpander")
        self.query_expander = QueryExpander()

        # Initialize Neo4j driver if available
        self.neo4j_driver = None
        if (
            NEO4J_AVAILABLE
            and self.config.neo4j_uri
            and self.config.neo4j_user
            and self.config.neo4j_password
        ):
            try:
                self.neo4j_driver = GraphDatabase.driver(
                    self.config.neo4j_uri,
                    auth=(self.config.neo4j_user, self.config.neo4j_password),
                )
                # Set the driver in the query expander
                self.query_expander.set_neo4j_driver(self.neo4j_driver)
                logger.info("Neo4j driver initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Neo4j driver: {str(e)}")
                self.neo4j_driver = None
        elif not NEO4J_AVAILABLE:
            logger.warning("Neo4j driver not available - install neo4j package")
        else:
            logger.error("Neo4j configuration incomplete - cannot proceed without Neo4j")
            raise RuntimeError("Neo4j configuration is required for pipeline operation")

        # Step 5: Context Coalescer
        self.context_coalescer = ContextCoalescer(
            config={
                "max_entities": self.config.max_entities,
                "max_claims": self.config.max_claims,
                "max_relationships": self.config.max_relationships,
                "include_timestamps": True,
                "include_full_text": True,
            }
        )

        # Step 6: Result Outputter
        self.result_outputter = ResultOutputter(
            openai_api_key=self.config.openai_api_key,
            config={
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "include_token_usage": self.config.include_token_usage,
                "include_timing": self.config.include_timing,
                "verbose": self.config.verbose,
            },
        )

        logger.info("Pipeline components initialized successfully")



    def process_query(self, user_query: str) -> Any:
        """
        Process a user query through the complete pipeline.

        Args:
            user_query: The user's natural language query

        Returns:
            QueryResult object with the complete result
        """
        logger.info(f"Starting pipeline for query: {user_query}")
        pipeline_start_time = time.time()

        try:
            # Step 0: Preprocess query (optional)
            processed_query = user_query
            if self.query_preprocessor:
                logger.info("Step 0: Preprocessing user query")
                preprocess_result = self.query_preprocessor.process(user_query)
                if preprocess_result.success:
                    processed_query = preprocess_result.data.get("cleaned_query", user_query)
                    logger.info(f"Query preprocessed: '{user_query}' -> '{processed_query}'")
                else:
                    logger.warning(f"Query preprocessing failed: {preprocess_result.error}")

            # Step 1: Hybrid Search
            logger.info("Step 1: Performing hybrid search")
            search_results = self.searcher.execute(processed_query)

            # Step 2: Rerank Search Results (optional)
            if self.reranker:
                logger.info("Step 2: Reranking search results")
                search_results = self.reranker.execute(search_results)

            # Step 3: Generate Cypher Queries
            logger.info("Step 3: Generating Cypher queries")
            query_set = self.cypher_generator.execute(search_results)

            # Step 4: Execute Cypher Queries and Build Graph Context
            logger.info("Step 4: Executing query expansion")
            graph_context = self.query_expander.execute(query_set)

            # Step 5: Coalesce Context into System Prompt
            logger.info("Step 5: Coalescing context into system prompt")
            system_prompt = self.context_coalescer.execute((graph_context, processed_query))

            # Calculate preprocessing time
            preprocessing_time = time.time() - pipeline_start_time

            # Step 6: Send to LLM and Output Results
            logger.info("Step 6: Generating and outputting final result")
            query_result = self.result_outputter.execute(
                (
                    system_prompt,
                    processed_query,
                    preprocessing_time,
                    self.config.output_format,
                    graph_context,
                )
            )

            total_time = time.time() - pipeline_start_time
            logger.info(f"Pipeline completed successfully in {total_time:.2f}s")

            return query_result

        except Exception as e:
            total_time = time.time() - pipeline_start_time
            logger.error(f"Pipeline failed after {total_time:.2f}s: {str(e)}")
            raise

    def close(self):
        """Close pipeline resources."""
        # Close Neo4j driver if open
        if hasattr(self, "neo4j_driver") and self.neo4j_driver:
            try:
                self.neo4j_driver.close()
                logger.info("Neo4j driver closed")
            except Exception as e:
                logger.warning(f"Error closing Neo4j driver: {str(e)}")

        steps = [
            "query_preprocessor",
            "cypher_generator",
            "query_expander",
            "context_coalescer",
            "result_outputter",
        ]

        for step_name in steps:
            if hasattr(self, step_name):
                step = getattr(self, step_name)
                if hasattr(step, "close"):
                    step.close()

        logger.info("Pipeline resources closed")


def create_pipeline_from_env() -> QueryPipeline:
    """
    Create a pipeline instance using environment variables.
    Automatically loads from .env file if available.

    Returns:
        Configured QueryPipeline instance
    """
    # Load environment variables if .env file exists
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    config = PipelineConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),  # Allow model override
        verbose=os.getenv("VERBOSE", "false").lower() == "true",
    )

    if not config.openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    return QueryPipeline(config)


def main():
    """Main entry point for running queries from command line."""
    import argparse
    import sys

    # Load environment variables automatically
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Query the chat knowledge base")
    parser.add_argument("query", help="The query to process")
    parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--save", help="Save result to file")
    parser.add_argument(
        "--model",
        default=None,
        help="Override the default LLM model (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--no-fulltext",
        action="store_true",
        help="Disable full-text search, use only vector search",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable graph search, use only vector search",
    )
    parser.add_argument(
        "--no-vector",
        action="store_true",
        help="Disable vector search, use only full-text search",
    )

    args = parser.parse_args()

    try:
        # Create pipeline from environment with CLI overrides
        config = PipelineConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_user=os.getenv("NEO4J_USER"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            model=args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            output_format=args.format,
            verbose=args.verbose,
            enable_fulltext_search=not args.no_fulltext,
            enable_vector_search=not args.no_vector,
            enable_graph_search=not args.no_graph,
        )

        if not config.openai_api_key:
            print("Error: OPENAI_API_KEY environment variable is required")
            print("Please set it in your .env file or environment")
            sys.exit(1)

        print(f"Using model: {config.model}")
        pipeline = QueryPipeline(config)

        try:
            # Process the query
            result = pipeline.process_query(args.query)

            # Save result if requested
            if args.save:
                pipeline.result_outputter.save_result(result, args.save, args.format)

        finally:
            pipeline.close()

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
