"""
Preprocessor Pipeline

Orchestrates all preprocessing strategies in a pipeline and combines their results.
Provides a unified interface for query preprocessing with configurable pipeline stages.
"""

import time
from typing import Any, Dict, Optional

from .base import BasePreprocessor, PreprocessorResult
from .entity_extractor import EntityExtractor
from .intent_classifier import IntentClassifier
from .keyword_extractor import KeywordExtractor
from .text_cleaner import TextCleaner


class PreprocessorPipeline(BasePreprocessor):
    """Pipeline preprocessor that combines all preprocessing strategies."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the preprocessor pipeline.

        Args:
            config: Configuration dictionary with options:
                - pipeline_stages: List of stages to run (default: all)
                - stage_configs: Individual configurations for each stage
                - fail_on_stage_error: Fail entire pipeline if one stage fails (default: False)
                - combine_strategies: How to combine results (default: "merge")
        """
        super().__init__(config)
        self.setup_pipeline()

    def setup_pipeline(self):
        """Setup the preprocessing pipeline."""
        # Default pipeline stages
        default_stages = [
            "text_cleaner",
            "entity_extractor",
            "intent_classifier",
            "keyword_extractor",
        ]
        self.pipeline_stages = self.get_config_value("pipeline_stages", default_stages)

        # Stage configurations
        stage_configs = self.get_config_value("stage_configs", {})

        # Initialize preprocessors
        self.preprocessors = {}

        if "text_cleaner" in self.pipeline_stages:
            self.preprocessors["text_cleaner"] = TextCleaner(
                stage_configs.get("text_cleaner", {})
            )

        if "entity_extractor" in self.pipeline_stages:
            self.preprocessors["entity_extractor"] = EntityExtractor(
                stage_configs.get("entity_extractor", {})
            )

        if "intent_classifier" in self.pipeline_stages:
            self.preprocessors["intent_classifier"] = IntentClassifier(
                stage_configs.get("intent_classifier", {})
            )

        if "keyword_extractor" in self.pipeline_stages:
            self.preprocessors["keyword_extractor"] = KeywordExtractor(
                stage_configs.get("keyword_extractor", {})
            )

    def get_name(self) -> str:
        """Get the name of this preprocessor."""
        return "PreprocessorPipeline"

    def run_stage(
        self, stage_name: str, text: str, context: Dict[str, Any]
    ) -> PreprocessorResult:
        """
        Run a single preprocessing stage.

        Args:
            stage_name: Name of the stage to run
            text: Input text
            context: Current context from previous stages

        Returns:
            PreprocessorResult from the stage
        """
        if stage_name not in self.preprocessors:
            return PreprocessorResult(
                success=False,
                error=f"Stage '{stage_name}' not found in pipeline",
                processing_time=0.0,
            )

        preprocessor = self.preprocessors[stage_name]

        try:
            result = preprocessor.process(text, context)
            self.logger.debug(
                f"Stage '{stage_name}' completed: success={result.success}"
            )
            return result
        except Exception as e:
            error_msg = f"Stage '{stage_name}' failed: {str(e)}"
            self.logger.error(error_msg)
            return PreprocessorResult(
                success=False,
                error=error_msg,
                processing_time=0.0,
            )

    def update_context(
        self, context: Dict[str, Any], stage_name: str, result: PreprocessorResult
    ) -> None:
        """
        Update context with results from a preprocessing stage.

        Args:
            context: Current context dictionary
            stage_name: Name of the stage that produced the result
            result: PreprocessorResult from the stage
        """
        if not result.success:
            context[f"{stage_name}_error"] = result.error
            return

        # Add stage-specific data to context
        if stage_name == "text_cleaner" and result.data:
            context["cleaned_text"] = result.data.get("cleaned_text")
            context["text_cleaning_metadata"] = result.data

        elif stage_name == "entity_extractor" and result.data:
            context["entities"] = result.data.get("entities", [])
            context["entity_types"] = result.data.get("entity_types", [])
            context["entity_extraction_metadata"] = result.data

        elif stage_name == "intent_classifier" and result.data:
            context["primary_intent"] = result.data.get("primary_intent")
            context["intent_confidence"] = result.data.get("confidence")
            context["all_intent_scores"] = result.data.get("all_scores", {})
            context["intent_classification_metadata"] = result.data

        elif stage_name == "keyword_extractor" and result.data:
            context["keywords"] = result.data.get("keywords", [])
            context["keyword_scores"] = result.data.get("keyword_scores", {})
            context["categorized_keywords"] = result.data.get(
                "categorized_keywords", {}
            )
            context["keyword_extraction_metadata"] = result.data

        # Add timing information
        context[f"{stage_name}_processing_time"] = result.processing_time

    def determine_query_type(self, context: Dict[str, Any]) -> str:
        """
        Determine the overall query type based on processed information.

        Args:
            context: Combined context from all stages

        Returns:
            String representing the query type
        """
        entities = context.get("entities", [])
        primary_intent = context.get("primary_intent", "general")

        # Analyze entity composition
        person_entities = [e for e in entities if e.get("type") == "person"]
        crypto_entities = [e for e in entities if e.get("type") == "cryptocurrency"]
        temporal_entities = [e for e in entities if e.get("type") == "temporal"]

        # Determine query type based on entities and intent
        if person_entities and crypto_entities:
            return "person_crypto_query"
        elif person_entities and primary_intent == "relationship":
            return "person_relationship_query"
        elif crypto_entities and primary_intent in ["analysis", "comparison"]:
            return "crypto_analysis_query"
        elif crypto_entities and primary_intent == "action":
            return "crypto_action_query"
        elif temporal_entities or primary_intent == "temporal":
            return "temporal_query"
        elif primary_intent == "sentiment":
            return "sentiment_query"
        elif primary_intent == "relationship":
            return "relationship_query"
        elif primary_intent == "comparison":
            return "comparison_query"
        elif crypto_entities:
            return "crypto_query"
        elif person_entities:
            return "person_query"
        else:
            return "general_query"

    def build_combined_result(
        self, context: Dict[str, Any], total_processing_time: float
    ) -> Dict[str, Any]:
        """
        Build the final combined result from all preprocessing stages.

        Args:
            context: Combined context from all stages
            total_processing_time: Total time for all stages

        Returns:
            Combined result dictionary
        """
        # Determine query type
        query_type = self.determine_query_type(context)

        # Build the final result
        combined_result = {
            # Original and cleaned text
            "original_query": context.get("original_query", ""),
            "cleaned_query": context.get("cleaned_text", ""),
            # Entities
            "entities": context.get("entities", []),
            "entity_count": len(context.get("entities", [])),
            "entity_types": context.get("entity_types", []),
            # Intent
            "intent": context.get("primary_intent", "general"),
            "intent_confidence": context.get("intent_confidence", 0.0),
            "all_intent_scores": context.get("all_intent_scores", {}),
            # Keywords
            "keywords": context.get("keywords", []),
            "keyword_count": len(context.get("keywords", [])),
            "keyword_scores": context.get("keyword_scores", {}),
            "categorized_keywords": context.get("categorized_keywords", {}),
            # Overall classification
            "query_type": query_type,
            # Processing metadata
            "processing_times": {
                stage: context.get(f"{stage}_processing_time", 0.0)
                for stage in self.pipeline_stages
            },
            "total_processing_time": total_processing_time,
            "pipeline_stages_run": self.pipeline_stages,
            "stage_errors": {
                stage: context.get(f"{stage}_error")
                for stage in self.pipeline_stages
                if f"{stage}_error" in context
            },
        }

        return combined_result

    def process(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> PreprocessorResult:
        """
        Process the input text through all preprocessing stages.

        Args:
            text: Input text to process
            context: Optional initial context

        Returns:
            PreprocessorResult with combined results from all stages
        """
        start_time = time.time()
        self.log_processing_start(text)

        try:
            # Validate input
            if not self.validate_input(text):
                return PreprocessorResult(
                    success=False,
                    error="Invalid input text",
                    processing_time=time.time() - start_time,
                )

            # Initialize context
            combined_context = context.copy() if context else {}
            combined_context["original_query"] = text

            fail_on_error = self.get_config_value("fail_on_stage_error", False)
            stage_results = {}
            failed_stages = []

            # Run each stage in the pipeline
            for stage_name in self.pipeline_stages:
                self.logger.debug(f"Running stage: {stage_name}")

                stage_result = self.run_stage(stage_name, text, combined_context)
                stage_results[stage_name] = stage_result

                if stage_result.success:
                    self.update_context(combined_context, stage_name, stage_result)
                else:
                    failed_stages.append(stage_name)
                    if fail_on_error:
                        processing_time = time.time() - start_time
                        return PreprocessorResult(
                            success=False,
                            error=f"Pipeline failed at stage '{stage_name}': {stage_result.error}",
                            processing_time=processing_time,
                        )

            # Build combined result
            total_processing_time = time.time() - start_time
            combined_result = self.build_combined_result(
                combined_context, total_processing_time
            )

            # Determine overall success
            success = len(failed_stages) == 0 or not fail_on_error

            result = PreprocessorResult(
                success=success,
                data=combined_result,
                processing_time=total_processing_time,
                metadata={
                    "pipeline_stages": self.pipeline_stages,
                    "successful_stages": [
                        s for s in self.pipeline_stages if s not in failed_stages
                    ],
                    "failed_stages": failed_stages,
                    "stage_count": len(self.pipeline_stages),
                    "stage_results": {s: r.success for s, r in stage_results.items()},
                },
            )

            self.log_processing_end(result)
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Composite preprocessing failed: {str(e)}"
            self.logger.error(error_msg)

            return PreprocessorResult(
                success=False,
                error=error_msg,
                processing_time=processing_time,
            )
