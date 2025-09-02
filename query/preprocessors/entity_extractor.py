"""
Entity Extractor Preprocessor

Extracts entities from queries using Google's Gemini LangExtract API.
"""

import time
from typing import Any, Dict, Optional

from .base import BasePreprocessor, PreprocessorResult


class EntityExtractor(BasePreprocessor):
    """Preprocessor for extracting entities from query text using Gemini LangExtract."""

    def get_name(self) -> str:
        """Return the name of this preprocessor."""
        return "entity_extractor"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the entity extractor.

        Args:
            config: Configuration dictionary with options:
                - api_key: Google API key for Gemini
                - model: Gemini model to use (default: 'gemini-pro')
                - confidence_threshold: Minimum confidence for entities (default: 0.5)
                - max_entities: Maximum number of entities to extract (default: 20)
        """
        super().__init__(config)

        # Configure Gemini API
        api_key = self.get_config_value("api_key")
        if api_key:
            try:
                import google.generativeai as genai

                genai.configure(api_key=api_key)
                self.genai = genai
            except ImportError:
                self.logger.warning(
                    "google.generativeai not available, entity extraction will be disabled"
                )
                self.genai = None
        else:
            self.genai = None

        self.model_name = self.get_config_value("model", "gemini-2.5-flash")
        if self.genai:
            self.model = self.genai.GenerativeModel(self.model_name)
        else:
            self.model = None

    def process(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> PreprocessorResult:
        """
        Extract entities from the input text using Gemini.

        Args:
            text: Input text to extract entities from
            context: Optional context from previous processors

        Returns:
            PreprocessorResult with extracted entities
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

            # Check if Gemini is available
            if not self.genai or not self.model:
                self.logger.warning(
                    "Gemini API not configured, skipping entity extraction"
                )
                return PreprocessorResult(
                    success=True,
                    data={
                        "entities": [],
                        "entity_count": 0,
                        "entity_types": [],
                    },
                    processing_time=time.time() - start_time,
                    metadata={
                        "model_used": "none",
                        "confidence_threshold": self.get_config_value(
                            "confidence_threshold", 0.5
                        ),
                        "entity_breakdown": {},
                    },
                )

            # Use cleaned text if available from context
            if context and "cleaned_text" in context:
                text_to_process = context["cleaned_text"]
            else:
                text_to_process = text

            # Extract entities using Gemini
            entities = self._extract_entities_with_gemini(text_to_process)

            # Post-process entities
            entities = self._filter_by_confidence(entities)
            entities = self._deduplicate_entities(entities)

            # Sort by confidence
            entities.sort(key=lambda x: x["confidence"], reverse=True)

            processing_time = time.time() - start_time
            result = PreprocessorResult(
                success=True,
                data={
                    "entities": entities,
                    "entity_count": len(entities),
                    "entity_types": list(set(e["type"] for e in entities)),
                },
                processing_time=processing_time,
                metadata={
                    "model_used": self.model_name,
                    "confidence_threshold": self.get_config_value(
                        "confidence_threshold", 0.5
                    ),
                    "entity_breakdown": self._get_entity_breakdown(entities),
                },
            )

            self.log_processing_end(result)
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Entity extraction failed: {str(e)}"
            self.logger.error(error_msg)

            return PreprocessorResult(
                success=False,
                error=error_msg,
                processing_time=processing_time,
            )

    def _extract_entities_with_gemini(self, text: str) -> list:
        """Extract entities using Gemini LangExtract."""
        if not self.model:
            return []

        prompt = f"""
        Extract named entities from the following text. Return a JSON list of entities with the following format:
        [
            {{
                "text": "entity text",
                "type": "PERSON|ORGANIZATION|LOCATION|CRYPTOCURRENCY|FINANCIAL|DATE|MISC",
                "confidence": 0.95,
                "start": 0,
                "end": 10
            }}
        ]

        Focus on:
        - People names
        - Cryptocurrency symbols and names (BTC, Bitcoin, ETH, Ethereum, etc.)
        - Organizations and companies
        - Financial instruments and prices
        - Dates and time expressions
        - Locations

        Text: {text}

        Return only the JSON array, no additional text.
        """

        try:
            response = self.model.generate_content(prompt)

            if not response or not response.text:
                self.logger.warning("Gemini returned empty response")
                return []

            response_text = response.text.strip()
            self.logger.debug(
                f"Gemini response: {response_text[:500]}..."
            )  # Log first 500 chars

            # Clean the response text - sometimes Gemini returns extra text
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            # Find JSON array in the response
            import json

            start = response_text.find("[")
            end = response_text.rfind("]") + 1

            if start == -1 or end == 0:
                self.logger.warning(f"No JSON array found in response: {response_text}")
                return []

            json_text = response_text[start:end]
            entities_data = json.loads(json_text)

            # Limit number of entities
            max_entities = self.get_config_value("max_entities", 20)
            if len(entities_data) > max_entities:
                entities_data = entities_data[:max_entities]

            self.logger.debug(f"Extracted {len(entities_data)} entities from Gemini")
            return entities_data

        except Exception as e:
            self.logger.warning(f"Gemini entity extraction failed: {e}")
            return []

    def _filter_by_confidence(self, entities: list) -> list:
        """Filter entities by confidence threshold."""
        threshold = self.get_config_value("confidence_threshold", 0.5)
        return [e for e in entities if e.get("confidence", 0) >= threshold]

    def _deduplicate_entities(self, entities: list) -> list:
        """Remove duplicate entities based on text and type."""
        seen = set()
        deduplicated = []

        for entity in entities:
            key = (entity["text"].lower(), entity["type"])
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)

        return deduplicated

    def _get_entity_breakdown(self, entities: list) -> Dict[str, int]:
        """Get breakdown of entities by type."""
        breakdown = {}
        for entity in entities:
            entity_type = entity.get("type", "unknown")
            breakdown[entity_type] = breakdown.get(entity_type, 0) + 1
        return breakdown
