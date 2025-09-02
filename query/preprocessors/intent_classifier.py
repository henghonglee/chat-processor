"""
Intent Classifier Preprocessor

Classifies user intent from queries including:
- Search intents (find, show, what, who, etc.)
- Analysis intents (compare, analyze, trend)
- Sentiment intents (opinion, feeling)
- Relationship intents (connection, between)
- Temporal intents (timeline, history, when)
- Action intents (buy, sell, trade)
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from .base import BasePreprocessor, PreprocessorResult


class IntentClassifier(BasePreprocessor):
    """Preprocessor for classifying user intent from query text."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the intent classifier.

        Args:
            config: Configuration dictionary with options:
                - intent_categories: Custom intent categories to use
                - confidence_threshold: Minimum confidence for intent (default: 0.3)
                - use_entity_context: Use extracted entities for intent (default: True)
                - multi_intent: Allow multiple intents (default: False)
        """
        super().__init__(config)
        self.setup_intent_patterns()

    def setup_intent_patterns(self):
        """Setup intent patterns and keywords."""
        default_intent_patterns = {
            "search": {
                "keywords": [
                    "find",
                    "search",
                    "look for",
                    "show me",
                    "what",
                    "who",
                    "when",
                    "where",
                    "which",
                    "how",
                    "get",
                    "retrieve",
                    "display",
                    "list",
                ],
                "patterns": [
                    r"\bwhat\s+(is|are|was|were)\b",
                    r"\bwho\s+(is|are|was|were)\b",
                    r"\bshow\s+me\b",
                    r"\bfind\s+(all|any)\b",
                    r"\btell\s+me\b",
                ],
                "weight": 1.0,
            },
            "analysis": {
                "keywords": [
                    "analyze",
                    "compare",
                    "trend",
                    "pattern",
                    "correlation",
                    "performance",
                    "statistics",
                    "metrics",
                    "data",
                    "insight",
                    "analysis",
                    "versus",
                    "vs",
                ],
                "patterns": [
                    r"\bcompare\s+\w+\s+(to|with|against)\b",
                    r"\banalyze\s+the\b",
                    r"\btrend\s+(analysis|of)\b",
                    r"\bhow\s+(does|did)\s+\w+\s+perform\b",
                ],
                "weight": 1.2,
            },
            "sentiment": {
                "keywords": [
                    "feel",
                    "think",
                    "opinion",
                    "sentiment",
                    "positive",
                    "negative",
                    "bullish",
                    "bearish",
                    "optimistic",
                    "pessimistic",
                    "mood",
                    "vibe",
                    "reaction",
                    "impression",
                    "view",
                ],
                "patterns": [
                    r"\bhow\s+(do|did)\s+\w+\s+feel\b",
                    r"\bwhat\s+(do|did)\s+\w+\s+think\b",
                    r"\bsentiment\s+(around|about|on)\b",
                    r"\b(positive|negative|bullish|bearish)\s+(sentiment|opinion)\b",
                ],
                "weight": 1.1,
            },
            "relationship": {
                "keywords": [
                    "relationship",
                    "connection",
                    "link",
                    "between",
                    "related to",
                    "connected to",
                    "associated with",
                    "correlation",
                    "interaction",
                    "network",
                    "ties",
                    "bonds",
                ],
                "patterns": [
                    r"\bbetween\s+\w+\s+and\s+\w+\b",
                    r"\brelationship\s+between\b",
                    r"\bconnection\s+(between|to)\b",
                    r"\bhow\s+\w+\s+(relates?|connects?)\s+to\b",
                ],
                "weight": 1.3,
            },
            "temporal": {
                "keywords": [
                    "timeline",
                    "history",
                    "chronological",
                    "over time",
                    "sequence",
                    "when",
                    "time",
                    "date",
                    "period",
                    "duration",
                    "recent",
                    "past",
                    "future",
                    "yesterday",
                    "today",
                    "tomorrow",
                    "last",
                    "next",
                ],
                "patterns": [
                    r"\btimeline\s+of\b",
                    r"\bover\s+(time|the\s+(past|last))\b",
                    r"\b(last|past|recent)\s+\w+\b",
                    r"\bwhen\s+(did|was|were)\b",
                    r"\bhistory\s+of\b",
                ],
                "weight": 1.1,
            },
            "action": {
                "keywords": [
                    "buy",
                    "sell",
                    "trade",
                    "purchase",
                    "acquire",
                    "invest",
                    "divest",
                    "exchange",
                    "swap",
                    "convert",
                    "transfer",
                    "send",
                    "receive",
                ],
                "patterns": [
                    r"\b(buy|sell|trade|purchase)\s+\w+\b",
                    r"\binvest\s+in\b",
                    r"\bshould\s+(i|we)\s+(buy|sell)\b",
                    r"\bhow\s+to\s+(buy|sell|trade)\b",
                ],
                "weight": 1.4,
            },
            "informational": {
                "keywords": [
                    "explain",
                    "definition",
                    "meaning",
                    "understand",
                    "learn",
                    "about",
                    "information",
                    "details",
                    "overview",
                    "summary",
                ],
                "patterns": [
                    r"\bexplain\s+\w+\b",
                    r"\bwhat\s+is\s+\w+\b",
                    r"\btell\s+me\s+about\b",
                    r"\bi\s+(want\s+to\s+)?(understand|learn)\b",
                ],
                "weight": 1.0,
            },
            "comparison": {
                "keywords": [
                    "compare",
                    "versus",
                    "vs",
                    "difference",
                    "better",
                    "worse",
                    "which",
                    "between",
                    "against",
                    "than",
                    "similar",
                    "different",
                ],
                "patterns": [
                    r"\bcompare\s+\w+\s+(vs|versus|to|with|against)\s+\w+\b",
                    r"\b\w+\s+(vs|versus)\s+\w+\b",
                    r"\bwhich\s+is\s+(better|worse)\b",
                    r"\bdifference\s+between\b",
                ],
                "weight": 1.2,
            },
        }

        self.intent_patterns = self.get_config_value(
            "intent_categories", default_intent_patterns
        )

    def get_name(self) -> str:
        """Get the name of this preprocessor."""
        return "IntentClassifier"

    def calculate_keyword_scores(self, text: str) -> Dict[str, float]:
        """Calculate intent scores based on keyword matching."""
        text_lower = text.lower()
        scores = {}

        for intent, config in self.intent_patterns.items():
            score = 0.0
            keywords = config.get("keywords", [])
            weight = config.get("weight", 1.0)

            # Count keyword matches
            keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)

            if keyword_matches > 0:
                # Normalize by keyword count and apply weight
                score = (keyword_matches / len(keywords)) * weight

            scores[intent] = score

        return scores

    def calculate_pattern_scores(self, text: str) -> Dict[str, float]:
        """Calculate intent scores based on regex pattern matching."""
        import re

        scores = {}

        for intent, config in self.intent_patterns.items():
            score = 0.0
            patterns = config.get("patterns", [])
            weight = config.get("weight", 1.0)

            # Count pattern matches
            pattern_matches = 0
            for pattern_str in patterns:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                if pattern.search(text):
                    pattern_matches += 1

            if pattern_matches > 0:
                # Normalize by pattern count and apply weight
                score = (pattern_matches / len(patterns)) * weight

            scores[intent] = score

        return scores

    def incorporate_entity_context(
        self, text: str, entities: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Adjust intent scores based on extracted entities."""
        scores = {}

        if not entities:
            return scores

        # Entity type to intent mapping
        entity_intent_mapping = {
            "cryptocurrency": ["search", "analysis", "sentiment", "action"],
            "person": ["search", "relationship", "sentiment"],
            "price": ["action", "analysis", "comparison"],
            "temporal": ["temporal", "analysis"],
            "organization": ["search", "analysis", "relationship"],
            "protocol": ["search", "analysis", "action"],
        }

        entity_types = [e.get("type") for e in entities]

        for entity_type in entity_types:
            if entity_type in entity_intent_mapping:
                for intent in entity_intent_mapping[entity_type]:
                    scores[intent] = scores.get(intent, 0) + 0.2

        # Specific entity combinations
        if any(e.get("type") == "person" for e in entities) and any(
            e.get("type") == "cryptocurrency" for e in entities
        ):
            scores["relationship"] = scores.get("relationship", 0) + 0.3

        if len([e for e in entities if e.get("type") == "cryptocurrency"]) >= 2:
            scores["comparison"] = scores.get("comparison", 0) + 0.3

        return scores

    def detect_question_type(self, text: str) -> Tuple[str, float]:
        """Detect question type for better intent classification."""
        text_lower = text.lower().strip()

        question_patterns = {
            "what": ("informational", 0.8),
            "who": ("search", 0.9),
            "when": ("temporal", 0.9),
            "where": ("search", 0.8),
            "why": ("analysis", 0.8),
            "how": ("analysis", 0.7),
            "which": ("comparison", 0.8),
            "should": ("action", 0.7),
            "can": ("action", 0.6),
        }

        for question_word, (intent, confidence) in question_patterns.items():
            if text_lower.startswith(question_word):
                return intent, confidence

        return "general", 0.5

    def classify_intent(
        self, text: str, entities: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify the intent of the input text.

        Args:
            text: Input text to classify
            entities: Optional list of extracted entities

        Returns:
            Tuple of (primary_intent, confidence, all_scores)
        """
        # Calculate base scores
        keyword_scores = self.calculate_keyword_scores(text)
        pattern_scores = self.calculate_pattern_scores(text)

        # Combine scores
        combined_scores = {}
        for intent in self.intent_patterns.keys():
            combined_scores[intent] = keyword_scores.get(
                intent, 0
            ) + pattern_scores.get(intent, 0)

        # Incorporate entity context if available and enabled
        if self.get_config_value("use_entity_context", True) and entities:
            entity_scores = self.incorporate_entity_context(text, entities)
            for intent, score in entity_scores.items():
                combined_scores[intent] = combined_scores.get(intent, 0) + score

        # Detect question type
        question_intent, question_confidence = self.detect_question_type(text)
        if question_intent in combined_scores:
            combined_scores[question_intent] += question_confidence

        # Find the best intent
        if not combined_scores or max(combined_scores.values()) == 0:
            return "general", 0.5, combined_scores

        best_intent = max(combined_scores.items(), key=lambda x: x[1])
        primary_intent, raw_score = best_intent

        # Normalize confidence to 0-1 range
        max_possible_score = 3.0  # Rough estimate of max possible score
        confidence = min(raw_score / max_possible_score, 1.0)

        # Apply minimum confidence threshold
        threshold = self.get_config_value("confidence_threshold", 0.3)
        if confidence < threshold:
            return "general", confidence, combined_scores

        return primary_intent, confidence, combined_scores

    def get_multiple_intents(
        self, all_scores: Dict[str, float], threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """Get multiple intents above threshold, sorted by confidence."""
        intents = [
            (intent, score)
            for intent, score in all_scores.items()
            if score >= threshold
        ]
        return sorted(intents, key=lambda x: x[1], reverse=True)

    def process(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> PreprocessorResult:
        """
        Classify the intent of the input text.

        Args:
            text: Input text to classify intent from
            context: Optional context from previous processors

        Returns:
            PreprocessorResult with classified intent
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

            # Use cleaned text if available from context
            if context and "cleaned_text" in context:
                text_to_process = context["cleaned_text"]
            else:
                text_to_process = text

            # Get entities from context if available
            entities = context.get("entities", []) if context else []

            # Classify intent
            primary_intent, confidence, all_scores = self.classify_intent(
                text_to_process, entities
            )

            # Get multiple intents if configured
            result_data = {
                "primary_intent": primary_intent,
                "confidence": confidence,
                "all_scores": all_scores,
            }

            if self.get_config_value("multi_intent", False):
                multi_threshold = self.get_config_value("multi_intent_threshold", 0.5)
                multiple_intents = self.get_multiple_intents(
                    all_scores, multi_threshold
                )
                result_data["multiple_intents"] = multiple_intents

            processing_time = time.time() - start_time
            result = PreprocessorResult(
                success=True,
                data=result_data,
                processing_time=processing_time,
                metadata={
                    "classification_methods": [
                        "keyword_matching",
                        "pattern_matching",
                        "question_detection",
                    ],
                    "entity_context_used": bool(entities),
                    "confidence_threshold": self.get_config_value(
                        "confidence_threshold", 0.3
                    ),
                    "intent_categories": list(self.intent_patterns.keys()),
                },
            )

            self.log_processing_end(result)
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Intent classification failed: {str(e)}"
            self.logger.error(error_msg)

            return PreprocessorResult(
                success=False,
                error=error_msg,
                processing_time=processing_time,
            )
