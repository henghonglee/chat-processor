"""
Keyword Extractor Preprocessor

Extracts important keywords from queries using:
- Stop word filtering
- TF-IDF scoring (simple implementation)
- N-gram extraction
- Domain-specific keyword detection
- Context-aware filtering
"""

import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .base import BasePreprocessor, PreprocessorResult


class KeywordExtractor(BasePreprocessor):
    """Preprocessor for extracting important keywords from query text."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the keyword extractor.

        Args:
            config: Configuration dictionary with options:
                - max_keywords: Maximum number of keywords to extract (default: 10)
                - min_word_length: Minimum word length (default: 2)
                - include_ngrams: Include n-grams (default: True)
                - ngram_range: Range of n-grams to extract (default: (1, 2))
                - use_tfidf: Use TF-IDF scoring (default: True)
                - custom_stop_words: Additional stop words
                - boost_domain_terms: Boost crypto/finance terms (default: True)
        """
        super().__init__(config)
        self.setup_stop_words()
        self.setup_domain_terms()

    def setup_stop_words(self):
        """Setup stop words for filtering."""
        default_stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "under",
            "between",
            "among",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "us",
            "them",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "now",
        }

        # Add custom stop words if provided
        custom_stop_words = set(self.get_config_value("custom_stop_words", []))
        self.stop_words = default_stop_words.union(custom_stop_words)

    def setup_domain_terms(self):
        """Setup domain-specific terms for boosting."""
        self.crypto_terms = {
            "bitcoin",
            "ethereum",
            "blockchain",
            "cryptocurrency",
            "crypto",
            "defi",
            "nft",
            "token",
            "coin",
            "wallet",
            "exchange",
            "trading",
            "mining",
            "staking",
            "yield",
            "liquidity",
            "protocol",
            "dapp",
            "smart contract",
            "consensus",
            "proof of stake",
            "proof of work",
            "node",
            "validator",
            "mempool",
            "gas",
            "gwei",
            "satoshi",
            "wei",
            "bull",
            "bear",
            "hodl",
            "fomo",
            "fud",
            "ath",
            "atl",
            "market cap",
            "volume",
            "whale",
        }

        self.finance_terms = {
            "investment",
            "portfolio",
            "asset",
            "equity",
            "bond",
            "stock",
            "share",
            "dividend",
            "return",
            "yield",
            "interest",
            "risk",
            "volatility",
            "market",
            "price",
            "value",
            "valuation",
            "growth",
            "profit",
            "loss",
            "bull market",
            "bear market",
            "correction",
            "recession",
            "inflation",
        }

        self.analysis_terms = {
            "trend",
            "pattern",
            "correlation",
            "analysis",
            "metric",
            "data",
            "performance",
            "comparison",
            "sentiment",
            "indicator",
            "signal",
            "statistics",
            "average",
            "median",
            "variance",
            "deviation",
        }

        self.domain_terms = self.crypto_terms.union(self.finance_terms).union(
            self.analysis_terms
        )

    def get_name(self) -> str:
        """Get the name of this preprocessor."""
        return "KeywordExtractor"

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for keyword extraction."""
        # Convert to lowercase and split into words
        words = text.lower().split()

        # Remove punctuation and filter by length
        min_length = self.get_config_value("min_word_length", 2)
        processed_words = []

        for word in words:
            # Remove punctuation
            cleaned_word = "".join(c for c in word if c.isalnum())

            # Filter by length and stop words
            if (
                len(cleaned_word) >= min_length
                and cleaned_word not in self.stop_words
                and cleaned_word.isalpha()
            ):
                processed_words.append(cleaned_word)

        return processed_words

    def extract_ngrams(self, words: List[str], n: int) -> List[str]:
        """Extract n-grams from word list."""
        if len(words) < n:
            return []

        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            ngrams.append(ngram)

        return ngrams

    def calculate_word_frequency(self, words: List[str]) -> Dict[str, int]:
        """Calculate word frequency."""
        return dict(Counter(words))

    def calculate_tfidf_scores(
        self, words: List[str], frequencies: Dict[str, int]
    ) -> Dict[str, float]:
        """Calculate simple TF-IDF scores."""
        total_words = len(words)
        tfidf_scores = {}

        for word, freq in frequencies.items():
            # Term Frequency
            tf = freq / total_words

            # Simple IDF approximation (higher score for domain terms)
            if word in self.domain_terms:
                idf = 2.0  # Boost domain terms
            elif freq == 1:
                idf = 1.5  # Boost rare terms
            else:
                idf = 1.0  # Normal terms

            tfidf_scores[word] = tf * idf

        return tfidf_scores

    def boost_domain_keywords(
        self, keyword_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Boost domain-specific keywords."""
        if not self.get_config_value("boost_domain_terms", True):
            return keyword_scores

        boosted_scores = keyword_scores.copy()

        for keyword in keyword_scores:
            # Check if keyword or any part of it is a domain term
            keyword_lower = keyword.lower()

            if keyword_lower in self.domain_terms:
                boosted_scores[keyword] *= 1.5
            elif any(term in keyword_lower for term in self.domain_terms):
                boosted_scores[keyword] *= 1.2

        return boosted_scores

    def filter_entity_overlap(
        self, keywords: List[str], entities: List[Dict[str, Any]]
    ) -> List[str]:
        """Remove keywords that are already captured as entities."""
        if not entities:
            return keywords

        entity_texts = {e.get("text", "").lower() for e in entities}
        entity_words = set()

        for entity_text in entity_texts:
            entity_words.update(entity_text.split())

        filtered_keywords = []
        for keyword in keywords:
            keyword_words = set(keyword.lower().split())
            # Keep keyword if it doesn't completely overlap with entities
            if not keyword_words.issubset(entity_words):
                filtered_keywords.append(keyword)

        return filtered_keywords

    def extract_keywords(
        self, text: str, entities: Optional[List[Dict[str, Any]]] = None
    ) -> List[Tuple[str, float]]:
        """
        Extract keywords from text with scoring.

        Args:
            text: Input text
            entities: Optional entities to avoid duplication

        Returns:
            List of (keyword, score) tuples sorted by score
        """
        # Preprocess text
        words = self.preprocess_text(text)

        if not words:
            return []

        # Calculate frequencies
        frequencies = self.calculate_word_frequency(words)

        # Calculate scores
        if self.get_config_value("use_tfidf", True):
            keyword_scores = self.calculate_tfidf_scores(words, frequencies)
        else:
            # Simple frequency scoring
            max_freq = max(frequencies.values()) if frequencies else 1
            keyword_scores = {
                word: freq / max_freq for word, freq in frequencies.items()
            }

        # Add n-grams if enabled
        if self.get_config_value("include_ngrams", True):
            ngram_range = self.get_config_value("ngram_range", (1, 2))
            for n in range(ngram_range[0], ngram_range[1] + 1):
                if n > 1:  # Skip unigrams as they're already included
                    ngrams = self.extract_ngrams(words, n)
                    ngram_freqs = self.calculate_word_frequency(ngrams)

                    # Score n-grams
                    for ngram, freq in ngram_freqs.items():
                        if (
                            freq >= 1
                        ):  # Only include n-grams that appear multiple times or are domain terms
                            score = freq * 0.8  # Slightly lower than unigrams
                            if ngram.lower() in self.domain_terms:
                                score *= 1.3
                            keyword_scores[ngram] = score

        # Boost domain keywords
        keyword_scores = self.boost_domain_keywords(keyword_scores)

        # Convert to list and sort
        keyword_list = list(keyword_scores.items())
        keyword_list.sort(key=lambda x: x[1], reverse=True)

        # Filter entity overlap
        keywords_only = [kw for kw, _ in keyword_list]
        filtered_keywords = self.filter_entity_overlap(keywords_only, entities)

        # Rebuild list with scores
        filtered_keyword_list = [
            (kw, score) for kw, score in keyword_list if kw in filtered_keywords
        ]

        # Limit to max keywords
        max_keywords = self.get_config_value("max_keywords", 10)
        return filtered_keyword_list[:max_keywords]

    def categorize_keywords(
        self, keywords: List[Tuple[str, float]]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Categorize keywords by domain."""
        categories = {"crypto": [], "finance": [], "analysis": [], "general": []}

        for keyword, score in keywords:
            keyword_lower = keyword.lower()

            if keyword_lower in self.crypto_terms or any(
                term in keyword_lower for term in self.crypto_terms
            ):
                categories["crypto"].append((keyword, score))
            elif keyword_lower in self.finance_terms or any(
                term in keyword_lower for term in self.finance_terms
            ):
                categories["finance"].append((keyword, score))
            elif keyword_lower in self.analysis_terms or any(
                term in keyword_lower for term in self.analysis_terms
            ):
                categories["analysis"].append((keyword, score))
            else:
                categories["general"].append((keyword, score))

        return categories

    def process(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> PreprocessorResult:
        """
        Extract keywords from the input text.

        Args:
            text: Input text to extract keywords from
            context: Optional context from previous processors

        Returns:
            PreprocessorResult with extracted keywords
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

            # Extract keywords
            keyword_tuples = self.extract_keywords(text_to_process, entities)

            # Separate keywords and scores
            keywords = [kw for kw, _ in keyword_tuples]
            keyword_scores = {kw: score for kw, score in keyword_tuples}

            # Categorize keywords
            categorized_keywords = self.categorize_keywords(keyword_tuples)

            processing_time = time.time() - start_time
            result = PreprocessorResult(
                success=True,
                data={
                    "keywords": keywords,
                    "keyword_scores": keyword_scores,
                    "keyword_count": len(keywords),
                    "categorized_keywords": categorized_keywords,
                },
                processing_time=processing_time,
                metadata={
                    "extraction_method": (
                        "tfidf"
                        if self.get_config_value("use_tfidf", True)
                        else "frequency"
                    ),
                    "ngrams_included": self.get_config_value("include_ngrams", True),
                    "domain_boost_applied": self.get_config_value(
                        "boost_domain_terms", True
                    ),
                    "max_keywords": self.get_config_value("max_keywords", 10),
                    "entity_overlap_filtered": bool(entities),
                },
            )

            self.log_processing_end(result)
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Keyword extraction failed: {str(e)}"
            self.logger.error(error_msg)

            return PreprocessorResult(
                success=False,
                error=error_msg,
                processing_time=processing_time,
            )
