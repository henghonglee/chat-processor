"""
Base reranker abstract class.

Defines the interface that all reranker implementations must follow.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..searchers.base_searcher import SearchResult


@dataclass
class RerankConfig:
    """Configuration for reranking step."""

    # Model selection
    method: str = "zerank"  # zerank, cross-encoder, cohere, jina
    model_name: str = "zerank/zerank-v2-large"

    # Reranking parameters
    top_k_before: int = 500   # How many results to rerank
    top_k_after: int = 50     # How many to return after reranking
    min_score_threshold: float = 0.0

    # Feature toggles
    use_query_expansion: bool = True
    boost_exact_matches: bool = True
    boost_recent_results: bool = False

    # Boosting factors
    exact_match_boost: float = 1.3
    recent_boost_factor: float = 1.1
    entity_match_boost: float = 1.2

    # Model settings
    batch_size: int = 32
    device: str = "cpu"  # or "cuda"
    max_length: int = 512

    # API settings
    api_key: Optional[str] = None
    api_timeout: int = 30


class BaseReranker(ABC):
    """Abstract base class for all reranking implementations."""

    def __init__(self, config: RerankConfig):
        """Initialize the reranker with configuration."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.api_key = config.api_key
        self._setup_model()

    def _setup_model(self):
        """Setup method for model initialization (override in subclasses)."""
        pass

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[Tuple[SearchResult, float]]:
        """
        Rerank results by semantic relevance.

        Args:
            query: The user's search query
            results: List of SearchResult objects to rerank

        Returns:
            List of (SearchResult, rerank_score) tuples sorted by score (descending)
        """
        pass
