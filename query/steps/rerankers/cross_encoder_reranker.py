"""
Cross-encoder reranker implementation.

Uses cross-encoder models that jointly encode query+document pairs
to directly model relevance. Most accurate reranking method.
"""

import math
from typing import List, Tuple

from .base_reranker import BaseReranker
from ..searchers.base_searcher import SearchResult


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranking using sentence-transformers.

    Most accurate reranking method. Jointly encodes query+document pairs
    to directly model relevance. Best for quality when latency is acceptable.

    Recommended models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (default, fast)
    - cross-encoder/ms-marco-MiniLM-L-12-v2 (better quality)
    - BAAI/bge-reranker-base (multilingual)
    - BAAI/bge-reranker-large (best quality, slowest)
    """

    def _setup_model(self):
        """Load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(
                self.config.model_name,
                max_length=self.config.max_length,
                device=self.config.device
            )
            self.logger.info(f"Loaded cross-encoder model: {self.config.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load cross-encoder model: {e}")
            raise

    def rerank(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[Tuple[SearchResult, float]]:
        """Rerank using cross-encoder model."""
        if not results:
            return []

        # Prepare query-document pairs
        pairs = [[query, r.document_text] for r in results]

        # Get relevance scores from cross-encoder
        scores = self.model.predict(pairs, batch_size=self.config.batch_size)

        # Normalize scores using sigmoid to [0, 1] range
        normalized_scores = [1.0 / (1.0 + math.exp(-score)) for score in scores]

        # Pair results with normalized scores
        reranked = list(zip(results, normalized_scores))

        # Sort by score (descending)
        reranked.sort(key=lambda x: x[1], reverse=True)

        self.logger.info(f"Cross-encoder reranked {len(results)} results")
        return reranked
