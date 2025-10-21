"""
Cohere API reranker implementation.

Cloud-based reranking using Cohere's rerank API.
Requires API key and internet connection.
"""

import os
from typing import List, Tuple

from .base_reranker import BaseReranker
from ..searchers.base_searcher import SearchResult


class CohereReranker(BaseReranker):
    """
    Cloud-based reranking using Cohere's rerank API.

    High-quality reranking without managing models locally.
    Requires API key and internet connection. Good for production
    when you want quality without infrastructure complexity.

    Available models:
    - rerank-english-v2.0 (English)
    - rerank-multilingual-v2.0 (Multilingual)

    Pricing: ~$1 per 1,000 searches (check Cohere pricing)

    Environment variables:
    - COHERE_API_KEY: Your Cohere API key
    """

    def _setup_model(self):
        """Validate API key is available."""
        if not self.api_key:
            self.api_key = os.getenv("COHERE_API_KEY")

        if not self.api_key:
            raise ValueError("COHERE_API_KEY not found in config or environment")

        self.logger.info("Cohere reranker initialized")

    def rerank(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[Tuple[SearchResult, float]]:
        """Rerank using Cohere API."""
        if not results:
            return []

        try:
            import cohere

            co = cohere.Client(self.api_key)
            documents = [r.document_text for r in results]

            response = co.rerank(
                query=query,
                documents=documents,
                top_n=len(results),  # Get scores for all documents
                model=self.config.model_name or "rerank-english-v2.0"
            )

            # Map back to original results using index
            reranked = [
                (results[res.index], res.relevance_score)
                for res in response.results
            ]

            self.logger.info(f"Cohere reranked {len(results)} results")
            return reranked

        except Exception as e:
            self.logger.error(f"Cohere reranking failed: {e}")
            # Fallback: return original results with their scores
            return [(r, r.similarity_score) for r in results]
