"""
Jina AI API reranker implementation.

Cloud-based reranking using Jina AI's rerank API.
Requires API key and internet connection.
"""

import os
from typing import List, Tuple

from .base_reranker import BaseReranker
from ..searchers.base_searcher import SearchResult


class JinaReranker(BaseReranker):
    """
    Cloud-based reranking using Jina AI's rerank API.

    Alternative to Cohere with competitive pricing and quality.
    Supports multilingual reranking. Requires API key and internet.

    Available models:
    - jina-reranker-v1-base-en (English)
    - jina-reranker-v1-turbo-en (Faster)

    Pricing: ~$0.50 per 1,000 searches (check Jina pricing)

    Environment variables:
    - JINA_API_KEY: Your Jina AI API key
    """

    def _setup_model(self):
        """Validate API key is available."""
        if not self.api_key:
            self.api_key = os.getenv("JINA_API_KEY")

        if not self.api_key:
            raise ValueError("JINA_API_KEY not found in config or environment")

        self.logger.info("Jina reranker initialized")

    def rerank(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[Tuple[SearchResult, float]]:
        """Rerank using Jina API."""
        if not results:
            return []

        try:
            import requests

            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.config.model_name or "jina-reranker-v1-base-en",
                    "query": query,
                    "documents": [r.document_text for r in results],
                    "top_n": len(results)
                },
                timeout=self.config.api_timeout
            )

            response.raise_for_status()
            data = response.json()

            # Map back to original results
            reranked = [
                (results[res["index"]], res["relevance_score"])
                for res in data["results"]
            ]

            self.logger.info(f"Jina reranked {len(results)} results")
            return reranked

        except Exception as e:
            self.logger.error(f"Jina reranking failed: {e}")
            # Fallback: return original results with their scores
            return [(r, r.similarity_score) for r in results]
