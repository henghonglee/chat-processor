"""
Zero-shot bi-encoder reranker implementation.

Uses bi-encoder models to compute semantic similarity between
query and document embeddings. Faster than cross-encoders.
"""

from typing import List, Tuple

from .base_reranker import BaseReranker
from ..searchers.base_searcher import SearchResult


class ZeRankReranker(BaseReranker):
    """
    Zero-shot bi-encoder reranking using sentence-transformers.

    Faster than cross-encoders but less accurate. Computes cosine similarity
    between query and document embeddings. Good for low-latency scenarios.

    Recommended models:
    - all-MiniLM-L6-v2 (default, fast, good quality)
    - all-mpnet-base-v2 (higher quality, slower)
    - all-distilroberta-v1 (balanced)
    """

    def _setup_model(self):
        """Load the bi-encoder model."""
        try:
            from sentence_transformers import SentenceTransformer, util

            self.model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device
            )
            self.util = util
            self.logger.info(f"Loaded zerank model: {self.config.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load zerank model: {e}")
            raise

    def rerank(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[Tuple[SearchResult, float]]:
        """Rerank using bi-encoder embeddings and cosine similarity."""
        if not results:
            return []

        # Encode query
        query_emb = self.model.encode(query, convert_to_tensor=True)

        # Encode documents
        doc_texts = [r.document_text for r in results]
        doc_embs = self.model.encode(doc_texts, convert_to_tensor=True)

        # Compute cosine similarities
        similarities = self.util.cos_sim(query_emb, doc_embs)[0]

        # Combine with original hybrid scores (average)
        # This leverages both the hybrid search score and semantic similarity
        reranked = []
        for result, sim in zip(results, similarities):
            combined_score = (float(sim) + result.similarity_score) / 2.0
            reranked.append((result, combined_score))

        # Sort by combined score (descending)
        reranked.sort(key=lambda x: x[1], reverse=True)

        self.logger.info(f"ZeRank reranked {len(results)} results")
        return reranked
