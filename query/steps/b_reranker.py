"""
Step 1.5: Semantic Reranker

Reranks search results from hybrid search using advanced semantic similarity models.
Positioned between hybrid search (Step 1) and Cypher generation (Step 2) to improve
result quality through cross-encoder or bi-encoder reranking.

Supported Methods:
1. Cross-Encoder - Most accurate, models query-document interaction directly
2. ZeRank - Fast bi-encoder similarity, works offline
3. Cohere API - Cloud-based reranking service
4. Jina API - Alternative cloud-based reranking

Architecture:
Input: VectorSearchResults from hybrid search
Process: Semantic reranking + boosting + filtering
Output: VectorSearchResults with reranked results
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

from .base import BaseQueryStep
from .a_hybrid_search import VectorSearchResults
from .searchers.base_searcher import SearchResult
from .rerankers import (
    BaseReranker,
    RerankConfig,
    CrossEncoderReranker,
    ZeRankReranker,
    CohereReranker,
    JinaReranker,
)

load_dotenv()


class RerankerStep(BaseQueryStep):
    """
    Main reranker step that integrates into the query pipeline.

    Takes VectorSearchResults from hybrid search, reranks the top-K results
    using semantic similarity models, applies boosting heuristics, filters
    by threshold, and returns refined VectorSearchResults.
    """

    def setup(self):
        """Setup reranker configuration and model."""
        # Load configuration from config dict and environment variables
        self.rerank_config = self._load_config()

        # Create appropriate reranker instance
        self.reranker = self._create_reranker()

        self.logger.info(f"Reranker initialized: {self.rerank_config.method}")
        self.logger.info(f"  Model: {self.rerank_config.model_name}")
        self.logger.info(f"  Top-K before: {self.rerank_config.top_k_before}")
        self.logger.info(f"  Top-K after: {self.rerank_config.top_k_after}")

    def _load_config(self) -> RerankConfig:
        """Load configuration from config dict and environment variables."""
        return RerankConfig(
            # Method and model
            method=self.config.get(
                "rerank_method",
                os.getenv("RERANK_METHOD", "cross-encoder")
            ),
            model_name=self.config.get(
                "rerank_model",
                os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            ),

            # Result counts
            top_k_before=int(self.config.get(
                "rerank_top_k_before",
                os.getenv("RERANK_TOP_K_BEFORE", "50")
            )),
            top_k_after=int(self.config.get(
                "rerank_top_k_after",
                os.getenv("RERANK_TOP_K_AFTER", "20")
            )),
            min_score_threshold=float(self.config.get(
                "rerank_min_score",
                os.getenv("RERANK_MIN_SCORE", "0.0")
            )),

            # Feature toggles
            use_query_expansion=self.config.get(
                "rerank_use_query_expansion",
                os.getenv("RERANK_USE_QUERY_EXPANSION", "true").lower() == "true"
            ),
            boost_exact_matches=self.config.get(
                "rerank_boost_exact_matches",
                os.getenv("RERANK_BOOST_EXACT_MATCHES", "true").lower() == "true"
            ),
            boost_recent_results=self.config.get(
                "rerank_boost_recent",
                os.getenv("RERANK_BOOST_RECENT", "false").lower() == "true"
            ),

            # Boosting factors
            exact_match_boost=float(self.config.get(
                "rerank_exact_match_boost",
                os.getenv("RERANK_EXACT_MATCH_BOOST", "1.3")
            )),
            entity_match_boost=float(self.config.get(
                "rerank_entity_boost",
                os.getenv("RERANK_ENTITY_BOOST", "1.2")
            )),
            recent_boost_factor=float(self.config.get(
                "rerank_recent_boost",
                os.getenv("RERANK_RECENT_BOOST", "1.1")
            )),

            # Model settings
            batch_size=int(self.config.get(
                "rerank_batch_size",
                os.getenv("RERANK_BATCH_SIZE", "32")
            )),
            device=self.config.get(
                "rerank_device",
                os.getenv("RERANK_DEVICE", "cpu")
            ),
            max_length=int(self.config.get(
                "rerank_max_length",
                os.getenv("RERANK_MAX_LENGTH", "512")
            )),

            # API settings
            api_key=self.config.get("rerank_api_key"),
            api_timeout=int(self.config.get(
                "rerank_api_timeout",
                os.getenv("RERANK_API_TIMEOUT", "30")
            ))
        )

    def _create_reranker(self) -> BaseReranker:
        """Factory method to create the appropriate reranker instance."""
        method = self.rerank_config.method.lower()

        if method == "zerank":
            return ZeRankReranker(self.rerank_config)
        elif method == "cross-encoder":
            return CrossEncoderReranker(self.rerank_config)
        elif method == "cohere":
            return CohereReranker(self.rerank_config)
        elif method == "jina":
            return JinaReranker(self.rerank_config)
        else:
            raise ValueError(
                f"Unknown rerank method: {method}. "
                f"Supported: zerank, cross-encoder, cohere, jina"
            )

    def process(self, input_data: VectorSearchResults) -> VectorSearchResults:
        """
        Main processing method - reranks hybrid search results.

        Args:
            input_data: VectorSearchResults from hybrid search step

        Returns:
            VectorSearchResults with reranked and filtered results
        """
        self.logger.info(f"Reranking {len(input_data.results)} results")

        # 1. Extract top-K results to rerank
        results_to_rerank = input_data.results[:self.rerank_config.top_k_before]

        if not results_to_rerank:
            self.logger.warning("No results to rerank")
            return input_data

        # Log BEFORE reranking
        self.logger.info("=" * 80)
        self.logger.info("BEFORE RERANKING:")
        self.logger.info("=" * 80)
        for i, result in enumerate(results_to_rerank[:10], 1):
            self.logger.info(
                f"  {i}. [{result.node_type}] {result.node_id[:50]} "
                f"(score: {result.similarity_score:.4f}, method: {result.search_method})"
            )
            self.logger.info(f"     Text: {result.document_text[:100]}...")
        if len(results_to_rerank) > 10:
            self.logger.info(f"  ... and {len(results_to_rerank) - 10} more results")

        # 2. Rerank using selected method
        reranked = self.reranker.rerank(
            input_data.original_query,
            results_to_rerank
        )

        # 3. Apply boosting factors (if enabled)
        if self.rerank_config.use_query_expansion or self.rerank_config.boost_exact_matches:
            reranked = self._apply_boosting(
                reranked,
                input_data.original_query,
                input_data.extracted_entities,
                input_data.extracted_keywords
            )

        # 4. Filter by threshold
        reranked = self._filter_by_threshold(reranked)

        # 5. Limit to top_k_after
        reranked = reranked[:self.rerank_config.top_k_after]

        # Log AFTER reranking
        self.logger.info("=" * 80)
        self.logger.info("AFTER RERANKING:")
        self.logger.info("=" * 80)
        for i, (result, score) in enumerate(reranked[:10], 1):
            self.logger.info(
                f"  {i}. [{result.node_type}] {result.node_id[:50]} "
                f"(score: {score:.4f}, method: {result.search_method})"
            )
            self.logger.info(f"     Text: {result.document_text[:100]}...")
        if len(reranked) > 10:
            self.logger.info(f"  ... and {len(reranked) - 10} more results")
        self.logger.info("=" * 80)

        self.logger.info(f"After reranking and filtering: {len(reranked)} results")

        # 6. Update SearchResult objects with new scores
        updated_results = self._update_results_with_scores(reranked)

        # 7. Rebuild VectorSearchResults with reranked results
        return self._rebuild_vector_search_results(updated_results, input_data)

    def _apply_boosting(
        self,
        results_with_scores: List[Tuple[SearchResult, float]],
        query: str,
        entities: List[str],
        keywords: List[str]
    ) -> List[Tuple[SearchResult, float]]:
        """Apply all boosting factors to reranked results."""
        boosted = []

        for result, score in results_with_scores:
            boost = 1.0

            # Exact match boosting
            if self.rerank_config.boost_exact_matches:
                boost *= self._apply_exact_match_boost(result, keywords)

            # Entity match boosting
            if self.rerank_config.use_query_expansion:
                boost *= self._apply_entity_boost(result, entities)

            # Recency boosting
            if self.rerank_config.boost_recent_results:
                boost *= self._apply_recency_boost(result)

            boosted_score = score * boost
            boosted.append((result, boosted_score))

        # Re-sort by boosted scores
        boosted.sort(key=lambda x: x[1], reverse=True)

        return boosted

    def _apply_exact_match_boost(
        self,
        result: SearchResult,
        keywords: List[str]
    ) -> float:
        """Boost results containing exact query keywords."""
        doc_lower = result.document_text.lower()

        for keyword in keywords:
            if keyword.lower() in doc_lower:
                return self.rerank_config.exact_match_boost

        return 1.0  # No boost

    def _apply_entity_boost(
        self,
        result: SearchResult,
        entities: List[str]
    ) -> float:
        """Boost results containing extracted entities."""
        doc_lower = result.document_text.lower()

        for entity in entities:
            if entity.lower() in doc_lower:
                return self.rerank_config.entity_match_boost

        return 1.0

    def _apply_recency_boost(self, result: SearchResult) -> float:
        """Boost more recent results based on timestamp."""
        timestamp = result.metadata.get("timestamp")

        if not timestamp:
            return 1.0

        try:
            # Parse timestamp if it's a string
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)

            # Calculate age in days
            age_days = (datetime.now() - timestamp).days

            # Decay factor: boost decays over time
            if age_days < 7:
                return self.rerank_config.recent_boost_factor
            elif age_days < 30:
                return 1.05
            else:
                return 1.0
        except Exception as e:
            self.logger.debug(f"Error applying recency boost: {e}")
            return 1.0

    def _filter_by_threshold(
        self,
        results_with_scores: List[Tuple[SearchResult, float]]
    ) -> List[Tuple[SearchResult, float]]:
        """Filter results by minimum score threshold."""
        threshold = self.rerank_config.min_score_threshold

        if threshold <= 0.0:
            return results_with_scores

        filtered = [
            (result, score)
            for result, score in results_with_scores
            if score >= threshold
        ]

        num_filtered = len(results_with_scores) - len(filtered)
        if num_filtered > 0:
            self.logger.info(f"Filtered out {num_filtered} results below threshold {threshold}")

        return filtered

    def _update_results_with_scores(
        self,
        results_with_scores: List[Tuple[SearchResult, float]]
    ) -> List[SearchResult]:
        """Update SearchResult objects with new reranked scores."""
        updated = []

        for result, new_score in results_with_scores:
            # Create new SearchResult with updated score
            updated_result = SearchResult(
                node_id=result.node_id,
                similarity_score=new_score,
                document_text=result.document_text,
                node_type=result.node_type,
                chat_name=result.chat_name,
                metadata=result.metadata,
                search_method=f"{result.search_method}_reranked"
            )
            updated.append(updated_result)

        return updated

    def _rebuild_vector_search_results(
        self,
        updated_results: List[SearchResult],
        original_data: VectorSearchResults
    ) -> VectorSearchResults:
        """Rebuild VectorSearchResults with reranked results and updated ID sets."""
        # Rebuild entity/claim/fulltext ID sets based on remaining results
        entity_ids = set()
        person_ids = set()
        claim_ids = set()
        full_text_ids = set()

        for result in updated_results:
            if result.node_type == "entity":
                entity_ids.add(result.node_id)
            elif result.node_type == "person":
                person_ids.add(result.node_id)
            elif result.node_type == "claim":
                claim_ids.add(result.node_id)
            elif result.node_type == "full_text":
                full_text_ids.add(result.node_id)

        return VectorSearchResults(
            entity_ids=entity_ids,
            person_ids=person_ids,
            claim_ids=claim_ids,
            full_text_ids=full_text_ids,
            results=updated_results,
            original_query=original_data.original_query,
            extracted_entities=original_data.extracted_entities,
            extracted_keywords=original_data.extracted_keywords,
            extracted_variations=original_data.extracted_variations
        )

    def _log_step_result(self, result: VectorSearchResults):
        """Log summary of reranking results."""
        self.logger.info(f"Reranking summary:")
        self.logger.info(f"  Final results: {len(result.results)}")
        self.logger.info(f"  Entity IDs: {len(result.entity_ids)}")
        self.logger.info(f"  Person IDs: {len(result.person_ids)}")
        self.logger.info(f"  Claim IDs: {len(result.claim_ids)}")
        self.logger.info(f"  Full-text IDs: {len(result.full_text_ids)}")

        if self.logger.isEnabledFor(logging.DEBUG):
            for i, search_result in enumerate(result.results[:5]):
                self.logger.debug(
                    f"Result {i+1}: {search_result.search_method} "
                    f"'{search_result.node_id}' (score: {search_result.similarity_score:.3f})"
                )
