"""
Step 2: Cypher Query Generation

Generates Neo4j Cypher queries based on vector search results.
Uses specific entity and claim IDs found through vector search.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from .base import BaseQueryStep


def _import_hybrid_search_module():
    """Import the hybrid search module to get VectorSearchResults."""
    import importlib.util
    import sys
    from pathlib import Path

    current_dir = Path(__file__).parent
    step_path = current_dir / "a_hybrid_search.py"

    project_root = str(current_dir.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        spec = importlib.util.spec_from_file_location("hybrid_search", step_path)
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "query.steps"
        spec.loader.exec_module(module)
        return module
    finally:
        if project_root in sys.path:
            sys.path.remove(project_root)


# Import VectorSearchResults from step 1 (compatibility wrapper in hybrid search)
_hybrid_search_module = _import_hybrid_search_module()
VectorSearchResults = _hybrid_search_module.VectorSearchResults


@dataclass
class CypherQuery:
    """Container for a generated Cypher query."""

    query: str
    parameters: Dict[str, Any]
    query_type: str
    description: str


@dataclass
class CypherQuerySet:
    """Container for all generated Cypher queries."""

    entity_queries: List[CypherQuery]
    claim_queries: List[CypherQuery]
    person_queries: List[CypherQuery]


class CypherGenerator(BaseQueryStep):
    """Generates Cypher queries for entity and claim search patterns.
    """

    def setup(self):
        """Setup Cypher query templates."""
        self.templates = {
            "entity_by_id": """
                MATCH (e:Entity)
                WHERE e.id CONTAINS $entity_id
                OPTIONAL MATCH path = (p:Person)-[r:SAID|REACTED]->(c)
                OPTIONAL MATCH mention = (c)-[:MENTION]->(e)
                RETURN c, path, mention, e, p
                ORDER BY c.valid_at DESC
                """,
            "claim_by_id": """
                MATCH (c:Claim)
                WHERE c.id CONTAINS $claim_id
                OPTIONAL MATCH path = (p:Person)-[r:SAID|REACTED]->(c)
                OPTIONAL MATCH mention = (c)-[:MENTION]->(e:Entity)
                RETURN c, path, mention, e, p
                ORDER BY c.valid_at DESC
                """,
            "full_text_by_id": """
                MATCH (c:Claim)
                WHERE c.id CONTAINS $claim_id
                OPTIONAL MATCH path = (p:Person)-[r:SAID|REACTED]->(c)
                OPTIONAL MATCH mention = (c)-[:MENTION]->(e:Entity)
                RETURN c, path, mention, e, p
                ORDER BY c.valid_at DESC
            """,
            "person_by_short_name_and_search_text": """
                MATCH (p:Person)
                WHERE p.id CONTAINS $person_id
                OPTIONAL MATCH path = (p)-[r:SAID|REACTED]->(c:Claim)
                WHERE ANY(txt IN r.full_text WHERE ANY(search_term IN $search_text WHERE toLower(txt) CONTAINS toLower(search_term)))
                OR ANY(search_term IN $search_text WHERE toLower(c.summary_text) CONTAINS toLower(search_term))
                OPTIONAL MATCH mention = (c)-[:MENTION]->(e)
                RETURN c, path, mention, e
                ORDER BY c.valid_at DESC
                """,
            "claims_containing_search_terms": """
                MATCH (c:Claim)
                WHERE ANY(search_term IN $search_text WHERE toLower(c.summary_text) CONTAINS toLower(search_term))
                OPTIONAL MATCH path = (p:Person)-[r:SAID|REACTED]->(c)
                OPTIONAL MATCH mention = (c)-[:MENTION]->(e)
                RETURN c, path, mention, e, p
                ORDER BY c.valid_at DESC
                """,
        }

    def process(self, vector_search_results: VectorSearchResults) -> CypherQuerySet:
        """
        Generate Cypher queries based on vector search results.

        Args:
            vector_search_results: Results from vector search step

        Returns:
            CypherQuerySet containing entity and claim queries
        """
        self.logger.info(
            f"Generating Cypher queries for: {vector_search_results.original_query}"
        )

        self.logger.info(
            f"Using vector search results: {len(vector_search_results.entity_ids)} entities, "
            f"{len(vector_search_results.claim_ids)} claims, "
            f"{len(vector_search_results.person_ids)} people, "
            f"{len(vector_search_results.full_text_ids)} full-text matches"
        )

        entity_queries = self._generate_entity_queries(vector_search_results)
        claim_queries = self._generate_claim_queries(vector_search_results)
        person_queries = self._generate_person_queries(vector_search_results)

        query_set = CypherQuerySet(
            entity_queries=entity_queries,
            claim_queries=claim_queries,
            person_queries=person_queries,
        )

        total_queries = len(entity_queries) + len(claim_queries) + len(person_queries)
        self.logger.info(f"Generated {total_queries} queries total")

        # Log all generated queries
        self._log_generated_queries(query_set)

        return query_set

    def _generate_entity_queries(
        self, vector_search_results: VectorSearchResults
    ) -> List[CypherQuery]:
        """Generate entity search queries using vector search results."""
        queries = []

        # Use vector search results for specific entity IDs
        for entity_id in vector_search_results.entity_ids:
            query = CypherQuery(
                query=self.templates["entity_by_id"].strip(),
                parameters={"entity_id": entity_id},
                query_type="entity_by_id",
                description=f"Find entity by ID: {entity_id}",
            )
            queries.append(query)

        self.logger.info(
            f"Generated {len(queries)} entity ID queries from vector search"
        )

        return queries

    def _generate_claim_queries(
        self, vector_search_results: VectorSearchResults
    ) -> List[CypherQuery]:
        """Generate claim search queries using vector search results."""
        queries = []

        # Use claim IDs directly
        for claim_id in vector_search_results.claim_ids:
            query = CypherQuery(
                query=self.templates["claim_by_id"].strip(),
                parameters={"claim_id": claim_id},
                query_type="claim_by_id",
                description=f"Find claim by ID: {claim_id}",
            )
            queries.append(query)

        # Use full-text IDs as potential claim IDs
        for full_text_id in vector_search_results.full_text_ids:
            query = CypherQuery(
                query=self.templates["full_text_by_id"].strip(),
                parameters={"claim_id": full_text_id},
                query_type="full_text_by_id",
                description=f"Find claim by full-text ID: {full_text_id}",
            )
            queries.append(query)

        self.logger.info(
            f"Generated {len(queries)} claim ID queries from vector search"
        )

        return queries


    def _generate_person_queries(
        self, vector_search_results: VectorSearchResults
    ) -> List[CypherQuery]:
        """Generate person search queries using vector search results."""
        queries = []

        # Generate person-specific queries for found people
        for person_id in vector_search_results.person_ids:
            query = CypherQuery(
                query=self.templates["person_by_short_name_and_search_text"].strip(),
                parameters={"person_id": person_id, "search_text": vector_search_results.extracted_entities},
                query_type="person_by_short_name_and_search_text",
                description=f"Find person by ID: {person_id}",
            )
            queries.append(query)

        # Also add a general query to find claims containing the search terms
        if vector_search_results.extracted_entities:
            query = CypherQuery(
                query=self.templates["claims_containing_search_terms"].strip(),
                parameters={"search_text": vector_search_results.extracted_entities},
                query_type="claims_containing_search_terms",
                description="Find claims containing search terms",
            )
            queries.append(query)

        self.logger.info(
            f"Generated {len(queries)} person and claim queries from vector search"
        )

        return queries
    
    def _log_generated_queries(self, query_set: CypherQuerySet):
        """Log all generated Cypher queries with full details."""
        if query_set.entity_queries:
            self.logger.info("=== Generated Entity Queries ===")
            for i, query in enumerate(query_set.entity_queries, 1):
                self.logger.info(f"Entity Query {i}: {query.description}")
                self.logger.info(f"Parameters: {query.parameters}")
                self.logger.info(f"Cypher:\n{query.query}")
                self.logger.info("-" * 50)

        if query_set.claim_queries:
            self.logger.info("=== Generated Claim Queries ===")
            for i, query in enumerate(query_set.claim_queries, 1):
                self.logger.info(f"Claim Query {i}: {query.description}")
                self.logger.info(f"Parameters: {query.parameters}")
                self.logger.info(f"Cypher:\n{query.query}")
                self.logger.info("-" * 50)
        if query_set.person_queries:
            self.logger.info("=== Generated Person Queries ===")
            for i, query in enumerate(query_set.person_queries, 1):
                self.logger.info(f"Person Query {i}: {query.description}")
                self.logger.info(f"Parameters: {query.parameters}")
                self.logger.info(f"Cypher:\n{query.query}")
                self.logger.info("-" * 50)


    def _log_step_result(self, result: CypherQuerySet):
        """Log summary of generated queries."""
        self.logger.debug(f"Entity queries: {len(result.entity_queries)}")
        self.logger.debug(f"Claim queries: {len(result.claim_queries)}")

        # Log sample query descriptions for debugging
        if self.logger.isEnabledFor(logging.DEBUG):
            for i, query in enumerate(result.entity_queries[:2]):
                self.logger.debug(f"Entity query {i+1}: {query.description}")
            for i, query in enumerate(result.claim_queries[:2]):
                self.logger.debug(f"Claim query {i+1}: {query.description}")
