"""
Step 3: Query Expansion and Neo4j Execution

Executes Cypher queries against Neo4j database and builds comprehensive graph context.
Provides detailed logging of query execution and results.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import BaseQueryStep


def _import_cypher_module():
    """Import the Cypher generation module to get CypherQuerySet."""
    import importlib.util
    import sys
    from pathlib import Path

    current_dir = Path(__file__).parent
    step_path = current_dir / "c_cypher_generation.py"

    project_root = str(current_dir.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        spec = importlib.util.spec_from_file_location("cypher_generation", step_path)
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "query.steps"
        spec.loader.exec_module(module)
        return module
    finally:
        if project_root in sys.path:
            sys.path.remove(project_root)


# Import CypherQuerySet from step 2
_cypher_module = _import_cypher_module()
CypherQuerySet = _cypher_module.CypherQuerySet


@dataclass
class GraphContext:
    """Container for graph query results and metadata."""

    entities: Dict[int, Dict[str, Any]]
    people: Dict[int, Dict[str, Any]]
    claims: Dict[int, Dict[str, Any]]
    said_relationships: List[Dict[str, Any]]
    mention_relationships: List[Dict[str, Any]]
    reaction_relationships: List[Dict[str, Any]]
    paths: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class QueryExpander(BaseQueryStep):
    """Executes Cypher queries and builds comprehensive graph context."""

    def setup(self):
        """Setup query expander configuration."""
        self.max_results_per_query = self.config.get("max_results_per_query", 50)
        self.log_detailed_results = self.config.get("log_detailed_results", True)
        self.max_logged_records = self.config.get("max_logged_records", 5)

    def process(self, cypher_queries: CypherQuerySet) -> GraphContext:
        """
        Process Cypher queries and return graph context.

        Args:
            cypher_queries: CypherQuerySet from Step 2

        Returns:
            GraphContext with query results from Neo4j
        """
        start_time = time.time()
        self.logger.info("Starting query expansion")

        try:
            # This will be set by the pipeline if Neo4j is available
            neo4j_driver = getattr(self, "_neo4j_driver", None)

            if neo4j_driver:
                graph_context = self._execute_against_neo4j(
                    cypher_queries, neo4j_driver
                )
            else:
                self.logger.error("No Neo4j driver available - cannot proceed")
                raise RuntimeError("Neo4j driver is required for query expansion")

            processing_time = time.time() - start_time
            self.logger.info(f"Query expansion completed in {processing_time:.2f}s")

            return graph_context

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(
                f"Query expansion failed after {processing_time:.2f}s: {str(e)}"
            )
            # Re-raise the error - no fallback behavior
            raise

    def _execute_against_neo4j(
        self, cypher_queries: CypherQuerySet, neo4j_driver
    ) -> GraphContext:
        """Execute Cypher queries against Neo4j and log results."""
        start_time = time.time()
        self.logger.info("=== EXECUTING CYPHER QUERIES AGAINST NEO4J ===")

        # Initialize result collections
        entities = {}
        people = {}
        claims = {}
        said_relationships = []
        mention_relationships = []
        reaction_relationships = []
        paths = []

        total_queries = 0
        successful_queries = 0
        failed_queries = 0
        total_records = 0

        try:
            with neo4j_driver.session() as session:
                # Execute entity queries
                for i, query in enumerate(cypher_queries.entity_queries, 1):
                    total_queries += 1
                    records = self._execute_single_query(
                        session, query, f"Entity Query {i}"
                    )
                    if records is not None:
                        successful_queries += 1
                        total_records += len(records)
                        self._process_query_results(
                            records,
                            entities,
                            people,
                            claims,
                            said_relationships,
                            mention_relationships,
                            reaction_relationships,
                            paths,
                        )
                    else:
                        failed_queries += 1

                # Execute claim queries
                for i, query in enumerate(cypher_queries.claim_queries, 1):
                    total_queries += 1
                    records = self._execute_single_query(
                        session, query, f"Claim Query {i}"
                    )
                    if records is not None:
                        successful_queries += 1
                        total_records += len(records)
                        self._process_query_results(
                            records,
                            entities,
                            people,
                            claims,
                            said_relationships,
                            mention_relationships,
                            reaction_relationships,
                            paths,
                        )
                    else:
                        failed_queries += 1

                # Execute person queries
                for i, query in enumerate(cypher_queries.person_queries, 1):
                    total_queries += 1
                    records = self._execute_single_query(
                        session, query, f"Person Query {i}"
                    )
                    if records is not None:
                        successful_queries += 1
                        total_records += len(records)
                        self._process_query_results(
                            records,
                            entities,
                            people,
                            claims,
                            said_relationships,
                            mention_relationships,
                            reaction_relationships,
                            paths,
                        )
                    else:
                        failed_queries += 1

            execution_time = time.time() - start_time

            # Log execution summary
            self.logger.info("=== NEO4J EXECUTION SUMMARY ===")
            self.logger.info(f"Total queries executed: {total_queries}")
            self.logger.info(f"Successful queries: {successful_queries}")
            self.logger.info(f"Failed queries: {failed_queries}")
            self.logger.info(f"Total records retrieved: {total_records}")
            self.logger.info(f"Execution time: {execution_time:.2f}s")
            self.logger.info(f"Unique entities found: {len(entities)}")
            self.logger.info(f"Unique people found: {len(people)}")
            self.logger.info(f"Unique claims found: {len(claims)}")
            self.logger.info(
                f"Total relationships: {len(said_relationships) + len(mention_relationships) + len(reaction_relationships)}"
            )

            # Create GraphContext with real results
            return GraphContext(
                entities=entities,
                people=people,
                claims=claims,
                said_relationships=said_relationships,
                mention_relationships=mention_relationships,
                reaction_relationships=reaction_relationships,
                paths=paths,
                metadata={
                    "total_entities": len(entities),
                    "total_people": len(people),
                    "total_claims": len(claims),
                    "total_said_relationships": len(said_relationships),
                    "total_mention_relationships": len(mention_relationships),
                    "total_reaction_relationships": len(reaction_relationships),
                    "total_queries": total_queries,
                    "successful_queries": successful_queries,
                    "failed_queries": failed_queries,
                    "total_records": total_records,
                    "query_execution_time": execution_time,
                    "data_source": "neo4j",
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(
                f"Neo4j session failed after {execution_time:.2f}s: {str(e)}"
            )
            raise

    def _execute_single_query(self, session, query, query_name: str) -> Optional[List]:
        """Execute a single Cypher query with error handling and logging."""
        try:
            self.logger.info(f"Executing {query_name}: {query.description}")
            self.logger.debug(f"Cypher: {query.query}")
            self.logger.debug(f"Parameters: {query.parameters}")

            result = session.run(query.query, query.parameters)
            records = list(result)

            self.logger.info(f"✅ {query_name} returned {len(records)} records")

            if self.log_detailed_results:
                self._log_query_results(query_name, records)

            return records

        except Exception as e:
            self.logger.error(f"❌ {query_name} failed: {str(e)}")
            return None

    def _log_query_results(self, query_name: str, records: List):
        """Log detailed results of a Cypher query."""
        if not records:
            self.logger.info(f"{query_name} - No results returned")
            return


    def _process_query_results(
        self,
        records: List,
        entities: Dict,
        people: Dict,
        claims: Dict,
        said_relationships: List,
        mention_relationships: List,
        reaction_relationships: List,
        paths: List,
    ):
        """Process and organize query results into appropriate collections."""
        for record in records:
            for record_key, value in record.items():
                if hasattr(value, "labels") and hasattr(value, "id"):  # Node
                    node_labels = list(value.labels)
                    node_props = dict(value)
                    node_id = value.id

                    if "Entity" in node_labels:
                        if node_id not in entities:
                            entities[node_id] = {
                                "labels": node_labels,
                                "properties": node_props,
                            }
                    elif "Person" in node_labels:
                        if node_id not in people:
                            people[node_id] = {
                                "labels": node_labels,
                                "properties": node_props,
                            }
                    elif "Claim" in node_labels:
                        if node_id not in claims:
                            claims[node_id] = {
                                "labels": node_labels,
                                "properties": node_props,
                            }

                elif hasattr(value, "type") and hasattr(
                    value, "start_node"
                ):  # Relationship
                    rel_type = value.type
                    rel_props = dict(value)
                    rel_data = {
                        "type": rel_type,
                        "properties": rel_props,
                        "start_node": value.start_node.id,
                        "end_node": value.end_node.id,
                    }

                    if rel_type == "SAID":
                        if not self._relationship_exists(said_relationships, rel_data):
                            said_relationships.append(rel_data)
                    elif rel_type == "MENTION":
                        if not self._relationship_exists(mention_relationships, rel_data):
                            mention_relationships.append(rel_data)
                    elif rel_type == "REACTED":
                        if not self._relationship_exists(reaction_relationships, rel_data):
                            reaction_relationships.append(rel_data)

                elif hasattr(value, "start_node") and hasattr(value, "end_node") and hasattr(value, "nodes") and hasattr(value, "relationships"):  # Path
                    # Process all nodes in the path
                    for node in value.nodes:
                        node_labels = list(node.labels)
                        node_props = dict(node)
                        node_id = node.id

                        if "Entity" in node_labels and node_id not in entities:
                            entities[node_id] = {
                                "labels": node_labels,
                                "properties": node_props,
                            }
                        elif "Person" in node_labels and node_id not in people:
                            people[node_id] = {
                                "labels": node_labels,
                                "properties": node_props,
                            }
                        elif "Claim" in node_labels and node_id not in claims:
                            claims[node_id] = {
                                "labels": node_labels,
                                "properties": node_props,
                            }

                    # Process all relationships in the path
                    for relationship in value.relationships:
                        rel_type = relationship.type
                        rel_props = dict(relationship)
                        rel_data = {
                            "type": rel_type,
                            "properties": rel_props,
                            "start_node": relationship.start_node.id,
                            "end_node": relationship.end_node.id,
                        }

                        # Avoid duplicates by checking if this exact relationship already exists
                        if rel_type == "SAID":
                            if not self._relationship_exists(said_relationships, rel_data):
                                said_relationships.append(rel_data)
                        elif rel_type == "MENTION":
                            if not self._relationship_exists(mention_relationships, rel_data):
                                mention_relationships.append(rel_data)
                        elif rel_type == "REACTED":
                            if not self._relationship_exists(reaction_relationships, rel_data):
                                reaction_relationships.append(rel_data)

                    # Store the path information for potential future use
                    path_data = {
                        "start_node": value.start_node.id,
                        "end_node": value.end_node.id,
                        "length": len(value.relationships),
                        "nodes": [node.id for node in value.nodes],
                        "relationships": [{"type": rel.type, "start": rel.start_node.id, "end": rel.end_node.id} for rel in value.relationships]
                    }

                    # Avoid duplicate paths
                    if not self._path_exists(paths, path_data):
                        paths.append(path_data)

    def _relationship_exists(self, relationships: List[Dict], rel_data: Dict) -> bool:
        """Check if a relationship already exists in the collection."""
        return any(
            r["start_node"] == rel_data["start_node"]
            and r["end_node"] == rel_data["end_node"]
            and r["type"] == rel_data["type"]
            for r in relationships
        )

    def _path_exists(self, paths: List[Dict], path_data: Dict) -> bool:
        """Check if a path already exists in the collection."""
        return any(
            p["start_node"] == path_data["start_node"]
            and p["end_node"] == path_data["end_node"]
            and p["relationships"] == path_data["relationships"]
            for p in paths
        )



    def set_neo4j_driver(self, driver):
        """Set the Neo4j driver for this expander instance."""
        self._neo4j_driver = driver

    def _log_step_result(self, result: GraphContext):
        """Log summary of graph context results."""
        metadata = result.metadata
        self.logger.info(
            f"Graph context built: {metadata.get('data_source', 'unknown')} source"
        )
        self.logger.info(f"Total entities: {metadata.get('total_entities', 0)}")
        self.logger.info(f"Total people: {metadata.get('total_people', 0)}")
        self.logger.info(f"Total claims: {metadata.get('total_claims', 0)}")
        self.logger.info(
            f"Total relationships: {metadata.get('total_said_relationships', 0) + metadata.get('total_mention_relationships', 0) + metadata.get('total_reaction_relationships', 0)}"
        )

    def close(self):
        """Close any open connections or resources."""
        # In a real implementation, this would close Neo4j connections
        pass



