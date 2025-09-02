"""
Graph Searcher

Neo4j-based graph search for finding people and relationships.
"""

import logging
import os
from typing import List

from .base_searcher import BaseSearcher, SearchResult


class GraphSearcher(BaseSearcher):
    """Graph-based search for finding people using Neo4j."""
    
    def setup(self):
        """Setup Neo4j connection."""
        try:
            from neo4j import GraphDatabase
            
            neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            neo4j_user = os.getenv("NEO4J_USER", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
            
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            
            # Test connection
            with self.driver.session() as session:
                session.run("MATCH (n) RETURN count(n) LIMIT 1")
            
            logging.info("Neo4j connection established")
            
        except Exception as e:
            logging.warning(f"Failed to connect to Neo4j: {e}")
            self.driver = None
    
    def search(self, query: str, **kwargs) -> List[SearchResult]:
        """
        Search for people in the graph database.
        
        Args:
            query: Search query string
            **kwargs: Additional parameters (query_terms as List[str])
            
        Returns:
            List of SearchResult objects for people found
        """
        # Support both direct query and query_terms list
        query_terms = kwargs.get('query_terms', [query]) if kwargs.get('query_terms') else [query]
        return self.search_people(query_terms)
    
    def search_people(self, query_terms: List[str]) -> List[SearchResult]:
        """Search for people in the graph database."""
        if not self.driver:
            return []
        
        results = []
        
        try:
            with self.driver.session() as session:
                # Search for people by ID or name containing query terms
                for term in query_terms:
                    cypher_query = """
                    MATCH (p:Person)
                    WHERE toLower(p.id) CONTAINS toLower($term) 
                       OR toLower(p.name) CONTAINS toLower($term)
                    RETURN p.id as id, p.name as name, p.description as description
                    LIMIT 10
                    """
                    
                    result = session.run(cypher_query, term=term)
                    
                    for record in result:
                        person_id = record.get("id", "")
                        person_name = record.get("name", "")
                        person_description = record.get("description", "")
                        
                        # Create document text from available fields
                        doc_parts = []
                        if person_name:
                            doc_parts.append(f"Name: {person_name}")
                        if person_description:
                            doc_parts.append(f"Description: {person_description}")
                        
                        document_text = " | ".join(doc_parts) if doc_parts else person_id
                        
                        # Calculate simple relevance score based on term matching
                        score = 0.8  # Base score for graph matches
                        if term.lower() in person_name.lower():
                            score += 0.2  # Bonus for name match
                        if term.lower() in person_id.lower():
                            score += 0.1  # Bonus for ID match
                        
                        search_result = SearchResult(
                            node_id=person_id,
                            similarity_score=score,
                            document_text=document_text,
                            node_type="person",
                            chat_name="graph",
                            metadata={
                                "name": person_name,
                                "description": person_description,
                                "source": "neo4j_graph"
                            },
                            search_method="graph"
                        )
                        
                        results.append(search_result)
        
        except Exception as e:
            logging.error(f"Error in graph search: {e}")
        
        # Remove duplicates and sort by score
        seen_ids = set()
        unique_results = []
        for result in results:
            if result.node_id not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result.node_id)
        
        unique_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return unique_results[:10]  # Limit to top 10 results
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
