"""
Step 1: Hybrid Search (Enhanced)

Combines vector similarity search with full-text search for improved retrieval.
Uses both semantic understanding and keyword matching to find relevant content.

Architecture:
1. Vector Search - Semantic similarity using embeddings
2. Full-Text Search - Keyword matching with BM25 scoring
3. Graph Search - Quick person lookup using Neo4j graph database
4. Result Fusion - Combines and ranks results from all methods
"""

import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import chromadb
import openai
import requests
from dotenv import load_dotenv

from .base import BaseQueryStep
from .searchers import BaseSearcher, SearchResult, VectorSearcher, FullTextSearcher, GraphSearcher

load_dotenv()


# Import SearchResult from searchers module
# Alias for backward compatibility
VectorSearchResult = SearchResult


@dataclass
class HybridSearchResults:
    """Container for all hybrid search results."""
    
    entity_ids: Set[str]
    claim_ids: Set[str]
    full_text_ids: Set[str]
    results: List[SearchResult]
    original_query: str
    vector_results_count: int
    fulltext_results_count: int
    graph_results_count: int


# For compatibility with existing VectorSearchResults interface
@dataclass
class VectorSearchResults:
    """Compatibility wrapper for VectorSearchResults interface."""
    
    entity_ids: Set[str]
    person_ids: Set[str]
    claim_ids: Set[str]
    full_text_ids: Set[str]
    results: List[SearchResult]
    original_query: str
    extracted_entities: List[str]
    extracted_keywords: List[str]
    extracted_variations: List[str]








class HybridSearcher(BaseQueryStep):
    """Enhanced search combining vector similarity, full-text search, and graph search."""
    
    def setup(self):
        """Setup hybrid search configuration."""
        # Search configuration
        self.vector_search_top_k = int(os.getenv("VECTOR_SEARCH_TOP_K", "30"))
        self.fulltext_search_top_k = int(os.getenv("FULLTEXT_SEARCH_TOP_K", "20"))
        
        # Feature toggles
        self.enable_fulltext_search = self.config.get("enable_fulltext_search", True)
        self.enable_vector_search = self.config.get("enable_vector_search", True)
        self.enable_graph_search = self.config.get("enable_graph_search", True)
        
        # Hybrid search weights
        self.vector_weight = float(os.getenv("VECTOR_WEIGHT", "0.5"))
        self.fulltext_weight = float(os.getenv("FULLTEXT_WEIGHT", "0.3"))
        self.graph_weight = float(os.getenv("GRAPH_WEIGHT", "0.2"))
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="./vector_store")
        
        # Initialize searchers
        self._initialize_searchers()
    
    def _initialize_searchers(self):
        """Initialize the individual searcher instances."""
        searcher_config = {
            'chroma_client': self.chroma_client,
            'top_k': self.vector_search_top_k
        }
        
        # Initialize vector searcher only if enabled
        if self.enable_vector_search:
            self.vector_searcher = VectorSearcher(searcher_config)
            self.logger.info("Vector search enabled")
        else:
            self.vector_searcher = None
            self.logger.info("Vector search disabled")
        
        # Initialize full-text searcher only if enabled
        if self.enable_fulltext_search:
            self.fulltext_searcher = FullTextSearcher(searcher_config)
            self.logger.info("Full-text search enabled")
        else:
            self.fulltext_searcher = None
            self.logger.info("Full-text search disabled")
        
        # Initialize graph searcher only if enabled
        if self.enable_graph_search:
            self.graph_searcher = GraphSearcher(searcher_config)
            self.logger.info("Graph search enabled")
        else:
            self.graph_searcher = None
            self.logger.info("Graph search disabled")
    

    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms using TF-IDF scoring."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Preprocess text
            cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
            words = cleaned_text.split()
            
            if len(words) < 2:
                return words
            
            documents = [text.lower(), ' '.join(words)]
            
            vectorizer = TfidfVectorizer(
                max_features=5,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                token_pattern=r'\b[a-zA-Z]{3,}\b'
            )
            
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix[0].toarray()[0]
            
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [keyword for keyword, score in keyword_scores if score > 0][:5]
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}")
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            return list(set(words))[:5]

    def _extract_potential_entities(self, text: str) -> List[str]:
        """Extract potential entities using langextract for better entity recognition."""
        try:
            import langextract
        except ImportError:
            self.logger.warning("langextract not available, skipping entity extraction")
            return []
        
        # Use langextract to detect and extract entities
        entities = []
        
        # Check if we have a Gemini API key for langextract
        import os
        api_key = os.getenv("LANGEXTRACT_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if api_key:
            try:
                # Create examples for langextract
                examples = [
                    langextract.data.ExampleData(
                        text="Who did John Smith work with at Microsoft on the Azure project?",
                        extractions=[
                            langextract.data.Extraction(
                                extraction_class="PEOPLE",
                                extraction_text="John Smith",
                            ),
                            langextract.data.Extraction(
                                extraction_class="ORGANIZATION",
                                extraction_text="Microsoft",
                            ),
                            langextract.data.Extraction(
                                extraction_class="PROJECT",
                                extraction_text="Azure",
                            ),
                            langextract.data.Extraction(
                                extraction_class="TOPIC",
                                extraction_text="climbing",
                            ),

                        ],
                    ),
                    langextract.data.ExampleData(
                        text="What did Sarah Johnson say about the meeting in New York with OpenAI?",
                        extractions=[
                            langextract.data.Extraction(
                                extraction_class="PEOPLE",
                                extraction_text="Sarah Johnson",
                                attributes={"type": "person", "action": "say"},
                            ),
                            langextract.data.Extraction(
                                extraction_class="LOCATION",
                                extraction_text="New York",
                                attributes={"type": "city", "context": "meeting"},
                            ),
                            langextract.data.Extraction(
                                extraction_class="ORGANIZATION",
                                extraction_text="OpenAI",
                                attributes={"type": "company", "context": "meeting"},
                            ),
                        ],
                    ),
                    langextract.data.ExampleData(
                        text="Tell me about the DFX hack and who was affected",
                        extractions=[
                            langextract.data.Extraction(
                                extraction_class="ORGANIZATION",
                                extraction_text="DFX",
                                attributes={"type": "platform", "event": "hack"},
                            ),
                        ],
                    ),
                ]
                
                result = langextract.extract(
                    text,
                    prompt_description="Extract all person names, organization names, place names, and project names from the text. Focus on entities that could be found in a chat knowledge base.",
                    examples=examples,
                    api_key=api_key,
                    model_id="gemini-2.5-flash",
                    temperature=0.1,
                    debug=False
                )

                # Extract entities from the result
                if hasattr(result, 'extractions'):
                    for extraction in result.extractions:
                        if hasattr(extraction, 'extraction_text') and extraction.extraction_text:
                            entity_clean = str(extraction.extraction_text).strip()
                            entity_type = getattr(extraction, 'extraction_class', 'UNKNOWN')
                            if len(entity_clean) > 2:
                                # Store entity with its type as a tuple
                                entities.append((entity_clean, entity_type))
                                
            except Exception as e:
                self.logger.error(f"Error in langextract entity extraction: {e}")
        else:
            self.logger.debug("No API key found for langextract, skipping LLM-based entity extraction")

        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity_text, entity_type in entities:
            if entity_text.lower() not in seen:
                seen.add(entity_text.lower())
                unique_entities.append((entity_text, entity_type))

        self.logger.debug(f"Extracted entities: {[f'{text} ({type_})' for text, type_ in unique_entities]}")
        return unique_entities[:10]  # Limit to top 10 entities

    def _enhanced_fulltext_search(self, query: str, keywords: List[str], entities: List[str]) -> List[SearchResult]:
        """Perform enhanced full-text search using query, keywords, and entities."""
        # Return empty results if fulltext search is disabled
        if not self.fulltext_searcher:
            return []
            
        all_results = []
        seen_ids = set()
        
        # Search with original query
        original_results = self.fulltext_searcher.search(query, top_k=self.fulltext_search_top_k // 3)
        for result in original_results:
            if result.node_id not in seen_ids:
                all_results.append(result)
                seen_ids.add(result.node_id)
        
        # Search with extracted entities for better recall
        for entity in entities[:3]:  # Limit to top 3 entities to avoid too many searches
            entity_results = self.fulltext_searcher.search(entity, top_k=max(5, self.fulltext_search_top_k // 6))
            for result in entity_results:
                if result.node_id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result.node_id)
        
        # Search with high-value keywords
        for keyword in keywords[:2]:  # Limit to top 2 keywords
            keyword_results = self.fulltext_searcher.search(keyword, top_k=max(3, self.fulltext_search_top_k // 8))
            for result in keyword_results:
                if result.node_id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result.node_id)
        
        # Sort by enhanced similarity score and return top results
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return all_results[:self.fulltext_search_top_k]

    def _enhanced_vector_search(self, query: str, entities: List[str]) -> List[SearchResult]:
        """Perform enhanced vector search using query and extracted entities."""
        if not self.vector_searcher:
            return []
            
        all_results = []
        seen_ids = set()
        
        # Search with original query find nodes that are similar to the query
        original_results = self.vector_searcher.search(query)
        for result in original_results:
            if result.node_id not in seen_ids:
                all_results.append(result)
                seen_ids.add(result.node_id)
        
        # Search with extracted entities for additional context
        # Filter entities into people and non-person entities based on type
        non_person_entities = []
        
        for entity in entities:
            # Check if entity is a tuple with type information
            if isinstance(entity, tuple):
                entity_text, entity_type = entity
                if entity_type != 'PEOPLE':
                    non_person_entities.append(entity_text)
            else:
                non_person_entities.append(entity)
        
        # Search with non-person entities for additional context
        for entity in non_person_entities:
            entity_results = self.vector_searcher.search(entity, collection="entity")
            for result in entity_results:
                if result.node_id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result.node_id)

        # Sort by enhanced similarity score and return top results
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return all_results[:self.vector_search_top_k]

    def _enhanced_graph_search(self, query: str, keywords: List[str], entities: List[str]) -> List[SearchResult]:
        """Perform graph-based search for people using query terms."""
        if not self.graph_searcher:
            return []
        
        # Combine query terms, keywords, and person entities for graph search
        search_terms = [query]
        search_terms.extend(keywords)
        
        # Add person entities
        person_entities = []
        for entity in entities:
            if isinstance(entity, tuple):
                entity_text, entity_type = entity
                if entity_type == 'PEOPLE':
                    person_entities.append(entity_text)
            else:
                person_entities.append(entity)
        
        search_terms.extend(person_entities)
        
        # Remove duplicates while preserving order
        unique_terms = []
        seen = set()
        for term in search_terms:
            if term.lower() not in seen:
                unique_terms.append(term)
                seen.add(term.lower())
        
        return self.graph_searcher.search(query, query_terms=unique_terms)
   

    
    def _fuse_results(self, vector_results: List[SearchResult], fulltext_results: List[SearchResult], graph_results: List[SearchResult]) -> List[SearchResult]:
        """Combine and rank results from vector, full-text, and graph search."""
        # Create a map of results by node_id
        combined_results = {}
        
        # Add vector results
        for result in vector_results:
            node_id = result.node_id
            combined_results[node_id] = SearchResult(
                node_id=node_id,
                similarity_score=result.similarity_score * self.vector_weight,
                document_text=result.document_text,
                node_type=result.node_type,
                chat_name=result.chat_name,
                metadata=result.metadata,
                search_method="vector"
            )
        
        # Add or combine full-text results
        for result in fulltext_results:
            node_id = result.node_id
            
            if node_id in combined_results:
                # Combine scores if found in both
                existing = combined_results[node_id]
                combined_score = existing.similarity_score + (result.similarity_score * self.fulltext_weight)
                combined_results[node_id] = SearchResult(
                    node_id=node_id,
                    similarity_score=combined_score,
                    document_text=existing.document_text,
                    node_type=existing.node_type,
                    chat_name=existing.chat_name,
                    metadata=existing.metadata,
                    search_method="hybrid"
                )
            else:
                # Add new full-text result
                combined_results[node_id] = SearchResult(
                    node_id=node_id,
                    similarity_score=result.similarity_score * self.fulltext_weight,
                    document_text=result.document_text,
                    node_type=result.node_type,
                    chat_name=result.chat_name,
                    metadata=result.metadata,
                    search_method="fulltext"
                )
        
        # Add or combine graph results
        for result in graph_results:
            node_id = result.node_id
            
            if node_id in combined_results:
                # Combine scores if found in other methods
                existing = combined_results[node_id]
                combined_score = existing.similarity_score + (result.similarity_score * self.graph_weight)
                combined_results[node_id] = SearchResult(
                    node_id=node_id,
                    similarity_score=combined_score,
                    document_text=existing.document_text,
                    node_type=existing.node_type,
                    chat_name=existing.chat_name,
                    metadata=existing.metadata,
                    search_method="hybrid"
                )
            else:
                # Add new graph result
                combined_results[node_id] = SearchResult(
                    node_id=node_id,
                    similarity_score=result.similarity_score * self.graph_weight,
                    document_text=result.document_text,
                    node_type=result.node_type,
                    chat_name=result.chat_name,
                    metadata=result.metadata,
                    search_method="graph"
                )
        
        # Sort by combined score
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return final_results
    
    def process(self, user_query: str) -> VectorSearchResults:
        """
        Perform hybrid search combining vector, full-text, and graph search.
        
        Args:
            user_query: Raw user query string
            
        Returns:
            VectorSearchResults containing ranked results from all methods
        """
        self.logger.info(f"Performing hybrid search for: {user_query}")
        
        # Extract keywords and entities for enhanced search
        keywords = self._extract_keywords(user_query)
        entities = self._extract_potential_entities(user_query)
        self.logger.info(f"Extracted keywords: {keywords}")
        self.logger.info(f"Extracted entities: {entities}")
        
        # Perform enhanced vector search with query and entities (if enabled)
        if self.enable_vector_search:
            vector_results = self._enhanced_vector_search(user_query, entities)
            self.logger.info(f"Enhanced vector search found {len(vector_results)} results")
        else:
            vector_results = []
            self.logger.info("Vector search skipped (disabled)")
        
        # Perform full-text search with enhanced query (if enabled)
        if self.enable_fulltext_search:
            fulltext_results = self._enhanced_fulltext_search(user_query, keywords, [entity[0] for entity in entities])
            self.logger.info(f"Enhanced full-text search found {len(fulltext_results)} results")
        else:
            fulltext_results = []
            self.logger.info("Full-text search skipped (disabled)")
        
        # Perform graph search for people (if enabled)
        if self.enable_graph_search:
            person_results = self._enhanced_graph_search(user_query, keywords, entities)
            self.logger.info(f"Graph search found: {person_results}")
        else:
            person_results = []
            self.logger.info("Graph search skipped (disabled)")
        

        # Fuse results
        fused_results = self._fuse_results(vector_results, fulltext_results, person_results)
        self.logger.info(f"Fused search produced {len(fused_results)} results")
        
        # Organize results
        entity_ids = set()
        person_ids = set()
        claim_ids = set()
        full_text_ids = set()
        
        for result in fused_results:
            if result.node_type == "entity":
                entity_ids.add(result.node_id)
            elif result.node_type == "person":
                person_ids.add(result.node_id)  # Add people to entity_ids for compatibility
            elif result.node_type in ["claim", "full_text"]:
                if result.node_type == "claim":
                    claim_ids.add(result.node_id)
                else:
                    full_text_ids.add(result.node_id)
        
        # Return VectorSearchResults for compatibility with existing pipeline
        non_person_entities = []
        for entity in entities:
            if entity[1] != "PERSON":
                non_person_entities.append(entity[0])

        return VectorSearchResults(
            entity_ids=entity_ids,
            person_ids=person_ids,
            claim_ids=claim_ids,
            full_text_ids=full_text_ids,
            results=fused_results,
            original_query=user_query,
            extracted_entities=non_person_entities,
            extracted_keywords=keywords,
            extracted_variations=[]
        )
    
    def _log_step_result(self, result: VectorSearchResults):
        """Log summary of hybrid search results."""
        self.logger.info(f"Hybrid search summary:")
        self.logger.info(f"  Final fused results: {len(result.results)}")
        self.logger.info(f"  Entity IDs: {len(result.entity_ids)}")
        self.logger.info(f"  Claim IDs: {len(result.claim_ids)}")
        self.logger.info(f"  Full-text IDs: {len(result.full_text_ids)}")
        
        if self.logger.isEnabledFor(logging.DEBUG):
            for i, search_result in enumerate(result.results[:5]):
                self.logger.debug(f"Result {i+1}: {search_result.search_method} "
                                f"'{search_result.node_id}' (score: {search_result.similarity_score:.3f})")
    
    def __del__(self):
        """Cleanup connections when object is destroyed."""
        if hasattr(self, 'graph_searcher') and self.graph_searcher:
            self.graph_searcher.close()
        if hasattr(self, 'vector_searcher') and self.vector_searcher:
            self.vector_searcher.close()
        if hasattr(self, 'fulltext_searcher') and self.fulltext_searcher:
            self.fulltext_searcher.close()
