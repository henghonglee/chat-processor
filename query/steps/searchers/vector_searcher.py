"""
Vector Searcher

Embedding-based semantic similarity search implementation.
"""

import logging
import os
from typing import List

import chromadb
import openai
import requests

from .base_searcher import BaseSearcher, SearchResult


class VectorSearcher(BaseSearcher):
    """Vector similarity search implementation using embeddings."""
    
    def setup(self):
        """Setup vector search configuration and collections."""
        # Embedding configuration
        self.embedding_provider = self.config.get("embedding_provider", os.getenv("EMBEDDING_PROVIDER", "openai"))
        self.embedding_model = self.config.get("embedding_model", os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"))
        self.ollama_url = self.config.get("ollama_url", os.getenv("OLLAMA_URL", "http://localhost:11434"))
        
        # Search configuration
        self.top_k = self.config.get("top_k", int(os.getenv("VECTOR_SEARCH_TOP_K", "30")))
        self.similarity_threshold = self.config.get("similarity_threshold", float(os.getenv("SIMILARITY_THRESHOLD", "0.05")))
        
        # Initialize OpenAI client for embeddings
        if self.embedding_provider == "openai":
            openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize ChromaDB client
        self.chroma_client = self.config.get('chroma_client')
        if not self.chroma_client:
            raise ValueError("chroma_client is required in config")
        
        # Setup collections
        self._setup_collections()
    
    def _setup_collections(self):
        """Setup ChromaDB collections."""
        try:
            self.entity_collection = self.chroma_client.get_collection(name="entity")
            logging.info("Connected to entity collection")
        except Exception as e:
            logging.warning(f"Entity collection not found: {e}")
            self.entity_collection = None
        
        try:
            self.full_text_collection = self.chroma_client.get_collection(name="full_text")
            logging.info("Connected to full_text collection")
        except Exception as e:
            logging.warning(f"Full_text collection not found: {e}")
            self.full_text_collection = None
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using configured provider."""
        try:
            if self.embedding_provider == "openai":
                response = openai.embeddings.create(
                    model=self.embedding_model, input=text
                )
                return response.data[0].embedding
            elif self.embedding_provider == "ollama":
                response = requests.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={"model": self.embedding_model, "prompt": text},
                    timeout=30,
                )
                response.raise_for_status()
                return response.json()["embedding"]
            else:
                logging.error(f"Unsupported embedding provider: {self.embedding_provider}")
                return []
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            return []
    
    def search(self, query: str, collection: str = None, **kwargs) -> List[SearchResult]:
        """
        Perform vector similarity search.
        
        Args:
            query: Search query string
            collection: Optional specific collection to search ("entity" or "full_text")
            **kwargs: Additional search parameters (top_k override)
            
        Returns:
            List of SearchResult objects sorted by similarity
        """
        top_k = kwargs.get('top_k', self.top_k)
        
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            return []
        
        results = []
        
        # Determine which collections to search
        collections_to_search = []
        
        if collection == "entity" and self.entity_collection:
            collections_to_search.append(("entity", self.entity_collection))
        elif collection == "full_text" and self.full_text_collection:
            collections_to_search.append(("full_text", self.full_text_collection))
        else:
            if self.entity_collection:
                collections_to_search.append(("entity", self.entity_collection))
            if self.full_text_collection:
                collections_to_search.append(("full_text", self.full_text_collection))
        
        # Search each collection
        for collection_name, collection_obj in collections_to_search:
            search_results = collection_obj.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            if search_results["ids"] and len(search_results["ids"]) > 0:
                for i, doc_id in enumerate(search_results["ids"][0]):
                    distance = search_results["distances"][0][i]
                    similarity_score = 1 - distance
                    
                    if similarity_score >= self.similarity_threshold:
                        metadata = search_results["metadatas"][0][i]
                        document = search_results["documents"][0][i]
                        
                        results.append(SearchResult(
                            node_id=metadata.get("node_id", doc_id),
                            similarity_score=similarity_score,
                            document_text=document,
                            node_type=metadata.get("node_type", collection_name),
                            chat_name=metadata.get("chat_name", "unknown"),
                            metadata=metadata,
                            search_method="vector"
                        ))
        
        # Sort by similarity score
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]
