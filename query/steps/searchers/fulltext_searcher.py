"""
Full-Text Searcher

BM25-based full-text search implementation for keyword matching.
"""

import logging
import re
from typing import List

import chromadb

from .base_searcher import BaseSearcher, SearchResult


class FullTextSearcher(BaseSearcher):
    """Full-text search implementation using BM25 algorithm."""
    
    def setup(self):
        """Setup full-text search index from ChromaDB data."""
        self.chroma_client = self.config.get('chroma_client')
        if not self.chroma_client:
            raise ValueError("chroma_client is required in config")

        # Initialize documents to empty dict to prevent AttributeError
        self.documents = {}

        try:
            self.full_text_collection = self.chroma_client.get_collection(name="full_text")
            self.entity_collection = self.chroma_client.get_collection(name="entity")

            # Get all documents for indexing
            self._build_search_index()

        except Exception as e:
            logging.warning(f"Failed to setup full-text search: {e}")
            self.full_text_collection = None
            self.entity_collection = None
            # Keep documents as empty dict
    
    def _build_search_index(self):
        """Build in-memory search index for fast text search."""
        self.documents = {}
        self.term_frequencies = {}
        self.document_frequencies = {}
        self.total_documents = 0
        
        # Get all full-text documents
        if self.full_text_collection:
            all_docs = self.full_text_collection.get(include=['documents', 'metadatas'])
            
            for doc_id, document, metadata in zip(
                all_docs['ids'], all_docs['documents'], all_docs['metadatas']
            ):
                self.documents[doc_id] = {
                    'text': document,
                    'metadata': metadata,
                    'terms': self._tokenize(document)
                }
                self.total_documents += 1
        
        # Build term frequency and document frequency indexes
        self._build_tf_idf_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for search indexing."""
        # Convert to lowercase and extract words
        text = text.lower()
        # Extract alphanumeric words, keep important symbols
        terms = re.findall(r'\b\w+\b', text)
        return terms
    
    def _build_tf_idf_index(self):
        """Build TF-IDF index for BM25 scoring."""
        # Calculate term frequencies for each document
        for doc_id, doc_data in self.documents.items():
            terms = doc_data['terms']
            term_count = len(terms)
            term_freq = {}
            
            for term in terms:
                term_freq[term] = term_freq.get(term, 0) + 1
            
            # Normalize by document length
            for term in term_freq:
                term_freq[term] = term_freq[term] / term_count
            
            self.term_frequencies[doc_id] = term_freq
        
        # Calculate document frequencies
        for doc_id, term_freq in self.term_frequencies.items():
            for term in term_freq:
                if term not in self.document_frequencies:
                    self.document_frequencies[term] = 0
                self.document_frequencies[term] += 1
    
    def search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """Perform BM25-based full-text search."""
        if not hasattr(self, 'documents') or not self.documents:
            return []
        
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        
        # BM25 parameters
        k1 = 1.5  # Controls term frequency saturation
        b = 0.75  # Controls length normalization
        
        # Calculate average document length
        total_length = sum(len(doc['terms']) for doc in self.documents.values())
        avg_doc_length = total_length / self.total_documents if self.total_documents > 0 else 0
        
        scores = {}
        
        # Calculate BM25 score for each document
        for doc_id, doc_data in self.documents.items():
            score = 0.0
            doc_length = len(doc_data['terms'])
            
            for term in query_terms:
                if term in self.term_frequencies[doc_id]:
                    # Term frequency in document
                    tf = self.term_frequencies[doc_id][term]
                    
                    # Document frequency
                    df = self.document_frequencies.get(term, 0)
                    
                    # IDF calculation
                    idf = max(0, (self.total_documents - df + 0.5) / (df + 0.5))
                    
                    # BM25 formula
                    tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
                    score += idf * tf_component
            
            if score > 0:
                scores[doc_id] = score
        
        # Sort by score and return top results
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, score in sorted_results:
            doc_data = self.documents[doc_id]
            metadata = doc_data['metadata']
            
            result = SearchResult(
                node_id=metadata.get('node_id', doc_id),
                similarity_score=score,
                document_text=doc_data['text'],
                node_type=metadata.get('node_type', 'unknown'),
                chat_name=metadata.get('chat_name', 'unknown'),
                metadata=metadata,
                search_method='fulltext'
            )
            results.append(result)
        
        return results
