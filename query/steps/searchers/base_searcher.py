"""
Abstract Base Searcher

Defines the interface that all searchers must implement
for the hybrid search system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SearchResult:
    """Container for individual search results."""
    
    node_id: str
    similarity_score: float
    document_text: str
    node_type: str  # "entity", "claim", "full_text", "person"
    chat_name: str
    metadata: Dict[str, Any]
    search_method: str  # "vector", "fulltext", "hybrid", "graph"


class BaseSearcher(ABC):
    """Abstract base class for all search implementations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the searcher with configuration.
        
        Args:
            config: Dictionary containing searcher-specific configuration
        """
        self.config = config or {}
        self.setup()
    
    @abstractmethod
    def setup(self):
        """Setup the searcher (connections, indexes, etc.)."""
        pass
    
    @abstractmethod
    def search(self, query: str, **kwargs) -> List[SearchResult]:
        """
        Perform search with the given query.
        
        Args:
            query: The search query string
            **kwargs: Additional search parameters
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        pass
    
    def close(self):
        """
        Clean up resources (optional override).
        
        Default implementation does nothing. Subclasses should override
        if they need to clean up connections, files, etc.
        """
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
