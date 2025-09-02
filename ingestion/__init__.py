"""
Ingestion Subsystem

This module contains all components related to data ingestion and processing:
- Parsing chat data from various sources
- Chunking strategies for breaking down conversations
- Chunk processing strategies for cleaning and enriching data
- Entity extraction from processed chunks
- Final ingestion into the knowledge base

The ingestion pipeline processes raw chat data through multiple stages to prepare
it for storage and later querying.
"""

# Main ingestion pipeline steps - import as modules
from . import prompts, steps, strategies

__all__ = ["steps", "prompts", "strategies"]
