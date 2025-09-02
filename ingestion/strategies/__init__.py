"""
Ingestion Strategies Package

This package contains all the strategy implementations used throughout the ingestion pipeline:
- Parsing strategies for different chat formats (WhatsApp, Facebook, Telegram)
- Chunking strategies for breaking down conversations (time-based, message-count, etc.)
- Chunk processing strategies for cleaning and enriching data (URL expansion, Twitter processing, etc.)
- Cypher query generation strategies for different LLM providers (OpenAI, Groq, Ollama)

All strategies follow their respective abstract base classes and implement the Strategy pattern
for maximum flexibility and extensibility.
"""

# Import all strategy subpackages
from . import chunk_processing, chunking, cypher_query_generation, parsing

__all__ = ["chunk_processing", "chunking", "cypher_query_generation", "parsing"]
