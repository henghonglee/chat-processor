"""
Chat Text Processor Pipeline Steps

This directory contains the numbered pipeline steps for the chat text processor.
Each step represents a stage in the processing pipeline:

1. Parsing - Creates intermediate representation from raw chat exports
2. Chunking - Splits messages into chunks using strategy pattern
3. Chunk Processing - Processes chunks with chain of responsibility pattern
4. Post Processing - Final operations like aggregation and exports
5. Entity Extraction - Extracts entities using LangExtract
6. Ingestion - Ingests data into Neo4j database

All steps use JSONL format throughout for consistency and efficiency.
The files are numbered for easy identification of the processing order
in your file explorer. All pipeline components are organized within
the steps/ directory for a clean, consolidated structure.
"""
