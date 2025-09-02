# Chat Text Processor

A comprehensive system for processing chat conversations and enabling natural language queries against the processed data.

## 📁 Project Structure

The project is organized into two main subsystems:

### 🔄 `ingestion/` - Data Ingestion Subsystem

Handles the processing of raw chat data and preparation for storage:

- **`steps/`** - Sequential processing pipeline:
  - `a_parsing.py` - Parse raw chat files (WhatsApp, Facebook, etc.)
  - `b_chunking.py` - Break conversations into logical chunks
  - `c_chunk_processing.py` - Clean and enhance chunk data
  - `d_post_processing.py` - Final processing and validation
  - `e_entity_extraction.py` - Extract entities using LLMs
  - `f_cypher_query_generation.py` - Generate Cypher queries using AI
  - `g_cypher_query_post_processing.py` - Post-process and sanitize Cypher
  - `h_ingestion.py` - Store processed data in Neo4j

- **`chunk_processing_strategies/`** - Different approaches to chunking:
  - `base.py` - Abstract base class for all strategies
  - `message_count.py` - Chunk by number of messages
  - `time_based.py` - Chunk by time gaps between messages
  - `line_count.py` - Chunk by number of lines
  - `hybrid.py` - Combination of multiple strategies

- **`preprocessors/`** - Text preprocessing utilities:
  - `base.py` - Base processor interface
  - `link_cleaner.py` - Clean and expand URLs
  - `twitter_processor.py` - Handle Twitter-specific content
  - `image_processor.py` - Process image attachments
  - `url_expander.py` - Expand shortened URLs

- **`prompts/`** - LLM prompts and configurations:
  - `cypher_prompt.json` - Templates for graph queries
  - `langextract_entities.py` - Entity extraction configurations

- **`cypher_query_ai_strategies/`** - AI strategies for Cypher generation:
  - `base.py` - Abstract base strategy interface
  - `ollama_strategy.py` - Ollama API implementation
  - `openai_strategy.py` - OpenAI API implementation
  - `groq_strategy.py` - Groq API implementation

### 🔍 `query/` - Query Processing Subsystem

Handles natural language queries against the processed data:

- **`steps/`** - Query processing pipeline:
  - `a_vector_search.py` - Vector similarity search for relevant entities and claims
  - `b_cypher_generation.py` - Generate graph database queries
  - `c_query_expansion.py` - Expand queries to get context
  - `d_context_coalescence.py` - Combine results into prompts
  - `e_result_output.py` - Generate final responses
  - `pipeline.py` - Main orchestration pipeline
  - `example.py` - Usage examples

- **`preprocessors/`** - Modular query preprocessing:
  - `base.py` - Abstract base for all preprocessors
  - `text_cleaner.py` - Clean and normalize text
  - `entity_extractor.py` - Extract entities from queries
  - `intent_classifier.py` - Classify user intent
  - `keyword_extractor.py` - Extract important keywords
  - `preprocessor_pipeline.py` - Orchestrate all preprocessing
