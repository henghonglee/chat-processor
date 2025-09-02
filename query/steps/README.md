# Query Steps Pipeline

This directory contains a complete pipeline for querying the chat text processor knowledge base. The pipeline processes natural language queries and returns contextual answers using the Neo4j graph database and OpenAI's language models.

## Pipeline Overview

The query processing pipeline consists of 5 main steps:

1. **Query Preprocessing** (`1_query_preprocessing.py`)
   - Cleans and normalizes user queries
   - Extracts entities, keywords, and intent
   - Classifies query type for optimized processing

2. **Cypher Generation** (`2_cypher_generation.py`)
   - Generates Neo4j Cypher queries based on preprocessed input
   - Creates queries for entities, claims, mentions, and said relationships
   - Handles different query patterns and types

3. **Query Expansion** (`3_query_expansion.py`)
   - Executes Cypher queries against Neo4j database
   - Expands results to get all neighbors (claims, mentions, entities, people, saids)
   - Builds comprehensive graph context

4. **Context Coalescence** (`4_context_coalescence.py`)
   - Combines all retrieved information into structured system prompt
   - Formats context for optimal LLM consumption
   - Includes entities, people, claims, relationships, and temporal information

5. **Result Output** (`5_result_output.py`)
   - Sends system prompt and user query to OpenAI chat completions
   - Formats and prints final results
   - Supports multiple output formats (text/JSON)

## Quick Start

### Prerequisites

1. **Environment Setup**

   ```bash
   # Copy environment template
   cp env.example .env

   # Edit .env with your credentials:
   # - OPENAI_API_KEY
   # - NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
   ```

2. **Dependencies**
   ```bash
   pip install openai neo4j python-dotenv
   ```

### Basic Usage

```python
from query.pipeline import create_pipeline_from_env

# Create pipeline from environment variables
pipeline = create_pipeline_from_env()

try:
    # Process a query
    result = pipeline.process_query("What did Shaun say about BTC last week?")

finally:
    pipeline.close()
```

### Command Line Usage

```bash
# Basic query
python -m query.pipeline "What did Shaun say about BTC?"

# JSON output
python -m query.pipeline "Show me conversations about Ethereum" --format json

# Verbose logging
python -m query.pipeline "Find relationships between John and crypto" --verbose

# Save result to file
python -m query.pipeline "Timeline of SOL discussions" --save results.json --format json
```

## Configuration

### Pipeline Configuration

The pipeline can be configured via `PipelineConfig`:

```python
from query.pipeline import PipelineConfig, QueryPipeline

config = PipelineConfig(
    # Database connection
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",

    # OpenAI API
    openai_api_key="sk-...",

    # Processing limits
    max_entities=20,
    max_claims=30,
    max_relationships=50,

    # LLM settings
    model="gpt-4-turbo-preview",
    max_tokens=1000,
    temperature=0.7,

    # Output options
    output_format="text",  # or "json"
    include_token_usage=True,
    include_timing=True,
    verbose=False
)

pipeline = QueryPipeline(config)
```

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-your-openai-key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Optional
VERBOSE=false
```

## Architecture

### Data Flow

```text
User Query
    ↓
[1] Query Preprocessing
    ↓
[2] Cypher Generation
    ↓
[3] Query Expansion (Neo4j)
    ↓
[4] Context Coalescence
    ↓
[5] Result Output (OpenAI)
    ↓
Formatted Response
```

### Key Components

#### QueryPreprocessor

- Text cleaning and normalization
- Entity extraction (people, crypto, etc.)
- Intent classification (search, analysis, sentiment, etc.)
- Keyword extraction

#### CypherGenerator

- Template-based query generation
- Support for entity, claim, mention, and said queries
- Relationship discovery queries
- Fuzzy search capabilities

#### QueryExpander

- Neo4j query execution
- Neighbor expansion for comprehensive context
- Error handling and performance optimization
- Result consolidation

#### ContextCoalescer

- Structured prompt generation
- Token estimation and optimization
- Temporal context formatting
- Relationship summarization

#### ResultOutputter

- OpenAI API integration
- Multiple output formats
- Performance metrics
- Result persistence

## Query Types Supported

### Entity Queries

- "What is BTC?"
- "Show me information about Ethereum"
- "Find all cryptocurrency mentions"

### Person Queries

- "What did Shaun say?"
- "Find conversations involving John"
- "Show me Shaun's opinions"

### Relationship Queries

- "What did Shaun say about BTC?"
- "Find connections between John and crypto trading"
- "Show interactions between Shaun and Desmond"

### Temporal Queries

- "What happened last week?"
- "Timeline of BTC discussions"
- "Recent conversations about SOL"

### Sentiment Queries

- "What's the sentiment around USDT?"
- "How do people feel about the market?"
- "Find positive reactions to BTC"

## Performance Considerations

### Query Optimization

- Limit result sets to prevent memory issues
- Use indexed queries where possible
- Batch relationship expansion

### Token Management

- Estimate token usage before LLM calls
- Truncate context if needed
- Optimize prompt structure

### Error Handling

- Graceful degradation for failed queries
- Retry logic for transient failures
- Comprehensive error reporting

## Extending the Pipeline

### Adding New Query Types

1. **Update QueryPreprocessor**: Add new intent/entity patterns
2. **Extend CypherGenerator**: Add new query templates
3. **Modify ContextCoalescer**: Add new formatting for the query type

### Custom Processors

```python
from query.steps.a_query_preprocessing import QueryPreprocessor

class CustomPreprocessor(QueryPreprocessor):
    def extract_entities(self, text: str):
        # Custom entity extraction logic
        entities = super().extract_entities(text)
        # Add your custom entities
        return entities
```

### Integration with Other LLMs

```python
from query.steps.e_result_output import ResultOutputter

class CustomOutputter(ResultOutputter):
    def send_to_llm(self, system_prompt, user_query):
        # Custom LLM integration
        pass
```

## Troubleshooting

### Common Issues

1. **Neo4j Connection Errors**
   - Check Neo4j is running: `docker ps` or service status
   - Verify connection credentials in `.env`
   - Verify connection with Neo4j Browser

2. **OpenAI API Errors**
   - Verify API key is valid and has credits
   - Check rate limits
   - Monitor token usage

3. **Empty Results**
   - Check if data exists in Neo4j: `MATCH (n) RETURN count(n)`
   - Verify graph schema matches expected structure
   - Enable verbose logging to debug query execution

4. **Performance Issues**
   - Reduce `max_entities`, `max_claims`, `max_relationships`
   - Optimize Neo4j queries with indexes
   - Monitor memory usage during expansion

### Debugging

```bash
# Enable verbose logging
python -m query.pipeline "your query" --verbose

# Check individual components
python -c "
from query.steps.a_query_preprocessing import QueryPreprocessor
proc = QueryPreprocessor()
result = proc.process('your query')
print(result)
"
```

## Contributing

When extending the pipeline:

1. Follow the existing step naming convention
2. Add comprehensive logging
3. Include error handling
4. Update this README with new features
5. Add example usage in the module docstrings

## License

This pipeline is part of the chat-text-processor project.
