# Query Preprocessors

This package contains modular preprocessing strategies for user queries. Each preprocessor handles a specific aspect of query preprocessing, allowing for flexible and extensible query processing pipelines.

## Architecture

### Base Classes

- **`BasePreprocessor`**: Abstract base class that defines the interface for all preprocessors
- **`PreprocessorResult`**: Data container for preprocessor results with metadata and error handling

### Individual Preprocessors

#### 1. TextCleaner (`text_cleaner.py`)

Handles text cleaning and normalization:

- URL removal
- Whitespace normalization
- Special character handling
- Text encoding fixes
- Mention/hashtag removal (optional)

**Configuration:**

```python
{
    "remove_urls": True,
    "remove_mentions": False,
    "remove_hashtags": False,
    "normalize_whitespace": True,
    "strip_text": True,
    "preserve_line_breaks": False,
    "remove_special_chars": False
}
```

#### 2. EntityExtractor (`entity_extractor.py`)

Extracts entities from queries:

- Cryptocurrency names and symbols
- Person names (various patterns)
- Financial instruments and prices
- Organizations and companies
- Dates and time expressions
- DeFi protocols and concepts

**Configuration:**

```python
{
    "extract_crypto": True,
    "extract_people": True,
    "extract_prices": True,
    "extract_dates": True,
    "extract_organizations": True,
    "crypto_symbols": ["BTC", "ETH", "SOL", ...],
    "confidence_threshold": 0.5
}
```

**Entity Types:**

- `cryptocurrency`: Crypto symbols and names
- `person`: Individual names
- `price`: Monetary values
- `temporal`: Dates and time expressions
- `organization`: Companies and institutions
- `protocol`: DeFi protocols and platforms

#### 3. IntentClassifier (`intent_classifier.py`)

Classifies user intent from queries:

- Search intents (find, show, what, who, etc.)
- Analysis intents (compare, analyze, trend)
- Sentiment intents (opinion, feeling)
- Relationship intents (connection, between)
- Temporal intents (timeline, history, when)
- Action intents (buy, sell, trade)

**Configuration:**

```python
{
    "confidence_threshold": 0.3,
    "use_entity_context": True,
    "multi_intent": False,
    "multi_intent_threshold": 0.5,
    "intent_categories": {...}  # Custom intent definitions
}
```

**Intent Categories:**

- `search`: Information seeking queries
- `analysis`: Analytical and comparative queries
- `sentiment`: Opinion and feeling queries
- `relationship`: Connection and association queries
- `temporal`: Time-based queries
- `action`: Transaction and action queries
- `informational`: Educational queries
- `comparison`: Comparative queries

#### 4. KeywordExtractor (`keyword_extractor.py`)

Extracts important keywords using:

- Stop word filtering
- TF-IDF scoring
- N-gram extraction
- Domain-specific keyword detection
- Context-aware filtering

**Configuration:**

```python
{
    "max_keywords": 10,
    "min_word_length": 2,
    "include_ngrams": True,
    "ngram_range": (1, 2),
    "use_tfidf": True,
    "boost_domain_terms": True,
    "custom_stop_words": [...]
}
```

### Preprocessor Pipeline

#### PreprocessorPipeline (`preprocessor_pipeline.py`)

Orchestrates all preprocessing strategies in a configurable pipeline:

- Runs preprocessors in sequence
- Passes context between stages
- Combines results into unified output
- Handles errors gracefully
- Provides comprehensive metadata

**Configuration:**

```python
{
    "pipeline_stages": ["text_cleaner", "entity_extractor", "intent_classifier", "keyword_extractor"],
    "fail_on_stage_error": False,
    "stage_configs": {
        "text_cleaner": {...},
        "entity_extractor": {...},
        "intent_classifier": {...},
        "keyword_extractor": {...}
    }
}
```

## Usage Examples

### Basic Usage

```python
from query.preprocessors import PreprocessorPipeline

# Use default configuration
preprocessor = PreprocessorPipeline()
result = preprocessor.process("What did Shaun say about BTC last week?")

if result.success:
    data = result.data
    print(f"Intent: {data['intent']}")
    print(f"Entities: {data['entities']}")
    print(f"Keywords: {data['keywords']}")
```

### Custom Configuration

```python
config = {
    "pipeline_stages": ["text_cleaner", "entity_extractor", "intent_classifier"],
    "stage_configs": {
        "entity_extractor": {
            "confidence_threshold": 0.7,
            "extract_dates": False
        },
        "intent_classifier": {
            "multi_intent": True,
            "confidence_threshold": 0.4
        }
    }
}

preprocessor = PreprocessorPipeline(config)
result = preprocessor.process("Compare BTC and ETH performance")
```

### Individual Preprocessor Usage

```python
from query.preprocessors import TextCleaner, EntityExtractor

# Text cleaning only
cleaner = TextCleaner({"remove_urls": True})
clean_result = cleaner.process("Check out https://example.com for BTC info")

# Entity extraction with custom config
extractor = EntityExtractor({
    "crypto_symbols": ["BTC", "ETH", "SOL", "ADA"],
    "confidence_threshold": 0.8
})
entity_result = extractor.process("What's the BTC price trend?")
```

### Using with Main QueryPreprocessor

```python
from query.steps.a_query_preprocessing import QueryPreprocessor

# Custom preprocessor configuration
config = {
    "stage_configs": {
        "entity_extractor": {
            "extract_organizations": False,
            "boost_crypto_terms": True
        },
        "keyword_extractor": {
            "max_keywords": 15,
            "include_ngrams": True
        }
    }
}

processor = QueryPreprocessor(config)
result = processor.process("Find all conversations about Ethereum")
```

## Extending the System

### Adding New Preprocessors

1. **Create a new preprocessor class** inheriting from `BasePreprocessor`:

```python
from .base import BasePreprocessor, PreprocessorResult

class MyCustomPreprocessor(BasePreprocessor):
    def get_name(self) -> str:
        return "MyCustomPreprocessor"

    def process(self, text: str, context: Optional[Dict[str, Any]] = None) -> PreprocessorResult:
        # Your processing logic here
        return PreprocessorResult(
            success=True,
            data={"custom_data": "processed_result"},
            processing_time=0.1
        )
```

2. **Add to PreprocessorPipeline** by updating the `setup_pipeline` method
3. **Update package exports** in `__init__.py`

### Custom Intent Categories

```python
custom_intents = {
    "trading": {
        "keywords": ["buy", "sell", "trade", "exchange"],
        "patterns": [r"\b(should|can)\s+(i|we)\s+(buy|sell)\b"],
        "weight": 1.5
    }
}

config = {
    "stage_configs": {
        "intent_classifier": {
            "intent_categories": custom_intents
        }
    }
}
```

### Custom Entity Types

```python
config = {
    "stage_configs": {
        "entity_extractor": {
            "crypto_symbols": ["CUSTOM1", "CUSTOM2"],
            "extract_custom_entities": True
        }
    }
}
```

## Performance Considerations

- **Individual preprocessors** can be used for specific tasks to avoid overhead
- **Pipeline stages** can be selectively enabled/disabled
- **Confidence thresholds** can be adjusted to balance precision vs recall
- **Domain boosting** helps with cryptocurrency/finance specific queries
- **Context passing** between stages improves accuracy

## Error Handling

- Each preprocessor returns a `PreprocessorResult` with success/error status
- The preprocessor pipeline can continue on individual stage failures
- Comprehensive error messages and processing times are provided
- Failed stages are tracked in metadata

## Configuration Best Practices

1. **Start with defaults** and customize only what you need
2. **Adjust confidence thresholds** based on your accuracy requirements
3. **Enable domain boosting** for crypto/finance queries
4. **Use multi-intent** for complex queries that might have multiple purposes
5. **Configure pipeline stages** based on your specific use case

## Example Usage

Each preprocessor includes example usage in its `if __name__ == "__main__"` block:

```bash
python -m query.preprocessors.entity_extractor
python -m query.preprocessors.intent_classifier
python -m query.preprocessors.keyword_extractor
```
