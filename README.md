# Chat Text Processor

An AI-first ingestion and query pipeline that turns WhatsApp, Telegram, and Facebook chat exports into a richly linked Neo4j knowledge graph plus a hybrid (vector + full-text + graph) search index backed by ChromaDB.

## TL;DR

- **One command ingestion** turns raw chat exports into Neo4j nodes, relationships, and embeddings using modular parsing, chunking, enrichment, and Cypher generation strategies.
- **Hybrid query pipeline** fuses vector similarity, BM25, and graph traversal, reranks with cross-encoders or hosted APIs, and feeds a context-rich prompt to LLMs.
- **Strategy-driven architecture** (parsers, chunkers, chunk processors, Cypher generators, searchers, rerankers) makes every stage configurable and easy to extend.
- **Multi-provider support**: OpenAI + Groq + Gemini + Ollama for LLM, embedding, and entity extraction workloads; optional Cohere and Jina rerankers.

## Architecture Overview

```
INGESTION PIPELINE
┌───────────────┐   ┌────────────┐   ┌──────────────────┐   ┌──────────────┐
| Chat Exports  | → | Parsing    | → | Chunking &        | → | Chunk         |
| (TXT/JSON)    |   | (strategy) |   | Processing chain  |   | Post-processing|
└───────────────┘   └────────────┘   └──────────────────┘   └──────┬───────┘
                                                                    │
                                                ┌───────────────────┴────────────────┐
                                                | Entity extraction (Gemini LangExtract) |
                                                └──────────────┬──────────────────────┘
                                                               │
                                               ┌───────────────▼──────────────┐
                                               | Cypher generation (OpenAI/   |
                                               | Groq/Ollama strategies)      |
                                               └───────────────┬──────────────┘
                                                               │
                                        ┌──────────────────────▼──────────────────────┐
                                        | Cypher post-processing + Neo4j ingestion +  |
                                        | ChromaDB embeddings                         |
                                        └─────────────────────────────────────────────┘
```

```
QUERY PIPELINE
┌──────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
| User Query   |→ | Preprocessors |→ | Hybrid Search |→ | Rerankers      |
└──────────────┘  └───────────────┘  └───────────────┘  └───────────────┘
                                                         │
     ┌───────────────────────────────┬───────────────────┴─────────────────────────┐
     | Cypher generation (step 3)    │ Query expansion (Neo4j, step 4)             |
     └───────────────┬───────────────┴────────────────────────────┬────────────────┘
                     │                                            │
          ┌──────────▼──────────┐                        ┌────────▼─────────┐
          | Context coalescence |                        | Result outputter |
          └─────────────────────┘                        └──────────────────┘
```

## Table of Contents

1. [🚀 Quick Start](#-quick-start)
2. [📁 Project Structure](#-project-structure)
3. [💬 Supported Chat Platforms](#-supported-chat-platforms)
4. [🧵 Ingestion Pipeline](#-ingestion-pipeline)
5. [🧱 Chunking Strategies](#-chunking-strategies)
6. [🧼 Chunk Processing Strategies](#-chunk-processing-strategies)
7. [🧠 Entity Extraction](#-entity-extraction)
8. [🧾 Cypher Query Generation Strategies](#-cypher-query-generation-strategies)
9. [🌳 Neo4j Ingestion & Embeddings](#-neo4j-ingestion--embeddings)
10. [🔎 Query Pipeline](#-query-pipeline)
11. [📦 Searcher Types](#-searcher-types)
12. [📈 Reranking Options](#-reranking-options)
13. [🧯 Query Preprocessors](#-query-preprocessors)
14. [🧰 API Reference](#-api-reference)
15. [⚙️ Configuration & Environment Variables](#%EF%B8%8F-configuration--environment-variables)
16. [🧪 Advanced Usage & Extensibility](#-advanced-usage--extensibility)
17. [🐛 Troubleshooting](#-troubleshooting)
18. [🤝 Contributing & Support](#-contributing--support)

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.8+**
- **Neo4j 5.x + APOC** (run via `docker-compose.yml` or locally)
- **ChromaDB** (auto-created in `vector_store/`)
- API keys (see [Configuration](#%EF%B8%8F-configuration--environment-variables))

### 2. Environment Setup

```bash
git clone <repository-url>
cd chat-text-processor
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
nano .env
```

### 3. Launch Neo4j & Supporting Services

```bash
docker-compose up -d          # Runs Neo4j with APOC and persists data
```

### 4. Process a Chat Export

```bash
# Parse + chunk + enrich + ingest a WhatsApp group export
python3 ingestion/pipeline.py whatsapp-yourgroup \
  --model groq/qwen2.5-72b-instruct \
  --chunking-config configs/chunking/time_40min.json \
  --processors-config configs/processors/default.json
```

Common CLI flags (per ingestion step):

| Step | Entry point | Useful flags |
|------|-------------|--------------|
| Parsing | `python3 ingestion/steps/a_parsing.py whatsapp-yourgroup` | `--chat-name` |
| Chunking | `python3 ingestion/steps/b_chunking.py whatsapp-yourgroup --config configs/chunking/hybrid.json` | `--strategy-config` |
| Chunk processing | `python3 ingestion/steps/c_chunk_processing.py whatsapp-yourgroup --max-workers 8` | `--processors-config`, `--max-workers` |
| Entity extraction | `python3 ingestion/steps/e_entity_extraction.py whatsapp-yourgroup` | `--chunk-id` |
| Cypher generation | `python3 ingestion/steps/f_cypher_query_generation.py whatsapp-yourgroup --model openai/gpt-4o-mini` | `--prompt-version`, `--chunk-id` |
| Neo4j ingestion | `python3 ingestion/steps/g_ingestion.py whatsapp-yourgroup --dry-run` | `--list`, `--start-chunk`, `--ingest-only` |

### 5. Ask Questions

```bash
# Default hybrid pipeline (vector + full-text + graph + reranker)
python3 query/pipeline.py "What did Sam say about the launch timeline?" \
  --model gpt-4o-mini --top-k 20

# Disable graph search and reranking for faster iteration
python3 query/pipeline.py "Summaries about invoices" --no-graph --no-fulltext
```

## 📁 Project Structure

```
├── chats/                          # Raw exports per provider
├── chats-parsed/                   # JSONL intermediate representation
├── chats-parsed-chunked/           # Chunk files per chat
├── chats-processed/                # Chunk-processor outputs
├── chats-enriched/                 # Entity-enriched chunks
├── cypher-queries/                 # Raw Cypher per chunk
├── cypher-processed/               # Sanitized Cypher ready for Neo4j
├── vector_store/                   # ChromaDB persistent workspace
├── ingestion/
│   ├── steps/                      # a_parsing … g_ingestion
│   └── strategies/                 # parsing, chunking, chunk_processing, cypher_query_generation
└── query/
    ├── pipeline.py
    ├── preprocessors/              # modular query preprocessing
    └── steps/                      # hybrid search → result output
```

## 💬 Supported Chat Platforms

| Platform | Export Format | Parser | Notes |
|----------|---------------|--------|-------|
| WhatsApp | TXT (UTF-8) from **Export Chat > Without Media** | `ingestion/strategies/parsing/whatsapp_parser.py` | Supports legacy timestamp formats and media placeholders (“image omitted”). Optional `context.txt` used for metadata. |
| Telegram | JSON from **Export chat history** with “Machine-readable JSON” option | `ingestion/strategies/parsing/telegram_parser.py` | Handles text entities, polls, replies, forwards, stickers, media attachments, and service messages. |
| Facebook Messenger | JSON from **Download Your Information** (Messages > JSON) | `ingestion/strategies/parsing/facebook_parser.py` | Normalizes reactions, media attachments, and converts reverse chronological exports into timeline order. |

How to prepare exports:

1. Place each chat under `chats/<provider>/<chat-folder>/`.
2. Optional `context.txt` can include small human summaries, pinned message references, or channel notes.
3. Use prefixes (`whatsapp-`, `telegram-`, `facebook-`) when invoking ingestion CLIs to auto-select the parser (`whatsapp-groupslack` → `chats/whatsapp/groupslack/`).

## 🧵 Ingestion Pipeline

Each stage runs independently (CLI or orchestrated via `ingestion/pipeline.py`), so you can re-run/parallelize specific steps.

1. ### Parsing (`ingestion/steps/a_parsing.py`)
   - `ChatParser` auto-detects provider, delegates to strategy classes, and writes JSONL with context metadata + message rows.
   - Output lives in `chats-parsed/<chat-name>.jsonl`.
   - Use `--chat-name` to process multiples; logs include per-message counts.

2. ### Chunking (`ingestion/steps/b_chunking.py`)
   - `ChatChunker` applies a `ChunkingStrategy` (time, message count, line count, or hybrid) to produce coherent context windows.
   - Writes `chunk_id`, `page_content`, and metadata (chat name, message count, context) into `chats-parsed-chunked/<chat>/`.
   - Configurable through JSON (see [Chunking Strategies](#-chunking-strategies)).

3. ### Chunk Processing (`ingestion/steps/c_chunk_processing.py`)
   - Chain-of-responsibility pipeline composed of processors from `ingestion/strategies/chunk_processing`.
   - Default chain: `LinkCleaner → URLExpander → TwitterProcessor → ImageProcessor`.
   - Supports parallelism (`--max-workers`) and URL cache re-use (`url_cache.csv`).
   - Output stored under `chats-processed/<chat>/`.

4. ### Post-Processing (`ingestion/steps/d_post_processing.py`)
   - Consolidates processed chunk files back into a single JSONL (`chats-enriched/<chat>.jsonl`) while preserving context metadata.
   - Use for quick exports or validating intermediate states before entity extraction.

5. ### Entity Extraction (`ingestion/steps/e_entity_extraction.py`)
   - Uses Gemini LangExtract (`langextract` package) plus prompts defined in `ingestion/prompts/langextract_entities.py`.
   - Annotates each chunk with `extracted_entities` (type, text, attributes) and skips already-processed chunks automatically.
   - Requires `GOOGLE_API_KEY` or `LANGEXTRACT_API_KEY`.

6. ### Cypher Generation (`ingestion/steps/f_cypher_query_generation.py`)
   - Converts enriched chunks to Neo4j Cypher via pluggable AI strategies (OpenAI, Groq, Ollama).
   - Loads prompts from `ingestion/prompts/cypher_prompt_v*.json`.
   - One Cypher file per chunk in `cypher-queries/<chat>/chunk_###.cypher`.

7. ### Cypher Post-Processing & Neo4j Ingestion (`ingestion/steps/g_ingestion.py`)
   - Sanitizes SAID/MENTION conflicts, normalizes variable names, extracts text/entity embeddings, and persists them to Chroma (`full_text`, `entity` collections).
   - Executes Cypher against Neo4j via the official driver and tracks success/failure counts.
   - Supports dry runs, chunk ranges, post-process-only, or ingest-only modes, making it safe to resume after failures.

## 🧱 Chunking Strategies

Located in `ingestion/strategies/chunking/` and instantiated via `ChatChunker.create_strategy_from_config`.

| Strategy | Class | Key config | Behavior |
|----------|-------|------------|----------|
| Time-based | `TimeBasedChunkingStrategy` | `gap_minutes` | Starts a new chunk when the time gap between messages exceeds threshold. Great for asynchronous group chats. |
| Message-count | `MessageCountChunkingStrategy` | `max_messages` | Forces uniform chunk sizes by message count. Use for analytics-friendly batches. |
| Line-count | `LineCountChunkingStrategy` | `max_lines` | Useful when dealing with multi-line media or forwarded blocks. |
| Hybrid | `HybridChunkingStrategy` | list of strategies | Runs multiple strategies, starting a new chunk when **any** inner strategy fires (e.g., time gap **or** 200 lines). |

Example config (`configs/chunking/hybrid.json`):

```json
{
  "type": "hybrid",
  "strategies": [
    {"type": "time", "gap_minutes": 30},
    {"type": "message_count", "max_messages": 120}
  ]
}
```

Pass via CLI: `python3 ingestion/steps/b_chunking.py whatsapp-yourchat --config configs/chunking/hybrid.json`.

## 🧼 Chunk Processing Strategies

Modules live in `ingestion/strategies/chunk_processing/` and plug into `ChainOfResponsibilityProcessor`.

| Processor | Purpose | Highlights |
|-----------|---------|------------|
| `LinkCleanerProcessor` | Removes marketing/tracking query params (UTM, fbclid, etc.) | Keeps content canonical for embeddings. |
| `URLExpanderProcessor` | Fetches article content, handles binary files, caches responses | Uses `requests`, optional `trafilatura`/BeautifulSoup, respects timeouts. |
| `TwitterProcessor` | Captures tweet text/media via multiple strategies (syndication, guest token, API, Selenium fallback) | Caches successes/failures in `url_cache.csv`; handles removed tweets gracefully. |
| `ImageProcessor` | Logs image URLs to maintain references | Appends `--- Image: url ---` markers to chunks. |

Configure processors via JSON:

```json
{
  "processors": [
    {"type": "link_cleaner"},
    {"type": "url_expander", "params": {"timeout": 20}},
    {"type": "twitter", "params": {"timeout": 15}},
    {"type": "image"}
  ]
}
```

## 🧠 Entity Extraction

- Backed by Gemini LangExtract through `langextract`.
- Configurable prompt/examples in `ingestion/prompts/langextract_entities.py`.
- CLI resumes from existing outputs and supports single-chunk reprocessing via `--chunk-id`.
- Entities are stored as `{"type": "...", "text": "...", "attributes": {...}}` per chunk, enabling downstream Cypher generation to reason about people, organizations, locations, protocols, etc.

## 🧾 Cypher Query Generation Strategies

Strategy classes in `ingestion/strategies/cypher_query_generation/` share the `BaseCypherQueryAIStrategy` contract:

| Strategy | When to use | Env requirements |
|----------|-------------|------------------|
| `OpenAICypherQueryAIStrategy` | Highest quality with GPT-4o/4o-mini | `OPENAI_API_KEY`, optional `OPENAI_MODEL`, temperature control |
| `GroqCypherQueryAIStrategy` | Fast, cost-efficient long-context models (e.g., Qwen/QwQ) | `GROQ_API_KEY`, `groq` Python SDK |
| `OllamaCypherQueryAIStrategy` | Fully local or air-gapped deployments | `OLLAMA_URL`, running Ollama model (e.g., `llama3`, `codellama`) |

All strategies:
- Load versioned prompts (`CYPHER_PROMPT_VERSION` env).
- Clean LLM responses (remove code fences, `<think>` tags).
- Write queries per chunk and skip previously generated files.

## 🌳 Neo4j Ingestion & Embeddings

`ingestion/steps/g_ingestion.py` orchestrates post-processing + ingestion:

- **Sanitization:** deduplicates conflicting SAID/MENTION edges, enforces valid variable names.
- **Embedding extraction:** collects `full_text` from SAID/REACTED relationships and entity IDs; stores embeddings in Chroma (`full_text`, `entity` collections) via OpenAI or Ollama providers.
- **Neo4j execution:** splits statements, executes sequentially, and prints counters for nodes/relationships created. Requires `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.
- **CLI Quality-of-life:** `--dry-run`, `--list`, chunk range filters, `--post-process-only`, `--ingest-only`.

Neo4j must have APOC installed to support downstream graph procedures (recommended: official Docker image with APOC plugin enabled).

## 🔎 Query Pipeline

`query/pipeline.py` composes the runtime pipeline. Steps:

1. **Query Preprocessing (optional)** – `PreprocessorPipeline` cleans URLs, normalizes whitespace, extracts intent/entities/keywords. Controlled through `enable_query_preprocessing` + `replace_personal_pronouns`.
2. **Hybrid Search (`query/steps/a_hybrid_search.py`)** – `HybridSearcher` combines:
   - Vector search (`VectorSearcher`) over Chroma embeddings.
   - Full-text BM25 search (`FullTextSearcher`) using local inverted index.
   - Graph search (`GraphSearcher`) querying Neo4j for person nodes.
   - Weighted fusion controlled by `VECTOR_WEIGHT`, `FULLTEXT_WEIGHT`, `GRAPH_WEIGHT`.
3. **Reranking (`query/steps/b_reranker.py`)** – `RerankerStep` applies one of four rerankers (see next section) to the top-K search results and boosts exact/entity matches or recent timestamps.
4. **Cypher Generation (`query/steps/c_cypher_generation.py`)** – Builds templated Cypher queries targeting the IDs returned by hybrid search (entity, claim, person, and text IDs).
5. **Query Expansion (`query/steps/d_query_expansion.py`)** – Executes Cypher against Neo4j, gathers nodes/relationships/path metadata into a `GraphContext`.
6. **Context Coalescence (`query/steps/e_context_coalescence.py`)** – Formats the graph/search context plus user query into a system prompt using `query/steps/prompts/system_prompt.md`.
7. **Result Output (`query/steps/f_result_output.py`)** – Sends the prompt to the configured LLM (OpenAI or other provider via `OPENAI_API_KEY`/`OPENAI_MODEL`) and prints JSON or text with optional timing/token stats.

## 📦 Searcher Types

| Searcher | File | Description |
|----------|------|-------------|
| `VectorSearcher` | `query/steps/searchers/vector_searcher.py` | Generates embeddings via OpenAI (`text-embedding-3-large` by default) or Ollama, queries Chroma collections (`full_text`, `entity`), and returns similarity-scored documents. |
| `FullTextSearcher` | `query/steps/searchers/fulltext_searcher.py` | Builds an in-memory BM25 index from Chroma docs for precise keyword matching (useful for short proper nouns, hashtags, etc.). |
| `GraphSearcher` | `query/steps/searchers/graph_searcher.py` | Direct Neo4j lookup for person nodes by ID/name, returning small context snippets and boosting graph-aware queries. |

Environment knobs:
- `VECTOR_SEARCH_TOP_K`, `FULLTEXT_SEARCH_TOP_K`, `SIMILARITY_THRESHOLD`.
- Disable features via CLI flags (`--no-vector`, `--no-fulltext`, `--no-graph`).

## 📈 Reranking Options

All rerankers implement `BaseReranker` and accept a `RerankConfig`.

| Method | Class | Best for | Requirements |
|--------|-------|----------|--------------|
| Cross-Encoder | `CrossEncoderReranker` | Highest accuracy, cross-encoder interactions (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) | Downloads models via `sentence-transformers`, GPU optional. |
| ZeRank (bi-encoder) | `ZeRankReranker` | Fast offline reranking with cosine similarity (e.g., `all-MiniLM-L6-v2`) | Requires local embeddings; good default when GPU not available. |
| Cohere API | `CohereReranker` | Managed reranking (English/multilingual) | `COHERE_API_KEY`, internet access. |
| Jina API | `JinaReranker` | Cost-effective managed reranking | `JINA_API_KEY`, internet access. |

Config (env or CLI):

```
RERANK_METHOD=cross-encoder
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
RERANK_TOP_K_BEFORE=60
RERANK_TOP_K_AFTER=15
RERANK_MIN_SCORE=0.25
```

## 🧯 Query Preprocessors

Reference: `query/preprocessors/README.md`.

| Preprocessor | File | Capabilities | Key config |
|--------------|------|--------------|------------|
| TextCleaner | `text_cleaner.py` | URL stripping, whitespace normalization, optional mention/hashtag removal | `remove_urls`, `normalize_whitespace`, `strip_text`, etc. |
| EntityExtractor | `entity_extractor.py` | Regex/statistical extraction for crypto symbols, people, organizations, dates, protocols | `extract_people`, `extract_dates`, `confidence_threshold`. |
| IntentClassifier | `intent_classifier.py` | Pattern-based classification (search, analysis, sentiment, timeline, action, etc.) | `intent_categories`, `multi_intent`, `use_entity_context`. |
| KeywordExtractor | `keyword_extractor.py` | TF-IDF + n-grams + domain boosting | `max_keywords`, `ngram_range`, `use_tfidf`, `custom_stop_words`. |
| PreprocessorPipeline | `preprocessor_pipeline.py` | Orchestrates ordered stages, passes context, handles errors | `pipeline_stages`, `fail_on_stage_error`, per-stage configs. |

Example custom preprocessor setup:

```python
from query.preprocessors import PreprocessorPipeline

config = {
    "pipeline_stages": [
        "text_cleaner",
        "entity_extractor",
        "intent_classifier",
        "keyword_extractor"
    ],
    "stage_configs": {
        "entity_extractor": {"extract_prices": False},
        "intent_classifier": {"multi_intent": True, "confidence_threshold": 0.4}
    }
}

preprocessor = PreprocessorPipeline(config)
result = preprocessor.process("Compare BTC and ETH performance last month")
```

## 🧰 API Reference

| Module | Class / Function | Description | Key methods |
|--------|------------------|-------------|-------------|
| `ingestion/steps/a_parsing.py` | `ChatParser` | Detects chat provider, runs parser strategies, writes JSONL | `process_single_chat(chat_name)` |
| `ingestion/steps/b_chunking.py` | `ChatChunker` | Applies chunking strategies to JSONL messages | `chunk_messages`, `process_single_chat`, `create_strategy_from_config` |
| `ingestion/steps/c_chunk_processing.py` | `ChainOfResponsibilityProcessor` | Builds processor chain, processes chunks in parallel | `process_single_chat`, `create_processor_chain_from_config` |
| `ingestion/steps/f_cypher_query_generation.py` | `generate_cypher_for_chat` | CLI entry for AI-powered Cypher generation | `create_cypher_query_ai_strategy`, `generate_cypher_for_chunk` |
| `ingestion/steps/g_ingestion.py` | `process_cypher_chunks` | Combined post-process + Neo4j ingest driver | `process_single_chunk_post_processing`, `process_single_chunk_ingestion` |
| `query/pipeline.py` | `QueryPipeline` | High-level orchestrator for preprocess → output | `process_query`, `close`, `create_pipeline_from_env` |
| `query/steps/a_hybrid_search.py` | `HybridSearcher` | Executes vector/full-text/graph search fusion | `execute`, `_fuse_results` |
| `query/steps/b_reranker.py` | `RerankerStep` | Config-driven reranking + boosting | `process`, `_apply_boosting` |
| `query/steps/d_query_expansion.py` | `QueryExpander` | Runs Cypher against Neo4j and builds `GraphContext` | `set_neo4j_driver`, `_process_query_results` |
| `query/steps/e_context_coalescence.py` | `ContextCoalescer` | Converts graph/search context into LLM prompt | `build_system_prompt`, `process` |

Search within each module for `main()` to see CLI usage, or import the classes directly inside custom workflows/notebooks.

## ⚙️ Configuration & Environment Variables

### Core AI Providers

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` / `OPENAI_MODEL` | Required for embeddings, Cypher generation, and final answer synthesis (unless using Ollama). |
| `GOOGLE_API_KEY` or `LANGEXTRACT_API_KEY` | Required for Gemini LangExtract entity extraction. |
| `GROQ_API_KEY` | Optional LLM provider for Cypher generation. |
| `OLLAMA_URL`, `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL` | Use Ollama for embeddings and/or Cypher generation. |

### Graph & Storage

| Variable | Default |
|----------|---------|
| `NEO4J_URI` | `bolt://localhost:7687` |
| `NEO4J_USER` / `NEO4J_PASSWORD` | `neo4j` / `password` |
| `VECTOR_STORE_PATH` | `./vector_store` (implicit via `chromadb.PersistentClient`) |

### Search & Ranking

| Variable | Default | Purpose |
|----------|---------|---------|
| `VECTOR_WEIGHT`, `FULLTEXT_WEIGHT`, `GRAPH_WEIGHT` | `0.5 / 0.3 / 0.2` | Fuse hybrid scores. |
| `VECTOR_SEARCH_TOP_K`, `FULLTEXT_SEARCH_TOP_K` | `30 / 20` | Max hits per modality. |
| `SIMILARITY_THRESHOLD` | `0.05` | Filter vector hits. |
| `RERANK_METHOD`, `RERANK_MODEL`, `RERANK_TOP_K_BEFORE`, `RERANK_TOP_K_AFTER` | see `.env.example` | Configure reranking approach. |
| `COHERE_API_KEY`, `JINA_API_KEY` | Optional cloud rerankers. |

### Other optional knobs

- `OPENROUTER_API_KEY` – use OpenRouter for query answering instead of direct OpenAI.
- `TWITTER_API_KEY`, `TWITTER_BEARER_TOKEN` – power advanced tweet expansion paths.
- `VERBOSE=true` – enable detailed logging across ingestion/query steps.

### Sample `.env`

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
GOOGLE_API_KEY=your-gemini-key
GROQ_API_KEY=groq-...
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4jpassword
EMBEDDING_PROVIDER=openai
CYPHER_PROMPT_VERSION=v2
VECTOR_WEIGHT=0.5
FULLTEXT_WEIGHT=0.3
GRAPH_WEIGHT=0.2
RERANK_METHOD=cross-encoder
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
COHERE_API_KEY=optional
JINA_API_KEY=optional
```

## 🧪 Advanced Usage & Extensibility

### Custom Chunking Strategy

```python
from ingestion.steps.b_chunking import ChatChunker
from ingestion.strategies.chunking import TimeBasedChunkingStrategy, MessageCountChunkingStrategy, HybridChunkingStrategy

strategy = HybridChunkingStrategy([
    TimeBasedChunkingStrategy(gap_minutes=20),
    MessageCountChunkingStrategy(max_messages=80)
])
chunker = ChatChunker(strategy)
chunker.process_single_chat("whatsapp-myfamily")
```

### Adding a New Chunk Processor

```python
from ingestion.strategies.chunk_processing.base import BaseProcessor

class EmojiNormalizer(BaseProcessor):
    def _process_content(self, content: str, context: dict) -> str:
        return content.replace("👍🏻", "👍")  # custom logic
```

Insert into the processor config before URL expansion to ensure normalized content flows downstream.

### Custom Query Pipeline Configuration

```python
from query.pipeline import PipelineConfig, QueryPipeline

config = PipelineConfig(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    neo4j_uri=os.environ["NEO4J_URI"],
    neo4j_user=os.environ["NEO4J_USER"],
    neo4j_password=os.environ["NEO4J_PASSWORD"],
    enable_fulltext_search=True,
    enable_vector_search=True,
    enable_graph_search=False,  # disable for local prototyping
    enable_reranking=False,
    model="gpt-4o-mini"
)

pipeline = QueryPipeline(config)
print(pipeline.process_query("Summaries about the Zurich offsite"))
pipeline.close()
```

### Production Deployment Tips

- **Neo4j**: run in Aura or Docker with mounted volumes; enable nightly APOC upgrades; consider multi-database setups for staging vs prod.
- **ChromaDB**: host behind persistent disk or upgrade to server mode; snapshot `vector_store/` alongside Neo4j exports.
- **Secrets management**: load `.env` via Vault/SSM and restrict per-service API scopes.
- **Observability**: set `VERBOSE=true` to streaming logs, wrap ingestion steps with job schedulers (Airflow, Dagster) to monitor chunk counts and ingestion success/failure metrics.

## 🐛 Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: langextract` | Optional dependency missing | `pip install langextract` or disable entity extraction. |
| Neo4j connection failures | Container not running / credentials wrong | `docker-compose logs neo4j`, verify env, ensure APOC plugin is enabled. |
| Embedding failures | `OPENAI_API_KEY` invalid or missing | Refresh key, or set `EMBEDDING_PROVIDER=ollama`. |
| Twitter expansion slow | Rate limits / missing API keys | Provide `TWITTER_BEARER_TOKEN` or set processor config to skip twitter. |
| Query pipeline stuck | Neo4j unreachable during query expansion | Run with `--no-graph` temporarily and re-enable after Neo4j recovers. |

Enable verbose logs per step:

```bash
export VERBOSE=true
python3 ingestion/steps/g_ingestion.py whatsapp-yourchat --dry-run
```

## 🤝 Contributing & Support

1. Fork the repo.
2. Create a branch (`git checkout -b feat/my-change`).
3. Add tests or CLI examples where possible.
4. Run linting/formatting (`ruff`, `black`, etc., if configured).
5. Submit a PR referencing any related issues.

Need help?
- Review this README and linked module docs.
- Search existing GitHub issues / discussions.
- File a new issue with logs, command used, and environment info.
