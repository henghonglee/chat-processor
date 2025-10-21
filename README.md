# Chat Text Processor

A comprehensive AI-powered pipeline for processing, analyzing, and querying chat conversations from platforms like WhatsApp, Telegram, and Facebook. Uses advanced NLP techniques to extract entities, build knowledge graphs, and enable intelligent search across chat histories.

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Neo4j** (for graph database)
- **API Keys** for AI services (OpenAI, Google Gemini, Groq)

### 1. Environment Setup

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd chat-text-processor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Neo4j Database

```bash
# Using Docker (recommended)
docker-compose up -d

# Or install Neo4j locally and start it
```

### 3. Configure API Keys

```bash
# Copy environment template
cp env.example .env

# Edit .env with your API keys
nano .env  # or your preferred editor
```

**Required API Keys:**

- `OPENAI_API_KEY` - From [OpenAI Platform](https://platform.openai.com/)
- `GOOGLE_API_KEY` - From [Google AI Studio](https://aistudio.google.com/app/apikey)
- `GROQ_API_KEY` - From [Groq Console](https://console.groq.com/keys)

**Optional API Keys:**

- `OPENROUTER_API_KEY` - For enhanced query output
- `TWITTER_API_KEY` & `TWITTER_BEARER_TOKEN` - For tweet URL expansion

### 4. Process Your First Chat

```bash
# Example: Process a WhatsApp chat
# groq/qwen2.5-72b-instruct is the default model, pass model with --model
python3 ingestion/pipeline.py whatsapp--group

```

### 5. Query Your Data

```bash
# Start the query interface
python3 query/pipeline.py "What did we discuss about the new project?"

# Or use specific search parameters
python3 query/pipeline.py "AI investments" --model gpt-4o-mini --top-k 20
```

## 📁 Project Structure

```
├── chats/                 # Raw chat export files
│   ├── whatsapp/         # WhatsApp exports (.txt)
│   ├── telegram/         # Telegram exports (.json)
│   └── facebook/         # Facebook exports
├── chats-parsed/         # Parsed chat data (JSONL)
├── chats-parsed-chunked/ # Text chunks for processing
├── chats-processed/      # Enriched chunks with entities
├── cypher-queries/       # Generated Cypher queries
├── cypher-processed/     # Executed Cypher files
├── ingestion/            # Data processing pipeline
│   ├── steps/           # Processing steps
│   └── strategies/      # Processing strategies
├── query/               # Query and search system
├── vector_store/        # Embeddings and vector search
└── ingestion/           # Graph database (Neo4j)
```

## 🔧 Configuration

### Environment Variables

| Variable             | Default                  | Description                    |
| -------------------- | ------------------------ | ------------------------------ |
| `OPENAI_API_KEY`     | -                        | Required for OpenAI models     |
| `GOOGLE_API_KEY`     | -                        | Required for entity extraction |
| `GROQ_API_KEY`       | -                        | Required for Cypher generation |
| `NEO4J_URI`          | `bolt://localhost:7687`  | Neo4j connection               |
| `EMBEDDING_PROVIDER` | `openai`                 | `openai` or `ollama`           |
| `OLLAMA_URL`         | `http://localhost:11434` | Local Ollama server            |

### Advanced Search Tuning

```bash
# Adjust search weights (optional)
VECTOR_WEIGHT=0.5      # Vector similarity weight
FULLTEXT_WEIGHT=0.3    # Full-text search weight
GRAPH_WEIGHT=0.2       # Graph relationships weight
VECTOR_SEARCH_TOP_K=30 # Results per vector search
```

## 📊 Pipeline Overview

1. **Parse** raw chat exports into structured JSON
2. **Chunk** conversations into manageable text segments
3. **Enrich** chunks with metadata and entity extraction
4. **Generate** Cypher queries using AI models
5. **Execute** queries against Neo4j graph database
6. **Index** for hybrid search (vector + full-text + graph)

## 🔍 Query Features

- **Hybrid Search**: Combines vector similarity, full-text, and graph relationships
- **Entity-Aware**: Finds mentions of people, organizations, and concepts
- **Contextual**: Maintains conversation context and relationships
- **Multi-Modal**: Supports different AI providers (OpenAI, Groq, Ollama)

## 🐛 Troubleshooting

### Common Issues

**"No auth credentials found"**

- Check your API keys in `.env` file
- Ensure keys are valid and have proper permissions

**"Failed to connect to Neo4j"**

- Verify Neo4j is running: `docker-compose ps`
- Check connection details in `.env`

**"Module not found"**

- Activate virtual environment: `source venv/bin/activate`
- Reinstall requirements: `pip install -r requirements.txt`

**Memory issues with large chats**

- Process smaller batches
- Increase system memory or use cloud instance

### Logs and Debugging

```bash
# Enable verbose logging
export VERBOSE=true
python3 ingestion/pipeline.py your-chat --model groq/qwen2.5-72b-instruct
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

[Add your license information here]

## 📞 Support

For issues and questions:

- Check the troubleshooting section above
- Review existing issues on GitHub
- Create a new issue with detailed information
