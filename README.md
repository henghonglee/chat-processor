# 📱 Chat Text Processor

<p align="center">
  <em>Transform your chat conversations into an intelligent knowledge base with AI-powered analysis and search</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/neo4j-latest-green.svg" alt="Neo4j">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
</p>

## 🎯 What is Chat Text Processor?

**Chat Text Processor** is a sophisticated AI-powered pipeline that transforms raw chat exports from messaging platforms into structured, searchable knowledge bases. It combines advanced natural language processing, graph databases, and multiple AI models to create an intelligent system for analyzing and querying chat conversations.

### ✨ Key Features

- **🔄 Multi-Platform Support**: Process chats from WhatsApp, Telegram, and Facebook
- **🧠 AI-Powered Analysis**: Extract entities, relationships, and insights using OpenAI, Groq, and Google Gemini
- **🔍 Hybrid Search**: Combine vector similarity, full-text search, and graph relationships
- **📊 Knowledge Graph**: Store structured data in Neo4j for complex relationship queries
- **🎛️ Flexible Architecture**: Modular pipeline with configurable processing steps
- **⚡ Multiple AI Providers**: Support for OpenAI, Groq, Ollama, and cloud reranking services
- **🔧 Production Ready**: Docker deployment, comprehensive configuration, and error handling

### 🚀 Use Cases

- **Personal Knowledge Management**: Turn chat history into searchable knowledge base
- **Team Communication Analysis**: Analyze group discussions and extract insights
- **Research & Journalism**: Process interview transcripts and communication records
- **Business Intelligence**: Extract relationships and entities from business communications
- **Content Creation**: Generate insights from community discussions and feedback

## 🚀 Installation & Setup

### 📋 Prerequisites

Before installing Chat Text Processor, ensure your system meets these requirements:

#### System Requirements
- **Python 3.8 or higher**
- **Docker & Docker Compose** (for Neo4j database)
- **4GB RAM minimum** (8GB+ recommended for large chat files)
- **10GB free disk space** (for database and embeddings)

#### Operating System Support
- ✅ **Linux** (Ubuntu 18.04+, CentOS 7+, Fedora)
- ✅ **macOS** (10.15+ with Docker Desktop)
- ✅ **Windows** (10/11 with WSL2 + Docker Desktop)

#### Required API Keys
You'll need API keys from at least one AI provider:

| Provider | Purpose | Get Key |
|----------|---------|---------|
| **OpenAI** | Embeddings, Cypher generation, queries | [platform.openai.com](https://platform.openai.com/) |
| **Google Gemini** | Entity extraction | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| **Groq** | Fast Cypher generation | [console.groq.com](https://console.groq.com/keys) |

### 🛠️ Installation Steps

#### Step 1: Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd chat-text-processor

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 2: Start Neo4j Database

Chat Text Processor uses Neo4j for storing the knowledge graph. The easiest setup is using Docker:

```bash
# Start Neo4j with Docker Compose (recommended)
docker-compose up -d

# Verify Neo4j is running
docker-compose ps
curl http://localhost:7474  # Should show Neo4j browser login
```

**Alternative: Local Neo4j Installation**
```bash
# Download and install Neo4j Desktop from neo4j.com
# OR use system package manager
sudo apt install neo4j  # Ubuntu/Debian
brew install neo4j      # macOS

# Start Neo4j service
sudo systemctl start neo4j    # Linux
neo4j start                   # macOS
```

#### Step 3: Configure Environment Variables

```bash
# Copy the environment template
cp .env.example .env

# Edit with your API keys and preferences
nano .env  # or use your preferred editor
```

**Essential Configuration:**
```bash
# Required: Choose at least one AI provider
OPENAI_API_KEY=sk-your-openai-key-here
GOOGLE_API_KEY=your-google-gemini-key-here
GROQ_API_KEY=your-groq-key-here

# Database connection (default works with Docker setup)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Embedding provider (OpenAI recommended for quality)
EMBEDDING_PROVIDER=openai
```

#### Step 4: Verify Installation

```bash
# Test Python environment
python3 -c "import torch, neo4j, chromadb; print('✅ Core dependencies installed')"

# Test database connection
python3 -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
with driver.session() as session:
    result = session.run('RETURN 1 as test')
    print('✅ Neo4j connection successful')
driver.close()
"

# Test AI provider connectivity (optional)
python3 -c "
import openai
client = openai.OpenAI(api_key='your-key-here')
# This will test if your key works
print('✅ OpenAI API accessible')
"
```

### 🐳 Docker Installation (Alternative)

For a completely containerized setup:

```bash
# Build and run the complete stack
docker-compose -f docker-compose.full.yml up --build

# Access the application
# Web interface: http://localhost:3000
# Neo4j Browser: http://localhost:7474
```

### ☁️ Cloud Deployment Options

#### Google Cloud Platform
```bash
# Deploy to Cloud Run with Neo4j Aura
export GCP_PROJECT=your-project-id
export NEO4J_AURA_CONNECTION_STRING=your-aura-uri

# Deploy with Terraform
cd infrastructure/gcp
terraform init && terraform apply
```

#### AWS Deployment
```bash
# Use ECS with Aurora or DocumentDB
cd infrastructure/aws
terraform init && terraform apply
```

#### Local Development with GPU
```bash
# For GPU-accelerated embeddings (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

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

## 💡 Usage Examples

This section provides practical examples for common use cases and scenarios.

### 📱 Processing Different Chat Platforms

#### WhatsApp Chats
```bash
# Process a WhatsApp group chat
python3 ingestion/pipeline.py whatsapp/my-group-chat --model groq/qwen2.5-72b-instruct

# Process with custom chunking strategy
python3 ingestion/pipeline.py whatsapp/family-group --model openai/gpt-4o-mini --chunking-strategy message_count

# Start from specific step (skip parsing if already done)
python3 ingestion/pipeline.py whatsapp/work-discussion --step 3 --model groq/qwen2.5-72b-instruct
```

#### Telegram Chats
```bash
# Process Telegram export (JSON format)
python3 ingestion/pipeline.py telegram/bot-support-channel --model groq/qwen2.5-72b-instruct

# Process with larger chunks for context preservation
python3 ingestion/pipeline.py telegram/tech-discussion --chunking-strategy time_based --model openai/gpt-4o
```

#### Facebook Conversations
```bash
# Process Facebook Messenger export
python3 ingestion/pipeline.py facebook/friend-conversation --model groq/qwen2.5-72b-instruct

# Process multiple conversations
python3 ingestion/pipeline.py facebook/project-team --model openai/gpt-4o-mini
```

### 🔍 Querying Your Knowledge Base

#### Basic Queries
```bash
# Simple question
python3 query/pipeline.py "What did we discuss about the new project?"

# Search with specific model
python3 query/pipeline.py "Find conversations about cryptocurrency" --model gpt-4o-mini

# Get results in JSON format
python3 query/pipeline.py "Who mentioned the budget?" --format json

# Save results to file
python3 query/pipeline.py "Summarize weekend plans" --save weekend_summary.json
```

#### Advanced Search Options
```bash
# Disable vector search (text-only)
python3 query/pipeline.py "Technical discussions" --no-vector

# Disable full-text search (vector-only)
python3 query/pipeline.py "AI and machine learning" --no-fulltext

# Disable graph search (no relationship queries)
python3 query/pipeline.py "Meeting notes" --no-graph

# Enable verbose output for debugging
python3 query/pipeline.py "Project timeline" --verbose
```

### 🎛️ Working with Different AI Providers

#### OpenAI Models
```bash
# Use GPT-4 for high-quality results
python3 ingestion/pipeline.py whatsapp/tech-chat --model openai/gpt-4o

# Use GPT-3.5-turbo for faster processing
python3 ingestion/pipeline.py telegram/support-chat --model openai/gpt-3.5-turbo

# Query with specific model
python3 query/pipeline.py "Complex analysis question" --model openai/gpt-4o
```

#### Groq Models (Fast Inference)
```bash
# Use Llama 3 70B for Cypher generation
python3 ingestion/pipeline.py whatsapp/large-group --model groq/llama3-70b-8192

# Use Mixtral for entity extraction
python3 ingestion/pipeline.py facebook/business-chat --model groq/mixtral-8x7b-32768

# Fast querying with Groq
python3 query/pipeline.py "Quick information lookup" --model groq/llama3-8b-8192
```

#### Local Ollama Models
```bash
# Use local embeddings (requires Ollama running)
export EMBEDDING_PROVIDER=ollama
python3 ingestion/pipeline.py whatsapp/personal-chat --model openai/gpt-3.5-turbo

# Start Ollama service
ollama serve &
ollama pull nomic-embed-text
```

### 🔧 Pipeline Step Management

#### Running Specific Steps
```bash
# Run only parsing (step 1)
python3 ingestion/pipeline.py whatsapp/chat --step 1 --dry-run

# Start from chunking (step 2)
python3 ingestion/pipeline.py whatsapp/chat --step 2

# Start from entity extraction (step 5)
python3 ingestion/pipeline.py whatsapp/chat --step 5 --model groq/qwen2.5-72b-instruct

# Run entity extraction + Cypher generation (steps 5-6)
python3 ingestion/pipeline.py whatsapp/chat --step 5 --model openai/gpt-4o-mini
```

#### Debugging Pipeline Issues
```bash
# Dry run to see what would be processed
python3 ingestion/pipeline.py whatsapp/test-chat --dry-run --verbose

# Check intermediate results
ls -la chats-parsed/whatsapp/
ls -la chats-parsed-chunked/whatsapp/
ls -la cypher-queries/whatsapp/
```

### 📊 Analyzing Results

#### Check Processing Status
```bash
# View parsed chat structure
head -n 20 chats-parsed/whatsapp/group-chat.jsonl

# Check generated Cypher queries
cat cypher-queries/whatsapp/group-chat.cypher | head -n 50

# Verify Neo4j ingestion
echo "MATCH (n) RETURN count(n) as node_count" | neo4j-admin database query neo4j
```

#### Query Analysis
```bash
# Get conversation statistics
python3 query/pipeline.py "How many messages were processed?" --format json

# Find most active participants
python3 query/pipeline.py "Who are the most active people in the conversation?"

# Search by date range
python3 query/pipeline.py "What was discussed last week about the project?"
```

### 🔄 Batch Processing

#### Processing Multiple Chats
```bash
# Process multiple chats in sequence
for chat in whatsapp/group1 whatsapp/group2 telegram/channel1; do
    echo "Processing $chat..."
    python3 ingestion/pipeline.py $chat --model groq/qwen2.5-72b-instruct
done
```

#### Large File Handling
```bash
# Process large chats in smaller batches
export CHUNK_BATCH_SIZE=50
python3 ingestion/pipeline.py whatsapp/large-group --model groq/qwen2.5-72b-instruct

# Monitor memory usage
python3 -m memory_profiler ingestion/pipeline.py whatsapp/big-chat --model openai/gpt-3.5-turbo
```

### 🎨 Custom Configurations

#### High-Quality Processing
```bash
# Best quality settings
export OPENAI_MODEL=gpt-4o
export RERANK_METHOD=cross-encoder
export RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
export VECTOR_SEARCH_TOP_K=100
python3 ingestion/pipeline.py whatsapp/important-chat
```

#### Fast Processing
```bash
# Speed-optimized settings
export OPENAI_MODEL=gpt-3.5-turbo
export RERANK_METHOD=zerank
export VECTOR_SEARCH_TOP_K=30
export CHUNK_BATCH_SIZE=200
python3 ingestion/pipeline.py whatsapp/quick-chat
```

#### Memory-Constrained Environment
```bash
# Low-memory settings
export VECTOR_SEARCH_TOP_K=20
export RERANK_TOP_K_BEFORE=30
export CHUNK_BATCH_SIZE=25
export EMBEDDING_BATCH_SIZE=10
python3 ingestion/pipeline.py whatsapp/small-chat
```

## 📁 Project Structure & Data Flow

Chat Text Processor follows a clear data pipeline architecture with dedicated directories for each processing stage.

### 📂 Directory Structure

```
chat-text-processor/
├── 📱 chats/                           # Raw chat export files (manual upload)
│   ├── whatsapp/                       # WhatsApp exports (.txt files)
│   │   ├── family-group/
│   │   │   ├── chat.txt               # Raw WhatsApp export
│   │   │   └── context.txt            # Optional context information
│   │   └── work-team/
│   ├── telegram/                       # Telegram exports (.json files)
│   │   └── tech-channel/
│   │       ├── result.json            # Telegram export
│   │       └── context.txt
│   └── facebook/                       # Facebook Messenger exports
│       └── friend-chat/
│           ├── context.txt
│           └── messages.json
│
├── 🔄 chats-parsed/                   # Step 1: Parsed structured data (JSONL)
│   └── whatsapp/family-group.jsonl    # Normalized message format
│
├── ✂️ chats-parsed-chunked/          # Step 2: Text segmentation
│   └── whatsapp/family-group/         # Chunked conversations
│       ├── chunk_001.json
│       └── chunk_002.json
│
├── 🧹 chats-processed/               # Step 3-4: Enhanced chunks
│   └── whatsapp/family-group/         # Processed with metadata
│       ├── processed_chunk_001.json
│       └── processed_chunk_002.json
│
├── 🏷️ entities/                      # Step 5: Extracted entities
│   └── whatsapp/family-group/
│       └── entities.json
│
├── ⚡ cypher-queries/                # Step 6: Generated graph queries
│   └── whatsapp/family-group.cypher  # Neo4j Cypher commands
│
├── 💾 cypher-processed/             # Step 7: Execution logs
│   └── whatsapp/family-group.log     # Ingestion results
│
├── 🧠 vector_store/                 # Embeddings for vector search
│   └── chroma_db/                    # ChromaDB vector database
│       ├── collections/
│       └── embeddings/
│
├── ingestion/                        # Ingestion pipeline code
│   ├── pipeline.py                   # Main orchestration script
│   ├── steps/                        # Individual processing steps
│   │   ├── a_parsing.py             # Raw file parsing
│   │   ├── b_chunking.py            # Text chunking logic
│   │   ├── c_chunk_processing.py    # Content enhancement
│   │   ├── d_post_processing.py     # Validation & cleanup
│   │   ├── e_entity_extraction.py   # AI entity extraction
│   │   ├── f_cypher_query_generation.py # Graph query creation
│   │   └── g_ingestion.py           # Neo4j execution
│   ├── strategies/                   # Configurable strategies
│   │   ├── parsing/                 # Platform-specific parsers
│   │   │   ├── whatsapp_parser.py
│   │   │   ├── telegram_parser.py
│   │   └── chunking/                # Chunking algorithms
│   │       ├── message_count_strategy.py
│   │       └── time_based_strategy.py
│   └── chunking_strategies/          # Legacy chunking configs
│
├── query/                           # Query and search system
│   ├── pipeline.py                  # Main query orchestration
│   ├── steps/                       # Query processing steps
│   │   ├── a_hybrid_search.py      # Multi-modal search
│   │   ├── b_reranker.py           # Result ranking
│   │   ├── c_cypher_generation.py  # Dynamic graph queries
│   │   ├── d_query_expansion.py    # Query enhancement
│   │   └── e_context_coalescence.py # Result formatting
│   ├── preprocessors/              # Query preprocessing
│   │   ├── entity_extractor.py     # Query entity detection
│   │   ├── intent_classifier.py    # Query intent analysis
│   │   └── text_cleaner.py         # Query normalization
│   └── rerankers/                  # Reranking implementations
│       ├── cross_encoder_reranker.py
│       ├── cohere_reranker.py
│       └── jina_reranker.py
│
├── 🐳 docker-compose.yml            # Neo4j container setup
├── 📋 requirements.txt             # Python dependencies
├── ⚙️ .env.example                 # Configuration template
└── 📖 README.md                    # This documentation
```

### 🔄 Data Flow Explanation

#### Stage 1: Raw Data Ingestion
- **Input**: Manual upload of chat export files to `chats/` directory
- **Format**: Platform-specific (WhatsApp `.txt`, Telegram `.json`, Facebook various)
- **Purpose**: Raw conversation data as exported from messaging apps

#### Stage 2: Structured Parsing
- **Process**: Convert platform-specific formats to unified JSONL
- **Output**: `chats-parsed/` - Normalized message objects with timestamps, senders, content
- **Benefit**: Consistent data format regardless of source platform

#### Stage 3: Intelligent Chunking
- **Process**: Split long conversations into meaningful segments
- **Strategies**: Message count, time-based, or hybrid chunking
- **Output**: `chats-parsed-chunked/` - Conversation chunks with context preservation
- **Benefit**: Maintains conversation flow while enabling efficient processing

#### Stage 4: Content Enhancement
- **Process**: Add metadata, clean content, extract URLs, mentions
- **Output**: `chats-processed/` - Enriched chunks ready for AI processing
- **Benefit**: Improved data quality for downstream AI tasks

#### Stage 5: Entity Extraction
- **Process**: AI models identify people, organizations, concepts, dates
- **Output**: `entities/` - Structured entity data with confidence scores
- **Benefit**: Enables entity-aware search and relationship mapping

#### Stage 6: Graph Query Generation
- **Process**: AI generates Cypher queries to create knowledge graph
- **Output**: `cypher-queries/` - Neo4j commands for nodes and relationships
- **Benefit**: Transforms conversations into queryable graph structure

#### Stage 7: Database Ingestion
- **Process**: Execute Cypher queries against Neo4j database
- **Output**: `cypher-processed/` - Execution logs and success metrics
- **Result**: Persistent knowledge graph in Neo4j

#### Query Processing (Runtime)
- **Input**: Natural language queries
- **Process**: Hybrid search across vector, text, and graph indexes
- **Output**: Ranked, contextual results with conversation snippets

### 💾 Storage Architecture

#### Neo4j Graph Database
- **Nodes**: Messages, People, Organizations, Concepts, Dates
- **Relationships**: MENTIONS, REPLIES_TO, PARTICIPATES_IN, DISCUSSES
- **Indexes**: Full-text and vector indexes for performance

#### Vector Store (ChromaDB)
- **Purpose**: Semantic search over conversation content
- **Embeddings**: OpenAI or Ollama-generated text embeddings
- **Indexing**: HNSW for fast approximate nearest neighbor search

#### File System
- **Intermediate Results**: JSON/JSONL files for debugging and recovery
- **Configuration**: Environment variables and strategy configurations
- **Logs**: Processing logs and performance metrics

## ⚙️ Configuration Guide

Chat Text Processor is highly configurable through environment variables. This section documents all available options with examples and recommendations.

### 🔑 Core API Keys

#### Required Keys (Choose at least one provider)
| Variable | Description | Required | Get Key |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API for embeddings, chat, and Cypher generation | **Required** | [platform.openai.com](https://platform.openai.com/) |
| `GOOGLE_API_KEY` | Google Gemini for entity extraction | **Required** | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `GROQ_API_KEY` | Groq API for fast Cypher query generation | **Required** | [console.groq.com](https://console.groq.com/keys) |

#### Optional Keys (Enhanced functionality)
| Variable | Description | Use Case |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | Enhanced query output with multiple models | Advanced queries |
| `TWITTER_API_KEY` | Twitter API for URL expansion | Social media links |
| `TWITTER_BEARER_TOKEN` | Twitter Bearer token | Social media links |
| `COHERE_API_KEY` | Cohere reranking API | Cloud-based reranking |
| `JINA_API_KEY` | Jina AI reranking API | Cloud-based reranking |

### 🗄️ Database Configuration

#### Neo4j Connection
| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection string |
| `NEO4J_USER` | `neo4j` | Database username |
| `NEO4J_PASSWORD` | `password` | Database password |

**Neo4j Aura (Cloud) Example:**
```bash
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-aura-password
```

### 🤖 AI Model Configuration

#### Embedding Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `openai` | `openai` or `ollama` |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI embedding model |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL (if using ollama) |

#### Model Selection Examples
```bash
# OpenAI (recommended for quality)
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large

# Ollama (local, free)
EMBEDDING_PROVIDER=ollama
OLLAMA_URL=http://localhost:11434
```

#### Override Default Models
```bash
# Custom OpenAI model for Cypher generation
OPENAI_MODEL=gpt-4o-mini

# Custom Groq model (faster inference)
GROQ_MODEL=llama3-8b-8192
```

### 🎯 Search & Retrieval Configuration

#### Hybrid Search Weights
| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_WEIGHT` | `0.5` | Vector similarity importance (0.0-1.0) |
| `FULLTEXT_WEIGHT` | `0.3` | Full-text search importance (0.0-1.0) |
| `GRAPH_WEIGHT` | `0.2` | Graph relationships importance (0.0-1.0) |

#### Search Limits
| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_SEARCH_TOP_K` | `30` | Results per vector search query |
| `FULLTEXT_SEARCH_TOP_K` | `50` | Results per full-text search query |
| `GRAPH_SEARCH_TOP_K` | `20` | Results per graph search query |

#### Hybrid Search Toggle
```bash
# Enable/disable hybrid search (recommended: true)
USE_HYBRID_SEARCH=true
```

### 🔄 Reranking Configuration

#### Reranker Method Selection
| Variable | Default | Description |
|----------|---------|-------------|
| `RERANK_METHOD` | `cross-encoder` | `cross-encoder`, `zerank`, `cohere`, `jina` |

#### Reranker Model Examples
```bash
# Cross-encoder (best quality, local)
RERANK_METHOD=cross-encoder
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Zero-shot reranking (fast, local)
RERANK_METHOD=zerank
RERANK_MODEL=all-MiniLM-L6-v2

# Cohere (cloud, high quality)
RERANK_METHOD=cohere
RERANK_MODEL=rerank-english-v2.0

# Jina (cloud, multilingual)
RERANK_METHOD=jina
RERANK_MODEL=jina-reranker-v1-base-en
```

#### Reranking Parameters
| Variable | Default | Description |
|----------|---------|-------------|
| `RERANK_TOP_K_BEFORE` | `50` | Results to rerank from hybrid search |
| `RERANK_TOP_K_AFTER` | `20` | Final results after reranking |
| `RERANK_MIN_SCORE` | `0.0` | Minimum score threshold (0.0 = no filter) |

#### Advanced Reranking Options
```bash
# Boosting multipliers
RERANK_EXACT_MATCH_BOOST=1.3      # Boost exact keyword matches
RERANK_ENTITY_BOOST=1.2           # Boost entity matches
RERANK_RECENT_BOOST=1.1           # Boost recent results

# Feature toggles
RERANK_USE_QUERY_EXPANSION=true   # Use extracted entities
RERANK_BOOST_EXACT_MATCHES=true   # Enable exact match boosting
RERANK_BOOST_RECENT=false         # Enable recency boosting

# Performance settings
RERANK_BATCH_SIZE=32              # Processing batch size
RERANK_DEVICE=cpu                 # cpu or cuda
RERANK_MAX_LENGTH=512             # Maximum token length
RERANK_API_TIMEOUT=30             # API timeout (seconds)
```

### 📁 File Paths & Directories

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_DIR` | `.` | Base directory for all operations |
| `VERBOSE` | `false` | Enable verbose logging |

### 🔧 Performance Tuning

#### Memory & Resource Management
```bash
# Increase for large chat files
PYTHONUNBUFFERED=1

# GPU acceleration (if available)
CUDA_VISIBLE_DEVICES=0

# Threading options
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
```

#### Batch Processing
```bash
# Processing batch sizes
CHUNK_BATCH_SIZE=100
ENTITY_BATCH_SIZE=50
INGESTION_BATCH_SIZE=25
```

### 📊 Monitoring & Debugging

```bash
# Enable detailed logging
LOG_LEVEL=INFO
LOG_FORMAT=detailed

# Performance profiling
ENABLE_PROFILING=false
PROFILE_OUTPUT_DIR=profiles/
```

### 🛠️ Example Configuration Files

#### Minimal Production Setup
```bash
# .env.production
OPENAI_API_KEY=sk-production-key
GOOGLE_API_KEY=production-gemini-key
GROQ_API_KEY=production-groq-key
NEO4J_URI=neo4j+s://production-instance.neo4j.io
EMBEDDING_PROVIDER=openai
RERANK_METHOD=cross-encoder
USE_HYBRID_SEARCH=true
```

#### Local Development Setup
```bash
# .env.local
OPENAI_API_KEY=sk-development-key
GOOGLE_API_KEY=dev-gemini-key
GROQ_API_KEY=dev-groq-key
NEO4J_URI=bolt://localhost:7687
OLLAMA_URL=http://localhost:11434
EMBEDDING_PROVIDER=ollama
RERANK_METHOD=zerank
VERBOSE=true
```

#### High-Performance Setup
```bash
# .env.performance
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
RERANK_METHOD=cross-encoder
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
RERANK_DEVICE=cuda
VECTOR_SEARCH_TOP_K=100
RERANK_TOP_K_BEFORE=100
OMP_NUM_THREADS=8
```

## 🏗️ Architecture Overview

Chat Text Processor consists of two main pipelines: **Ingestion Pipeline** and **Query Pipeline**, working together to transform chat data into an intelligent knowledge system.

### 📥 Ingestion Pipeline (7 Steps)

The ingestion pipeline processes raw chat exports through a series of sophisticated transformations:

```
Raw Chat Files → Parsed Data → Chunks → Enriched Chunks → Entities → Cypher Queries → Neo4j Graph
     ↓              ↓             ↓            ↓                ↓             ↓              ↓
   chats/     chats-parsed/  chats-chunked/  chats-processed/  entities/  cypher-queries/  cypher-processed/
```

1. **📄 Parse** (`a_parsing.py`): Convert raw chat exports into structured JSONL format
2. **✂️ Chunk** (`b_chunking.py`): Split conversations into logical text segments
3. **🧹 Process** (`c_chunk_processing.py`): Clean and enhance chunks with metadata
4. **🔄 Post-Process** (`d_post_processing.py`): Final validation and sanitization
5. **🏷️ Extract Entities** (`e_entity_extraction.py`): Use AI to identify people, organizations, concepts
6. **⚡ Generate Cypher** (`f_cypher_query_generation.py`): Create graph database queries with AI
7. **💾 Ingest** (`g_ingestion.py`): Execute queries and store in Neo4j

### 🔍 Query Pipeline (5 Steps)

The query pipeline provides intelligent search and retrieval capabilities:

```
User Query → Hybrid Search → Rerank → Cypher Generation → Context Coalescence
     ↓             ↓              ↓            ↓                      ↓
  Preprocessing  Vector+Text+Graph  AI Reranking  Graph Queries    Final Results
```

1. **🔍 Hybrid Search** (`a_hybrid_search.py`): Combine vector similarity, full-text, and graph search
2. **🎯 Rerank** (`b_reranker.py`): Use AI models to improve result ranking
3. **⚡ Generate Cypher** (`c_cypher_generation.py`): Create complex graph queries from natural language
4. **🔗 Query Expansion** (`d_query_expansion.py`): Expand queries with extracted entities
5. **📋 Coalesce Context** (`e_context_coalescence.py`): Combine and format final results

### 🗂️ Data Flow & Storage

```
chats/                           # Raw exports (manual upload)
├── whatsapp/group-name/
├── telegram/chat-name/
└── facebook/conversation/

chats-parsed/                   # Structured JSONL
chats-parsed-chunked/          # Text segments
chats-processed/               # Enhanced chunks
cypher-queries/               # Generated queries
cypher-processed/             # Execution logs

vector_store/                  # Embeddings (ChromaDB)
├── collections/
└── embeddings/

ingestion/                     # Neo4j Graph Database
└── neo4j/                     # Nodes: Messages, People, Entities
                               # Relationships: MENTIONS, REPLIES_TO, etc.
```

## 🔍 Query Features

- **Hybrid Search**: Combines vector similarity, full-text, and graph relationships
- **Entity-Aware**: Finds mentions of people, organizations, and concepts
- **Contextual**: Maintains conversation context and relationships
- **Multi-Modal**: Supports different AI providers (OpenAI, Groq, Ollama)

## 🖥️ API & CLI Reference

### Ingestion Pipeline Commands

#### Main Command
```bash
python3 ingestion/pipeline.py [OPTIONS] CHAT_NAME
```

#### Positional Arguments
| Argument | Description | Example |
|----------|-------------|---------|
| `CHAT_NAME` | Path to chat directory relative to `chats/` | `whatsapp/family-group` |

#### Core Options
| Option | Default | Description |
|--------|---------|-------------|
| `--step` | `1` | Starting pipeline step (1-7) |
| `--model` | `groq/qwen/qwen3-32b` | AI model for Cypher generation |
| `--dry-run` | `false` | Show what would be done without executing |

#### Advanced Options
| Option | Default | Description |
|--------|---------|-------------|
| `--chunking-strategy` | `hybrid` | `line_count`, `message_count`, `time_based`, `hybrid` |
| `--force` | `false` | Overwrite existing processed data |
| `--verbose` | `false` | Enable detailed logging |
| `--batch-size` | `100` | Processing batch size for large files |

#### Examples
```bash
# Complete pipeline run
python3 ingestion/pipeline.py whatsapp/work-team

# Start from entity extraction (skip earlier steps)
python3 ingestion/pipeline.py telegram/tech-chat --step 5

# Use specific AI model
python3 ingestion/pipeline.py facebook/friend-chat --model openai/gpt-4o

# Dry run to preview processing
python3 ingestion/pipeline.py whatsapp/group --dry-run --verbose

# Custom chunking for long conversations
python3 ingestion/pipeline.py whatsapp/large-group --chunking-strategy time_based
```

### Query Pipeline Commands

#### Main Command
```bash
python3 query/pipeline.py [OPTIONS] "QUERY_TEXT"
```

#### Positional Arguments
| Argument | Description | Example |
|----------|-------------|---------|
| `QUERY_TEXT` | Natural language search query | `"What did Sarah say about the project?"` |

#### Output Options
| Option | Default | Description |
|--------|---------|-------------|
| `--format` | `text` | Output format: `text`, `json` |
| `--save` | - | Save results to file |
| `--verbose` | `false` | Enable detailed logging |

#### Search Control Options
| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `gpt-4o-mini` | Override default LLM model |
| `--no-vector` | `false` | Disable vector search |
| `--no-fulltext` | `false` | Disable full-text search |
| `--no-graph` | `false` | Disable graph search |
| `--top-k` | `20` | Number of results to return |

#### Advanced Query Options
| Option | Description |
|--------|-------------|
| `--entity-boost` | Boost results containing extracted entities |
| `--recent-boost` | Prioritize more recent messages |
| `--exact-match-boost` | Boost exact keyword matches |
| `--rerank-method` | Override reranking method |

#### Examples
```bash
# Basic search
python3 query/pipeline.py "Project deadline discussions"

# Advanced search with options
python3 query/pipeline.py "AI investments" --model gpt-4o --top-k 30 --verbose

# JSON output for programmatic use
python3 query/pipeline.py "Meeting summaries" --format json --save results.json

# Disable specific search types
python3 query/pipeline.py "Technical questions" --no-graph --no-vector

# Entity-focused search
python3 query/pipeline.py "What did John say?" --entity-boost
```

### Pipeline Step Details

#### Ingestion Steps
| Step | Name | Description | Typical Duration |
|------|------|-------------|------------------|
| 1 | Parse | Convert raw exports to JSONL | 10-60 seconds |
| 2 | Chunk | Split into text segments | 5-30 seconds |
| 3 | Process | Clean and enhance content | 10-45 seconds |
| 4 | Post-process | Final validation | 5-15 seconds |
| 5 | Extract Entities | AI entity identification | 2-10 minutes |
| 6 | Generate Cypher | Create graph queries | 5-20 minutes |
| 7 | Ingest | Execute against Neo4j | 1-5 minutes |

#### Query Steps
| Step | Name | Description | Typical Duration |
|------|------|-------------|------------------|
| 1 | Hybrid Search | Multi-modal retrieval | 100-500ms |
| 2 | Rerank | AI result ranking | 200-1000ms |
| 3 | Generate Cypher | Dynamic graph queries | 500-2000ms |
| 4 | Query Expansion | Entity-based enhancement | 100-300ms |
| 5 | Coalesce Context | Result formatting | 50-200ms |

### Configuration Files

#### Environment Variables
All configuration is done through `.env` file. Key variables:

```bash
# AI Providers
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
GROQ_API_KEY=...

# Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Search Settings
EMBEDDING_PROVIDER=openai
RERANK_METHOD=cross-encoder
VECTOR_WEIGHT=0.5
FULLTEXT_WEIGHT=0.3
GRAPH_WEIGHT=0.2
```

#### Chunking Strategies
Available in `ingestion/strategies/chunking/`:

- **`line_count_strategy.py`**: Fixed number of lines per chunk
- **`message_count_strategy.py`**: Fixed number of messages per chunk
- **`time_based_strategy.py`**: Time-window based chunking
- **`hybrid_strategy.py`**: Adaptive chunking based on content

#### Reranking Methods
Available in `query/rerankers/`:

- **`cross_encoder_reranker.py`**: Local transformer-based reranking
- **`cohere_reranker.py`**: Cohere cloud API reranking
- **`jina_reranker.py`**: Jina AI cloud reranking
- **`zerank_reranker.py`**: Zero-shot local reranking

### Programmatic Usage

#### Python API Example
```python
from ingestion.pipeline import ChatIngestionPipeline
from query.pipeline import ChatQueryPipeline

# Process a chat
ingestion = ChatIngestionPipeline()
result = ingestion.process_chat("whatsapp/group-chat", model="groq/qwen2.5-72b-instruct")

# Query the knowledge base
query = ChatQueryPipeline()
results = query.search("What was discussed about the budget?", top_k=20)
```

#### Docker Integration
```bash
# Run ingestion in container
docker run --rm -v $(pwd):/app chat-processor ingestion whatsapp/group --model groq/qwen2.5-72b-instruct

# Run query in container
docker run --rm -v $(pwd):/app chat-processor query "Search query here"
```

#### REST API (Future)
The project is designed for easy addition of REST endpoints for web integration.

## 🐛 Troubleshooting Guide

This section covers common issues, debugging techniques, and performance optimization tips.

### 🚨 Common Issues & Solutions

#### Authentication & API Errors

**"No auth credentials found" or "Invalid API key"**
```bash
# Check API key configuration
grep "API_KEY" .env

# Test OpenAI key
python3 -c "
import openai
client = openai.OpenAI()
try:
    client.models.list()
    print('✅ OpenAI API working')
except Exception as e:
    print(f'❌ OpenAI API error: {e}')
"

# Test Groq key
python3 -c "
import groq
client = groq.Groq()
try:
    client.models.list()
    print('✅ Groq API working')
except Exception as e:
    print(f'❌ Groq API error: {e}')
"
```

**"Google API quota exceeded"**
- Check Google AI Studio usage dashboard
- Consider upgrading to paid plan or switching to OpenAI
- Implement rate limiting in your scripts

#### Database Connection Issues

**"Failed to connect to Neo4j"**
```bash
# Check if Neo4j container is running
docker-compose ps

# View Neo4j logs
docker-compose logs neo4j

# Test connection manually
python3 -c "
from neo4j import GraphDatabase
try:
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
    with driver.session() as session:
        result = session.run('RETURN 1 as test')
        print('✅ Neo4j connection successful')
    driver.close()
except Exception as e:
    print(f'❌ Neo4j connection failed: {e}')
"

# Reset Neo4j if corrupted
docker-compose down -v
docker-compose up -d
```

**"Neo4j authentication failed"**
- Verify credentials in `.env` match Docker Compose
- Default Docker setup uses `neo4j`/`password`
- For Neo4j Aura, use the connection string provided

#### Processing Errors

**"No such file or directory: 'chats/whatsapp/group'"**
```bash
# Check directory structure
ls -la chats/
ls -la chats/whatsapp/

# Ensure chat export exists
find chats/ -name "*.txt" -o -name "*.json"

# Create directory structure if missing
mkdir -p chats/whatsapp/your-chat-name
# Place your chat export file here
```

**"Chunking failed: insufficient content"**
- Chat file might be too small or empty
- Check file size: `wc -l chats/whatsapp/your-chat/chat.txt`
- Verify export format matches expected structure

#### Memory & Performance Issues

**"Out of memory" during processing**
```bash
# Reduce batch sizes
export CHUNK_BATCH_SIZE=25
export ENTITY_BATCH_SIZE=10
export INGESTION_BATCH_SIZE=5

# Process in smaller chunks
python3 ingestion/pipeline.py whatsapp/large-chat --chunking-strategy line_count

# Monitor memory usage
python3 -m memory_profiler ingestion/pipeline.py your-chat
```

**Slow processing on large chats**
```bash
# Use faster models
export OPENAI_MODEL=gpt-3.5-turbo
export RERANK_METHOD=zerank

# Increase parallelism (if hardware allows)
export OMP_NUM_THREADS=8
export CHUNK_BATCH_SIZE=200

# Use GPU acceleration if available
export CUDA_VISIBLE_DEVICES=0
```

### 🔍 Debugging Techniques

#### Enable Detailed Logging
```bash
# Global verbose mode
export VERBOSE=true
python3 ingestion/pipeline.py whatsapp/debug-chat --model groq/qwen2.5-72b-instruct

# Pipeline-specific logging
export LOG_LEVEL=DEBUG
python3 query/pipeline.py "Debug query" --verbose

# Save logs to file
python3 ingestion/pipeline.py whatsapp/chat 2>&1 | tee processing.log
```

#### Inspect Intermediate Results
```bash
# Check parsing output
head -n 20 chats-parsed/whatsapp/chat.jsonl
jq '.messages[0]' chats-parsed/whatsapp/chat.jsonl

# Examine chunks
ls -la chats-parsed-chunked/whatsapp/chat/
cat chats-parsed-chunked/whatsapp/chat/chunk_001.json

# Review Cypher queries
head -n 50 cypher-queries/whatsapp/chat.cypher

# Check Neo4j data
echo "MATCH (n) RETURN labels(n), count(*) as count" | neo4j-admin database query neo4j
```

#### Query Debugging
```bash
# Test search components individually
python3 query/pipeline.py "Test query" --no-vector --no-graph  # Text-only
python3 query/pipeline.py "Test query" --no-fulltext --no-graph  # Vector-only
python3 query/pipeline.py "Test query" --no-vector --no-fulltext  # Graph-only

# Check reranking performance
time python3 query/pipeline.py "Complex query" --rerank-method zerank
time python3 query/pipeline.py "Complex query" --rerank-method cross-encoder
```

### 🚀 Performance Optimization

#### Hardware Optimization
```bash
# Use GPU for embeddings (if available)
export CUDA_VISIBLE_DEVICES=0
export RERANK_DEVICE=cuda

# Optimize thread usage
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)

# Increase memory limits
export NEO4J_dbms_memory_heap_initial__size=2G
export NEO4J_dbms_memory_heap_max__size=4G
```

#### Configuration Tuning
```bash
# High-performance settings
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
RERANK_METHOD=cross-encoder
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
VECTOR_SEARCH_TOP_K=100
RERANK_TOP_K_BEFORE=100

# Fast processing settings
EMBEDDING_PROVIDER=ollama
RERANK_METHOD=zerank
VECTOR_SEARCH_TOP_K=30
CHUNK_BATCH_SIZE=200

# Memory-efficient settings
VECTOR_SEARCH_TOP_K=20
RERANK_TOP_K_BEFORE=30
CHUNK_BATCH_SIZE=25
EMBEDDING_BATCH_SIZE=10
```

#### Database Optimization
```bash
# Neo4j performance tuning
# Add to neo4j/conf/neo4j.conf:
dbms.memory.heap.initial_size=2G
dbms.memory.heap.max_size=4G
dbms.memory.pagecache.size=2G

# Create indexes for better query performance
echo "
CREATE INDEX message_content IF NOT EXISTS FOR (m:Message) ON (m.content);
CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name);
CREATE FULLTEXT INDEX message_fts IF NOT EXISTS FOR (m:Message) ON EACH [m.content];
" | neo4j-admin database query neo4j
```

### 🔧 Advanced Troubleshooting

#### Reset and Recovery
```bash
# Clean restart (keeps data)
docker-compose restart neo4j
docker-compose logs -f neo4j

# Full reset (removes all data)
docker-compose down -v
docker-compose up -d

# Clear processed files (keep raw chats)
rm -rf chats-parsed/ chats-parsed-chunked/ chats-processed/
rm -rf cypher-queries/ cypher-processed/ entities/
rm -rf vector_store/
```

#### Network & Proxy Issues
```bash
# Test API connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models

# Use proxy if needed
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080

# Disable SSL verification (not recommended for production)
export REQUESTS_CA_BUNDLE=""
```

#### Platform-Specific Issues

**WhatsApp exports**
- Ensure export includes media captions if needed
- Check for special characters encoding
- Verify date format in export

**Telegram exports**
- Use "Export Chat History" from Telegram Desktop
- Include media if relevant to conversations
- Check for multi-byte character encoding

**Facebook Messenger**
- Use official export tools
- Ensure JSON format is valid
- Check for truncated long messages

### 📞 Getting Help

#### Diagnostic Information
When reporting issues, include:
```bash
# System information
uname -a
python3 --version
docker --version

# Package versions
pip list | grep -E "(neo4j|openai|torch|chromadb)"

# Configuration (mask sensitive data)
grep -v "KEY\|PASSWORD" .env

# Error logs
tail -n 50 processing.log
```

#### Support Resources
- **GitHub Issues**: [Create an issue](https://github.com/your-repo/issues) with diagnostic info
- **Documentation**: Check this README and inline code documentation
- **Community**: Join discussions for tips from other users

### ⚠️ Known Limitations & Workarounds

- **Large chats (>100MB)**: Split into smaller files or use cloud instance
- **Rate limiting**: Implement exponential backoff in scripts
- **Memory usage**: Monitor with `htop` or `nvidia-smi` during processing
- **GPU memory**: Cross-encoder reranking may require significant GPU RAM
- **Neo4j Aura**: Free tier has limitations on node/relationship counts

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help improve Chat Text Processor.

### 🚀 Ways to Contribute

- **🐛 Bug Reports**: Found a bug? [Open an issue](https://github.com/your-repo/issues/new?template=bug_report.md) with detailed steps to reproduce
- **💡 Feature Requests**: Have an idea? [Submit a feature request](https://github.com/your-repo/issues/new?template=feature_request.md)
- **📖 Documentation**: Help improve documentation, tutorials, or examples
- **🔧 Code Contributions**: Fix bugs, add features, or improve performance
- **🧪 Testing**: Add test cases or help with testing on different platforms

### 🛠️ Development Setup

#### Prerequisites for Contributors
```bash
# Install development dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Setup pre-commit for commit message formatting
pre-commit install --hook-type commit-msg
```

#### Development Workflow
```bash
# 1. Fork and clone the repository
git clone https://github.com/your-username/chat-text-processor.git
cd chat-text-processor

# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Setup development environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# 4. Make your changes
# - Follow the existing code style
# - Add tests for new features
# - Update documentation

# 5. Run tests and linting
pytest tests/
black --check .
isort --check-only .
flake8 .

# 6. Commit your changes
git add .
git commit -m "feat: add your feature description"

# 7. Push and create pull request
git push origin feature/your-feature-name
```

### 📋 Contribution Guidelines

#### Code Style
- **Python**: Follow PEP 8 with Black formatting
- **Imports**: Use `isort` for import sorting
- **Docstrings**: Use Google-style docstrings
- **Type Hints**: Add type annotations for function parameters and return values

#### Commit Messages
Follow conventional commit format:
```bash
type(scope): description

# Types: feat, fix, docs, style, refactor, test, chore
# Examples:
feat(parser): add support for Telegram animated stickers
fix(chunking): resolve memory leak in large file processing
docs(readme): update installation instructions
test(query): add integration tests for hybrid search
```

#### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_ingestion_pipeline.py

# Run with coverage
pytest --cov=ingestion --cov=query --cov-report=html

# Run performance tests
pytest tests/test_performance.py -v
```

#### Pull Request Process
1. **Update Documentation**: Ensure README and docstrings are updated
2. **Add Tests**: New features should include appropriate test coverage
3. **Update Changelog**: Add entry to `CHANGELOG.md` if applicable
4. **Squash Commits**: Clean up commit history before merging
5. **Review**: Request review from maintainers

### 🏗️ Architecture Decisions

#### Adding New Features
- **Parsers**: Add to `ingestion/strategies/parsing/`
- **Chunking Strategies**: Add to `ingestion/strategies/chunking/`
- **Query Processors**: Add to `query/preprocessors/`
- **Rerankers**: Add to `query/rerankers/`

#### Configuration Management
- All user-configurable options should be in `.env.example`
- Use environment variables for runtime configuration
- Document all new configuration options in this README

### 📊 Performance Considerations

When contributing performance improvements:
- **Benchmark**: Include before/after performance metrics
- **Memory**: Monitor memory usage impact
- **Scalability**: Test with various data sizes
- **Backwards Compatibility**: Ensure existing functionality isn't broken

### 🔒 Security Considerations

- **API Keys**: Never commit real API keys or secrets
- **Input Validation**: Validate all user inputs and file uploads
- **Dependencies**: Keep dependencies updated and audit for vulnerabilities
- **Data Privacy**: Handle user chat data responsibly

### 📈 Roadmap & Vision

We're working towards:
- **Web Interface**: Browser-based chat upload and querying
- **REST API**: Programmatic access to all features
- **Multi-language Support**: Beyond English conversations
- **Real-time Processing**: Live chat stream processing
- **Advanced Analytics**: Conversation insights and trends

## 🧪 Development

### Project Structure for Developers

```
├── 📁 ingestion/           # Data ingestion pipeline
│   ├── steps/             # Individual pipeline steps
│   └── strategies/        # Configurable algorithms
├── 📁 query/              # Query and search system
│   ├── steps/            # Query processing pipeline
│   ├── preprocessors/    # Query enhancement
│   └── rerankers/        # Result ranking algorithms
├── 📁 tests/              # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── fixtures/         # Test data
├── 📁 docs/               # Documentation
└── 📁 scripts/            # Development utilities
```

### 🐳 Development with Docker

```bash
# Development environment with hot reload
docker-compose -f docker-compose.dev.yml up --build

# Run tests in container
docker-compose -f docker-compose.dev.yml exec app pytest

# Access development database
docker-compose -f docker-compose.dev.yml exec neo4j cypher-shell
```

### 🔍 Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export VERBOSE=true

# Profile performance
python3 -m cProfile -s time ingestion/pipeline.py test-chat

# Memory profiling
python3 -m memory_profiler ingestion/pipeline.py test-chat

# Debug with breakpoint
python3 -c "
import pdb
from ingestion.pipeline import ChatIngestionPipeline
pdb.run('pipeline = ChatIngestionPipeline(); pipeline.process_chat(\"test\")')
"
```

### 📊 Testing Strategy

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test pipeline end-to-end
- **Performance Tests**: Benchmark processing speed and memory usage
- **Compatibility Tests**: Test across different Python versions and platforms

### 🚀 Release Process

1. **Version Bump**: Update version in `setup.py` and `__init__.py`
2. **Changelog**: Update `CHANGELOG.md` with new features and fixes
3. **Tests**: Ensure all tests pass on CI/CD
4. **Documentation**: Update README and any relevant docs
5. **Tag Release**: Create git tag and GitHub release
6. **Publish**: Upload to PyPI if applicable

## 📄 License

```
MIT License

Copyright (c) 2024 Chat Text Processor

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgments

- **OpenAI** for powerful language models and embeddings
- **Neo4j** for the excellent graph database platform
- **Google** for Gemini AI and related services
- **Groq** for fast inference capabilities
- **Chroma** for vector database functionality
- **Sentence Transformers** for local embeddings and reranking

## 📞 Support & Community

### Getting Help
- **📖 Documentation**: This README and inline code documentation
- **🐛 Bug Reports**: [GitHub Issues](https://github.com/your-repo/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions) for questions
- **📧 Email**: For security issues, email maintainers directly

### Community Guidelines
- Be respectful and inclusive
- Provide detailed information when reporting issues
- Help others when possible
- Follow the code of conduct

---

<p align="center">
  <em>Built with ❤️ for turning chat conversations into actionable knowledge</em>
</p>

<p align="center">
  <strong>⭐ Star this repository if you find it useful!</strong>
</p>
