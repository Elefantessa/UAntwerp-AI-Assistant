# 🎓 UAntwerp Academic RAG System

> **Enterprise-Grade Retrieval-Augmented Generation Pipeline for University Programme Information**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.x-green.svg)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange.svg)](https://trychroma.com)

A production-ready RAG (Retrieval-Augmented Generation) system designed to provide accurate, context-aware answers about the University of Antwerp's Master in Computer Science programmes. Built with a focus on **modularity**, **scalability**, and **maintainability**.

---

## 🌟 Key Highlights

| Feature | Description |
|---------|-------------|
| 🏗️ **Modular Architecture** | Clean separation of concerns across 8 distinct modules |
| ⚡ **High Performance** | Async web scraping, GPU-accelerated embeddings, batch processing |
| 🎯 **Advanced Retrieval** | MMR diversity search + Cross-Encoder reranking for precision |
| 🧠 **Intelligent Processing** | Entity extraction, intent classification, confidence scoring |
| 🔄 **LangGraph Pipeline** | State-machine orchestration for complex workflows |
| 📊 **Multi-Factor Confidence** | 5-factor scoring with semantic coherence validation |

---

## 🧠 How It Works: Question-Answering Pipeline

The system processes queries through a sophisticated multi-stage pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                           │
│                 "What are the admission requirements?"                      │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: QUERY ANALYSIS                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • Intent Classification (FACTUAL, PROCEDURAL, COMPARISON, etc.)     │   │
│  │ • Entity Extraction (programmes, courses, lecturers, dates)         │   │
│  │ • Keyword Expansion & Query Refinement                              │   │
│  │ • Metadata Filter Generation                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: INTELLIGENT RETRIEVAL                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • MMR Search (Maximal Marginal Relevance) for diversity             │   │
│  │ • Semantic similarity via SFR-Embedding-Mistral (4096-dim)          │   │
│  │ • Metadata filtering by programme/page_type                         │   │
│  │ • Fetch k=100 candidates → Return top k=50                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: CROSS-ENCODER RERANKING                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • ms-marco-MiniLM cross-encoder for query-document scoring          │   │
│  │ • Re-scores all candidates with full attention                      │   │
│  │ • Selects top-12 most relevant documents                            │   │
│  │ • Provides reranking confidence scores                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: CONTEXT MANAGEMENT                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ • Token budget management (2000 tokens max)                         │   │
│  │ • Source de-duplication                                             │   │
│  │ • Context expansion for completeness                                │   │
│  │ • Priority ranking by relevance score                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: ANSWER GENERATION                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STRICT MODE (Primary):                                              │   │
│  │ • JSON-structured output                                            │   │
│  │ • Answers ONLY from retrieved context                               │   │
│  │ • Explicit "I don't know" for missing information                   │   │
│  │                                                                      │   │
│  │ FLEXIBLE MODE (Fallback):                                           │   │
│  │ • Can use general knowledge                                         │   │
│  │ • Clear distinction of sourced vs. general info                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: CONFIDENCE SCORING                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Multi-Factor Confidence Calculation:                                │   │
│  │                                                                      │   │
│  │   Reranking Score ────────────────────── 30%                        │   │
│  │   Entity Match ───────────────────────── 20%                        │   │
│  │   Semantic Coherence (LLM-based) ─────── 20%                        │   │
│  │   Source Diversity ───────────────────── 15%                        │   │
│  │   Context Completeness ───────────────── 15%                        │   │
│  │   ─────────────────────────────────────────                         │   │
│  │   Final Confidence Score ─────────────── 100%                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RESPONSE                                             │
│  {                                                                          │
│    "answer": "Detailed answer with source citations...",                    │
│    "confidence": 0.85,                                                      │
│    "sources": ["url1", "url2"],                                            │
│    "contexts": [...]                                                        │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture

### System Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE                                │
│                    (Flask API + Web Chat Interface)                     │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           API LAYER                                     │
│         ┌─────────────┐    ┌──────────────┐    ┌────────────────┐      │
│         │ /api/query  │    │ /api/health  │    │ /api/chat      │      │
│         └─────────────┘    └──────────────┘    └────────────────┘      │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         SERVICE LAYER                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      RAG SERVICE                                 │   │
│  │    (LangGraph State Machine Orchestration)                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────┐          ┌───────────────────────────────┐   │
│  │    Ollama Service    │          │      Query Processor          │   │
│  │   (LLM Interface)    │          │  (Intent + Entity Extraction) │   │
│  └──────────────────────┘          └───────────────────────────────┘   │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          CORE LAYER                                     │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────────────┐   │
│  │   Retrieval    │  │   Generation   │  │      Processors         │   │
│  │ ┌────────────┐ │  │ ┌────────────┐ │  │ ┌─────────────────────┐ │   │
│  │ │VectorStore │ │  │ │StrictGen   │ │  │ │QueryProcessor       │ │   │
│  │ │Reranker    │ │  │ │FlexibleGen │ │  │ │ConfidenceCalculator │ │   │
│  │ │Expander    │ │  │ └────────────┘ │  │ └─────────────────────┘ │   │
│  │ └────────────┘ │  └────────────────┘  └─────────────────────────┘   │
│  └────────────────┘                                                     │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                        │
│  ┌──────────────────────────────┐   ┌───────────────────────────────┐  │
│  │       ChromaDB               │   │      Ollama (LLM)             │  │
│  │  (Vector Embeddings Store)   │   │   llama3.1:latest             │  │
│  └──────────────────────────────┘   └───────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
pipline/
├── main.py                      # Application entry point
├── run_indexing.py              # Indexing pipeline CLI
├── run_evaluation.py            # 📈 RAGAS evaluation CLI
├── requirements.txt             # Dependencies
│
├── config/                      # ⚙️ Configuration
│   ├── settings.py              # Dataclass-based settings
│   └── logging_config.py        # Logging setup
│
├── core/                        # 🧠 Core Business Logic
│   ├── models/                  # Data models
│   │   ├── state.py             # LangGraph state schema
│   │   ├── response.py          # Response dataclasses
│   │   └── entities.py          # Query entities
│   │
│   ├── processors/              # Processing logic
│   │   ├── query_processor.py   # Intent & entity extraction
│   │   └── confidence_calculator.py  # Multi-factor scoring
│   │
│   ├── retrieval/               # Retrieval components
│   │   ├── vector_store.py      # ChromaDB manager
│   │   ├── reranker.py          # Cross-encoder reranking
│   │   └── context_expander.py  # Token budget management
│   │
│   └── generation/              # Answer generation
│       ├── base_generator.py    # Abstract base class
│       ├── strict_generator.py  # Context-only answers
│       └── flexible_generator.py # General knowledge fallback
│
├── evaluation/                  # 📈 RAGAS Evaluation
│   ├── __init__.py              # Module exports
│   ├── config.py                # Evaluation configuration
│   ├── ragas_evaluator.py       # RAGAS metrics wrapper
│   └── tester.py                # RAG testing framework
│
├── services/                    # 🔧 Service Layer
│   ├── rag_service.py           # LangGraph orchestrator
│   └── ollama_service.py        # LLM client wrapper
│
├── api/                         # 🌐 Web API
│   ├── app.py                   # Flask factory
│   ├── routes/
│   │   ├── chat.py              # Query endpoints
│   │   └── health.py            # Health checks
│   └── templates/
│       └── chat.html            # Web interface
│
├── utils/                       # 🛠️ Utilities
│   ├── json_utils.py            # JSON parsing
│   └── text_utils.py            # Text processing
│
└── indexing/                    # 📥 Data Ingestion
    ├── scraper/                 # Web scraping
    │   ├── config.py
    │   ├── url_utils.py
    │   ├── html_cleaner.py
    │   ├── markdown_converter.py
    │   └── scraper.py
    │
    ├── chunker/                 # Text chunking
    │   ├── config.py
    │   ├── token_estimator.py
    │   ├── text_utils.py
    │   └── chunker.py
    │
    └── ingestor/                # ChromaDB ingestion
        ├── config.py
        ├── metadata_utils.py
        ├── device_planner.py
        ├── embeddings.py
        └── ingestor.py
```

---

## 🎯 Technical Strengths

### 1. **Production-Ready Architecture**
- Clean separation of concerns following SOLID principles
- Dependency injection for testability
- Dataclass-based configuration for type safety
- Comprehensive error handling with graceful fallbacks

### 2. **Advanced NLP Pipeline**
- **State-of-the-Art Embeddings**: Salesforce SFR-Embedding-Mistral (4096 dimensions)
- **Two-Stage Retrieval**: MMR search + Cross-encoder reranking
- **Entity-Aware Processing**: Automatic extraction of programmes, courses, lecturers
- **Intent Classification**: FACTUAL, PROCEDURAL, COMPARISON, EXPLORATORY, SPECIFIC

### 3. **Robust Confidence Scoring**
```python
confidence = (
    rerank_score * 0.30 +      # Cross-encoder relevance
    entity_match * 0.20 +       # Query-answer entity overlap
    semantic_coherence * 0.20 + # LLM-based validation
    source_diversity * 0.15 +   # Multiple source agreement
    context_completeness * 0.15 # Coverage of query aspects
)
```

### 4. **Scalable Indexing Pipeline**
- Async web scraping with concurrency control
- Robots.txt compliance
- Content deduplication (per-programme)
- GPU-accelerated batch embedding
- Token-aware hybrid chunking

### 5. **Developer Experience**
- Unified CLI for all operations
- Comprehensive logging
- Modular design for easy extension
- Well-documented codebase

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Ollama with llama3.1 model
- CUDA-capable GPU (recommended)

### Installation

```bash
# Clone and setup
pip install -r web_pipline/pipline/requirements.txt

# Start Ollama (if not running)
ollama serve &
ollama pull llama3.1
```

### Run Indexing Pipeline

```bash
cd web_pipline/pipline

# Full pipeline (recommended for first run)
python run_indexing.py --full

# With custom settings
python run_indexing.py --full --max-pages 50 --recreate
```

### Start API Server

```bash
python main.py \
  --persist-dir /path/to/chroma_db \
  --collection uantwerp_cs_web \
  --port 5006
```

### Test the System

```bash
# Health check
curl http://localhost:5006/api/health

# Query example
curl -X POST http://localhost:5006/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the admission requirements for Data Science?"}'
```

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Indexing Speed** | 34 pages → 92 chunks in 126s |
| **Query Latency** | < 5 seconds (with GPU) |
| **Embedding Model** | SFR-Embedding-Mistral (4096-dim) |
| **Context Window** | 2000 tokens |
| **Confidence Range** | 0.0 - 1.0 |

---

## 🛠️ Configuration Reference

### Environment Settings (`config/settings.py`)

| Config | Parameter | Default |
|--------|-----------|---------|
| **Model** | `ollama_model` | llama3.1:latest |
| **Model** | `embed_model` | SFR-Embedding-Mistral |
| **RAG** | `k` | 50 documents |
| **RAG** | `token_budget` | 2000 tokens |
| **Indexing** | `max_pages_per_seed` | 100 |
| **Indexing** | `target_tokens` | 350 per chunk |

---

## 📈 RAGAS Evaluation

The system includes comprehensive evaluation capabilities using **RAGAS** (Retrieval Augmented Generation Assessment) framework with local Ollama models.

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Answer Relevancy** | How relevant the answer is to the question |
| **Context Precision** | Proportion of relevant context chunks |
| **Context Recall** | How well the retrieved context covers the ground truth |
| **Faithfulness** | How grounded the answer is in the retrieved context |
| **Answer Correctness** | Semantic similarity to ground truth answers |

### Recommended Models for Evaluation

| Model | Size | JSON Quality | Speed | Notes |
|-------|------|--------------|-------|-------|
| **qwen2.5:14b** ⭐ | 9GB | ⭐⭐⭐⭐⭐ | Medium | Best balance - all metrics work |
| **qwen2.5:7b** | 4.7GB | ⭐⭐⭐⭐ | Fast | Good for quick evaluations |
| **mistral:7b** | 4.1GB | ⭐⭐⭐⭐ | Fast | Reliable JSON output |
| **gemma2:9b** | 5.4GB | ⭐⭐⭐⭐⭐ | Medium | From Google |
| **llama3.1:8b** | 4.9GB | ⭐⭐⭐ | Fast | Some metrics may fail |
| **gpt-oss:latest** | 13GB | ⭐⭐⭐⭐ | Slow | May timeout on some metrics |

### Running Evaluation

```bash
cd web_pipline/pipline

# Make sure the RAG API is running first
python main.py --persist-dir ../data/db/unified_chroma_db --port 5007

# Run evaluation with recommended model (in another terminal)
python run_evaluation.py \
  --questions /web_pipline/data/evaluation/sample_questions.json \
  --provider ollama \
  --llm-model qwen2.5:14b \
  --api-url http://127.0.0.1:5007

# Generate sample questions file
python run_evaluation.py --generate-sample
```

### Evaluation Results Example

```
📈 RAGAS Scores:
  answer_relevancy: 0.5275
  context_precision: 0.3000
  context_recall: 0.4000
  faithfulness: 0.6867
  answer_correctness: 0.3982

  📊 Average: 0.4625
```

### Installing Evaluation Models

```bash
# Recommended model (best balance)
ollama pull qwen2.5:14b

# Alternative models
ollama pull qwen2.5:7b
ollama pull mistral:7b-instruct
ollama pull gemma2:9b

# Required for embeddings
ollama pull nomic-embed-text
```

### Evaluation Configuration

The evaluation uses:
- **LLM**: Configurable via `--llm-model` (default: `gpt-oss:latest`)
- **Embeddings**: `nomic-embed-text` via Ollama
- **Provider**: `ollama` (local) or `openai` (API)

### Output Files

Results are saved to `/web_pipline/data/evaluation/`:
- `rag_results_TIMESTAMP.jsonl` - Detailed RAG responses
- `ragas_scores_TIMESTAMP.json` - RAGAS metric scores
- `evaluation_report_TIMESTAMP.md` - Human-readable report

---

## 📄 License

University of Antwerp - Master in Computer Science Project

---

## 👤 Author

**Hala Alramli**
