# RFC RAG System

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A Retrieval-Augmented Generation (RAG) system for searching and answering questions about RFC (Request for Comments) documents using semantic search and Claude AI.

## Overview

This project implements a complete RAG pipeline that:
- Downloads and processes RFC documents from XML sources
- Creates semantic embeddings using sentence-transformers
- Builds a FAISS vector index for efficient similarity search
- Provides semantic search and Q&A capabilities via CLI
- Integrates with Claude AI for intelligent answer generation

## Features

- **Semantic Search**: Find relevant RFC sections using natural language queries
- **AI-Powered Q&A**: Get intelligent answers from Claude based on RFC context
- **Performance Metrics**: Detailed timing and relevance analysis
- **CLI Interface**: Easy-to-use command-line tools
- **CCDS Structure**: Following Cookiecutter Data Science conventions

## Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd rfc-rag

# Create and activate virtual environment (using uv - recommended)
uv venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install dependencies
uv pip install -r requirements.txt

# Alternative: using standard Python venv
# python -m venv .venv
# source .venv/bin/activate
# pip install -r requirements.txt

# Set up environment variables
cp .env.template .env
# Edit .env with your actual API keys and settings
export ANTHROPIC_API_KEY="your_claude_api_key"
```

### Usage

1. **Download and process RFC data**:
```bash
# Download RFC XML files
python data/download_bulk.py

# Process XML to chunks, generate embeddings, and build index
make data       # Process XML files to JSONL chunks
make features   # Generate embeddings 
make train      # Build FAISS index

# Or run the complete pipeline:
make data features train
```

2. **Search RFCs**:
```bash
# Semantic search
python -m rfc_rag.modeling.predict search "TCP congestion control"

# Get AI-powered answers
python -m rfc_rag.modeling.predict answer "How does TCP handle packet loss?"
```

## Commands

### Data Processing Pipeline
```bash
# 1. Process RFC XML files to chunks
python -m rfc_rag.dataset [OPTIONS]

# 2. Generate embeddings (requires embed subcommand)
python -m rfc_rag.features embed [OPTIONS]

# 3. Build FAISS index
python -m rfc_rag.modeling.train [OPTIONS]
```

### Search Command
```bash
python -m rfc_rag.modeling.predict search "your query" [OPTIONS]
```
Options:
- `--k`: Number of results (default: 5)
- `--model-name`: Embedding model (default: "all-MiniLM-L6-v2")

### Answer Command
```bash
python -m rfc_rag.modeling.predict answer "your question" [OPTIONS]
```
Options:
- `--k`: Number of context chunks (default: 5)
- `--claude-model`: Claude model (default: "claude-3-haiku-20240307")
- `--model-name`: Embedding model (default: "all-MiniLM-L6-v2")

## Requirements

- Python 3.10+
- FAISS (CPU version)
- sentence-transformers
- Anthropic Claude API access
- See `requirements.txt` for full dependencies

## Project Organization

```
├── LICENSE            <- Open-source license
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project
├── data
│   ├── external       <- Data from third party sources (RFC XML files)
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   │   ├── rfc_chunks.jsonl    <- Processed RFC text chunks
│   │   └── rfc_embeddings.npz  <- Precomputed embeddings
│   └── raw            <- The original, immutable data dump
│       └── xmlsource-all/      <- Downloaded RFC XML files
│
├── docs               <- Documentation using mkdocs
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│   ├── rfc_faiss_index.bin     <- FAISS vector index
│   └── rfc_metadata.pkl        <- RFC chunk metadata
│
├── notebooks          <- Jupyter notebooks for exploration and analysis
│
├── pyproject.toml     <- Project configuration file with package metadata
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
│
├── tests/             <- Unit tests
│
└── rfc_rag/          <- Source code for use in this project
    │
    ├── __init__.py             <- Makes rfc_rag a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling/               <- Scripts to train models and make predictions
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Architecture

The system follows a standard RAG architecture:

1. **Data Ingestion**: RFC XML files are downloaded and parsed
2. **Text Processing**: Documents are chunked into semantic segments
3. **Embedding Generation**: Text chunks are converted to vector embeddings
4. **Vector Indexing**: FAISS index is built for efficient similarity search
5. **Query Processing**: User queries are embedded and matched against the index
6. **Context Retrieval**: Top-K most relevant chunks are retrieved
7. **Answer Generation**: Claude AI generates answers based on retrieved context

## Performance

The system provides detailed metrics including:
- End-to-end response times
- Individual component timing (embedding, search, API calls)
- Relevance scores and categorization
- Source attribution with similarity scores

## Contributing

This project follows [Cookiecutter Data Science (CCDS)](https://cookiecutter-data-science.drivendata.org/) conventions:
- Use `make` commands for common tasks
- Keep data processing scripts in `data/`
- Model training in `rfc_rag/modeling/train.py`
- Inference in `rfc_rag/modeling/predict.py`
- Tests in `tests/`
- Documentation in `docs/`

## License

See LICENSE file for details.

