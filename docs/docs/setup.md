# Setup Guide

## Prerequisites

- Python 3.10 or higher
- Git
- Anthropic Claude API key

## Installation Steps

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd rfc-rag
```

### 2. Set Up Python Environment

Using pip:
```bash
pip install -r requirements.txt
```

Using conda:
```bash
conda create -n rfc-rag python=3.10
conda activate rfc-rag
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Required for Claude AI integration
ANTHROPIC_API_KEY=your_claude_api_key_here
```

### 4. Download and Process RFC Data

```bash
# Download RFC XML files (this may take a while)
python data/download_bulk.py

# Process the data and create embeddings
python -m rfc_rag.modeling.train
```

This will:
- Parse XML files and extract text chunks
- Generate embeddings using sentence-transformers
- Build a FAISS vector index
- Save processed data to `data/processed/` and models to `models/`

## Troubleshooting

### macOS Segmentation Fault

If you encounter segmentation faults on macOS, this is a known issue with torch and sentence-transformers. The code includes a fix:

```python
import torch
torch.set_num_threads(1)
```

This is already included in the prediction scripts.

### Missing Dependencies

If you get import errors, ensure all requirements are installed:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `faiss-cpu` for vector search
- `sentence-transformers` for embeddings
- `anthropic` for Claude API
- `typer` for CLI
- `loguru` for logging

### API Key Issues

Make sure your Anthropic API key is:
1. Valid and active
2. Set in the `.env` file
3. Has sufficient credits

### Large File Issues

If you have issues with large files (embeddings, FAISS index), they are excluded from git by `.gitignore`. You need to regenerate them:

```bash
python -m rfc_rag.modeling.train
```

## Verification

Test your installation:

```bash
# Test search functionality
python -m rfc_rag.modeling.predict search "TCP protocol"

# Test answer functionality (requires API key)
python -m rfc_rag.modeling.predict answer "What is TCP?"
```

If both commands work without errors, your setup is complete!
