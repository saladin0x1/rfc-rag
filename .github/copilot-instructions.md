# Copilot instructions for rfc-rag

This repo is a compact RAG pipeline over IETF RFCs following Cookiecutter Data Science (CCDS) conventions. It follows a simple, linear flow with Typer CLIs and Loguru logging.

Big picture architecture and data flow
- Data ingestion: `rfc_rag/dataset.py` parses RFC XML (`data/raw/xmlsource-all/*.xml`) and writes chunked JSONL to `data/processed/rfc_chunks.jsonl`.
  - Chunking: word-based, max_tokens=500, overlap=50 (`chunk_text`).
  - JSONL schema per line: `{ rfc_id, title, section, chunk_id, text, metadata:{ file, section_name, chunk_index } }`.
- Embedding: `rfc_rag/features.py` command `embed` uses sentence-transformers (default `all-MiniLM-L6-v2`) to produce `data/processed/rfc_embeddings.npz` with two arrays: `embeddings` (float32) and `chunks` (list of dicts mirroring JSONL).
- Indexing: `rfc_rag/modeling/train.py` builds a FAISS `IndexFlatIP` over L2-normalized embeddings; saves `models/rfc_faiss_index.bin` and metadata `models/rfc_metadata.pkl` (pickled list(chunks)).
- Querying: `rfc_rag/modeling/predict.py` commands `search` and `answer` embed a query with the same model and return top-k results from FAISS. The `answer` command uses Claude for RAG synthesis.
- Evaluation: `rfc_rag/evaluation.py` provides `evaluate_relevance` and `benchmark` commands for quality assessment and performance testing.
- Paths/config: `rfc_rag/config.py` defines `PROJ_ROOT`, `DATA_DIR`, `MODELS_DIR`, etc., and wires Loguru to tqdm. Most CLIs default to these paths.

Developer workflows (macOS/zsh)
- Run commands from the repo root: `rfc-rag/rfc-rag` (paths like `data/...` are relative to this directory).
- Environment: Python 3.10. Use uv if available.
  - Create venv: `make create_environment`
  - Install deps: `make requirements`
- Build pipeline (preferred):
  - `make data` → `make features` → `make train` → `make evaluate`
- Direct CLI usage (Typer subcommands are required):
  - Data: `python rfc_rag/dataset.py main --input-path data/raw/xmlsource-all --output-path data/processed/rfc_chunks.jsonl`
  - Embed: `python -m rfc_rag.features embed --input-path data/processed/rfc_chunks.jsonl --output-path data/processed/rfc_embeddings.npz --model all-MiniLM-L6-v2`
  - Train: `python -m rfc_rag.modeling.train main --embeddings-path data/processed/rfc_embeddings.npz --index-path models/rfc_faiss_index.bin --metadata-path models/rfc_metadata.pkl`
  - Search: `python -m rfc_rag.modeling.predict search "HTTP authentication methods" --k 5`
  - Answer: `python -m rfc_rag.modeling.predict answer "How does HTTP authentication work?" --k 5` (requires ANTHROPIC_API_KEY)
  - Evaluate: `python -m rfc_rag.evaluation evaluate_relevance --k 10`
  - Benchmark: `python -m rfc_rag.evaluation benchmark --k 5`
- Tests:
  - `make test` (requires `faiss-cpu` and `sentence-transformers`, and that the index/metadata exist).
  - Alternative to avoid threading issues: `python tests/standalone_test.py`.
- Docs (mkdocs): run from `rfc-rag/rfc-rag/docs` directory: `mkdocs serve` or `mkdocs build` (see `rfc-rag/rfc-rag/docs/README.md`).

Conventions and patterns specific to this repo
- **CCDS Structure**: Follows Cookiecutter Data Science project template with standard directories (`data/`, `models/`, `notebooks/`, `reports/`, etc.)
- CLIs are Typer apps; command names follow function names (`@app.command("embed")`, or the function name if unspecified). Always pass the subcommand.
- Similarity uses cosine via inner product on normalized vectors: embeddings are L2-normalized before index.add and query.
- Logging: use `loguru.logger.info/success`; tqdm is integrated (avoid raw prints in pipelines).
- Style/tooling: Python 3.10, Ruff configured in `pyproject.toml` (line-length 99; import sorting). Run `make lint`/`make format`.
- Data locations and defaults come from `config.py`; prefer Path defaults and these constants in new code.

External deps and integration notes
- Core libs: sentence-transformers, faiss-cpu, torch, numpy<2, tqdm, typer, loguru, anthropic.
- On macOS, install `faiss-cpu` (CPU-only) unless you have GPU support. If missing, CLIs log a helpful message and exit.
- Claude integration: set `ANTHROPIC_API_KEY` environment variable for the `answer` command.
- To reduce tokenizer/threading noise in tests/scripts, set: `TRANSFORMERS_VERBOSITY=error`, `TOKENIZERS_PARALLELISM=false`, optionally `TQDM_DISABLE=1`.

Extending the system (follow existing patterns)
- New data sources: emit the same JSONL schema so downstream embedding/indexing works unchanged.
- Changing the embedding model: keep consistent during index build and query (predict loads model by name).
- If you add per-chunk metadata, keep storing `chunks` alongside `embeddings` in the `.npz` and update `train.py`/`predict.py` accordingly.

Key files/directories
- `rfc_rag/dataset.py`, `rfc_rag/features.py`, `rfc_rag/modeling/train.py`, `rfc_rag/modeling/predict.py`, `rfc_rag/evaluation.py`
- `rfc_rag/config.py` (paths/logging), `Makefile` (end-to-end tasks), `tests/test_search.py` and `tests/standalone_test.py`

Known mismatches to watch for
- Makefile targets call Typer modules without subcommands (e.g., `features`/`train`). When running manually, include `embed`/`main` as shown above.

Questions or unclear areas to confirm
- Should Makefile be updated to include Typer subcommands (`embed`, `main`, `search`) or add default commands?
- Should `data/download_bulk.py` be the canonical fetcher, and what’s the expected layout under `data/raw/`?
