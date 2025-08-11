#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = rfc-rag
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) rfc_rag/dataset.py

## Generate embeddings (feature engineering stage)
.PHONY: features
features: data
	$(PYTHON_INTERPRETER) -m rfc_rag.features \
		--input-path data/processed/rfc_chunks.jsonl \
		--output-path data/processed/rfc_embeddings.npz

## Build FAISS vector index (modeling stage)
.PHONY: train
train: features
	$(PYTHON_INTERPRETER) -m rfc_rag.modeling.train \
		--embeddings-path data/processed/rfc_embeddings.npz \
		--index-path models/rfc_faiss_index.bin \
		--metadata-path models/rfc_metadata.pkl

## Evaluate search relevance quality
.PHONY: evaluate
evaluate: train
	$(PYTHON_INTERPRETER) -m rfc_rag.evaluation evaluate_relevance

## Benchmark RAG performance
.PHONY: benchmark
benchmark: train
	$(PYTHON_INTERPRETER) -m rfc_rag.evaluation benchmark


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
