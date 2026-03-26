# Baseline Reproduction Interface

This directory contains scripts and patches for reproducing the three baseline systems evaluated in the paper. Each baseline is adapted to work with Saturn's unified benchmark data format.

## Directory Structure

```
ref/
├── birdie/        # BIRDIE (DSI-based table discovery)
├── pneuma/        # Pneuma (LLM summary + hybrid retrieval)
└── solo/          # Solo (dense passage retrieval)
```

## Common Workflow

All three baselines follow a consistent pipeline:

1. **Setup**: `setup.sh` — Clone original repo, create conda env, apply patches
2. **Data Conversion**: `scripts/convert_unified_to_*.py` — Convert from Saturn's unified format
3. **Evaluation**: `evaluate.sh` — Run the full pipeline (index → train → evaluate)
4. **Batch Run**: `run_all_datasets.sh` — Evaluate across all 7 benchmark datasets

## Baselines

### Birdie

DSI (Differentiable Search Index) approach with hierarchical clustering and constrained beam search. Our modifications include:
- Multi-answer query support (original only supports single-answer)
- BGE-M3 embeddings (consistent with Saturn)
- vLLM-based query generation (replaces local Transformer inference)
- LoRA fine-tuning for dataset-specific query generation
- Extended beam search (K=100) for Hit@100 evaluation

Key patches in `birdie/patches/` modify the original BIRDIE code for the above features.

### Pneuma

LLM-based table summarization with hybrid BM25+vector retrieval. Our modifications include:
- Unified data format conversion
- Faithful reproduction of the official summarization prompt
- ChromaDB + BM25 hybrid retrieval with configurable alpha
- LLM reranking support
- Multiple retrieval mode experiments (BM25-only, vector-only, hybrid+rerank)

### Solo

Dense passage retrieval with bi-encoder training. Our modifications include:
- FAISS GPU acceleration with K-value limiting (K < 2048)
- Single-process training fix for multi-GPU environments
- Auto-continue training support
- Passage count limiting for large datasets
- Table-level metric aggregation (P@K from passage-level predictions)

Key patches in `solo/patches/` are applied via `setup_solo.sh`.

## Prerequisites

Each baseline requires its own conda environment. See the respective `setup.sh` / `setup_solo.sh` scripts for environment creation.

All baselines expect benchmark data in Saturn's unified format at `data/benchmark/unified/{dataset}/`. See the main README for data preparation instructions.
