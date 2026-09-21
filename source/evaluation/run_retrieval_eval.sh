#!/usr/bin/env bash
# Evaluate cached UPO-aligned queries with BM25, vector or hybrid retrieval.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/source${PYTHONPATH:+:$PYTHONPATH}"

DATASETS="adventure_works bird chembl chicago fetaqa fetaqapn openwikitable public_bi"
RETRIEVERS="bm25 vector hybrid"
LLM="local"
NUM_QUERIES=-1
TOP_K=100
while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets) DATASETS="$2"; shift 2 ;;
        --retriever) RETRIEVERS="$2"; shift 2 ;;
        --llm) LLM="$2"; shift 2 ;;
        --num-queries|-n) NUM_QUERIES="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --gpu) export CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
        --help|-h)
            printf '%s\n' 'Usage: run_retrieval_eval.sh [--datasets "d1 d2"] [--retriever "bm25 vector hybrid"] [--llm local] [-n N] [--top-k K] [--gpu N]'
            exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 1 ;;
    esac
done
cd "$PROJECT_ROOT"
for dataset in $DATASETS; do
    for retriever in $RETRIEVERS; do
        python -m evaluation.runners.hyde_retrieval -d "$dataset" \
            --llm "$LLM" --retriever "$retriever" --compare-combined \
            --num-queries "$NUM_QUERIES" --top-k "$TOP_K"
    done
done
