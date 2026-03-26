#!/bin/bash
#
# Run BM25-only evaluation (alpha=1.0) with LLM rerank
# No GPU needed - runs on CPU (no embedding model loaded)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output_bm25_only"
mkdir -p "${OUTPUT_DIR}"

# Force CPU mode (no GPU needed for BM25-only)
export CUDA_VISIBLE_DEVICES=""

# LLM rerank settings
RERANK_MODE="llm"
RERANK_TOP_K=10
OPENAI_URL="http://10.120.47.91:8000/v1"
LLM_MODEL="Qwen3-Next-80B-A3B-Instruct"

DATASETS=(
    "adventure_works"
    "bird"
    "chembl"
    "chicago"
    "public_bi"
    "fetaqapn"
    "fetaqa"
)

echo "=========================================="
echo "Running BM25-only evaluation (alpha=1.0)"
echo "LLM Rerank: ${RERANK_MODE} (top-k=${RERANK_TOP_K})"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

for ds in "${DATASETS[@]}"; do
    echo ""
    echo ">>> Evaluating: ${ds}"
    
    "${SCRIPT_DIR}/evaluate.sh" \
        --dataset "${ds}" \
        --skip-convert \
        --skip-summaries \
        --skip-index \
        --alpha 1.0 \
        --rerank "${RERANK_MODE}" \
        --rerank-top-k "${RERANK_TOP_K}" \
        --openai-url "${OPENAI_URL}" \
        --openai-model "${LLM_MODEL}" \
        2>&1 | tee "${OUTPUT_DIR}/${ds}_eval.log"
    
    # Copy result to output dir
    if [ -f "${SCRIPT_DIR}/output/${ds}_results.json" ]; then
        cp "${SCRIPT_DIR}/output/${ds}_results.json" "${OUTPUT_DIR}/"
    fi
    
    echo ">>> Done: ${ds}"
done

echo ""
echo "=========================================="
echo "BM25-only evaluation complete"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="
