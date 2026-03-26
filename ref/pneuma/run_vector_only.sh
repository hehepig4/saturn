#!/bin/bash
#
# Run Vector-only evaluation (alpha=0.0) with LLM rerank
# Uses GPU 0
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output_vector_only"
mkdir -p "${OUTPUT_DIR}"

# Use GPU 0
export CUDA_VISIBLE_DEVICES="0"

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
echo "Running Vector-only evaluation (alpha=0.0)"
echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"
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
        --alpha 0.0 \
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
echo "Vector-only evaluation complete"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="
