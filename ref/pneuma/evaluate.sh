#!/bin/bash
#
# Pneuma Evaluation Script
#
# Usage:
#   ./evaluate.sh --dataset chembl [options]
#
# Example:
#   ./evaluate.sh --dataset fetaqapn --skip-summaries
#   ./evaluate.sh --dataset adventure_works --skip-convert --skip-summaries --skip-index
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF_DIR="${SCRIPT_DIR}/../.."
SATURN_ROOT="${REF_DIR}/.."

# Default paths
UNIFIED_BASE="${SATURN_ROOT}/data/benchmark/unified"
WORK_DIR="${REF_DIR}/pneuma_work"
OUTPUT_DIR="${SCRIPT_DIR}/output"

# Model paths
LOCAL_MODEL_DIR="${SATURN_ROOT}/model"
EMBED_MODEL="${LOCAL_MODEL_DIR}/bge-m3"

# OpenAI API configuration (for LLM summaries)
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://10.120.47.91:8000/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-token-abc123}"
OPENAI_MODEL="${OPENAI_MODEL:-Qwen3-Next-80B-A3B-Instruct}"

# Default values
DATASET=""
SKIP_CONVERT=false
SKIP_SUMMARIES=false
SKIP_INDEX=false
TOP_K="1,3,5,10,20,100"
ALPHA=0.5
RERANK_MODE="none"
RERANK_TOP_K=""
LLM_RERANK_MODEL=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --unified-dir)
            UNIFIED_BASE="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --skip-convert)
            SKIP_CONVERT=true
            shift
            ;;
        --skip-summaries)
            SKIP_SUMMARIES=true
            shift
            ;;
        --skip-index)
            SKIP_INDEX=true
            shift
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --rerank)
            RERANK_MODE="$2"
            shift 2
            ;;
        --rerank-top-k)
            RERANK_TOP_K="$2"
            shift 2
            ;;
        --llm-rerank-model)
            LLM_RERANK_MODEL="$2"
            shift 2
            ;;
        --openai-url)
            OPENAI_BASE_URL="$2"
            shift 2
            ;;
        --openai-model)
            OPENAI_MODEL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --dataset DATASET [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --dataset NAME         Dataset name (chembl, adventure_works, chicago,"
            echo "                         public_bi, fetaqa, fetaqapn, bird)"
            echo ""
            echo "Data options:"
            echo "  --unified-dir PATH     Unified benchmark directory"
            echo "  --work-dir PATH        Working directory for intermediate files"
            echo ""
            echo "Pipeline options:"
            echo "  --skip-convert         Skip data conversion (use existing)"
            echo "  --skip-summaries       Skip summary generation (use existing)"
            echo "  --skip-index           Skip index building (use existing)"
            echo ""
            echo "Evaluation options:"
            echo "  --top-k K1,K2,...      Hit@K values to report (default: 1,3,5,10,20,100)"
            echo "  --alpha ALPHA          BM25 weight (default: 0.5)"
            echo "  --rerank MODE          Reranking mode: none, cosine, direct, llm (default: none)"
            echo "  --rerank-top-k K       Number of tables to rerank (default: max of top-k)"
            echo "  --llm-rerank-model     Path to LLM model for llm rerank mode"
            echo ""
            echo "LLM options:"
            echo "  --openai-url URL       OpenAI-compatible API URL"
            echo "  --openai-model MODEL   LLM model name"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Validate arguments
if [ -z "$DATASET" ]; then
    echo -e "${RED}Error: --dataset is required${NC}"
    echo "Use --help for usage information"
    exit 1
fi

# Setup directories
DATASET_WORK_DIR="${WORK_DIR}/${DATASET}"
SUMMARIES_DIR="${DATASET_WORK_DIR}/summaries"
INDICES_DIR="${DATASET_WORK_DIR}/indices"
CONVERTED_DIR="${DATASET_WORK_DIR}/converted"

mkdir -p "${DATASET_WORK_DIR}"
mkdir -p "${SUMMARIES_DIR}"
mkdir -p "${INDICES_DIR}"
mkdir -p "${CONVERTED_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}    Pneuma Evaluation Pipeline${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Dataset: ${DATASET}"
echo "Unified data: ${UNIFIED_BASE}/${DATASET}"
echo "Work directory: ${DATASET_WORK_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Activate conda environment
echo -e "${GREEN}Activating conda environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate pneuma || {
    echo -e "${RED}Error: pneuma environment not found${NC}"
    exit 1
}

# Export OpenAI configuration
export OPENAI_BASE_URL
export OPENAI_API_KEY

# ============================================
# Step 1: Convert data to Pneuma format
# ============================================
echo ""
echo -e "${GREEN}[Step 1/4] Converting data to Pneuma format...${NC}"

if [ "$SKIP_CONVERT" = true ] && [ -f "${CONVERTED_DIR}/tables/done.flag" ]; then
    echo -e "${YELLOW}Skipping conversion (--skip-convert flag set)${NC}"
else
    python3 "${SCRIPT_DIR}/scripts/convert_unified_to_pneuma.py" \
        --dataset "${DATASET}" \
        --unified-dir "${UNIFIED_BASE}" \
        --output-dir "${CONVERTED_DIR}"
    
    touch "${CONVERTED_DIR}/tables/done.flag"
    echo -e "${GREEN}Conversion complete${NC}"
fi

# Check table count
TABLE_COUNT=$(ls -1 "${CONVERTED_DIR}/tables/"*.csv 2>/dev/null | wc -l || echo "0")
QUERY_COUNT=$(wc -l < "${CONVERTED_DIR}/queries/test.jsonl" 2>/dev/null || echo "0")
echo "  Tables: ${TABLE_COUNT}"
echo "  Test queries: ${QUERY_COUNT}"

# ============================================
# Step 2: Generate LLM summaries
# ============================================
echo ""
echo -e "${GREEN}[Step 2/4] Generating summaries...${NC}"

SCHEMA_FILE="${SUMMARIES_DIR}/schema_narrations.jsonl"
ROW_FILE="${SUMMARIES_DIR}/sample_rows.jsonl"

if [ "$SKIP_SUMMARIES" = true ] && [ -f "${SCHEMA_FILE}" ] && [ -f "${ROW_FILE}" ]; then
    echo -e "${YELLOW}Skipping summary generation (--skip-summaries flag set)${NC}"
else
    echo "Generating schema narrations using ${OPENAI_MODEL}..."
    python3 "${SCRIPT_DIR}/scripts/generate_summaries.py" \
        --dataset "${DATASET}" \
        --tables-dir "${CONVERTED_DIR}/tables" \
        --output-dir "${SUMMARIES_DIR}" \
        --openai-url "${OPENAI_BASE_URL}" \
        --openai-model "${OPENAI_MODEL}"
    
    echo -e "${GREEN}Summary generation complete${NC}"
fi

# Check summary counts
SCHEMA_COUNT=$(wc -l < "${SCHEMA_FILE}" 2>/dev/null || echo "0")
ROW_COUNT=$(wc -l < "${ROW_FILE}" 2>/dev/null || echo "0")
echo "  Schema narrations: ${SCHEMA_COUNT}"
echo "  Row samples: ${ROW_COUNT}"

# ============================================
# Step 3 & 4: Build indices and evaluate
# ============================================
echo ""
echo -e "${GREEN}[Step 3/4] Building indices and evaluating...${NC}"

RESULT_FILE="${OUTPUT_DIR}/${DATASET}_results.json"

# Determine mode based on flags
if [ "$SKIP_INDEX" = true ]; then
    MODE="evaluate-only"
else
    MODE="full"
fi

# Build rerank arguments
RERANK_ARGS=""
if [ "${RERANK_MODE}" != "none" ]; then
    RERANK_ARGS="--rerank ${RERANK_MODE}"
    if [ -n "${RERANK_TOP_K}" ]; then
        RERANK_ARGS="${RERANK_ARGS} --rerank-top-k ${RERANK_TOP_K}"
    fi
    if [ "${RERANK_MODE}" = "llm" ]; then
        RERANK_ARGS="${RERANK_ARGS} --openai-url ${OPENAI_BASE_URL} --llm-model ${OPENAI_MODEL}"
    fi
fi

python3 "${SCRIPT_DIR}/scripts/run_evaluation.py" \
    --dataset "${DATASET}" \
    --work-dir "${WORK_DIR}" \
    --embed-model "${EMBED_MODEL}" \
    --mode "${MODE}" \
    --top-k "${TOP_K}" \
    --alpha "${ALPHA}" \
    --output-file "${RESULT_FILE}" \
    ${RERANK_ARGS}

echo -e "${GREEN}Evaluation complete${NC}"

# ============================================
# Summary
# ============================================
echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}    Evaluation Results${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# Display results
if [ -f "${RESULT_FILE}" ]; then
    python3 -c "
import json
with open('${RESULT_FILE}', 'r') as f:
    results = json.load(f)

print(f\"Dataset: {results['dataset']}\")
print(f\"Total queries: {results['total_queries']}\")
print()
print('Hit Rate@K:')
hit_rates = results.get('hit_rates', results.get('hit_rate', results.get('recall', {})))
for k, v in sorted(hit_rates.items(), key=lambda x: int(x[0])):
    print(f'  Hit@{k}: {v:.2f}%')
print()
print(f\"MRR: {results['mrr']:.4f}\")
"
fi

echo ""
echo "Results saved to: ${RESULT_FILE}"
