#!/bin/bash
#
# Generate Query Analysis Cache for Retrieval Evaluation
#
# This script generates RAG-enhanced unified query analysis for all datasets
# with existing retrieval indexes. The analysis results are cached for
# subsequent evaluation runs.
#
# Usage:
#   ./generate_query_analysis.sh [OPTIONS]
#
# Options:
#   --gpu GPU_ID        GPU to use (default: 0)
#   --llm LLM           LLM to use: default, gemini, local (default: default)
#   --parallel N        Number of parallel workers (default: 10)
#   --rag-top-k K       RAG retrieval top-k (default: 3)
#   --split SPLIT       Query split: test, train, entries (default: test)
#   --num-queries N     Number of queries per dataset (-1 for all, default: -1)
#   --datasets "d1 d2"  Override dataset list (space-separated)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/../.."
DATA_DIR="${SOURCE_DIR}/../data"
INDEX_DIR="${DATA_DIR}/lake/indexes"
LOG_DIR="${DATA_DIR}/lake/lancedb/eval_results/logs"

# ==================== Configuration ====================
# Datasets to process (comment out to skip)
ALL_DATASETS=(
    "adventure_works"
    "chembl"
    "fetaqapn"
    "public_bi"
    "bird"
    "chicago"
    "fetaqa"
)
# ALL_DATASETS=(
#     "fetaqapn"
#     "public_bi"
# )
# Default parameters
GPU_ID=0
LLM="local"
PARALLEL=10
RAG_TOP_K=3
SPLIT="test"
NUM_QUERIES=-1
CUSTOM_DATASETS=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ==================== Parse Arguments ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --llm)
            LLM="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --rag-top-k)
            RAG_TOP_K="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --num-queries)
            NUM_QUERIES="$2"
            shift 2
            ;;
        --datasets)
            CUSTOM_DATASETS="$2"
            shift 2
            ;;
        -h|--help)
            head -30 "$0" | tail -20
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Use custom datasets if provided
if [ -n "$CUSTOM_DATASETS" ]; then
    IFS=' ' read -ra DATASETS <<< "$CUSTOM_DATASETS"
else
    DATASETS=("${ALL_DATASETS[@]}")
fi

# ==================== Check Prerequisites ====================
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}    Query Analysis Cache Generator${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "Configuration:"
echo "  GPU: ${GPU_ID}"
echo "  LLM: ${LLM}"
echo "  Parallel: ${PARALLEL}"
echo "  RAG Top-K: ${RAG_TOP_K}"
echo "  Split: ${SPLIT}"
echo "  Num Queries: ${NUM_QUERIES}"
echo ""

# ==================== Detect Available Datasets ====================
AVAILABLE_DATASETS=()

echo -e "${YELLOW}Checking datasets with existing indexes...${NC}"
for dataset in "${DATASETS[@]}"; do
    faiss_dir="${INDEX_DIR}/${dataset}/faiss"
    bm25_dir="${INDEX_DIR}/${dataset}/bm25"
    
    if [ -d "$faiss_dir" ] || [ -d "$bm25_dir" ]; then
        AVAILABLE_DATASETS+=("$dataset")
        echo -e "  ${GREEN}✓${NC} ${dataset}"
    else
        echo -e "  ${RED}✗${NC} ${dataset} (no index found)"
    fi
done

if [ ${#AVAILABLE_DATASETS[@]} -eq 0 ]; then
    echo -e "${RED}No datasets with indexes found!${NC}"
    echo "Run the UPO pipeline (stage 5) first to generate indexes."
    exit 1
fi

echo ""
echo "Datasets to process: ${AVAILABLE_DATASETS[*]}"
echo ""

# ==================== Process Each Dataset ====================
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Create log directory with timestamp
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_LOG_DIR="${LOG_DIR}/generate_${TIMESTAMP}"
mkdir -p "${RUN_LOG_DIR}"
echo "Log directory: ${RUN_LOG_DIR}"
echo ""

declare -A RESULTS
FAILED_DATASETS=""

for dataset in "${AVAILABLE_DATASETS[@]}"; do
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}Processing: ${dataset}${NC}"
    echo -e "${BLUE}================================================${NC}"
    
    start_time=$(date +%s)
    LOG_FILE="${RUN_LOG_DIR}/${dataset}.log"
    
    echo "  Log file: ${LOG_FILE}"
    
    if conda run -n saturn python "${SOURCE_DIR}/demos/retrieval.py" \
        --analyze-queries \
        -d "${dataset}" \
        --split "${SPLIT}" \
        -n ${NUM_QUERIES} \
        --use-rag \
        --rag-top-k ${RAG_TOP_K} \
        --llm "${LLM}" \
        --parallel ${PARALLEL} 2>&1 | tee "${LOG_FILE}"; then
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        RESULTS["${dataset}"]="success (${duration}s)"
        echo -e "${GREEN}✓ ${dataset} completed in ${duration}s${NC}"
    else
        RESULTS["${dataset}"]="failed"
        FAILED_DATASETS="${FAILED_DATASETS} ${dataset}"
        echo -e "${RED}✗ ${dataset} failed${NC}"
    fi
    echo ""
done

# ==================== Summary ====================
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}    Summary${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

for dataset in "${AVAILABLE_DATASETS[@]}"; do
    result="${RESULTS[$dataset]}"
    if [[ "$result" == success* ]]; then
        echo -e "  ${GREEN}✓${NC} ${dataset}: ${result}"
    else
        echo -e "  ${RED}✗${NC} ${dataset}: ${result}"
    fi
done

echo ""
echo "Output directory: ${DATA_DIR}/lake/lancedb/eval_results/"
echo "Log directory: ${RUN_LOG_DIR}"
echo ""

# Report failures
if [ -n "${FAILED_DATASETS}" ]; then
    echo -e "${RED}Failed datasets:${FAILED_DATASETS}${NC}"
    exit 1
fi

echo -e "${GREEN}All datasets completed successfully!${NC}"
