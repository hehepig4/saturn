#!/bin/bash
#
# Layer2All Batch Execution - Stage 2+3+4+5 Processing
#
# This script runs layer2_all (Stage 2+3+4+5) for all datasets in the configuration.
# This assumes Stage 1 (federated_primitive_tbox) has already been completed.
#
# Stages included:
#   - Stage 2: Column Summary (classification + data properties)
#   - Stage 3: Table Annotation (roles, relations)
#   - Stage 4: Export (summaries, descriptions)
#   - Stage 5: Retrieval Index (FAISS + BM25)
#
# Usage:
#   ./run_layer2_all.sh [OPTIONS]
#
# Options:
#   --gpu GPU_ID              GPU to use (default: 0)
#   --llm LLM_PURPOSE         LLM purpose key (default: local)
#   --log-level LEVEL         Logging level (default: INFO)
#   --max-tables N            Max tables per dataset (-1 for all, default: -1)
#   --datasets "d1 d2 ..."    Override dataset list (space-separated)
#   --fresh                   Clear transform contracts before Stage 2
#   --disable-virtual-columns Disable virtual column extraction
#   --dry-run                 Show commands without executing
#
# Example:
#   ./run_layer2_all.sh --gpu 0 --max-tables 100 --datasets "fetaqa adventure_works"
#

set -e

# Set loguru log level
export LOGURU_LEVEL="INFO"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/.."

# ==================== Dataset Configuration ====================
# Default list of datasets to process
ALL_DATASETS=(
    "adventure_works"
    "chembl"
    "fetaqa"
    "fetaqapn"
    "public_bi"
    "chicago"
    "bird"
)

# ==================== Common Hyperparameters ====================
# Parallel execution config
BATCH_SIZE=1000
TABLE_MAX_WORKERS=128
ANALYZE_MAX_WORKERS=32
SH_MAX_WORKERS=2

# BF Index config
BF_UNIQUE_RATIO=0.1
BF_TARGET_FPR=1e-6

# Retrieval index config
SEARCH_FIELDS="table_description,column_descriptions"
EMBEDDING_BATCH_SIZE=32

# ==================== Default Parameters ====================
GPU_ID=0
LLM_PURPOSE="local"
LOG_LEVEL="INFO"
MAX_TABLES=-1
CUSTOM_DATASETS=""
FRESH_START=false
DISABLE_VIRTUAL_COLUMNS=false
DRY_RUN=false
SKIP_EXISTING=true

# ==================== Colors ====================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ==================== Parse Arguments ====================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --llm)
            LLM_PURPOSE="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --max-tables)
            MAX_TABLES="$2"
            shift 2
            ;;
        --datasets)
            CUSTOM_DATASETS="$2"
            shift 2
            ;;
        --fresh)
            FRESH_START=true
            shift
            ;;
        --disable-virtual-columns)
            DISABLE_VIRTUAL_COLUMNS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-skip)
            SKIP_EXISTING=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu GPU_ID              GPU to use (default: 0)"
            echo "  --llm LLM_PURPOSE         LLM purpose key (default: local)"
            echo "  --log-level LEVEL         Logging level (default: INFO)"
            echo "  --max-tables N            Max tables per dataset (-1 for all, default: -1)"
            echo "  --datasets \"d1 d2 ...\"  Override dataset list"
            echo "  --fresh                   Clear transform contracts before Stage 2"
            echo "  --disable-virtual-columns Disable virtual column extraction"
            echo "  --dry-run                 Show commands without executing"
            echo "  --no-skip                 Force re-run even if indexes exist"
            echo ""
            echo "Default datasets: ${ALL_DATASETS[*]}"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Use custom datasets if provided, otherwise use ALL_DATASETS
if [[ -n "$CUSTOM_DATASETS" ]]; then
    IFS=' ' read -ra DATASETS <<< "$CUSTOM_DATASETS"
else
    DATASETS=("${ALL_DATASETS[@]}")
fi

# ==================== Main Execution ====================
echo -e "${CYAN}======================================${NC}"
echo -e "${CYAN}  Layer2All Batch Execution${NC}"
echo -e "${CYAN}======================================${NC}"
echo ""
echo -e "GPU: ${YELLOW}${GPU_ID}${NC}"
echo -e "LLM Purpose: ${YELLOW}${LLM_PURPOSE}${NC}"
echo -e "Log Level: ${YELLOW}${LOG_LEVEL}${NC}"
echo -e "Max Tables: ${YELLOW}${MAX_TABLES}${NC}"
echo -e "Fresh Start: ${YELLOW}${FRESH_START}${NC}"
echo -e "Disable Virtual Columns: ${YELLOW}${DISABLE_VIRTUAL_COLUMNS}${NC}"
echo -e "Skip Existing: ${YELLOW}${SKIP_EXISTING}${NC}"
echo -e "Datasets: ${YELLOW}${DATASETS[*]}${NC}"
echo ""

# Export GPU
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Index base path for checking existing
INDEX_DIR="${SOURCE_DIR}/../data/lake/indexes"

# Track results
declare -A RESULTS
TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0

cd "${SOURCE_DIR}"

for dataset in "${DATASETS[@]}"; do
    TOTAL=$((TOTAL + 1))
    
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}[$TOTAL/${#DATASETS[@]}] Dataset: ${YELLOW}${dataset}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Check if FAISS index already exists (indicates Stage 5 completed)
    if [[ "$SKIP_EXISTING" == true ]]; then
        FAISS_INDEX="${INDEX_DIR}/${dataset}/faiss/index.faiss"
        if [[ -f "$FAISS_INDEX" ]]; then
            echo -e "${YELLOW}✓ FAISS index already exists: ${FAISS_INDEX}${NC}"
            echo -e "${YELLOW}  Skipping (use --no-skip to force re-run)${NC}"
            RESULTS[$dataset]="SKIPPED (exists)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
    fi
    
    # Build command
    CMD="python -m demos.run_upo_pipeline"
    CMD+=" --dataset ${dataset}"
    CMD+=" --step layer2_all"
    CMD+=" --max-tables ${MAX_TABLES}"
    CMD+=" --llm-purpose ${LLM_PURPOSE}"
    CMD+=" --log-level ${LOG_LEVEL}"
    CMD+=" --batch-size ${BATCH_SIZE}"
    CMD+=" --table-max-workers ${TABLE_MAX_WORKERS}"
    CMD+=" --analyze-max-workers ${ANALYZE_MAX_WORKERS}"
    CMD+=" --sh-max-workers ${SH_MAX_WORKERS}"
    CMD+=" --bf-unique-ratio ${BF_UNIQUE_RATIO}"
    CMD+=" --bf-target-fpr ${BF_TARGET_FPR}"
    CMD+=" --search-fields ${SEARCH_FIELDS}"
    CMD+=" --embedding-batch-size ${EMBEDDING_BATCH_SIZE}"
    
    if [[ "$FRESH_START" == true ]]; then
        CMD+=" --fresh"
    fi
    
    if [[ "$DISABLE_VIRTUAL_COLUMNS" == true ]]; then
        CMD+=" --disable-virtual-columns"
    fi
    
    echo -e "\n${CYAN}Command:${NC}"
    echo -e "  ${CMD}"
    echo ""
    
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}[DRY RUN] Skipping execution${NC}"
        RESULTS[$dataset]="SKIPPED"
        continue
    fi
    
    # Execute
    START_TIME=$(date +%s)
    
    if eval "$CMD"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo -e "\n${GREEN}✓ ${dataset} completed in ${DURATION}s${NC}"
        RESULTS[$dataset]="SUCCESS (${DURATION}s)"
        SUCCESS=$((SUCCESS + 1))
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo -e "\n${RED}✗ ${dataset} FAILED after ${DURATION}s${NC}"
        RESULTS[$dataset]="FAILED"
        FAILED=$((FAILED + 1))
    fi
done

# ==================== Summary ====================
echo -e "\n${CYAN}======================================${NC}"
echo -e "${CYAN}  Layer2All Batch Summary${NC}"
echo -e "${CYAN}======================================${NC}"
echo ""
echo -e "Total: ${TOTAL}, Success: ${GREEN}${SUCCESS}${NC}, Skipped: ${YELLOW}${SKIPPED}${NC}, Failed: ${RED}${FAILED}${NC}"
echo ""
echo -e "Results:"
for dataset in "${DATASETS[@]}"; do
    status="${RESULTS[$dataset]}"
    if [[ "$status" == *"SUCCESS"* ]]; then
        echo -e "  ${GREEN}✓${NC} ${dataset}: ${status}"
    elif [[ "$status" == *"SKIPPED"* ]] || [[ "$status" == "DRY_RUN" ]]; then
        echo -e "  ${YELLOW}○${NC} ${dataset}: ${status}"
    else
        echo -e "  ${RED}✗${NC} ${dataset}: ${status}"
    fi
done

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
