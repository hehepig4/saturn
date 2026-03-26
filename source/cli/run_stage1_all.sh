#!/bin/bash
#
# Stage 1 Batch Execution - Federated Primitive TBox Generation
#
# This script runs Stage 1 (federated_primitive_tbox) for all datasets in the configuration.
# Each dataset can have different train query counts while sharing other hyperparameters.
#
# Usage:
#   ./run_stage1_all.sh [OPTIONS]
#
# Options:
#   --gpu GPU_ID              GPU to use (default: 0)
#   --llm LLM_PURPOSE         LLM purpose key (default: local)
#   --log-level LEVEL         Logging level (default: INFO)
#   --datasets "d1 d2 ..."    Override dataset list (space-separated)
#   --dry-run                 Show commands without executing
#
# Example:
#   ./run_stage1_all.sh --gpu 0 --datasets "fetaqa adventure_works"
#

set -e

# Set loguru log level
export LOGURU_LEVEL="INFO"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/.."

# ==================== Dataset Configuration ====================
# Each dataset can have different num_train_queries
# Format: "dataset_name:num_train_queries"
# Use -1 for ALL queries
declare -A DATASET_TRAIN_QUERIES=(
    ["adventure_works"]=100
    ["chembl"]=100
    ["fetaqa"]=100
    ["fetaqapn"]=100
    ["public_bi"]=100
    ["chicago"]=100
    ["bird"]=100
)

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
# These are shared across all datasets
CQ_MAX_CONCURRENT=128
DP_MAX_CONCURRENT=16

# ==================== Default Parameters ====================
GPU_ID=0
LLM_PURPOSE="gemini"
LOG_LEVEL="INFO"
CUSTOM_DATASETS=""
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
        --datasets)
            CUSTOM_DATASETS="$2"
            shift 2
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
            echo "  --datasets \"d1 d2 ...\"  Override dataset list"
            echo "  --dry-run                 Show commands without executing"
            echo "  --no-skip                 Force re-run even if TBox exists"
            echo ""
            echo "Configured datasets and train queries:"
            for ds in "${ALL_DATASETS[@]}"; do
                echo "  $ds: ${DATASET_TRAIN_QUERIES[$ds]} queries"
            done
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
echo -e "${CYAN}  Stage 1 Batch Execution${NC}"
echo -e "${CYAN}======================================${NC}"
echo ""
echo -e "GPU: ${YELLOW}${GPU_ID}${NC}"
echo -e "LLM Purpose: ${YELLOW}${LLM_PURPOSE}${NC}"
echo -e "Log Level: ${YELLOW}${LOG_LEVEL}${NC}"
echo -e "Skip Existing: ${YELLOW}${SKIP_EXISTING}${NC}"
echo -e "Datasets: ${YELLOW}${DATASETS[*]}${NC}"
echo ""

# Export GPU
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# TBox base path for checking existing
TBOX_DIR="${SOURCE_DIR}/../data/lake/tbox"

# Track results
declare -A RESULTS
TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0

cd "${SOURCE_DIR}"

for dataset in "${DATASETS[@]}"; do
    TOTAL=$((TOTAL + 1))
    
    # Get train query count for this dataset
    TRAIN_QUERIES=${DATASET_TRAIN_QUERIES[$dataset]:-100}
    
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}[$TOTAL/${#DATASETS[@]}] Dataset: ${YELLOW}${dataset}${NC}"
    echo -e "${BLUE}           Train Queries: ${YELLOW}${TRAIN_QUERIES}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Check if TBox already exists (federated_{dataset}_*)
    if [[ "$SKIP_EXISTING" == true ]]; then
        TBOX_PATTERN="${TBOX_DIR}/federated_${dataset}_*"
        if ls -d $TBOX_PATTERN 1>/dev/null 2>&1; then
            EXISTING_TBOX=$(ls -d $TBOX_PATTERN | head -1)
            echo -e "${YELLOW}✓ TBox already exists: $(basename $EXISTING_TBOX)${NC}"
            echo -e "${YELLOW}  Skipping (use --no-skip to force re-run)${NC}"
            RESULTS[$dataset]="SKIPPED (exists)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
    fi
    
    # Build command
    CMD="python -m demos.run_upo_pipeline"
    CMD+=" --dataset ${dataset}"
    CMD+=" --step federated_primitive_tbox"
    CMD+=" --total-queries ${TRAIN_QUERIES}"
    CMD+=" --llm-purpose ${LLM_PURPOSE}"
    CMD+=" --log-level ${LOG_LEVEL}"
    CMD+=" --cq-max-concurrent ${CQ_MAX_CONCURRENT}"
    CMD+=" --dp-max-concurrent ${DP_MAX_CONCURRENT}"
    
    echo -e "\n${CYAN}Command:${NC}"
    echo -e "  ${CMD}"
    echo ""
    
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}[DRY RUN] Skipping execution${NC}"
        RESULTS[$dataset]="DRY_RUN"
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
echo -e "${CYAN}  Stage 1 Batch Summary${NC}"
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
