#!/bin/bash
# BIRDIE Off-the-shelf Baseline Experiment
# Uses Qwen3-Next-80B-A3B-Instruct as query generator (no LoRA training)
#
# This is a baseline variant for comparison:
# - TLlama+LoRA: Trains a LoRA adapter on Llama-3-8B-table-base for each dataset
# - Off-the-shelf: Uses Qwen3-Next directly without any training
#
# Key differences:
# - No --use-tllama flag: skips LoRA training (Step 3.5)
# - Uses VLLM_BASE_URL for query generation
# 
# Usage:
#   ./exp_birdie_offshelf.sh [dataset1] [dataset2] ...
#   ./exp_birdie_offshelf.sh adventure_works chembl
#   ./exp_birdie_offshelf.sh  # runs all datasets

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs/offshelf_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Qwen3-Next configuration from llm_models.json
VLLM_BASE_URL="http://10.120.47.94:8000/v1"
VLLM_MODEL="Qwen3-Next-80B-A3B-Instruct"
VLLM_API_KEY="token-abc123"

# Default datasets (all evaluation datasets)
DEFAULT_DATASETS="adventure_works chembl"
# For full experiment add: fetaqa fetaqapn public_bi chicago

# Parse arguments or use defaults
if [ $# -gt 0 ]; then
    DATASETS="$@"
else
    DATASETS="$DEFAULT_DATASETS"
fi

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=============================================="
echo "  BIRDIE Off-the-shelf Baseline Experiment"
echo "=============================================="
echo "Timestamp: $TIMESTAMP"
echo "Query Generator: $VLLM_MODEL"
echo "vLLM URL: $VLLM_BASE_URL"
echo "Datasets: $DATASETS"
echo "Log directory: $LOG_DIR"
echo ""

# Check vLLM server
echo -e "${YELLOW}Checking vLLM server availability...${NC}"
if ! curl -s -H "Authorization: Bearer ${VLLM_API_KEY}" "${VLLM_BASE_URL}/models" > /dev/null 2>&1; then
    echo -e "${RED}Error: vLLM server not reachable at ${VLLM_BASE_URL}${NC}"
    echo "Please ensure the Qwen3-Next server is running."
    exit 1
fi
echo -e "${GREEN}vLLM server is available.${NC}"
echo ""

# Create summary file
SUMMARY_FILE="${LOG_DIR}/summary.md"
cat > "$SUMMARY_FILE" << EOF
# BIRDIE Off-the-shelf Baseline Results

**Date:** $(date)
**Query Generator:** ${VLLM_MODEL}
**vLLM URL:** ${VLLM_BASE_URL}
**Mode:** Off-the-shelf (no LoRA training)

EOF

for DATASET in $DATASETS; do
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Processing: $DATASET${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    LOG_FILE="${LOG_DIR}/${DATASET}.log"
    
    # Remove existing train_query.json to force regeneration
    TRAIN_QUERY="${SCRIPT_DIR}/dataset/${DATASET}/train_query.json"
    if [ -f "$TRAIN_QUERY" ]; then
        echo -e "${YELLOW}Removing existing train_query.json to force regeneration...${NC}"
        rm -f "$TRAIN_QUERY"
    fi
    
    # Run evaluate.sh WITHOUT --use-tllama (triggers off-the-shelf mode)
    echo "Running BIRDIE pipeline..."
    CUDA_VISIBLE_DEVICES=6,7 ./evaluate.sh \
        --dataset "$DATASET" \
        --vllm-base-url "$VLLM_BASE_URL" \
        --vllm-api-key "$VLLM_API_KEY" \
        --vllm-model "$VLLM_MODEL" \
        --epochs 30 \
        --num-gpus 2 \
        2>&1 | tee "$LOG_FILE"
    
    # Extract results
    echo "" >> "$SUMMARY_FILE"
    echo "## $DATASET" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    grep -E "eval_Hits@" "$LOG_FILE" | tail -6 >> "$SUMMARY_FILE" || echo "No results found" >> "$SUMMARY_FILE"
    
    echo -e "${GREEN}Completed: $DATASET${NC}"
done

echo ""
echo "=============================================="
echo "  Experiment Complete"
echo "=============================================="
echo "Results saved to: $SUMMARY_FILE"
cat "$SUMMARY_FILE"
