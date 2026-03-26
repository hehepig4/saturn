#!/bin/bash
# ============================================================================
# Unified Benchmark Data Conversion Script
# ============================================================================
# 
# Converts raw benchmark data from Pneuma/Solo formats into unified format.
# 
# Usage:
#   ./convert_benchmark.sh                 # Convert all datasets with config array
#   ./convert_benchmark.sh --dataset bird  # Convert only 'bird' dataset with its config
#
# Configuration:
#   Modify DATASET_CONFIG array below to customize train queries and translation.
#   Format: "name:train_queries:translate"
#   Example: "fetaqapn:2000:true" means 2000 train queries with translation
#
# Output structure:
#   data/benchmark/unified/<dataset>/
#   ├── table/                 # JSON files, one per table
#   └── query/
#       ├── train.jsonl        # Training queries
#       └── test.jsonl         # Test queries (for evaluation)
#
# ============================================================================

set -e  # Exit on error

# ============ Dataset Configuration Array ============
# Format: "name:train_queries:translate"
# Modify this array to customize processing for each dataset
DATASET_CONFIG=(
    "adventure_works:200:false"
    "bird:200:false"
    "chembl:200:false"
    "chicago:200:false"
    "public_bi:200:false"
    "fetaqa:1000:false"
    "fetaqapn:200:false"   
)

# ============ Path Configuration ============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
SOURCE_DIR="$PROJECT_ROOT/source"
PYTHON_SCRIPT="$SOURCE_DIR/scripts/unify_benchmark_data.py"
CONDA_ENV="saturn"
LLM_MODEL="local"

# Runtime variables
SELECTED_DATASET=""

# ============ Parse Arguments ============
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset|-d)
            SELECTED_DATASET="$2"
            shift 2
            ;;
        --llm-model)
            LLM_MODEL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --dataset, -d NAME   Process specific dataset (default: all)"
            echo "  --llm-model MODEL    LLM model for translation (default: local)"
            echo "  --help, -h           Show this help message"
            echo ""
            echo "Configuration:"
            echo "  Modify DATASET_CONFIG array in script to set train_queries and translate per dataset"
            echo ""
            echo "Supported datasets:"
            echo "  adventure_works, bird, chembl, chicago, fetaqapn, public_bi, fetaqa"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============ Helper Function ============
get_config() {
    local dataset_name="$1"
    for config in "${DATASET_CONFIG[@]}"; do
        IFS=':' read -r name train translate <<< "$config"
        if [ "$name" == "$dataset_name" ]; then
            echo "$train:$translate"
            return 0
        fi
    done
    echo "1000:false"  # Default
}

# ============ Validate Environment ============
echo "=============================================="
echo "Unified Benchmark Data Conversion"
echo "=============================================="

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

echo "Project root: $PROJECT_ROOT"
echo "Python script: $PYTHON_SCRIPT"
echo "Conda environment: $CONDA_ENV"
echo ""

# ============ Show Processing Plan ============
echo "Processing Plan:"
echo "----------------"

if [ -n "$SELECTED_DATASET" ]; then
    DATASETS_TO_PROCESS=("$SELECTED_DATASET")
else
    DATASETS_TO_PROCESS=()
    for config in "${DATASET_CONFIG[@]}"; do
        IFS=':' read -r name _ _ <<< "$config"
        DATASETS_TO_PROCESS+=("$name")
    done
fi

for ds in "${DATASETS_TO_PROCESS[@]}"; do
    IFS=':' read -r train translate <<< "$(get_config "$ds")"
    trans_mark=""
    [ "$translate" == "true" ] && trans_mark="[translate]"
    echo "  $ds: train=$train $trans_mark"
done
echo ""

# ============ Run Conversion ============
echo "Starting conversion..."
echo ""

cd "$SOURCE_DIR"

for ds in "${DATASETS_TO_PROCESS[@]}"; do
    IFS=':' read -r train translate <<< "$(get_config "$ds")"
    
    echo "=============================================="
    echo "Processing: $ds (train=$train, translate=$translate)"
    echo "=============================================="
    
    CMD="conda run -n $CONDA_ENV python3 $PYTHON_SCRIPT --dataset $ds --train-queries $train"
    
    if [ "$translate" == "true" ]; then
        CMD="$CMD --translate --llm-model $LLM_MODEL"
    fi
    
    echo "Running: $CMD"
    eval $CMD
    echo ""
done

# ============ Verify Output ============
echo ""
echo "=============================================="
echo "Conversion Complete - Verifying Output"
echo "=============================================="

UNIFIED_DIR="$PROJECT_ROOT/data/benchmark/unified"

for ds in "${DATASETS_TO_PROCESS[@]}"; do
    DS_DIR="$UNIFIED_DIR/$ds"
    if [ -d "$DS_DIR" ]; then
        TABLE_COUNT=$(ls -1 "$DS_DIR/table" 2>/dev/null | wc -l)
        TRAIN_COUNT=$(wc -l < "$DS_DIR/query/train.jsonl" 2>/dev/null || echo 0)
        TEST_COUNT=$(wc -l < "$DS_DIR/query/test.jsonl" 2>/dev/null || echo 0)
        echo "✓ $ds: $TABLE_COUNT tables, $TRAIN_COUNT train, $TEST_COUNT test"
    else
        echo "✗ $ds: output directory not found"
    fi
done

echo ""
echo "=============================================="
echo "Benchmark data conversion completed!"
echo "=============================================="
