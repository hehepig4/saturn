#!/bin/bash
#
# Solo Train and Evaluate Script
# Convert data, build index, train and evaluate Solo on a single dataset.
#
# Usage:
#     CUDA_VISIBLE_DEVICES=0,1 ./train_and_evaluate.sh --dataset chembl
#

# Use set -e only for critical sections, not globally
# set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
SOLO_DIR="$BASE_DIR/solo"

export SOLO_AUTO_CONTINUE=1

# Default values
DATASET=""
PASSAGE_LIMIT=1000000000  # 1G
UNIFIED_DIR="$BASE_DIR/data/benchmark/unified"
WORK_DIR="$BASE_DIR"
SKIP_CONVERT=false
SKIP_INDEX=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --passage-limit)
            PASSAGE_LIMIT="$2"
            shift 2
            ;;
        --unified-dir)
            UNIFIED_DIR="$2"
            shift 2
            ;;
        --skip-convert)
            SKIP_CONVERT=true
            shift
            ;;
        --skip-index)
            SKIP_INDEX=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --dataset DATASET [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET    Dataset name (required)"
            echo "  --passage-limit N    Max passages (default: 1G, skip if exceeded)"
            echo "  --unified-dir DIR    Unified benchmark directory"
            echo "  --skip-convert       Skip data conversion"
            echo "  --skip-index         Skip index building"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "Error: --dataset is required"
    exit 1
fi

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate s2ld

DATA_DIR="$WORK_DIR/data"
DATASET_DIR="$DATA_DIR/$DATASET"
STATS_FILE="$DATASET_DIR/dataset_stats.json"

# Setup PYTHONPATH for Solo
# Note: Do NOT include sql2question/transformers/src - use pip installed version instead
export PYTHONPATH="${SOLO_DIR}:${SOLO_DIR}/relevance:${SOLO_DIR}/sql2question:${PYTHONPATH}"

echo "============================================"
echo "Solo Train and Evaluate"
echo "============================================"
echo "Dataset:        $DATASET"
echo "Work directory: $WORK_DIR"
echo "Solo directory: $SOLO_DIR"
echo "Passage limit:  $PASSAGE_LIMIT"
echo "============================================"

# Step 1: Convert data
echo ""
echo "[Step 1/4] Converting data..."
if [ "$SKIP_CONVERT" = false ] || [ ! -f "$STATS_FILE" ]; then
    python3 "$SCRIPT_DIR/converters/convert_unified_to_solo.py" \
        --dataset "$DATASET" \
        --unified-dir "$UNIFIED_DIR" \
        --output-dir "$DATA_DIR" \
        --passage-limit "$PASSAGE_LIMIT"
else
    echo "  Skipped (using existing)"
fi

# Check passage limit
if [ -f "$STATS_FILE" ]; then
    EXCEEDED=$(python3 -c "import json; print('true' if json.load(open('$STATS_FILE')).get('exceeded_limit') else 'false')")
    if [ "$EXCEEDED" = "true" ]; then
        PASSAGES=$(python3 -c "import json; print(json.load(open('$STATS_FILE'))['total_passages'])")
        echo "⚠️  SKIPPED: $DATASET has $PASSAGES passages > limit $PASSAGE_LIMIT"
        exit 0
    fi
fi

# Step 2: Build index
echo ""
echo "[Step 2/4] Building index..."
INDEX_FILE="$WORK_DIR/index/on_disk_index_${DATASET}_rel_graph/populated.index"

if [ "$SKIP_INDEX" = false ] || [ ! -f "$INDEX_FILE" ]; then
    cd "$SOLO_DIR"
    
    # index_tables.py handles everything: table2graph + encoding + index building
    # It reads table_chunk_size and table_import_batch from system.config
    python3 index_tables.py \
        --work_dir "$WORK_DIR" \
        --dataset "$DATASET"
    
    echo "  ✓ Index built"
else
    echo "  Skipped (using existing)"
fi

# Step 3: Train
echo ""
echo "[Step 3/4] Training..."

# Check if training already completed (has train_* directories with models)
TRAIN_OUTPUT_DIR="$BASE_DIR/open_table_discovery/output/$DATASET"
EXISTING_TRAIN=$(ls -td "$TRAIN_OUTPUT_DIR"/train_*/train_* 2>/dev/null | head -1)

if [ -n "$EXISTING_TRAIN" ] && ls "$EXISTING_TRAIN"/*.pt >/dev/null 2>&1; then
    echo "  Skipped (using existing training: $(dirname $EXISTING_TRAIN))"
else
    cd "$SOLO_DIR"
    python3 trainer.py \
        --work_dir "$WORK_DIR" \
        --dataset "$DATASET"
fi

# Step 4: Evaluate
echo ""
echo "[Step 4/4] Evaluating..."

# Prepare test data (convert test_queries.jsonl to fusion_query.jsonl format)
echo "  Preparing test data..."
python3 "$SCRIPT_DIR/prepare_test_data.py" \
    --data-dir "$DATA_DIR" \
    --dataset "$DATASET"

# Find the latest trained model
# Structure: output/<dataset>/train_<timestamp>/best_model/ or train_N/
OUTPUT_DIR="$BASE_DIR/open_table_discovery/output/$DATASET"
LATEST_TRAIN_DIR=$(ls -td "$OUTPUT_DIR"/train_* 2>/dev/null | head -1)

if [ -z "$LATEST_TRAIN_DIR" ]; then
    echo "Error: No training output found in $OUTPUT_DIR"
    exit 1
fi

# Prefer best_model directory if exists, otherwise use the last train_N iteration
if [ -d "$LATEST_TRAIN_DIR/best_model" ] && [ -f "$LATEST_TRAIN_DIR/best_model/best_metric_info.json" ]; then
    MODEL_DIR="$LATEST_TRAIN_DIR/best_model"
else
    # Find the last train_N iteration that has best_metric_info.json
    MODEL_DIR=$(ls -td "$LATEST_TRAIN_DIR"/train_* 2>/dev/null | while read dir; do
        if [ -f "$dir/best_metric_info.json" ]; then
            echo "$dir"
            break
        fi
    done)
fi

if [ -z "$MODEL_DIR" ] || [ ! -f "$MODEL_DIR/best_metric_info.json" ]; then
    echo "Error: No trained model with best_metric_info.json found in $LATEST_TRAIN_DIR"
    exit 1
fi

echo "  Using model: $MODEL_DIR"

cd "$SOLO_DIR"
python3 tester.py \
    --work_dir "$WORK_DIR" \
    --dataset "$DATASET" \
    --query_dir query \
    --table_repre rel_graph \
    --bnn 1 \
    --train_model_dir "$MODEL_DIR"

# Compute extended P@K metrics
echo ""
echo "Computing extended metrics (P@1,3,5,10,20,100)..."

LATEST_TEST=$(ls -td "$OUTPUT_DIR"/test_* 2>/dev/null | head -1)
if [ -n "$LATEST_TEST" ]; then
    PRED_FILE=$(find "$LATEST_TEST" -name "pred_*.jsonl" 2>/dev/null | head -1)
fi

if [ -n "$PRED_FILE" ] && [ -f "$PRED_FILE" ]; then
    python3 "$SCRIPT_DIR/compute_metrics.py" "$PRED_FILE" --k-values 1 3 5 10 20 100
else
    echo "Warning: No prediction file found for extended metrics"
fi

echo ""
echo "============================================"
echo "Training and Evaluation Complete!"
echo "============================================"
