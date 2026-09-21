#!/bin/bash
#
# Solo Batch Training and Evaluation
# Train and evaluate on all datasets in unified benchmark directory.
#
# Usage:
#     CUDA_VISIBLE_DEVICES=0,1 ./run_all_datasets.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
UNIFIED_DIR="$BASE_DIR/data/benchmark/unified"
PASSAGE_LIMIT=1000000000  # 1G
EXCLUDE_DATASETS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unified-dir)
            UNIFIED_DIR="$2"
            shift 2
            ;;
        --passage-limit)
            PASSAGE_LIMIT="$2"
            shift 2
            ;;
        --exclude)
            EXCLUDE_DATASETS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --unified-dir DIR     Unified benchmark directory"
            echo "  --passage-limit N     Max passages (skip if exceeded)"
            echo "  --exclude DATASETS    Comma-separated list of datasets to skip"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Activate environment
eval "$(conda shell.bash hook)"
conda activate s2ld

# Create run directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$BASE_DIR/runs/solo_$TIMESTAMP"
mkdir -p "$RUN_DIR"

# Initialize summary
SUMMARY="$RUN_DIR/summary.md"
cat > "$SUMMARY" << EOF
# Solo Training and Evaluation Results

- **Timestamp**: $(date '+%Y-%m-%d %H:%M:%S')
- **GPUs**: ${CUDA_VISIBLE_DEVICES:-all}
- **Passage Limit**: $PASSAGE_LIMIT
- **Excluded**: ${EXCLUDE_DATASETS:-none}

## Results

| Dataset | Tables | Queries | P@1 | P@3 | P@5 | P@10 | P@20 | P@100 | Status |
|---------|--------|---------|-----|-----|-----|------|------|-------|--------|
EOF

echo "============================================"
echo "Solo Batch Training and Evaluation"
echo "============================================"
echo "Run directory: $RUN_DIR"
echo "Excluded: ${EXCLUDE_DATASETS:-none}"
echo ""

# Get all datasets
DATASETS=$(ls -d "$UNIFIED_DIR"/*/ 2>/dev/null | xargs -n1 basename | sort)

# Helper function to check if dataset is excluded
is_excluded() {
    local ds="$1"
    IFS=',' read -ra EXCLUDED <<< "$EXCLUDE_DATASETS"
    for ex in "${EXCLUDED[@]}"; do
        if [ "$ds" = "$ex" ]; then
            return 0
        fi
    done
    return 1
}

for DATASET in $DATASETS; do
    echo "=========================================="
    echo "Processing: $DATASET"
    echo "=========================================="
    
    # Check if dataset is excluded
    if is_excluded "$DATASET"; then
        echo "⏭️  Skipped (excluded)"
        echo ""
        continue
    fi
    
    LOG_FILE="$RUN_DIR/${DATASET}.log"
    
    # Count tables and queries
    DATASET_PATH="$UNIFIED_DIR/$DATASET"
    if [ -d "$DATASET_PATH/table" ]; then
        NUM_TABLES=$(ls "$DATASET_PATH/table/"table_*.json 2>/dev/null | wc -l)
    else
        NUM_TABLES=0
    fi
    
    if [ -f "$DATASET_PATH/query/test.jsonl" ]; then
        NUM_QUERIES=$(wc -l < "$DATASET_PATH/query/test.jsonl")
    else
        NUM_QUERIES=0
    fi
    
    if [ "$NUM_TABLES" -eq 0 ] || [ "$NUM_QUERIES" -eq 0 ]; then
        echo "⚠️  Missing data, skipping"
        echo "| $DATASET | $NUM_TABLES | $NUM_QUERIES | - | - | - | - | - | - | ❌ Missing data |" >> "$SUMMARY"
        continue
    fi
    
    # Run train and evaluate
    if bash "$SCRIPT_DIR/train_and_evaluate.sh" \
        --dataset "$DATASET" \
        --passage-limit "$PASSAGE_LIMIT" \
        --unified-dir "$UNIFIED_DIR" \
        > "$LOG_FILE" 2>&1; then
        
        # Extract metrics from log
        P1=$(grep -oP "P@1[=:]\s*\K[0-9.]+" "$LOG_FILE" 2>/dev/null | tail -1 || echo "-")
        P3=$(grep -oP "P@3[=:]\s*\K[0-9.]+" "$LOG_FILE" 2>/dev/null | tail -1 || echo "-")
        P5=$(grep -oP "P@5[=:]\s*\K[0-9.]+" "$LOG_FILE" 2>/dev/null | tail -1 || echo "-")
        P10=$(grep -oP "P@10[=:]\s*\K[0-9.]+" "$LOG_FILE" 2>/dev/null | tail -1 || echo "-")
        P20=$(grep -oP "P@20[=:]\s*\K[0-9.]+" "$LOG_FILE" 2>/dev/null | tail -1 || echo "-")
        P100=$(grep -oP "P@100[=:]\s*\K[0-9.]+" "$LOG_FILE" 2>/dev/null | tail -1 || echo "-")
        
        echo "| $DATASET | $NUM_TABLES | $NUM_QUERIES | $P1 | $P3 | $P5 | $P10 | $P20 | $P100 | ✅ |" >> "$SUMMARY"
        echo "✅ Complete: P@1=$P1 P@5=$P5"
        
    elif grep -q "SKIPPED" "$LOG_FILE" 2>/dev/null; then
        echo "| $DATASET | $NUM_TABLES | $NUM_QUERIES | - | - | - | - | - | - | ⏭️ OOM |" >> "$SUMMARY"
        echo "⏭️ Skipped (OOM)"
        
    else
        echo "| $DATASET | $NUM_TABLES | $NUM_QUERIES | - | - | - | - | - | - | ❌ Error |" >> "$SUMMARY"
        echo "❌ Error - see $LOG_FILE"
    fi
    
    echo ""
done

echo ""
echo "============================================"
echo "Batch Complete!"
echo "============================================"
echo "Results: $SUMMARY"
cat "$SUMMARY"
