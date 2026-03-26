#!/bin/bash
#
# Run BIRDIE evaluation on all remaining datasets with proper logging
# Datasets: public_bi, fetaqapn, chicago, bird (in order)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create logs directory
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Datasets to process
DATASETS=("public_bi" "fetaqapn" "chicago" "bird")

# Common parameters
EPOCHS=30
BATCH_SIZE=32

# Main log file
MAIN_LOG="${LOG_DIR}/birdie_all_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${MAIN_LOG}"
}

log "=========================================="
log "BIRDIE All Datasets Evaluation"
log "Started at: $(date)"
log "Datasets: ${DATASETS[*]}"
log "Log directory: ${LOG_DIR}"
log "=========================================="

for dataset in "${DATASETS[@]}"; do
    DATASET_LOG="${LOG_DIR}/${dataset}_$(date +%Y%m%d_%H%M%S).log"
    
    log ""
    log "======================================"
    log "Processing: $dataset"
    log "Started at: $(date)"
    log "Dataset log: ${DATASET_LOG}"
    log "======================================"
    
    # Run evaluation with TLlama + vLLM (SGLang)
    # Redirect both stdout and stderr to log file while also showing on terminal
    if ./evaluate.sh \
        --dataset "$dataset" \
        --use-tllama \
        --use-tllama-vllm \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        2>&1 | tee "${DATASET_LOG}"; then
        
        log "SUCCESS: $dataset completed at $(date)"
        
        # Extract and save results
        RESULT_LINES=$(grep -E "Hits@|eval_" "${DATASET_LOG}" | tail -20)
        if [ -n "$RESULT_LINES" ]; then
            echo "" >> "${LOG_DIR}/${dataset}_results.txt"
            echo "=== ${dataset} Results ===" >> "${LOG_DIR}/${dataset}_results.txt"
            echo "Date: $(date)" >> "${LOG_DIR}/${dataset}_results.txt"
            echo "$RESULT_LINES" >> "${LOG_DIR}/${dataset}_results.txt"
            log "Results saved to ${LOG_DIR}/${dataset}_results.txt"
        fi
    else
        log "ERROR: $dataset failed at $(date)"
        log "Check ${DATASET_LOG} for details"
        # Continue with next dataset even if one fails
    fi
    
    # Append dataset log to main log
    echo "" >> "${MAIN_LOG}"
    echo "=== ${dataset} Summary ===" >> "${MAIN_LOG}"
    tail -50 "${DATASET_LOG}" >> "${MAIN_LOG}" 2>/dev/null || true
done

log ""
log "=========================================="
log "All datasets completed at: $(date)"
log "=========================================="

# Final summary
log ""
log "=== Final Results Summary ==="
for dataset in "${DATASETS[@]}"; do
    if [ -f "${LOG_DIR}/${dataset}_results.txt" ]; then
        log ""
        cat "${LOG_DIR}/${dataset}_results.txt" | tee -a "${MAIN_LOG}"
    fi
done
