#!/bin/bash
# Experiment 1: Iteration Ablation
# Single run to 10 iterations, extract per-iteration metrics from review_log
# Dataset: fetaqa, Queries: 100, LLM: gemini (OpenRouter)

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SOURCE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PROJECT_DIR=$(cd "$SOURCE_DIR/.." && pwd)

cd "$SOURCE_DIR"
source "$(conda info --base)/bin/activate" saturn

# Proxy for OpenRouter (Singapore node to avoid region ban)
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export no_proxy=localhost,127.0.0.1,::1,10.0.0.0/8
export NO_PROXY=localhost,127.0.0.1,::1,10.0.0.0/8

LOG_DIR=$PROJECT_DIR/logs/experiments
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=${LOG_DIR}/iter_ablation_${TS}.log

echo "============================================="
echo "[Exp 1] Iteration Ablation"
echo "  Dataset: fetaqa"
echo "  Max iterations: 10"
echo "  Queries: 100"
echo "  Target classes: 50"
echo "  LLM: gemini"
echo "  Log: ${LOG_FILE}"
echo "  Started: $(date)"
echo "============================================="

python -m cli.run_experiment iteration-ablation \
    -d fetaqa \
    -i 10 \
    -q 100 \
    --target-classes 50 \
    --llm-purpose gemini \
    --name iter_ablation_fetaqa_${TS} \
    2>&1 | tee ${LOG_FILE}

echo ""
echo "[Exp 1] Finished at $(date)"
