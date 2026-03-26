#!/bin/bash
# Experiment 3: Concept Count Ablation
# 5 variants: target_classes = 10, 25, 50, 75, 100
# Dataset: fetaqa, Queries: 100, Iterations: 5, LLM: gemini (OpenRouter)

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
LOG_FILE=${LOG_DIR}/concept_ablation_${TS}.log

echo "============================================="
echo "[Exp 3] Concept Count Ablation"
echo "  Dataset: fetaqa"
echo "  Target classes: 10 25 50 75 100"
echo "  Queries: 100"
echo "  Iterations: 5"
echo "  LLM: gemini"
echo "  Log: ${LOG_FILE}"
echo "  Started: $(date)"
echo "============================================="

python -m cli.run_experiment concept-ablation \
    -d fetaqa \
    -t 10 25 50 75 100 \
    -q 100 \
    -i 5 \
    --llm-purpose gemini \
    --name concept_ablation_fetaqa_${TS} \
    2>&1 | tee ${LOG_FILE}

echo ""
echo "[Exp 3] Finished at $(date)"
