#!/bin/bash
#
# Pneuma Full Benchmark Evaluation
#
# Runs Pneuma evaluation on all benchmark datasets and generates a summary report.
#
# Usage:
#   ./run_all_datasets.sh [OPTIONS]
#
# Options:
#   --skip-summaries    Skip LLM summary generation (use existing)
#   --skip-index        Skip index building (use existing)
#   --rerank            Enable LLM reranking
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF_DIR="${SCRIPT_DIR}/../.."
OUTPUT_DIR="${SCRIPT_DIR}/output"

# Configuration - (dataset_name, table_count)
# table_count is used to adjust Top-K for small datasets
# Updated to match paper: bird=597, chicago=802
DATASETS=(
    "adventure_works:88"
    "bird:597"
    "chembl:78"
    "chicago:802"
    "public_bi:203"
    "fetaqapn:10330"
    "fetaqa:10330"
)

# Default Top-K values
DEFAULT_TOP_K="1,3,5,10,20,100"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
EXTRA_ARGS=""
RERANK_ARGS=""
SELECTED_DATASETS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-summaries)
            EXTRA_ARGS="${EXTRA_ARGS} --skip-summaries"
            shift
            ;;
        --skip-index)
            EXTRA_ARGS="${EXTRA_ARGS} --skip-index"
            shift
            ;;
        --skip-convert)
            EXTRA_ARGS="${EXTRA_ARGS} --skip-convert"
            shift
            ;;
        --datasets)
            SELECTED_DATASETS="$2"
            shift 2
            ;;
        --rerank)
            RERANK_ARGS="${RERANK_ARGS} --rerank $2"
            shift 2
            ;;
        --rerank-top-k)
            RERANK_ARGS="${RERANK_ARGS} --rerank-top-k $2"
            shift 2
            ;;
        --llm-rerank-model)
            RERANK_ARGS="${RERANK_ARGS} --llm-rerank-model $2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-summaries    Skip LLM summary generation"
            echo "  --skip-index        Skip index building"
            echo "  --skip-convert      Skip data conversion"
            echo "  --rerank MODE       Reranking mode: none, cosine, direct, llm"
            echo "  --rerank-top-k K    Number of tables to rerank"
            echo "  --llm-rerank-model  Path to LLM model for llm rerank"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Filter datasets if --datasets specified
if [ -n "$SELECTED_DATASETS" ]; then
    FILTERED_DATASETS=()
    IFS=',' read -ra SEL_ARRAY <<< "$SELECTED_DATASETS"
    for dataset_config in "${DATASETS[@]}"; do
        IFS=':' read -r ds_name table_count <<< "${dataset_config}"
        for sel in "${SEL_ARRAY[@]}"; do
            if [ "$ds_name" == "$sel" ]; then
                FILTERED_DATASETS+=("$dataset_config")
                break
            fi
        done
    done
    DATASETS=("${FILTERED_DATASETS[@]}")
fi

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_DIR}/run_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}    Pneuma Full Benchmark Evaluation${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Output directory: ${RUN_DIR}"
echo "Datasets: ${#DATASETS[@]}"
echo "Extra args: ${EXTRA_ARGS}"
echo "Rerank args: ${RERANK_ARGS}"
echo ""

# Track results
declare -A RESULTS
FAILED_DATASETS=""

# Run evaluation for each dataset
for dataset_config in "${DATASETS[@]}"; do
    IFS=':' read -r dataset table_count <<< "${dataset_config}"
    
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Dataset: ${dataset} (${table_count} tables)${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    # Adjust Top-K for small datasets
    if [ "${table_count}" -lt 100 ]; then
        # Replace 100 with actual table count
        TOP_K=$(echo "${DEFAULT_TOP_K}" | sed "s/100/${table_count}/g")
        echo -e "${YELLOW}Adjusted Top-K: ${TOP_K} (table count < 100)${NC}"
    else
        TOP_K="${DEFAULT_TOP_K}"
    fi
    
    # Log file
    LOG_FILE="${RUN_DIR}/${dataset}.log"
    
    # Run evaluation
    if "${SCRIPT_DIR}/evaluate.sh" \
        --dataset "${dataset}" \
        --top-k "${TOP_K}" \
        ${EXTRA_ARGS} \
        ${RERANK_ARGS} \
        2>&1 | tee "${LOG_FILE}"; then
        
        # Copy result file
        if [ -f "${OUTPUT_DIR}/${dataset}_results.json" ]; then
            cp "${OUTPUT_DIR}/${dataset}_results.json" "${RUN_DIR}/"
            RESULTS["${dataset}"]="success"
            echo -e "${GREEN}✓ ${dataset} completed${NC}"
        else
            RESULTS["${dataset}"]="no_result"
            echo -e "${YELLOW}⚠ ${dataset} completed but no result file${NC}"
        fi
    else
        RESULTS["${dataset}"]="failed"
        FAILED_DATASETS="${FAILED_DATASETS} ${dataset}"
        echo -e "${RED}✗ ${dataset} failed${NC}"
    fi
done

# Generate summary report
echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}    Generating Summary Report${NC}"
echo -e "${GREEN}================================================${NC}"

SUMMARY_FILE="${RUN_DIR}/summary.md"

cat > "${SUMMARY_FILE}" << 'HEADER'
# Pneuma Benchmark Evaluation Results

HEADER

echo "Run timestamp: $(date '+%Y-%m-%d %H:%M:%S')" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# Generate results table
python3 << EOF
import json
import os
from pathlib import Path

run_dir = Path("${RUN_DIR}")
datasets = [d.split(":")[0] for d in """${DATASETS[*]}""".split()]

# Collect all results
all_results = []
for dataset in datasets:
    result_file = run_dir / f"{dataset}_results.json"
    if result_file.exists():
        with open(result_file) as f:
            all_results.append(json.load(f))

if not all_results:
    print("No results found!")
    exit(0)

# Get all K values
all_k = set()
for r in all_results:
    all_k.update(r.get('hit_rate', r.get('recall', {})).keys())
k_values = sorted([int(k) for k in all_k])

# Generate markdown table
with open("${SUMMARY_FILE}", "a") as f:
    f.write("## Results Summary\n\n")
    
    # Header
    header = "| Dataset | Queries |"
    for k in k_values:
        header += f" Hit@{k} |"
    header += " MRR |\n"
    f.write(header)
    
    # Separator
    sep = "|---------|---------|"
    for _ in k_values:
        sep += "--------|"
    sep += "------|\n"
    f.write(sep)
    
    # Data rows
    for r in all_results:
        dataset = r['dataset']
        queries = r['total_queries']
        hit_rates = r.get('hit_rate', r.get('recall', {}))
        mrr = r.get('mrr', 0)
        
        row = f"| {dataset} | {queries} |"
        for k in k_values:
            rate = hit_rates.get(str(k), 0) * 100
            row += f" {rate:.1f}% |"
        row += f" {mrr:.4f} |\n"
        f.write(row)
    
    f.write("\n")
    
    # Per-dataset details
    f.write("## Detailed Results\n\n")
    for r in all_results:
        f.write(f"### {r['dataset']}\n\n")
        f.write(f"- Total queries: {r['total_queries']}\n")
        f.write(f"- MRR: {r.get('mrr', 0):.4f}\n")
        f.write("\n")

print("Summary generated successfully!")
EOF

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}    Evaluation Complete${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Results directory: ${RUN_DIR}"
echo "Summary report: ${SUMMARY_FILE}"
echo ""

# Show summary
if [ -f "${SUMMARY_FILE}" ]; then
    cat "${SUMMARY_FILE}"
fi

# Report failures
if [ -n "${FAILED_DATASETS}" ]; then
    echo ""
    echo -e "${RED}Failed datasets:${FAILED_DATASETS}${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}All datasets completed successfully!${NC}"
