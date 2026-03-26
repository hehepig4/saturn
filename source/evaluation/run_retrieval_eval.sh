#!/bin/bash
#
# Retrieval Evaluation - Compare Different Methods
#
# This script evaluates retrieval performance using cached query analysis,
# comparing three main retrieval approaches:
#
# Retrieval Methods (--methods):
#   - semantic:   HyDE-enhanced Vector + BM25 hybrid search (uses analyze_hyde_retrieval.py)
#                 Compares raw query vs HyDE combined (table + column descriptions)
#   - structural: TBox/ABox constraint-based retrieval (ScorerV3)
#   - hybrid:     Combined structural + semantic retrieval
#
# Usage:
#   ./run_retrieval_eval.sh [OPTIONS]
#
# Options:
#   --gpu GPU_ID        GPU to use (default: 0)
#   --llm LLM           LLM used for analysis cache (default: local)
#   --num-queries N     Number of queries (-1 for all, default: -1)
#   --datasets "d1 d2"  Override dataset list (space-separated)
#   --methods "m1 m2"   Methods to test (default: "semantic structural hybrid")
#   --retriever "r1 r2"  Retriever(s) for semantic: bm25, vector, hybrid (space-separated, default: bm25)
#   --top-k K           Max retrieval candidates (default: 100)
#   --compare           Run comparison mode (all methods + analysis)
#   --aggregate         Aggregate existing exports and generate visualizations (skip evaluation)
#   --experiment-dir D  Use experiment directory D for indexes and eval_results
#
# Example:
#   ./run_retrieval_eval.sh --gpu 0 --datasets "fetaqa fetaqapn" --num-queries 100
#   ./run_retrieval_eval.sh --aggregate  # Aggregate results and generate visualizations
#

set -e

# Set loguru log level to INFO to reduce verbose output
export LOGURU_LEVEL="INFO"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/.."
DATA_DIR="${SOURCE_DIR}/../data"
INDEX_DIR="${DATA_DIR}/lake/indexes"
# Analysis cache is stored in lancedb/eval_results (via get_db_path())
CACHE_DIR="${DATA_DIR}/lake/lancedb/eval_results"
# Run outputs also go to lancedb/eval_results/runs for consistency
EVAL_DIR="${DATA_DIR}/lake/lancedb/eval_results"

# ==================== Configuration ====================
ALL_DATASETS=(
    "adventure_works"
    "chembl"
    "fetaqa"
    "fetaqapn"
    "public_bi"
    "bird"
    "chicago"
)
# ALL_DATASETS=(
#     "public_bi"
# )

# Default parameters
GPU_ID=0
LLM="local"
NUM_QUERIES=-1
CUSTOM_DATASETS=""
METHODS="semantic"
RETRIEVERS="bm25"  # space-separated list: bm25, vector, hybrid, hybrid-sum
TOP_K=100
COMPARE_MODE=false
AGGREGATE_MODE=false
SPLIT="test"
EXPERIMENT_DIR=""

# P@K values to report (aligned with Pneuma)
K_VALUES="1 3 5 10 20 100"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ==================== Parse Arguments ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --llm)
            LLM="$2"
            shift 2
            ;;
        --num-queries)
            NUM_QUERIES="$2"
            shift 2
            ;;
        --datasets)
            CUSTOM_DATASETS="$2"
            shift 2
            ;;
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --retriever)
            RETRIEVERS="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --compare)
            COMPARE_MODE=true
            shift
            ;;
        --aggregate)
            AGGREGATE_MODE=true
            shift
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --experiment-dir)
            EXPERIMENT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            head -28 "$0" | tail -22
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Override paths if experiment directory is specified
if [ -n "$EXPERIMENT_DIR" ]; then
    # Resolve to absolute path
    EXPERIMENT_DIR=$(cd "$EXPERIMENT_DIR" 2>/dev/null && pwd || echo "$EXPERIMENT_DIR")
    INDEX_DIR="${EXPERIMENT_DIR}/indexes"
    CACHE_DIR="${EXPERIMENT_DIR}/lancedb/eval_results"
    EVAL_DIR="${EXPERIMENT_DIR}/lancedb/eval_results"
    export SATURN_DB_PATH="${EXPERIMENT_DIR}/lancedb"
    export SATURN_INDEX_PATH="${EXPERIMENT_DIR}"
    echo -e "${YELLOW}Using experiment directory: ${EXPERIMENT_DIR}${NC}"
    echo "  INDEX_DIR: ${INDEX_DIR}"
    echo "  CACHE_DIR: ${CACHE_DIR}"
    echo "  SATURN_DB_PATH: ${SATURN_DB_PATH}"
    echo "  SATURN_INDEX_PATH: ${SATURN_INDEX_PATH}"
    echo ""
fi

# Use custom datasets if provided
if [ -n "$CUSTOM_DATASETS" ]; then
    IFS=' ' read -ra DATASETS <<< "$CUSTOM_DATASETS"
else
    DATASETS=("${ALL_DATASETS[@]}")
fi

IFS=' ' read -ra METHOD_ARRAY <<< "$METHODS"
IFS=' ' read -ra RETRIEVER_ARRAY <<< "$RETRIEVERS"

# ==================== Check Prerequisites ====================
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}    Retrieval Evaluation${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "Configuration:"
echo "  GPU: ${GPU_ID}"
echo "  LLM: ${LLM}"
echo "  Split: ${SPLIT}"
echo "  Num Queries: ${NUM_QUERIES}"
echo "  Methods: ${METHODS}"
echo "  Retrievers (for semantic): ${RETRIEVERS}"
echo "  Top-K: ${TOP_K}"
echo "  Compare Mode: ${COMPARE_MODE}"
echo "  Aggregate Mode: ${AGGREGATE_MODE}"
echo "  P@K values: ${K_VALUES}"
echo ""

# ==================== Aggregate Mode (skip evaluation) ====================
if [ "$AGGREGATE_MODE" = true ]; then
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}    Running Aggregation Mode${NC}"
    echo -e "${BLUE}================================================${NC}"
    
    EXPORT_DIR="${EVAL_DIR}/exports/parquet"
    AGGREGATED_FILE="${EVAL_DIR}/exports/aggregated/all_results.parquet"
    SUMMARY_CSV="${EVAL_DIR}/exports/aggregated/summary.csv"
    
    mkdir -p "$(dirname ${AGGREGATED_FILE})"
    
    echo "Input directory: ${EXPORT_DIR}"
    echo "Output file: ${AGGREGATED_FILE}"
    
    # Run aggregation (cd to SOURCE_DIR so evaluation module is importable)
    cd "${SOURCE_DIR}" && conda run -n saturn python -m evaluation.aggregate_results \
        --input-dir "${EXPORT_DIR}" \
        --output "${AGGREGATED_FILE}" \
        --summary "${SUMMARY_CSV}"
    
    if [ -f "${AGGREGATED_FILE}" ]; then
        echo -e "${GREEN}✓${NC} Aggregation complete: ${AGGREGATED_FILE}"
        echo ""
        
        # Generate visualizations
        VIZ_DIR="${EVAL_DIR}/visualizations"
        mkdir -p "${VIZ_DIR}"
        
        echo "Generating visualizations..."
        conda run -n saturn python -m evaluation.visualize_results \
            --data "${AGGREGATED_FILE}" \
            --output-dir "${VIZ_DIR}"
        
        echo -e "${GREEN}✓${NC} Visualizations saved to: ${VIZ_DIR}"
    else
        echo -e "${RED}✗${NC} Aggregation failed"
    fi
    
    echo -e "${GREEN}Done!${NC}"
    exit 0
fi

# Create output directory with timestamp
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_DIR="${EVAL_DIR}/runs/eval_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

echo "Output directory: ${RUN_DIR}"
echo ""

# ==================== Detect Available Datasets ====================
AVAILABLE_DATASETS=()

echo -e "${YELLOW}Checking datasets with analysis cache and indexes...${NC}"
for dataset in "${DATASETS[@]}"; do
    # Check for index
    index_path="${INDEX_DIR}/${dataset}"
    if [ ! -d "$index_path" ]; then
        echo -e "  ${RED}✗${NC} ${dataset} (no index)"
        continue
    fi
    
    # Check for analysis cache (unified analysis file)
    # Supports multiple naming patterns:
    # - {dataset}_unified_analysis_*_{llm}*.json (standard)
    # - {dataset}_test_unified_analysis_*_{llm}*.json (with split)
    # - {dataset}_{split}_unified_analysis_*_{llm}_rag*.json (with RAG)
    cache_file=""
    for pattern in \
        "${CACHE_DIR}/${dataset}_unified_analysis_*_${LLM}*.json" \
        "${CACHE_DIR}/${dataset}_test_unified_analysis_*_${LLM}*.json" \
        "${CACHE_DIR}/${dataset}_*_unified_analysis_*_${LLM}*.json"; do
        cache_file=$(ls ${pattern} 2>/dev/null | head -1)
        if [ -n "$cache_file" ]; then
            break
        fi
    done
    
    if [ -n "$cache_file" ]; then
        AVAILABLE_DATASETS+=("$dataset")
        echo -e "  ${GREEN}✓${NC} ${dataset}"
        echo "      Index: ${index_path}"
        echo "      Cache: $(basename ${cache_file})"
    else
        echo -e "  ${YELLOW}!${NC} ${dataset} (no analysis cache, run generate_query_analysis.sh first)"
    fi
done

if [ ${#AVAILABLE_DATASETS[@]} -eq 0 ]; then
    echo -e "${RED}No datasets ready for evaluation!${NC}"
    echo "Run generate_query_analysis.sh first to generate analysis cache."
    exit 1
fi

echo ""
echo "Datasets to evaluate: ${AVAILABLE_DATASETS[*]}"
echo ""

# ==================== Run Evaluation ====================
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Store all results for summary
declare -A ALL_RESULTS

for dataset in "${AVAILABLE_DATASETS[@]}"; do
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}Evaluating: ${dataset}${NC}"
    echo -e "${BLUE}================================================${NC}"
    
    # Comparison mode - runs all methods and provides analysis
    if [ "$COMPARE_MODE" = true ]; then
        echo -e "${CYAN}>>> Running comparison mode for ${dataset}${NC}"
        
        result_file="${RUN_DIR}/${dataset}_compare.json"
        log_file="${RUN_DIR}/${dataset}_compare.log"
        
        CMD="cd ${SOURCE_DIR} && conda run -n saturn python -m evaluation.runners.structural_retrieval"
        CMD="${CMD} -d ${dataset}"
        CMD="${CMD} --llm ${LLM}"
        CMD="${CMD} --compare"
        CMD="${CMD} --top-k ${TOP_K}"
        CMD="${CMD} --split ${SPLIT}"
        
        if [ ${NUM_QUERIES} -gt 0 ]; then
            CMD="${CMD} -n ${NUM_QUERIES}"
        fi
        
        echo "Running: $CMD"
        
        if eval "$CMD" 2>&1 | tee "${log_file}"; then
            echo -e "${GREEN}✓${NC} ${dataset} comparison completed"
        else
            echo -e "${RED}✗${NC} ${dataset} comparison failed"
        fi
        continue
    fi
    
    # Test individual methods
    for method in "${METHOD_ARRAY[@]}"; do
        echo ""
        echo -e "${CYAN}>>> ${dataset} | Method: ${method}${NC}"
        
        # Build command based on method
        # semantic uses analyze_hyde_retrieval.py, others use analyze_structural_retrieval.py
        if [ "$method" = "semantic" ]; then
            # Semantic: loop through each retriever type
            for retriever in "${RETRIEVER_ARRAY[@]}"; do
                echo -e "${CYAN}    Retriever: ${retriever}${NC}"
                
                result_file="${RUN_DIR}/${dataset}_${method}_${retriever}.log"
                
                # Semantic: use HyDE combined mode from analyze_hyde_retrieval.py
                CMD="cd ${SOURCE_DIR} && conda run -n saturn python -m evaluation.runners.hyde_retrieval"
                CMD="${CMD} -d ${dataset}"
                CMD="${CMD} --llm ${LLM}"
                CMD="${CMD} --top-k ${TOP_K}"
                CMD="${CMD} --compare-combined"  # Compare raw vs HyDE combined
                CMD="${CMD} --retriever ${retriever}"  # Use specified retriever (bm25/vector/hybrid)
                
                # Add num queries if specified
                if [ ${NUM_QUERIES} -gt 0 ]; then
                    CMD="${CMD} -n ${NUM_QUERIES}"
                fi
                
                echo "Running: $CMD"
                
                if eval "$CMD" 2>&1 | tee "${result_file}"; then
                    echo -e "${GREEN}✓${NC} ${dataset}/${method}/${retriever} completed"
                    ALL_RESULTS["${dataset}_${method}_${retriever}"]="${result_file}"
                else
                    echo -e "${RED}✗${NC} ${dataset}/${method}/${retriever} failed"
                fi
            done
        else
            result_file="${RUN_DIR}/${dataset}_${method}.log"
            
            # Structural and Hybrid: use analyze_structural_retrieval.py
            CMD="cd ${SOURCE_DIR} && conda run -n saturn python -m evaluation.runners.structural_retrieval"
            CMD="${CMD} -d ${dataset}"
            CMD="${CMD} --llm ${LLM}"
            CMD="${CMD} --top-k ${TOP_K}"
            CMD="${CMD} --split ${SPLIT}"
            
            # Add num queries if specified
            if [ ${NUM_QUERIES} -gt 0 ]; then
                CMD="${CMD} -n ${NUM_QUERIES}"
            fi
            
            # Set retrieval method flag
            case $method in
                structural)
                    CMD="${CMD} --test-retrieval"
                    ;;
                hybrid)
                    CMD="${CMD} --test-hybrid"
                    ;;
                *)
                    echo "Unknown method: $method"
                    continue
                    ;;
            esac
            
            echo "Running: $CMD"
            
            if eval "$CMD" 2>&1 | tee "${result_file}"; then
                echo -e "${GREEN}✓${NC} ${dataset}/${method} completed"
                ALL_RESULTS["${dataset}_${method}"]="${result_file}"
            else
                echo -e "${RED}✗${NC} ${dataset}/${method} failed"
            fi
        fi
    done
done

# ==================== Generate Summary Report ====================
echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}    Generating Summary Report${NC}"
echo -e "${GREEN}================================================${NC}"

SUMMARY_FILE="${RUN_DIR}/summary.md"

cat > "${SUMMARY_FILE}" << EOF
# Retrieval Evaluation Results

Run timestamp: $(date '+%Y-%m-%d %H:%M:%S')

## Configuration
- LLM: ${LLM}
- Num Queries: ${NUM_QUERIES}
- Methods: ${METHODS}
- Top-K: ${TOP_K}
- Compare Mode: ${COMPARE_MODE}

## Datasets Evaluated
EOF

for dataset in "${AVAILABLE_DATASETS[@]}"; do
    echo "- ${dataset}" >> "${SUMMARY_FILE}"
done

echo "" >> "${SUMMARY_FILE}"
echo "## Results" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# Parse results from log files using Python
RUN_DIR_FOR_PYTHON="${RUN_DIR}" python3 << 'EOF'
import re
import sys
import os
from pathlib import Path

run_dir = Path(os.environ.get("RUN_DIR_FOR_PYTHON", "."))

# Find all log files
log_files = list(run_dir.glob("*.log"))

if not log_files:
    print("No log files found.")
    sys.exit(0)

results = {}

# Parse log files for recall metrics
for log_file in log_files:
    content = log_file.read_text()
    
    # Parse dataset and method from filename
    # Semantic logs: {dataset}_semantic_{retriever}.log (e.g. adventure_works_semantic_bm25.log)
    # Other logs: {dataset}_{method}.log
    stem = log_file.stem
    retriever = None
    if '_semantic_' in stem:
        idx = stem.index('_semantic_')
        dataset = stem[:idx] + '_semantic'
        retriever = stem[idx + len('_semantic_'):]
        method = 'semantic'
    elif "_" in stem:
        dataset, method = stem.rsplit("_", 1)
    else:
        continue
    
    # For semantic method, extract both RAW and COMBINED results
    if method == "semantic":
        # Find the Summary Table section
        summary_match = re.search(r"📈 Summary Table\s*─+\s*(.*?)(?:\n\s*\n|\Z)", content, re.DOTALL)
        if summary_match:
            summary_text = summary_match.group(1)
            
            # Parse header to find column positions
            header_match = re.search(r"Metric\s+(.*)", summary_text)
            if header_match:
                header_line = header_match.group(0)
                # Modes are uppercase (RAW, COMBINED, etc.)
                modes_in_header = re.findall(r"\b(RAW|COMBINED|TABLE_DESC|COLUMN_DESC)\b", header_line)
                
                # Parse each metric line
                metric_lines = re.findall(r"(hit@\d+|mrr)\s+([\d.%\s]+)", summary_text)
                
                for metric_name, values_str in metric_lines:
                    # Split values by whitespace, filter out empty strings
                    values = [v.strip() for v in values_str.split() if v.strip()]
                    
                    # Map each mode to its value
                    for i, hyde_mode in enumerate(modes_in_header):
                        if i < len(values):
                            value_str = values[i].rstrip('%')
                            try:
                                value = float(value_str)
                            except ValueError:
                                continue
                            
                            # Initialize storage for this mode
                            if retriever:
                                mode_suffix = f"{retriever} w/o HyDE" if hyde_mode == "RAW" else retriever
                            else:
                                mode_suffix = "w/o HyDE" if hyde_mode == "RAW" else method
                            if (dataset, mode_suffix) not in results:
                                results[(dataset, mode_suffix)] = {'hit': {}, 'mrr': 0.0}
                            
                            if metric_name == 'mrr':
                                results[(dataset, mode_suffix)]['mrr'] = value
                            else:
                                k = metric_name.split('@')[1]
                                results[(dataset, mode_suffix)]['hit'][k] = value
    else:
        # For non-semantic methods (structural, hybrid), use original parsing
        recall_pattern = r"Hit@(\d+):\s*([\d.]+)%?"
        mrr_pattern = r"MRR:\s*([\d.]+)"
        
        recalls = {}
        for match in re.finditer(recall_pattern, content):
            k = match.group(1)
            value = float(match.group(2))
            recalls[k] = value
        
        mrr_match = re.search(mrr_pattern, content)
        mrr = float(mrr_match.group(1)) if mrr_match else 0.0
        
        if recalls:
            results[(dataset, method)] = {
                'hit': recalls,
                'mrr': mrr
            }

# Generate markdown table
summary_path = run_dir / "summary.md"
with open(summary_path, "a") as f:
    if results:
        # Get all K values
        all_ks = sorted(set(k for r in results.values() for k in r['hit'].keys()), key=int)
        
        # Header
        header = "| Dataset | Method |"
        for k in all_ks:
            header += f" R@{k} |"
        header += " MRR |\n"
        f.write(header)
        
        # Separator
        sep = "|---------|--------|"
        for _ in all_ks:
            sep += "------|"
        sep += "------|\n"
        f.write(sep)
        
        # Rows - sort to group by dataset and show w/o HyDE before HyDE
        def sort_key(item):
            dataset, method = item[0]
            # Sort order: dataset name, then w/o HyDE before others
            priority = 0 if method == "w/o HyDE" else 1
            return (dataset, priority, method)
        
        for (dataset, method), data in sorted(results.items(), key=sort_key):
            row = f"| {dataset} | {method} |"
            for k in all_ks:
                val = data['hit'].get(k, 0)
                row += f" {val:.1f}% |"
            row += f" {data['mrr']:.4f} |\n"
            f.write(row)
    else:
        f.write("No results parsed from log files.\n")
        f.write("\nNote: Results are printed to stdout during evaluation.\n")
        f.write("Check individual log files in the run directory for details.\n")

print(f"Summary written to: {summary_path}")
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
    echo "--- Summary ---"
    cat "${SUMMARY_FILE}"
fi

echo ""

# ==================== Export Results for Visualization ====================
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}    Exporting Results for Visualization${NC}"
echo -e "${BLUE}================================================${NC}"

EXPORT_DIR="${EVAL_DIR}/exports/parquet"
mkdir -p "${EXPORT_DIR}"

# Export results using Python evaluation module
EXPORT_TIMESTAMP="${TIMESTAMP}" RUN_DIR_FOR_EXPORT="${RUN_DIR}" \
EXPORT_DIR_FOR_EXPORT="${EXPORT_DIR}" SOURCE_DIR_FOR_EXPORT="${SOURCE_DIR}" LLM="${LLM}" python3 << 'EXPORT_EOF'
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Add source to path (use SOURCE_DIR passed from bash)
source_dir = os.environ.get("SOURCE_DIR_FOR_EXPORT", "")
if source_dir:
    sys.path.insert(0, source_dir)
else:
    sys.path.insert(0, str(Path(os.environ.get("RUN_DIR_FOR_EXPORT", ".")).parent.parent.parent.parent))

try:
    from evaluation.export_utils import EvaluationResults, EvaluationRun, QueryResult
    HAS_EXPORT_MODULE = True
except ImportError:
    HAS_EXPORT_MODULE = False
    print("Warning: evaluation.export_utils not available, skipping structured export")

if HAS_EXPORT_MODULE:
    run_dir = Path(os.environ.get("RUN_DIR_FOR_EXPORT", "."))
    export_dir = Path(os.environ.get("EXPORT_DIR_FOR_EXPORT", "./exports"))
    timestamp = os.environ.get("EXPORT_TIMESTAMP", datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Parse log files and export
    exported_files = []
    
    for log_file in run_dir.glob("*.log"):
        stem = log_file.stem
        if "_" not in stem:
            continue
        
        # Parse filename: {dataset}_semantic_{retriever}.log or {dataset}_{method}.log
        if '_semantic_' in stem:
            idx = stem.index('_semantic_')
            dataset = stem[:idx] + '_semantic'
            retriever = stem[idx + len('_semantic_'):]
            method = 'semantic'
        else:
            dataset, method = stem.rsplit("_", 1)
            retriever = method  # For non-semantic, method IS the retriever fallback
        
        content = log_file.read_text()
        
        # Parse Summary Table for metrics
        summary_match = re.search(r"📈 Summary Table\s*─+\s*(.*?)(?:\n\s*\n|\Z)", content, re.DOTALL)
        if not summary_match:
            continue
        
        summary_text = summary_match.group(1)
        header_match = re.search(r"Metric\s+(.*)", summary_text)
        if not header_match:
            continue
        
        header_line = header_match.group(0)
        modes_in_header = re.findall(r"\b(RAW|COMBINED|TABLE_DESC|COLUMN_DESC)\b", header_line)
        
        # Extract metrics for each mode
        for mode in modes_in_header:
            mode_lower = mode.lower()
            hyde_mode = "raw" if mode == "RAW" else mode_lower
            
            run = EvaluationRun(
                run_id=f"{dataset}_{method}_{retriever}_{hyde_mode}_{timestamp}",
                timestamp=datetime.now().isoformat(),
                dataset=dataset,
                method=method,
                hyde_mode=hyde_mode,
                retriever_type=retriever,
                llm=os.environ.get("LLM", "local")
            )
            
            results = EvaluationResults(run)
            
            # Parse metrics
            metrics = {}
            metric_lines = re.findall(r"(hit@\d+|mrr)\s+([\d.%\s]+)", summary_text)
            mode_index = modes_in_header.index(mode)
            
            for metric_name, values_str in metric_lines:
                values = [v.strip() for v in values_str.split() if v.strip()]
                if mode_index < len(values):
                    value_str = values[mode_index].rstrip('%')
                    try:
                        value = float(value_str)
                        if metric_name != 'mrr':
                            value = value / 100.0  # Convert percentage
                        metrics[metric_name] = value
                    except ValueError:
                        continue
            
            # Manually set metrics from parsed data
            if metrics:
                results.set_aggregated_metrics(
                    hit_at_1=metrics.get('hit@1', 0),
                    hit_at_5=metrics.get('hit@5', 0),
                    hit_at_10=metrics.get('hit@10', 0),
                    hit_at_50=metrics.get('hit@50', 0),
                    hit_at_100=metrics.get('hit@100', 0),
                    mrr=metrics.get('mrr', 0),
                    num_queries=1,
                    num_found=0
                )
                
                # Export (summary mode - one row per run)
                output_base = export_dir / f"{dataset}_{method}_{retriever}_{hyde_mode}_{timestamp}"
                try:
                    results.save_parquet(f"{output_base}.parquet", summary_only=True)
                    exported_files.append(f"{output_base}.parquet")
                except Exception as e:
                    print(f"Warning: Failed to export {output_base}: {e}")
    
    if exported_files:
        print(f"Exported {len(exported_files)} result files to: {export_dir}")
        for f in exported_files:
            print(f"  - {Path(f).name}")
    else:
        print("No results exported (no parseable log files found)")
EXPORT_EOF

# ==================== Generate Visualization ====================
echo ""
echo -e "${BLUE}Generating Visualizations...${NC}"

VIZ_DIR="${EVAL_DIR}/visualizations"
mkdir -p "${VIZ_DIR}"

# Check if we have aggregated data
AGGREGATED_FILE="${EVAL_DIR}/exports/aggregated/all_results.parquet"
if [ -f "${AGGREGATED_FILE}" ]; then
    echo "Using existing aggregated data: ${AGGREGATED_FILE}"
    
    # Generate visualizations
    conda run -n saturn python -m evaluation.visualize_results \
        --data "${AGGREGATED_FILE}" \
        --output-dir "${VIZ_DIR}" \
        --plots hyde_comparison retriever_comparison hit_curve dataset_comparison performance_matrix 2>/dev/null || \
        echo "Note: Visualization generation skipped (may need more data)"
else
    echo "Note: Aggregated data not found. Run --aggregate to combine results across runs."
    echo "  To aggregate: python -m evaluation.aggregate_results --input-dir ${EXPORT_DIR} --output ${AGGREGATED_FILE}"
fi

echo ""
echo -e "${GREEN}Done!${NC}"
