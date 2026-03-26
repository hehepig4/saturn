#!/bin/bash
#
# Birdie Evaluation Script
#
# This script automates the full Birdie evaluation pipeline following the paper:
# 1. Data preparation (train/test split for Query Generator)
# 2. Query Generator LoRA training (if no adapter exists)
# 3. Embedding generation (using BGE-M3)
# 4. Semantic ID generation (hierarchical clustering)
# 5. Synthetic query generation (using trained LoRA)
# 6. DSI model training (MT5-base)
# 7. Evaluation with Hits@K metrics
#
# Usage:
#   ./evaluate.sh --dataset <name> [options]
#
# Example:
#   ./evaluate.sh --dataset fetaqa --epochs 30
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BIRDIE_DIR="$PROJECT_ROOT/ref/birdie"
MODEL_DIR="$PROJECT_ROOT/model"

# Default values
DATASET_NAME=""
UNIFIED_PATH=""
TABLE_JSON=""
QUERY_JSON=""
NUM_QUERIES=20
EPOCHS=30
BATCH_SIZE=64  # Paper: 64 (distributed across GPUs)
LEARNING_RATE=5e-4  # Paper: 5e-4
NUM_GPUS=2  # Should match the number of GPUs in CUDA_VISIBLE_DEVICES
SKIP_EMBEDDING=false
SKIP_CLUSTERING=false
SKIP_QUERY_GEN=false
SKIP_TRAINING=false

# Unified benchmark base path
UNIFIED_BASE="$PROJECT_ROOT/data/benchmark/unified"

# Model paths
EMBEDDING_MODEL="${MODEL_DIR}/bge-m3"
MT5_MODEL="${MODEL_DIR}/mt5-base"

# Local vLLM server for query generation (replaces author's LoRA adapters)
VLLM_BASE_URL="http://10.120.47.91:8000/v1"
VLLM_API_KEY="token-abc123"
VLLM_MODEL="Qwen3-Next-80B-A3B-Instruct"

# TLlama model for query generation (local model option)
TLLAMA_MODEL=""
TLLAMA_MODEL_PATH="${MODEL_DIR}/kingb/Llama-3-8B-table-base"
USE_TLLAMA=false
# LoRA adapter for TLlama (author's trained adapters)
TLLAMA_LORA_PATH=""
# Use local vLLM server to serve TLlama+LoRA (much faster than transformers)
USE_TLLAMA_VLLM=false
TLLAMA_VLLM_PORT=8001
TLLAMA_VLLM_PID=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        --unified-path)
            UNIFIED_PATH="$2"
            shift 2
            ;;
        --table-json)
            TABLE_JSON="$2"
            shift 2
            ;;
        --query-json)
            QUERY_JSON="$2"
            shift 2
            ;;
        --num-queries)
            NUM_QUERIES="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --vllm-base-url)
            VLLM_BASE_URL="$2"
            shift 2
            ;;
        --vllm-api-key)
            VLLM_API_KEY="$2"
            shift 2
            ;;
        --vllm-model)
            VLLM_MODEL="$2"
            shift 2
            ;;
        --embedding-model)
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        --skip-embedding)
            SKIP_EMBEDDING=true
            shift
            ;;
        --skip-clustering)
            SKIP_CLUSTERING=true
            shift
            ;;
        --skip-query-gen)
            SKIP_QUERY_GEN=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --use-tllama)
            USE_TLLAMA=true
            shift
            ;;
        --tllama-model)
            TLLAMA_MODEL_PATH="$2"
            USE_TLLAMA=true
            shift 2
            ;;
        --tllama-lora)
            TLLAMA_LORA_PATH="$2"
            USE_TLLAMA=true
            shift 2
            ;;
        --use-tllama-vllm)
            USE_TLLAMA_VLLM=true
            USE_TLLAMA=true
            shift
            ;;
        --tllama-vllm-port)
            TLLAMA_VLLM_PORT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --dataset NAME         Dataset name (e.g., fetaqa)"
            echo ""
            echo "Data source (one of):"
            echo "  --unified-path PATH    Path to unified format directory"
            echo "                         (default: /data/benchmark/unified/<dataset>)"
            echo "  --table-json PATH      Path to table_data.json"
            echo "  --query-json PATH      Path to question_data.json"
            echo ""
            echo "Training options:"
            echo "  --num-queries N        Number of queries per table (default: 20)"
            echo "  --epochs N             Training epochs (default: 30)"
            echo "  --batch-size N         Batch size (default: 64)"
            echo "  --learning-rate R      Learning rate (default: 5e-4)"
            echo "  --num-gpus N           Number of GPUs (default: 4)"
            echo ""
            echo "Model paths:"
            echo "  --embedding-model PATH  BGE-M3 model path"
            echo "  --vllm-base-url URL     Local vLLM server URL (default: ${VLLM_BASE_URL})"
            echo "  --vllm-api-key KEY      vLLM API key (default: ${VLLM_API_KEY})"
            echo "  --vllm-model NAME       vLLM model name (default: ${VLLM_MODEL})"
            echo ""
            echo "Skip options:"
            echo "  --skip-embedding       Skip embedding generation"
            echo "  --skip-clustering      Skip semantic ID generation"
            echo "  --skip-query-gen       Skip synthetic query generation"
            echo "  --skip-training        Skip DSI training (evaluation only)"
            echo ""
            echo "TLlama options:"
            echo "  --use-tllama           Use local TLlama model for query generation"
            echo "  --tllama-model PATH    Path to TLlama model (default: ${TLLAMA_MODEL_PATH})"
            echo "  --tllama-lora PATH     Path to LoRA adapter for TLlama (optional)"
            echo "  --use-tllama-vllm      Use vLLM to serve TLlama+LoRA (much faster)"
            echo "  --tllama-vllm-port N   Port for local TLlama vLLM server (default: 8001)"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Validate arguments
if [ -z "$DATASET_NAME" ]; then
    echo -e "${RED}Error: --dataset is required${NC}"
    exit 1
fi

# Setup paths
DATASET_DIR="${BIRDIE_DIR}/dataset/${DATASET_NAME}"
TABLEID_DIR="${BIRDIE_DIR}/tableid"
QUERY_GEN_DIR="${SCRIPT_DIR}/patches"  # Use patched query_g.py
OUTPUT_DIR="${BIRDIE_DIR}/output/${DATASET_NAME}"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoint"

# Auto-set TLLAMA_LORA_PATH if using TLlama but no LoRA path specified
if [ "$USE_TLLAMA" = true ] && [ -z "${TLLAMA_LORA_PATH}" ]; then
    TLLAMA_LORA_PATH="${MODEL_DIR}/BIRDIE/lora_adapter/${DATASET_NAME}"
    echo "Auto-setting LoRA path: ${TLLAMA_LORA_PATH}"
fi

mkdir -p "${DATASET_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}    Birdie Evaluation Pipeline${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Dataset: ${DATASET_NAME}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate birdie

# Function to start local vLLM server for TLlama+LoRA
start_tllama_vllm_server() {
    local model_path="$1"
    local lora_path="$2"
    local port="$3"
    
    echo "Starting local SGLang server for TLlama+LoRA on port ${port}..."
    
    # First, ensure port is free (kill any existing process on this port)
    fuser -k ${port}/tcp 2>/dev/null || true
    pkill -9 -f "sglang" 2>/dev/null || true
    sleep 2
    
    # Build SGLang command - use GPU 2,3 with data parallel (SGLang supports LoRA + DP!)
    # Use eval + conda activate instead of conda run for better reliability
    local sglang_cmd="CUDA_VISIBLE_DEVICES=2,3 python -m sglang.launch_server \
        --model-path ${model_path} \
        --port ${port} \
        --host 0.0.0.0 \
        --context-length 4096 \
        --mem-fraction-static 0.85 \
        --dp 2 \
        --log-level warning"
    
    # Add LoRA if provided
    if [ -n "${lora_path}" ]; then
        local lora_name=$(basename "${lora_path}")
        sglang_cmd="${sglang_cmd} --enable-lora --max-lora-rank 64 \
            --lora-paths ${lora_name}=${lora_path}"
    fi
    
    # Start server in background (activate sglang environment first)
    echo "SGLang command: ${sglang_cmd}"
    (
        eval "$(conda shell.bash hook)"
        conda activate sglang
        eval "${sglang_cmd}"
    ) > /tmp/tllama_sglang_${port}.log 2>&1 &
    TLLAMA_VLLM_PID=$!
    
    # Wait for server to be ready (increased timeout for model loading and JIT compilation)
    echo "Waiting for SGLang server to start (PID: ${TLLAMA_VLLM_PID})..."
    local max_wait=900  # 15 minutes timeout for DP mode (includes JIT compilation on first run)
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo -e "${GREEN}SGLang server is ready on port ${port}${NC}"
            return 0
        fi
        # Check if process is still running
        if ! ps -p ${TLLAMA_VLLM_PID} > /dev/null 2>&1; then
            echo -e "${RED}SGLang server process died${NC}"
            echo "Check log: /tmp/tllama_sglang_${port}.log"
            cat /tmp/tllama_sglang_${port}.log | tail -50
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
        echo -n "."
    done
    
    echo -e "${RED}Failed to start SGLang server within ${max_wait} seconds${NC}"
    echo "Check log: /tmp/tllama_sglang_${port}.log"
    cat /tmp/tllama_sglang_${port}.log | tail -30
    return 1
}

# Function to stop local SGLang server
stop_tllama_vllm_server() {
    echo "Stopping SGLang server..."
    
    # First, try to stop by PID if available
    if [ -n "${TLLAMA_VLLM_PID}" ]; then
        echo "Stopping SGLang server (PID: ${TLLAMA_VLLM_PID})..."
        kill ${TLLAMA_VLLM_PID} 2>/dev/null || true
        pkill -P ${TLLAMA_VLLM_PID} 2>/dev/null || true
    fi
    
    # Kill all processes on the SGLang port using fuser (more portable than lsof)
    fuser -k ${TLLAMA_VLLM_PORT:-8001}/tcp 2>/dev/null || true
    
    # Also kill any lingering SGLang processes
    pkill -9 -f "sglang.launch_server.*--port ${TLLAMA_VLLM_PORT:-8001}" 2>/dev/null || true
    pkill -9 -f "sglang" 2>/dev/null || true
    
    # Wait for processes to terminate
    sleep 2
    
    TLLAMA_VLLM_PID=""
    echo "SGLang server stopped successfully"
}

# Trap to clean up vLLM server on exit
trap stop_tllama_vllm_server EXIT

# Step 1: Data Conversion
echo -e "${GREEN}[1/6] Data Conversion${NC}"

# Try unified format first (default for datasets)
if [ -z "$UNIFIED_PATH" ] && [ -z "$TABLE_JSON" ]; then
    # Auto-detect unified format
    AUTO_UNIFIED="${UNIFIED_BASE}/${DATASET_NAME}"
    if [ -d "$AUTO_UNIFIED" ]; then
        UNIFIED_PATH="$AUTO_UNIFIED"
        echo "Auto-detected unified format: ${UNIFIED_PATH}"
    fi
fi

if [ -n "$UNIFIED_PATH" ]; then
    echo "Converting from unified format: ${UNIFIED_PATH}"
    
    # Convert from unified format
    python "${SCRIPT_DIR}/scripts/convert_unified_to_birdie.py" \
        --unified_path "${UNIFIED_PATH}" \
        --output_dir "${DATASET_DIR}"
    
    TABLE_JSON="${DATASET_DIR}/table_data.json"
    QUERY_JSON="${DATASET_DIR}/question_data.json"
    
elif [ -n "$TABLE_JSON" ]; then
    echo "Using existing JSON files"
    cp "${TABLE_JSON}" "${DATASET_DIR}/table_data.json"
    if [ -n "$QUERY_JSON" ]; then
        cp "${QUERY_JSON}" "${DATASET_DIR}/question_data.json"
    fi
else
    # Check if data already exists
    if [ -f "${DATASET_DIR}/table_data.json" ]; then
        echo -e "${YELLOW}Using existing data in ${DATASET_DIR}${NC}"
        TABLE_JSON="${DATASET_DIR}/table_data.json"
        QUERY_JSON="${DATASET_DIR}/question_data.json"
    else
        echo -e "${RED}Error: No data source specified${NC}"
        echo "Use --unified-path or --table-json"
        exit 1
    fi
fi

# Count tables
NUM_TABLES=$(python -c "import json; print(len(json.load(open('${TABLE_JSON}'))))")
echo "Number of tables: ${NUM_TABLES}"

# Step 2: Embedding Generation
echo ""
echo -e "${GREEN}[2/6] Embedding Generation${NC}"

EMBED_DIR="${TABLEID_DIR}/embeddings/${DATASET_NAME}"
mkdir -p "${EMBED_DIR}"

if [ "$SKIP_EMBEDDING" = true ]; then
    echo -e "${YELLOW}Skipping embedding generation${NC}"
elif [ -f "${EMBED_DIR}/table_embeddings.npy" ]; then
    echo -e "${YELLOW}Embeddings already exist${NC}"
else
    echo "Generating embeddings using BGE-M3..."
    cd "${SCRIPT_DIR}"
    
    python patches/emb.py \
        --table_data_path "${TABLE_JSON}" \
        --output_dir "${EMBED_DIR}" \
        --model_name "${EMBEDDING_MODEL}" \
        --batch_size 32
    
    echo -e "${GREEN}Embeddings saved to ${EMBED_DIR}${NC}"
fi

# Step 3: Semantic ID Generation
echo ""
echo -e "${GREEN}[3/6] Semantic ID Generation${NC}"

DOCID_DIR="${TABLEID_DIR}/docid/${DATASET_NAME}"
mkdir -p "${DOCID_DIR}"

if [ "$SKIP_CLUSTERING" = true ]; then
    echo -e "${YELLOW}Skipping clustering${NC}"
elif [ -f "${DOCID_DIR}/id_map.json" ]; then
    echo -e "${YELLOW}Semantic IDs already exist${NC}"
else
    echo "Running hierarchical clustering..."
    cd "${SCRIPT_DIR}"
    
    python patches/hierarchical_clustering.py \
        --embedding_path "${EMBED_DIR}/table_embeddings.npy" \
        --table_data_path "${TABLE_JSON}" \
        --output_dir "${DOCID_DIR}" \
        --depth 3
    
    echo -e "${GREEN}Semantic IDs saved to ${DOCID_DIR}${NC}"
fi

# Step 3.5: Train Query Generator LoRA Adapter (if needed)
if [ "$USE_TLLAMA" = true ] && [ -n "${TLLAMA_LORA_PATH}" ]; then
    echo ""
    echo -e "${GREEN}[3.5/6] Query Generator LoRA Training${NC}"
    
    if [ -f "${TLLAMA_LORA_PATH}/adapter_model.safetensors" ]; then
        echo -e "${YELLOW}LoRA adapter already exists: ${TLLAMA_LORA_PATH}${NC}"
        echo -e "${YELLOW}Skipping training${NC}"
    else
        echo "Training LoRA adapter for dataset: ${DATASET_NAME}"
        echo "Output: ${TLLAMA_LORA_PATH}"
        
        # Run LoRA training script
        cd "${SCRIPT_DIR}"
        ./train_query_generator.sh \
            --dataset "${DATASET_NAME}" \
            --epochs 1 \
            --batch-size 2 \
            --gradient-accumulation 2 \
            --bf16
        
        if [ -f "${TLLAMA_LORA_PATH}/adapter_model.safetensors" ]; then
            echo -e "${GREEN}LoRA adapter trained: ${TLLAMA_LORA_PATH}${NC}"
        else
            echo -e "${RED}LoRA training failed!${NC}"
            exit 1
        fi
    fi
fi

# Step 4: Query Generation
echo ""
echo -e "${GREEN}[4/6] Query Generation${NC}"

# TRAIN_DATA is for query_g.py output (synthetic queries)
# TRAIN_QUERIES_UNIFIED is for unified benchmark train split (human annotations)
TRAIN_DATA="${DATASET_DIR}/train_query.json"
TRAIN_QUERIES_UNIFIED="${DATASET_DIR}/train_queries.json"
QUERY_GEN_USED=false

if [ "$SKIP_QUERY_GEN" = true ]; then
    echo -e "${YELLOW}Skipping query generation${NC}"
elif [ -f "${TRAIN_DATA}" ] && [ -s "${TRAIN_DATA}" ]; then
    # File exists and is non-empty
    echo -e "${YELLOW}Synthetic training queries already exist (${TRAIN_DATA})${NC}"
    QUERY_GEN_USED=true
    # Ensure birdie_train.json is up to date (convert 'question' field to 'text')
    sed 's/"question":/"text":/g' "${TRAIN_DATA}" > "${DATASET_DIR}/birdie_train.json"
else
    # Remove empty file if it exists
    [ -f "${TRAIN_DATA}" ] && [ ! -s "${TRAIN_DATA}" ] && rm -f "${TRAIN_DATA}"
    
    if [ "$USE_TLLAMA" = true ]; then
        # Use local TLlama model for query generation
        if [ "$USE_TLLAMA_VLLM" = true ]; then
            # Start vLLM server for TLlama+LoRA (much faster)
            echo "Starting vLLM server for TLlama+LoRA..."
            start_tllama_vllm_server "${TLLAMA_MODEL_PATH}" "${TLLAMA_LORA_PATH}" "${TLLAMA_VLLM_PORT}"
            
            echo "Generating synthetic queries using TLlama via vLLM server..."
            cd "${QUERY_GEN_DIR}"
            
            QUERY_OUT_PARENT="$(dirname "${DATASET_DIR}")"
            
            # Determine model name for vLLM
            if [ -n "${TLLAMA_LORA_PATH}" ]; then
                # Use LoRA adapter name
                VLLM_TLLAMA_MODEL=$(basename "${TLLAMA_LORA_PATH}")
            else
                # Use base model path
                VLLM_TLLAMA_MODEL="${TLLAMA_MODEL_PATH}"
            fi
            
            python query_g.py \
                --dataset_name "${DATASET_NAME}" \
                --tableid_path "${DOCID_DIR}/id_map.json" \
                --table_data_path "${TABLE_JSON}" \
                --out_train_path "${QUERY_OUT_PARENT}" \
                --num "${NUM_QUERIES}" \
                --vllm_url "http://localhost:${TLLAMA_VLLM_PORT}/v1" \
                --model_name "${VLLM_TLLAMA_MODEL}"
            
            # Stop vLLM server after query generation to free GPU memory for training
            echo "Stopping vLLM server to free GPU memory for training..."
            stop_tllama_vllm_server
        else
            # Use transformers for TLlama (slower but no vLLM dependency)
            if [ -n "${TLLAMA_LORA_PATH}" ]; then
                echo "Generating synthetic queries using TLlama with LoRA: ${TLLAMA_LORA_PATH}..."
            else
                echo "Generating synthetic queries using TLlama model: ${TLLAMA_MODEL_PATH}..."
            fi
            cd "${QUERY_GEN_DIR}"
            
            # query_g.py creates output at {out_train_path}/{dataset_name}/train_query.json
            QUERY_OUT_PARENT="$(dirname "${DATASET_DIR}")"
            
            # Build command with optional LoRA
            # Note: query_g.py uses --model_name (for vLLM) not --model_path
            TLLAMA_CMD="python query_g.py \
                --dataset_name \"${DATASET_NAME}\" \
                --tableid_path \"${DOCID_DIR}/id_map.json\" \
                --table_data_path \"${TABLE_JSON}\" \
                --out_train_path \"${QUERY_OUT_PARENT}\" \
                --num ${NUM_QUERIES} \
                --model_name \"${DATASET_NAME}\""
            
            # Note: LoRA is loaded by vLLM server at startup, not passed to query_g.py
            
            eval ${TLLAMA_CMD}
        fi
        
        # Move output to expected location (handle case where source=dest)
        QUERY_GEN_OUTPUT="${QUERY_OUT_PARENT}/${DATASET_NAME}/train_query.json"
        if [ -f "${QUERY_GEN_OUTPUT}" ]; then
            if [ "${QUERY_GEN_OUTPUT}" != "${TRAIN_DATA}" ]; then
                mv "${QUERY_GEN_OUTPUT}" "${TRAIN_DATA}"
                rmdir "${QUERY_OUT_PARENT}/${DATASET_NAME}" 2>/dev/null || true
            fi
        fi
        
        echo -e "${GREEN}Queries generated using TLlama${NC}"
    else
        # Check if local vLLM server is reachable
        echo "Checking vLLM server at ${VLLM_BASE_URL}..."
        if ! curl -s -H "Authorization: Bearer ${VLLM_API_KEY}" "${VLLM_BASE_URL}/models" > /dev/null; then
            echo -e "${RED}Error: vLLM server not reachable at ${VLLM_BASE_URL}${NC}"
            echo "Please ensure the vLLM server is running."
            exit 1
        fi
        echo "vLLM server is available."
        
        # Generate queries using local vLLM server
        echo "Generating synthetic queries using ${VLLM_MODEL}..."
        cd "${QUERY_GEN_DIR}"
        
        # query_g.py creates output at {out_train_path}/{dataset_name}/train_query.json
        # So pass the parent of DATASET_DIR
        QUERY_OUT_PARENT="$(dirname "${DATASET_DIR}")"
        python query_g.py \
            --dataset_name "${DATASET_NAME}" \
            --tableid_path "${DOCID_DIR}/id_map.json" \
            --table_data_path "${TABLE_JSON}" \
            --out_train_path "${QUERY_OUT_PARENT}" \
            --num "${NUM_QUERIES}" \
            --vllm_url "${VLLM_BASE_URL}" \
            --model_name "${VLLM_MODEL}" \
            --api_key "${VLLM_API_KEY}"
        
        # Move output to expected location (handle case where source=dest)
        QUERY_GEN_OUTPUT="${QUERY_OUT_PARENT}/${DATASET_NAME}/train_query.json"
        if [ -f "${QUERY_GEN_OUTPUT}" ]; then
            if [ "${QUERY_GEN_OUTPUT}" != "${TRAIN_DATA}" ]; then
                mv "${QUERY_GEN_OUTPUT}" "${TRAIN_DATA}"
                rmdir "${QUERY_OUT_PARENT}/${DATASET_NAME}" 2>/dev/null || true
            fi
        fi
        
        echo -e "${GREEN}Queries generated${NC}"
    fi
    
    # query_g.py outputs with 'question' field, convert to 'text' for BIRDIE
    if [ -f "${TRAIN_DATA}" ]; then
        sed 's/"question":/"text":/g' "${TRAIN_DATA}" > "${DATASET_DIR}/birdie_train.json"
        QUERY_GEN_USED=true
    fi
fi

# Check if train_query.json exists (from previous query_g.py run)
if [ -f "${TRAIN_DATA}" ] && [ "$(wc -l < "${TRAIN_DATA}")" -gt 0 ]; then
    echo "Found existing synthetic queries in ${TRAIN_DATA}"
    sed 's/"question":/"text":/g' "${TRAIN_DATA}" > "${DATASET_DIR}/birdie_train.json"
    QUERY_GEN_USED=true
fi

# Prepare final training data - MUST use synthetic queries
if [ "${QUERY_GEN_USED:-false}" = true ]; then
    echo "Using synthetic queries from query_g.py (already in BIRDIE format)..."
else
    # ERROR: No synthetic queries generated - this should not happen
    echo -e "${RED}ERROR: No synthetic queries found!${NC}"
    echo "BIRDIE requires synthetic queries generated by query_g.py or query_g_tllama.py"
    echo "Please ensure query generation completed successfully."
    echo ""
    echo "Expected file: ${TRAIN_DATA}"
    echo ""
    echo "To regenerate queries, run without --skip-query-gen flag."
    exit 1
fi

# Count training samples (JSONL format - count lines)
NUM_TRAIN=$(wc -l < "${DATASET_DIR}/birdie_train.json")
echo "Number of training samples: ${NUM_TRAIN}"

# Prepare test data in BIRDIE format (needed for both training validation and final evaluation)
TEST_DATA_ORIG="${DATASET_DIR}/test_queries.json"
TEST_DATA="${DATASET_DIR}/birdie_test.json"
if [ -f "${TEST_DATA_ORIG}" ]; then
    echo "Converting test queries to BIRDIE format..."
    python "${SCRIPT_DIR}/scripts/prepare_birdie_test.py" \
        --query_data "${TEST_DATA_ORIG}" \
        --id_map "${DOCID_DIR}/id_map.json" \
        --output "${TEST_DATA}"
    NUM_TEST=$(wc -l < "${TEST_DATA}")
    echo "Number of test queries: ${NUM_TEST}"
fi

# Step 5: Model Training
echo ""
echo -e "${GREEN}[5/6] Model Training${NC}"

if [ "$SKIP_TRAINING" = true ]; then
    echo -e "${YELLOW}Skipping training${NC}"
else
    echo "Training BIRDIE model..."
    cd "${BIRDIE_DIR}"
    
    # Note: run.py uses:
    # - base_model_path (not model_name_or_path)
    # - train_file (for training data)
    # - valid_file (for validation during training, use BIRDIE format test data)
    # Paper uses max_steps=8000 instead of num_train_epochs
    torchrun --nproc_per_node=${NUM_GPUS} run.py \
        --output_dir "${OUTPUT_DIR}" \
        --do_train \
        --save_steps 2000 \
        --train_file "${DATASET_DIR}/birdie_train.json" \
        --valid_file "${DATASET_DIR}/birdie_test.json" \
        --base_model_path "${MT5_MODEL}" \
        --bf16 \
        --per_device_train_batch_size "${BATCH_SIZE}" \
        --learning_rate "${LEARNING_RATE}" \
        --warmup_ratio 0.1 \
        --max_steps 8000 \
        --max_length 512 \
        --dataloader_num_workers 8
    
    echo -e "${GREEN}Training completed${NC}"
fi

# Step 6: Evaluation
echo ""
echo -e "${GREEN}[6/6] Evaluation${NC}"

# Find latest checkpoint
LATEST_CHECKPOINT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo -e "${RED}No checkpoint found!${NC}"
    exit 1
fi

echo "Using checkpoint: ${LATEST_CHECKPOINT}"

# Test data should already be converted in Step 4
TEST_DATA="${DATASET_DIR}/birdie_test.json"
if [ ! -f "${TEST_DATA}" ]; then
    echo -e "${RED}No test data found! (${TEST_DATA})${NC}"
    exit 1
fi

NUM_TEST=$(wc -l < "${TEST_DATA}")
echo "Number of test queries: ${NUM_TEST}"

# Run evaluation
cd "${BIRDIE_DIR}"

# Note: run.py uses:
# - valid_file (for test data in Search mode)
# - train_file (for getting valid_ids in Search mode)
# - base_model_path (for loading trained checkpoint)
echo "Running search evaluation..."
python run.py \
    --output_dir "${OUTPUT_DIR}" \
    --do_predict \
    --task Search \
    --valid_file "${TEST_DATA}" \
    --train_file "${DATASET_DIR}/birdie_train.json" \
    --base_model_path "${LATEST_CHECKPOINT}" \
    --bf16 \
    --per_device_eval_batch_size 1 \
    --max_length 512

# Parse results
echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}    Evaluation Results${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Dataset: ${DATASET_NAME}"
echo "Tables: ${NUM_TABLES}"
echo "Training samples: ${NUM_TRAIN}"
echo "Checkpoint: ${LATEST_CHECKPOINT}"
echo ""

# Extract Hits@K from log
RESULT_FILE="${OUTPUT_DIR}/search_results.txt"
if [ -f "${RESULT_FILE}" ]; then
    cat "${RESULT_FILE}"
else
    echo "Check ${OUTPUT_DIR} for detailed results"
fi

echo ""
echo -e "${GREEN}Evaluation complete!${NC}"
