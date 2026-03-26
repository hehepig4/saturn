#!/bin/bash
#
# Train Query Generator LoRA Adapter
#
# This script trains a LoRA adapter on TLlama3 for a specific dataset.
#
# Usage:
#   ./train_query_generator.sh --dataset <name> [options]
#
# Example:
#   ./train_query_generator.sh --dataset public_bi --epochs 3

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/model"
BIRDIE_DIR="$PROJECT_ROOT/ref/birdie"
UNIFIED_BASE="$PROJECT_ROOT/data/benchmark/unified"

# Default paths
TLLAMA_MODEL="${MODEL_DIR}/kingb/Llama-3-8B-table-base"
OUTPUT_BASE="${MODEL_DIR}/BIRDIE/lora_adapter"

# Default training parameters (from paper)
EPOCHS=3
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4
LEARNING_RATE=2e-4
LORA_RANK=8
LORA_ALPHA=32
MAX_INPUT_LENGTH=1024
MAX_OUTPUT_LENGTH=256

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
DATASET_NAME=""
TRAIN_DATA=""
TABLE_DATA=""
UNIFIED_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        --train-data)
            TRAIN_DATA="$2"
            shift 2
            ;;
        --table-data)
            TABLE_DATA="$2"
            shift 2
            ;;
        --unified-path)
            UNIFIED_PATH="$2"
            shift 2
            ;;
        --model)
            TLLAMA_MODEL="$2"
            shift 2
            ;;
        --output-base)
            OUTPUT_BASE="$2"
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
        --gradient-accumulation)
            GRADIENT_ACCUMULATION="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --lora-rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --bf16)
            BF16="--bf16"
            shift
            ;;
        --fp16)
            FP16="--fp16"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --dataset NAME         Dataset name"
            echo ""
            echo "Data source (choose one):"
            echo "  --unified-path PATH    Path to unified benchmark directory"
            echo "  --train-data PATH      Path to prepared training data (JSONL)"
            echo ""
            echo "Training options:"
            echo "  --epochs N             Number of epochs (default: ${EPOCHS})"
            echo "  --batch-size N         Batch size (default: ${BATCH_SIZE})"
            echo "  --gradient-accumulation N  Gradient accumulation steps (default: ${GRADIENT_ACCUMULATION})"
            echo "  --learning-rate R      Learning rate (default: ${LEARNING_RATE})"
            echo "  --lora-rank N          LoRA rank (default: ${LORA_RANK})"
            echo "  --bf16                 Use BF16 training"
            echo "  --fp16                 Use FP16 training"
            echo ""
            echo "Paths:"
            echo "  --model PATH           TLlama3 model path"
            echo "  --output-base PATH     Base output directory for LoRA adapters"
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

# Determine output directory
OUTPUT_DIR="${OUTPUT_BASE}/${DATASET_NAME}"
mkdir -p "${OUTPUT_DIR}"

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Query Generator LoRA Training${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Dataset: ${DATASET_NAME}"
echo "Model: ${TLLAMA_MODEL}"
echo "Output: ${OUTPUT_DIR}"
echo "Epochs: ${EPOCHS}"
echo "LoRA rank: ${LORA_RANK}"
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate birdie

# Determine training data source
if [ -z "$TRAIN_DATA" ]; then
    # Auto-detect from unified format
    if [ -z "$UNIFIED_PATH" ]; then
        UNIFIED_PATH="${UNIFIED_BASE}/${DATASET_NAME}"
    fi
    
    if [ -d "$UNIFIED_PATH" ]; then
        echo "Using unified benchmark format: ${UNIFIED_PATH}"
        
        # Prepare training data
        PREPARED_DATA="${OUTPUT_DIR}/train_lora_data.jsonl"
        
        echo "Preparing training data..."
        python "${SCRIPT_DIR}/scripts/prepare_lora_training.py" \
            --unified_path "${UNIFIED_PATH}" \
            --output_path "${PREPARED_DATA}" \
            --split train
        
        TRAIN_DATA="${PREPARED_DATA}"
    else
        # Try BIRDIE format
        BIRDIE_TRAIN="${BIRDIE_DIR}/dataset/${DATASET_NAME}/train.json"
        BIRDIE_TABLES="${BIRDIE_DIR}/dataset/${DATASET_NAME}/table_data.json"
        
        if [ -f "${BIRDIE_TRAIN}" ] && [ -f "${BIRDIE_TABLES}" ]; then
            echo "Using BIRDIE format"
            
            PREPARED_DATA="${OUTPUT_DIR}/train_lora_data.jsonl"
            
            echo "Preparing training data..."
            python "${SCRIPT_DIR}/scripts/prepare_lora_training.py" \
                --birdie_train "${BIRDIE_TRAIN}" \
                --table_data "${BIRDIE_TABLES}" \
                --output_path "${PREPARED_DATA}"
            
            TRAIN_DATA="${PREPARED_DATA}"
            TABLE_DATA="${BIRDIE_TABLES}"
        else
            echo -e "${RED}Error: Cannot find training data for dataset ${DATASET_NAME}${NC}"
            echo "Searched:"
            echo "  - ${UNIFIED_PATH}"
            echo "  - ${BIRDIE_TRAIN}"
            exit 1
        fi
    fi
fi

# Count training samples
NUM_SAMPLES=$(wc -l < "${TRAIN_DATA}")
echo "Training samples: ${NUM_SAMPLES}"

# Check if LoRA already exists
if [ -d "${OUTPUT_DIR}/adapter_model.safetensors" ] || [ -f "${OUTPUT_DIR}/adapter_model.safetensors" ] || [ -f "${OUTPUT_DIR}/adapter_config.json" ]; then
    echo -e "${YELLOW}LoRA adapter already exists at ${OUTPUT_DIR}${NC}"
    read -p "Overwrite? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping training."
        exit 0
    fi
fi

# Build training command
TRAIN_CMD="python ${SCRIPT_DIR}/patches/train_query_generator.py \
    --model_path ${TLLAMA_MODEL} \
    --train_data ${TRAIN_DATA} \
    --output_dir ${OUTPUT_DIR} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --learning_rate ${LEARNING_RATE} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --max_input_length ${MAX_INPUT_LENGTH} \
    --max_output_length ${MAX_OUTPUT_LENGTH}"

# Add table data if available
if [ -n "${TABLE_DATA}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --table_data ${TABLE_DATA}"
fi

# Add precision flags
if [ -n "${BF16}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --bf16"
elif [ -n "${FP16}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --fp16"
fi

# Run training
echo ""
echo -e "${BLUE}Starting training...${NC}"
echo "Command: ${TRAIN_CMD}"
echo ""

eval ${TRAIN_CMD}

# Verify output
if [ -f "${OUTPUT_DIR}/adapter_config.json" ]; then
    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}  Training Complete!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo "LoRA adapter saved to: ${OUTPUT_DIR}"
    echo ""
    echo "To use this adapter:"
    echo "  ./evaluate.sh --dataset ${DATASET_NAME} --use-tllama --tllama-lora ${OUTPUT_DIR}"
else
    echo -e "${RED}Error: Training may have failed. Check logs.${NC}"
    exit 1
fi
