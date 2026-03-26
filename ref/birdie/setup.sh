#!/bin/bash
#
# Birdie Setup Script
# 
# This script automates the setup of the Birdie table retrieval model:
# 1. Clone the BIRDIE repository
# 2. Create and configure conda environment
# 3. Download base models (Llama-3-8B-table-base, BGE-M3)
# 4. Download LoRA adapters for query generation
# 5. Apply custom patches for compatibility
#
# Usage:
#   ./setup.sh [--install-dir /path/to/install] [--skip-download] [--skip-models]
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF_DIR="${SCRIPT_DIR}/../.."
DEFAULT_INSTALL_DIR="${REF_DIR}"
INSTALL_DIR="${DEFAULT_INSTALL_DIR}"

# Model paths (Hugging Face)
LLAMA_TABLE_BASE_HF="kingb/Llama-3-8B-table-base"
BGE_M3_HF="BAAI/bge-m3"
MT5_BASE_HF="google/mt5-base"

# Repository
BIRDIE_REPO="https://github.com/ZJU-DAILY/BIRDIE.git"
BIRDIE_CLONE_NAME="birdie"

# Local vLLM server for query generation (replaces author's LoRA adapters)
VLLM_BASE_URL="http://10.120.47.91:8000/v1"
VLLM_API_KEY="token-abc123"
VLLM_MODEL="Qwen3-Next-80B-A3B-Instruct"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command line arguments
SKIP_DOWNLOAD=false
SKIP_MODELS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --install-dir PATH   Installation directory (default: ${DEFAULT_INSTALL_DIR})"
            echo "  --skip-download      Skip downloading models (use existing)"
            echo "  --skip-models        Skip all model downloads"
            echo "  --help, -h           Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}    Birdie Setup Script${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Installation directory: ${INSTALL_DIR}"
echo ""

# Create installation directory
mkdir -p "${INSTALL_DIR}"
cd "${INSTALL_DIR}"

# Step 1: Clone BIRDIE repository
echo -e "${GREEN}[1/5] Cloning BIRDIE repository...${NC}"
if [ -d "${BIRDIE_CLONE_NAME}" ]; then
    echo -e "${YELLOW}${BIRDIE_CLONE_NAME} directory exists, skipping clone${NC}"
else
    git clone "${BIRDIE_REPO}" "${BIRDIE_CLONE_NAME}"
    echo -e "${GREEN}Cloned successfully to ${BIRDIE_CLONE_NAME}${NC}"
fi

# Step 2: Create conda environment
echo -e "${GREEN}[2/5] Setting up conda environment...${NC}"
if conda env list | grep -q "^birdie "; then
    echo -e "${YELLOW}birdie environment exists${NC}"
else
    echo "Creating birdie conda environment..."
    conda create -n birdie python=3.10 -y
fi

# Install dependencies
echo "Installing dependencies..."
eval "$(conda shell.bash hook)"
conda activate birdie

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.35.2
pip install datasets
pip install accelerate
pip install sentencepiece
pip install scikit-learn
pip install scikit-learn-intelex  # Intel optimized sklearn for clustering
pip install tqdm
pip install openai
pip install peft
pip install FlagEmbedding
pip install wandb  # Required for training logging

echo -e "${GREEN}Dependencies installed${NC}"

# Step 3: Create model directories
echo -e "${GREEN}[3/5] Setting up model directories...${NC}"

# Use existing model directory if available, otherwise create new one
EXISTING_MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/model"
if [ -d "${EXISTING_MODEL_DIR}" ]; then
    MODEL_DIR="${EXISTING_MODEL_DIR}"
    echo "Using existing model directory: ${MODEL_DIR}"
else
    MODEL_DIR="${INSTALL_DIR}/models"
    mkdir -p "${MODEL_DIR}"
fi

if [ "$SKIP_MODELS" = false ]; then
    # Download models using huggingface-cli
    echo "Downloading models from Hugging Face..."
    
    if [ "$SKIP_DOWNLOAD" = false ]; then
        # BGE-M3 for embeddings
        echo "Downloading BGE-M3..."
        if [ ! -d "${MODEL_DIR}/bge-m3" ]; then
            huggingface-cli download ${BGE_M3_HF} --local-dir "${MODEL_DIR}/bge-m3"
        else
            echo -e "${YELLOW}BGE-M3 already exists${NC}"
        fi
        
        # MT5-base for BIRDIE model
        echo "Downloading MT5-base..."
        if [ ! -d "${MODEL_DIR}/mt5-base" ]; then
            huggingface-cli download ${MT5_BASE_HF} --local-dir "${MODEL_DIR}/mt5-base"
        else
            echo -e "${YELLOW}MT5-base already exists${NC}"
        fi
        
        # Llama-3-8B-table-base for query generation (optional, we use local vLLM instead)
        echo "Downloading Llama-3-8B-table-base..."
        if [ ! -d "${MODEL_DIR}/Llama-3-8B-table-base" ] && [ ! -d "${MODEL_DIR}/kingb/Llama-3-8B-table-base" ]; then
            echo -e "${YELLOW}Skipping Llama-3-8B-table-base download (using local vLLM server)${NC}"
        else
            echo -e "${YELLOW}Llama-3-8B-table-base already exists${NC}"
        fi
    fi
else
    echo -e "${YELLOW}Skipping model downloads${NC}"
fi

# Step 4: Apply custom patches
echo -e "${GREEN}[4/5] Applying custom patches...${NC}"
PATCH_DIR="${SCRIPT_DIR}/patches"
BIRDIE_DIR="${INSTALL_DIR}/${BIRDIE_CLONE_NAME}"

if [ -d "${PATCH_DIR}" ]; then
    # Copy patched files
    if [ -f "${PATCH_DIR}/emb.py" ]; then
        cp "${PATCH_DIR}/emb.py" "${BIRDIE_DIR}/tableid/emb.py"
        echo "Applied: emb.py (BGE-M3 default embedding model)"
    fi
    
    if [ -f "${PATCH_DIR}/run.py" ]; then
        cp "${PATCH_DIR}/run.py" "${BIRDIE_DIR}/run.py"
        echo "Applied: run.py (multi-answer evaluation + warmup_ratio)"
    fi
    
    if [ -f "${PATCH_DIR}/data.py" ]; then
        cp "${PATCH_DIR}/data.py" "${BIRDIE_DIR}/data.py"
        echo "Applied: data.py (multi-answer dataset support)"
    fi
    
    if [ -f "${PATCH_DIR}/trainer.py" ]; then
        cp "${PATCH_DIR}/trainer.py" "${BIRDIE_DIR}/trainer.py"
        echo "Applied: trainer.py (configurable beam size for Hits@100)"
    fi
    
    if [ -f "${PATCH_DIR}/query_g.py" ]; then
        cp "${PATCH_DIR}/query_g.py" "${BIRDIE_DIR}/query_generate/query_g.py"
        echo "Applied: query_g.py (model_name parameter)"
    fi
    
    if [ -f "${PATCH_DIR}/hierarchical_clustering.py" ]; then
        cp "${PATCH_DIR}/hierarchical_clustering.py" "${BIRDIE_DIR}/tableid/hierarchical_clustering.py"
        echo "Applied: hierarchical_clustering.py (adapted for our embedding format)"
    fi
else
    echo -e "${YELLOW}No patches directory found, using original BIRDIE code${NC}"
fi

# Copy interface scripts
echo "Copying interface scripts..."
SCRIPTS_DIR="${BIRDIE_DIR}/scripts"
mkdir -p "${SCRIPTS_DIR}"

if [ -d "${SCRIPT_DIR}/scripts" ]; then
    cp -r "${SCRIPT_DIR}/scripts/"* "${SCRIPTS_DIR}/"
    echo "Copied custom scripts to ${SCRIPTS_DIR}"
fi

# Step 5: Create configuration file
echo -e "${GREEN}[5/5] Creating configuration...${NC}"
CONFIG_FILE="${BIRDIE_DIR}/tableid/data_info.json"

cat > "${CONFIG_FILE}" << EOF
{
    "fetaqa": {
        "table_data": "${BIRDIE_DIR}/dataset/fetaqa/table_data.json",
        "question_data": "${BIRDIE_DIR}/dataset/fetaqa/question_data.json"
    },
    "fetaqa_lancedb": {
        "table_data": "${BIRDIE_DIR}/dataset/fetaqa_lancedb/table_data.json",
        "question_data": "${BIRDIE_DIR}/dataset/fetaqa_lancedb/question_data.json"
    },
    "adventure_works": {
        "table_data": "${BIRDIE_DIR}/dataset/adventure_works/table_data.json",
        "question_data": "${BIRDIE_DIR}/dataset/adventure_works/question_data.json"
    }
}
EOF

echo -e "${GREEN}Configuration saved to ${CONFIG_FILE}${NC}"

# Create environment info file
ENV_INFO="${INSTALL_DIR}/environment_info.txt"
cat > "${ENV_INFO}" << EOF
Birdie Installation Info
========================
Installation Date: $(date)
Installation Directory: ${INSTALL_DIR}
BIRDIE Directory: ${BIRDIE_DIR}
Model Directory: ${MODEL_DIR}

Conda Environment: birdie
Python Version: $(python --version 2>&1)

Models:
- BGE-M3: ${MODEL_DIR}/bge-m3
- MT5-base: ${MODEL_DIR}/mt5-base
- Llama-3-8B-table-base: ${MODEL_DIR}/Llama-3-8B-table-base

To activate environment:
  conda activate birdie

To run training:
  cd ${BIRDIE_DIR}
  python run.py --dataset_name <dataset_name> --output_dir <output_dir>

To evaluate:
  python run.py --search --dataset_name <dataset_name> --checkpoint_path <checkpoint_path>
EOF

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}    Setup Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Installation directory: ${INSTALL_DIR}"
echo "To activate environment: conda activate birdie"
echo "Environment info saved to: ${ENV_INFO}"
echo ""
echo "Next steps:"
echo "  1. Prepare your dataset in BIRDIE format"
echo "  2. Generate embeddings: python tableid/emb.py"
echo "  3. Generate semantic IDs: python tableid/hierarchical_clustering.py"
echo "  4. Generate queries (if needed): python query_generate/query_g.py"
echo "  5. Train model: python run.py"
echo ""
