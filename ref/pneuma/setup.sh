#!/bin/bash
#
# Pneuma Setup Script
#
# This script sets up the Pneuma table retrieval baseline:
# 1. Clone/copy Pneuma repository
# 2. Create conda environment (pneuma)
# 3. Install dependencies
# 4. Apply compatibility patches
# 5. Download embedding model
#
# Usage:
#   ./setup.sh [--install-dir /path/to/install] [--skip-patches] [--from-existing]
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF_DIR="${SCRIPT_DIR}/../.."
DEFAULT_INSTALL_DIR="${REF_DIR}"
SATURN_ROOT="${REF_DIR}/.."

# Pneuma repository
PNEUMA_REPO="https://github.com/TheDataStation/Pneuma.git"
PNEUMA_CLONE_NAME="pneuma"

# Model paths
EMBED_MODEL_HF="BAAI/bge-base-en-v1.5"
LOCAL_MODEL_DIR="${SATURN_ROOT}/model"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parse arguments
INSTALL_DIR="${DEFAULT_INSTALL_DIR}"
SKIP_PATCHES=false
FROM_EXISTING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --skip-patches)
            SKIP_PATCHES=true
            shift
            ;;
        --from-existing)
            FROM_EXISTING=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --install-dir PATH   Installation directory (default: ${DEFAULT_INSTALL_DIR})"
            echo "  --skip-patches       Skip applying compatibility patches"
            echo "  --from-existing      Use existing pneuma_original instead of cloning"
            echo "  --help, -h           Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

PNEUMA_DIR="${INSTALL_DIR}/${PNEUMA_CLONE_NAME}"

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}    Pneuma Setup Script${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Installation directory: ${INSTALL_DIR}"
echo "Pneuma directory: ${PNEUMA_DIR}"
echo ""

# Step 1: Get Pneuma repository
echo -e "${GREEN}[1/5] Getting Pneuma repository...${NC}"

if [ "$FROM_EXISTING" = true ]; then
    # Copy from existing pneuma_original
    if [ -d "${REF_DIR}/pneuma_original" ]; then
        if [ -d "${PNEUMA_DIR}" ]; then
            echo -e "${YELLOW}${PNEUMA_CLONE_NAME} directory exists, skipping copy${NC}"
        else
            echo "Copying from pneuma_original..."
            cp -r "${REF_DIR}/pneuma_original" "${PNEUMA_DIR}"
            echo -e "${GREEN}Copied successfully${NC}"
        fi
    else
        echo -e "${RED}Error: pneuma_original not found at ${REF_DIR}/pneuma_original${NC}"
        exit 1
    fi
else
    # Clone from GitHub
    if [ -d "${PNEUMA_DIR}" ]; then
        echo -e "${YELLOW}${PNEUMA_CLONE_NAME} directory exists, skipping clone${NC}"
    else
        cd "${INSTALL_DIR}"
        git clone "${PNEUMA_REPO}" "${PNEUMA_CLONE_NAME}"
        echo -e "${GREEN}Cloned successfully${NC}"
    fi
fi

# Step 2: Create conda environment
echo ""
echo -e "${GREEN}[2/5] Setting up conda environment...${NC}"

if conda env list | grep -q "^pneuma "; then
    echo -e "${YELLOW}pneuma environment exists${NC}"
else
    echo "Creating pneuma conda environment..."
    conda create -n pneuma python=3.12 -y
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate pneuma

# Step 3: Install dependencies
echo ""
echo -e "${GREEN}[3/5] Installing dependencies...${NC}"

cd "${PNEUMA_DIR}"

# Install Pneuma package
if [ -f "pyproject.toml" ]; then
    pip install -e .
elif [ -f "setup.py" ]; then
    pip install -e .
else
    echo -e "${YELLOW}No setup file found, installing dependencies manually${NC}"
fi

# Install additional dependencies
pip install chromadb bm25s sentence-transformers openai lancedb pyarrow tqdm loguru

# Install PyStemmer for BM25
pip install PyStemmer

echo -e "${GREEN}Dependencies installed${NC}"

# Step 4: Apply patches
echo ""
echo -e "${GREEN}[4/5] Applying compatibility patches...${NC}"

if [ "$SKIP_PATCHES" = true ]; then
    echo -e "${YELLOW}Skipping patches (--skip-patches flag set)${NC}"
else
    PATCHES_DIR="${SCRIPT_DIR}/patches"
    
    if [ -d "${PATCHES_DIR}" ]; then
        echo "Applying patches from ${PATCHES_DIR}..."
        
        # Patch pneuma.py (optional LLM loading)
        if [ -f "${PATCHES_DIR}/pneuma.py.patch" ]; then
            cd "${PNEUMA_DIR}"
            patch -p1 < "${PATCHES_DIR}/pneuma.py.patch" || echo "Patch may already be applied"
        fi
        
        # Patch query_processor.py (optional reranking)
        if [ -f "${PATCHES_DIR}/query_processor.py.patch" ]; then
            cd "${PNEUMA_DIR}"
            patch -p1 < "${PATCHES_DIR}/query_processor.py.patch" || echo "Patch may already be applied"
        fi
        
        # Patch hybrid_search.py (table-level aggregation)
        if [ -f "${PATCHES_DIR}/hybrid_search_table_aggregation.patch" ]; then
            cd "${PNEUMA_DIR}"
            patch -p1 < "${PATCHES_DIR}/hybrid_search_table_aggregation.patch" || echo "Patch may already be applied"
        fi
        
        echo -e "${GREEN}Patches applied${NC}"
    else
        echo -e "${YELLOW}No patches directory found at ${PATCHES_DIR}${NC}"
        echo "You may need to manually apply compatibility patches"
    fi
fi

# Step 5: Download embedding model
echo ""
echo -e "${GREEN}[5/5] Setting up embedding model...${NC}"

# Check if BGE-M3 is available locally
if [ -d "${LOCAL_MODEL_DIR}/bge-m3" ]; then
    echo -e "${GREEN}BGE-M3 model found at ${LOCAL_MODEL_DIR}/bge-m3${NC}"
else
    echo "Downloading ${EMBED_MODEL_HF}..."
    python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('${EMBED_MODEL_HF}')
print('Model downloaded successfully')
"
fi

# Create symlinks for scripts
echo ""
echo -e "${GREEN}Creating symlinks for evaluation scripts...${NC}"

SCRIPTS_DIR="${PNEUMA_DIR}/experiments/pneuma_retriever"
mkdir -p "${SCRIPTS_DIR}"

# Copy evaluation scripts if not exists
for script in generate_summaries.py evaluate_benchmark.py; do
    if [ -f "${SCRIPT_DIR}/scripts/${script}" ] && [ ! -f "${SCRIPTS_DIR}/${script}" ]; then
        cp "${SCRIPT_DIR}/scripts/${script}" "${SCRIPTS_DIR}/"
        echo "  Copied ${script}"
    fi
done

# Summary
echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}    Pneuma Setup Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Pneuma directory: ${PNEUMA_DIR}"
echo "Conda environment: pneuma"
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate pneuma"
echo "  2. Generate summaries: ./evaluate.sh --dataset chembl --generate-summaries"
echo "  3. Run evaluation: ./evaluate.sh --dataset chembl"
echo ""
