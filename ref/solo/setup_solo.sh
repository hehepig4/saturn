#!/bin/bash
#
# Solo Setup Script
# This script sets up the Solo table retrieval system from scratch.
#
# Usage:
#     ./setup_solo.sh [--install-dir /path/to/install] [--skip-models] [--skip-patches]
#
# The script will:
# 1. Clone the Solo repository from GitHub
# 2. Create and activate conda environment (s2ld)
# 3. Install all dependencies
# 4. Download pre-trained models (t5-base, sql2nlg)
# 5. Apply necessary GPU compatibility patches
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Install to ref/ directory by default (grandparent of interface/solo, i.e., parent of interface)
DEFAULT_INSTALL_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Parse arguments
INSTALL_DIR="$DEFAULT_INSTALL_DIR"
SKIP_MODELS=false
SKIP_PATCHES=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        --skip-patches)
            SKIP_PATCHES=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--install-dir DIR] [--skip-models] [--skip-patches]"
            echo ""
            echo "Options:"
            echo "  --install-dir DIR    Installation directory (default: $DEFAULT_INSTALL_DIR)"
            echo "  --skip-models        Skip downloading pre-trained models"
            echo "  --skip-patches       Skip applying GPU compatibility patches"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SOLO_DIR="$INSTALL_DIR/solo"
MODELS_DIR="$INSTALL_DIR/models"
INDEX_DIR="$INSTALL_DIR/index"
DATA_DIR="$INSTALL_DIR/data"

echo "============================================"
echo "Solo Setup Script"
echo "============================================"
echo "Installation directory: $INSTALL_DIR"
echo "Solo directory: $SOLO_DIR"
echo "Models directory: $MODELS_DIR"
echo "============================================"

# Step 1: Clone Solo repository
echo ""
echo "[Step 1/6] Cloning Solo repository..."
if [ -d "$SOLO_DIR" ]; then
    echo "  Solo directory already exists: $SOLO_DIR"
    echo "  Skipping clone. Use 'rm -rf $SOLO_DIR' to force re-clone."
else
    cd "$INSTALL_DIR"
    git clone https://github.com/TheDataStation/solo.git
    echo "  Repository cloned successfully."
fi

# Step 2: Create conda environment
echo ""
echo "[Step 2/6] Setting up conda environment..."
if conda info --envs | grep -q "^s2ld "; then
    echo "  Conda environment 's2ld' already exists."
    echo "  Activating existing environment..."
else
    echo "  Creating conda environment 's2ld'..."
    conda create -y --name s2ld python=3.7.9
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate s2ld

# Step 3: Install dependencies
echo ""
echo "[Step 3/6] Installing dependencies..."
cd "$SOLO_DIR"

# Install PyTorch with CUDA FIRST (foundational dependency)
echo "  Installing PyTorch (must be installed first)..."
conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch || echo "  PyTorch may already be installed"

# Install faiss-gpu (depends on PyTorch)
echo "  Installing faiss-gpu..."
conda install -y -c pytorch faiss-gpu || echo "  faiss-gpu may already be installed"

# Install requirements (excluding packages we manage separately)
echo "  Installing requirements..."
# Filter out torch/transformers from requirements to avoid version conflicts
grep -v -E "^torch|^transformers" ./requirements.txt > /tmp/requirements_filtered.txt || true
conda install -y --file /tmp/requirements_filtered.txt 2>/dev/null || pip install -r /tmp/requirements_filtered.txt || echo "  Some requirements may need manual installation"

# Install custom transformers LAST (to override any transformers from requirements)
echo "  Installing custom transformers..."
cd "$SOLO_DIR/relevance/transformers-3.0.2"
pip install -e .
cd "$SOLO_DIR"

# Step 4: Setup sql2question environment
echo ""
echo "[Step 4/6] Setting up sql2question environment..."
cd "$SOLO_DIR/sql2question"
if [ -d "$INSTALL_DIR/pyenv/sql2question" ]; then
    echo "  sql2question virtual environment already exists."
else
    bash prep_env.sh
fi
cd "$SOLO_DIR"

# Step 5: Download models
echo ""
echo "[Step 5/6] Setting up models..."
mkdir -p "$MODELS_DIR"

if [ "$SKIP_MODELS" = true ]; then
    echo "  Skipping model download (--skip-models flag set)"
else
    # Download t5-base from HuggingFace
    if [ -d "$MODELS_DIR/t5-base" ]; then
        echo "  t5-base model already exists."
    else
        echo "  Downloading t5-base from HuggingFace..."
        python3 -c "
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
model_path = '$MODELS_DIR/t5-base'
print(f'Downloading t5-base to {model_path}...')
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)
print('t5-base downloaded successfully.')
"
    fi

    # Download sql2nlg model
    SQL2NLG_MODEL="$MODELS_DIR/sql2nlg-t5-base_2022_01_21.ckpt"
    if [ -f "$SQL2NLG_MODEL" ]; then
        echo "  sql2nlg model already exists."
    else
        echo "  Downloading sql2nlg model..."
        echo "  NOTE: You need to download sql2nlg-t5-base_2022_01_21.ckpt manually from the Solo release page:"
        echo "  https://github.com/TheDataStation/solo/releases"
        echo "  Place it in: $SQL2NLG_MODEL"
    fi
fi

# Step 6: Apply patches
echo ""
echo "[Step 6/6] Applying GPU compatibility patches..."
PATCHES_DIR="$SCRIPT_DIR/patches"

if [ "$SKIP_PATCHES" = true ]; then
    echo "  Skipping patches (--skip-patches flag set)"
else
    if [ -d "$PATCHES_DIR" ]; then
        for patch_file in "$PATCHES_DIR"/*.patch; do
            if [ -f "$patch_file" ]; then
                patch_name=$(basename "$patch_file" .patch)
                echo "  Applying patch: $patch_name"
                cd "$SOLO_DIR"
                patch -p1 < "$patch_file" || echo "  Patch may already be applied or conflicts exist"
            fi
        done
    else
        echo "  No patches directory found at $PATCHES_DIR"
        echo "  Creating patches directory..."
        mkdir -p "$PATCHES_DIR"
    fi
fi

# Create necessary directories and symlinks
echo ""
echo "Setting up directories and symlinks..."
mkdir -p "$INDEX_DIR"
mkdir -p "$DATA_DIR"

# Create symlinks if they don't exist
cd "$SOLO_DIR"
if [ ! -L "data" ]; then
    ln -s "$INDEX_DIR" data || echo "  data symlink may already exist"
fi

cd "$INSTALL_DIR"
if [ ! -L "open_table_discovery" ]; then
    ln -s solo open_table_discovery || echo "  open_table_discovery symlink may already exist"
fi

# Update system.config if needed
echo ""
echo "Checking system.config..."
CONFIG_FILE="$SOLO_DIR/system.config"
if [ -f "$CONFIG_FILE" ]; then
    echo "  system.config exists."
else
    echo "  Creating default system.config..."
    cat > "$CONFIG_FILE" << 'EOF'
{
    "sql_batch_size": 50,
    "min_sql_row_num": 3,
    "index_batch_size": 5000000,
    "experiment": "test",
    "top_n": 100,
    "top_n_train": 30,
    "n_probe": 128,
    "min_tables": 5,
    "max_retr": 1000,
    "debug": false
}
EOF
fi

echo ""
echo "============================================"
echo "Solo setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate s2ld"
echo "2. Convert your data using: python converters/convert_unified_to_solo.py --dataset <name>"
echo "3. Run evaluation using: ./evaluate_solo.sh --dataset <name>"
echo ""
echo "Directory structure:"
echo "  Solo code: $SOLO_DIR"
echo "  Models: $MODELS_DIR"
echo "  Index: $INDEX_DIR"
echo "  Data: $DATA_DIR"
echo ""
