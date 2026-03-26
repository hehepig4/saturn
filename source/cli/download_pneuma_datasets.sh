#!/bin/bash
# Download Pneuma Content Benchmark datasets
# Download Pneuma Content Benchmark datasets for evaluation

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
DATA_DIR="${PROJECT_ROOT}/data/raw"

# Expected table counts from Pneuma paper
declare -A EXPECTED_TABLES
EXPECTED_TABLES["chembl"]=78
EXPECTED_TABLES["public_bi"]=203
EXPECTED_TABLES["fetaqapn"]=10330
EXPECTED_TABLES["chicago"]=802
EXPECTED_TABLES["bird"]=597
EXPECTED_TABLES["adventure_works"]=88

# Dataset configurations
declare -A DATASET_TAR
DATASET_TAR["chembl"]="pneuma_chembl_10K.tar"
DATASET_TAR["public_bi"]="pneuma_public_bi.tar"
DATASET_TAR["fetaqapn"]="pneuma_fetaqa.tar"
DATASET_TAR["chicago"]="pneuma_chicago_10K.tar"
DATASET_TAR["bird"]="pneuma_bird.tar"
DATASET_TAR["adventure_works"]="pneuma_adventure_works.tar"

declare -A DATASET_FOLDER
DATASET_FOLDER["chembl"]="pneuma_chembl_10K"
DATASET_FOLDER["public_bi"]="pneuma_public_bi"
DATASET_FOLDER["fetaqapn"]="pneuma_fetaqa"
DATASET_FOLDER["chicago"]="pneuma_chicago_10K"
DATASET_FOLDER["bird"]="pneuma_bird"
DATASET_FOLDER["adventure_works"]="pneuma_adventure_works"

declare -A DATASET_QUESTIONS
DATASET_QUESTIONS["chembl"]="pneuma_chembl_10K_questions_annotated.jsonl"
DATASET_QUESTIONS["public_bi"]="pneuma_public_bi_questions_annotated.jsonl"
DATASET_QUESTIONS["fetaqapn"]="pneuma_fetaqa_questions_annotated.jsonl"
DATASET_QUESTIONS["chicago"]="pneuma_chicago_10K_questions_annotated.jsonl"
DATASET_QUESTIONS["bird"]="pneuma_bird_questions_annotated.jsonl"
DATASET_QUESTIONS["adventure_works"]="pneuma_adventure_works_questions_annotated.jsonl"

BASE_URL="https://storage.googleapis.com/pneuma_open"

# Parse arguments
FORCE_DOWNLOAD=false
CHECK_ONLY=false
DATASETS=""

print_usage() {
    echo "Usage: $0 [OPTIONS] [DATASETS...]"
    echo ""
    echo "Options:"
    echo "  -f, --force       Force re-download even if files exist"
    echo "  -a, --all         Download all datasets"
    echo "  -c, --check-only  Only check completeness, don't download"
    echo "  -h, --help        Show this help"
    echo ""
    echo "Datasets: chembl, public_bi, fetaqapn, chicago, bird, adventure_works"
    echo ""
    echo "Examples:"
    echo "  $0 chicago bird           # Download Chicago and BIRD"
    echo "  $0 -a                      # Download all datasets"
    echo "  $0 -c                      # Check completeness only"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--force)
            FORCE_DOWNLOAD=true
            shift
            ;;
        -a|--all)
            DATASETS="chembl public_bi fetaqapn chicago bird adventure_works"
            shift
            ;;
        -c|--check-only)
            CHECK_ONLY=true
            DATASETS="chembl public_bi fetaqapn chicago bird adventure_works"
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            DATASETS="$DATASETS $1"
            shift
            ;;
    esac
done

# Default to all if no datasets specified
if [ -z "$DATASETS" ]; then
    DATASETS="chembl public_bi fetaqapn chicago bird adventure_works"
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Pneuma Content Benchmark Data Downloader${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Data directory: ${DATA_DIR}"
echo -e "Datasets: ${DATASETS}"
echo ""

# Function to check dataset completeness
check_completeness() {
    local dataset=$1
    local dir="${DATA_DIR}/${dataset}"
    local folder="${DATASET_FOLDER[$dataset]}"
    local expected=${EXPECTED_TABLES[$dataset]}
    
    if [ ! -d "${dir}/${folder}" ]; then
        echo -e "${RED}NOT FOUND${NC}"
        return 1
    fi
    
    local actual=$(ls "${dir}/${folder}/"*.csv 2>/dev/null | wc -l)
    
    if [ "$actual" -ge "$expected" ]; then
        echo -e "${GREEN}COMPLETE${NC} (${actual}/${expected} tables)"
        return 0
    else
        echo -e "${YELLOW}INCOMPLETE${NC} (${actual}/${expected} tables, missing $((expected - actual)))"
        return 1
    fi
}

# Function to download and extract a dataset
download_dataset() {
    local dataset=$1
    local dir="${DATA_DIR}/${dataset}"
    local tar_file="${DATASET_TAR[$dataset]}"
    local folder="${DATASET_FOLDER[$dataset]}"
    local questions="${DATASET_QUESTIONS[$dataset]}"
    local expected=${EXPECTED_TABLES[$dataset]}
    
    echo -e "\n${YELLOW}Processing ${dataset}...${NC}"
    mkdir -p "${dir}"
    cd "${dir}"
    
    # Check if already complete
    if [ "$FORCE_DOWNLOAD" = false ]; then
        if [ -d "${folder}" ]; then
            local actual=$(ls "${folder}/"*.csv 2>/dev/null | wc -l)
            if [ "$actual" -ge "$expected" ]; then
                echo -e "  ${GREEN}Already complete (${actual}/${expected} tables), skipping.${NC}"
                return 0
            else
                echo -e "  ${YELLOW}Incomplete (${actual}/${expected} tables), re-downloading...${NC}"
                rm -rf "${folder}"
            fi
        fi
    else
        echo -e "  Force download enabled, removing existing data..."
        rm -rf "${folder}" "${tar_file}"
    fi
    
    # Download tar
    if [ ! -f "${tar_file}" ]; then
        echo -e "  Downloading ${tar_file}..."
        if ! wget -q --show-progress "${BASE_URL}/${tar_file}"; then
            echo -e "  ${RED}Failed to download ${tar_file}${NC}"
            return 1
        fi
    else
        echo -e "  ${tar_file} already exists."
    fi
    
    # Download questions if available
    if [ ! -f "${questions}" ]; then
        echo -e "  Downloading ${questions}..."
        wget -q --show-progress "${BASE_URL}/${questions}" 2>/dev/null || true
    fi
    
    # Extract
    echo -e "  Extracting ${tar_file}..."
    tar -xf "${tar_file}"
    
    # Verify
    local actual=$(ls "${folder}/"*.csv 2>/dev/null | wc -l)
    if [ "$actual" -ge "$expected" ]; then
        echo -e "  ${GREEN}Success: ${actual}/${expected} tables extracted.${NC}"
    else
        echo -e "  ${RED}Warning: Only ${actual}/${expected} tables extracted!${NC}"
    fi
}

# Check completeness first
echo -e "${BLUE}Checking dataset completeness...${NC}"
echo "----------------------------------------"
printf "%-20s %s\n" "Dataset" "Status"
echo "----------------------------------------"

declare -A NEED_DOWNLOAD
for dataset in $DATASETS; do
    printf "%-20s " "$dataset"
    if check_completeness "$dataset"; then
        NEED_DOWNLOAD[$dataset]=false
    else
        NEED_DOWNLOAD[$dataset]=true
    fi
done
echo "----------------------------------------"

if [ "$CHECK_ONLY" = true ]; then
    echo -e "\n${GREEN}Check complete.${NC}"
    exit 0
fi

# Download incomplete datasets
echo ""
for dataset in $DATASETS; do
    if [ "${NEED_DOWNLOAD[$dataset]}" = true ] || [ "$FORCE_DOWNLOAD" = true ]; then
        download_dataset "$dataset"
    fi
done

# Final summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Download Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${BLUE}Final Status:${NC}"
echo "----------------------------------------"
printf "%-20s %-10s %-10s %s\n" "Dataset" "Tables" "Expected" "Status"
echo "----------------------------------------"

for dataset in $DATASETS; do
    dir="${DATA_DIR}/${dataset}"
    folder="${DATASET_FOLDER[$dataset]}"
    expected=${EXPECTED_TABLES[$dataset]}
    
    if [ -d "${dir}/${folder}" ]; then
        actual=$(ls "${dir}/${folder}/"*.csv 2>/dev/null | wc -l)
        if [ "$actual" -ge "$expected" ]; then
            status="${GREEN}OK${NC}"
        else
            status="${RED}INCOMPLETE${NC}"
        fi
        printf "%-20s %-10s %-10s %b\n" "$dataset" "$actual" "$expected" "$status"
    else
        printf "%-20s %-10s %-10s %b\n" "$dataset" "0" "$expected" "${RED}MISSING${NC}"
    fi
done
echo "----------------------------------------"

echo -e "\n${GREEN}Done!${NC}"
