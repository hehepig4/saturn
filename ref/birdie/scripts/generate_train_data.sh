#!/bin/bash
#
# Generate Birdie train data from unified benchmark train queries
#
# Usage:
#   bash generate_train_data.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BIRDIE_DIR="$PROJECT_ROOT/ref/birdie"
UNIFIED_BASE="$PROJECT_ROOT/data/benchmark/unified"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

DATASETS="adventure_works bird chembl chicago fetaqa fetaqapn public_bi"

for DATASET in ${DATASETS}; do
    echo ""
    echo -e "${GREEN}Processing: ${DATASET}${NC}"
    
    # Check for id_map.json
    ID_MAP="${BIRDIE_DIR}/tableid/docid/${DATASET}/id_map.json"
    if [ ! -f "${ID_MAP}" ]; then
        echo -e "${RED}No id_map found for ${DATASET}, skipping...${NC}"
        continue
    fi
    
    # Check for train.jsonl (unified format)
    TRAIN_QUERIES="${UNIFIED_BASE}/${DATASET}/query/train.jsonl"
    if [ ! -f "${TRAIN_QUERIES}" ]; then
        echo -e "${RED}No train.jsonl found for ${DATASET}, skipping...${NC}"
        continue
    fi
    
    # Create datasets directory
    DATASET_DIR="${BIRDIE_DIR}/datasets/${DATASET}"
    mkdir -p "${DATASET_DIR}"
    
    # Convert train.jsonl to BIRDIE format
    OUTPUT="${DATASET_DIR}/birdie_train.json"
    
    python3 << EOF
import json

# Load id_map
with open("${ID_MAP}", "r") as f:
    id_map_raw = json.load(f)

# Convert to dict if list
if isinstance(id_map_raw, list):
    id_map = {item["tableID"]: item["semantic_id"] for item in id_map_raw}
else:
    id_map = id_map_raw

print(f"Loaded {len(id_map)} semantic IDs")

# Load train queries (JSONL format)
train_queries = []
with open("${TRAIN_QUERIES}", "r") as f:
    for line in f:
        if line.strip():
            train_queries.append(json.loads(line))

print(f"Loaded {len(train_queries)} train queries")

# Convert to BIRDIE format
birdie_train = []
skipped = 0

for query in train_queries:
    question = query.get("question", "")
    # answer_tables is a list in unified format
    answer_tables = query.get("answer_tables", [])
    
    if not question or not answer_tables:
        skipped += 1
        continue
    
    # Primary table is the first one
    table_id = answer_tables[0]
    
    # Get semantic ID for primary answer
    semantic_id = id_map.get(table_id)
    if not semantic_id:
        skipped += 1
        continue
    
    sample = {
        "text_id": semantic_id,
        "text": question,
        "tableID": table_id
    }
    birdie_train.append(sample)

print(f"Generated {len(birdie_train)} train samples (skipped {skipped})")

# Save as JSONL
with open("${OUTPUT}", "w") as f:
    for sample in birdie_train:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"Saved to ${OUTPUT}")
EOF
    
    if [ -f "${OUTPUT}" ]; then
        NUM_LINES=$(wc -l < "${OUTPUT}")
        echo -e "${GREEN}Generated ${NUM_LINES} train samples for ${DATASET}${NC}"
    fi
done

echo ""
echo -e "${GREEN}Done!${NC}"
