#!/bin/bash
#
# Generate Birdie test data from unified benchmark test queries
#
# This script converts unified test_queries.json to BIRDIE format (birdie_test.json)
# Required files:
#   - /ref/birdie/tableid/docid/{dataset}/id_map.json (semantic IDs)
#   - /data/benchmark/unified/{dataset}/test_queries.json (test queries)
#
# Usage:
#   bash generate_test_data.sh
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
    
    # Check for test.jsonl (unified format)
    TEST_QUERIES="${UNIFIED_BASE}/${DATASET}/query/test.jsonl"
    if [ ! -f "${TEST_QUERIES}" ]; then
        echo -e "${RED}No test.jsonl found for ${DATASET}, skipping...${NC}"
        continue
    fi
    
    # Create datasets directory
    DATASET_DIR="${BIRDIE_DIR}/datasets/${DATASET}"
    mkdir -p "${DATASET_DIR}"
    
    # Convert test.jsonl to BIRDIE format
    OUTPUT="${DATASET_DIR}/birdie_test.json"
    
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

# Load test queries (JSONL format)
test_queries = []
with open("${TEST_QUERIES}", "r") as f:
    for line in f:
        if line.strip():
            test_queries.append(json.loads(line))

print(f"Loaded {len(test_queries)} test queries")

# Convert to BIRDIE format
birdie_test = []
skipped = 0

for query in test_queries:
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
    
    # Get all semantic IDs for multi-answer evaluation
    all_semantic_ids = []
    for ans_table in answer_tables:
        sid = id_map.get(ans_table)
        if sid:
            all_semantic_ids.append(sid)
    
    sample = {
        "text_id": semantic_id,
        "text": question,
        "tableID": table_id,
        "all_answer_tables": answer_tables,
        "all_semantic_ids": all_semantic_ids
    }
    birdie_test.append(sample)

print(f"Generated {len(birdie_test)} test samples (skipped {skipped})")

# Save as JSONL
with open("${OUTPUT}", "w") as f:
    for sample in birdie_test:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"Saved to ${OUTPUT}")
EOF
    
    if [ -f "${OUTPUT}" ]; then
        NUM_LINES=$(wc -l < "${OUTPUT}")
        echo -e "${GREEN}Generated ${NUM_LINES} test samples for ${DATASET}${NC}"
    fi
done

echo ""
echo -e "${GREEN}Done!${NC}"
