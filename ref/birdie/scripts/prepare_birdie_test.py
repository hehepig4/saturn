#!/usr/bin/env python3
"""
Prepare Birdie Test Data

Converts test queries to BIRDIE JSONL format while preserving multi-answer info.

BIRDIE expected format (JSONL):
{"text_id": "<semantic_id>", "text": "<query_text>", "tableID": "<table_id>", "all_answer_tables": [...]}

Usage:
    python prepare_birdie_test.py \
        --query_data test_queries.json \
        --id_map id_map.json \
        --output birdie_test.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List


def load_json(path: str):
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_test_data(query_data_path: str,
                      id_map_path: str,
                      output_path: str):
    """Prepare test data in Birdie JSONL format."""
    
    # Load data
    print(f"Loading query data from {query_data_path}")
    query_data = load_json(query_data_path)
    
    print(f"Loading ID map from {id_map_path}")
    id_map_raw = load_json(id_map_path)
    
    # Convert id_map to dict format if it's a list
    if isinstance(id_map_raw, list):
        id_map = {item["tableID"]: item["semantic_id"] for item in id_map_raw}
    else:
        id_map = id_map_raw
    
    # Handle different query data formats
    if isinstance(query_data, dict):
        queries = []
        for table_id, table_queries in query_data.items():
            for q in table_queries:
                if isinstance(q, str):
                    queries.append({"question": q, "table_id": table_id, "all_answer_tables": [table_id]})
                elif isinstance(q, dict):
                    queries.append({
                        "question": q.get("question", q.get("query", "")),
                        "table_id": table_id,
                        "all_answer_tables": q.get("all_answer_tables", [table_id])
                    })
    elif isinstance(query_data, list):
        queries = query_data
    else:
        raise ValueError(f"Unsupported query data format: {type(query_data)}")
    
    print(f"Total queries: {len(queries)}")
    
    # Prepare test samples
    test_data = []
    skipped = 0
    
    for query in queries:
        question = query.get("question", query.get("query", ""))
        table_id = query.get("table_id", query.get("target_table", ""))
        all_answer_tables = query.get("all_answer_tables", [table_id])
        
        if not question or not table_id:
            skipped += 1
            continue
        
        # Get semantic ID for primary table
        semantic_id = id_map.get(table_id, None)
        if semantic_id is None:
            skipped += 1
            continue
        
        # Convert all_answer_tables to semantic IDs
        all_answer_semantic_ids = []
        for ans_table in all_answer_tables:
            ans_semantic_id = id_map.get(ans_table, None)
            if ans_semantic_id is not None:
                all_answer_semantic_ids.append(ans_semantic_id)
        
        if not all_answer_semantic_ids:
            all_answer_semantic_ids = [semantic_id]
        
        # Create test sample in BIRDIE format
        # NOTE: Use 'all_semantic_ids' key to match MultiAnswerQueryDataset in data.py
        sample = {
            "text_id": semantic_id,
            "text": question,
            "tableID": table_id,
            "all_semantic_ids": all_answer_semantic_ids  # Fixed: was 'all_answer_tables'
        }
        
        test_data.append(sample)
    
    print(f"Test samples: {len(test_data)}")
    print(f"Skipped: {skipped}")
    
    # Count multi-answer queries
    multi_answer = sum(1 for s in test_data if len(s.get("all_semantic_ids", [])) > 1)
    print(f"Multi-answer queries: {multi_answer}/{len(test_data)}")
    
    # Save test data in JSONL format
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in test_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Birdie test data")
    parser.add_argument("--query_data", required=True, type=str,
                        help="Path to test query data")
    parser.add_argument("--id_map", required=True, type=str,
                        help="Path to id_map.json (semantic IDs)")
    parser.add_argument("--output", required=True, type=str,
                        help="Output path for test data")
    
    args = parser.parse_args()
    
    prepare_test_data(
        query_data_path=args.query_data,
        id_map_path=args.id_map,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
