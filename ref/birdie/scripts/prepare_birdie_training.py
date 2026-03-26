#!/usr/bin/env python3
"""
Prepare Birdie Training Data

Combines table data, queries, and semantic IDs into Birdie training format.

BIRDIE expected format (JSONL):
{"text_id": "<semantic_id>", "text": "<query_text>", "tableID": "<table_id>"}

Usage:
    python prepare_birdie_training.py \
        --table_data table_data.json \
        --query_data queries.json \
        --id_map id_map.json \
        --output birdie_train.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List


def load_json(path: str) -> Dict:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def json_to_markdown(table: Dict) -> str:
    """Convert table JSON to markdown format."""
    columns = [col.get('text', '') for col in table.get('columns', [])]
    rows = table.get('rows', [])
    
    # Header
    header = '|' + '|'.join(col if col else ' ' for col in columns) + '|'
    separator = '|' + '|'.join(['---'] * len(columns)) + '|'
    
    # Rows
    row_lines = []
    for row in rows:
        cells = [cell.get('text', '') for cell in row.get('cells', [])]
        row_lines.append('|' + '|'.join(cells) + '|')
    
    return '\n'.join([header, separator] + row_lines)


def prepare_training_data(table_data_path: str,
                          query_data_path: str,
                          id_map_path: str,
                          output_path: str):
    """Prepare training data in Birdie format."""
    
    # Load data
    print(f"Loading table data from {table_data_path}")
    table_data = load_json(table_data_path)
    
    print(f"Loading query data from {query_data_path}")
    query_data = load_json(query_data_path)
    
    print(f"Loading ID map from {id_map_path}")
    id_map_raw = load_json(id_map_path)
    
    # Convert id_map to dict format if it's a list
    # List format: [{"tableID": "...", "semantic_id": "..."}, ...]
    # Dict format: {"tableID": "semantic_id", ...}
    if isinstance(id_map_raw, list):
        id_map = {item["tableID"]: item["semantic_id"] for item in id_map_raw}
    else:
        id_map = id_map_raw
    
    # Handle different query data formats
    if isinstance(query_data, dict):
        # Format: {"table_id": [queries...]}
        queries = []
        for table_id, table_queries in query_data.items():
            for q in table_queries:
                if isinstance(q, str):
                    queries.append({"question": q, "table_id": table_id})
                elif isinstance(q, dict):
                    queries.append({
                        "question": q.get("question", q.get("query", "")),
                        "table_id": table_id
                    })
    elif isinstance(query_data, list):
        # Format: [{"question": ..., "table_id": ...}]
        queries = query_data
    else:
        raise ValueError(f"Unsupported query data format: {type(query_data)}")
    
    print(f"Total queries: {len(queries)}")
    
    # Prepare training samples
    training_data = []
    skipped = 0
    
    for query in queries:
        question = query.get("question", query.get("query", ""))
        table_id = query.get("table_id", query.get("target_table", ""))
        
        if not question or not table_id:
            skipped += 1
            continue
        
        # Get semantic ID
        semantic_id = id_map.get(table_id, None)
        if semantic_id is None:
            skipped += 1
            continue
        
        # Get table content (for context if needed)
        table = table_data.get(table_id, None)
        
        # Create training sample in BIRDIE format
        # BIRDIE expects: {"text_id": "<semantic_id>", "text": "<query_text>", "tableID": "<table_id>"}
        sample = {
            "text_id": semantic_id,
            "text": question,
            "tableID": table_id
        }
        
        training_data.append(sample)
    
    print(f"Training samples: {len(training_data)}")
    print(f"Skipped: {skipped}")
    
    # Save training data in JSONL format (BIRDIE expected format)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in training_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Birdie training data")
    parser.add_argument("--table_data", required=True, type=str,
                        help="Path to table_data.json")
    parser.add_argument("--query_data", required=True, type=str,
                        help="Path to query data (queries.json or question_data.json)")
    parser.add_argument("--id_map", required=True, type=str,
                        help="Path to id_map.json (semantic IDs)")
    parser.add_argument("--output", required=True, type=str,
                        help="Output path for training data")
    
    args = parser.parse_args()
    
    prepare_training_data(
        table_data_path=args.table_data,
        query_data_path=args.query_data,
        id_map_path=args.id_map,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
