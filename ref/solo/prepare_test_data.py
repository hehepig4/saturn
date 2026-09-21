#!/usr/bin/env python3
"""
Prepare test data for Solo evaluation.

Converts test_queries.jsonl format to fusion_query.jsonl format expected by Solo tester.
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare test data for Solo evaluation")
    parser.add_argument("--data-dir", required=True, help="Solo data directory")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    dataset_dir = data_dir / args.dataset
    test_query_dir = dataset_dir / "query" / "test"
    
    # Check for existing test_queries.jsonl from converter
    test_queries_file = test_query_dir / "test_queries.jsonl"
    fusion_query_file = test_query_dir / "fusion_query.jsonl"
    
    # Also check the train directory for the pattern
    train_query_dir = dataset_dir / "query" / "train"
    
    # If fusion_query.jsonl already exists with content, skip
    if fusion_query_file.exists() and fusion_query_file.stat().st_size > 0:
        print(f"  fusion_query.jsonl already exists ({fusion_query_file.stat().st_size} bytes), skipping")
        return
    
    # Check if test_queries.jsonl exists
    if not test_queries_file.exists():
        print(f"Warning: {test_queries_file} not found")
        # Try to find queries from train directory format
        if not test_query_dir.exists():
            print(f"Error: {test_query_dir} does not exist")
            return
        
        # List files in the directory
        files = list(test_query_dir.glob("*.jsonl"))
        print(f"  Files in {test_query_dir}: {[f.name for f in files]}")
        
        if not files:
            print("Error: No jsonl files found in test query directory")
            return
    
    # Convert test_queries.jsonl to fusion_query.jsonl
    print(f"  Converting {test_queries_file} to {fusion_query_file}...")
    
    count = 0
    with open(test_queries_file, 'r') as f_in, open(fusion_query_file, 'w') as f_out:
        for line in f_in:
            if not line.strip():
                continue
            
            data = json.loads(line)
            
            # Convert format:
            # Input: {qid, question, table_id_lst}
            # Output: {id, question, table_id_lst, answers, ctxs}
            converted = {
                "id": data.get("qid", data.get("id", str(count))),
                "question": data["question"],
                "table_id_lst": data["table_id_lst"],
                "answers": ["N/A"],  # Solo format requires answers field
                "ctxs": []  # Empty contexts, will be filled by retrieval
            }
            
            f_out.write(json.dumps(converted) + "\n")
            count += 1
    
    print(f"  Converted {count} test queries to fusion_query.jsonl")


if __name__ == "__main__":
    main()
