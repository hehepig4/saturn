#!/usr/bin/env python3
"""
Convert Unified Benchmark Format to Birdie Format.

This script converts data from the unified benchmark format
(/data/benchmark/unified/<dataset>/) to Birdie's JSON format.

Unified format:
- table/: Directory of JSON files, one per table
- query/train.jsonl, query/test.jsonl: JSONL files with queries

Birdie format:
- table_data.json: Dict mapping table_id -> table content
- question_data.json: List of {question, table_id} dicts

Usage:
    python convert_unified_to_birdie.py \
        --unified_path /path/to/unified/fetaqa \
        --output_dir /path/to/output
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm


def load_unified_tables(table_dir: str) -> Dict[str, Dict]:
    """Load tables from unified format (one JSON per table)."""
    table_data = {}
    table_path = Path(table_dir)
    
    table_files = list(table_path.glob("*.json"))
    print(f"Found {len(table_files)} table files")
    
    for table_file in tqdm(table_files, desc="Loading tables"):
        with open(table_file, 'r', encoding='utf-8') as f:
            table = json.load(f)
        
        # Get table ID
        table_id = table.get('table_id', table_file.stem)
        
        # Convert to Birdie format
        birdie_table = convert_table_to_birdie(table)
        table_data[table_id] = birdie_table
    
    return table_data


def convert_table_to_birdie(table: Dict) -> Dict:
    """
    Convert a table from unified format to Birdie format.
    
    Unified format:
    {
        "table_id": "...",
        "title": "...",
        "columns": ["col1", "col2"],
        "rows": [["val1", "val2"], ...]
    }
    
    Birdie format:
    {
        "documentTitle": "...",
        "columns": [{"text": "col1"}, {"text": "col2"}],
        "rows": [{"cells": [{"text": "val1"}, {"text": "val2"}]}]
    }
    """
    # Get title
    title = table.get('title', table.get('table_id', 'Unknown'))
    
    # Convert columns
    columns = table.get('columns', [])
    birdie_columns = [{"text": str(col)} for col in columns]
    
    # Convert rows
    rows = table.get('rows', [])
    birdie_rows = []
    for row in rows:
        if isinstance(row, list):
            cells = [{"text": str(cell) if cell is not None else ""} for cell in row]
        elif isinstance(row, dict):
            # Handle dict format (cells already as objects)
            cells = [{"text": str(cell.get('text', cell))} for cell in row.get('cells', [])]
        else:
            continue
        birdie_rows.append({"cells": cells})
    
    return {
        "documentTitle": title,
        "columns": birdie_columns,
        "rows": birdie_rows
    }


def load_unified_queries(query_dir: str, splits: List[str] = None) -> List[Dict]:
    """Load queries from unified format (JSONL files)."""
    if splits is None:
        splits = ['train', 'test']
    
    queries = []
    query_path = Path(query_dir)
    
    for split in splits:
        query_file = query_path / f"{split}.jsonl"
        if not query_file.exists():
            print(f"Warning: {query_file} not found")
            continue
        
        print(f"Loading queries from {query_file}")
        with open(query_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    query = json.loads(line)
                    queries.append(query)
    
    return queries


def convert_queries_to_birdie(queries: List[Dict], expand_multi_answer: bool = True) -> List[Dict]:
    """
    Convert queries from unified format to Birdie format.
    
    Unified format:
    {
        "query_id": "...",
        "question": "...",
        "answer_tables": ["table_id_1", ...],
        ...
    }
    
    Birdie format (for training, expand_multi_answer=True):
    {
        "question": "...",
        "table_id": "...",  # One table per entry (expanded)
        "all_answer_tables": ["table_id_1", ...]  # For reference
    }
    
    Birdie format (for evaluation, expand_multi_answer=False):
    {
        "question": "...",
        "table_id": "...",  # First answer table
        "all_answer_tables": ["table_id_1", ...]  # All correct answers for Hit@K
    }
    
    Args:
        queries: List of query dicts in unified format
        expand_multi_answer: If True, create one entry per answer table (for training).
                             If False, keep one entry per query with all answers (for eval).
    """
    birdie_queries = []
    
    for query in queries:
        question = query.get('question', '')
        answer_tables = query.get('answer_tables', [])
        query_id = query.get('query_id', '')
        
        if not question or not answer_tables:
            continue
        
        if expand_multi_answer:
            # For training: create one entry for each answer table
            for table_id in answer_tables:
                birdie_queries.append({
                    "question": question,
                    "table_id": table_id,
                    "all_answer_tables": answer_tables,
                    "query_id": query_id
                })
        else:
            # For evaluation: keep one entry with all answers
            birdie_queries.append({
                "question": question,
                "table_id": answer_tables[0],  # Primary answer for compatibility
                "all_answer_tables": answer_tables,  # All valid answers for Hit@K
                "query_id": query_id
            })
    
    return birdie_queries


def convert_unified_to_birdie(unified_path: str, 
                               output_dir: str,
                               query_splits: List[str] = None,
                               expand_multi_answer_train: bool = True):
    """Convert unified format to Birdie format.
    
    Args:
        unified_path: Path to unified dataset directory
        output_dir: Output directory for Birdie format files
        query_splits: Query splits to include
        expand_multi_answer_train: If True, expand multi-answer queries for training
    """
    os.makedirs(output_dir, exist_ok=True)
    
    unified_path = Path(unified_path)
    
    # Load and convert tables
    table_dir = unified_path / 'table'
    if table_dir.exists():
        table_data = load_unified_tables(str(table_dir))
        
        table_output = os.path.join(output_dir, 'table_data.json')
        with open(table_output, 'w', encoding='utf-8') as f:
            json.dump(table_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(table_data)} tables to {table_output}")
    else:
        print(f"Warning: Table directory not found: {table_dir}")
        table_data = {}
    
    # Load and convert queries
    query_dir = unified_path / 'query'
    if query_dir.exists():
        queries = load_unified_queries(str(query_dir), query_splits)
        
        # For general output, don't expand (keep one entry per query)
        birdie_queries = convert_queries_to_birdie(queries, expand_multi_answer=False)
        
        query_output = os.path.join(output_dir, 'question_data.json')
        with open(query_output, 'w', encoding='utf-8') as f:
            json.dump(birdie_queries, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(birdie_queries)} queries to {query_output}")
        
        # Only save test split (train queries should NOT be saved to avoid label leakage)
        # BIRDIE training must use synthetic queries generated by query_g.py / query_g_tllama.py
        test_queries = [q for q in queries if q.get('split') == 'test']
        
        # NOTE: train_queries.json is intentionally NOT generated here
        # to prevent accidental use of human-annotated training data
        
        if test_queries:
            # Test: keep one entry per query with all answers for evaluation
            test_output = os.path.join(output_dir, 'test_queries.json')
            test_birdie = convert_queries_to_birdie(
                test_queries, 
                expand_multi_answer=False  # Keep all answers for evaluation
            )
            with open(test_output, 'w', encoding='utf-8') as f:
                json.dump(test_birdie, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(test_birdie)} test queries to {test_output}")
            
            # Count multi-answer queries
            multi_answer_count = sum(1 for q in test_birdie if len(q.get('all_answer_tables', [])) > 1)
            print(f"  Multi-answer queries: {multi_answer_count}/{len(test_birdie)}")
    else:
        print(f"Warning: Query directory not found: {query_dir}")
    
    # Print summary
    print("\n=== Conversion Summary ===")
    print(f"Tables: {len(table_data)}")
    print(f"Queries: {len(birdie_queries) if 'birdie_queries' in dir() else 0}")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert unified format to Birdie format")
    parser.add_argument("--unified_path", required=True, type=str,
                        help="Path to unified dataset directory (e.g., /data/benchmark/unified/fetaqa)")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Output directory for Birdie format files")
    parser.add_argument("--query_splits", nargs='+', default=['train', 'test'],
                        help="Query splits to include (default: train test)")
    
    args = parser.parse_args()
    
    convert_unified_to_birdie(
        unified_path=args.unified_path,
        output_dir=args.output_dir,
        query_splits=args.query_splits
    )


if __name__ == "__main__":
    main()
