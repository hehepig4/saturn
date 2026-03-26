#!/usr/bin/env python3
"""
Convert unified benchmark data format to Pneuma format.

Unified format:
- table/*.json: {"table_id", "title", "columns": [], "rows": [[]]}
- query/test.jsonl: {"query_id", "question", "answer_tables": []}

Pneuma format:
- tables/*.csv: CSV files with header row
- queries/test.jsonl: {"id", "question", "answer_tables": []}

Usage:
    python convert_unified_to_pneuma.py --dataset chembl
    python convert_unified_to_pneuma.py --dataset chembl --unified-dir /path/to/unified
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


# Default paths
DEFAULT_UNIFIED_BASE = str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "benchmark" / "unified")


def sanitize_filename(name: str) -> str:
    """Convert table_id to safe filename."""
    # Replace problematic characters
    safe = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Limit length
    if len(safe) > 200:
        safe = safe[:200]
    return safe


def convert_table_to_csv(table_file: Path, output_dir: Path) -> str:
    """
    Convert a unified table JSON to Pneuma CSV format.
    
    Args:
        table_file: Path to unified table JSON
        output_dir: Output directory for CSV files
        
    Returns:
        table_id for mapping
    """
    with open(table_file, 'r', encoding='utf-8') as f:
        table = json.load(f)
    
    table_id = table['table_id']
    columns = table.get('columns', [])
    rows = table.get('rows', [])
    
    # Create safe filename
    safe_name = sanitize_filename(table_id)
    output_file = output_dir / f"{safe_name}.csv"
    
    # Write CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            # Ensure row has same length as columns
            padded_row = row + [''] * (len(columns) - len(row))
            writer.writerow(padded_row[:len(columns)])
    
    return table_id


def convert_tables(unified_dir: Path, output_dir: Path) -> Dict[str, str]:
    """
    Convert all unified tables to Pneuma CSV format.
    
    Args:
        unified_dir: Unified dataset directory
        output_dir: Output tables directory
        
    Returns:
        Dict mapping table_id to CSV filename
    """
    tables_dir = unified_dir / "table"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    table_files = list(tables_dir.glob("*.json"))
    print(f"Found {len(table_files)} tables")
    
    table_mapping = {}
    for table_file in tqdm(table_files, desc="Converting tables"):
        table_id = convert_table_to_csv(table_file, output_dir)
        safe_name = sanitize_filename(table_id)
        table_mapping[table_id] = f"{safe_name}.csv"
    
    # Save mapping
    mapping_file = output_dir.parent / "table_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(table_mapping, f, indent=2)
    
    print(f"Converted {len(table_mapping)} tables")
    return table_mapping


def convert_queries(unified_dir: Path, output_dir: Path, table_mapping: Dict[str, str]) -> int:
    """
    Convert unified queries to Pneuma format.
    
    Args:
        unified_dir: Unified dataset directory
        output_dir: Output queries directory
        table_mapping: Dict mapping table_id to CSV filename
        
    Returns:
        Number of queries converted
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    
    for split in ['train', 'test']:
        input_file = unified_dir / "query" / f"{split}.jsonl"
        if not input_file.exists():
            continue
        
        output_file = output_dir / f"{split}.jsonl"
        
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                
                query = json.loads(line)
                
                # Convert to Pneuma format
                pneuma_query = {
                    "id": query.get("query_id", ""),
                    "question": query.get("question", ""),
                    "answer_tables": query.get("answer_tables", []),
                }
                
                # Add answer if available
                if "answer" in query:
                    pneuma_query["answer"] = query["answer"]
                
                f_out.write(json.dumps(pneuma_query) + "\n")
                count += 1
    
    print(f"Converted {count} queries")
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert unified to Pneuma format")
    parser.add_argument("--dataset", "-d", required=True, help="Dataset name")
    parser.add_argument("--unified-dir", default=DEFAULT_UNIFIED_BASE, help="Unified benchmark base directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    unified_dir = Path(args.unified_dir) / args.dataset
    output_dir = Path(args.output_dir)
    
    if not unified_dir.exists():
        print(f"Error: Unified directory not found: {unified_dir}")
        sys.exit(1)
    
    print(f"Converting {args.dataset} from unified to Pneuma format")
    print(f"  Input: {unified_dir}")
    print(f"  Output: {output_dir}")
    print()
    
    # Convert tables
    tables_output = output_dir / "tables"
    table_mapping = convert_tables(unified_dir, tables_output)
    
    # Convert queries
    queries_output = output_dir / "queries"
    convert_queries(unified_dir, queries_output, table_mapping)
    
    print()
    print("Conversion complete!")


if __name__ == "__main__":
    main()
