#!/usr/bin/env python3
"""
Convert unified benchmark data format to Solo format.

Unified format:
- table/table_X.json: {"table_id", "title", "columns": [], "rows": [[]]}
- query/test.jsonl: {"query_id", "question", "answer_tables": []}

Solo format:
- tables/tables.jsonl: {"tableId", "documentTitle", "columns": [{"text":...}], "rows": [{"cells": [{"text":...}]}]}
- test/test_queries.jsonl: {"qid", "question", "table_id", "answer_tables": []}

Usage:
    python convert_unified_to_solo.py --dataset chembl
    python convert_unified_to_solo.py --dataset chembl --unified-dir /path/to/unified --output-dir /path/to/output
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


# Default paths
DEFAULT_UNIFIED_BASE = str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "benchmark" / "unified")
DEFAULT_OUTPUT_BASE = str(Path(__file__).resolve().parent.parent.parent.parent / "ref" / "data")

# Passage limit (1 billion = 1G)
DEFAULT_PASSAGE_LIMIT = 1_000_000_000


def convert_table_to_solo(table_file: str, dataset_name: str) -> dict:
    """
    Convert a unified table JSON file to Solo format.
    
    Args:
        table_file: Path to the unified table JSON file
        dataset_name: Name of the dataset for URL field
        
    Returns:
        Table dict in Solo format
    """
    with open(table_file, 'r', encoding='utf-8') as f:
        unified_table = json.load(f)
    
    # Convert to Solo format
    solo_table = {
        'tableId': unified_table['table_id'],
        'documentTitle': unified_table.get('title', ''),
        'documentUrl': f"{dataset_name}/{unified_table.get('title', unified_table['table_id'])}",
        'columns': [{'text': col} for col in unified_table['columns']],
        'rows': [
            {'cells': [{'text': str(cell) if cell is not None else ''} for cell in row]}
            for row in unified_table['rows']
        ]
    }
    
    return solo_table


def convert_tables(
    input_dir: str, 
    output_file: str, 
    dataset_name: str,
    passage_limit: int = DEFAULT_PASSAGE_LIMIT
) -> Tuple[Dict[str, str], int, bool]:
    """
    Convert all unified table JSON files to Solo format.
    
    Args:
        input_dir: Directory containing unified table JSON files
        output_file: Path to output tables.jsonl
        dataset_name: Name of the dataset
        passage_limit: Maximum number of passages (cells) allowed
        
    Returns:
        Tuple of (table_mapping, total_passages, exceeded_limit)
    """
    # Support both table_*.json and *.json filename patterns
    table_files = sorted(Path(input_dir).glob('table_*.json'))
    
    if not table_files:
        # Fallback to any JSON files (for datasets like fetaqa with UUID-style names)
        table_files = sorted(Path(input_dir).glob('*.json'))
    
    if not table_files:
        raise ValueError(f"No table files found in {input_dir}")
    
    print(f"Found {len(table_files)} table files")
    
    table_mapping = {}
    total_passages = 0
    exceeded_limit = False
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # First pass: count passages
    print("Counting passages...")
    for table_file in tqdm(table_files, desc="Counting"):
        with open(table_file, 'r', encoding='utf-8') as f:
            unified_table = json.load(f)
        num_rows = len(unified_table.get('rows', []))
        num_cols = len(unified_table.get('columns', []))
        total_passages += num_rows * num_cols
    
    print(f"Total passages: {total_passages:,}")
    
    if total_passages > passage_limit:
        print(f"⚠️  Passage count ({total_passages:,}) exceeds limit ({passage_limit:,})")
        print("    This dataset will be skipped during training/evaluation (treated as OOM)")
        exceeded_limit = True
    
    # Second pass: convert
    total_rows = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for table_file in tqdm(table_files, desc="Converting tables"):
            solo_table = convert_table_to_solo(str(table_file), dataset_name)
            table_mapping[solo_table['tableId']] = solo_table['documentTitle']
            total_rows += len(solo_table['rows'])
            f_out.write(json.dumps(solo_table) + '\n')
    
    print(f"Converted {len(table_files)} tables with {total_rows} total rows")
    
    return table_mapping, total_passages, exceeded_limit


def convert_queries(
    input_file: str, 
    output_file: str, 
    table_mapping: Dict[str, str],
    split_name: str = "test"
) -> int:
    """
    Convert unified query JSONL to Solo format.
    
    Args:
        input_file: Path to unified query JSONL file
        output_file: Path to output queries JSONL file
        table_mapping: Dict mapping table_id to table_title
        split_name: Name of the split (test, train, etc.)
        
    Returns:
        Number of queries converted
    
    Note:
        Solo uses `table_id_lst` (list) for multi-table answer support.
        P@K calculation checks if retrieved table_id is in table_id_lst.
    """
    if not os.path.exists(input_file):
        print(f"Query file not found: {input_file}")
        return 0
    
    queries = []
    multi_table_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            # Get answer tables list
            answer_tables = item.get('answer_tables', [])
            if len(answer_tables) > 1:
                multi_table_count += 1
            
            # Build Solo query format
            # Solo expects `table_id_lst` (list) for multi-table P@K evaluation
            query = {
                'qid': item.get('query_id', item.get('id', '')),
                'question': item['question'],
                'table_id_lst': answer_tables,  # Solo uses this for P@K
            }
            
            # Add table titles if available
            table_titles = [table_mapping.get(tid, '') for tid in answer_tables if tid in table_mapping]
            if table_titles:
                query['table_title_lst'] = table_titles
            
            # Preserve metadata if present
            if 'metadata' in item:
                query['meta'] = item['metadata']
            
            queries.append(query)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for query in queries:
            f_out.write(json.dumps(query) + '\n')
    
    print(f"Converted {len(queries)} {split_name} queries ({multi_table_count} with multiple answer tables)")
    return len(queries)


def convert_dataset(
    dataset: str,
    unified_dir: str = DEFAULT_UNIFIED_BASE,
    output_dir: str = DEFAULT_OUTPUT_BASE,
    passage_limit: int = DEFAULT_PASSAGE_LIMIT
) -> dict:
    """
    Convert a dataset from unified format to Solo format.
    
    Args:
        dataset: Name of the dataset (e.g., 'chembl', 'fetaqapn')
        unified_dir: Base directory for unified benchmark data
        output_dir: Base directory for Solo format output
        passage_limit: Maximum number of passages allowed
        
    Returns:
        Conversion statistics
    """
    unified_dataset_dir = os.path.join(unified_dir, dataset)
    output_dataset_dir = os.path.join(output_dir, dataset)
    
    if not os.path.exists(unified_dataset_dir):
        raise ValueError(f"Dataset directory not found: {unified_dataset_dir}")
    
    print(f"\n{'='*60}")
    print(f"Converting dataset: {dataset}")
    print(f"{'='*60}")
    print(f"Input:  {unified_dataset_dir}")
    print(f"Output: {output_dataset_dir}")
    
    # Convert tables
    tables_input_dir = os.path.join(unified_dataset_dir, 'table')
    tables_output_file = os.path.join(output_dataset_dir, 'tables', 'tables.jsonl')
    
    table_mapping, total_passages, exceeded_limit = convert_tables(
        tables_input_dir, 
        tables_output_file, 
        dataset,
        passage_limit
    )
    
    # Save table mapping
    mapping_file = os.path.join(output_dataset_dir, 'table_id_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(table_mapping, f, indent=2)
    print(f"Saved table mapping to {mapping_file}")
    
    # Save passage count and limit info
    stats_file = os.path.join(output_dataset_dir, 'dataset_stats.json')
    stats = {
        'dataset': dataset,
        'num_tables': len(table_mapping),
        'total_passages': total_passages,
        'passage_limit': passage_limit,
        'exceeded_limit': exceeded_limit,
    }
    
    # Convert queries (test and train if available)
    queries_dir = os.path.join(unified_dataset_dir, 'query')
    
    for split in ['test', 'train', 'dev']:
        query_input = os.path.join(queries_dir, f'{split}.jsonl')
        if os.path.exists(query_input):
            query_output = os.path.join(output_dataset_dir, split, f'{split}_queries.jsonl')
            num_queries = convert_queries(query_input, query_output, table_mapping, split)
            stats[f'{split}_queries'] = num_queries
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved dataset stats to {stats_file}")
    
    print(f"\n✓ Conversion complete for {dataset}")
    if exceeded_limit:
        print(f"⚠️  WARNING: Dataset has {total_passages:,} passages, exceeding limit of {passage_limit:,}")
        print("   This dataset will be automatically skipped during training/evaluation.")
    
    return stats


def list_available_datasets(unified_dir: str = DEFAULT_UNIFIED_BASE) -> List[str]:
    """List all available datasets in the unified directory."""
    datasets = []
    for item in Path(unified_dir).iterdir():
        if item.is_dir():
            table_dir = item / 'table'
            if table_dir.exists():
                datasets.append(item.name)
    return sorted(datasets)


def main():
    parser = argparse.ArgumentParser(
        description='Convert unified benchmark data to Solo format'
    )
    parser.add_argument(
        '--dataset', type=str, default=None,
        help='Dataset name to convert (e.g., chembl, fetaqapn). If not specified, list available datasets.'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Convert all available datasets'
    )
    parser.add_argument(
        '--unified-dir', type=str, default=DEFAULT_UNIFIED_BASE,
        help=f'Base directory for unified benchmark data (default: {DEFAULT_UNIFIED_BASE})'
    )
    parser.add_argument(
        '--output-dir', type=str, default=DEFAULT_OUTPUT_BASE,
        help=f'Base directory for Solo format output (default: {DEFAULT_OUTPUT_BASE})'
    )
    parser.add_argument(
        '--passage-limit', type=int, default=DEFAULT_PASSAGE_LIMIT,
        help=f'Maximum number of passages allowed (default: {DEFAULT_PASSAGE_LIMIT:,})'
    )
    
    args = parser.parse_args()
    
    if args.dataset is None and not args.all:
        # List available datasets
        print(f"Available datasets in {args.unified_dir}:")
        datasets = list_available_datasets(args.unified_dir)
        for ds in datasets:
            print(f"  - {ds}")
        print(f"\nUse --dataset <name> to convert a specific dataset")
        print(f"Use --all to convert all datasets")
        return
    
    if args.all:
        datasets = list_available_datasets(args.unified_dir)
        print(f"Converting all {len(datasets)} datasets...")
        all_stats = {}
        for ds in datasets:
            try:
                stats = convert_dataset(
                    ds, args.unified_dir, args.output_dir, args.passage_limit
                )
                all_stats[ds] = stats
            except Exception as e:
                print(f"Error converting {ds}: {e}")
                all_stats[ds] = {'error': str(e)}
        
        # Print summary
        print(f"\n{'='*60}")
        print("Conversion Summary")
        print(f"{'='*60}")
        for ds, stats in all_stats.items():
            if 'error' in stats:
                print(f"  {ds}: ERROR - {stats['error']}")
            elif stats.get('exceeded_limit'):
                print(f"  {ds}: ⚠️  SKIPPED (passages: {stats['total_passages']:,} > {stats['passage_limit']:,})")
            else:
                print(f"  {ds}: ✓ {stats['num_tables']} tables, {stats.get('test_queries', 0)} test queries")
    else:
        convert_dataset(
            args.dataset, args.unified_dir, args.output_dir, args.passage_limit
        )


if __name__ == '__main__':
    main()
