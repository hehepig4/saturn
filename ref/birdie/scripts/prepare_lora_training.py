#!/usr/bin/env python3
"""
Prepare Training Data for Query Generator LoRA

This script prepares training data from unified benchmark format or BIRDIE format
for training the Query Generator LoRA adapter.

Input formats supported:
1. Unified benchmark format: question_data.json + tables.jsonl
2. BIRDIE format: train.json with question-table pairs

Output format (JSONL):
{"tableId": "...", "question": "...", "caption": "...", "table_markdown": "..."}
or
{"tableId": "...", "text": "...", ...}  (if using existing BIRDIE train data)

Usage:
    # From unified benchmark
    python prepare_lora_training.py \
        --unified_path /path/to/unified/dataset \
        --output_path /path/to/output/train_lora.jsonl
    
    # From BIRDIE format
    python prepare_lora_training.py \
        --birdie_train /path/to/train.json \
        --table_data /path/to/table_data.json \
        --output_path /path/to/output/train_lora.jsonl
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_unified_tables(tables_path: str) -> Dict[str, Dict]:
    """Load tables from unified benchmark format."""
    tables = {}
    with open(tables_path, 'r', encoding='utf-8') as f:
        for line in f:
            table = json.loads(line.strip())
            table_id = table.get('table_id', table.get('tableId'))
            tables[table_id] = table
    return tables


def load_unified_questions(questions_path: str) -> List[Dict]:
    """Load questions from unified benchmark format."""
    questions = []
    with open(questions_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            questions = data
        elif isinstance(data, dict):
            # Handle different structures
            questions = data.get('questions', data.get('data', []))
    return questions


def load_birdie_tables(table_data_path: str) -> Dict[str, Dict]:
    """Load tables from BIRDIE table_data.json format."""
    tables = {}
    with open(table_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            table = json.loads(line.strip())
            tables[table['tableId']] = table
    return tables


def json_to_markdown(table_data: Dict) -> str:
    """Convert table JSON to markdown format."""
    columns = table_data.get('columns', [])
    rows = table_data.get('rows', [])
    
    if not columns:
        return ""
    
    # Extract column texts
    col_texts = []
    for col in columns:
        if isinstance(col, dict):
            col_texts.append(col.get('text', ''))
        else:
            col_texts.append(str(col))
    
    # Header
    header = '|' + '|'.join(col if col else ' ' for col in col_texts) + '|'
    separator = '|' + '|'.join(['---'] * len(col_texts)) + '|'
    
    # Rows
    row_lines = []
    for row in rows:
        if isinstance(row, dict):
            cells = row.get('cells', [])
            cell_texts = []
            for cell in cells:
                if isinstance(cell, dict):
                    cell_texts.append(cell.get('text', ''))
                else:
                    cell_texts.append(str(cell))
        else:
            cell_texts = [str(c) for c in row]
        row_lines.append('|' + '|'.join(cell_texts) + '|')
    
    return '\n'.join([header, separator] + row_lines)


def load_unified_tables_from_dir(table_dir: str) -> Dict[str, Dict]:
    """Load tables from unified benchmark format (directory of JSON files)."""
    from tqdm import tqdm
    tables = {}
    table_path = Path(table_dir)
    
    table_files = list(table_path.glob("*.json"))
    print(f"Found {len(table_files)} table files")
    
    for table_file in tqdm(table_files, desc="Loading tables"):
        with open(table_file, 'r', encoding='utf-8') as f:
            table = json.load(f)
        
        table_id = table.get('table_id', table_file.stem)
        tables[table_id] = table
    
    return tables


def prepare_from_unified(
    unified_path: str,
    output_path: str,
    split: str = 'train',
    max_samples: Optional[int] = None,
    include_table_markdown: bool = True,
) -> int:
    """Prepare training data from unified benchmark format."""
    unified_path = Path(unified_path)
    
    # Load tables - support both directory and JSONL format
    table_dir = unified_path / "table"
    tables_jsonl = unified_path / "tables" / "tables.jsonl"
    tables_jsonl_alt = unified_path / "tables.jsonl"
    
    if table_dir.exists() and table_dir.is_dir():
        print(f"Loading tables from directory {table_dir}")
        tables = load_unified_tables_from_dir(str(table_dir))
    elif tables_jsonl.exists():
        print(f"Loading tables from {tables_jsonl}")
        tables = load_unified_tables(str(tables_jsonl))
    elif tables_jsonl_alt.exists():
        print(f"Loading tables from {tables_jsonl_alt}")
        tables = load_unified_tables(str(tables_jsonl_alt))
    else:
        raise FileNotFoundError(f"No table data found in {unified_path}")
    
    print(f"Loaded {len(tables)} tables")
    
    # Load questions - support both query/train.jsonl and question_data.json
    query_file = unified_path / "query" / f"{split}.jsonl"
    questions_path = unified_path / "question_data.json"
    questions_path_alt = unified_path / "questions" / "question_data.json"
    
    if query_file.exists():
        print(f"Loading questions from {query_file}")
        questions = []
        with open(query_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    questions.append(q)
    elif questions_path.exists():
        print(f"Loading questions from {questions_path}")
        questions = load_unified_questions(str(questions_path))
        # Filter by split
        if split:
            questions = [q for q in questions if q.get('split') == split]
    elif questions_path_alt.exists():
        print(f"Loading questions from {questions_path_alt}")
        questions = load_unified_questions(str(questions_path_alt))
        # Filter by split
        if split:
            questions = [q for q in questions if q.get('split') == split]
    else:
        raise FileNotFoundError(f"No question data found in {unified_path}")
    
    print(f"Questions in '{split}' split: {len(questions)}")
    
    # Limit samples
    if max_samples and len(questions) > max_samples:
        questions = random.sample(questions, max_samples)
        print(f"Sampled {max_samples} questions")
    
    # Prepare output
    count = 0
    missing_tables = set()
    with open(output_path, 'w', encoding='utf-8') as f:
        for q in questions:
            # Support both formats:
            # - table_id: single table ID
            # - answer_tables: list of table IDs (create one sample per table)
            table_ids = q.get('answer_tables', [])
            if not table_ids:
                table_id = q.get('table_id', q.get('tableId'))
                if table_id:
                    table_ids = [table_id]
            
            question = q.get('question', q.get('text'))
            
            if not table_ids or not question:
                continue
            
            # Create one training sample per answer table
            for table_id in table_ids:
                table = tables.get(table_id)
                if not table:
                    missing_tables.add(table_id)
                    continue
                
                # Get caption/title
                caption = (
                    table.get('title') or
                    table.get('documentTitle') or
                    table.get('caption') or
                    table.get('page_title', '')
                )
                
                item = {
                    'tableId': table_id,
                    'text': question,
                    'caption': caption,
                }
                
                if include_table_markdown:
                    item['table_markdown'] = json_to_markdown(table)
                
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                count += 1
    
    if missing_tables:
        print(f"Warning: {len(missing_tables)} tables not found: {list(missing_tables)[:5]}...")
    
    return count


def prepare_from_birdie(
    train_data_path: str,
    table_data_path: str,
    output_path: str,
    max_samples: Optional[int] = None,
    include_table_markdown: bool = True,
) -> int:
    """Prepare training data from BIRDIE format."""
    
    # Load tables
    print(f"Loading tables from {table_data_path}")
    tables = load_birdie_tables(table_data_path)
    print(f"Loaded {len(tables)} tables")
    
    # Load existing training data
    print(f"Loading training data from {train_data_path}")
    samples = []
    with open(train_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    print(f"Loaded {len(samples)} training samples")
    
    # Limit samples
    if max_samples and len(samples) > max_samples:
        samples = random.sample(samples, max_samples)
        print(f"Sampled {max_samples} samples")
    
    # Prepare output
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            table_id = sample.get('tableID') or sample.get('tableId')
            question = sample.get('text') or sample.get('question')
            
            if not table_id or not question:
                continue
            
            table = tables.get(table_id)
            
            item = {
                'tableId': table_id,
                'text': question,
            }
            
            if table:
                item['caption'] = table.get('documentTitle', table.get('title', ''))
                if include_table_markdown:
                    item['table_markdown'] = json_to_markdown(table)
            else:
                item['caption'] = sample.get('caption', '')
            
            # Copy other fields
            if 'text_id' in sample:
                item['text_id'] = sample['text_id']
            
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            count += 1
    
    return count


def main():
    parser = argparse.ArgumentParser(description='Prepare LoRA Training Data')
    
    # Input options (choose one)
    parser.add_argument('--unified_path', type=str, default=None,
                        help='Path to unified benchmark dataset directory')
    parser.add_argument('--birdie_train', type=str, default=None,
                        help='Path to BIRDIE train.json')
    parser.add_argument('--table_data', type=str, default=None,
                        help='Path to table_data.json (for BIRDIE format)')
    
    # Output
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path for prepared training data')
    
    # Options
    parser.add_argument('--split', type=str, default='train',
                        help='Split to use (for unified format)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to include')
    parser.add_argument('--no_table_markdown', action='store_true',
                        help='Do not include table markdown in output')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    
    # Validate arguments
    if not args.unified_path and not args.birdie_train:
        parser.error("Either --unified_path or --birdie_train must be provided")
    
    if args.birdie_train and not args.table_data:
        parser.error("--table_data is required when using --birdie_train")
    
    # Prepare data
    include_table_markdown = not args.no_table_markdown
    
    if args.unified_path:
        count = prepare_from_unified(
            args.unified_path,
            args.output_path,
            split=args.split,
            max_samples=args.max_samples,
            include_table_markdown=include_table_markdown,
        )
    else:
        count = prepare_from_birdie(
            args.birdie_train,
            args.table_data,
            args.output_path,
            max_samples=args.max_samples,
            include_table_markdown=include_table_markdown,
        )
    
    print(f"\nPrepared {count} training samples")
    print(f"Output saved to: {args.output_path}")


if __name__ == '__main__':
    main()
