#!/usr/bin/env python3
"""
Unified Benchmark Data Processor

Converts raw benchmark data from various sources (Pneuma, Solo) into a unified
format for consistent evaluation across all datasets.

Supported datasets:
    - adventure_works: Pneuma format (CSV tables + annotated JSONL queries)
    - bird: Pneuma format (CSV tables + annotated JSONL queries)
    - chembl: Pneuma format (CSV tables + annotated JSONL queries)
    - chicago: Pneuma format (CSV tables + annotated JSONL queries)
    - fetaqapn: Pneuma format (CSV tables + annotated JSONL queries)
    - public_bi: Pneuma format (CSV tables + annotated JSONL queries)
    - fetaqa: Solo format (JSONL tables + query directories)

Output format:
    unified/<dataset>/
    ├── table/                 # One JSON file per table
    │   ├── <table_id>.json
    │   └── ...
    └── query/
        ├── train.jsonl        # Training queries
        └── test.jsonl         # Test queries

Usage:
    # Process all datasets with default config (defined in DEFAULT_PROCESSING_CONFIG)
    python scripts/unify_benchmark_data.py --all

    # Process single dataset with default config
    python scripts/unify_benchmark_data.py --dataset bird

    # Process single dataset with overrides
    python scripts/unify_benchmark_data.py --dataset fetaqapn --train-queries 2000 --translate

    # Use custom JSON config array: [[name, train_queries, translate], ...]
    python scripts/unify_benchmark_data.py --config '[["bird", 500, false], ["fetaqapn", 2000, true]]'

Configuration:
    Modify DEFAULT_PROCESSING_CONFIG array to customize default settings:
    Each entry is (dataset_name, train_queries, use_translate)
"""

import csv
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from loguru import logger
from tqdm import tqdm

# Increase CSV field size limit for large cells
csv.field_size_limit(sys.maxsize)

# Path setup for standalone execution
SOURCE_DIR = Path(__file__).resolve().parent.parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

PROJECT_ROOT = SOURCE_DIR.parent
BENCHMARK_RAW_DIR = PROJECT_ROOT / "data" / "benchmark" / "raw"
BENCHMARK_UNIFIED_DIR = PROJECT_ROOT / "data" / "benchmark" / "unified"


# ============ Dataset Configurations ============

# Dataset type: 'pneuma' or 'solo'
DATASET_CONFIGS = {
    "adventure_works": {"type": "pneuma"},
    "bird": {"type": "pneuma"},
    "chembl": {"type": "pneuma"},
    "chicago": {"type": "pneuma"},
    "fetaqapn": {"type": "pneuma"},  # FeTaQA Pneuma version
    "public_bi": {"type": "pneuma"},
    "fetaqa": {"type": "solo"},       # FeTaQA Solo version (original)
    "debug": {"type": "solo"},        # Debug mock dataset for testing
}


# ============ Utility Functions ============

def is_non_ascii(text: str) -> bool:
    """Check if text contains non-ASCII characters."""
    return bool(re.search(r'[^\x00-\x7F]', text))


def detect_language(text: str) -> str:
    """
    Simple language detection based on character analysis.
    Returns 'en' for ASCII-only text, or language hint for non-ASCII.
    """
    if not text:
        return "en"
    
    non_ascii_chars = [c for c in text if ord(c) > 127]
    if not non_ascii_chars:
        return "en"
    
    # Simple heuristic based on Unicode ranges
    for char in non_ascii_chars:
        code = ord(char)
        if 0x0E00 <= code <= 0x0E7F:  # Thai
            return "th"
        if 0x0400 <= code <= 0x04FF:  # Cyrillic
            return "ru"
        if 0x4E00 <= code <= 0x9FFF:  # CJK
            return "zh"
        if 0x0100 <= code <= 0x017F:  # Latin Extended-A
            return "extended_latin"
    
    return "non_ascii"


def parse_pneuma_filename(filename: str) -> Tuple[str, str]:
    """
    Parse Pneuma table filename to extract table_id and title.
    
    Formats:
        - "{folder},{subfolder}_SEP_{db_id}-#-{table_name}.csv" (BIRD/Chicago Pneuma format)
        - "{db_id}-#-{table_name}.csv" (simple BIRD format)
        - "{Title}_SEP_table_{num}.csv" (standard Pneuma format)
    
    Returns:
        Tuple of (table_id, title)
    """
    base_name = filename.replace(".csv", "")
    
    # BIRD/Chicago format with _SEP_ prefix: {folder},{subfolder}_SEP_{db_id}-#-{table}
    # Example: "address,alias_SEP_address-#-alias" -> table_id = "address-#-alias"
    if "_SEP_" in base_name and "-#-" in base_name:
        # Extract the part after _SEP_ as the actual table_id
        _, table_part = base_name.rsplit("_SEP_", 1)
        if "-#-" in table_part:
            db_id, table_name = table_part.split("-#-", 1)
            table_id = table_part  # "{db_id}-#-{table_name}"
            title = f"{db_id} - {table_name}"
            return table_id, title
    
    # Simple BIRD format: db_id-#-table_name (no _SEP_ prefix)
    if "-#-" in base_name and "_SEP_" not in base_name:
        parts = base_name.split("-#-", 1)
        db_id = parts[0]
        table_name = parts[1] if len(parts) > 1 else ""
        table_id = base_name  # Use full base_name as ID
        title = f"{db_id} - {table_name}"
        return table_id, title
    
    # Standard Pneuma format: Title_SEP_table_num
    if "_SEP_table_" in base_name:
        title_part, table_num = base_name.rsplit("_SEP_table_", 1)
        table_id = f"table_{table_num}"
        title = title_part.replace("_", " ")
        return table_id, title
    
    # Fallback: use full filename
    return base_name, base_name.replace("_", " ")


def load_csv_table(filepath: Path) -> Tuple[List[str], List[List[str]]]:
    """Load CSV table and return columns and rows."""
    columns = []
    rows = []
    
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                columns = row
            else:
                rows.append(row)
    
    return columns, rows


# ============ Pneuma Format Processing ============

def process_pneuma_tables(
    tables_dir: Path,
    dataset_name: str,
    output_dir: Path
) -> Dict[str, str]:
    """
    Process Pneuma format tables (CSV files) into unified JSON format.
    
    Args:
        tables_dir: Directory containing CSV tables
        dataset_name: Name of the dataset
        output_dir: Output directory for JSON files
    
    Returns:
        Mapping of original table IDs to unified table IDs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    table_id_mapping = {}
    csv_files = list(tables_dir.glob("*.csv"))
    
    logger.info(f"Processing {len(csv_files)} tables from {dataset_name}...")
    
    for csv_file in tqdm(csv_files, desc="Processing tables"):
        try:
            table_id, title = parse_pneuma_filename(csv_file.name)
            columns, rows = load_csv_table(csv_file)
            
            # Create unified table JSON
            table_data = {
                "table_id": table_id,
                "title": title,
                "section_title": "",
                "source_file": csv_file.name,
                "columns": columns,
                "column_count": len(columns),
                "row_count": len(rows),
                "rows": rows,
                "metadata": {
                    "dataset": dataset_name,
                    "original_id": csv_file.stem,
                    "source_format": "pneuma_csv"
                }
            }
            
            # Save to output directory
            # Use sanitized filename
            safe_id = re.sub(r'[^\w\-]', '_', table_id)
            output_file = output_dir / f"{safe_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(table_data, f, ensure_ascii=False, indent=2)
            
            # Map original filename (without extension) to table_id
            table_id_mapping[csv_file.stem] = table_id
            
        except Exception as e:
            logger.warning(f"Failed to process {csv_file.name}: {e}")
    
    logger.info(f"Processed {len(table_id_mapping)} tables")
    return table_id_mapping


def process_pneuma_queries(
    queries_file: Path,
    dataset_name: str,
    output_dir: Path,
    train_queries: int,
    translate: bool = False,
    llm=None
) -> Tuple[int, int]:
    """
    Process Pneuma format queries (annotated JSONL) into unified format.
    
    Args:
        queries_file: Path to annotated JSONL file
        dataset_name: Name of the dataset
        output_dir: Output directory for train/test JSONL files
        train_queries: Number of queries for training set
        translate: Whether to translate non-English queries
        llm: LLM instance for translation (required if translate=True)
    
    Returns:
        Tuple of (train_count, test_count)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all queries
    queries = []
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    
    logger.info(f"Loaded {len(queries)} queries from {queries_file.name}")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(queries)
    
    train_count = min(train_queries, len(queries))
    train_queries_list = queries[:train_count]
    test_queries_list = queries[train_count:]
    
    # Process and save
    def convert_query(raw_query: Dict, translate: bool, llm) -> Dict:
        """Convert raw Pneuma query to unified format."""
        question = raw_query.get("question", "")
        
        # Handle translation if needed
        question_original = None
        question_translated = None
        language = detect_language(question)
        
        if translate and is_non_ascii(question) and llm is not None:
            question_original = question
            try:
                question_translated = translate_query(question, llm)
                question = question_translated  # Use translated version
            except Exception as e:
                logger.warning(f"Translation failed: {e}")
                question_translated = None
        
        # Build unified query
        unified = {
            "query_id": str(raw_query.get("id", "")),
            "question": question,
            "answer_tables": raw_query.get("answer_tables", []),
            "metadata": {
                "dataset": dataset_name,
                "source": "pneuma_annotated",
                "language": language
            }
        }
        
        # Add optional fields
        if question_original:
            unified["question_original"] = question_original
        if question_translated:
            unified["question_translated"] = question_translated
        
        if "meta" in raw_query:
            unified["metadata"]["pneuma_meta"] = raw_query["meta"]
        
        if "answer" in raw_query:
            unified["answer"] = raw_query["answer"]
        
        return unified
    
    # Process training set
    train_file = output_dir / "train.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for q in tqdm(train_queries_list, desc="Processing train queries"):
            unified = convert_query(q, translate, llm)
            unified["split"] = "train"
            f.write(json.dumps(unified, ensure_ascii=False) + "\n")
    
    # Process test set
    test_file = output_dir / "test.jsonl"
    with open(test_file, 'w', encoding='utf-8') as f:
        for q in tqdm(test_queries_list, desc="Processing test queries"):
            unified = convert_query(q, translate, llm)
            unified["split"] = "test"
            f.write(json.dumps(unified, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved {train_count} train queries and {len(test_queries_list)} test queries")
    return train_count, len(test_queries_list)


# ============ Solo Format Processing ============

def process_solo_tables(
    tables_file: Path,
    dataset_name: str,
    output_dir: Path
) -> Dict[str, str]:
    """
    Process Solo format tables (single JSONL file) into unified JSON format.
    
    Args:
        tables_file: Path to tables.jsonl
        dataset_name: Name of the dataset
        output_dir: Output directory for JSON files
    
    Returns:
        Mapping of original table IDs to unified table IDs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    table_id_mapping = {}
    
    with open(tables_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    logger.info(f"Processing {len(lines)} tables from {tables_file.name}...")
    
    for line in tqdm(lines, desc="Processing tables"):
        if not line.strip():
            continue
        
        try:
            raw_table = json.loads(line)
            table_id = raw_table.get("tableId", "")
            
            # Parse Solo format columns and rows
            columns = [col.get("text", "") for col in raw_table.get("columns", [])]
            rows = [
                [cell.get("text", "") for cell in row.get("cells", [])]
                for row in raw_table.get("rows", [])
            ]
            
            # Extract title from tableId or documentTitle
            title = raw_table.get("documentTitle", table_id)
            
            # Create unified table JSON
            table_data = {
                "table_id": table_id,
                "title": title,
                "section_title": "",
                "source_file": "tables.jsonl",
                "columns": columns,
                "column_count": len(columns),
                "row_count": len(rows),
                "rows": rows,
                "metadata": {
                    "dataset": dataset_name,
                    "original_id": table_id,
                    "source_format": "solo_jsonl",
                    "document_url": raw_table.get("documentUrl", "")
                }
            }
            
            # Save to output directory (use hash for safe filename)
            import hashlib
            safe_id = hashlib.md5(table_id.encode()).hexdigest()[:16]
            output_file = output_dir / f"{safe_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(table_data, f, ensure_ascii=False, indent=2)
            
            table_id_mapping[table_id] = table_id
            
        except Exception as e:
            logger.warning(f"Failed to process table: {e}")
    
    logger.info(f"Processed {len(table_id_mapping)} tables")
    return table_id_mapping


def process_solo_queries(
    queries_dir: Path,
    dataset_name: str,
    output_dir: Path,
    train_queries: int,
    translate: bool = False,
    llm=None
) -> Tuple[int, int]:
    """
    Process Solo format queries (train/dev/test directories) into unified format.
    
    Args:
        queries_dir: Directory containing train/dev/test subdirectories
        dataset_name: Name of the dataset
        output_dir: Output directory for train/test JSONL files
        train_queries: Number of queries for training set
        translate: Whether to translate non-English queries
        llm: LLM instance for translation
    
    Returns:
        Tuple of (train_count, test_count)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all queries from all splits
    all_queries = []
    
    for split in ["train", "dev", "test"]:
        split_dir = queries_dir / split
        if not split_dir.exists():
            continue
        
        for jsonl_file in split_dir.glob("*.jsonl"):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        q = json.loads(line)
                        q["_original_split"] = split
                        all_queries.append(q)
    
    logger.info(f"Loaded {len(all_queries)} queries from Solo format")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(all_queries)
    
    train_count = min(train_queries, len(all_queries))
    train_queries_list = all_queries[:train_count]
    test_queries_list = all_queries[train_count:]
    
    def convert_query(raw_query: Dict, translate: bool, llm) -> Dict:
        """Convert raw Solo query to unified format."""
        question = raw_query.get("question", "")
        
        # Handle translation
        question_original = None
        question_translated = None
        language = detect_language(question)
        
        if translate and is_non_ascii(question) and llm is not None:
            question_original = question
            try:
                question_translated = translate_query(question, llm)
                question = question_translated
            except Exception:
                question_translated = None
        
        # Build unified query
        unified = {
            "query_id": str(raw_query.get("id", "")),
            "question": question,
            "answer_tables": raw_query.get("table_id_lst", []),
            "metadata": {
                "dataset": dataset_name,
                "source": "solo",
                "language": language,
                "original_split": raw_query.get("_original_split", "")
            }
        }
        
        if question_original:
            unified["question_original"] = question_original
        if question_translated:
            unified["question_translated"] = question_translated
        
        if "answers" in raw_query:
            unified["answer"] = raw_query["answers"][0] if raw_query["answers"] else ""
        
        return unified
    
    # Save training set
    train_file = output_dir / "train.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for q in tqdm(train_queries_list, desc="Processing train queries"):
            unified = convert_query(q, translate, llm)
            unified["split"] = "train"
            f.write(json.dumps(unified, ensure_ascii=False) + "\n")
    
    # Save test set
    test_file = output_dir / "test.jsonl"
    with open(test_file, 'w', encoding='utf-8') as f:
        for q in tqdm(test_queries_list, desc="Processing test queries"):
            unified = convert_query(q, translate, llm)
            unified["split"] = "test"
            f.write(json.dumps(unified, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved {train_count} train queries and {len(test_queries_list)} test queries")
    return train_count, len(test_queries_list)


# ============ Translation Function ============

def translate_query(question: str, llm) -> str:
    """
    Translate a non-English query to English using LLM.
    
    Args:
        question: Original question text
        llm: LLM instance
    
    Returns:
        Translated English question
    """
    from llm.invoke_with_stats import invoke_structured_llm
    
    prompt = f"""Translate the following question to English. 
Keep the meaning and intent exactly the same. 
Only output the translated question, nothing else.

Question: {question}

English translation:"""
    
    result = invoke_structured_llm(llm, prompt, max_retries=2, timeout=60)
    return result.content.strip()


# ============ Main Processing Function ============

def process_dataset(
    dataset_name: str,
    train_queries: int = 200,
    translate: bool = False,
    llm=None
) -> Dict[str, Any]:
    """
    Process a single dataset into unified format.
    
    Args:
        dataset_name: Name of the dataset
        train_queries: Number of training queries
        translate: Whether to translate non-English queries
        llm: LLM instance for translation
    
    Returns:
        Statistics about the processed dataset
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    config = DATASET_CONFIGS[dataset_name]
    raw_dir = BENCHMARK_RAW_DIR / dataset_name
    output_dir = BENCHMARK_UNIFIED_DIR / dataset_name
    
    stats = {"dataset": dataset_name}
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"{'='*60}")
    
    if config["type"] == "pneuma":
        # Process Pneuma format
        tables_dir = raw_dir / "tables"
        queries_file = raw_dir / "queries.jsonl"
        
        if not tables_dir.exists():
            raise FileNotFoundError(f"Tables directory not found: {tables_dir}")
        if not queries_file.exists():
            raise FileNotFoundError(f"Queries file not found: {queries_file}")
        
        # Process tables
        table_mapping = process_pneuma_tables(
            tables_dir,
            dataset_name,
            output_dir / "table"
        )
        stats["table_count"] = len(table_mapping)
        
        # Process queries
        train_count, test_count = process_pneuma_queries(
            queries_file,
            dataset_name,
            output_dir / "query",
            train_queries,
            translate,
            llm
        )
        stats["train_queries"] = train_count
        stats["test_queries"] = test_count
        
    elif config["type"] == "solo":
        # Process Solo format
        tables_file = raw_dir / "tables.jsonl"
        queries_dir = raw_dir / "queries"
        
        if not tables_file.exists():
            raise FileNotFoundError(f"Tables file not found: {tables_file}")
        if not queries_dir.exists():
            raise FileNotFoundError(f"Queries directory not found: {queries_dir}")
        
        # Process tables
        table_mapping = process_solo_tables(
            tables_file,
            dataset_name,
            output_dir / "table"
        )
        stats["table_count"] = len(table_mapping)
        
        # Process queries
        train_count, test_count = process_solo_queries(
            queries_dir,
            dataset_name,
            output_dir / "query",
            train_queries,
            translate,
            llm
        )
        stats["train_queries"] = train_count
        stats["test_queries"] = test_count
    
    logger.info(f"\nDataset {dataset_name} processed:")
    logger.info(f"  Tables: {stats['table_count']}")
    logger.info(f"  Train queries: {stats['train_queries']}")
    logger.info(f"  Test queries: {stats['test_queries']}")
    
    return stats


# ============ Default Processing Configuration ============

# Each entry: (dataset_name, train_queries, use_translate)
# Modify this array to customize processing for each dataset
DEFAULT_PROCESSING_CONFIG = [
    ("adventure_works", 200, False),
    ("bird", 200, False),
    ("chembl", 200, False),
    ("chicago", 200, False),  # Chicago has 1002 queries, use 200 for train, 802 for test
    ("fetaqapn", 200, False),  # Set via shell script if needed
    ("public_bi", 200, False),
    ("fetaqa", 1000, False),
    ("debug", 3, False),  # Debug mock dataset for testing
]


# ============ CLI Entry Point ============

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Unify benchmark data from various sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        help="Process a single dataset (uses default config for train_queries and translate)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Process all datasets using DEFAULT_PROCESSING_CONFIG"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="JSON config array: [[name, train_queries, translate], ...]. "
             "Example: '[[\"bird\", 500, false], [\"fetaqapn\", 2000, true]]'"
    )
    parser.add_argument(
        "--train-queries", "-t",
        type=int,
        default=None,
        help="Override train queries for --dataset mode"
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Override translate setting for --dataset mode"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="local",
        help="LLM model to use for translation (default: local)"
    )
    
    args = parser.parse_args()
    
    # Determine processing configuration
    processing_config = []
    
    if args.config:
        # Use JSON config from command line
        try:
            config_array = json.loads(args.config)
            for item in config_array:
                if len(item) >= 3:
                    processing_config.append((item[0], item[1], item[2]))
                elif len(item) == 2:
                    processing_config.append((item[0], item[1], False))
                else:
                    processing_config.append((item[0], 200, False))
        except json.JSONDecodeError as e:
            parser.error(f"Invalid JSON config: {e}")
    elif args.all:
        # Use default config for all datasets
        processing_config = DEFAULT_PROCESSING_CONFIG.copy()
    elif args.dataset:
        # Single dataset mode
        # Find default config for this dataset
        default_train = 200
        default_translate = False
        for name, train, translate in DEFAULT_PROCESSING_CONFIG:
            if name == args.dataset:
                default_train = train
                default_translate = translate
                break
        
        # Apply overrides if specified
        train_queries = args.train_queries if args.train_queries is not None else default_train
        translate = args.translate if args.translate else default_translate
        processing_config = [(args.dataset, train_queries, translate)]
    else:
        parser.error("One of --dataset, --all, or --config must be specified")
    
    # Check if any dataset needs translation
    needs_translation = any(translate for _, _, translate in processing_config)
    
    # Setup LLM if translation is needed
    llm = None
    if needs_translation:
        from llm import get_llm
        llm = get_llm(args.llm_model)
        logger.info(f"Translation enabled using {args.llm_model} model")
    
    # Log processing plan
    logger.info("\n" + "="*60)
    logger.info("PROCESSING PLAN")
    logger.info("="*60)
    for name, train_queries, translate in processing_config:
        trans_str = "✓ translate" if translate else ""
        logger.info(f"  {name}: train={train_queries} {trans_str}")
    logger.info("="*60 + "\n")
    
    # Process datasets
    all_stats = []
    for dataset_name, train_queries, translate in processing_config:
        try:
            stats = process_dataset(
                dataset_name,
                train_queries=train_queries,
                translate=translate,
                llm=llm if translate else None
            )
            all_stats.append(stats)
        except Exception as e:
            logger.error(f"Failed to process {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*60)
    
    total_tables = sum(s["table_count"] for s in all_stats)
    total_train = sum(s["train_queries"] for s in all_stats)
    total_test = sum(s["test_queries"] for s in all_stats)
    
    for stats in all_stats:
        logger.info(f"  {stats['dataset']}: {stats['table_count']} tables, "
                   f"{stats['train_queries']} train, {stats['test_queries']} test")
    
    logger.info(f"\nTotal: {total_tables} tables, {total_train} train, {total_test} test queries")


if __name__ == "__main__":
    main()
