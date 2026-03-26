#!/usr/bin/env python3
"""
BM25 Content Baseline Evaluation

This implements the "content-based keyword search" baseline from Pneuma:
- Flatten each table into a single document (title + columns + all rows)
- Build BM25 index on the flattened content
- Retrieve tables using raw query with BM25

This differs from our schema-only BM25 (raw index) which only indexes title + columns.

Usage:
    # Build content index for a dataset
    python -m evaluation.baselines.bm25_content --dataset adventure_works --build-index
    
    # Evaluate on test queries
    python -m evaluation.baselines.bm25_content --dataset adventure_works --evaluate
    
    # Build and evaluate all datasets
    python -m evaluation.baselines.bm25_content --all
    
    # Compare with schema-only BM25
    python -m evaluation.baselines.bm25_content --dataset adventure_works --compare
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bm25s
import lancedb
import numpy as np
import Stemmer
from tqdm import tqdm

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
import _path_setup  # noqa: F401

from loguru import logger
from core.paths import get_project_root

PROJECT_ROOT = get_project_root()

# Constants
DATASETS = ["adventure_works", "bird", "chembl", "chicago", "fetaqa", "fetaqapn", "public_bi"]
TOP_K_VALUES = [1, 3, 5, 10, 20, 50, 100]


def get_index_path(dataset_name: str) -> Path:
    """Get path to content BM25 index."""
    return PROJECT_ROOT / "data" / "lake" / "indexes" / dataset_name / "content" / "bm25"


def get_rows_db_path(dataset_name: str) -> Path:
    """Get path to rows LanceDB for a dataset."""
    return PROJECT_ROOT / "data" / "lake" / "lancedb_rows" / dataset_name


def get_main_db_path() -> Path:
    """Get path to main LanceDB."""
    return PROJECT_ROOT / "data" / "lake" / "lancedb"


def load_table_metadata(dataset_name: str) -> List[Dict[str, Any]]:
    """Load table metadata from main LanceDB."""
    db_path = get_main_db_path()
    db = lancedb.connect(str(db_path))
    
    table_name = f"{dataset_name}_tables_entries"
    if table_name not in db.table_names(limit=1000):
        raise ValueError(f"Table {table_name} not found in LanceDB")
    
    table = db.open_table(table_name)
    df = table.to_pandas()
    
    tables = []
    for _, row in df.iterrows():
        tables.append({
            "table_id": row["table_id"],
            "title": row.get("title", row["table_id"]),
            "section_title": row.get("section_title", ""),
            "columns": json.loads(row["columns"]) if isinstance(row["columns"], str) else row["columns"],
        })
    
    return tables


def load_table_rows(dataset_name: str, table_id: str) -> List[List[str]]:
    """Load all rows for a specific table from rows LanceDB."""
    rows_db_path = get_rows_db_path(dataset_name)
    
    if not rows_db_path.exists():
        logger.warning(f"Rows DB not found: {rows_db_path}")
        return []
    
    db = lancedb.connect(str(rows_db_path))
    
    # Sanitize table_id for LanceDB table name
    safe_table_id = table_id.replace("/", "_").replace("\\", "_").replace(".", "_")
    
    available_tables = db.table_names(limit=100000)
    if safe_table_id not in available_tables:
        # Try original table_id as fallback
        if table_id not in available_tables:
            return []
        safe_table_id = table_id
    
    try:
        table = db.open_table(safe_table_id)
        df = table.to_pandas()
        
        # Extract row values from columns
        rows = []
        for _, row in df.iterrows():
            row_values = []
            for col in df.columns:
                val = row[col]
                if val is not None and str(val).strip():
                    row_values.append(str(val))
            if row_values:
                rows.append(row_values)
        
        return rows
    except Exception as e:
        logger.warning(f"Error loading rows for {table_id}: {e}")
        return []


def flatten_table_content(
    table_id: str,
    title: str,
    section_title: str,
    columns: List[str],
    rows: List[List[str]],
    max_rows: int = 500,
) -> str:
    """
    Flatten table into a single text document.
    
    Format:
        {title} - {section_title}
        Columns: {col1}, {col2}, {col3}, ...
        Row 1: val1, val2, val3, ...
        Row 2: val1, val2, val3, ...
        ...
    """
    lines = []
    
    # Title
    full_title = f"{title} - {section_title}" if section_title else title
    lines.append(full_title)
    
    # Columns
    col_str = ", ".join(c for c in columns if c)
    lines.append(f"Columns: {col_str}")
    
    # Rows (limit to max_rows to avoid memory issues)
    for i, row in enumerate(rows[:max_rows]):
        row_str = ", ".join(str(v) for v in row if v)
        if row_str:
            lines.append(f"Row {i+1}: {row_str}")
    
    return "\n".join(lines)


def build_content_index(dataset_name: str, force: bool = False) -> Dict[str, Any]:
    """
    Build BM25 index on flattened table content.
    
    Returns statistics about the index.
    """
    index_path = get_index_path(dataset_name)
    
    if index_path.exists() and not force:
        logger.info(f"Content index already exists for {dataset_name}, use --force to rebuild")
        return {"status": "skipped", "reason": "index exists"}
    
    logger.info(f"Building content BM25 index for {dataset_name}...")
    
    # Load table metadata
    tables = load_table_metadata(dataset_name)
    logger.info(f"  Loaded {len(tables)} tables")
    
    # Flatten content for each table
    table_ids = []
    documents = []
    stats = {
        "total_tables": len(tables),
        "tables_with_rows": 0,
        "total_rows": 0,
        "avg_doc_length": 0,
    }
    
    for table in tqdm(tables, desc="Flattening tables"):
        table_id = table["table_id"]
        
        # Load rows
        rows = load_table_rows(dataset_name, table_id)
        if rows:
            stats["tables_with_rows"] += 1
            stats["total_rows"] += len(rows)
        
        # Flatten content
        content = flatten_table_content(
            table_id=table_id,
            title=table["title"],
            section_title=table.get("section_title", ""),
            columns=table["columns"],
            rows=rows,
        )
        
        table_ids.append(table_id)
        documents.append(content)
    
    # Calculate average document length
    total_chars = sum(len(doc) for doc in documents)
    stats["avg_doc_length"] = total_chars / len(documents) if documents else 0
    
    logger.info(f"  Tables with rows: {stats['tables_with_rows']}")
    logger.info(f"  Total rows: {stats['total_rows']}")
    logger.info(f"  Avg doc length: {stats['avg_doc_length']:.0f} chars")
    
    # Build BM25 index
    logger.info("  Building BM25 index...")
    stemmer = Stemmer.Stemmer("english")
    tokenized = bm25s.tokenize(documents, stemmer=stemmer, show_progress=True)
    
    bm25_retriever = bm25s.BM25(corpus=tokenized)
    bm25_retriever.index(tokenized, show_progress=True)
    
    # Save index
    index_path.mkdir(parents=True, exist_ok=True)
    bm25_retriever.save(str(index_path / "index"), corpus=tokenized)
    
    with open(index_path / "table_ids.pkl", "wb") as f:
        pickle.dump(table_ids, f)
    
    with open(index_path / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"  ✓ Saved content index to {index_path}")
    stats["status"] = "success"
    return stats


def load_content_index(dataset_name: str) -> Tuple[bm25s.BM25, List[str]]:
    """Load content BM25 index and table IDs."""
    index_path = get_index_path(dataset_name)
    
    if not index_path.exists():
        raise FileNotFoundError(f"Content index not found: {index_path}")
    
    # Load BM25 index
    retriever = bm25s.BM25.load(str(index_path / "index"), load_corpus=False)
    
    # Load table IDs
    with open(index_path / "table_ids.pkl", "rb") as f:
        table_ids = pickle.load(f)
    
    return retriever, table_ids


def load_test_queries(dataset_name: str) -> List[Dict[str, Any]]:
    """Load test queries from LanceDB."""
    db_path = get_main_db_path()
    db = lancedb.connect(str(db_path))
    
    table_name = f"{dataset_name}_test_queries"
    if table_name not in db.table_names(limit=1000):
        raise ValueError(f"Table {table_name} not found in LanceDB")
    
    table = db.open_table(table_name)
    df = table.to_pandas()
    
    queries = []
    for _, row in df.iterrows():
        gt_tables = row.get("ground_truth_table_ids", "")
        if isinstance(gt_tables, str) and gt_tables:
            try:
                gt_tables = json.loads(gt_tables)
            except:
                gt_tables = [row["ground_truth_table_id"]]
        elif not gt_tables:
            gt_tables = [row["ground_truth_table_id"]]
        
        queries.append({
            "query_id": row["query_id"],
            "query_text": row["query_text"],
            "ground_truth_table_ids": gt_tables,
        })
    
    return queries


def bm25_retrieve(
    query: str,
    retriever: bm25s.BM25,
    table_ids: List[str],
    top_k: int = 100,
) -> List[Tuple[str, float]]:
    """
    Retrieve tables using BM25.
    
    Returns list of (table_id, score) tuples.
    """
    stemmer = Stemmer.Stemmer("english")
    query_tokens = bm25s.tokenize([query], stemmer=stemmer, show_progress=False)
    
    results, scores = retriever.retrieve(query_tokens, k=min(top_k, len(table_ids)))
    
    # Convert to (table_id, score) tuples
    retrieved = []
    for i, score in zip(results[0], scores[0]):
        if i < len(table_ids):
            retrieved.append((table_ids[i], float(score)))
    
    return retrieved


def evaluate_content_bm25(
    dataset_name: str,
    top_k_values: List[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate content BM25 baseline on test queries.
    
    Returns hit rates at different k values.
    """
    if top_k_values is None:
        top_k_values = TOP_K_VALUES
    
    logger.info(f"Evaluating content BM25 baseline on {dataset_name}...")
    
    # Load index
    retriever, table_ids = load_content_index(dataset_name)
    logger.info(f"  Loaded index with {len(table_ids)} tables")
    
    # Load test queries
    queries = load_test_queries(dataset_name)
    logger.info(f"  Loaded {len(queries)} test queries")
    
    # Evaluate
    max_k = max(top_k_values)
    hits_at_k = defaultdict(int)
    mrr_sum = 0.0
    
    for query in tqdm(queries, desc="Evaluating"):
        query_text = query["query_text"]
        gt_tables = query["ground_truth_table_ids"]
        
        # Retrieve
        results = bm25_retrieve(query_text, retriever, table_ids, top_k=max_k)
        retrieved_ids = [r[0] for r in results]
        
        # Check hits at each k
        for k in top_k_values:
            top_k_results = set(retrieved_ids[:k])
            if any(gt in top_k_results for gt in gt_tables):
                hits_at_k[k] += 1
        
        # MRR
        for rank, tid in enumerate(retrieved_ids, 1):
            if tid in gt_tables:
                mrr_sum += 1.0 / rank
                break
    
    # Calculate metrics
    n_queries = len(queries)
    results = {
        "dataset": dataset_name,
        "n_queries": n_queries,
        "n_tables": len(table_ids),
        "hit_rates": {},
        "mrr": mrr_sum / n_queries if n_queries > 0 else 0,
    }
    
    for k in top_k_values:
        hr = hits_at_k[k] / n_queries if n_queries > 0 else 0
        results["hit_rates"][f"HR@{k}"] = hr
    
    return results


def load_schema_index(dataset_name: str) -> Tuple[bm25s.BM25, List[str]]:
    """Load schema-only BM25 index (raw index)."""
    index_path = PROJECT_ROOT / "data" / "lake" / "indexes" / dataset_name / "raw" / "bm25"
    
    if not index_path.exists():
        raise FileNotFoundError(f"Schema index not found: {index_path}")
    
    retriever = bm25s.BM25.load(str(index_path / "index"), load_corpus=False)
    
    with open(index_path / "table_ids.pkl", "rb") as f:
        table_ids = pickle.load(f)
    
    return retriever, table_ids


def evaluate_schema_bm25(
    dataset_name: str,
    top_k_values: List[int] = None,
) -> Dict[str, Any]:
    """Evaluate schema-only BM25 baseline on test queries."""
    if top_k_values is None:
        top_k_values = TOP_K_VALUES
    
    logger.info(f"Evaluating schema BM25 baseline on {dataset_name}...")
    
    # Load index
    retriever, table_ids = load_schema_index(dataset_name)
    logger.info(f"  Loaded index with {len(table_ids)} tables")
    
    # Load test queries
    queries = load_test_queries(dataset_name)
    logger.info(f"  Loaded {len(queries)} test queries")
    
    # Evaluate
    max_k = max(top_k_values)
    hits_at_k = defaultdict(int)
    mrr_sum = 0.0
    
    for query in tqdm(queries, desc="Evaluating"):
        query_text = query["query_text"]
        gt_tables = query["ground_truth_table_ids"]
        
        results = bm25_retrieve(query_text, retriever, table_ids, top_k=max_k)
        retrieved_ids = [r[0] for r in results]
        
        for k in top_k_values:
            top_k_results = set(retrieved_ids[:k])
            if any(gt in top_k_results for gt in gt_tables):
                hits_at_k[k] += 1
        
        for rank, tid in enumerate(retrieved_ids, 1):
            if tid in gt_tables:
                mrr_sum += 1.0 / rank
                break
    
    n_queries = len(queries)
    results = {
        "dataset": dataset_name,
        "n_queries": n_queries,
        "n_tables": len(table_ids),
        "hit_rates": {},
        "mrr": mrr_sum / n_queries if n_queries > 0 else 0,
    }
    
    for k in top_k_values:
        hr = hits_at_k[k] / n_queries if n_queries > 0 else 0
        results["hit_rates"][f"HR@{k}"] = hr
    
    return results


def compare_baselines(dataset_name: str) -> None:
    """Compare content BM25 vs schema BM25."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Comparing BM25 baselines for {dataset_name}")
    logger.info(f"{'='*60}\n")
    
    # Evaluate both
    content_results = evaluate_content_bm25(dataset_name)
    schema_results = evaluate_schema_bm25(dataset_name)
    
    # Print comparison
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}")
    print(f"{'Metric':<15} {'Schema BM25':>15} {'Content BM25':>15} {'Delta':>15}")
    print(f"{'-'*70}")
    
    for k in TOP_K_VALUES:
        key = f"HR@{k}"
        s_hr = schema_results["hit_rates"][key]
        c_hr = content_results["hit_rates"][key]
        delta = c_hr - s_hr
        sign = "+" if delta > 0 else ""
        print(f"{key:<15} {s_hr:>14.2%} {c_hr:>14.2%} {sign}{delta:>14.2%}")
    
    s_mrr = schema_results["mrr"]
    c_mrr = content_results["mrr"]
    delta = c_mrr - s_mrr
    sign = "+" if delta > 0 else ""
    print(f"{'MRR':<15} {s_mrr:>14.4f} {c_mrr:>14.4f} {sign}{delta:>14.4f}")
    print(f"{'='*70}\n")


def run_all_datasets(build_index: bool = True, evaluate: bool = True) -> None:
    """Run content BM25 evaluation on all datasets."""
    all_results = {}
    
    for dataset in DATASETS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {dataset}")
        logger.info(f"{'='*60}\n")
        
        try:
            if build_index:
                build_content_index(dataset, force=False)
            
            if evaluate:
                results = evaluate_content_bm25(dataset)
                all_results[dataset] = results
        except Exception as e:
            logger.error(f"Error processing {dataset}: {e}")
            all_results[dataset] = {"error": str(e)}
    
    # Print summary table
    if evaluate and all_results:
        print_summary_table(all_results)
    
    # Save results
    output_path = PROJECT_ROOT / "data" / "experiments" / "bm25_content_baseline_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def print_summary_table(results: Dict[str, Any]) -> None:
    """Print summary table of all results."""
    print(f"\n{'='*100}")
    print("BM25 Content Baseline Results Summary")
    print(f"{'='*100}")
    
    # Header
    k_values = [1, 3, 5, 10, 20]
    header = f"{'Dataset':<20}"
    for k in k_values:
        header += f"{'HR@' + str(k):>12}"
    header += f"{'MRR':>12}"
    print(header)
    print("-" * 100)
    
    # Data rows
    for dataset, res in results.items():
        if "error" in res:
            print(f"{dataset:<20} ERROR: {res['error']}")
            continue
        
        row = f"{dataset:<20}"
        for k in k_values:
            hr = res["hit_rates"].get(f"HR@{k}", 0)
            row += f"{hr:>11.2%} "
        row += f"{res['mrr']:>11.4f}"
        print(row)
    
    print(f"{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(
        description="BM25 Content Baseline Evaluation"
    )
    parser.add_argument("-d", "--dataset", type=str, default=None,
                        help="Dataset name (default: all)")
    parser.add_argument("--build-index", action="store_true",
                        help="Build content BM25 index")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate on test queries")
    parser.add_argument("--compare", action="store_true",
                        help="Compare content vs schema BM25")
    parser.add_argument("--all", action="store_true",
                        help="Run on all datasets")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild index")
    parser.add_argument("--schema-only", action="store_true",
                        help="Evaluate schema-only BM25 (existing raw index)")
    
    args = parser.parse_args()
    
    # Default: if no action specified, do both build and evaluate
    if not any([args.build_index, args.evaluate, args.compare, args.schema_only]):
        args.build_index = True
        args.evaluate = True
    
    if args.all:
        run_all_datasets(build_index=args.build_index, evaluate=args.evaluate)
        return
    
    if args.dataset is None:
        parser.error("Please specify --dataset or use --all")
    
    if args.compare:
        compare_baselines(args.dataset)
        return
    
    if args.schema_only:
        results = evaluate_schema_bm25(args.dataset)
        print_summary_table({args.dataset: results})
        return
    
    if args.build_index:
        build_content_index(args.dataset, force=args.force)
    
    if args.evaluate:
        results = evaluate_content_bm25(args.dataset)
        print_summary_table({args.dataset: results})


if __name__ == "__main__":
    main()
