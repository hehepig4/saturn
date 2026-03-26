#!/usr/bin/env python3
"""
Unified Benchmark LanceDB Ingestion Script

Ingests all benchmark datasets from unified format into LanceDB.
Replaces the individual ingest_*.py scripts with a single unified approach.

Key features:
    - Single script for all datasets
    - Consistent table naming: {dataset}_tables_entries for tables
    - Separate train/test query tables: {dataset}_train_queries, {dataset}_test_queries
    - Independent indexes stored at: data/lake/indexes/{dataset}/raw/
    - Supports BM25-only mode (--index-mode bm25) to skip embedding computation

Output tables (per dataset):
    - {dataset}_tables_entries: Table metadata (no embeddings in LanceDB)
    - {dataset}_train_queries: Train queries only (for TBox training)
    - {dataset}_test_queries: Test queries only (for evaluation)

Independent indexes (per dataset):
    - data/lake/indexes/{dataset}/raw/faiss/: Table FAISS index
    - data/lake/indexes/{dataset}/raw/bm25/: Table BM25 index
    - data/lake/indexes/{dataset}/raw/train_query_faiss/: Train query FAISS index
    - data/lake/indexes/{dataset}/raw/train_query_bm25/: Train query BM25 index
    - data/lake/indexes/{dataset}/raw/test_query_faiss/: Test query FAISS index
    - data/lake/indexes/{dataset}/raw/test_query_bm25/: Test query BM25 index

Usage:
    cd source/
    python scripts/ingest_benchmark.py --dataset fetaqa
    python scripts/ingest_benchmark.py --all
    python scripts/ingest_benchmark.py --all --tables-only   # Only ingest tables
    python scripts/ingest_benchmark.py --all --queries-only  # Only ingest queries
    python scripts/ingest_benchmark.py --dataset fetaqa --index-mode bm25  # Skip embeddings
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger
from tqdm import tqdm

# Path setup for standalone execution
SOURCE_DIR = Path(__file__).resolve().parent.parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from store.store_manager import StoreManager
from store.embedding.custom_embeddings import get_bge_m3_embedding


# ============ Configuration ============

PROJECT_ROOT = SOURCE_DIR.parent
BENCHMARK_UNIFIED_DIR = PROJECT_ROOT / "data" / "benchmark" / "unified"

# Supported datasets
DATASETS = [
    "adventure_works",
    "bird",
    "chembl",
    "chicago",
    "fetaqapn",
    "public_bi",
    "fetaqa",
    "debug",  # Debug mock dataset for testing
]

# Table serialization settings
MAX_SAMPLE_ROWS = 3  # For display/prompt purposes
BATCH_SIZE = 64


# ============ Schema Definitions ============

def get_tables_schema() -> pa.Schema:
    """Get PyArrow schema for tables entries.
    
    NOTE: all_rows field is removed. Row data is stored in separate LanceDB:
          data/lake/lancedb_rows/{dataset}/{table_id}
    """
    return pa.schema([
        pa.field("table_id", pa.string()),
        pa.field("document_title", pa.string()),
        pa.field("section_title", pa.string()),
        pa.field("columns", pa.string()),  # JSON array
        pa.field("column_count", pa.int32()),
        pa.field("row_count", pa.int32()),
        pa.field("sample_rows", pa.string()),  # JSON array (3 rows for display)
        pa.field("table_text", pa.string()),
        pa.field("split", pa.string()),
        # NOTE: table_text_embedding moved to independent index at:
        # data/lake/indexes/{dataset}/raw/faiss/
    ])


def get_table_rows_schema(num_columns: int) -> pa.Schema:
    """Get PyArrow schema for individual table row data.
    
    Each table's rows are stored in a separate LanceDB table in:
    data/lake/lancedb_rows/{dataset}/{table_id}
    
    Schema is dynamically generated based on column count:
    - row_index: int32
    - col_0, col_1, ..., col_{N-1}: string (all values stored as strings)
    
    Args:
        num_columns: Number of columns in the table
    """
    fields = [pa.field("row_index", pa.int32())]
    for i in range(num_columns):
        fields.append(pa.field(f"col_{i}", pa.string()))
    return pa.schema(fields)


def get_test_queries_schema() -> pa.Schema:
    """Get PyArrow schema for test queries (evaluation-ready)."""
    return pa.schema([
        pa.field("query_id", pa.string()),
        pa.field("query_text", pa.string()),
        pa.field("answer", pa.string()),
        pa.field("ground_truth_table_id", pa.string()),
        pa.field("ground_truth_table_ids", pa.string()),  # JSON array for multi-answer
        pa.field("metadata", pa.string()),  # JSON object for extra info
    ])


def get_train_queries_schema() -> pa.Schema:
    """Get PyArrow schema for train queries (TBox training)."""
    return pa.schema([
        pa.field("query_id", pa.string()),
        pa.field("query_text", pa.string()),
        pa.field("answer", pa.string()),
        pa.field("ground_truth_table_id", pa.string()),
        pa.field("ground_truth_table_ids", pa.string()),  # JSON array for multi-answer
        pa.field("metadata", pa.string()),  # JSON object for extra info
    ])


# ============ Embedding Generation ============

_embedding_func = None

def get_embedding_func():
    """Get or create the embedding function instance."""
    global _embedding_func
    if _embedding_func is None:
        _embedding_func = get_bge_m3_embedding()
    return _embedding_func


def generate_embeddings(texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """
    Generate BGE-M3 embeddings for texts.
    
    Args:
        texts: List of text strings
        batch_size: Batch size for embedding
    
    Returns:
        numpy array of embeddings (N, dim)
    """
    logger.info(f"Generating embeddings for {len(texts)} texts...")
    
    embed_func = get_embedding_func()
    embeddings = embed_func.compute_source_embeddings(texts)
    embeddings = np.array(embeddings)
    
    logger.info(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings


# ============ Table Serialization ============

def serialize_table(
    title: str,
    columns: List[str],
    section_title: str = "",
) -> str:
    """
    Serialize table to text representation for embedding.
    
    NOTE: Only includes title and schema (column names), NOT sample values.
    This is for semantic matching of table structure, not content.
    
    Format:
        {title} - {section_title}
        Columns: {col1}, {col2}, {col3}, ...
    """
    # Build title with section if available
    full_title = f"{title} - {section_title}" if section_title else title
    lines = [full_title]
    
    # Columns line
    col_str = ", ".join(c for c in columns if c)
    lines.append(f"Columns: {col_str}")
    
    return "\n".join(lines)


# ============ Raw Index Generation ============

def generate_raw_indexes(
    dataset_name: str,
    tables: List[Dict[str, Any]],
    stats: Dict[str, Any],
    index_mode: str = "all",
) -> None:
    """
    Generate raw FAISS + BM25 indexes for unified search.
    
    These indexes are built on table_text (title + columns) and stored at:
    data/lake/indexes/{dataset}/raw/
    
    This enables Stage 1 to use unified_search with hybrid/bm25/vector modes
    before the full pipeline generates table summaries.
    
    Args:
        dataset_name: Dataset identifier
        tables: List of table dictionaries (with table_id, title, columns, etc.)
        stats: Statistics dict to update
        index_mode: Which indexes to generate ('all', 'bm25', 'vector')
    """
    import faiss
    import bm25s
    import Stemmer
    import pickle
    
    if not tables:
        logger.warning("No tables to index")
        return
    
    # Index path: data/lake/indexes/{dataset}/raw/
    index_base = PROJECT_ROOT / "data" / "lake" / "indexes" / dataset_name / "raw"
    faiss_path = index_base / "faiss"
    bm25_path = index_base / "bm25"
    
    # Clean up old indexes
    import shutil
    if index_base.exists():
        logger.info(f"Removing old raw indexes: {index_base}")
        shutil.rmtree(index_base)
    
    # Create directories based on index_mode
    if index_mode in ("all", "vector"):
        faiss_path.mkdir(parents=True, exist_ok=True)
    if index_mode in ("all", "bm25"):
        bm25_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Generating raw indexes for {dataset_name} (mode={index_mode})")
    logger.info(f"{'='*60}")
    
    # Prepare data
    table_ids = []
    table_texts = []
    metadata_list = []
    
    for table in tables:
        table_id = table.get("table_id", "")
        title = table.get("title", table_id)
        section_title = table.get("section_title", "")
        columns = table.get("columns", [])
        
        # Serialize for embedding/indexing
        table_text = serialize_table(title, columns, section_title)
        
        table_ids.append(table_id)
        table_texts.append(table_text)
        metadata_list.append({
            "table_id": table_id,
            "table_description": table_text,
            "document_title": title,
            "section_title": section_title,
            "columns": columns,
        })
    
    # === Generate FAISS index (if needed) ===
    faiss_index = None
    if index_mode in ("all", "vector"):
        logger.info(f"Generating FAISS index for {len(tables)} tables...")
        
        embeddings = generate_embeddings(table_texts)
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = embeddings / np.clip(norms, 1e-10, None)
        
        # Create FAISS index (IndexFlatIP for inner product = cosine on normalized vectors)
        dim = embeddings_normalized.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(embeddings_normalized)
        
        # Save FAISS index
        faiss.write_index(faiss_index, str(faiss_path / "index.faiss"))
        with open(faiss_path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata_list, f)
        
        logger.info(f"  ✓ FAISS index: {faiss_index.ntotal} vectors, dim={dim}")
    else:
        logger.info(f"Skipping FAISS index generation (mode={index_mode})")
    
    # === Generate BM25 index (if needed) ===
    if index_mode in ("all", "bm25"):
        logger.info(f"Generating BM25 index for {len(tables)} tables...")
        
        stemmer = Stemmer.Stemmer("english")
        tokenized = bm25s.tokenize(table_texts, stemmer=stemmer, show_progress=False)
        
        bm25_retriever = bm25s.BM25(corpus=tokenized)
        bm25_retriever.index(tokenized, show_progress=False)
        
        # Save BM25 index
        bm25_retriever.save(str(bm25_path / "index"), corpus=tokenized)
        with open(bm25_path / "table_ids.pkl", "wb") as f:
            pickle.dump(table_ids, f)
        
        logger.info(f"  ✓ BM25 index: {len(table_ids)} documents")
    else:
        logger.info(f"Skipping BM25 index generation (mode={index_mode})")
    
    # Update stats
    stats["raw_index_path"] = str(index_base)
    stats["raw_index_mode"] = index_mode
    if faiss_index is not None:
        stats["raw_faiss_vectors"] = faiss_index.ntotal
    if index_mode in ("all", "bm25"):
        stats["raw_bm25_docs"] = len(table_ids)
    
    logger.info(f"✓ Raw indexes saved to: {index_base}")


def generate_query_indexes(
    dataset_name: str,
    train_queries: List[Dict[str, Any]],
    test_queries: List[Dict[str, Any]],
    stats: Dict[str, Any],
    index_mode: str = "all",
) -> None:
    """
    Generate FAISS + BM25 indexes for queries (train and test separately).
    
    These indexes are built on query_text and stored at:
    data/lake/indexes/{dataset}/raw/train_query_faiss/
    data/lake/indexes/{dataset}/raw/train_query_bm25/
    data/lake/indexes/{dataset}/raw/test_query_faiss/
    data/lake/indexes/{dataset}/raw/test_query_bm25/
    
    This enables load_query_embeddings() to load from independent indexes
    instead of LanceDB, supporting both vector and BM25-only modes.
    
    Args:
        dataset_name: Dataset identifier
        train_queries: List of training query dictionaries
        test_queries: List of test query dictionaries
        stats: Statistics dict to update
        index_mode: Which indexes to generate ('all', 'bm25', 'vector')
    """
    import faiss
    import bm25s
    import Stemmer
    import pickle
    
    # Index path: data/lake/indexes/{dataset}/raw/
    index_base = PROJECT_ROOT / "data" / "lake" / "indexes" / dataset_name / "raw"
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Generating query indexes for {dataset_name} (mode={index_mode})")
    logger.info(f"{'='*60}")
    
    def _generate_indexes_for_split(
        queries: List[Dict[str, Any]], 
        split: str
    ) -> None:
        """Generate indexes for a single split (train or test)."""
        if not queries:
            logger.info(f"No {split} queries to index")
            return
        
        faiss_path = index_base / f"{split}_query_faiss"
        bm25_path = index_base / f"{split}_query_bm25"
        
        # Create directories based on index_mode
        if index_mode in ("all", "vector"):
            faiss_path.mkdir(parents=True, exist_ok=True)
        if index_mode in ("all", "bm25"):
            bm25_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        query_ids = []
        query_texts = []
        metadata_list = []
        
        for q in queries:
            query_id = str(q.get("query_id", ""))
            query_text = q.get("question", "")
            answer_tables = q.get("answer_tables", [])
            
            query_ids.append(query_id)
            query_texts.append(query_text)
            metadata_list.append({
                "query_id": query_id,
                "query_text": query_text,
                "split": split,
                "answer_tables": answer_tables,
                "answer": q.get("answer", ""),
            })
        
        # === Generate FAISS index (if needed) ===
        if index_mode in ("all", "vector"):
            logger.info(f"Generating {split} query FAISS index for {len(queries)} queries...")
            
            embeddings = generate_embeddings(query_texts)
            embeddings = np.array(embeddings, dtype=np.float32)
            
            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_normalized = embeddings / np.clip(norms, 1e-10, None)
            
            # Create FAISS index
            dim = embeddings_normalized.shape[1]
            faiss_index = faiss.IndexFlatIP(dim)
            faiss_index.add(embeddings_normalized)
            
            # Save FAISS index
            faiss.write_index(faiss_index, str(faiss_path / "index.faiss"))
            
            # Save metadata and embeddings (non-normalized, for load_query_embeddings)
            with open(faiss_path / "metadata.pkl", "wb") as f:
                pickle.dump({
                    "query_ids": query_ids,
                    "metadata_list": metadata_list,
                    "embeddings": embeddings,  # Non-normalized embeddings for external use
                }, f)
            
            logger.info(f"  ✓ {split.capitalize()} query FAISS index: {faiss_index.ntotal} vectors, dim={dim}")
        
        # === Generate BM25 index (if needed) ===
        if index_mode in ("all", "bm25"):
            logger.info(f"Generating {split} query BM25 index for {len(queries)} queries...")
            
            stemmer = Stemmer.Stemmer("english")
            tokenized = bm25s.tokenize(query_texts, stemmer=stemmer, show_progress=False)
            
            bm25_retriever = bm25s.BM25(corpus=tokenized)
            bm25_retriever.index(tokenized, show_progress=False)
            
            # Save BM25 index
            bm25_retriever.save(str(bm25_path / "index"), corpus=tokenized)
            with open(bm25_path / "metadata.pkl", "wb") as f:
                pickle.dump({
                    "query_ids": query_ids,
                    "metadata_list": metadata_list,
                }, f)
            
            logger.info(f"  ✓ {split.capitalize()} query BM25 index: {len(query_ids)} documents")
    
    # Generate indexes for train and test separately
    _generate_indexes_for_split(train_queries, "train")
    _generate_indexes_for_split(test_queries, "test")
    
    # Update stats
    stats["query_index_mode"] = index_mode
    stats["train_query_count"] = len(train_queries)
    stats["test_query_count"] = len(test_queries)
    
    logger.info(f"✓ Query indexes saved to: {index_base}")


# ============ Data Loading ============

def load_unified_tables(dataset_dir: Path) -> List[Dict[str, Any]]:
    """
    Load tables from unified JSON format.
    
    Args:
        dataset_dir: Path to unified dataset directory
    
    Returns:
        List of table dictionaries
    """
    tables_dir = dataset_dir / "table"
    if not tables_dir.exists():
        logger.warning(f"Tables directory not found: {tables_dir}")
        return []
    
    tables = []
    json_files = list(tables_dir.glob("*.json"))
    
    for json_file in tqdm(json_files, desc="Loading tables"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                table_data = json.load(f)
            tables.append(table_data)
        except Exception as e:
            logger.warning(f"Failed to load {json_file.name}: {e}")
    
    logger.info(f"Loaded {len(tables)} tables")
    return tables


def load_unified_queries(dataset_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """
    Load queries from unified JSONL format.
    
    Args:
        dataset_dir: Path to unified dataset directory
    
    Returns:
        Tuple of (train_queries, test_queries)
    """
    query_dir = dataset_dir / "query"
    if not query_dir.exists():
        logger.warning(f"Query directory not found: {query_dir}")
        return [], []
    
    train_queries = []
    test_queries = []
    
    # Load train queries
    train_file = query_dir / "train.jsonl"
    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    train_queries.append(json.loads(line))
    
    # Load test queries
    test_file = query_dir / "test.jsonl"
    if test_file.exists():
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_queries.append(json.loads(line))
    
    logger.info(f"Loaded {len(train_queries)} train and {len(test_queries)} test queries")
    return train_queries, test_queries


# ============ Ingestion Functions ============

def _get_rows_db_path(dataset_name: str) -> Path:
    """Get the path to the rows database for a dataset.
    
    Structure: data/lake/lancedb_rows/{dataset}/
    """
    return PROJECT_ROOT / "data" / "lake" / "lancedb_rows" / dataset_name


def _sanitize_table_id(table_id: str) -> str:
    """Sanitize table_id for use as LanceDB table name.
    
    LanceDB table names can only contain alphanumeric characters, 
    underscores, hyphens, and periods.
    """
    # Replace any invalid characters with underscores
    import re
    sanitized = re.sub(r'[^a-zA-Z0-9_\-.]', '_', table_id)
    return sanitized


def ingest_tables(
    store: StoreManager,
    dataset_name: str,
    tables: List[Dict[str, Any]],
    test_table_ids: set
) -> None:
    """
    Ingest tables into LanceDB.
    
    Table metadata goes to main DB: {dataset}_tables_entries
    Row data goes to separate rows DB: data/lake/lancedb_rows/{dataset}/{table_id}
    
    Args:
        store: StoreManager instance
        dataset_name: Name of the dataset
        tables: List of table dictionaries
        test_table_ids: Set of table IDs used in test queries
    """
    if not tables:
        logger.warning("No tables to ingest")
        return
    
    table_name = f"{dataset_name}_tables_entries"
    
    # Prepare records
    records = []
    
    for table_data in tqdm(tables, desc="Preparing tables"):
        table_id = table_data["table_id"]
        title = table_data.get("title", table_id)
        section_title = table_data.get("section_title", "")
        columns = table_data.get("columns", [])
        rows = table_data.get("rows", [])
        
        # Serialize for text-based indexing
        table_text = serialize_table(title, columns, section_title)
        
        # Determine split based on test query usage
        split = "test" if table_id in test_table_ids else "train"
        
        record = {
            "table_id": table_id,
            "document_title": title,
            "section_title": section_title,
            "columns": json.dumps(columns),
            "column_count": len(columns),
            "row_count": len(rows),
            "sample_rows": json.dumps(rows[:MAX_SAMPLE_ROWS]),
            # NOTE: all_rows stored in separate DB: lancedb_rows/{dataset}/{table_id}
            "table_text": table_text,
            "split": split,
            "_rows_data": rows,  # Temporary field for row data storage
            # NOTE: table_text_embedding moved to independent index at:
            # data/lake/indexes/{dataset}/raw/faiss/
        }
        records.append(record)
    
    # === Store row data in separate rows database ===
    logger.info(f"Storing row data for {len(records)} tables...")
    rows_db_path = _get_rows_db_path(dataset_name)
    
    # Clean up old rows database if exists
    import shutil
    if rows_db_path.exists():
        logger.info(f"Removing old rows database: {rows_db_path}")
        shutil.rmtree(rows_db_path)
    
    # Create rows database
    rows_db_path.mkdir(parents=True, exist_ok=True)
    import lancedb
    rows_db = lancedb.connect(str(rows_db_path))
    
    # Store each table's rows in the rows database
    for record in tqdm(records, desc="Storing row data"):
        table_id = record["table_id"]
        rows_data = record.pop("_rows_data")  # Remove temporary field
        
        if not rows_data:
            continue
        
        # Determine column count from first row (all rows have same length)
        num_columns = len(rows_data[0]) if rows_data else 0
        if num_columns == 0:
            continue
        
        # Create row records with dynamic columns: col_0, col_1, ...
        row_records = []
        for idx, row in enumerate(rows_data):
            row_record = {"row_index": idx}
            for col_idx, value in enumerate(row):
                # Store all values as strings (consistent with original JSON behavior)
                row_record[f"col_{col_idx}"] = str(value) if value is not None else ""
            row_records.append(row_record)
        
        # Sanitize table_id for use as table name
        safe_table_id = _sanitize_table_id(table_id)
        
        # Create row table in rows database with dynamic schema
        row_df = pd.DataFrame(row_records)
        rows_db.create_table(safe_table_id, row_df, schema=get_table_rows_schema(num_columns))
    
    logger.info(f"✓ Stored row data for {len(records)} tables in {rows_db_path}")
    
    # === Create main table in LanceDB ===
    logger.info(f"Creating table '{table_name}' with {len(records)} records...")
    
    df = pd.DataFrame(records)
    existing_tables = store.db.table_names(limit=1000000)
    
    # Drop existing table if exists
    if table_name in existing_tables:
        logger.info(f"Dropping existing table '{table_name}'...")
        store.db.drop_table(table_name)
    
    store.db.create_table(table_name, df, schema=get_tables_schema())
    logger.info(f"✓ Created table '{table_name}' with {len(records)} records")


def ingest_queries(
    store: StoreManager,
    dataset_name: str,
    train_queries: List[Dict],
    test_queries: List[Dict]
) -> None:
    """
    Ingest queries into LanceDB (train and test separately).
    
    Creates two tables:
        - {dataset}_train_queries: Train queries only (for TBox training)
        - {dataset}_test_queries: Test queries only (for evaluation)
    
    NOTE: query_text_embedding is stored in independent indexes at:
    data/lake/indexes/{dataset}/raw/train_query_faiss/ and train_query_bm25/
    data/lake/indexes/{dataset}/raw/test_query_faiss/ and test_query_bm25/
    
    Args:
        store: StoreManager instance
        dataset_name: Name of the dataset
        train_queries: List of train query dictionaries
        test_queries: List of test query dictionaries
    """
    existing_tables = store.db.table_names(limit=1000000)
    
    # === Ingest train queries ===
    if train_queries:
        train_table_name = f"{dataset_name}_train_queries"
        
        train_records = []
        for q in train_queries:
            answer_tables = q.get("answer_tables", [])
            gt_table_id = answer_tables[0] if answer_tables else ""
            
            record = {
                "query_id": str(q.get("query_id", "")),
                "query_text": q.get("question", ""),
                "answer": q.get("answer", ""),
                "ground_truth_table_id": gt_table_id,
                "ground_truth_table_ids": json.dumps(answer_tables),
                "metadata": json.dumps(q.get("metadata", {})),
            }
            train_records.append(record)
        
        logger.info(f"Creating table '{train_table_name}' with {len(train_records)} records...")
        
        df = pd.DataFrame(train_records)
        
        if train_table_name in existing_tables:
            logger.info(f"Dropping existing table '{train_table_name}'...")
            store.db.drop_table(train_table_name)
        
        store.db.create_table(train_table_name, df, schema=get_train_queries_schema())
        logger.info(f"✓ Created table '{train_table_name}' with {len(train_records)} records")
    else:
        logger.info("No train queries to ingest")
    
    # === Ingest test queries ===
    if test_queries:
        test_table_name = f"{dataset_name}_test_queries"
        
        test_records = []
        for q in test_queries:
            answer_tables = q.get("answer_tables", [])
            gt_table_id = answer_tables[0] if answer_tables else ""
            
            record = {
                "query_id": str(q.get("query_id", "")),
                "query_text": q.get("question", ""),
                "answer": q.get("answer", ""),
                "ground_truth_table_id": gt_table_id,
                "ground_truth_table_ids": json.dumps(answer_tables),
                "metadata": json.dumps(q.get("metadata", {})),
            }
            test_records.append(record)
        
        logger.info(f"Creating table '{test_table_name}' with {len(test_records)} records...")
        
        df = pd.DataFrame(test_records)
        
        if test_table_name in existing_tables:
            logger.info(f"Dropping existing table '{test_table_name}'...")
            store.db.drop_table(test_table_name)
        
        store.db.create_table(test_table_name, df, schema=get_test_queries_schema())
        logger.info(f"✓ Created table '{test_table_name}' with {len(test_records)} records")
    else:
        logger.info("No test queries to ingest")


# ============ Dataset Processing ============

def process_dataset(
    store: StoreManager,
    dataset_name: str,
    tables_only: bool = False,
    queries_only: bool = False,
    index_mode: str = "all"
) -> Dict[str, Any]:
    """
    Process and ingest a single dataset.
    
    Args:
        store: StoreManager instance
        dataset_name: Name of the dataset
        tables_only: Only ingest tables
        queries_only: Only ingest queries
        index_mode: Which indexes to generate ('all', 'bm25', 'vector')
    
    Returns:
        Statistics about the ingested data
    """
    dataset_dir = BENCHMARK_UNIFIED_DIR / dataset_name
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"{'='*60}")
    
    stats = {"dataset": dataset_name}
    
    # Load queries first (to determine test table IDs)
    train_queries, test_queries = load_unified_queries(dataset_dir)
    
    # Collect test table IDs
    test_table_ids = set()
    for q in test_queries:
        for tid in q.get("answer_tables", []):
            test_table_ids.add(tid)
    
    # Ingest tables
    if not queries_only:
        tables = load_unified_tables(dataset_dir)
        ingest_tables(store, dataset_name, tables, test_table_ids)
        stats["table_count"] = len(tables)
        
        # Generate raw indexes (FAISS + BM25) for unified search
        generate_raw_indexes(dataset_name, tables, stats, index_mode=index_mode)
    
    # Ingest queries
    if not tables_only:
        ingest_queries(store, dataset_name, train_queries, test_queries)
        stats["train_queries"] = len(train_queries)
        stats["test_queries"] = len(test_queries)
        
        # Generate query indexes (FAISS + BM25) for load_query_embeddings
        generate_query_indexes(
            dataset_name, train_queries, test_queries, stats, index_mode=index_mode
        )
    
    return stats


def verify_ingestion(store: StoreManager, dataset_name: str) -> None:
    """Verify ingestion by checking row counts."""
    logger.info(f"\n=== Verification for {dataset_name} ===")
    
    tables_table = f"{dataset_name}_tables_entries"
    train_queries_table = f"{dataset_name}_train_queries"
    test_queries_table = f"{dataset_name}_test_queries"
    
    existing = store.db.table_names(limit=1000000)
    
    if tables_table in existing:
        count = store.db.open_table(tables_table).count_rows()
        logger.info(f"  {tables_table}: {count} rows")
    
    if train_queries_table in existing:
        count = store.db.open_table(train_queries_table).count_rows()
        logger.info(f"  {train_queries_table}: {count} rows")
    
    if test_queries_table in existing:
        count = store.db.open_table(test_queries_table).count_rows()
        logger.info(f"  {test_queries_table}: {count} rows")


# ============ CLI Entry Point ============

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ingest benchmark data into LanceDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=DATASETS,
        help="Dataset to ingest"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Ingest all datasets"
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Only ingest tables (skip queries)"
    )
    parser.add_argument(
        "--queries-only",
        action="store_true",
        help="Only ingest queries (skip tables)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify ingestion after completion"
    )
    parser.add_argument(
        "--index-mode",
        type=str,
        choices=["all", "bm25", "vector"],
        default="all",
        help="Which indexes to generate: 'all' (both), 'bm25' (only bm25), 'vector' (only faiss)"
    )
    
    args = parser.parse_args()
    
    if not args.dataset and not args.all:
        parser.error("Either --dataset or --all must be specified")
    
    # Initialize store
    store = StoreManager()
    logger.info(f"Connected to LanceDB at: {store.db_path}")
    
    # Process datasets
    datasets_to_process = DATASETS if args.all else [args.dataset]
    
    all_stats = []
    for dataset in datasets_to_process:
        try:
            stats = process_dataset(
                store,
                dataset,
                tables_only=args.tables_only,
                queries_only=args.queries_only,
                index_mode=args.index_mode
            )
            all_stats.append(stats)
            
            if args.verify:
                verify_ingestion(store, dataset)
                
        except FileNotFoundError as e:
            logger.warning(f"Skipping {dataset}: {e}")
        except Exception as e:
            logger.error(f"Failed to process {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("INGESTION SUMMARY")
    logger.info("="*60)
    
    for stats in all_stats:
        line = f"  {stats['dataset']}: "
        if 'table_count' in stats:
            line += f"{stats['table_count']} tables, "
        if 'train_queries' in stats:
            line += f"{stats['train_queries']} train, {stats['test_queries']} test"
        logger.info(line)
    
    if all_stats:
        total_tables = sum(s.get("table_count", 0) for s in all_stats)
        total_train = sum(s.get("train_queries", 0) for s in all_stats)
        total_test = sum(s.get("test_queries", 0) for s in all_stats)
        logger.info(f"\nTotal: {total_tables} tables, {total_train} train, {total_test} test queries")


if __name__ == "__main__":
    main()
