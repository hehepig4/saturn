"""
Table retrieval utilities for federated primitive TBox workflow.

Provides table lookup, hard negative retrieval via unified search,
and cluster-level table retrieval for CQ generation.
"""

import json
import random
from typing import Dict, Any, List, Optional, Union
from loguru import logger
import numpy as np


def _parse_columns(columns: Union[str, List, None]) -> List[str]:
    """
    Parse columns field which may be a JSON string, list, or None.

    Args:
        columns: Raw columns value from database

    Returns:
        List of column names as strings
    """
    if columns is None:
        return []

    # If it's already a list, return it (flatten if nested)
    if isinstance(columns, list):
        # Handle nested list case [["col1", "col2", ...]]
        if columns and isinstance(columns[0], list):
            return [str(c) for c in columns[0]]
        return [str(c) for c in columns]

    # If it's a string, try to parse as JSON
    if isinstance(columns, str):
        try:
            parsed = json.loads(columns)
            if isinstance(parsed, list):
                # Handle nested list case [["col1", "col2", ...]]
                if parsed and isinstance(parsed[0], list):
                    return [str(c) for c in parsed[0]]
                return [str(c) for c in parsed]
            else:
                return [str(parsed)]  # Single value
        except json.JSONDecodeError:
            return [columns]  # Treat as single column name

    return [str(columns)]  # Fallback for other types


def _build_table_lookup(tables: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build table ID to table data mapping."""
    lookup = {}
    for table in tables:
        table_id = table.get("table_id") or table.get("fid") or table.get("id")
        if table_id:
            lookup[table_id] = table
    return lookup


def _retrieve_hard_negatives_unified(
    query_text: str,
    dataset_name: str,
    exclude_table_ids: Optional[set],
    top_k: int,
    rag_type: str = "hybrid",
    table_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-K similar tables using unified search interface.

    Uses the 'raw' index (built during ingest) for Stage 1 hard negative sampling.
    Supports BM25, Vector, or Hybrid retrieval modes.

    Args:
        query_text: Query text for retrieval
        dataset_name: Dataset identifier (e.g., 'fetaqa')
        exclude_table_ids: Set of table IDs to exclude (all ground truths)
        top_k: Number of hard negatives to retrieve
        rag_type: Retrieval type - 'bm25', 'vector', or 'hybrid'
        table_lookup: Optional table lookup for additional metadata

    Returns:
        List of hard negative table dicts
    """
    try:
        from workflows.retrieval.unified_search import unified_search
        from workflows.retrieval.config import INDEX_KEY_RAW

        # Retrieve more than needed to account for filtering multiple GTs
        n_exclude = len(exclude_table_ids) if exclude_table_ids else 0
        fetch_k = top_k + n_exclude + 5

        # Use unified search with 'raw' index
        results = unified_search(
            query=query_text,
            dataset_name=dataset_name,
            top_k=fetch_k,
            rag_type=rag_type,
            index_key=INDEX_KEY_RAW,
        )

        hard_negatives = []
        for table_id, score, meta in results:
            # Skip any ground truth table
            if exclude_table_ids and str(table_id) in exclude_table_ids:
                continue

            # Build table info from metadata
            table_info = {
                "table_id": table_id,
                "document_title": meta.get("document_title", ""),
                "section_title": meta.get("section_title", ""),
                "columns": meta.get("columns", []),
                "sample_rows": "",  # Not stored in raw index
                "table_text": meta.get("table_description", ""),
                "similarity_score": score,
            }

            # Enrich from table_lookup if available
            if table_lookup and table_id in table_lookup:
                lookup_info = table_lookup[table_id]
                table_info["sample_rows"] = lookup_info.get("sample_rows", "")
                if not table_info["columns"]:
                    table_info["columns"] = _parse_columns(lookup_info.get("columns", []))

            hard_negatives.append(table_info)

            if len(hard_negatives) >= top_k:
                break

        return hard_negatives

    except Exception as e:
        logger.error(f"Error retrieving hard negatives via unified search: {e}")
        return []


def _retrieve_tables_for_cluster(
    db,
    query_indices: List[int],
    queries: List[Dict],
    tables: List[Dict],
    table_store_name: str,
    table_embedding_field: str,  # Kept for interface compatibility
    retrieval_top_k: int,
    query_embedding_map: Optional[Dict[str, np.ndarray]] = None,  # Deprecated, not used
    dataset_name: Optional[str] = None,  # Required for unified search
    rag_type: str = "hybrid",  # Retrieval type: 'bm25', 'vector', or 'hybrid'
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant tables for a cluster's queries using unified search.

    Similar to sample_and_retrieve but for a subset of queries.
    Returns query-table pairs with Ground Truth + Hard Negatives.

    Args:
        db: LanceDB connection (kept for interface compatibility)
        query_indices: Indices of queries in this cluster
        queries: All queries list
        tables: All tables list
        table_store_name: Name of table store (used to infer dataset_name)
        table_embedding_field: Embedding field name (kept for interface compatibility)
        retrieval_top_k: Number of hard negatives per query
        query_embedding_map: Deprecated, not used (unified search uses text)
        dataset_name: Dataset identifier (required for unified search)
        rag_type: Retrieval type - 'bm25', 'vector', or 'hybrid'
    """
    # Get queries for this cluster
    cluster_queries = [queries[i] for i in query_indices]

    if not cluster_queries:
        return []

    # Build table lookup
    table_lookup = _build_table_lookup(tables)

    query_table_pairs = []

    for query in cluster_queries:
        query_id = str(query.get("query_id") or query.get("qid") or query.get("id"))
        query_text = query.get("query_text") or query.get("question") or query.get("query")

        # Support multiple ground truth tables
        gt_table_ids = query.get("ground_truth_table_ids", [])
        if isinstance(gt_table_ids, str):
            try:
                gt_table_ids = json.loads(gt_table_ids)
            except:
                gt_table_ids = []

        # Fallback to single ground_truth_table_id for data compatibility
        if not gt_table_ids:
            single_gt = query.get("ground_truth_table_id")
            if single_gt:
                gt_table_ids = [single_gt]

        # Get Ground Truth tables info (multiple)
        # For training: randomly sample ONE ground truth to avoid bias
        sampled_gt_id = random.choice(gt_table_ids) if gt_table_ids else None

        ground_truth_table = None
        if sampled_gt_id:
            gt_info = table_lookup.get(sampled_gt_id)
            if gt_info:
                ground_truth_table = {
                    "table_id": sampled_gt_id,
                    "document_title": gt_info.get("document_title", ""),
                    "section_title": gt_info.get("section_title", ""),
                    "columns": _parse_columns(gt_info.get("columns", [])),
                    "sample_rows": gt_info.get("sample_rows", ""),
                    "table_text": gt_info.get("table_text", ""),
                    "is_ground_truth": True,
                }

        # Retrieve Hard Negatives using unified search (supports BM25/Vector/Hybrid)
        hard_negatives = _retrieve_hard_negatives_unified(
            query_text=query_text,
            dataset_name=dataset_name,
            exclude_table_ids=set(gt_table_ids),
            top_k=retrieval_top_k,
            rag_type=rag_type,
            table_lookup=table_lookup,
        )

        for hn in hard_negatives:
            hn["is_ground_truth"] = False

        query_table_pairs.append({
            "query_id": query_id,
            "query_text": query_text,
            "has_ground_truth": ground_truth_table is not None,
            "ground_truth_table": ground_truth_table,  # Randomly sampled GT
            "ground_truth_table_ids": gt_table_ids,  # All GT IDs for reference
            "hard_negatives": hard_negatives,
            "retrieved_tables": (
                ([ground_truth_table] if ground_truth_table else []) + hard_negatives
            ),
        })

    return query_table_pairs


__all__ = [
    "_parse_columns",
    "_build_table_lookup",
    "_retrieve_hard_negatives_unified",
    "_retrieve_tables_for_cluster",
]
