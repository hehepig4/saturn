#!/usr/bin/env python3
"""
Analyze HyDE (Hypothetical Document Embedding) Effectiveness for Table Retrieval

This script evaluates whether HyDE improves semantic retrieval performance by comparing:
1. Raw Query: Original user query for semantic search
2. HyDE Table Description: LLM-generated hypothetical table description
3. HyDE Column Descriptions: LLM-generated hypothetical column descriptions
4. Combined HyDE: Both table and column descriptions combined

The script supports different retriever types:
- bm25: Only BM25 lexical search
- vector: Only FAISS vector search  
- hybrid: RRF fusion of BM25 + Vector (default)

Usage:
    # Compare raw query vs HyDE table description
    python scripts/eval/analyze_hyde_retrieval.py -d fetaqa --compare-table-desc

    # Compare raw query vs HyDE column descriptions
    python scripts/eval/analyze_hyde_retrieval.py -d fetaqa --compare-column-desc

    # Full comparison of all modes
    python scripts/eval/analyze_hyde_retrieval.py -d fetaqa --full-compare

    # Compare with specific retriever type
    python scripts/eval/analyze_hyde_retrieval.py -d fetaqa --retriever bm25 --full-compare

    # Comprehensive evaluation: all retriever types × all HyDE modes
    python scripts/eval/analyze_hyde_retrieval.py -d fetaqa --comprehensive

    # Analyze specific cases
    python scripts/eval/analyze_hyde_retrieval.py -d fetaqa --case 1 2 3

    # Show failure cases
    python scripts/eval/analyze_hyde_retrieval.py -d fetaqa --show-failures 5
"""

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
import _path_setup  # noqa: F401

from loguru import logger
from core.paths import get_db_path
from store.store_singleton import get_store


# ==================== Data Loading ====================

def load_unified_analysis(
    dataset: str, 
    llm_suffix: str = "local",
    rag_type: str = None,
    no_primitive_classes: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load unified analysis results from JSON file.
    
    Args:
        dataset: Dataset name
        llm_suffix: LLM identifier (e.g., "local", "gemini")
        rag_type: RAG retrieval type ("bm25", "vector", "hybrid") or None for any
        no_primitive_classes: If True, look for _no_pc suffix files
    
    Searches for files in order:
    1. Exact match with rag_type and no_pc if specified
    2. {dataset}_test_unified_analysis_all_{llm}_rag{k}_{type}[_no_pc].json
    3. {dataset}_test_unified_analysis_all_{llm}_rag{k}.json (legacy)
    4. {dataset}_unified_analysis_all_{llm}.json
    """
    eval_dir = get_db_path() / "eval_results"
    
    # Build search patterns based on options
    patterns = []
    
    # New format patterns with rag_type
    if rag_type:
        pc_suffix = "_no_pc" if no_primitive_classes else ""
        patterns.extend([
            f"{dataset}_test_unified_analysis_all_{llm_suffix}_rag3_{rag_type}{pc_suffix}.json",
            f"{dataset}_test_unified_analysis_all_{llm_suffix}_rag5_{rag_type}{pc_suffix}.json",
        ])
    
    # Legacy patterns (backward compatible)
    patterns.extend([
        f"{dataset}_test_unified_analysis_all_{llm_suffix}_rag3.json",
        f"{dataset}_unified_analysis_all_{llm_suffix}_rag3.json",
        f"{dataset}_unified_analysis_all_{llm_suffix}.json",
    ])
    
    for pattern in patterns:
        filepath = eval_dir / pattern
        if filepath.exists():
            logger.info(f"Loading unified analysis from: {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} query analyses")
            return data
    
    # Pattern 2: numbered files (find largest)
    import glob
    numbered_patterns = [
        f"{dataset}_test_unified_analysis_*_{llm_suffix}_rag*_{rag_type or '*'}.json",
        f"{dataset}_test_unified_analysis_*_{llm_suffix}_rag*.json",
        f"{dataset}_unified_analysis_*_{llm_suffix}_rag*.json",
        f"{dataset}_unified_analysis_*_{llm_suffix}.json",
    ]
    
    for pattern in numbered_patterns:
        files = list(eval_dir.glob(pattern))
        # Filter by no_pc if specified
        if no_primitive_classes:
            files = [f for f in files if "_no_pc" in f.name]
        elif not no_primitive_classes and rag_type:
            files = [f for f in files if "_no_pc" not in f.name]
        
        if files:
            # Sort by file size (largest first) to get most complete
            files.sort(key=lambda f: f.stat().st_size, reverse=True)
            filepath = files[0]
            logger.info(f"Loading unified analysis from: {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} query analyses")
            return data
    
    raise FileNotFoundError(
        f"No unified analysis file found for dataset: {dataset}\n"
        f"Searched patterns: {patterns}\n"
        f"Run: python demos/retrieval.py --analyze-queries -d {dataset} -n -1 --use-rag"
    )


def load_indexes_and_embedder(dataset: str):
    """Load FAISS index, metadata, BM25 retriever, and embedder using unified interface.
    
    Supports partial loading - if only one index type exists, only that index will be loaded.
    The embedder is only loaded if FAISS index exists.
    """
    from workflows.retrieval.unified_search import (
        load_unified_indexes, get_text_embedder
    )
    from workflows.retrieval.config import INDEX_KEY_TD_CD_CS
    
    logger.info("Loading indexes and embedder...")
    start = time.time()
    
    faiss_index, metadata_list, bm25_retriever, table_ids = load_unified_indexes(dataset, INDEX_KEY_TD_CD_CS)
    
    # Only load embedder if FAISS index exists (for vector search)
    embedder = None
    if faiss_index is not None:
        embedder = get_text_embedder()
    
    load_time = time.time() - start
    
    # Log what's available
    available_indexes = []
    if faiss_index is not None:
        available_indexes.append("FAISS")
        logger.info(f"  ✓ FAISS: {faiss_index.ntotal} vectors")
    else:
        logger.warning(f"  ✗ FAISS index not found (vector/hybrid search unavailable)")
    
    if bm25_retriever is not None:
        available_indexes.append("BM25")
        logger.info(f"  ✓ BM25: {len(table_ids)} documents")
    else:
        logger.warning(f"  ✗ BM25 index not found (bm25/hybrid search unavailable)")
    
    if not available_indexes:
        raise RuntimeError(
            f"No indexes found for dataset: {dataset}\n"
            f"Run: python demos/run_upo_pipeline.py -d {dataset} -s retrieval_index"
        )
    
    logger.info(f"  Available indexes: {', '.join(available_indexes)}")
    logger.info(f"  Loaded in {load_time:.2f}s")
    
    return faiss_index, metadata_list, bm25_retriever, table_ids, embedder


# ==================== Search Functions ====================

# Retriever types
RETRIEVER_BM25_ONLY = "bm25"  # For backward compatibility, keep variable name
RETRIEVER_VECTOR_ONLY = "vector"  # For backward compatibility, keep variable name
RETRIEVER_HYBRID = "hybrid"
RETRIEVER_HYBRID_SUM = "hybrid-sum"

RETRIEVER_TYPES = [RETRIEVER_BM25_ONLY, RETRIEVER_VECTOR_ONLY, RETRIEVER_HYBRID, RETRIEVER_HYBRID_SUM]

# HyDE modes
HYDE_RAW = "raw"
HYDE_TABLE_DESC = "table_desc"
HYDE_COLUMN_DESC = "column_desc"
HYDE_COMBINED = "combined"

HYDE_MODES = [HYDE_RAW, HYDE_TABLE_DESC, HYDE_COLUMN_DESC, HYDE_COMBINED]


def semantic_search(
    query_text: str,
    faiss_index,
    metadata_list: List[Dict],
    bm25_retriever,
    table_ids: List[str],
    embedder,
    top_k: int = 100,
    retriever_type: str = RETRIEVER_HYBRID,
) -> List[Tuple[str, float]]:
    """
    Perform semantic search using specified retriever type.
    
    Args:
        retriever_type: One of "bm25", "vector", "hybrid"
    
    Returns:
        List of (table_id, score) tuples
    """
    from workflows.retrieval.unified_search import (
        vector_search, bm25_search, reciprocal_rank_fusion, normalized_score_fusion
    )
    
    vec_results = []
    bm25_results = []
    
    # Vector search (if needed and available)
    if retriever_type in [RETRIEVER_VECTOR_ONLY, RETRIEVER_HYBRID, RETRIEVER_HYBRID_SUM]:
        if faiss_index is None:
            if retriever_type == RETRIEVER_VECTOR_ONLY:
                raise RuntimeError(
                    f"Vector-only search requested but FAISS index not available.\n"
                    f"Run: python demos/run_upo_pipeline.py -d <dataset> -s retrieval_index --rag-type vector"
                )
            logger.debug("FAISS index not available, skipping vector search in hybrid mode")
        elif embedder is None:
            raise RuntimeError("FAISS index exists but embedder not loaded")
        else:
            query_emb_list = embedder.compute_query_embeddings(query_text)
            query_emb = np.array(query_emb_list[0], dtype=np.float32)
            vec_results = vector_search(query_emb, faiss_index, metadata_list, top_k * 2)
    
    # BM25 search (if needed and available)
    if retriever_type in [RETRIEVER_BM25_ONLY, RETRIEVER_HYBRID, RETRIEVER_HYBRID_SUM]:
        if bm25_retriever is None:
            if retriever_type == RETRIEVER_BM25_ONLY:
                raise RuntimeError(
                    f"BM25-only search requested but BM25 index not available.\n"
                    f"Run: python demos/run_upo_pipeline.py -d <dataset> -s retrieval_index --rag-type bm25"
                )
            logger.debug("BM25 index not available, skipping BM25 search in hybrid mode")
        else:
            bm25_results = bm25_search(query_text, bm25_retriever, table_ids, metadata_list, top_k * 2)
    
    # Combine results based on retriever type
    if retriever_type == RETRIEVER_VECTOR_ONLY:
        results = vec_results
    elif retriever_type == RETRIEVER_BM25_ONLY:
        results = bm25_results
    else:  # RETRIEVER_HYBRID or RETRIEVER_HYBRID_SUM
        if vec_results and bm25_results:
            if retriever_type == RETRIEVER_HYBRID_SUM:
                results = normalized_score_fusion(vec_results, bm25_results)
            else:
                results = reciprocal_rank_fusion(vec_results, bm25_results)
        elif vec_results:
            logger.debug("Hybrid mode: Only vector results available")
            results = vec_results
        elif bm25_results:
            logger.debug("Hybrid mode: Only BM25 results available")
            results = bm25_results
        else:
            logger.warning("Hybrid mode: No results from either index")
            results = []
    
    return [(tid, score) for tid, score, _ in results[:top_k]]


def find_gt_rank(results: List[Tuple[str, float]], gt_tables: List[str]) -> Optional[int]:
    """Find the best (minimum) rank among all ground truth tables (1-indexed).
    
    Args:
        results: List of (table_id, score) tuples in retrieval order
        gt_tables: List of ground truth table IDs
        
    Returns:
        Best rank if any GT found, None if none found
    """
    result_ids = [tid for tid, _ in results]
    best_rank = None
    for gt in gt_tables:
        try:
            rank = result_ids.index(gt) + 1
            if best_rank is None or rank < best_rank:
                best_rank = rank
        except ValueError:
            continue
    return best_rank


# ==================== Metrics Computation ====================

def compute_metrics(ranks: List[Optional[int]], total: int) -> Dict[str, float]:
    """Compute retrieval metrics from rank list."""
    def hit_at_k(k: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= k) / total
    
    def mrr() -> float:
        rrs = [1.0 / r for r in ranks if r is not None]
        return sum(rrs) / total if rrs else 0.0
    
    return {
        'hit@1': hit_at_k(1),
        'hit@3': hit_at_k(3),
        'hit@5': hit_at_k(5),
        'hit@10': hit_at_k(10),
        'hit@50': hit_at_k(50),
        'hit@100': hit_at_k(100),
        'mrr': mrr(),
        'not_found': sum(1 for r in ranks if r is None),
    }


def print_metrics(metrics: Dict[str, float], label: str, total: int):
    """Pretty print metrics."""
    print(f"\n📊 {label} (n={total})")
    print(f"   Hit@1:   {metrics['hit@1']*100:.1f}%")
    print(f"   Hit@3:   {metrics['hit@3']*100:.1f}%")
    print(f"   Hit@5:   {metrics['hit@5']*100:.1f}%")
    print(f"   Hit@10:  {metrics['hit@10']*100:.1f}%")
    print(f"   Hit@50:  {metrics['hit@50']*100:.1f}%")
    print(f"   Hit@100: {metrics['hit@100']*100:.1f}%")
    print(f"   MRR:        {metrics['mrr']:.4f}")
    print(f"   Not Found:  {metrics['not_found']} ({metrics['not_found']/total*100:.1f}%)")


# ==================== Main Analysis Functions ====================

def analyze_hyde_mode(
    analysis_data: List[Dict[str, Any]],
    faiss_index,
    metadata_list: List[Dict],
    bm25_retriever,
    table_ids: List[str],
    embedder,
    mode: str = "table_desc",
    top_k: int = 100,
    num_queries: int = -1,
    progress_interval: int = 100,
    retriever_type: str = RETRIEVER_HYBRID,
    silent: bool = False,
) -> Dict[str, Any]:
    """
    Analyze retrieval performance for a specific HyDE mode and retriever type.
    
    Modes:
    - "raw": Original query (baseline)
    - "table_desc": Hypothetical table description
    - "column_desc": Hypothetical column descriptions
    - "combined": Table description + column descriptions
    
    Retriever Types:
    - "bm25-only": Only BM25 lexical search
    - "vector-only": Only FAISS vector search
    - "hybrid": RRF fusion of BM25 + Vector
    """
    
    if num_queries > 0:
        analysis_data = analysis_data[:num_queries]
    
    total = len(analysis_data)
    if not silent:
        logger.info(f"Analyzing {total} queries with mode={mode}, retriever={retriever_type}")
    
    ranks: List[Optional[int]] = []
    multi_gt_count = 0
    
    for i, item in enumerate(analysis_data):
        query = item['query']
        # Support multi-GT: use gt_tables if available, otherwise wrap single gt_table
        gt_tables = item.get('gt_tables', [item['gt_table']] if item.get('gt_table') else [])
        if len(gt_tables) > 1:
            multi_gt_count += 1
        analysis = item.get('analysis', {})
        
        # Skip if error
        if item.get('error') or not analysis:
            ranks.append(None)
            continue
        
        # Determine search text based on mode
        if mode == "raw":
            search_text = query
        elif mode == "table_desc":
            search_text = analysis.get('hypothetical_table_description', query)
            if not search_text:
                search_text = query
        elif mode == "column_desc":
            search_text = analysis.get('hypothetical_column_descriptions', query)
            if not search_text:
                search_text = query
        elif mode == "combined":
            # Build combined text from table_desc + column_desc
            table_desc = analysis.get('hypothetical_table_description', '')
            col_desc = analysis.get('hypothetical_column_descriptions', '')
            search_text = f"{table_desc}\n{col_desc}".strip()
            if not search_text:
                search_text = query
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Perform search with specified retriever type
        results = semantic_search(
            search_text, faiss_index, metadata_list, 
            bm25_retriever, table_ids, embedder, top_k,
            retriever_type=retriever_type
        )
        
        # Find GT rank (supports multi-GT)
        rank = find_gt_rank(results, gt_tables)
        ranks.append(rank)
        
        # Progress
        if not silent and (i + 1) % progress_interval == 0:
            found = sum(1 for r in ranks if r is not None)
            pct = found / (i + 1) * 100
            print(f"   [{retriever_type}/{mode}] Processed {i+1}/{total}, hit@{top_k}: {pct:.1f}%")
    
    metrics = compute_metrics(ranks, total)
    metrics['mode'] = mode
    metrics['retriever_type'] = retriever_type
    metrics['total'] = total
    metrics['multi_gt_queries'] = multi_gt_count
    metrics['ranks'] = ranks
    
    return metrics


def compare_hyde_modes(
    analysis_data: List[Dict[str, Any]],
    faiss_index,
    metadata_list: List[Dict],
    bm25_retriever,
    table_ids: List[str],
    embedder,
    modes: List[str] = ["raw", "table_desc"],
    top_k: int = 100,
    num_queries: int = -1,
    retriever_type: str = RETRIEVER_HYBRID,
    show_failures: int = 0,
    silent: bool = False,
) -> Dict[str, Any]:
    """
    Compare multiple HyDE modes for a specific retriever type.
    """
    if not silent:
        print(f"\n{'='*80}")
        print(f"🔬 HyDE Comparison Analysis")
        print(f"   Retriever: {retriever_type}")
        print(f"   Modes: {modes}")
        print(f"   Queries: {num_queries if num_queries > 0 else len(analysis_data)}")
        print(f"{'='*80}")
    
    # Run each mode
    results_by_mode = {}
    for mode in modes:
        if not silent:
            print(f"\n{'─'*40}")
            print(f"Running mode: {mode}")
            print(f"{'─'*40}")
        results_by_mode[mode] = analyze_hyde_mode(
            analysis_data, faiss_index, metadata_list,
            bm25_retriever, table_ids, embedder,
            mode=mode, top_k=top_k, num_queries=num_queries,
            retriever_type=retriever_type, silent=silent
        )
    
    if silent:
        return {
            'retriever_type': retriever_type,
            'modes': modes,
            'results': results_by_mode,
        }
    
    # Print individual metrics
    print(f"\n{'='*80}")
    print(f"📊 METRICS COMPARISON ({retriever_type})")
    print(f"{'='*80}")
    
    total = results_by_mode[modes[0]]['total']
    for mode in modes:
        print_metrics(results_by_mode[mode], f"{mode.upper()}", total)
    
    # Comparison table
    print(f"\n{'─'*80}")
    print("📈 Summary Table")
    print(f"{'─'*80}")
    
    headers = ["Metric"] + [m.upper() for m in modes]
    col_widths = [15] + [15] * len(modes)
    
    print("   " + "".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)))
    print("   " + "-" * sum(col_widths))
    
    for metric in ['hit@1', 'hit@3', 'hit@5', 'hit@10', 'hit@50', 'hit@100', 'mrr']:
        row = [metric]
        for mode in modes:
            val = results_by_mode[mode][metric]
            if metric == 'mrr':
                row.append(f"{val:.4f}")
            else:
                row.append(f"{val*100:.1f}%")
        print("   " + "".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row))))
    
    # Case-by-case analysis (for pairwise comparison)
    if len(modes) >= 2:
        print(f"\n{'='*80}")
        print("📊 Case-by-Case Analysis (Hit@10)")
        print(f"{'='*80}")
        
        baseline_mode = modes[0]
        baseline_ranks = results_by_mode[baseline_mode]['ranks']
        
        for compare_mode in modes[1:]:
            compare_ranks = results_by_mode[compare_mode]['ranks']
            
            baseline_wins = []
            compare_wins = []
            both_succeed = []
            both_fail = []
            
            for i in range(min(len(baseline_ranks), len(compare_ranks))):
                b_rank = baseline_ranks[i]
                c_rank = compare_ranks[i]
                
                b_hit = b_rank is not None and b_rank <= 10
                c_hit = c_rank is not None and c_rank <= 10
                
                if b_hit and not c_hit:
                    baseline_wins.append(i)
                elif c_hit and not b_hit:
                    compare_wins.append(i)
                elif b_hit and c_hit:
                    both_succeed.append(i)
                else:
                    both_fail.append(i)
            
            print(f"\n   {baseline_mode.upper()} vs {compare_mode.upper()}:")
            print(f"      {baseline_mode} wins: {len(baseline_wins)} queries")
            print(f"      {compare_mode} wins: {len(compare_wins)} queries")
            print(f"      Both succeed: {len(both_succeed)} queries")
            print(f"      Both fail: {len(both_fail)} queries")
            
            # Delta
            delta = len(compare_wins) - len(baseline_wins)
            delta_sign = "+" if delta > 0 else ""
            print(f"      Net improvement ({compare_mode} - {baseline_mode}): {delta_sign}{delta}")
            
            # Show example cases where HyDE wins
            if show_failures > 0 and compare_wins:
                print(f"\n   📌 Examples where {compare_mode} wins (showing {min(show_failures, len(compare_wins))}):")
                for idx in compare_wins[:show_failures]:
                    item = analysis_data[idx] if num_queries <= 0 else analysis_data[:num_queries][idx]
                    analysis = item.get('analysis', {})
                    gt_table_id = item['gt_table']
                    gt_summary = get_table_summary(gt_table_id, metadata_list)
                    
                    print(f"\n      ────────────────────────────────────────")
                    print(f"      Query: {item['query']}")
                    print(f"      GT Table: {gt_table_id}")
                    print(f"      Rank: {baseline_mode}={baseline_ranks[idx] or '>100'} → {compare_mode}={compare_ranks[idx]}")
                    
                    # Show GT table schema
                    print(f"      ─── GT Table Schema ───")
                    if gt_summary:
                        print(f"      table_desc: {gt_summary.get('table_description', 'N/A')}")
                        print(f"      column_desc: {gt_summary.get('column_descriptions', 'N/A')}")
                    else:
                        print(f"      (GT table not found in metadata)")
                    
                    # Show all HyDE generated content
                    print(f"      ─── HyDE Generated ───")
                    print(f"      table_desc: {analysis.get('hypothetical_table_description', 'N/A')}")
                    print(f"      column_desc: {analysis.get('hypothetical_column_descriptions', 'N/A')}")
                    print(f"      tbox: {analysis.get('tbox_constraints', [])}")
                    print(f"      abox: {analysis.get('abox_constraints', [])}")
            
            # Show example cases where baseline wins (HyDE hurts)
            if show_failures > 0 and baseline_wins:
                print(f"\n   📌 Examples where {baseline_mode} wins / {compare_mode} hurts (showing {min(show_failures, len(baseline_wins))}):")
                for idx in baseline_wins[:show_failures]:
                    item = analysis_data[idx] if num_queries <= 0 else analysis_data[:num_queries][idx]
                    analysis = item.get('analysis', {})
                    gt_table_id = item['gt_table']
                    gt_summary = get_table_summary(gt_table_id, metadata_list)
                    
                    print(f"\n      ────────────────────────────────────────")
                    print(f"      Query: {item['query']}")
                    print(f"      GT Table: {gt_table_id}")
                    print(f"      Rank: {baseline_mode}={baseline_ranks[idx]} → {compare_mode}={compare_ranks[idx] or '>100'}")
                    
                    # Show GT table schema
                    print(f"      ─── GT Table Schema ───")
                    if gt_summary:
                        print(f"      table_desc: {gt_summary.get('table_description', 'N/A')}")
                        print(f"      column_desc: {gt_summary.get('column_descriptions', 'N/A')}")
                    else:
                        print(f"      (GT table not found in metadata)")
                    
                    # Show all HyDE generated content
                    print(f"      ─── HyDE Generated ───")
                    print(f"      table_desc: {analysis.get('hypothetical_table_description', 'N/A')}")
                    print(f"      column_desc: {analysis.get('hypothetical_column_descriptions', 'N/A')}")
                    print(f"      tbox: {analysis.get('tbox_constraints', [])}")
                    print(f"      abox: {analysis.get('abox_constraints', [])}")
    
    return {
        'modes': modes,
        'results': results_by_mode,
    }


def get_table_summary(table_id: str, metadata_list: List[Dict]) -> Optional[Dict]:
    """Get table summary from metadata list by table_id."""
    for meta in metadata_list:
        if meta.get('table_id') == table_id:
            return meta
    return None


def analyze_specific_cases(
    analysis_data: List[Dict[str, Any]],
    faiss_index,
    metadata_list: List[Dict],
    bm25_retriever,
    table_ids: List[str],
    embedder,
    case_indices: List[int],
    top_k: int = 100,
    show_fn: int = 3,
) -> None:
    """
    Analyze specific cases in detail with GT table description, 
    HyDE description, and top false negatives comparison.
    """
    modes = ["raw", "table_desc", "column_desc", "combined"]
    
    for idx in case_indices:
        if idx < 1 or idx > len(analysis_data):
            print(f"Invalid case index: {idx} (valid: 1-{len(analysis_data)})")
            continue
        
        item = analysis_data[idx - 1]
        query = item['query']
        # Support multi-GT
        gt_tables = item.get('gt_tables', [item['gt_table']] if item.get('gt_table') else [])
        gt_table = gt_tables[0] if gt_tables else None  # Primary GT for display
        analysis = item.get('analysis', {})
        
        print(f"\n{'='*80}")
        print(f"📍 CASE {idx}")
        print(f"{'='*80}")
        
        print(f"\n📝 Query:")
        print(f"   {query}")
        
        print(f"\n📋 Ground Truth Table{'s' if len(gt_tables) > 1 else ''}: {gt_tables if len(gt_tables) > 1 else gt_table}")
        
        # Load GT table summary from index metadata
        gt_summary = get_table_summary(gt_table, metadata_list)
        if gt_summary:
            print(f"\n📊 GT Table Summary (from index):")
            print(f"   Description: {gt_summary.get('table_description', 'N/A')[:200]}...")
            col_desc = gt_summary.get('column_descriptions', '')
            if col_desc:
                print(f"   Columns: {col_desc[:150]}...")
        
        if analysis:
            print(f"\n📄 HyDE Table Description (generated):")
            print(f"   {analysis.get('hypothetical_table_description', 'N/A')}")
            
            print(f"\n📄 HyDE Column Descriptions (generated):")
            col_desc = analysis.get('hypothetical_column_descriptions', 'N/A')
            # HyDE output uses newlines to separate columns
            for line in col_desc.split('\n')[:5]:
                print(f"   {line}")
            if '\n' in col_desc and len(col_desc.split('\n')) > 5:
                print(f"   ... ({len(col_desc.split(chr(10)))} columns total)")
            
            print(f"\n🏷️  TBox Constraints: {analysis.get('tbox_constraints', [])}")
            print(f"📄 ABox Constraints: {analysis.get('abox_constraints', [])}")
        
        # Run retrieval for each mode and show top false negatives
        print(f"\n🔍 Retrieval Results:")
        for mode in modes:
            if mode == "raw":
                search_text = query
            elif mode == "table_desc":
                search_text = analysis.get('hypothetical_table_description', query) if analysis else query
            elif mode == "column_desc":
                search_text = analysis.get('hypothetical_column_descriptions', query) if analysis else query
            elif mode == "combined":
                # Build combined text from table_desc + column_desc
                table_desc = analysis.get('hypothetical_table_description', '') if analysis else ''
                col_desc = analysis.get('hypothetical_column_descriptions', '') if analysis else ''
                search_text = f"{table_desc}\n{col_desc}".strip() or query
            
            results = semantic_search(
                search_text, faiss_index, metadata_list,
                bm25_retriever, table_ids, embedder, top_k
            )
            rank = find_gt_rank(results, gt_tables)
            
            status = f"Rank {rank}" if rank else f"NOT FOUND (>top-{top_k})"
            print(f"   {mode.upper():15s}: {status}")
        
        # Show top false negatives for raw mode
        print(f"\n❌ Top False Negatives (Raw Query, showing {show_fn}):")
        raw_results = semantic_search(
            query, faiss_index, metadata_list,
            bm25_retriever, table_ids, embedder, top_k
        )
        fn_count = 0
        for rank_i, (tid, score) in enumerate(raw_results[:20], 1):
            if tid in gt_tables:
                continue  # Skip GT tables
            fn_count += 1
            if fn_count > show_fn:
                break
            
            fn_summary = get_table_summary(tid, metadata_list)
            print(f"\n   [{rank_i}] {tid[:60]}... (score={score:.4f})")
            if fn_summary:
                print(f"       Desc: {fn_summary.get('table_description', 'N/A')[:120]}...")


def analyze_detailed_comparison(
    analysis_data: List[Dict[str, Any]],
    faiss_index,
    metadata_list: List[Dict],
    bm25_retriever,
    table_ids: List[str],
    embedder,
    num_queries: int = 100,
    top_k: int = 100,
    show_cases: int = 5,
) -> None:
    """
    Detailed comparison showing GT description vs HyDE description vs top FN descriptions.
    Focus on cases where HyDE hurts performance.
    """
    if num_queries > 0:
        analysis_data = analysis_data[:num_queries]
    
    print(f"\n{'='*80}")
    print(f"🔬 Detailed HyDE vs GT Description Comparison")
    print(f"   Analyzing {len(analysis_data)} queries, showing {show_cases} HyDE-hurts cases")
    print(f"{'='*80}")
    
    hyde_hurts_cases = []
    
    for i, item in enumerate(analysis_data):
        query = item['query']
        # Support multi-GT
        gt_tables = item.get('gt_tables', [item['gt_table']] if item.get('gt_table') else [])
        gt_table = gt_tables[0] if gt_tables else None  # Primary GT for display
        analysis = item.get('analysis', {})
        
        if item.get('error') or not analysis:
            continue
        
        # Run raw query search
        raw_results = semantic_search(
            query, faiss_index, metadata_list,
            bm25_retriever, table_ids, embedder, top_k
        )
        raw_rank = find_gt_rank(raw_results, gt_tables)
        
        # Run HyDE table_desc search
        hyde_text = analysis.get('hypothetical_table_description', query)
        hyde_results = semantic_search(
            hyde_text, faiss_index, metadata_list,
            bm25_retriever, table_ids, embedder, top_k
        )
        hyde_rank = find_gt_rank(hyde_results, gt_tables)
        
        # Check if HyDE hurts: raw hits top-10 but HyDE misses
        raw_hit = raw_rank is not None and raw_rank <= 10
        hyde_hit = hyde_rank is not None and hyde_rank <= 10
        
        if raw_hit and not hyde_hit:
            hyde_hurts_cases.append({
                'index': i + 1,
                'query': query,
                'gt_table': gt_table,
                'raw_rank': raw_rank,
                'hyde_rank': hyde_rank,
                'hyde_desc': hyde_text,
                'raw_results': raw_results[:5],
                'hyde_results': hyde_results[:5],
            })
    
    print(f"\n📊 Found {len(hyde_hurts_cases)} cases where HyDE hurts (raw@10 hit, HyDE@10 miss)")
    
    # Show detailed cases
    for case in hyde_hurts_cases[:show_cases]:
        print(f"\n{'─'*80}")
        print(f"📍 Case {case['index']}")
        print(f"{'─'*80}")
        
        print(f"\n📝 Query: {case['query']}")
        print(f"\n📋 GT Table: {case['gt_table'][:60]}...")
        
        # GT table summary
        gt_summary = get_table_summary(case['gt_table'], metadata_list)
        if gt_summary:
            print(f"\n📊 GT Table Description (indexed):")
            print(f"   {gt_summary.get('table_description', 'N/A')[:200]}")
        
        print(f"\n📄 HyDE Description (generated):")
        print(f"   {case['hyde_desc'][:200]}")
        
        print(f"\n🎯 Performance:")
        print(f"   Raw Query Rank:  {case['raw_rank']}")
        print(f"   HyDE Desc Rank:  {case['hyde_rank'] or '>100'}")
        
        # Top results from HyDE search (false negatives)
        print(f"\n❌ HyDE Search Top Results (why they rank higher):")
        for rank_i, (tid, score) in enumerate(case['hyde_results'][:3], 1):
            fn_summary = get_table_summary(tid, metadata_list)
            is_gt = " ✅ GT" if tid == case['gt_table'] else ""
            print(f"\n   [{rank_i}] {tid[:50]}...{is_gt}")
            print(f"       Score: {score:.4f}")
            if fn_summary:
                print(f"       Desc: {fn_summary.get('table_description', 'N/A')[:150]}...")


# ==================== Comprehensive Evaluation ====================

def comprehensive_evaluation(
    analysis_data: List[Dict[str, Any]],
    faiss_index,
    metadata_list: List[Dict],
    bm25_retriever,
    table_ids: List[str],
    embedder,
    top_k: int = 100,
    num_queries: int = -1,
    retriever_types: List[str] = None,
    hyde_modes: List[str] = None,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation across all retriever types and HyDE modes.
    
    Produces a matrix of results:
    - Rows: Retriever types (bm25-only, vector-only, hybrid)
    - Columns: HyDE modes (raw, table_desc, column_desc, combined)
    """
    if retriever_types is None:
        retriever_types = RETRIEVER_TYPES
    if hyde_modes is None:
        hyde_modes = HYDE_MODES
    
    total_queries = len(analysis_data) if num_queries <= 0 else min(num_queries, len(analysis_data))
    
    print(f"\n{'='*100}")
    print(f"🔬 COMPREHENSIVE HYDE EVALUATION")
    print(f"{'='*100}")
    print(f"   Dataset queries: {total_queries}")
    print(f"   Retriever types: {retriever_types}")
    print(f"   HyDE modes: {hyde_modes}")
    print(f"   Top-K: {top_k}")
    print(f"{'='*100}")
    
    # Results matrix: retriever_type -> mode -> metrics
    all_results: Dict[str, Dict[str, Dict]] = {}
    
    total_combinations = len(retriever_types) * len(hyde_modes)
    current = 0
    
    for retriever_type in retriever_types:
        all_results[retriever_type] = {}
        
        for mode in hyde_modes:
            current += 1
            print(f"\n[{current}/{total_combinations}] Running: {retriever_type} × {mode}...")
            
            metrics = analyze_hyde_mode(
                analysis_data, faiss_index, metadata_list,
                bm25_retriever, table_ids, embedder,
                mode=mode, top_k=top_k, num_queries=num_queries,
                retriever_type=retriever_type, silent=True,
                progress_interval=500,  # Less frequent progress
            )
            
            # Remove ranks to save memory/output
            del metrics['ranks']
            all_results[retriever_type][mode] = metrics
            
            # Print quick result
            r1 = metrics['hit@1'] * 100
            r10 = metrics['hit@10'] * 100
            print(f"   R@1: {r1:.1f}% | R@10: {r10:.1f}%")
    
    # ==================== Print Summary Tables ====================
    
    print(f"\n{'='*100}")
    print(f"📊 COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'='*100}")
    
    # Table 1: Hit@1
    print(f"\n📈 Hit@1 Matrix")
    print(f"{'─'*80}")
    _print_matrix(all_results, retriever_types, hyde_modes, 'hit@1')
    
    # Table 2: Hit@10
    print(f"\n📈 Hit@10 Matrix")
    print(f"{'─'*80}")
    _print_matrix(all_results, retriever_types, hyde_modes, 'hit@10')
    
    # Table 3: MRR
    print(f"\n📈 MRR Matrix")
    print(f"{'─'*80}")
    _print_matrix(all_results, retriever_types, hyde_modes, 'mrr', is_percentage=False)
    
    # ==================== HyDE Delta Analysis ====================
    
    print(f"\n{'='*100}")
    print(f"📊 HYDE DELTA ANALYSIS (vs Raw Query)")
    print(f"{'='*100}")
    
    print(f"\n📈 Hit@10 Delta (HyDE - Raw)")
    print(f"{'─'*80}")
    _print_delta_matrix(all_results, retriever_types, hyde_modes, 'hit@10')
    
    print(f"\n📈 Hit@1 Delta (HyDE - Raw)")
    print(f"{'─'*80}")
    _print_delta_matrix(all_results, retriever_types, hyde_modes, 'hit@1')
    
    # ==================== Best Configuration ====================
    
    print(f"\n{'='*100}")
    print(f"🏆 BEST CONFIGURATIONS")
    print(f"{'='*100}")
    
    # Find best for each metric
    for metric in ['hit@1', 'hit@10', 'mrr']:
        best_val = -1
        best_config = None
        for rt in retriever_types:
            for mode in hyde_modes:
                val = all_results[rt][mode][metric]
                if val > best_val:
                    best_val = val
                    best_config = (rt, mode)
        
        if metric == 'mrr':
            print(f"   Best {metric}: {best_val:.4f} ({best_config[0]} + {best_config[1]})")
        else:
            print(f"   Best {metric}: {best_val*100:.1f}% ({best_config[0]} + {best_config[1]})")
    
    # Save to JSON if requested
    if output_file:
        output_path = get_db_path() / "eval_results" / output_file
        with open(output_path, 'w') as f:
            json.dump({
                'total_queries': total_queries,
                'retriever_types': retriever_types,
                'hyde_modes': hyde_modes,
                'results': all_results,
            }, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
    
    return all_results


def _print_matrix(
    results: Dict[str, Dict[str, Dict]],
    retriever_types: List[str],
    hyde_modes: List[str],
    metric: str,
    is_percentage: bool = True,
):
    """Print a matrix of results."""
    # Header
    col_width = 15
    header = "Retriever".ljust(col_width) + "".join(m.ljust(col_width) for m in hyde_modes)
    print(f"   {header}")
    print(f"   {'-' * len(header)}")
    
    # Rows
    for rt in retriever_types:
        row = rt.ljust(col_width)
        for mode in hyde_modes:
            val = results[rt][mode][metric]
            if is_percentage:
                row += f"{val*100:.1f}%".ljust(col_width)
            else:
                row += f"{val:.4f}".ljust(col_width)
        print(f"   {row}")


def _print_delta_matrix(
    results: Dict[str, Dict[str, Dict]],
    retriever_types: List[str],
    hyde_modes: List[str],
    metric: str,
):
    """Print a delta matrix (vs raw query baseline)."""
    col_width = 15
    # Only show non-raw modes for delta
    delta_modes = [m for m in hyde_modes if m != 'raw']
    
    header = "Retriever".ljust(col_width) + "".join(m.ljust(col_width) for m in delta_modes)
    print(f"   {header}")
    print(f"   {'-' * len(header)}")
    
    for rt in retriever_types:
        raw_val = results[rt]['raw'][metric]
        row = rt.ljust(col_width)
        for mode in delta_modes:
            hyde_val = results[rt][mode][metric]
            delta = (hyde_val - raw_val) * 100  # Convert to percentage points
            sign = "+" if delta > 0 else ""
            # Color coding hint in text
            marker = "✅" if delta > 0 else "❌" if delta < -0.5 else "─"
            row += f"{sign}{delta:.1f}pp {marker}".ljust(col_width)
        print(f"   {row}")


# ==================== Main Entry ====================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze HyDE effectiveness for table retrieval"
    )
    parser.add_argument("-d", "--dataset", default="fetaqa", help="Dataset name")
    parser.add_argument("-n", "--num-queries", type=int, default=-1, 
                        help="Number of queries to analyze (-1 for all)")
    parser.add_argument("--llm", default="local", help="LLM suffix in filename (default: local)")
    parser.add_argument("--top-k", type=int, default=100, help="Top-k for retrieval")
    
    # Retriever type
    parser.add_argument("--retriever", type=str, default=RETRIEVER_HYBRID,
                        choices=RETRIEVER_TYPES,
                        help=f"Retriever type: {RETRIEVER_TYPES}")
    
    # Comparison modes
    parser.add_argument("--compare-table-desc", action="store_true",
                        help="Compare raw query vs HyDE table description")
    parser.add_argument("--compare-column-desc", action="store_true",
                        help="Compare raw query vs HyDE column descriptions")
    parser.add_argument("--compare-combined", action="store_true",
                        help="Compare raw query vs combined HyDE")
    parser.add_argument("--full-compare", action="store_true",
                        help="Compare all HyDE modes")
    
    # Analysis options
    parser.add_argument("--case", type=int, nargs='+', 
                        help="Analyze specific case(s) by index (1-based)")
    parser.add_argument("--show-failures", type=int, default=0,
                        help="Show N example cases for each category")
    parser.add_argument("--detailed", action="store_true",
                        help="Show detailed comparison with GT/HyDE/FN descriptions")
    parser.add_argument("--show-cases", type=int, default=5,
                        help="Number of detailed cases to show (default: 5)")
    
    # Comprehensive evaluation
    parser.add_argument("--comprehensive", action="store_true",
                        help="Run comprehensive evaluation: all retriever types × all HyDE modes")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON filename for comprehensive results")
    
    args = parser.parse_args()
    
    # Load data
    analysis_data = load_unified_analysis(args.dataset, args.llm)
    faiss_index, metadata_list, bm25_retriever, table_ids, embedder = \
        load_indexes_and_embedder(args.dataset)
    
    # Comprehensive evaluation mode
    if args.comprehensive:
        output_file = args.output or f"{args.dataset}_comprehensive_hyde_eval.json"
        comprehensive_evaluation(
            analysis_data, faiss_index, metadata_list,
            bm25_retriever, table_ids, embedder,
            top_k=args.top_k, num_queries=args.num_queries,
            output_file=output_file,
        )
        return
    
    # Specific case analysis
    if args.case:
        analyze_specific_cases(
            analysis_data, faiss_index, metadata_list,
            bm25_retriever, table_ids, embedder,
            case_indices=args.case, top_k=args.top_k
        )
        return
    
    # Detailed comparison mode
    if args.detailed:
        analyze_detailed_comparison(
            analysis_data, faiss_index, metadata_list,
            bm25_retriever, table_ids, embedder,
            num_queries=args.num_queries, top_k=args.top_k,
            show_cases=args.show_cases,
        )
        return
    
    # Comparison modes (with retriever type support)
    retriever_type = args.retriever
    
    if args.full_compare:
        compare_hyde_modes(
            analysis_data, faiss_index, metadata_list,
            bm25_retriever, table_ids, embedder,
            modes=["raw", "table_desc", "column_desc", "combined"],
            top_k=args.top_k, num_queries=args.num_queries,
            retriever_type=retriever_type,
            show_failures=args.show_failures,
        )
    elif args.compare_table_desc:
        compare_hyde_modes(
            analysis_data, faiss_index, metadata_list,
            bm25_retriever, table_ids, embedder,
            modes=["raw", "table_desc"],
            top_k=args.top_k, num_queries=args.num_queries,
            retriever_type=retriever_type,
            show_failures=args.show_failures,
        )
    elif args.compare_column_desc:
        compare_hyde_modes(
            analysis_data, faiss_index, metadata_list,
            bm25_retriever, table_ids, embedder,
            modes=["raw", "column_desc"],
            top_k=args.top_k, num_queries=args.num_queries,
            retriever_type=retriever_type,
            show_failures=args.show_failures,
        )
    elif args.compare_combined:
        compare_hyde_modes(
            analysis_data, faiss_index, metadata_list,
            bm25_retriever, table_ids, embedder,
            modes=["raw", "combined"],
            top_k=args.top_k, num_queries=args.num_queries,
            retriever_type=retriever_type,
            show_failures=args.show_failures,
        )
    else:
        # Default: compare raw vs table_desc
        compare_hyde_modes(
            analysis_data, faiss_index, metadata_list,
            bm25_retriever, table_ids, embedder,
            modes=["raw", "table_desc"],
            top_k=args.top_k, num_queries=args.num_queries,
            retriever_type=retriever_type,
            show_failures=args.show_failures,
        )


if __name__ == "__main__":
    main()
