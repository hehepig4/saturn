#!/usr/bin/env python3
"""
Primitive Class Ablation Evaluation

This script evaluates the impact of primitive type classes (e.g., [Identifier > Name])
on semantic retrieval performance by comparing:
1. Full Index: Original index with primitive class markers like "[Type] ColName:"
2. No-PC Index: Ablated index without primitive class markers

The script supports different retriever types:
- bm25: Only BM25 lexical search
- vector: Only FAISS vector search  
- hybrid: RRF fusion of BM25 + Vector (default)

Usage:
    # Basic comparison: with vs without primitive classes
    python -m evaluation.runners.primitive_class_ablation -d adventure_works

    # Compare with specific retriever type
    python -m evaluation.runners.primitive_class_ablation -d adventure_works --retriever bm25

    # Comprehensive evaluation: all retriever types × with/without PC
    python -m evaluation.runners.primitive_class_ablation -d adventure_works --comprehensive

    # Use specific HyDE mode for query
    python -m evaluation.runners.primitive_class_ablation -d adventure_works --hyde-mode table_desc
    
    # Show example cases where removal helps/hurts
    python -m evaluation.runners.primitive_class_ablation -d adventure_works --show-cases 5
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
import _path_setup  # noqa: F401

from loguru import logger
from core.paths import get_db_path


# ==================== Constants ====================

# Retriever types
RETRIEVER_BM25 = "bm25"
RETRIEVER_VECTOR = "vector"
RETRIEVER_HYBRID = "hybrid"
RETRIEVER_HYBRID_SUM = "hybrid-sum"

RETRIEVER_TYPES = [RETRIEVER_BM25, RETRIEVER_VECTOR, RETRIEVER_HYBRID, RETRIEVER_HYBRID_SUM]

# HyDE modes for query generation
HYDE_RAW = "raw"
HYDE_TABLE_DESC = "table_desc"
HYDE_COLUMN_DESC = "column_desc"
HYDE_COMBINED = "combined"

HYDE_MODES = [HYDE_RAW, HYDE_TABLE_DESC, HYDE_COLUMN_DESC, HYDE_COMBINED]

# Index variants
INDEX_WITH_PC = "with_pc"
INDEX_NO_PC = "no_pc"


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
        no_primitive_classes: If True, load the _no_pc version of analysis
    
    Returns:
        List of query analysis dictionaries
    """
    eval_dir = get_db_path() / "eval_results"
    pc_suffix = "_no_pc" if no_primitive_classes else ""
    
    # Build search patterns
    patterns = []
    
    if rag_type:
        patterns.extend([
            f"{dataset}_test_unified_analysis_all_{llm_suffix}_rag3_{rag_type}{pc_suffix}.json",
            f"{dataset}_test_unified_analysis_all_{llm_suffix}_rag5_{rag_type}{pc_suffix}.json",
        ])
    
    patterns.extend([
        f"{dataset}_test_unified_analysis_all_{llm_suffix}_rag3{pc_suffix}.json",
        f"{dataset}_test_unified_analysis_all_{llm_suffix}_rag5{pc_suffix}.json",
        f"{dataset}_unified_analysis_all_{llm_suffix}_rag3{pc_suffix}.json",
        f"{dataset}_unified_analysis_all_{llm_suffix}{pc_suffix}.json",
    ])
    
    for pattern in patterns:
        filepath = eval_dir / pattern
        if filepath.exists():
            logger.info(f"Loading unified analysis from: {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} query analyses")
            return data
    
    # Fallback: search for any matching file
    numbered_patterns = [
        f"{dataset}_test_unified_analysis_*_{llm_suffix}_rag*{pc_suffix}.json",
        f"{dataset}_unified_analysis_*_{llm_suffix}_rag*{pc_suffix}.json",
        f"{dataset}_unified_analysis_*_{llm_suffix}{pc_suffix}.json",
    ]
    
    for pattern in numbered_patterns:
        files = list(eval_dir.glob(pattern))
        # Filter by pc_suffix
        if no_primitive_classes:
            files = [f for f in files if "_no_pc" in f.name]
        else:
            files = [f for f in files if "_no_pc" not in f.name]
        if files:
            files.sort(key=lambda f: f.stat().st_size, reverse=True)
            filepath = files[0]
            logger.info(f"Loading unified analysis from: {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} query analyses")
            return data
    
    pc_label = "no_pc" if no_primitive_classes else "with_pc"
    raise FileNotFoundError(
        f"No unified analysis file ({pc_label}) found for dataset: {dataset}\n"
        f"Searched in: {eval_dir}\n"
        f"Run: python -m cli.retrieval -d {dataset} --analyze-queries -n -1 --use-rag"
        + (" --no-primitive-classes" if no_primitive_classes else "")
    )


def load_both_analysis_variants(
    dataset: str,
    llm_suffix: str = "local",
    rag_type: str = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load both variants of unified analysis (with and without primitive classes).
    
    Returns:
        Dictionary mapping variant names to analysis data lists
    """
    variants = {}
    
    # Load original analysis (with primitive classes)
    logger.info("Loading query analysis with primitive classes...")
    try:
        variants[INDEX_WITH_PC] = load_unified_analysis(
            dataset, llm_suffix, rag_type, no_primitive_classes=False
        )
    except FileNotFoundError as e:
        logger.warning(f"Original analysis not found: {e}")
        variants[INDEX_WITH_PC] = None
    
    # Load ablated analysis (without primitive classes)
    logger.info("Loading query analysis without primitive classes...")
    try:
        variants[INDEX_NO_PC] = load_unified_analysis(
            dataset, llm_suffix, rag_type, no_primitive_classes=True
        )
    except FileNotFoundError as e:
        logger.warning(f"Ablated analysis not found: {e}")
        variants[INDEX_NO_PC] = None
    
    return variants


def load_indexes_for_variant(
    dataset: str, 
    no_primitive_classes: bool = False
) -> Tuple[Any, List[Dict], Any, List[str], Any]:
    """
    Load indexes for a specific variant (with or without primitive classes).
    
    Args:
        dataset: Dataset name
        no_primitive_classes: If True, load the _no_pc variant
        
    Returns:
        Tuple of (faiss_index, metadata_list, bm25_retriever, table_ids, embedder)
    """
    from workflows.retrieval.unified_search import (
        load_unified_indexes, get_text_embedder
    )
    from workflows.retrieval.config import INDEX_KEY_TD_CD_CS
    
    # Determine index key
    base_key = INDEX_KEY_TD_CD_CS
    if no_primitive_classes:
        index_key = f"{base_key}_no_pc"
    else:
        index_key = base_key
    
    variant_label = "no_pc" if no_primitive_classes else "with_pc"
    logger.info(f"Loading indexes for variant '{variant_label}': {index_key}")
    
    start = time.time()
    
    try:
        faiss_index, metadata_list, bm25_retriever, table_ids = load_unified_indexes(
            dataset, index_key
        )
    except Exception as e:
        logger.warning(f"Failed to load indexes for {variant_label}: {e}")
        return None, None, None, None, None
    
    # Load embedder only if FAISS index exists
    embedder = None
    if faiss_index is not None:
        embedder = get_text_embedder()
    
    load_time = time.time() - start
    
    # Log status
    available = []
    if faiss_index is not None:
        available.append(f"FAISS ({faiss_index.ntotal} vectors)")
    if bm25_retriever is not None:
        available.append(f"BM25 ({len(table_ids)} docs)")
    
    if available:
        logger.info(f"  [{variant_label}] Loaded in {load_time:.2f}s: {', '.join(available)}")
    else:
        logger.warning(f"  [{variant_label}] No indexes loaded")
    
    return faiss_index, metadata_list, bm25_retriever, table_ids, embedder


def load_both_index_variants(dataset: str) -> Dict[str, Tuple]:
    """
    Load both index variants (with and without primitive classes).
    
    Returns:
        Dictionary mapping variant names to index tuples
    """
    variants = {}
    
    # Load original index (with primitive classes)
    logger.info("Loading indexes with primitive classes...")
    variants[INDEX_WITH_PC] = load_indexes_for_variant(dataset, no_primitive_classes=False)
    
    # Load ablated index (without primitive classes)
    logger.info("Loading indexes without primitive classes...")
    variants[INDEX_NO_PC] = load_indexes_for_variant(dataset, no_primitive_classes=True)
    
    # Check what's available
    have_with_pc = variants[INDEX_WITH_PC][0] is not None or variants[INDEX_WITH_PC][2] is not None
    have_no_pc = variants[INDEX_NO_PC][0] is not None or variants[INDEX_NO_PC][2] is not None
    
    if not have_with_pc:
        raise RuntimeError(
            f"Original index not found for dataset: {dataset}\n"
            f"Run: python -m cli.run_pipeline -d {dataset} -s retrieval_index"
        )
    
    if not have_no_pc:
        logger.warning(
            f"Ablated index (no_pc) not found for dataset: {dataset}\n"
            f"To generate: python -m cli.run_pipeline -d {dataset} -s retrieval_index --no-primitive-classes"
        )
    
    return variants


# ==================== Search Functions ====================

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
        query_text: Query string
        faiss_index: FAISS index or None
        metadata_list: Metadata for each document
        bm25_retriever: BM25 retriever or None
        table_ids: List of table IDs
        embedder: Text embedder or None
        top_k: Number of results to return
        retriever_type: One of "bm25", "vector", "hybrid", "hybrid-sum"
    
    Returns:
        List of (table_id, score) tuples
    """
    from workflows.retrieval.unified_search import (
        vector_search, bm25_search, reciprocal_rank_fusion, normalized_score_fusion
    )
    
    vec_results = []
    bm25_results = []
    
    # Vector search
    if retriever_type in [RETRIEVER_VECTOR, RETRIEVER_HYBRID, RETRIEVER_HYBRID_SUM]:
        if faiss_index is not None and embedder is not None:
            query_emb_list = embedder.compute_query_embeddings(query_text)
            query_emb = np.array(query_emb_list[0], dtype=np.float32)
            vec_results = vector_search(query_emb, faiss_index, metadata_list, top_k * 2)
        elif retriever_type == RETRIEVER_VECTOR:
            logger.warning("Vector search requested but FAISS index not available")
    
    # BM25 search
    if retriever_type in [RETRIEVER_BM25, RETRIEVER_HYBRID, RETRIEVER_HYBRID_SUM]:
        if bm25_retriever is not None:
            bm25_results = bm25_search(query_text, bm25_retriever, table_ids, metadata_list, top_k * 2)
        elif retriever_type == RETRIEVER_BM25:
            logger.warning("BM25 search requested but BM25 index not available")
    
    # Combine results
    if retriever_type == RETRIEVER_VECTOR:
        results = vec_results
    elif retriever_type == RETRIEVER_BM25:
        results = bm25_results
    else:  # hybrid or hybrid-sum
        if vec_results and bm25_results:
            if retriever_type == RETRIEVER_HYBRID_SUM:
                results = normalized_score_fusion(vec_results, bm25_results)
            else:
                results = reciprocal_rank_fusion(vec_results, bm25_results)
        elif vec_results:
            results = vec_results
        elif bm25_results:
            results = bm25_results
        else:
            results = []
    
    return [(tid, score) for tid, score, _ in results[:top_k]]


def find_gt_rank(results: List[Tuple[str, float]], gt_tables: List[str]) -> Optional[int]:
    """Find the best (minimum) rank among all ground truth tables (1-indexed)."""
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


def get_search_text(item: Dict[str, Any], hyde_mode: str) -> str:
    """Get the search text based on HyDE mode."""
    query = item['query']
    analysis = item.get('analysis', {})
    
    if hyde_mode == HYDE_RAW:
        return query
    elif hyde_mode == HYDE_TABLE_DESC:
        return analysis.get('hypothetical_table_description', query) or query
    elif hyde_mode == HYDE_COLUMN_DESC:
        return analysis.get('hypothetical_column_descriptions', query) or query
    elif hyde_mode == HYDE_COMBINED:
        table_desc = analysis.get('hypothetical_table_description', '')
        col_desc = analysis.get('hypothetical_column_descriptions', '')
        combined = f"{table_desc}\n{col_desc}".strip()
        return combined or query
    else:
        return query


# ==================== Metrics ====================

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
    print(f"   MRR:     {metrics['mrr']:.4f}")
    print(f"   Not Found: {metrics['not_found']} ({metrics['not_found']/total*100:.1f}%)")


def print_comparison_table(results: Dict[str, Dict[str, float]]):
    """Print a comparison table of with_pc vs no_pc results."""
    print("\n" + "="*80)
    print("📈 PRIMITIVE CLASS ABLATION COMPARISON")
    print("="*80)
    
    metrics_to_show = ['hit@1', 'hit@3', 'hit@5', 'hit@10', 'hit@50', 'mrr']
    
    # Header
    print(f"\n{'Metric':<12} {'With PC':>12} {'No PC':>12} {'Delta':>12} {'Change':>10}")
    print("-"*60)
    
    with_pc = results.get(INDEX_WITH_PC, {})
    no_pc = results.get(INDEX_NO_PC, {})
    
    for metric in metrics_to_show:
        v1 = with_pc.get(metric, 0)
        v2 = no_pc.get(metric, 0)
        delta = v2 - v1
        
        if metric in ['hit@1', 'hit@3', 'hit@5', 'hit@10', 'hit@50']:
            # Percentage format
            v1_str = f"{v1*100:.1f}%"
            v2_str = f"{v2*100:.1f}%"
            delta_str = f"{delta*100:+.1f}%"
        else:
            v1_str = f"{v1:.4f}"
            v2_str = f"{v2:.4f}"
            delta_str = f"{delta:+.4f}"
        
        change = "📈" if delta > 0.001 else ("📉" if delta < -0.001 else "−")
        print(f"{metric:<12} {v1_str:>12} {v2_str:>12} {delta_str:>12} {change:>10}")
    
    print("-"*60)


# ==================== Evaluation Functions ====================

def evaluate_variant(
    analysis_data: List[Dict[str, Any]],
    faiss_index,
    metadata_list: List[Dict],
    bm25_retriever,
    table_ids: List[str],
    embedder,
    retriever_type: str = RETRIEVER_HYBRID,
    hyde_mode: str = HYDE_RAW,
    top_k: int = 100,
    num_queries: int = -1,
    silent: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate retrieval performance for a single index variant.
    
    Args:
        analysis_data: Query analysis data
        faiss_index, metadata_list, etc.: Index components
        retriever_type: Type of retriever to use
        hyde_mode: HyDE mode for query generation
        top_k: Top-k retrieval
        num_queries: Number of queries to evaluate (-1 for all)
        silent: Suppress progress output
    
    Returns:
        Dictionary containing metrics and detailed results
    """
    if num_queries > 0:
        analysis_data = analysis_data[:num_queries]
    
    total = len(analysis_data)
    ranks: List[Optional[int]] = []
    results_detail = []
    
    for i, item in enumerate(analysis_data):
        gt_tables = item.get('gt_tables', [item['gt_table']] if item.get('gt_table') else [])
        
        # Skip if error
        if item.get('error') or not item.get('analysis'):
            ranks.append(None)
            results_detail.append({
                'query': item.get('query', ''),
                'rank': None,
                'error': True,
            })
            continue
        
        # Get search text
        search_text = get_search_text(item, hyde_mode)
        
        # Perform search
        search_results = semantic_search(
            search_text, faiss_index, metadata_list,
            bm25_retriever, table_ids, embedder, top_k,
            retriever_type=retriever_type
        )
        
        # Find GT rank
        rank = find_gt_rank(search_results, gt_tables)
        ranks.append(rank)
        results_detail.append({
            'query': item.get('query', ''),
            'rank': rank,
            'gt_tables': gt_tables,
        })
        
        # Progress
        if not silent and (i + 1) % 50 == 0:
            found = sum(1 for r in ranks if r is not None)
            pct = found / (i + 1) * 100
            print(f"   Processed {i+1}/{total}, hit@{top_k}: {pct:.1f}%")
    
    metrics = compute_metrics(ranks, total)
    metrics['total'] = total
    metrics['ranks'] = ranks
    metrics['details'] = results_detail
    
    return metrics


def compare_variants(
    analysis_data_variants: Dict[str, List[Dict[str, Any]]],
    index_variants: Dict[str, Tuple],
    retriever_type: str = RETRIEVER_HYBRID,
    hyde_mode: str = HYDE_RAW,
    top_k: int = 100,
    num_queries: int = -1,
    show_cases: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare retrieval performance between index variants.
    
    Args:
        analysis_data_variants: Dict mapping variant names to their query analysis data
        index_variants: Dictionary of index variants (tuples)
        retriever_type: Type of retriever to use
        hyde_mode: HyDE mode for query generation
        top_k: Top-k retrieval
        num_queries: Number of queries (-1 for all)
        show_cases: Number of example cases to show for each category
    
    Returns:
        Dictionary mapping variant names to their evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Comparing variants with retriever={retriever_type}, hyde_mode={hyde_mode}")
    print(f"{'='*60}")
    
    results = {}
    
    for variant_name, (faiss_idx, meta, bm25, tids, emb) in index_variants.items():
        if faiss_idx is None and bm25 is None:
            logger.warning(f"Skipping variant '{variant_name}': no indexes loaded")
            continue
        
        # Use corresponding analysis data for each variant
        analysis_data = analysis_data_variants.get(variant_name)
        if analysis_data is None:
            logger.warning(f"Skipping variant '{variant_name}': no analysis data")
            continue
        
        print(f"\n🔍 Evaluating variant: {variant_name}")
        print(f"   Using {len(analysis_data)} query analyses")
        
        metrics = evaluate_variant(
            analysis_data,
            faiss_idx, meta, bm25, tids, emb,
            retriever_type=retriever_type,
            hyde_mode=hyde_mode,
            top_k=top_k,
            num_queries=num_queries,
        )
        
        results[variant_name] = metrics
        print_metrics(metrics, variant_name, metrics['total'])
    
    # Print comparison
    if len(results) == 2:
        print_comparison_table(results)
        
        # Analyze case differences (use with_pc analysis for query text display)
        if show_cases > 0 and INDEX_WITH_PC in results and INDEX_NO_PC in results:
            with_pc_analysis = analysis_data_variants.get(INDEX_WITH_PC, [])
            analyze_case_differences(
                with_pc_analysis[:num_queries] if num_queries > 0 else with_pc_analysis,
                results[INDEX_WITH_PC],
                results[INDEX_NO_PC],
                show_cases=show_cases,
            )
    
    return results


def analyze_case_differences(
    analysis_data: List[Dict[str, Any]],
    with_pc_results: Dict[str, Any],
    no_pc_results: Dict[str, Any],
    show_cases: int = 5,
):
    """
    Analyze and display cases where removing primitive classes helps or hurts.
    """
    with_ranks = with_pc_results.get('ranks', [])
    no_ranks = no_pc_results.get('ranks', [])
    
    if len(with_ranks) != len(no_ranks):
        logger.warning("Rank lists have different lengths, skipping case analysis")
        return
    
    # Categorize cases
    removal_helps = []  # Cases where no_pc rank is better (lower)
    removal_hurts = []  # Cases where no_pc rank is worse (higher)
    both_found = []     # Both found, same rank
    only_with_pc = []   # Only found with primitive classes
    only_no_pc = []     # Only found without primitive classes
    neither_found = []  # Neither variant found GT
    
    for i, (w_rank, n_rank) in enumerate(zip(with_ranks, no_ranks)):
        item = analysis_data[i] if i < len(analysis_data) else {}
        case = {
            'idx': i,
            'query': item.get('query', ''),
            'with_rank': w_rank,
            'no_rank': n_rank,
        }
        
        if w_rank is None and n_rank is None:
            neither_found.append(case)
        elif w_rank is None:
            only_no_pc.append(case)
        elif n_rank is None:
            only_with_pc.append(case)
        elif n_rank < w_rank:
            case['improvement'] = w_rank - n_rank
            removal_helps.append(case)
        elif n_rank > w_rank:
            case['degradation'] = n_rank - w_rank
            removal_hurts.append(case)
        else:
            both_found.append(case)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 CASE-BY-CASE ANALYSIS")
    print("="*70)
    
    total = len(with_ranks)
    print(f"\nCategory Distribution (n={total}):")
    print(f"  Removal HELPS (better rank):    {len(removal_helps):>4} ({len(removal_helps)/total*100:.1f}%)")
    print(f"  Removal HURTS (worse rank):     {len(removal_hurts):>4} ({len(removal_hurts)/total*100:.1f}%)")
    print(f"  Same rank (both found):         {len(both_found):>4} ({len(both_found)/total*100:.1f}%)")
    print(f"  Only found WITH PC:             {len(only_with_pc):>4} ({len(only_with_pc)/total*100:.1f}%)")
    print(f"  Only found WITHOUT PC:          {len(only_no_pc):>4} ({len(only_no_pc)/total*100:.1f}%)")
    print(f"  Neither found:                  {len(neither_found):>4} ({len(neither_found)/total*100:.1f}%)")
    
    # Show example cases
    def show_category_examples(cases: List[Dict], title: str, max_cases: int):
        if not cases or max_cases <= 0:
            return
        print(f"\n--- {title} (showing {min(len(cases), max_cases)}/{len(cases)}) ---")
        for case in cases[:max_cases]:
            query = case['query'][:80] + "..." if len(case['query']) > 80 else case['query']
            print(f"  [{case['idx']+1}] {query}")
            print(f"      With PC: rank={case['with_rank']}, No PC: rank={case['no_rank']}")
    
    show_category_examples(removal_helps, "📈 Removal HELPS (no_pc better)", show_cases)
    show_category_examples(removal_hurts, "📉 Removal HURTS (with_pc better)", show_cases)
    show_category_examples(only_no_pc, "🆕 Only found WITHOUT PC", show_cases)
    show_category_examples(only_with_pc, "⚠️ Only found WITH PC", show_cases)


def comprehensive_evaluation(
    analysis_data_variants: Dict[str, List[Dict[str, Any]]],
    index_variants: Dict[str, Tuple],
    top_k: int = 100,
    num_queries: int = -1,
    output_file: str = None,
):
    """
    Run comprehensive evaluation across all retriever types.
    
    Args:
        analysis_data_variants: Dict mapping variant names to their query analysis data
        index_variants: Dictionary of index variants
        top_k: Top-k retrieval
        num_queries: Number of queries (-1 for all)
        output_file: Optional output JSON file path
    """
    print("\n" + "="*70)
    print("🔬 COMPREHENSIVE PRIMITIVE CLASS ABLATION EVALUATION")
    print("="*70)
    
    all_results = {}
    
    for retriever_type in [RETRIEVER_BM25, RETRIEVER_VECTOR, RETRIEVER_HYBRID]:
        print(f"\n{'='*50}")
        print(f"Retriever: {retriever_type}")
        print(f"{'='*50}")
        
        run_results = {}
        
        for variant_name, (faiss_idx, meta, bm25, tids, emb) in index_variants.items():
            if faiss_idx is None and bm25 is None:
                continue
            
            # Get corresponding analysis data
            analysis_data = analysis_data_variants.get(variant_name)
            if analysis_data is None:
                logger.info(f"Skipping {variant_name}: no analysis data")
                continue
            
            # Check if retriever type is supported for this variant
            if retriever_type == RETRIEVER_VECTOR and faiss_idx is None:
                logger.info(f"Skipping {variant_name} for vector search (no FAISS)")
                continue
            if retriever_type == RETRIEVER_BM25 and bm25 is None:
                logger.info(f"Skipping {variant_name} for BM25 search (no BM25)")
                continue
            
            print(f"\n  Variant: {variant_name}")
            
            metrics = evaluate_variant(
                analysis_data,
                faiss_idx, meta, bm25, tids, emb,
                retriever_type=retriever_type,
                hyde_mode=HYDE_RAW,
                top_k=top_k,
                num_queries=num_queries,
                silent=True,
            )
            
            # Store without rank lists for JSON serialization
            metrics_clean = {k: v for k, v in metrics.items() if k not in ['ranks', 'details']}
            run_results[variant_name] = metrics_clean
            
            # Print summary
            print(f"    Hit@3: {metrics['hit@3']*100:.1f}%, "
                  f"Hit@10: {metrics['hit@10']*100:.1f}%, "
                  f"MRR: {metrics['mrr']:.4f}")
        
        all_results[retriever_type] = run_results
        
        # Print comparison for this retriever
        if len(run_results) == 2:
            print_comparison_table(run_results)
    
    # Save results
    if output_file:
        output_path = get_db_path() / "eval_results" / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n📁 Results saved to: {output_path}")
    
    # Print overall summary
    print("\n" + "="*70)
    print("📋 COMPREHENSIVE RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n{'Retriever':<12} {'Variant':<10} {'Hit@3':>10} {'Hit@10':>10} {'MRR':>10}")
    print("-"*55)
    
    for retriever_type, run_results in all_results.items():
        for variant_name, metrics in run_results.items():
            hit3 = f"{metrics.get('hit@3', 0)*100:.1f}%"
            hit10 = f"{metrics.get('hit@10', 0)*100:.1f}%"
            mrr = f"{metrics.get('mrr', 0):.4f}"
            print(f"{retriever_type:<12} {variant_name:<10} {hit3:>10} {hit10:>10} {mrr:>10}")
    
    return all_results


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the impact of primitive class markers on retrieval performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[0]
    )
    
    parser.add_argument("-d", "--dataset", default="adventure_works", 
                        help="Dataset name")
    parser.add_argument("-n", "--num-queries", type=int, default=-1, 
                        help="Number of queries to analyze (-1 for all)")
    parser.add_argument("--llm", default="local", 
                        help="LLM suffix in filename (default: local)")
    parser.add_argument("--top-k", type=int, default=100, 
                        help="Top-k for retrieval")
    
    # Retriever type
    parser.add_argument("--retriever", type=str, default=RETRIEVER_HYBRID,
                        choices=RETRIEVER_TYPES,
                        help=f"Retriever type: {RETRIEVER_TYPES}")
    
    # HyDE mode
    parser.add_argument("--hyde-mode", type=str, default=HYDE_RAW,
                        choices=HYDE_MODES,
                        help=f"HyDE mode for query: {HYDE_MODES}")
    
    # RAG type (for loading analysis files)
    parser.add_argument("--rag-type", type=str, default="vector",
                        choices=["bm25", "vector", "hybrid"],
                        help="RAG type used when generating query analysis (default: vector)")
    
    # Analysis options
    parser.add_argument("--show-cases", type=int, default=0,
                        help="Show N example cases for each category")
    
    # Comprehensive mode
    parser.add_argument("--comprehensive", action="store_true",
                        help="Run comprehensive evaluation across all retriever types")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON filename for results")
    
    args = parser.parse_args()
    
    # Load data
    print(f"\n🔧 Loading data for dataset: {args.dataset}")
    
    # Load both versions of query analysis
    analysis_variants = load_both_analysis_variants(args.dataset, args.llm, args.rag_type)
    
    # Load both index variants
    index_variants = load_both_index_variants(args.dataset)
    
    # Check if ablated versions exist
    have_no_pc_analysis = analysis_variants.get(INDEX_NO_PC) is not None
    have_no_pc_index = index_variants[INDEX_NO_PC][0] is not None or index_variants[INDEX_NO_PC][2] is not None
    
    if not have_no_pc_index:
        print("\n⚠️  Ablated index (no_pc) not found!")
        print(f"   To generate: python -m cli.run_pipeline -d {args.dataset} --step retrieval_index --no-primitive-classes")
    
    if not have_no_pc_analysis:
        print("\n⚠️  Ablated query analysis (no_pc) not found!")
        print(f"   To generate: python -m cli.retrieval -d {args.dataset} --analyze-queries -n -1 --use-rag --rag-type {args.rag_type} --no-primitive-classes --split test --llm {args.llm}")
    
    if not have_no_pc_index or not have_no_pc_analysis:
        print("\n   Running evaluation with original version only...")
        
        # Evaluate just the original
        result = evaluate_variant(
            analysis_variants.get(INDEX_WITH_PC, []),
            *index_variants[INDEX_WITH_PC],
            retriever_type=args.retriever,
            hyde_mode=args.hyde_mode,
            top_k=args.top_k,
            num_queries=args.num_queries,
        )
        print_metrics(result, "Original (with primitive classes)", result['total'])
        return
    
    # Comprehensive mode
    if args.comprehensive:
        output_file = args.output or f"{args.dataset}_primitive_class_ablation.json"
        comprehensive_evaluation(
            analysis_variants, index_variants,
            top_k=args.top_k,
            num_queries=args.num_queries,
            output_file=output_file,
        )
        return
    
    # Standard comparison
    results = compare_variants(
        analysis_variants, index_variants,
        retriever_type=args.retriever,
        hyde_mode=args.hyde_mode,
        top_k=args.top_k,
        num_queries=args.num_queries,
        show_cases=args.show_cases,
    )
    
    # Save results if requested
    if args.output:
        output_path = get_db_path() / "eval_results" / args.output
        # Clean results for JSON
        clean_results = {}
        for k, v in results.items():
            clean_results[k] = {
                key: val for key, val in v.items() 
                if key not in ['ranks', 'details']
            }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2)
        print(f"\n📁 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
