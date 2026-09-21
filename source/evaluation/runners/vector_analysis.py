#!/usr/bin/env python3
"""
Analyze Vector Retrieval Contribution from Primitive Class Tokens

This script analyzes how primitive class tokens (e.g., [Identifier > Name]) 
contribute to vector retrieval performance.

Key features:
1. Adhoc vector index: Dynamically compute embeddings without using pre-built indexes
2. Ablation experiments: Remove primitive class tokens from query/doc and measure recall
3. Contribution analysis: Quantify primitive class contribution via similarity delta

Usage:
    # Basic ablation analysis
    python -m evaluation.runners.vector_analysis -d fetaqa --mode ablation

    # Full analysis with case-by-case details
    python -m evaluation.runners.vector_analysis -d adventure_works --mode full --show-cases 10

    # Compare raw vs HyDE
    python -m evaluation.runners.vector_analysis -d fetaqa --compare-raw
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
import _path_setup  # noqa: F401

from loguru import logger
from core.paths import get_db_path, get_data_root
from workflows.retrieval.unified_search import load_unified_indexes, get_text_embedder
from store.embedding.embedding_registry import EmbeddingFunctionRegistry
from evaluation.runners.bm25_analysis import (
    extract_primitive_class_tokens,
    keep_only_primitive_class_tokens,
)


# ==================== Neutral Placeholder Replacement ====================

def replace_primitive_class_with_placeholder(text: str, placeholder: str = "[Type]") -> str:
    """
    Replace primitive class markers with a neutral placeholder.
    
    For vector embeddings, simply removing [Identifier > Name] is not fair
    because the model can still infer semantics from remaining context.
    
    Instead, we replace with a neutral placeholder to:
    1. Preserve structure (model knows there's a type marker)
    2. Remove specific semantic information
    
    Args:
        text: Text containing [Parent > Child] markers
        placeholder: Neutral placeholder to use (default: "[Type]")
        
    Returns:
        Text with markers replaced by placeholder
    """
    import re
    # Replace [Any > Content > Here] with placeholder
    return re.sub(r'\[[^\]]+\]', placeholder, text)


# ==================== Data Classes ====================

@dataclass
class VectorSearchResult:
    """Result of a single vector search."""
    doc_id: int
    table_id: str
    score: float  # cosine similarity
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== Adhoc Vector Index ====================

class AdhocVectorIndex:
    """
    Dynamically compute embeddings and perform vector search.
    
    This avoids using pre-built indexes, allowing us to modify
    query/document text for ablation experiments.
    """
    
    def __init__(self, embedder=None):
        """
        Initialize with an embedder.
        
        Args:
            embedder: Embedding function (default: BGE-M3)
        """
        if embedder is None:
            embedder = get_text_embedder()
        self.embedder = embedder
        
        # Cache for document embeddings
        self._doc_embeddings: Dict[int, np.ndarray] = {}
        self._doc_texts: Dict[int, str] = {}
        self._table_ids: List[str] = []
    
    def set_corpus(self, documents: List[str], table_ids: List[str]):
        """
        Set the document corpus.
        
        Args:
            documents: List of document texts
            table_ids: Corresponding table IDs
        """
        self._doc_texts = {i: doc for i, doc in enumerate(documents)}
        self._table_ids = table_ids
        self._doc_embeddings.clear()
        
        logger.info(f"Corpus set: {len(documents)} documents")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for a single text."""
        embeddings = self.embedder.compute_query_embeddings(text)
        return np.array(embeddings[0], dtype=np.float32)
    
    def _get_doc_embedding(self, doc_id: int, use_cache: bool = True) -> np.ndarray:
        """Get document embedding (with caching)."""
        if use_cache and doc_id in self._doc_embeddings:
            return self._doc_embeddings[doc_id]
        
        text = self._doc_texts.get(doc_id, "")
        emb = self._get_embedding(text)
        
        if use_cache:
            self._doc_embeddings[doc_id] = emb
        
        return emb
    
    def compute_all_doc_embeddings(self, batch_size: int = 32):
        """Pre-compute all document embeddings."""
        logger.info(f"Computing embeddings for {len(self._doc_texts)} documents...")
        
        doc_ids = sorted(self._doc_texts.keys())
        for i in range(0, len(doc_ids), batch_size):
            batch_ids = doc_ids[i:i+batch_size]
            batch_texts = [self._doc_texts[did] for did in batch_ids]
            
            # Batch embed
            embeddings = self.embedder.compute_source_embeddings(batch_texts)
            
            for j, did in enumerate(batch_ids):
                self._doc_embeddings[did] = np.array(embeddings[j], dtype=np.float32)
            
            if (i + batch_size) % 100 == 0:
                logger.debug(f"  Computed {min(i+batch_size, len(doc_ids))}/{len(doc_ids)}")
        
        logger.info(f"✓ Computed {len(self._doc_embeddings)} document embeddings")
    
    def search(
        self,
        query_text: str,
        top_k: int = 100,
        doc_text_override: Optional[Dict[int, str]] = None,
    ) -> List[VectorSearchResult]:
        """
        Perform vector search.
        
        Args:
            query_text: Query text to embed
            top_k: Number of results to return
            doc_text_override: Override document texts (for ablation)
            
        Returns:
            List of VectorSearchResult sorted by score (descending)
        """
        # Get query embedding
        query_emb = self._get_embedding(query_text)
        
        # Compute similarities
        scores = []
        for doc_id in range(len(self._doc_texts)):
            if doc_text_override and doc_id in doc_text_override:
                # Recompute embedding for modified doc
                doc_emb = self._get_embedding(doc_text_override[doc_id])
            else:
                doc_emb = self._get_doc_embedding(doc_id)
            
            # Cosine similarity
            sim = np.dot(query_emb, doc_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8
            )
            scores.append((doc_id, float(sim)))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for doc_id, score in scores[:top_k]:
            table_id = self._table_ids[doc_id] if doc_id < len(self._table_ids) else f"doc_{doc_id}"
            results.append(VectorSearchResult(
                doc_id=doc_id,
                table_id=table_id,
                score=score,
            ))
        
        return results
    
    def get_similarity(self, query_text: str, doc_id: int) -> float:
        """Get cosine similarity between query and a specific document."""
        query_emb = self._get_embedding(query_text)
        doc_emb = self._get_doc_embedding(doc_id)
        
        return float(np.dot(query_emb, doc_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8
        ))


# ==================== Analysis Functions ====================

def load_unified_analysis(
    dataset: str,
    rag_type: str = None,
) -> List[Dict[str, Any]]:
    """Load unified analysis results from JSON file."""
    eval_dir = get_db_path() / "eval_results"
    
    patterns = []
    if rag_type:
        patterns.extend([
            f"{dataset}_test_unified_analysis_all_local_rag3_{rag_type}.json",
            f"{dataset}_test_unified_analysis_all_local_rag5_{rag_type}.json",
        ])
    
    patterns.extend([
        f"{dataset}_test_unified_analysis_all_local_rag3_vector.json",
        f"{dataset}_test_unified_analysis_all_local_rag3_hybrid.json",
        f"{dataset}_test_unified_analysis_all_local_rag3.json",
        f"{dataset}_unified_analysis_all_local_rag3.json",
        f"{dataset}_unified_analysis_all_local.json",
    ])
    
    for pattern in patterns:
        filepath = eval_dir / pattern
        if filepath.exists():
            logger.info(f"Loading: {filepath}")
            with open(filepath, 'r') as f:
                return json.load(f)
    
    raise FileNotFoundError(f"No analysis file found for dataset={dataset}")


def build_corpus_from_metadata(metadata_list: List[Dict]) -> List[str]:
    """Build document corpus from metadata (same format as BM25 index)."""
    corpus = []
    for m in metadata_list:
        # Build text in same format as BM25 corpus
        parts = []
        
        table_desc = m.get('table_description', '')
        if table_desc:
            parts.append(f"== Table Description ==\n{table_desc}")
        
        col_descs = m.get('column_descriptions', '')
        if col_descs:
            parts.append(f"\n\n== Column Information ==\n{col_descs}")
        
        col_stats = m.get('column_stats', '')
        if col_stats:
            parts.append(f"\n\n== Statistics ==\n{col_stats}")
        
        corpus.append("\n".join(parts) if parts else "")
    
    return corpus


def get_gt_rank(results: List[VectorSearchResult], gt_tables: List[str]) -> Tuple[Optional[int], Optional[float]]:
    """Find the best rank and score among ground truth tables."""
    for i, r in enumerate(results):
        if r.table_id in gt_tables:
            return i + 1, r.score  # 1-indexed rank
    return None, None


def compute_metrics(ranks: List[Optional[int]], total: int) -> Dict[str, float]:
    """Compute retrieval metrics from rank list."""
    def hit_at_k(k: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= k) / total if total > 0 else 0.0
    
    def mrr() -> float:
        rrs = [1.0 / r for r in ranks if r is not None]
        return sum(rrs) / total if rrs else 0.0
    
    return {
        'hit@1': hit_at_k(1),
        'hit@3': hit_at_k(3),
        'hit@5': hit_at_k(5),
        'hit@10': hit_at_k(10),
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
    print(f"   MRR:     {metrics['mrr']:.4f}")


# ==================== Ablation Experiments ====================

def run_query_ablation(
    analysis_data: List[Dict[str, Any]],
    index: AdhocVectorIndex,
    table_ids: List[str],
    num_queries: int = -1,
    hyde_mode: str = "combined",
    top_k: int = 100,
) -> Dict[str, Any]:
    """
    Run query-side ablation: remove primitive class tokens from query.
    
    Compares:
    - Original: HyDE query with primitive class tokens
    - Ablated: HyDE query without primitive class tokens
    """
    logger.info("Running query-side ablation...")
    
    if num_queries > 0:
        analysis_data = analysis_data[:num_queries]
    
    # Build table_id to doc_id mapping
    table_to_doc = {tid: i for i, tid in enumerate(table_ids)}
    
    results = {
        'original_ranks': [],
        'ablated_ranks': [],
        'original_scores': [],
        'ablated_scores': [],
        'cases': [],
    }
    
    for i, item in enumerate(analysis_data):
        query = item['query']
        gt_tables = item.get('gt_tables', [item['gt_table']] if item.get('gt_table') else [])
        analysis = item.get('analysis', {})
        
        # Raw mode doesn't require analysis
        if hyde_mode != "raw" and (not analysis or item.get('error')):
            results['original_ranks'].append(None)
            results['ablated_ranks'].append(None)
            continue
        
        # Get query text
        if hyde_mode == "raw":
            query_text = query
        elif hyde_mode == "column_desc":
            query_text = analysis.get('hypothetical_column_descriptions', '')
        elif hyde_mode == "table_desc":
            query_text = analysis.get('hypothetical_table_description', '')
        elif hyde_mode == "combined":
            table_desc = analysis.get('hypothetical_table_description', '')
            col_desc = analysis.get('hypothetical_column_descriptions', '')
            query_text = f"{table_desc}\n{col_desc}"
        else:
            query_text = query
        
        if not query_text.strip():
            results['original_ranks'].append(None)
            results['ablated_ranks'].append(None)
            continue
        
        # Original search
        original_results = index.search(query_text, top_k=top_k)
        original_rank, original_score = get_gt_rank(original_results, gt_tables)
        
        # Ablated search (replace primitive class with neutral placeholder)
        ablated_query = replace_primitive_class_with_placeholder(query_text)
        ablated_results = index.search(ablated_query, top_k=top_k)
        ablated_rank, ablated_score = get_gt_rank(ablated_results, gt_tables)
        
        results['original_ranks'].append(original_rank)
        results['ablated_ranks'].append(ablated_rank)
        results['original_scores'].append(original_score)
        results['ablated_scores'].append(ablated_score)
        
        # Store case details
        if original_rank is not None or ablated_rank is not None:
            results['cases'].append({
                'query_idx': i,
                'query': query[:100],
                'gt_tables': gt_tables,
                'original_rank': original_rank,
                'ablated_rank': ablated_rank,
                'original_score': original_score,
                'ablated_score': ablated_score,
                'score_delta': (ablated_score - original_score) if original_score and ablated_score else None,
            })
        
        if (i + 1) % 20 == 0:
            logger.info(f"  Processed {i + 1}/{len(analysis_data)}")
    
    return results


def run_doc_ablation(
    analysis_data: List[Dict[str, Any]],
    index: AdhocVectorIndex,
    table_ids: List[str],
    doc_corpus: List[str],
    num_queries: int = -1,
    hyde_mode: str = "combined",
    top_k: int = 100,
) -> Dict[str, Any]:
    """
    Run document-side ablation: remove primitive class tokens from documents.
    
    This requires recomputing document embeddings with modified texts.
    """
    logger.info("Running document-side ablation...")
    
    # Pre-compute ablated document texts (use placeholder for fair semantic comparison)
    ablated_docs = {
        i: replace_primitive_class_with_placeholder(doc) 
        for i, doc in enumerate(doc_corpus)
    }
    
    if num_queries > 0:
        analysis_data = analysis_data[:num_queries]
    
    results = {
        'original_ranks': [],
        'ablated_ranks': [],
        'original_scores': [],
        'ablated_scores': [],
        'cases': [],
    }
    
    for i, item in enumerate(analysis_data):
        query = item['query']
        gt_tables = item.get('gt_tables', [item['gt_table']] if item.get('gt_table') else [])
        analysis = item.get('analysis', {})
        
        if hyde_mode != "raw" and (not analysis or item.get('error')):
            results['original_ranks'].append(None)
            results['ablated_ranks'].append(None)
            continue
        
        # Get query text
        if hyde_mode == "raw":
            query_text = query
        elif hyde_mode == "column_desc":
            query_text = analysis.get('hypothetical_column_descriptions', '')
        elif hyde_mode == "table_desc":
            query_text = analysis.get('hypothetical_table_description', '')
        elif hyde_mode == "combined":
            table_desc = analysis.get('hypothetical_table_description', '')
            col_desc = analysis.get('hypothetical_column_descriptions', '')
            query_text = f"{table_desc}\n{col_desc}"
        else:
            query_text = query
        
        if not query_text.strip():
            results['original_ranks'].append(None)
            results['ablated_ranks'].append(None)
            continue
        
        # Original search (with original docs)
        original_results = index.search(query_text, top_k=top_k)
        original_rank, original_score = get_gt_rank(original_results, gt_tables)
        
        # Ablated search (with modified docs)
        ablated_results = index.search(query_text, top_k=top_k, doc_text_override=ablated_docs)
        ablated_rank, ablated_score = get_gt_rank(ablated_results, gt_tables)
        
        results['original_ranks'].append(original_rank)
        results['ablated_ranks'].append(ablated_rank)
        results['original_scores'].append(original_score)
        results['ablated_scores'].append(ablated_score)
        
        if original_rank is not None or ablated_rank is not None:
            results['cases'].append({
                'query_idx': i,
                'query': query[:100],
                'gt_tables': gt_tables,
                'original_rank': original_rank,
                'ablated_rank': ablated_rank,
                'original_score': original_score,
                'ablated_score': ablated_score,
                'score_delta': (ablated_score - original_score) if original_score and ablated_score else None,
            })
        
        if (i + 1) % 20 == 0:
            logger.info(f"  Processed {i + 1}/{len(analysis_data)}")
    
    return results


def run_both_ablation(
    analysis_data: List[Dict[str, Any]],
    index: AdhocVectorIndex,
    table_ids: List[str],
    doc_corpus: List[str],
    num_queries: int = -1,
    hyde_mode: str = "combined",
    top_k: int = 100,
) -> Dict[str, Any]:
    """
    Run both-side ablation: remove primitive class tokens from both query and docs.
    """
    logger.info("Running both-side ablation...")
    
    # Pre-compute ablated document texts (use placeholder for fair semantic comparison)
    ablated_docs = {
        i: replace_primitive_class_with_placeholder(doc) 
        for i, doc in enumerate(doc_corpus)
    }
    
    if num_queries > 0:
        analysis_data = analysis_data[:num_queries]
    
    results = {
        'original_ranks': [],
        'ablated_ranks': [],
        'original_scores': [],
        'ablated_scores': [],
        'cases': [],
    }
    
    for i, item in enumerate(analysis_data):
        query = item['query']
        gt_tables = item.get('gt_tables', [item['gt_table']] if item.get('gt_table') else [])
        analysis = item.get('analysis', {})
        
        if hyde_mode != "raw" and (not analysis or item.get('error')):
            results['original_ranks'].append(None)
            results['ablated_ranks'].append(None)
            continue
        
        # Get query text
        if hyde_mode == "raw":
            query_text = query
        elif hyde_mode == "column_desc":
            query_text = analysis.get('hypothetical_column_descriptions', '')
        elif hyde_mode == "table_desc":
            query_text = analysis.get('hypothetical_table_description', '')
        elif hyde_mode == "combined":
            table_desc = analysis.get('hypothetical_table_description', '')
            col_desc = analysis.get('hypothetical_column_descriptions', '')
            query_text = f"{table_desc}\n{col_desc}"
        else:
            query_text = query
        
        if not query_text.strip():
            results['original_ranks'].append(None)
            results['ablated_ranks'].append(None)
            continue
        
        # Original search
        original_results = index.search(query_text, top_k=top_k)
        original_rank, original_score = get_gt_rank(original_results, gt_tables)
        
        # Ablated search (both query and docs modified with neutral placeholder)
        ablated_query = replace_primitive_class_with_placeholder(query_text)
        ablated_results = index.search(ablated_query, top_k=top_k, doc_text_override=ablated_docs)
        ablated_rank, ablated_score = get_gt_rank(ablated_results, gt_tables)
        
        results['original_ranks'].append(original_rank)
        results['ablated_ranks'].append(ablated_rank)
        results['original_scores'].append(original_score)
        results['ablated_scores'].append(ablated_score)
        
        if original_rank is not None or ablated_rank is not None:
            results['cases'].append({
                'query_idx': i,
                'query': query[:100],
                'gt_tables': gt_tables,
                'original_rank': original_rank,
                'ablated_rank': ablated_rank,
                'original_score': original_score,
                'ablated_score': ablated_score,
            })
        
        if (i + 1) % 20 == 0:
            logger.info(f"  Processed {i + 1}/{len(analysis_data)}")
    
    return results


# ==================== Contribution Analysis ====================

def analyze_primitive_contribution(
    analysis_data: List[Dict[str, Any]],
    index: AdhocVectorIndex,
    table_ids: List[str],
    num_queries: int = -1,
    hyde_mode: str = "combined",
    show_cases: int = 5,
) -> Dict[str, Any]:
    """
    Analyze contribution of primitive class tokens to vector similarity.
    
    For each query-GT pair, compare:
    - Original similarity: query vs GT doc
    - Ablated similarity: query (no primitive) vs GT doc
    - Delta = Original - Ablated = contribution from primitive tokens
    """
    logger.info("Analyzing primitive class contribution to vector similarity...")
    
    if num_queries > 0:
        analysis_data = analysis_data[:num_queries]
    
    table_to_doc = {tid: i for i, tid in enumerate(table_ids)}
    
    results = {
        'similarity_deltas': [],
        'original_similarities': [],
        'ablated_similarities': [],
        'cases': [],
    }
    
    for i, item in enumerate(analysis_data):
        query = item['query']
        gt_tables = item.get('gt_tables', [item['gt_table']] if item.get('gt_table') else [])
        analysis = item.get('analysis', {})
        
        if hyde_mode != "raw" and (not analysis or item.get('error')):
            continue
        
        # Get query text
        if hyde_mode == "raw":
            query_text = query
        elif hyde_mode == "column_desc":
            query_text = analysis.get('hypothetical_column_descriptions', '')
        elif hyde_mode == "table_desc":
            query_text = analysis.get('hypothetical_table_description', '')
        elif hyde_mode == "combined":
            table_desc = analysis.get('hypothetical_table_description', '')
            col_desc = analysis.get('hypothetical_column_descriptions', '')
            query_text = f"{table_desc}\n{col_desc}"
        else:
            query_text = query
        
        if not query_text.strip():
            continue
        
        # Get GT doc
        gt_doc_ids = [table_to_doc[gt] for gt in gt_tables if gt in table_to_doc]
        if not gt_doc_ids:
            continue
        
        gt_doc_id = gt_doc_ids[0]  # Use first GT
        
        # Compute similarities
        original_sim = index.get_similarity(query_text, gt_doc_id)
        
        ablated_query = replace_primitive_class_with_placeholder(query_text)
        ablated_sim = index.get_similarity(ablated_query, gt_doc_id)
        
        delta = original_sim - ablated_sim
        
        results['similarity_deltas'].append(delta)
        results['original_similarities'].append(original_sim)
        results['ablated_similarities'].append(ablated_sim)
        
        # Extract primitive class tokens for case analysis
        primitive_tokens = extract_primitive_class_tokens(query_text)
        
        results['cases'].append({
            'query_idx': i,
            'query': query[:100],
            'gt_table': gt_tables[0] if gt_tables else '',
            'original_sim': original_sim,
            'ablated_sim': ablated_sim,
            'delta': delta,
            'primitive_tokens': list(primitive_tokens)[:10],
        })
        
        if (i + 1) % 20 == 0:
            logger.info(f"  Processed {i + 1}/{len(analysis_data)}")
    
    return results


def print_ablation_results(results: Dict[str, Any], label: str):
    """Print ablation experiment results."""
    original_ranks = results['original_ranks']
    ablated_ranks = results['ablated_ranks']
    
    valid = [i for i in range(len(original_ranks)) 
             if original_ranks[i] is not None or ablated_ranks[i] is not None]
    total = len(valid)
    
    if total == 0:
        print(f"\n⚠️  {label}: No valid results")
        return
    
    # Compute metrics
    original_metrics = compute_metrics(original_ranks, len(original_ranks))
    ablated_metrics = compute_metrics(ablated_ranks, len(ablated_ranks))
    
    print(f"\n{'='*60}")
    print(f"📊 {label}")
    print(f"{'='*60}")
    
    print_metrics(original_metrics, "Original (with primitive class)", total)
    print_metrics(ablated_metrics, "Ablated (without primitive class)", total)
    
    # Compare Hit@K
    print("\n📈 Hit@K Comparison:")
    for k in [1, 3, 5, 10]:
        orig = original_metrics[f'hit@{k}'] * 100
        ablated = ablated_metrics[f'hit@{k}'] * 100
        delta = ablated - orig
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"   Hit@{k}: {orig:.1f}% → {ablated:.1f}% ({arrow}{abs(delta):.1f}%)")
    
    # Count improvements/regressions
    improvements = 0
    regressions = 0
    for orig, ablated in zip(original_ranks, ablated_ranks):
        if orig is not None and ablated is not None:
            if ablated < orig:
                improvements += 1
            elif ablated > orig:
                regressions += 1
    
    print(f"\n   Improvements (rank ↓ after ablation): {improvements}")
    print(f"   Regressions (rank ↑ after ablation):  {regressions}")


def print_contribution_results(results: Dict[str, Any], show_cases: int = 5):
    """Print contribution analysis results."""
    deltas = results['similarity_deltas']
    
    if not deltas:
        print("\n⚠️  No contribution data")
        return
    
    print(f"\n{'='*60}")
    print("📊 Primitive Class Contribution to Vector Similarity")
    print(f"{'='*60}")
    
    deltas_arr = np.array(deltas)
    print(f"\n   Mean delta:   {np.mean(deltas_arr):.4f}")
    print(f"   Median delta: {np.median(deltas_arr):.4f}")
    print(f"   Std:          {np.std(deltas_arr):.4f}")
    print(f"   Min delta:    {np.min(deltas_arr):.4f}")
    print(f"   Max delta:    {np.max(deltas_arr):.4f}")
    
    # Interpretation
    positive_count = sum(1 for d in deltas if d > 0)
    negative_count = sum(1 for d in deltas if d < 0)
    print(f"\n   Positive delta (primitive helps): {positive_count} ({positive_count/len(deltas)*100:.1f}%)")
    print(f"   Negative delta (primitive hurts): {negative_count} ({negative_count/len(deltas)*100:.1f}%)")
    
    # Show case examples
    if show_cases > 0 and results['cases']:
        print(f"\n📝 Case Examples (sorted by delta):")
        
        # Sort by delta
        sorted_cases = sorted(results['cases'], key=lambda x: x['delta'], reverse=True)
        
        for j, case in enumerate(sorted_cases[:show_cases]):
            print(f"\n   ─── Case {j+1} (Query {case['query_idx']+1}) ───")
            print(f"   Query: {case['query']}...")
            print(f"   GT: {case['gt_table']}")
            print(f"   Original similarity: {case['original_sim']:.4f}")
            print(f"   Ablated similarity:  {case['ablated_sim']:.4f}")
            print(f"   Delta (contribution): {case['delta']:.4f}")
            if case['primitive_tokens']:
                print(f"   Primitive tokens: {case['primitive_tokens']}")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze Vector Retrieval Contribution from Primitive Class Tokens"
    )
    parser.add_argument("-d", "--dataset", default="fetaqa", help="Dataset name")
    parser.add_argument("-n", "--num-queries", type=int, default=-1,
                        help="Number of queries to analyze (-1 for all)")
    parser.add_argument("--mode", type=str, default="query-ablation",
                        choices=["query-ablation", "doc-ablation", "both-ablation", "contribution", "full"],
                        help="Analysis mode")
    parser.add_argument("--hyde-mode", type=str, default="combined",
                        choices=["raw", "table_desc", "column_desc", "combined"],
                        help="HyDE text mode for analysis")
    parser.add_argument("--show-cases", type=int, default=5,
                        help="Number of case analyses to show")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Top-k for retrieval")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON filename")
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data for dataset: {args.dataset}")
    analysis_data = load_unified_analysis(args.dataset)
    
    # Load indexes to get metadata and table_ids
    logger.info("Loading indexes for metadata...")
    faiss_index, metadata_list, bm25_retriever, table_ids = load_unified_indexes(args.dataset)
    
    if not metadata_list:
        logger.error("No metadata found!")
        return
    
    # Build corpus from metadata
    logger.info("Building document corpus...")
    doc_corpus = build_corpus_from_metadata(metadata_list)
    
    # Initialize adhoc vector index
    logger.info("Initializing adhoc vector index...")
    index = AdhocVectorIndex()
    index.set_corpus(doc_corpus, table_ids)
    
    # Pre-compute document embeddings
    index.compute_all_doc_embeddings()
    
    # Run analysis
    results = {}
    
    if args.mode in ["query-ablation", "full"]:
        print("\n" + "=" * 80)
        print("🔬 QUERY-SIDE ABLATION")
        print("=" * 80)
        
        query_results = run_query_ablation(
            analysis_data, index, table_ids,
            num_queries=args.num_queries,
            hyde_mode=args.hyde_mode,
            top_k=args.top_k,
        )
        print_ablation_results(query_results, "Query Ablation (remove primitive from query)")
        results['query_ablation'] = query_results
    
    if args.mode in ["doc-ablation", "full"]:
        print("\n" + "=" * 80)
        print("🔬 DOCUMENT-SIDE ABLATION")
        print("=" * 80)
        
        doc_results = run_doc_ablation(
            analysis_data, index, table_ids, doc_corpus,
            num_queries=args.num_queries,
            hyde_mode=args.hyde_mode,
            top_k=args.top_k,
        )
        print_ablation_results(doc_results, "Doc Ablation (remove primitive from docs)")
        results['doc_ablation'] = doc_results
    
    if args.mode in ["both-ablation", "full"]:
        print("\n" + "=" * 80)
        print("🔬 BOTH-SIDE ABLATION")
        print("=" * 80)
        
        both_results = run_both_ablation(
            analysis_data, index, table_ids, doc_corpus,
            num_queries=args.num_queries,
            hyde_mode=args.hyde_mode,
            top_k=args.top_k,
        )
        print_ablation_results(both_results, "Both Ablation (remove primitive from both)")
        results['both_ablation'] = both_results
    
    if args.mode in ["contribution", "full"]:
        print("\n" + "=" * 80)
        print("🔬 PRIMITIVE CLASS CONTRIBUTION ANALYSIS")
        print("=" * 80)
        
        contrib_results = analyze_primitive_contribution(
            analysis_data, index, table_ids,
            num_queries=args.num_queries,
            hyde_mode=args.hyde_mode,
            show_cases=args.show_cases,
        )
        print_contribution_results(contrib_results, show_cases=args.show_cases)
        results['contribution'] = contrib_results
    
    # Save results
    if args.output:
        output_path = get_db_path() / "eval_results" / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Clean for JSON
        def clean_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [clean_for_json(v) for v in obj]
            return obj
        
        with open(output_path, 'w') as f:
            json.dump(clean_for_json(results), f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
