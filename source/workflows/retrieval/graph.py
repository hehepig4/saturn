"""
Retrieval Subgraph - Dual-Path Parallel Recall Architecture

Architecture:
┌──────────────────────────────────────────────────────────────┐
│                    extract_constraints                        │
│                           │                                   │
│         ┌─────────────────┴─────────────────┐                │
│         │                                   │                │
│         ▼                                   ▼                │
│   SEMANTIC PATH                      STRUCTURAL PATH         │
│   (Vector + BM25)                    ┌──────────────┐        │
│         │                            │  tbox_filter │        │
│         │                            └──────┬───────┘        │
│         │                                   │                │
│         │                            ┌──────▼───────┐        │
│         │                            │  abox_verify │        │
│         │                            └──────┬───────┘        │
│         │                                   │                │
│         └─────────────────┬─────────────────┘                │
│                           ▼                                   │
│                      rrf_fusion                               │
│                           │                                   │
│                          END                                  │
└──────────────────────────────────────────────────────────────┘

Design:
- Semantic: Vector + BM25 hybrid on table summaries (configurable fields)
- Structural: TBox filter → ABox verify (serial chain)
- Fusion: RRF combines both paths
"""

from typing import Dict, Any, List
import time
from loguru import logger
from langgraph.graph import StateGraph, END

from workflows.retrieval.state import RetrievalState, RetrievalResult, ColumnTypeMatch
from workflows.retrieval.nodes.extract_constraints import (
    extract_constraints_node,
    convert_to_path_constraints,
)
from workflows.retrieval.nodes.semantic_search import semantic_search_node
from workflows.retrieval.matcher import get_scorer_v3


# ==================== Structural Path ======================================

def structural_path_node(state: RetrievalState) -> Dict[str, Any]:
    """
    Execute structural path using ScorerV3 (TBox + ABox).
    
    ScorerV3 implements the v3 scoring formula with capped ABox and IDF ratio.
    """
    if not state.enable_structural:
        logger.info("Structural path disabled")
        return {'structural_results': []}
    
    start_time = time.time()
    
    logger.info("\n" + "="*60)
    logger.info("[STRUCTURAL PATH] ScorerV3 (TBox + ABox)")
    logger.info("="*60)
    
    # Get constraints from state
    tbox_constraints = getattr(state, 'tbox_constraints', None)
    abox_constraints = getattr(state, 'abox_constraints', None)
    
    if not tbox_constraints or not tbox_constraints.primitive_class_constraints:
        logger.info("  No TBox constraints, skipping structural path")
        return {
            'structural_results': [],
            'structural_time': time.time() - start_time,
        }
    
    # Convert to PathConstraint format
    logger.info("\n[Step 1/2] Converting constraints to PathConstraint format")
    path_constraints = convert_to_path_constraints(tbox_constraints, abox_constraints)
    
    if len(path_constraints) == 0:
        logger.info("  No path constraints after conversion")
        return {
            'structural_results': [],
            'structural_time': time.time() - start_time,
        }
    
    # Log constraints
    logger.info(f"  PathConstraints ({len(path_constraints)}):")
    for i, pc in enumerate(path_constraints):
        if pc.has_value:
            logger.info(f"    {i+1}. {pc.class_name} + value='{pc.abox.value}'")
        else:
            logger.info(f"    {i+1}. {pc.class_name} (TBox-only)")
    
    # Use ScorerV3 for retrieval
    logger.info("\n[Step 2/2] ScorerV3 retrieval")
    scorer = get_scorer_v3(state.dataset_name)
    
    retrieval_results = scorer.retrieve(
        constraints=path_constraints,
        score_threshold=0.0,  # Return all candidates with score > 0
    )
    
    logger.info(f"  ScorerV3 returned {len(retrieval_results)} candidates")
    
    if not retrieval_results:
        return {
            'structural_results': [],
            'structural_time': time.time() - start_time,
        }
    
    # Convert to RetrievalResult format
    # ScorerV3 returns List[Dict] with keys: required_class, score, deepest_matched/best_ancestor, has_abox, weight
    structural_results = []
    for table_id, score, details in retrieval_results:
        # Extract match info from details (which are dicts from ScorerV3)
        tbox_matches = []
        abox_matches = []
        has_fallback_only = True
        
        for detail in details:
            # ScorerV3 uses dict format
            matched_class = detail.get('deepest_matched') or detail.get('best_ancestor')
            constraint_score = detail.get('score', 0.0)
            has_abox = detail.get('has_abox', False)
            
            if matched_class:
                tbox_matches.append(ColumnTypeMatch(
                    column_type=matched_class,
                    confidence=constraint_score,  # Use constraint score as confidence
                    is_fallback=(matched_class == 'Column'),
                ))
                if matched_class != 'Column':
                    has_fallback_only = False
            
            if has_abox and detail.get('best_ancestor'):
                abox_matches.append(f"value exists in {detail.get('best_ancestor')}")
        
        result = RetrievalResult(
            table_id=table_id,
            structural_score=score,
            source='structural',
            tbox_matches=tbox_matches,
            abox_matches=abox_matches,
            has_fallback_only=has_fallback_only,
        )
        structural_results.append(result)
    
    elapsed = time.time() - start_time
    logger.info(f"\n  ✓ Structural path: {len(structural_results)} results in {elapsed:.2f}s")
    
    return {
        'structural_results': structural_results,
        'structural_time': elapsed,
    }


# ==================== RRF Fusion ====================

def rrf_fusion_node(state: RetrievalState) -> Dict[str, Any]:
    """
    Fuse semantic and structural results using Reciprocal Rank Fusion.
    
    RRF: score(d) = sum(weight / (k + rank(d)))
    """
    start_time = time.time()
    
    semantic_results = getattr(state, 'semantic_results', []) or []
    structural_results = getattr(state, 'structural_results', []) or []
    
    semantic_weight = getattr(state, 'semantic_weight', 0.5)
    structural_weight = getattr(state, 'structural_weight', 0.5)
    top_k = state.top_k
    k = 60  # RRF constant
    
    logger.info("\n" + "="*60)
    logger.info("[RRF FUSION]")
    logger.info("="*60)
    logger.info(f"  Semantic: {len(semantic_results)} results")
    logger.info(f"  Structural: {len(structural_results)} results")
    logger.info(f"  Weights: sem={semantic_weight}, struct={structural_weight}")
    
    # Edge cases: single path - copy score to final_score
    if not semantic_results and not structural_results:
        return {'final_results': [], 'fusion_time': 0.0}
    
    if not semantic_results:
        # Structural only: use structural_score as final_score
        for r in structural_results:
            r.final_score = r.structural_score
        sorted_struct = sorted(structural_results, key=lambda x: x.final_score, reverse=True)
        return {'final_results': sorted_struct[:top_k], 'fusion_time': 0.0}
    
    if not structural_results:
        # Semantic only: use semantic_score as final_score
        for r in semantic_results:
            r.final_score = r.semantic_score
        sorted_sem = sorted(semantic_results, key=lambda x: x.final_score, reverse=True)
        return {'final_results': sorted_sem[:top_k], 'fusion_time': 0.0}
    
    # Build rank maps
    sem_ranked = sorted(semantic_results, key=lambda r: r.semantic_score, reverse=True)
    sem_ranks = {r.table_id: i + 1 for i, r in enumerate(sem_ranked)}
    sem_map = {r.table_id: r for r in semantic_results}
    
    struct_ranked = sorted(structural_results, key=lambda r: r.structural_score, reverse=True)
    struct_ranks = {r.table_id: i + 1 for i, r in enumerate(struct_ranked)}
    struct_map = {r.table_id: r for r in structural_results}
    
    # Compute RRF
    all_tables = set(sem_ranks.keys()) | set(struct_ranks.keys())
    
    fused = []
    for tid in all_tables:
        score = 0.0
        if tid in sem_ranks:
            score += semantic_weight / (k + sem_ranks[tid])
        if tid in struct_ranks:
            score += structural_weight / (k + struct_ranks[tid])
        
        # Determine source
        if tid in sem_ranks and tid in struct_ranks:
            source = 'hybrid'
        elif tid in sem_ranks:
            source = 'semantic'
        else:
            source = 'structural'
        
        # Get base result
        sem_r = sem_map.get(tid)
        struct_r = struct_map.get(tid)
        base = sem_r or struct_r
        
        merged = RetrievalResult(
            table_id=tid,
            final_score=score,
            semantic_score=sem_r.semantic_score if sem_r else 0.0,
            structural_score=struct_r.structural_score if struct_r else 0.0,
            source=source,
            tbox_matches=struct_r.tbox_matches if struct_r else (sem_r.tbox_matches if sem_r else []),
            abox_matches=struct_r.abox_matches if struct_r else [],
            has_fallback_only=base.has_fallback_only if base else False,
            metadata=base.metadata if base else {},
        )
        fused.append(merged)
    
    fused.sort(key=lambda r: r.final_score, reverse=True)
    
    # Log stats
    hybrid = sum(1 for r in fused if r.source == 'hybrid')
    sem_only = sum(1 for r in fused if r.source == 'semantic')
    struct_only = sum(1 for r in fused if r.source == 'structural')
    
    logger.info(f"\n  Fusion: {hybrid} hybrid, {sem_only} sem-only, {struct_only} struct-only")
    logger.info(f"\n  Top 5:")
    for i, r in enumerate(fused[:5], 1):
        logger.info(f"    {i}. {r.table_id[:50]}... [{r.source}] score={r.final_score:.4f}")
    
    elapsed = time.time() - start_time
    
    return {
        'final_results': fused[:top_k],
        'fusion_time': elapsed,
    }


# ==================== Parallel Recall ====================

def parallel_recall_node(state: RetrievalState) -> Dict[str, Any]:
    """
    Execute semantic and structural paths (logically parallel).
    """
    logger.info("\n" + "="*60)
    logger.info("PARALLEL RECALL")
    logger.info("="*60)
    
    start_time = time.time()
    results = {}
    
    # Semantic path
    if state.enable_semantic:
        logger.info("\n[Path 1] SEMANTIC (Vector + BM25)")
        sem_updates = semantic_search_node(state)
        results.update(sem_updates)
    else:
        results['semantic_results'] = []
    
    # Structural path
    if state.enable_structural:
        logger.info("\n[Path 2] STRUCTURAL (PathMatcher)")
        struct_updates = structural_path_node(state)
        results.update(struct_updates)
    else:
        results['structural_results'] = []
    
    logger.info(f"\n✓ Parallel recall done in {time.time() - start_time:.2f}s")
    
    return results


# ==================== Graph Builder ====================

def build_retrieval_graph():
    """
    Build retrieval graph:
        extract_constraints → parallel_recall → rrf_fusion → END
    """
    graph = StateGraph(RetrievalState)
    
    graph.add_node("extract_constraints", extract_constraints_node)
    graph.add_node("parallel_recall", parallel_recall_node)
    graph.add_node("rrf_fusion", rrf_fusion_node)
    
    graph.set_entry_point("extract_constraints")
    graph.add_edge("extract_constraints", "parallel_recall")
    graph.add_edge("parallel_recall", "rrf_fusion")
    graph.add_edge("rrf_fusion", END)
    
    return graph.compile()


def create_retrieval_graph():
    """Alias for build_retrieval_graph."""
    return build_retrieval_graph()


# ==================== Entry Point ====================

def run_retrieval(
    query: str,
    dataset_name: str = "fetaqa",
    top_k: int = 20,
    semantic_top_k: int = 100,
    enable_semantic: bool = True,
    enable_structural: bool = True,
    enable_bm25: bool = True,
    search_fields: List[str] = None,
    semantic_weight: float = 0.5,
    structural_weight: float = 0.5,
) -> RetrievalState:
    """
    Run dual-path retrieval pipeline.
    
    Args:
        query: Natural language query
        dataset_name: Dataset to search
        top_k: Final results count
        semantic_top_k: Candidates per path
        enable_semantic: Enable semantic path
        enable_structural: Enable structural path
        enable_bm25: Enable BM25 in semantic path
        search_fields: Fields for semantic search
        semantic_weight: Weight in RRF
        structural_weight: Weight in RRF
    """
    logger.info("="*60)
    logger.info(f"RETRIEVAL: {query[:50]}...")
    logger.info("="*60)
    
    start_time = time.time()
    
    if search_fields is None:
        search_fields = ['table_description', 'column_descriptions', 'column_stats', 'relationship_view']
    
    initial_state = RetrievalState(
        query=query,
        dataset_name=dataset_name,
        top_k=top_k,
        semantic_top_k=semantic_top_k,
        enable_semantic=enable_semantic,
        enable_structural=enable_structural,
        enable_bm25=enable_bm25,
        search_fields=search_fields,
        semantic_weight=semantic_weight,
        structural_weight=structural_weight,
    )
    
    graph = build_retrieval_graph()
    
    try:
        result_dict = graph.invoke(initial_state)
        
        # LangGraph returns dict, convert back to RetrievalState
        if isinstance(result_dict, dict):
            # Merge dict updates into initial_state
            for key, value in result_dict.items():
                if hasattr(initial_state, key):
                    setattr(initial_state, key, value)
            final_state = initial_state
        else:
            final_state = result_dict
        
        final_state.success = True
        final_state.total_time = time.time() - start_time
        
        logger.info("="*60)
        logger.info(f"DONE: {len(final_state.final_results)} results in {final_state.total_time:.2f}s")
        logger.info("="*60)
        
        return final_state
        
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        
        initial_state.success = False
        initial_state.error = str(e)
        return initial_state
