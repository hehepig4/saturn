#!/usr/bin/env python3
"""
UPO Pipeline Runner

Complete 4-stage pipeline for User Preference Ontology based Table Discovery.

Stages:
    1. Primitive TBox (Layer 1): Federated CQ-driven ontology generation
        - Auto-compute agent tree structure from capacity constraints
        - Balanced k-means clustering of queries
        - Generate Scoping CQs (SCQ) and Validating CQs (VCQ) per query
        - Global-Local agent collaboration for TBox synthesis
        - Validate EL++ consistency, store to LanceDB
    
    2. Column Summary: Code-based column statistics
        - Analyze each column with LLM-generated code
        - Compute statistics (type, range, cardinality, etc.)
        - Store column_summaries to LanceDB
    
    3. Layer 2 TBox (Table Annotation): Defined class generation
        - Use Primitive Classes + Column Summaries
        - LLM annotation: description, role, match_level
        - Create Column and Table Defined Classes
        - Store column_mappings and table_defined_classes
    
    4. Table Summarization: Multi-view serialization
        - Generate role-based views (key, temporal, measure, attribute)
        - Serialize for LM embedding (L1/L2/L3 levels)
        - Export UPO and PNEUMA-compatible formats

Usage:
    # Run Federated Primitive TBox (recommended)
    python cli/run_pipeline.py --total-queries 100 --agent-cq-capacity 30
    
    # Run with specific cluster count (0 = auto-compute)
    python cli/run_pipeline.py --n-clusters 5 --total-queries 200
    
    # Run full pipeline from Federated Stage 1 to export
    python cli/run_pipeline.py --step all --total-queries 100 --max-tables 50
    
    # Run specific stages
    python cli/run_pipeline.py --step column_summary
    python cli/run_pipeline.py --step layer2_annotation
    python cli/run_pipeline.py --step summarize
    
    # Run from Layer 2 onwards (skip primitive_tbox)
    python cli/run_pipeline.py --step layer2_all --max-tables 100
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add source directory to path so _path_setup can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
import _path_setup  # noqa: F401 - Sets up source path

from loguru import logger
from core.paths import lake_data_path, ensure_project_cwd

# Ensure working directory is project root to avoid writing to source/data/
ensure_project_cwd()


# ============== Configuration ==============

DEFAULT_DATASET = "fetaqa"
DEFAULT_OUTPUT_BASE = lake_data_path("upo_summaries")



# ============== Stage 1 (Federated): Global-Local Agent Collaboration ==============

def run_federated_primitive_tbox(
    dataset_name: str = DEFAULT_DATASET,
    n_clusters: int = 3,
    n_iterations: int = 2,
    agent_cq_capacity: int = 30,
    global_agent_span: int = 10,
    proposal_capacity: int = 30,
    target_classes: int = 0,
    scq_per_query: int = 1,
    vcq_per_query: int = 1,
    llm_purpose: str = "gemini",
    max_tables: Optional[int] = None,
    max_queries: Optional[int] = None,
    dp_only: bool = False,
    cq_max_concurrent: int = 16,
    dp_max_concurrent: int = 5,
    rag_type: str = "hybrid",
) -> Dict[str, Any]:
    """
    Stage 1 (Federated): Global-Local agent collaboration for TBox generation.
    
    This workflow implements the Federated Primitive TBox architecture:
    1. CQ Generation: Generate Competency Questions per-query (async parallel)
    2. Global Init: Global Agent initializes TBox from backbone CQs
    3. Local Proposals: Each Local Agent proposes changes (parallel)
    4. Global Synthesis: Global Agent synthesizes all proposals
    5. Iteration: Repeat steps 3-4 for n_iterations
    6. Local Voting: Each Local Agent votes on class usefulness
    7. Export: Aggregate scores and export final TBox
    
    DP-Only Mode:
    - When dp_only=True, skip steps 1-6 and only regenerate DataProperties
    - Loads existing TBox classes from LanceDB (latest snapshot)
    
    Args:
        dataset_name: Dataset to use (fetaqa, wikitableqa, etc.)
        n_clusters: Number of Local Agents (0 = auto-compute from capacity)
        n_iterations: Number of proposal-synthesis iterations
        agent_cq_capacity: K parameter - max CQs per agent
        global_agent_span: B parameter - max branching factor
        proposal_capacity: P parameter - max class proposals for global synthesis
        target_classes: Target number of classes (0 = no specific target)
        scq_per_query: Number of Scoping CQs per query (default: 1)
        vcq_per_query: Number of Validating CQs per query (default: 1)
        llm_purpose: LLM purpose key ("gemini", "default")
        max_tables: Max tables to load (None = all, not used in Stage 1)
        max_queries: Max queries to load (None = all)
        dp_only: If True, skip iterations and only regenerate DataProperties (load from LanceDB)
        cq_max_concurrent: Max concurrent CQ generation LLM calls
        dp_max_concurrent: Max concurrent DP generation LLM calls
        rag_type: Retrieval type for hard negative sampling ('bm25', 'vector', 'hybrid')
    
    Returns:
        Result dict with success status, statistics, paths
    """
    from workflows.conceptualization import run_federated_tbox
    
    logger.info("=" * 70)
    if dp_only:
        logger.info("Stage 1 (DP-Only): DataProperty Regeneration")
        logger.info("=" * 70)
        logger.info(f"  Dataset: {dataset_name}")
        logger.info(f"  TBox Source: LanceDB (latest snapshot)")
        logger.info(f"  LLM Purpose: {llm_purpose}")
        logger.info(f"  DP Max Concurrent: {dp_max_concurrent}")
    else:
        logger.info("Stage 1 (Federated): Global-Local Agent Collaboration")
        logger.info("=" * 70)
        logger.info(f"  Dataset: {dataset_name}")
        logger.info(f"  Local Agents (clusters): {n_clusters} (0 = auto)")
        logger.info(f"  Iterations: {n_iterations}")
        logger.info(f"  Target Classes: {target_classes} (0 = no target)")
        logger.info(f"  CQs per Query: SCQ={scq_per_query}, VCQ={vcq_per_query}")
        logger.info(f"  Agent CQ Capacity (K): {agent_cq_capacity}")
        logger.info(f"  Global Agent Span (B): {global_agent_span}")
        logger.info(f"  Proposal Capacity (P): {proposal_capacity}")
        logger.info(f"  LLM Purpose: {llm_purpose}")
        logger.info(f"  CQ Max Concurrent: {cq_max_concurrent}")
        logger.info(f"  DP Max Concurrent: {dp_max_concurrent}")
    
    start_time = time.time()
    
    # Run federated workflow
    result_state = run_federated_tbox(
        dataset_name=dataset_name,
        table_store_name=f"{dataset_name}_tables_entries",
        query_store_name=f"{dataset_name}_train_queries",
        n_clusters=n_clusters,
        n_iterations=n_iterations,
        target_classes=target_classes,
        scq_per_query=scq_per_query,
        vcq_per_query=vcq_per_query,
        agent_cq_capacity=agent_cq_capacity,
        global_agent_span=global_agent_span,
        proposal_capacity=proposal_capacity,
        llm_purpose=llm_purpose,
        max_tables=max_tables,  # Not used in Stage 1
        max_queries=max_queries,
        dp_only=dp_only,
        cq_max_concurrent=cq_max_concurrent,
        dp_max_concurrent=dp_max_concurrent,
        rag_type=rag_type,
    )
    
    elapsed = time.time() - start_time
    
    # Check success
    if not result_state.export_success:
        logger.error(f"  ✗ Federated Stage 1 failed: {result_state.export_error}")
        return {
            'success': False,
            'error': result_state.export_error or "Unknown error",
            'elapsed': elapsed,
        }
    
    # Extract statistics
    final_tbox = result_state.current_tbox or {}
    stats = {
        'n_clusters': len(result_state.cluster_assignments or {}),
        'n_cqs': len(result_state.competency_questions or []),
        'n_classes': len(final_tbox.get('classes', [])),
        'n_data_properties': len(final_tbox.get('data_properties', [])),
        'n_iterations_run': len(result_state.synthesis_log or []),
    }
    
    logger.info(f"  ✓ Federated Stage 1 completed in {elapsed:.2f}s")
    logger.info(f"  ✓ Clusters: {stats['n_clusters']}")
    logger.info(f"  ✓ CQs Generated: {stats['n_cqs']}")
    logger.info(f"  ✓ Classes: {stats['n_classes']}")
    logger.info(f"  ✓ DataProperties: {stats['n_data_properties']}")
    logger.info(f"  ✓ Iterations: {stats['n_iterations_run']}")
    logger.info(f"  ✓ OWL Path: {result_state.owl_path or 'N/A'}")
    
    return {
        'success': True,
        'statistics': stats,
        'owl_path': result_state.owl_path,
        'report_path': result_state.report_path,
        'elapsed': elapsed,
    }


# ============== Stage 2: Column Summary ==============

def run_column_summary(
    dataset_name: str = DEFAULT_DATASET,
    max_tables: Optional[int] = None,
    fresh_start: bool = False,
    budget_multiplier: float = 1.0,
    budget_cap: int = 1000,
    disable_virtual_columns: bool = False,
    batch_size: int = 300,
    table_max_workers: int = 128,
    analyze_max_workers: int = 32,
    sh_max_workers: int = 8,
    llm_purpose: str = "default",
    tbox_iteration: int = -1,
    disable_transform_reuse: bool = False,
    stats_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stage 2: Column Summary Agent using workflows.population subgraph.
    
    Args:
        dataset_name: Dataset name (fetaqa, wikitableqa, etc.)
        max_tables: Maximum tables to process (None for all)
        fresh_start: If True, clear Transform Contracts before starting
        budget_multiplier: Successive Halving budget = k * num_rows * n_candidates
        budget_cap: Upper limit on Successive Halving budget (prevents excessive CPU on large columns)
        disable_virtual_columns: If True, skip virtual column extraction (default: False)
        batch_size: Number of tables per batch
        table_max_workers: Max parallel workers for expand/classify nodes (no nested parallelism)
        analyze_max_workers: Max parallel workers for analyze_columns (has nested SH parallelism)
        sh_max_workers: Max parallel workers for Successive Halving evaluation
        tbox_iteration: TBox iteration to use (-1 = latest)
        disable_transform_reuse: If True, skip transform repository lookup (ablation)
        stats_dir: If given, save LLM stats JSON to this directory (default: logs/llm_calls/)
    
    Returns:
        Result dict with statistics
    """
    from workflows.population.graph import build_column_summary_graph
    from workflows.population.state import ColumnSummaryState
    
    logger.info("=" * 70)
    logger.info("Stage 2: Column Summary Agent")
    logger.info("=" * 70)
    logger.info(f"  Dataset: {dataset_name}")
    logger.info(f"  Max Tables: {max_tables or 'ALL'}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Table Max Workers (expand/classify): {table_max_workers}")
    logger.info(f"  Analyze Max Workers (analyze_columns): {analyze_max_workers}")
    logger.info(f"  Successive Halving Max Workers: {sh_max_workers}")
    logger.info(f"  Budget Multiplier: {budget_multiplier}")
    logger.info(f"  Budget Cap: {budget_cap}")
    logger.info(f"  Virtual Columns: {'disabled' if disable_virtual_columns else 'enabled'}")
    logger.info(f"  LLM Purpose: {llm_purpose}")
    logger.info(f"  TBox Iteration: {tbox_iteration}")
    if disable_transform_reuse:
        logger.info(f"  Transform Reuse: DISABLED (ablation mode)")
    
    start_time = time.time()
    
    # Create initial state
    initial_state = ColumnSummaryState(
        dataset_name=dataset_name,
        table_store_name=f"{dataset_name}_tables_entries",
        batch_size=batch_size,
        max_tables=max_tables,
        fresh_start=fresh_start,
        budget_multiplier=budget_multiplier,
        enable_virtual_columns=not disable_virtual_columns,
        table_max_workers=table_max_workers,
        analyze_max_workers=analyze_max_workers,
        sh_max_workers=sh_max_workers,
        budget_cap=budget_cap,
        llm_purpose=llm_purpose,
        tbox_iteration=tbox_iteration,
        disable_transform_reuse=disable_transform_reuse,
        # context_fields uses default from state: ['document_title', 'section_title']
    )
    
    # Build and run the graph with higher recursion limit
    graph = build_column_summary_graph()
    result = graph.invoke(initial_state, config={"recursion_limit": 5000})
    elapsed = time.time() - start_time
    
    # Extract statistics
    total_tables = result.get('total_tables_processed', 0)
    total_columns = result.get('total_columns_analyzed', 0)
    total_llm_calls = result.get('total_llm_calls', 0)
    code_reuse_count = result.get('code_reuse_count', 0)
    llm_stats_timeline = result.get('llm_stats_timeline', [])
    
    logger.info(f"  ✓ Stage 2 completed in {elapsed:.2f}s")
    logger.info(f"  ✓ Tables Processed: {total_tables}")
    logger.info(f"  ✓ Columns Analyzed: {total_columns}")
    if llm_stats_timeline:
        logger.info(f"  ✓ LLM Stats Timeline: {len(llm_stats_timeline)} snapshots")
    
    # Save LLM stats timeline to a dedicated file with date suffix
    if llm_stats_timeline:
        from pathlib import Path
        if stats_dir:
            out_dir = Path(stats_dir)
        else:
            out_dir = Path(__file__).parent.parent.parent / 'logs' / 'llm_calls'
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        reuse_tag = "no_reuse" if disable_transform_reuse else "with_reuse"
        stats_filename = f"{dataset_name}_llm_stats_{reuse_tag}_{ts}.json"
        stats_path = out_dir / stats_filename
        stats_data = {
            "dataset_name": dataset_name,
            "disable_transform_reuse": disable_transform_reuse,
            "elapsed_seconds": elapsed,
            "total_tables": total_tables,
            "total_columns": total_columns,
            "code_reuse_count": code_reuse_count,
            "timeline": llm_stats_timeline,
        }
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        logger.info(f"  ✓ LLM Stats saved to {stats_path}")
    
    return {
        'success': True,
        'total_tables': total_tables,
        'total_columns': total_columns,
        'total_llm_calls': total_llm_calls,
        'code_reuse_count': code_reuse_count,
        'llm_stats_timeline': llm_stats_timeline,
        'elapsed': elapsed,
    }


# ============== Stage 3: Layer 2 Annotation ==============

def run_layer2_annotation(
    dataset_name: str = DEFAULT_DATASET,
    max_tables: Optional[int] = None,
    export_owl: bool = True,
    batch_size: int = 300,
    table_max_workers: int = 128,
) -> Dict[str, Any]:
    """
    Stage 3: Layer 2 Annotation using workflows.indexing.annotation subgraph.
    
    Args:
        dataset_name: Dataset name
        max_tables: Maximum tables to annotate (None for all)
        export_owl: Export OWL files
        batch_size: Batch size for processing tables
        table_max_workers: Maximum concurrent workers for table processing
    
    Returns:
        Result dict with Layer 2 statistics
    """
    from workflows.indexing.annotation import (
        create_table_discovery_layer2_graph,
        TableDiscoveryLayer2State,
    )
    
    logger.info("=" * 70)
    logger.info("Stage 3: Layer 2 TBox - Table Annotation")
    logger.info("=" * 70)
    logger.info(f"  Dataset: {dataset_name}")
    logger.info(f"  Max Tables: {max_tables or 'ALL'}")
    logger.info(f"  Export OWL: {export_owl}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Table Max Workers: {table_max_workers}")
    
    start_time = time.time()
    
    # Use the subgraph directly - it handles parallelization internally
    initial_state = TableDiscoveryLayer2State(
        dataset_name=dataset_name,
        max_tables=max_tables,
        export_owl=export_owl,
        batch_size=batch_size,
        llm_max_workers=table_max_workers,
    )
    
    graph = create_table_discovery_layer2_graph()
    result = graph.invoke(
        initial_state,
        {"recursion_limit": 3000}
    )
    elapsed = time.time() - start_time
    
    if not result.get('success'):
        logger.error(f"  ✗ Stage 3 failed: {result.get('error')}")
        return {'success': False, 'error': result.get('error')}
    
    column_classes = result.get('column_defined_classes', [])
    table_classes = result.get('table_defined_classes', [])
    
    logger.info(f"  ✓ Stage 3 completed in {elapsed:.2f}s")
    logger.info(f"  ✓ Column Defined Classes: {len(column_classes)}")
    logger.info(f"  ✓ Table Defined Classes: {len(table_classes)}")
    
    return {
        'success': True,
        'num_column_classes': len(column_classes),
        'num_table_classes': len(table_classes),
        'ontology_id': result.get('layer2_ontology_id'),
        'elapsed': elapsed,
    }


# ============== Stage 4: Table Summarization ==============

def run_summarization(
    dataset_name: str = DEFAULT_DATASET,
    output_format: str = "both",
    output_base: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stage 4: Table Summarization using workflows.indexing subgraph.
    
    Args:
        dataset_name: Dataset name
        output_format: "upo", "pneuma", or "both"
        output_base: Base output directory (default: data/lake/upo_summaries)
    
    Returns:
        Result dict with summary statistics and paths
    """
    from workflows.indexing import (
        create_table_summarization_graph,
        TableSummarizationState,
    )
    
    logger.info("=" * 70)
    logger.info("Stage 4: Table Summarization")
    logger.info("=" * 70)
    logger.info(f"  Dataset: {dataset_name}")
    logger.info(f"  Output Format: {output_format}")
    
    if output_base is None:
        output_base = str(DEFAULT_OUTPUT_BASE)
    
    start_time = time.time()
    
    # Create initial state
    initial_state = TableSummarizationState(
        dataset_name=dataset_name,
        output_base=output_base,
        output_format=output_format,
    )
    
    # Create and run the graph
    graph = create_table_summarization_graph()
    result = graph.invoke(initial_state)
    elapsed = time.time() - start_time
    
    if not result.get('success'):
        logger.error(f"  ✗ Stage 4 failed: {result.get('error')}")
        return {'success': False, 'error': result.get('error')}
    
    output = result.get('output')
    num_summaries = output.num_summaries if output else 0
    
    logger.info(f"  ✓ Stage 4 completed in {elapsed:.2f}s")
    logger.info(f"  ✓ Table Summaries: {num_summaries}")
    
    return {
        'success': True,
        'num_summaries': num_summaries,
        'output_dir': output_base,
        'elapsed': elapsed,
    }


# ============== Stage 5: Retrieval Index Generation ==============

def run_retrieval_index(
    dataset_name: str = DEFAULT_DATASET,
    index_key: Optional[str] = None,
    enable_faiss: bool = True,
    enable_bm25: bool = True,
    embedding_batch_size: int = 32,
    remove_primitive_classes: bool = False,
    output_base_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Stage 5: Generate Retrieval Indexes (FAISS Vector + BM25).
    
    Requires: Stage 4 completed ({dataset}_table_summaries_retrieval exists).
    
    Args:
        dataset_name: Dataset name
        index_key: Index configuration key ('td', 'td_cd', 'td_cd_cs')
        enable_faiss: Generate FAISS vector index
        enable_bm25: Generate BM25 keyword index
        embedding_batch_size: Batch size for embedding generation
        remove_primitive_classes: If True, remove [Type] markers from documents
            (ablation mode). Indexes will be saved to {index_key}_no_pc/
        output_base_path: Base path for output. If provided, indexes are saved to
            {output_base_path}/indexes/{index_key}/
    
    Returns:
        Result dict with index statistics and paths
    """
    from workflows.retrieval.index_generator import generate_retrieval_index
    
    logger.info("=" * 70)
    logger.info("Stage 5: Retrieval Index Generation")
    logger.info("=" * 70)
    
    logger.info(f"  Dataset: {dataset_name}")
    logger.info(f"  Index Key: {index_key or 'td_cd_cs (default)'}")
    logger.info(f"  Enable FAISS: {enable_faiss}")
    logger.info(f"  Enable BM25: {enable_bm25}")
    logger.info(f"  Embedding Batch Size: {embedding_batch_size}")
    if remove_primitive_classes:
        logger.info(f"  Ablation Mode: Removing primitive class markers")
    if output_base_path:
        logger.info(f"  Output Base Path: {output_base_path}")
    
    start_time = time.time()
    
    try:
        result = generate_retrieval_index(
            dataset_name=dataset_name,
            index_key=index_key,
            enable_faiss=enable_faiss,
            enable_bm25=enable_bm25,
            batch_size=embedding_batch_size,
            remove_primitive_classes=remove_primitive_classes,
            output_base_path=output_base_path,
        )
        elapsed = time.time() - start_time
        
        if not result.get('success'):
            logger.error(f"  ✗ Stage 5 failed: {result.get('error')}")
            return {'success': False, 'error': result.get('error'), 'elapsed': elapsed}
        
        logger.info(f"  ✓ Stage 5 completed in {elapsed:.2f}s")
        logger.info(f"  ✓ Tables indexed: {result.get('total_tables', 0)}")
        if enable_faiss:
            logger.info(f"  ✓ FAISS index: {result.get('faiss_path', 'N/A')}")
        if enable_bm25:
            logger.info(f"  ✓ BM25 index: {result.get('bm25_path', 'N/A')}")
        
        result['elapsed'] = elapsed
        return result
        
    except Exception as e:
        import traceback
        elapsed = time.time() - start_time
        logger.error(f"  ✗ Stage 5 failed with exception: {e}")
        return {
            'success': False, 
            'error': str(e), 
            'traceback': traceback.format_exc(),
            'elapsed': elapsed,
        }


# ============== Main Pipeline ==============

def run_pipeline(
    dataset_name: str = DEFAULT_DATASET,
    step: str = "all",
    max_tables: Optional[int] = None,
    output_format: str = "both",
    fresh_start: bool = False,
    budget_multiplier: float = 1.0,
    budget_cap: int = 1000,
    disable_virtual_columns: bool = False,
    # Federated options (primary mode)
    n_clusters: int = 0,  # Auto-compute
    total_queries: int = 100,
    n_iterations: int = 1,
    scq_per_query: int = 1,
    vcq_per_query: int = 1,
    agent_cq_capacity: int = 30,
    global_agent_span: int = 10,
    proposal_capacity: int = 30,
    target_classes: int = 0,
    llm_purpose: str = "gemini",
    dp_only: bool = False,
    # Global parallel execution config
    batch_size: int = 300,
    table_max_workers: int = 128,
    analyze_max_workers: int = 32,
    cq_max_concurrent: int = 16,
    dp_max_concurrent: int = 5,
    sh_max_workers: int = 8,
    # Stage 5: Retrieval Index options
    enable_faiss: bool = True,
    enable_bm25: bool = True,
    index_key: Optional[str] = None,  # td, td_cd, td_cd_cs
    rag_type: str = "hybrid",  # bm25, vector, hybrid
    embedding_batch_size: int = 256,
    remove_primitive_classes: bool = False,
    # TBox iteration selection
    tbox_iteration: int = -1,
    # Ablation options
    disable_transform_reuse: bool = False,
    # Stats output directory
    stats_dir: Optional[str] = None,
    # Experiment DB isolation
    output_base_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the UPO pipeline with selected stages.
    
    Args:
        dataset_name: Dataset to use
        step: Which stage(s) to run (federated_primitive_tbox, column_summary, layer2, export, retrieval_index, all, layer2_all)
        max_tables: Max tables for Stage 2/3/4
        output_format: Output format for Stage 4
        fresh_start: If True, clear Transform Contracts before Stage 2
        budget_multiplier: Successive Halving budget = k * num_rows * n_candidates (Stage 2)
        budget_cap: Upper limit on Successive Halving budget (Stage 2)
        disable_virtual_columns: If True, skip virtual column extraction in Stage 2
        n_clusters: Number of clusters (0 = auto-compute from capacity)
        total_queries: Total queries to load
        n_iterations: Number of proposal-synthesis iterations
        scq_per_query: Scoping CQs per query
        vcq_per_query: Validating CQs per query
        agent_cq_capacity: K - max CQs per agent
        global_agent_span: B - max branching factor
        proposal_capacity: P - max class proposals for global synthesis
        target_classes: Target number of classes (0 = no specific target)
        llm_purpose: LLM purpose key
        batch_size: Batch size for table processing in Stage 2/3/4
        table_max_workers: Max parallel workers for expand/classify/Stage3 (no nested parallelism)
        analyze_max_workers: Max parallel workers for analyze_columns (has nested SH)
        cq_max_concurrent: Stage 1 CQ generation max concurrent calls
        dp_max_concurrent: Stage 1 DP generation max concurrent calls
        sh_max_workers: Stage 2 max parallel workers for Successive Halving
        enable_faiss: Stage 5 - Generate FAISS vector index
        enable_bm25: Stage 5 - Generate BM25 keyword index
        index_key: Stage 5 - Index configuration key (td, td_cd, td_cd_cs)
        rag_type: Retrieval type for unified search ('bm25', 'vector', 'hybrid')
        embedding_batch_size: Stage 5 - Batch size for embedding generation
        tbox_iteration: TBox iteration to use for Stage 2+ (-1 = latest)
    
    Returns:
        Results from all executed stages
    """
    # Convert -1 to None for "all" semantics
    if max_tables == -1:
        max_tables = None
    if total_queries == -1:
        total_queries = None
    
    results = {}
    start_total = time.time()
    
    def _safe_run_stage(stage_name: str, stage_func, **kwargs) -> Dict[str, Any]:
        """Run a stage with exception handling."""
        try:
            return stage_func(**kwargs)
        except Exception as e:
            import traceback
            error_msg = str(e)
            tb_str = traceback.format_exc()
            logger.error(f"  ✗ {stage_name} raised exception: {error_msg}")
            logger.error(f"  Traceback:\n{tb_str}")
            return {
                'success': False,
                'error': error_msg,
                'traceback': tb_str,
                'elapsed': time.time() - start_total,
            }
    
    # Stage 1: Primitive TBox (Federated mode)
    if step in ("federated_primitive_tbox", "all"):
        results['stage1'] = _safe_run_stage(
            "Stage 1 (Federated)",
            run_federated_primitive_tbox,
            dataset_name=dataset_name,
            n_clusters=n_clusters,
            n_iterations=n_iterations,
            target_classes=target_classes,
            scq_per_query=scq_per_query,
            vcq_per_query=vcq_per_query,
            llm_purpose=llm_purpose,
            max_tables=max_tables,
            max_queries=total_queries,
            agent_cq_capacity=agent_cq_capacity,
            global_agent_span=global_agent_span,
            proposal_capacity=proposal_capacity,
            dp_only=dp_only,
            cq_max_concurrent=cq_max_concurrent,
            dp_max_concurrent=dp_max_concurrent,
            rag_type=rag_type,
        )
        if not results['stage1'].get('success'):
            results['total_elapsed'] = time.time() - start_total
            return results
    
    # Stage 2: Column Summary
    if step in ("column_summary", "all", "layer2_all"):
        results['stage2'] = _safe_run_stage(
            "Stage 2",
            run_column_summary,
            dataset_name=dataset_name,
            max_tables=max_tables,
            fresh_start=fresh_start,
            budget_multiplier=budget_multiplier,
            budget_cap=budget_cap,
            disable_virtual_columns=disable_virtual_columns,
            batch_size=batch_size,
            table_max_workers=table_max_workers,
            analyze_max_workers=analyze_max_workers,
            sh_max_workers=sh_max_workers,
            llm_purpose=llm_purpose,
            tbox_iteration=tbox_iteration,
            disable_transform_reuse=disable_transform_reuse,
            stats_dir=stats_dir,
        )
        if not results['stage2'].get('success'):
            results['total_elapsed'] = time.time() - start_total
            return results
    
    # Stage 3: Layer 2 Annotation
    if step in ("layer2_annotation", "all", "layer2_all"):
        results['stage3'] = _safe_run_stage(
            "Stage 3",
            run_layer2_annotation,
            dataset_name=dataset_name,
            max_tables=max_tables,
            batch_size=batch_size,
            table_max_workers=table_max_workers,
        )
        if not results['stage3'].get('success'):
            results['total_elapsed'] = time.time() - start_total
            return results
    
    # Stage 4: Summarization
    if step in ("summarize", "export", "all", "layer2_all"):
        results['stage4'] = _safe_run_stage(
            "Stage 4",
            run_summarization,
            dataset_name=dataset_name,
            output_format=output_format,
        )
        # Stage 4 is not critical, continue even on failure
    
    # Stage 5: Retrieval Index Generation
    if step in ("retrieval_index", "all", "layer2_all"):
        stage5_kwargs = dict(
            dataset_name=dataset_name,
            index_key=index_key,
            enable_faiss=enable_faiss,
            enable_bm25=enable_bm25,
            embedding_batch_size=embedding_batch_size,
            remove_primitive_classes=remove_primitive_classes,
        )
        if output_base_path:
            stage5_kwargs['output_base_path'] = Path(output_base_path)
        results['stage5'] = _safe_run_stage(
            "Stage 5",
            run_retrieval_index,
            **stage5_kwargs,
        )
        # Stage 5 is not critical, continue even on failure
    
    total_elapsed = time.time() - start_total
    results['total_elapsed'] = total_elapsed
    
    return results


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(
        description="UPO Pipeline Runner - 5 Stage Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  federated_primitive_tbox Stage 1: Federated Global-Local agent collaboration
  column_summary           Stage 2: Analyze columns with code-based statistics  
  layer2_annotation        Stage 3: LLM annotation to create Defined Classes
  summarize                Stage 4: Multi-view serialization for retrieval
  retrieval_index          Stage 5: Generate FAISS + BM25 indexes for semantic search
  all                      Run all 5 stages sequentially
  layer2_all               Run Stage 2+3+4+5 (skip primitive_tbox generation)

Key Parameters:
  --total-queries N        Total queries to load (default: 100)
  --n-clusters N           Number of clusters (0 = auto-compute from capacity)
  --agent-cq-capacity K    Max CQs per agent (default: 30)
  --global-agent-span B    Max branching factor (default: 10)
  --n-iterations N         Proposal-synthesis iterations (default: 1)

Stage 5 (Retrieval Index) Parameters:
  --enable-faiss/--disable-faiss   Generate FAISS vector index (default: enabled)
  --enable-bm25/--disable-bm25     Generate BM25 keyword index (default: enabled)
  --index-key KEY                  Index fields: td, td_cd, td_cd_cs (default: td_cd_cs)
  --embedding-batch-size N         Batch size for embedding generation (default: 256)

Examples:
  # Run Federated Stage 1 with auto-computed clusters
  python demos/run_upo_pipeline.py --total-queries 100 --agent-cq-capacity 30
  
  # Run with manual cluster count
  python demos/run_upo_pipeline.py --n-clusters 5 --total-queries 200
  
  # Run full pipeline (all 5 stages)
  python demos/run_upo_pipeline.py --step all --total-queries 100 --max-tables 50
  
  # Run Layer 2 pipeline using existing Layer 1
  python demos/run_upo_pipeline.py --step layer2_all --max-tables 50
  
  # Only generate retrieval indexes
  python demos/run_upo_pipeline.py --step retrieval_index
  
  # Multiple iterations for consensus
  python demos/run_upo_pipeline.py --total-queries 100 --n-iterations 2
        """
    )
    parser.add_argument(
        "-d", "--dataset",
        default=DEFAULT_DATASET,
        help=f"Dataset name (default: {DEFAULT_DATASET})"
    )
    parser.add_argument(
        "--step",
        choices=["federated_primitive_tbox", "column_summary", "layer2_annotation", "summarize", "export", "retrieval_index", "all", "layer2_all"],
        default="federated_primitive_tbox",
        help="Which stage to run (federated_primitive_tbox is the primary Stage 1 mode)"
    )
    parser.add_argument(
        "--max-tables",
        type=int,
        default=-1,
        help="Maximum tables for Stage 2/3/4 (default: -1, use -1 for ALL)"
    )
    parser.add_argument(
        "--output-format",
        choices=["upo", "pneuma", "both"],
        default="both",
        help="Output format for summaries"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Fresh start: clear Transform Contracts before Stage 2 (default: reuse existing)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file for synchronous write (keeps terminal colors, writes plain text to file)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for console output (default: INFO)"
    )
    parser.add_argument(
        "--disable-virtual-columns",
        action="store_true",
        help="Disable virtual column extraction from context in Stage 2 (enabled by default)"
    )
    
    # ========== Global Parallel Execution Config ==========
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for table processing in Stage 2/3/4 (default: 1000)"
    )
    parser.add_argument(
        "--table-max-workers",
        type=int,
        default=128,
        help="Max parallel workers for expand_virtual_columns, classify_columns, and Stage 3. "
             "These nodes have no nested parallelism, so high values are safe. (default: 128)"
    )
    parser.add_argument(
        "--analyze-max-workers",
        type=int,
        default=32,
        help="Max parallel workers for analyze_columns node. Lower than table-max-workers because "
             "analyze_columns has nested Successive Halving parallelism. "
             "Total threads ≈ analyze-max-workers × sh-max-workers. (default: 32)"
    )
    parser.add_argument(
        "--cq-max-concurrent",
        type=int,
        default=128,
        help="Stage 1: Max concurrent CQ generation calls (default: 16)"
    )
    parser.add_argument(
        "--dp-max-concurrent",
        type=int,
        default=5,
        help="Stage 1: Max concurrent DP generation calls (default: 5)"
    )
    parser.add_argument(
        "--sh-max-workers",
        type=int,
        default=2,
        help="Stage 2: Max parallel workers for Successive Halving evaluation. "
             "WARNING: Creates nested threads. Total threads ≈ analyze-max-workers × sh-max-workers. "
             "Keep low (4-8) to avoid thread explosion. (default: 2)"
    )
    parser.add_argument(
        "--budget-cap",
        type=int,
        default=1000,
        help="Stage 2: Upper limit on Successive Halving budget. "
             "Prevents excessive CPU time on large columns. (default: 1000)"
    )
    
    # ========== Stage 2 Ablation Options ==========
    parser.add_argument(
        "--disable-reuse",
        action="store_true",
        default=False,
        help="Stage 2: Disable transform reuse (always generate with LLM, for ablation experiments)"
    )
    
    parser.add_argument(
        "--stats-dir",
        type=str,
        default=None,
        help="Directory for LLM stats output files (default: logs/llm_calls/)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Override LanceDB path (for experiment isolation). Sets SATURN_DB_PATH env var."
    )
    parser.add_argument(
        "--output-base-path",
        type=str,
        default=None,
        help="Base path for Stage 5 index output (default: data/lake/)"
    )
    
    # ========== Stage 5: Retrieval Index Configuration ==========
    parser.add_argument(
        "--enable-faiss",
        action="store_true",
        default=True,
        help="Generate FAISS vector index (default: enabled)"
    )
    parser.add_argument(
        "--disable-faiss",
        action="store_true",
        help="Disable FAISS vector index generation"
    )
    parser.add_argument(
        "--enable-bm25",
        action="store_true",
        default=True,
        help="Generate BM25 keyword index (default: enabled)"
    )
    parser.add_argument(
        "--disable-bm25",
        action="store_true",
        help="Disable BM25 index generation"
    )
    parser.add_argument(
        "--index-key",
        type=str,
        default=None,
        choices=["td", "td_cd", "td_cd_cs"],
        help="Index fields: td (table_desc), td_cd (+column_desc), td_cd_cs (full, default)"
    )
    parser.add_argument(
        "--rag-type",
        type=str,
        default="hybrid",
        choices=["bm25", "vector", "hybrid"],
        help="Retrieval type for unified search: bm25, vector, or hybrid (default: hybrid)"
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation in Stage 5 (default: 256)"
    )
    parser.add_argument(
        "--no-primitive-classes",
        action="store_true",
        help="Ablation: remove [Type] markers from index documents (saves to {index_key}_no_pc/)"
    )
    
    # ========== TBox Iteration Selection ==========
    parser.add_argument(
        "--tbox-iteration",
        type=int,
        default=-1,
        help="TBox iteration to use for Stage 2+ (-1 = latest, default: -1)"
    )
    
    # ========== Stage 1 Options ==========
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=0,
        help="Number of clusters (0 = auto-compute from capacity, default: 0)"
    )
    parser.add_argument(
        "--total-queries",
        type=int,
        default=100,
        help="Total queries to load (default: 100, use -1 for ALL)"
    )
    parser.add_argument(
        "--llm-purpose",
        type=str,
        default="local",
        help="LLM purpose key (default: local)"
    )
    
    # ========== Federated Options ==========
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=1,
        help="Number of proposal-synthesis iterations (default: 1)"
    )
    parser.add_argument(
        "--scq-per-query",
        type=int,
        default=1,
        help="Number of Scoping CQs per query (default: 1)"
    )
    parser.add_argument(
        "--vcq-per-query",
        type=int,
        default=1,
        help="Number of Validating CQs per query (default: 1)"
    )
    parser.add_argument(
        "--agent-cq-capacity",
        type=int,
        default=30,
        help="K: max CQs per agent (default: 30)"
    )
    parser.add_argument(
        "--global-agent-span",
        type=int,
        default=10,
        help="B: max branching factor (default: 10)"
    )
    parser.add_argument(
        "--proposal-capacity",
        type=int,
        default=30,
        help="P: max class proposals for global synthesis per iteration (default: 30)"
    )
    parser.add_argument(
        "--target-classes",
        type=int,
        default=50,
        help="Target number of classes (0 = no specific target, let agents decide). "
             "Guides agents on the desired scale of the ontology."
    )
    parser.add_argument(
        "--dp-only",
        action="store_true",
        help="DP-only mode: skip class iterations, only regenerate DataProperties (load from LanceDB)"
    )
    
    args = parser.parse_args()
    
    # Set SATURN_DB_PATH if --db-path provided (before any store initialization)
    if args.db_path:
        os.environ['SATURN_DB_PATH'] = args.db_path
    
    # Configure console logger level
    logger.remove()  # Remove default handler
    logger.add(
        lambda msg: print(msg, end=""),
        format="<level>{time:HH:mm:ss}</level> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=args.log_level.upper(),
        colorize=True,
    )
    
    # Configure log file if specified
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add file handler without colors (plain text for file)
        # Use serialize=False to get human-readable format
        logger.add(
            args.log_file,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="100 MB",  # Rotate when file exceeds 100MB
            retention="7 days",  # Keep logs for 7 days
            compression="gz",  # Compress rotated files
            enqueue=False,  # Synchronous write (no buffering)
        )
        logger.info(f"Logging to file: {args.log_file}")
    
    # Estimate table count
    table_count_str = str(args.max_tables) if args.max_tables else "ALL"
    
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " UPO Pipeline Runner (5 Stages) ".center(68) + "║")
    logger.info("╚" + "═" * 68 + "╝")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Stage: {args.step}")
    if args.step == "federated_primitive_tbox":
        logger.info(f"  Stage 1 (Federated): {args.n_clusters} clusters, {args.n_iterations} iterations")
        logger.info(f"  CQs per Query: SCQ={args.scq_per_query}, VCQ={args.vcq_per_query}")
        logger.info(f"  Total Queries: {args.total_queries}, LLM: {args.llm_purpose}")
    elif args.step in ["layer2_all", "column_summary", "layer2_annotation", "summarize"]:
        logger.info(f"  Skipping Stage 1 (using existing TBox)")
    else:
        logger.info(f"  Stage 1 (legacy)")
    logger.info(f"  Stage 2/3/4: {table_count_str} tables")
    logger.info("")
    
    # Run pipeline
    results = None
    pipeline_error = None
    
    try:
        results = run_pipeline(
            dataset_name=args.dataset,
            step=args.step,
            max_tables=args.max_tables,
            output_format=args.output_format,
            fresh_start=args.fresh,
            disable_virtual_columns=args.disable_virtual_columns,
            # Federated options (primary mode)
            n_clusters=args.n_clusters,
            total_queries=args.total_queries,
            n_iterations=args.n_iterations,
            scq_per_query=args.scq_per_query,
            vcq_per_query=args.vcq_per_query,
            agent_cq_capacity=args.agent_cq_capacity,
            global_agent_span=args.global_agent_span,
            proposal_capacity=args.proposal_capacity,
            target_classes=args.target_classes,
            llm_purpose=args.llm_purpose,
            dp_only=args.dp_only,
            # Global parallel execution config
            batch_size=args.batch_size,
            table_max_workers=args.table_max_workers,
            analyze_max_workers=args.analyze_max_workers,
            cq_max_concurrent=args.cq_max_concurrent,
            dp_max_concurrent=args.dp_max_concurrent,
            sh_max_workers=args.sh_max_workers,
            budget_cap=args.budget_cap,
            # Stage 5: Retrieval Index options
            # Auto-determine enable_faiss/enable_bm25 based on rag_type if not explicitly disabled
            enable_faiss=(args.enable_faiss and not args.disable_faiss and args.rag_type in ("vector", "hybrid")),
            enable_bm25=(args.enable_bm25 and not args.disable_bm25 and args.rag_type in ("bm25", "hybrid")),
            index_key=args.index_key,
            embedding_batch_size=args.embedding_batch_size,
            remove_primitive_classes=args.no_primitive_classes,
            # TBox iteration selection
            tbox_iteration=args.tbox_iteration,
            # RAG type for retrieval
            rag_type=args.rag_type,
            # Ablation options
            disable_transform_reuse=args.disable_reuse,
            # Stats output
            stats_dir=args.stats_dir,
            # Experiment DB isolation
            output_base_path=args.output_base_path,
        )
    except Exception as e:
        pipeline_error = str(e)
        logger.error(f"Pipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results = {'error': pipeline_error, 'total_elapsed': 0}
    
    # Print summary
    logger.info("")
    logger.info("╔" + "═" * 68 + "╗")
    if pipeline_error:
        logger.info("║" + " Pipeline FAILED - Starting Keep-Alive ".center(68) + "║")
    else:
        logger.info("║" + " Pipeline Complete ".center(68) + "║")
    logger.info("╚" + "═" * 68 + "╝")
    
    if results:
        logger.info(f"  Total Time: {results.get('total_elapsed', 0):.2f}s")
        
        for stage_key in ['stage1', 'stage2', 'stage3', 'stage4', 'stage5']:
            if stage_key in results:
                stage_result = results[stage_key]
                status = "✓" if stage_result.get('success') else "✗"
                elapsed = stage_result.get('elapsed', 0)
                error_msg = f" - {stage_result.get('error', '')[:50]}" if not stage_result.get('success') else ""
                logger.info(f"  {status} {stage_key}: {elapsed:.2f}s{error_msg}")
    
    if pipeline_error:
        logger.error(f"  Error: {pipeline_error[:100]}")
    
    return results


if __name__ == "__main__":
    main()
