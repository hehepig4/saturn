"""
Column Summary Agent Graph Builder

Builds the LangGraph workflow for column analysis.

Four-stage approach:
1. expand_virtual_columns: Extract virtual columns from context (optional)
2. classify_columns: LLM assigns primitive_class to each column
3. analyze_columns (Stage 1): Try to reuse existing code from repository (0 LLM calls)
4. analyze_columns (Stage 2): Generate new code with LLM if no match found
"""

from typing import Literal, Optional
from langgraph.graph import StateGraph, START, END
from loguru import logger

from workflows.population.state import ColumnSummaryState
from workflows.population.nodes import (
    load_primitive_tbox_node,
    load_tables_batch_node,
    expand_virtual_columns_node,
    classify_columns_node,
    analyze_columns_node,
    save_batch_results_node,
    aggregate_summaries_node,
)


def should_continue(state: ColumnSummaryState) -> Literal["continue", "finish"]:
    """
    Router: Check if more batches need to be processed.
    
    Returns:
        "continue" if more batches remaining
        "finish" if all batches processed
    """
    if state.current_batch_index < state.total_batches:
        logger.info(f"Batch progress: {state.current_batch_index}/{state.total_batches}")
        return "continue"
    else:
        logger.info("All batches processed, finishing...")
        return "finish"


def build_column_summary_graph():
    """
    Build the Column Summary Agent subgraph.
    
    Graph Structure:
    
        START
          │
          ▼
        load_primitive_tbox
          │
          ▼
        load_tables_batch  ◄────────────┐
          │                              │
          ▼                              │
        expand_virtual_columns           │
          │                              │
          ▼                              │
        classify_columns                 │
          │                              │
          ▼                              │
        analyze_columns                  │
          │                              │
          ▼                              │
        save_batch_results               │
          │                              │
          ▼                              │
        [should_continue?] ──continue───┘
          │
          │ finish
          ▼
        aggregate_summaries
          │
          ▼
         END
    
    Returns:
        Compiled LangGraph workflow
    """
    logger.info("Building Column Summary Agent graph...")
    
    graph = StateGraph(ColumnSummaryState)
    
    # Add nodes
    graph.add_node("load_primitive_tbox", load_primitive_tbox_node)
    graph.add_node("load_tables_batch", load_tables_batch_node)
    graph.add_node("expand_virtual_columns", expand_virtual_columns_node)
    graph.add_node("classify_columns", classify_columns_node)
    graph.add_node("analyze_columns", analyze_columns_node)
    graph.add_node("save_batch_results", save_batch_results_node)
    graph.add_node("aggregate_summaries", aggregate_summaries_node)
    
    # Add edges
    graph.add_edge(START, "load_primitive_tbox")
    graph.add_edge("load_primitive_tbox", "load_tables_batch")
    graph.add_edge("load_tables_batch", "expand_virtual_columns")
    graph.add_edge("expand_virtual_columns", "classify_columns")
    graph.add_edge("classify_columns", "analyze_columns")
    graph.add_edge("analyze_columns", "save_batch_results")
    
    # Conditional routing for batch loop
    graph.add_conditional_edges(
        "save_batch_results",
        should_continue,
        {
            "continue": "load_tables_batch",
            "finish": "aggregate_summaries",
        }
    )
    
    graph.add_edge("aggregate_summaries", END)
    
    compiled = graph.compile()
    logger.info("Column Summary Agent graph compiled successfully")
    
    return compiled


def run_column_summary_workflow(
    dataset_name: str,
    table_store_name: str,
    output_dir: str = "data/cache/column_summaries",
    batch_size: int = 20,
    max_tables: Optional[int] = None,
    table_max_workers: int = 128,
    analyze_max_workers: int = 32,
    sh_max_workers: int = 8,
) -> ColumnSummaryState:
    """
    Run the column summary workflow.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'fetaqa')
        table_store_name: LanceDB table name
        output_dir: Directory for checkpoints and output
        batch_size: Tables per batch
        max_tables: Maximum tables to process (None for all)
        table_max_workers: Max workers for expand/classify nodes (no nested parallelism)
        analyze_max_workers: Max workers for analyze_columns (has nested SH parallelism)
        sh_max_workers: Max workers for Successive Halving evaluation
        
    Returns:
        Final workflow state with all_summaries
    """
    logger.info(f"Starting Column Summary workflow for {dataset_name}")
    logger.info(f"  Table store: {table_store_name}")
    logger.info(f"  Batch size: {batch_size}, Max tables: {max_tables}")
    logger.info(f"  Table Max Workers (expand/classify): {table_max_workers}")
    logger.info(f"  Analyze Max Workers: {analyze_max_workers}")
    logger.info(f"  Successive Halving Max Workers: {sh_max_workers}")
    
    # Build graph
    graph = build_column_summary_graph()
    
    # Initialize state
    initial_state = ColumnSummaryState(
        dataset_name=dataset_name,
        table_store_name=table_store_name,
        output_dir=output_dir,
        batch_size=batch_size,
        max_tables=max_tables,
        table_max_workers=table_max_workers,
        analyze_max_workers=analyze_max_workers,
        sh_max_workers=sh_max_workers,
    )
    
    # Run workflow
    final_state = graph.invoke(initial_state)
    
    # Log summary
    logger.info("=" * 60)
    logger.info("Column Summary Workflow Complete")
    logger.info(f"  Tables processed: {final_state.get('completed_tables', 0)}")
    logger.info(f"  Columns analyzed: {final_state.get('total_columns_analyzed', 0)}")
    logger.info(f"  LLM calls: {final_state.get('total_llm_calls', 0)}")
    logger.info(f"  Code reused: {final_state.get('code_reuse_count', 0)}")
    logger.info("=" * 60)
    
    return final_state
