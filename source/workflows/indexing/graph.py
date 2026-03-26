"""
Table Summarization Graph

Creates a LangGraph workflow for generating multi-view summaries
from Layer 2 Defined Classes for retrieval and embedding.

Workflow:
    1. load_layer2_data: Load column mappings, table classes, column summaries
    2. generate_summaries: Create multi-view summaries for each table
    3. export_summaries: Save to JSON and Parquet formats
    4. export_for_retrieval: Export to LanceDB for vector retrieval
"""

from typing import Optional
from loguru import logger
from langgraph.graph import StateGraph, END

from .state import TableSummarizationState, SummarizationOutput
from .nodes import (
    load_layer2_data_node,
    generate_summaries_node,
    export_summaries_node,
    export_for_retrieval_node,
)


def create_table_summarization_graph() -> StateGraph:
    """
    Create the Table Summarization workflow graph.
    
    Workflow:
        load_layer2_data → generate_summaries → export_summaries → export_for_retrieval → END
    
    Returns:
        Compiled StateGraph workflow
    """
    workflow = StateGraph(TableSummarizationState)
    
    # Add nodes
    workflow.add_node("load_layer2_data", load_layer2_data_node)
    workflow.add_node("generate_summaries", generate_summaries_node)
    workflow.add_node("export_summaries", export_summaries_node)
    workflow.add_node("export_for_retrieval", export_for_retrieval_node)
    
    # Define flow
    workflow.set_entry_point("load_layer2_data")
    workflow.add_edge("load_layer2_data", "generate_summaries")
    workflow.add_edge("generate_summaries", "export_summaries")
    workflow.add_edge("export_summaries", "export_for_retrieval")
    workflow.add_edge("export_for_retrieval", END)

    return workflow.compile()


def invoke_table_summarization(
    dataset_name: str = "fetaqa",
    output_base: str = "",  # Will use lake_data_path('summaries') if empty
    serialization_level: int = 2,
) -> SummarizationOutput:
    """
    Invoke the Table Summarization workflow.
    
    This workflow generates multi-view summaries for retrieval:
    - View Summaries: Key, Temporal, Measure, Attribute, Core
    - Blocked Views: Table-level serializations for embedding
    - PNEUMA-compatible format for benchmark comparison
    
    Args:
        dataset_name: Name of the dataset (e.g., 'fetaqa')
        output_base: Base output directory (uses lake_data_path if empty)
        serialization_level: Column serialization level (1, 2, or 3)
        
    Returns:
        SummarizationOutput with paths to exported files
    """
    from core.paths import lake_data_path
    
    # Use absolute path from lake_data_path if not specified
    if not output_base:
        output_base = str(lake_data_path('summaries'))
    
    logger.info("=" * 80)
    logger.info("Table Summarization Workflow")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Output Base: {output_base}")
    logger.info(f"Serialization Level: {serialization_level}")
    
    # Create initial state
    initial_state = TableSummarizationState(
        dataset_name=dataset_name,
        output_base=output_base,
        serialization_level=serialization_level,
    )
    
    # Create and run the graph
    graph = create_table_summarization_graph()
    result = graph.invoke(initial_state)
    
    # Log summary
    if result.get('success'):
        output = result.get('output')
        logger.info("")
        logger.info("=" * 80)
        logger.info("Summarization Complete")
        logger.info("=" * 80)
        if output:
            logger.info(f"  Table Summaries: {output.num_summaries}")
            if output.pneuma_json_path:
                logger.info(f"  PNEUMA JSON: {output.pneuma_json_path}")
    else:
        logger.error(f"Summarization failed: {result.get('error')}")
    
    return result.get('output')
