"""
Table Discovery Layer 2 Graph

Creates a LangGraph workflow for generating Layer 2 Defined Classes
from tables using Layer 1 Primitive Classes.

Workflow (Batch Mode):
    1. load_data: Load tables, column summaries, and Layer 1 primitives (batched)
    2. annotate_tables: Annotate tables with Layer 1 classes (parallel, per batch)
    3. create_defined_classes: Create Column and Table Defined Classes (per batch)
    4. save_batch_results: Save checkpoint and update LanceDB (per batch)
    5. [loop back to load_data for next batch until all batches done]
    6. validate_layer2: Validate with OWL reasoner (optional)
    7. export_layer2: Final export to OWL files
"""

from typing import Optional, Literal
from loguru import logger
from langgraph.graph import StateGraph, START, END

from .state import TableDiscoveryLayer2State
from .nodes import (
    load_data_node,
    annotate_tables_node,
    create_defined_classes_node,
    save_batch_results_node,
    validate_layer2_node,
    export_layer2_node,
)


def should_continue_batches(state: TableDiscoveryLayer2State) -> Literal["continue", "finish"]:
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
        logger.info("All batches processed, proceeding to validation...")
        return "finish"


def create_table_discovery_layer2_graph() -> StateGraph:
    """
    Create the Table Discovery Layer 2 workflow graph with batch processing.
    
    Workflow:
        load_data → annotate_tables → create_defined_classes → save_batch_results
        → [should_continue_batches?] 
            → continue: loop back to load_data
            → finish: validate_layer2 → export_layer2 → END
    
    Returns:
        Compiled StateGraph workflow
    """
    workflow = StateGraph(TableDiscoveryLayer2State)
    
    # Add nodes
    workflow.add_node("load_data", load_data_node)
    workflow.add_node("annotate_tables", annotate_tables_node)
    workflow.add_node("create_defined_classes", create_defined_classes_node)
    workflow.add_node("save_batch_results", save_batch_results_node)
    workflow.add_node("validate_layer2", validate_layer2_node)
    workflow.add_node("export_layer2", export_layer2_node)
    
    # Define flow - Batch loop
    workflow.add_edge(START, "load_data")
    workflow.add_edge("load_data", "annotate_tables")
    workflow.add_edge("annotate_tables", "create_defined_classes")
    workflow.add_edge("create_defined_classes", "save_batch_results")
    
    # Conditional routing for batch loop
    workflow.add_conditional_edges(
        "save_batch_results",
        should_continue_batches,
        {
            "continue": "load_data",  # Loop back for next batch
            "finish": "validate_layer2",  # All batches done, skip to validation
        }
    )
    
    # Post-batch processing
    workflow.add_edge("validate_layer2", "export_layer2")
    workflow.add_edge("export_layer2", END)
    
    return workflow.compile()


def invoke_table_discovery_layer2(
    dataset_name: str = "fetaqa",
    max_tables: Optional[int] = None,
    llm_max_workers: int = 128,
    batch_size: int = 50,
    output_dir: str = "data/cache/layer2_checkpoints",
) -> TableDiscoveryLayer2State:
    """
    Invoke the Table Discovery Layer 2 workflow with batch processing.
    
    This workflow generates Layer 2 Defined Classes from:
    - Tables loaded from LanceDB (processed in batches)
    - Layer 1 Primitive Classes for classification
    - Column summaries for enhanced annotation
    
    Batch processing enables:
    - Checkpointing after each batch (resume on failure)
    - Incremental updates to LanceDB
    - Progress tracking
    
    Args:
        dataset_name: Name of the dataset (e.g., 'fetaqa')
        max_tables: Maximum tables to process (None = all)
        llm_max_workers: Max parallel workers for LLM calls
        batch_size: Number of tables per batch for checkpointing (default: 50)
        output_dir: Directory for checkpoint files
        
    Returns:
        Final workflow state with Layer 2 Defined Classes
    """
    logger.info("=" * 80)
    logger.info("Table Discovery Layer 2 Workflow (Batch Mode)")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Max Tables: {max_tables or 'all'}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"LLM Max Workers: {llm_max_workers}")
    logger.info(f"Output Dir: {output_dir}")
    
    # Create initial state
    initial_state = TableDiscoveryLayer2State(
        dataset_name=dataset_name,
        max_tables=max_tables,
        llm_max_workers=llm_max_workers,
        batch_size=batch_size,
        output_dir=output_dir,
    )
    
    # Create and run the graph
    graph = create_table_discovery_layer2_graph()
    result = graph.invoke(initial_state)
    
    # Log summary
    if result.get('success'):
        logger.info("")
        logger.info("=" * 80)
        logger.info("Layer 2 Generation Complete")
        logger.info("=" * 80)
        logger.info(f"  Column Defined Classes: {len(result.get('column_defined_classes', []))}")
        logger.info(f"  Table Defined Classes: {len(result.get('table_defined_classes', []))}")
        logger.info(f"  Ontology ID: {result.get('layer2_ontology_id')}")
        if result.get('owl_xml_path'):
            logger.info(f"  OWL/XML: {result.get('owl_xml_path')}")
    else:
        logger.error(f"Layer 2 generation failed: {result.get('error')}")
    
    return result
