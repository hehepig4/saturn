"""
Node: Save Batch Results

Saves the current batch results to disk and LanceDB for checkpointing.
"""

from typing import Dict, Any
from pathlib import Path
import json
from datetime import datetime
from loguru import logger
import pandas as pd

from workflows.common.node_decorators import graph_node
from workflows.indexing.annotation.state import (
    TableDiscoveryLayer2State,
    ColumnDefinedClass,
    TableDefinedClass,
)


@graph_node(node_type="processing", on_error="continue")
def save_batch_results_node(state: TableDiscoveryLayer2State) -> Dict[str, Any]:
    """
    Save current batch results to checkpoint file and LanceDB.
    
    Creates:
    1. JSON checkpoint file with annotations and defined classes
    2. Incremental updates to LanceDB tables
    
    Returns:
        Updated state with checkpoint info and cleared batch data
    """
    if not state.current_batch_table_classes:
        logger.info(f"Batch {state.current_batch_index + 1} empty, skipping checkpoint")
        return {
            "current_batch_index": state.current_batch_index + 1,
            "current_batch_tables": [],
            "current_batch_annotations": [],
            "current_batch_column_classes": [],
            "current_batch_table_classes": [],
        }
    
    logger.info(f"Saving batch {state.current_batch_index + 1}/{state.total_batches} results...")
    
    # Save checkpoint JSON file
    checkpoint_path = _save_checkpoint_file(state)
    
    # Save to LanceDB incrementally
    _save_to_lancedb(state)
    
    # Update progress
    new_completed = state.completed_tables + len(state.current_batch_table_classes)
    new_checkpoint_paths = state.checkpoint_paths + [checkpoint_path]
    new_batch_index = state.current_batch_index + 1
    
    logger.info(f"  ✓ Checkpoint saved: {checkpoint_path}")
    logger.info(f"  ✓ Progress: {new_completed}/{len(state.tables)} tables")
    
    return {
        "checkpoint_paths": new_checkpoint_paths,
        "completed_tables": new_completed,
        "current_batch_index": new_batch_index,
        # Clear batch data for next iteration
        "current_batch_tables": [],
        "current_batch_annotations": [],
        "current_batch_column_classes": [],
        "current_batch_table_classes": [],
    }


def _save_checkpoint_file(state: TableDiscoveryLayer2State) -> str:
    """Save batch results to JSON checkpoint file."""
    output_dir = Path(state.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"{state.dataset_name}_layer2_batch_{state.current_batch_index:04d}_{timestamp}.json"
    checkpoint_path = output_dir / checkpoint_name
    
    # Serialize data
    batch_data = {
        "metadata": {
            "dataset_name": state.dataset_name,
            "batch_index": state.current_batch_index,
            "total_batches": state.total_batches,
            "tables_in_batch": len(state.current_batch_table_classes),
            "timestamp": timestamp,
        },
        "annotations": [ann.model_dump() for ann in state.current_batch_annotations],
        "column_classes": [cc.model_dump() for cc in state.current_batch_column_classes],
        "table_classes": [tc.model_dump() for tc in state.current_batch_table_classes],
    }
    
    with open(checkpoint_path, 'w') as f:
        json.dump(batch_data, f, indent=2)
    
    return str(checkpoint_path)


def _save_to_lancedb(state: TableDiscoveryLayer2State) -> None:
    """Save batch results to LanceDB incrementally."""
    try:
        from store.store_singleton import get_store
        
        store = get_store()
        dataset_name = state.dataset_name
        
        # Save table defined classes
        if state.current_batch_table_classes:
            table_class_rows = []
            for tbl_class in state.current_batch_table_classes:
                table_class_rows.append({
                    "class_name": tbl_class.class_name,
                    "definition": tbl_class.el_definition,
                    "description": tbl_class.description,
                    "summary": tbl_class.summary,
                    "source_table": tbl_class.table_id,
                    "dataset": dataset_name,
                    "created_at": datetime.now().isoformat(),
                })
            
            class_table_name = f"{dataset_name}_table_defined_classes"
            class_df = pd.DataFrame(table_class_rows)
            
            try:
                if class_table_name in store.db.table_names(limit=1000000):
                    tbl = store.db.open_table(class_table_name)
                    tbl.add(class_df)
                else:
                    store.db.create_table(class_table_name, class_df)
            except ValueError as e:
                if "already exists" in str(e):
                    tbl = store.db.open_table(class_table_name)
                    tbl.add(class_df)
                else:
                    raise
            
            logger.info(f"    ✓ Saved {len(table_class_rows)} table classes to {class_table_name}")
        
        # Save column mappings
        if state.current_batch_column_classes:
            mapping_rows = []
            for col in state.current_batch_column_classes:
                source_table = col.column_id.split("::")[0] if "::" in col.column_id else ""
                mapping_rows.append({
                    "class_name": col.column_id,
                    "column_name": col.label,
                    "primitive_class": col.primitive_class,
                    "description": col.description,
                    "source_table": source_table,
                    "contract_id": col.contract_id,  # From Stage 2 TransformContract
                    "dataset": dataset_name,
                    "created_at": datetime.now().isoformat(),
                })
            
            mapping_table_name = f"{dataset_name}_column_mappings"
            mapping_df = pd.DataFrame(mapping_rows)
            
            try:
                if mapping_table_name in store.db.table_names(limit=1000000):
                    tbl = store.db.open_table(mapping_table_name)
                    tbl.add(mapping_df)
                else:
                    store.db.create_table(mapping_table_name, mapping_df)
            except ValueError as e:
                if "already exists" in str(e):
                    tbl = store.db.open_table(mapping_table_name)
                    tbl.add(mapping_df)
                else:
                    raise
            
            logger.info(f"    ✓ Saved {len(mapping_rows)} column mappings to {mapping_table_name}")
        
    except Exception as e:
        logger.error(f"Failed to save to LanceDB: {e}")
        raise
