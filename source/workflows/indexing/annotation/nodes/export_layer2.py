"""
Node 5: Export Layer 2

Stores Layer 2 Defined Classes to LanceDB.

This node persists to LanceDB for downstream retrieval.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from loguru import logger

from workflows.common.node_decorators import graph_node
from workflows.indexing.annotation.state import (
    TableDiscoveryLayer2State,
    ColumnDefinedClass,
    TableDefinedClass,
)


@graph_node(node_type="finalization")
def export_layer2_node(state: TableDiscoveryLayer2State) -> Dict[str, Any]:
    """
    Export Layer 2 to LanceDB.
    
    Stores:
        - Ontology metadata
        - Column Defined Classes (to {dataset}_column_mappings)
        - Table Defined Classes (to {dataset}_table_defined_classes)
    
    Returns:
        State updates with export status
    """
    from store.store_singleton import get_store
    
    logger.info("=" * 70)
    logger.info("Node 5: Exporting Layer 2 to LanceDB")
    logger.info("=" * 70)
    
    column_classes = state.column_defined_classes
    table_classes = state.table_defined_classes
    primitive_usage = state.primitive_classes_used
    dataset_name = state.dataset_name
    timestamp = state.timestamp
    
    layer2_ontology_id = None
    
    # ========== Store to LanceDB ==========
    try:
        layer2_ontology_id = _store_to_lancedb(
            column_classes=column_classes,
            table_classes=table_classes,
            primitive_usage=primitive_usage,
            dataset_name=dataset_name,
            layer1_ontology_id=state.layer1_ontology_id,
            correction_stats=state.correction_stats,
            timestamp=timestamp,
        )
        logger.info(f"  ✓ Stored to LanceDB: {layer2_ontology_id}")
    except Exception as e:
        logger.error(f"  ✗ Failed to store to LanceDB: {e}")
        import traceback
        traceback.print_exc()
    
    # Log summary - read actual counts from LanceDB (since batch data is cleared)
    store = get_store()
    actual_column_count = 0
    actual_table_count = 0
    
    try:
        col_table = store.db.open_table(f"{dataset_name}_column_mappings")
        actual_column_count = len(col_table.to_pandas())
    except:
        pass
    
    try:
        tbl_table = store.db.open_table(f"{dataset_name}_table_defined_classes")
        actual_table_count = len(tbl_table.to_pandas())
    except:
        pass
    
    logger.info("")
    logger.info("  Summary (from LanceDB):")
    logger.info(f"    Column Classes: {actual_column_count}")
    logger.info(f"    Table Classes: {actual_table_count}")
    
    return {
        "export_done": True,
        "layer2_ontology_id": layer2_ontology_id,
        "success": True,
        "actual_counts": {
            "column_classes": actual_column_count,
            "table_classes": actual_table_count,
        }
    }


def _store_to_lancedb(
    column_classes: List[ColumnDefinedClass],
    table_classes: List[TableDefinedClass],
    primitive_usage: Dict[str, int],
    dataset_name: str,
    layer1_ontology_id: Optional[str],
    correction_stats: Dict[str, int],
    timestamp: str,
) -> str:
    """Store Layer 2 Defined Classes and Relationships to LanceDB."""
    from store.store_singleton import get_store
    from store.ontology.ontology_table import (
        OntologyTableManager,
        OntologyClassTableManager,
    )
    import pandas as pd
    
    store = get_store()
    
    # Create ontology metadata
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    ontology_id = f"table_upo_layer2_{ts_str}"
    
    meta_mgr = OntologyTableManager(store.db)
    meta_mgr.create_ontology(
        name="TableUPO_Layer2",
        ontology_type="table_upo_layer2",
        namespace="http://www.example.org/table_upo/layer2#",
        version=f"v1_{ts_str}",
        domain=["table_discovery"],
        dataset_name=dataset_name,
        metrics={
            "num_column_classes": len(column_classes),
            "num_table_classes": len(table_classes),
            "primitive_usage": primitive_usage,
            "layer1_ontology_id": layer1_ontology_id,
            "validation_corrections": correction_stats,
        },
        notes=f"Layer 2 Defined Classes with P0/P1 enhancements. Generated at {timestamp}",
        ontology_id=ontology_id,
    )
    
    # Store classes - BATCH INSERT
    class_mgr = OntologyClassTableManager(store.db)
    
    column_class_batch = []
    for col_class in column_classes:
        safe_name = col_class.column_id.replace("::", "_").replace(" ", "_").replace("-", "_")
        class_id = f"{ontology_id}:{safe_name}"
        column_class_batch.append({
            "class_id": class_id,
            "ontology_id": ontology_id,
            "class_name": safe_name,
            "label": col_class.label,
            "description": col_class.description,
            "parent_classes": [col_class.primitive_class],
            "equivalent_classes": [],
            "disjoint_with": [],
            "created_at": datetime.now(),
        })
    
    table_class_batch = []
    for tbl_class in table_classes:
        class_id = f"{ontology_id}:{tbl_class.class_name}"
        table_class_batch.append({
            "class_id": class_id,
            "ontology_id": ontology_id,
            "class_name": tbl_class.class_name,
            "label": tbl_class.label,
            "description": tbl_class.description,
            "parent_classes": ["Table"],
            "equivalent_classes": [],
            "disjoint_with": [],
            "created_at": datetime.now(),
        })
    
    all_classes = column_class_batch + table_class_batch
    if all_classes:
        class_mgr.insert(all_classes)
    
    meta_mgr.update_counts(
        ontology_id=ontology_id,
        num_classes=len(column_classes) + len(table_classes),
        num_object_properties=0,
        num_data_properties=0,
        num_axioms=0,
    )
    
    # NOTE: Dataset-specific tables ({dataset}_column_mappings, {dataset}_table_defined_classes)
    # are now saved incrementally in save_batch_results_node during batch processing.
    # This avoids duplicate writes and ensures contract_id is preserved.
    # Relationships are also saved incrementally in save_batch_results_node.
    
    return ontology_id
