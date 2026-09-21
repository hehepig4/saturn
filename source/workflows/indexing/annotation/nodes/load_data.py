"""
Node 1: Load Data

Loads tables, column summaries, and Layer 1 primitive classes from LanceDB.
"""

from typing import Dict, Any, List
from loguru import logger

from workflows.common.node_decorators import graph_node
from workflows.indexing.annotation.state import TableDiscoveryLayer2State


@graph_node(node_type="processing")
def load_data_node(state: TableDiscoveryLayer2State) -> Dict[str, Any]:
    """
    Load all required data for Layer 2 annotation.
    
    Loads:
        1. Tables from LanceDB
        2. Column summaries (pre-computed statistics)
        3. Layer 1 Primitive Classes
        4. Layer 1 Data Properties
    
    Returns:
        State updates with loaded data
    """
    from store.store_singleton import get_store
    from store.ontology.ontology_table import (
        OntologyTableManager,
        OntologyClassTableManager,
        OntologyPropertyTableManager,
    )
    
    # For first batch: load all data and setup batches
    # For subsequent batches: just load next batch of tables
    is_first_batch = state.current_batch_index == 0
    
    if is_first_batch:
        logger.info("=" * 70)
        logger.info("Node 1: Loading Data for Layer 2 (Batch Mode)")
        logger.info("=" * 70)
    else:
        logger.info(f"Loading batch {state.current_batch_index + 1}/{state.total_batches}...")
    
    dataset_name = state.dataset_name
    max_tables = state.max_tables
    table_offset = state.table_offset
    batch_size = state.batch_size
    
    store = get_store()
    
    # ========== Load Tables (First Batch Only) ==========
    if is_first_batch:
        table_store_name = f"{dataset_name}_tables_entries"
        try:
            table_store = store.db.open_table(table_store_name)
            df = table_store.to_pandas()
            tables = df.to_dict('records')
            
            # Apply offset and limit
            if table_offset > 0:
                tables = tables[table_offset:]
            
            if max_tables is not None and max_tables > 0:
                tables = tables[:max_tables]
            
            if not tables:
                logger.warning(f"  ⚠ No tables found after offset={table_offset}")
                return {
                    "success": False,
                    "error": f"No tables available at offset {table_offset}",
                }
            
            # Check for already processed tables and filter them out
            processed_table_ids = _get_processed_table_ids(store, dataset_name)
            starting_batch_index = 0
            
            if processed_table_ids:
                original_count = len(tables)
                # Filter out already processed tables
                tables = [t for t in tables if t.get('table_id') not in processed_table_ids]
                skipped_count = original_count - len(tables)
                
                if skipped_count > 0:
                    logger.info(f"  ✓ Found {len(processed_table_ids)} already processed tables")
                    logger.info(f"  ✓ Skipped {skipped_count} processed tables, {len(tables)} remaining")
            
            logger.info(f"  ✓ Loaded {len(tables)} tables from {table_store_name} (offset={table_offset})")
        except Exception as e:
            logger.error(f"  ✗ Failed to load tables: {e}")
            return {
                "success": False,
                "error": f"Failed to load tables: {e}",
            }
    else:
        # Use tables from state
        tables = state.tables
        starting_batch_index = state.current_batch_index
    
    # ========== Load Column Summaries ==========
    summary_table_name = f"{dataset_name}_column_summaries"
    column_summaries = {}
    
    try:
        summary_table = store.db.open_table(summary_table_name)
        summary_df = summary_table.to_pandas()
        
        for _, row in summary_df.iterrows():
            table_id = row['table_id']
            if table_id not in column_summaries:
                column_summaries[table_id] = []
            column_summaries[table_id].append(row.to_dict())
        
        logger.info(f"  ✓ Loaded column summaries for {len(column_summaries)} tables")
    except Exception as e:
        logger.warning(f"  ⚠ Column summaries not found: {e}")
        logger.warning("    Continuing without column summaries...")
    
    # Filter tables to only include those with column summaries
    # Tables without headers are skipped entirely in Stage 2 and won't have summaries
    # Tables with headers but no data rows still get summaries (with primitive_class but no data property values)
    if is_first_batch and column_summaries:
        original_count = len(tables)
        tables = [t for t in tables if t.get('table_id') in column_summaries]
        filtered_count = original_count - len(tables)
        if filtered_count > 0:
            logger.info(f"  ✓ Filtered {filtered_count} tables without column summaries (no headers)")
    
    # ========== Load Layer 1 Primitive Classes ==========
    primitive_classes = []
    layer1_ontology_id = None
    
    try:
        meta_mgr = OntologyTableManager(store.db)
        
        # Try federated_primitive_tbox first, then fallback to primitive_tbox
        ontology = meta_mgr.get_latest_version("federated_primitive_tbox", dataset_name)
        if not ontology:
            ontology = meta_mgr.get_latest_version("primitive_tbox", dataset_name)
        
        if not ontology:
            raise ValueError(f"No primitive_tbox or federated_primitive_tbox ontology found for {dataset_name}")
        
        layer1_ontology_id = ontology['ontology_id']
        logger.info(f"  ✓ Found Layer 1 ontology: {layer1_ontology_id}")
        
        # Load classes
        class_mgr = OntologyClassTableManager(store.db)
        class_records = class_mgr.get_classes_by_ontology(layer1_ontology_id)
        
        for rec in class_records:
            primitive_classes.append({
                "name": rec['class_name'],
                "label": rec.get('label', rec['class_name']),
                "description": rec.get('description', ''),
                "parent_classes": rec.get('parent_classes', []),
            })
        
        logger.info(f"  ✓ Loaded {len(primitive_classes)} primitive classes")
        
    except Exception as e:
        logger.error(f"  ✗ Failed to load Layer 1 primitive classes from LanceDB: {e}")
        return {
            "success": False,
            "error": f"Failed to load Layer 1 primitive classes: {e}. Run Stage 1 first.",
        }
    
    if len(primitive_classes) == 0:
        return {
            "success": False,
            "error": "No Layer 1 primitive classes found. Run primitive_tbox subgraph first.",
        }
    
    # Extract class names for validation
    layer1_class_names = [p['name'] for p in primitive_classes]
    
    # ========== Load Layer 1 Data Properties ==========
    data_properties = []
    
    if layer1_ontology_id:
        try:
            prop_mgr = OntologyPropertyTableManager(store.db)
            prop_records = prop_mgr.get_properties_by_ontology(layer1_ontology_id)
            
            for rec in prop_records:
                if rec.get('property_type') == 'data':
                    data_properties.append({
                        'name': rec['property_name'],
                        'label': rec.get('label', rec['property_name']),
                        'description': rec.get('description', ''),
                        'domain': rec.get('domain', []),
                        'range_type': rec.get('range_type', 'xsd:string'),
                        'readout_template': rec.get('readout_template', ''),
                        'statistics_requirements': rec.get('statistics_requirements', []),
                    })
            
            logger.info(f"  ✓ Loaded {len(data_properties)} DataProperties")
            
        except Exception as e:
            logger.warning(f"  ⚠ Failed to load Layer 1 Properties: {e}")
    
    # ========== Setup Batches (First Batch Only) ==========
    if is_first_batch:
        import math
        total_batches = math.ceil(len(tables) / batch_size) if tables else 0
        logger.info(f"  ✓ Total: {total_batches} batches of {batch_size} tables each")
        
        # Always start from batch 0 since we've filtered out processed tables
        current_batch_index = 0
    else:
        total_batches = state.total_batches
        current_batch_index = state.current_batch_index
    
    # ========== Get Current Batch Tables ==========
    start_idx = current_batch_index * batch_size
    end_idx = min(start_idx + batch_size, len(tables))
    current_batch_tables = tables[start_idx:end_idx]
    
    if not current_batch_tables:
        logger.warning(f"  ⚠ No tables in batch {current_batch_index + 1}")
        return {
            "data_loaded": True,
            "current_batch_tables": [],
        }
    
    logger.info(f"  ✓ Batch {current_batch_index + 1}/{total_batches}: {len(current_batch_tables)} tables (indices {start_idx}-{end_idx-1})")
    
    # ========== Summary ==========
    if is_first_batch:
        logger.info("")
        logger.info("  Data Loading Summary:")
        logger.info(f"    Total Tables: {len(tables)}")
        logger.info(f"    Column Summaries: {len(column_summaries)} tables")
        logger.info(f"    Primitive Classes: {len(primitive_classes)}")
        logger.info(f"    Data Properties: {len(data_properties)}")
        logger.info(f"    Total Batches: {total_batches}")
    
    return {
        "data_loaded": True,
        "tables": tables if is_first_batch else state.tables,
        "column_summaries": column_summaries if is_first_batch else state.column_summaries,
        "primitive_classes": primitive_classes if is_first_batch else state.primitive_classes,
        "layer1_class_names": layer1_class_names if is_first_batch else state.layer1_class_names,
        "data_properties": data_properties if is_first_batch else state.data_properties,
        "layer1_ontology_id": layer1_ontology_id if is_first_batch else state.layer1_ontology_id,
        "total_batches": total_batches,
        "current_batch_index": current_batch_index,
        "current_batch_tables": current_batch_tables,
        "completed_tables": 0 if is_first_batch else state.completed_tables,  # Start from 0 since we filtered out processed tables
    }


def _get_processed_table_ids(store, dataset_name: str) -> set:
    """Get set of already processed table IDs from LanceDB."""
    try:
        table_name = f"{dataset_name}_table_defined_classes"
        # Use direct open_table instead of table_names() which has bugs
        tbl = store.db.open_table(table_name)
        df = tbl.to_pandas()
        return set(df['source_table'].unique())
    except Exception as e:
        logger.debug(f"Could not load processed tables: {e}")
    return set()

