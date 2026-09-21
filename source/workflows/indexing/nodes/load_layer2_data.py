"""
Node: Load Layer 2 Data

Loads column summaries, column mappings (Layer 2 column classes), 
table defined classes, and column relationships from LanceDB.
"""

from typing import Dict, Any
from loguru import logger

from workflows.common.node_decorators import graph_node


@graph_node()
def load_layer2_data_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load Layer 2 data from LanceDB for summarization.
    
    Loads:
        - Column summaries (statistics)
        - Column mappings (Layer 2 column classes)
        - Table defined classes (Layer 2 table classes)
        - Column relationships (discovered semantic relationships)
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with loaded data and lookup structures
    """
    dataset_name = state.dataset_name
    
    logger.info(f"Loading Layer 2 data for dataset: {dataset_name}")
    
    from store.store_singleton import get_store
    store = get_store()
    
    # Load column summaries
    try:
        col_summary_table = store.db.open_table(f"{dataset_name}_column_summaries")
        column_summaries_df = col_summary_table.to_pandas()
        column_summaries = column_summaries_df.to_dict('records')
        logger.debug(f"  ✓ Loaded {len(column_summaries)} column summaries")
    except Exception as e:
        logger.warning(f"  ⚠ Failed to load column summaries: {e}")
        column_summaries = []
    
    # Load table defined classes
    try:
        table_classes_table = store.db.open_table(f"{dataset_name}_table_defined_classes")
        table_classes_df = table_classes_table.to_pandas()
        table_classes = table_classes_df.to_dict('records')
        logger.debug(f"  ✓ Loaded {len(table_classes)} table defined classes")
    except Exception as e:
        logger.warning(f"  ⚠ Failed to load table defined classes: {e}")
        table_classes = []
    
    # Load column mappings (Layer 2 column classes)
    try:
        col_mappings_table = store.db.open_table(f"{dataset_name}_column_mappings")
        column_mappings_df = col_mappings_table.to_pandas()
        column_mappings = column_mappings_df.to_dict('records')
        logger.debug(f"  ✓ Loaded {len(column_mappings)} column mappings")
    except Exception as e:
        logger.warning(f"  ⚠ Failed to load column mappings: {e}")
        column_mappings = []
    
    # Load column relationships (discovered in Layer 2)
    relationships = []
    relationships_by_table = {}
    try:
        rel_table_name = f"{dataset_name}_column_relationships"
        if rel_table_name in store.db.table_names(limit=1000000):
            rel_table = store.db.open_table(rel_table_name)
            rel_df = rel_table.to_pandas()
            relationships = rel_df.to_dict('records')
            
            # Build lookup by table_id
            for rel in relationships:
                table_id = rel.get('table_id', '')
                if table_id not in relationships_by_table:
                    relationships_by_table[table_id] = []
                relationships_by_table[table_id].append(rel)
            
            logger.debug(f"  ✓ Loaded {len(relationships)} column relationships")
        else:
            logger.debug(f"  ⚠ No relationships table found ({rel_table_name})")
    except Exception as e:
        logger.warning(f"  ⚠ Failed to load column relationships: {e}")
    
    # Build lookup: column summaries by table_id -> column_name
    col_summaries_by_table = {}
    for row in column_summaries:
        table_id = row.get('table_id', '')
        if table_id not in col_summaries_by_table:
            col_summaries_by_table[table_id] = {}
        col_name = row.get('column_name', '')
        col_summaries_by_table[table_id][col_name] = row
    
    # Build lookup: column classes by table_id
    col_classes_by_table = {}
    for row in column_mappings:
        col_id = row.get('class_name', '')
        if '::' in col_id:
            table_id = col_id.rsplit('::', 1)[0]
        else:
            table_id = row.get('source_table', '')
        if table_id not in col_classes_by_table:
            col_classes_by_table[table_id] = []
        col_classes_by_table[table_id].append(row)
    
    logger.debug(f"  ✓ Built lookup structures for {len(col_summaries_by_table)} tables")
    
    return {
        'data_loaded': True,
        'column_summaries': column_summaries,
        'column_mappings': column_mappings,
        'table_classes': table_classes,
        'col_summaries_by_table': col_summaries_by_table,
        'col_classes_by_table': col_classes_by_table,
        'relationships': relationships,
        'relationships_by_table': relationships_by_table,
    }
