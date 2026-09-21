"""
Load Data Node for Federated Primitive TBox

Loads tables and queries from LanceDB.
Reuses logic from primitive_tbox.nodes.load_data.
"""

from typing import Dict, Any
from loguru import logger

from workflows.common.node_decorators import graph_node
from workflows.conceptualization.state import FederatedPrimitiveTBoxState


@graph_node(node_type="processing", log_level="INFO")
def load_data_node(state: FederatedPrimitiveTBoxState) -> Dict[str, Any]:
    """
    Load tables and queries from LanceDB.
    
    Reuses logic from primitive_tbox.nodes.load_data but adapted for
    FederatedPrimitiveTBoxState.
    
    Note: max_tables is NOT used in Stage 1 (federated_primitive_tbox).
    Tables are retrieved per-query during clustering phase.
    """
    from store.store_singleton import get_store
    
    logger.info("Loading data from LanceDB...")
    logger.info(f"  Table Store: {state.table_store_name}")
    logger.info(f"  Query Store: {state.query_store_name}")
    
    try:
        store = get_store()
        db = store.db
        
        # Load ALL tables (no max_tables limit in Stage 1)
        # Tables are retrieved per-query during clustering phase
        table_store = db.open_table(state.table_store_name)
        tables_df = table_store.to_pandas()
        tables = tables_df.to_dict('records')
        
        logger.info(f"  Loaded {len(tables)} tables (all available)")
        
        # Load queries (with optional limit)
        query_store = db.open_table(state.query_store_name)
        queries_df = query_store.to_pandas()
        queries = queries_df.to_dict('records')
        
        if state.max_queries is not None:
            queries = queries[:state.max_queries]
        
        logger.info(f"  Loaded {len(queries)} queries")
        
        return {
            "tables": tables,
            "queries": queries,
        }
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {
            "error_message": str(e),
            "phase_errors": {"load_data": [str(e)]},
        }
