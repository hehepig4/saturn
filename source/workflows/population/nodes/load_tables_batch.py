"""
Node 2: Load Tables Batch

Loads a batch of tables from LanceDB for column analysis.
Supports pagination for large datasets.
"""

from typing import Dict, Any, List
import json
from loguru import logger

from workflows.common.node_decorators import graph_node
from workflows.population.state import ColumnSummaryState


@graph_node(node_type="processing", on_error="continue")
def load_tables_batch_node(state: ColumnSummaryState) -> Dict[str, Any]:
    """
    Load a batch of tables from LanceDB.
    
    Uses offset-based pagination to process tables in batches.
    Loads full row data for column analysis.
    Supports resume: skips tables already in {dataset}_column_summaries.
    
    Returns:
        Updated state with current_batch_tables and batch metadata
    """
    from store.store_singleton import get_store
    
    logger.info(f"Loading table batch {state.current_batch_index + 1}...")
    
    store = get_store()
    
    # Load processed table IDs ONLY ONCE on first batch, then cache in state
    if state.processed_table_ids is None:
        processed_table_ids = _get_processed_table_ids(store.db, state.dataset_name)
        if processed_table_ids:
            logger.info(f"  Found {len(processed_table_ids)} already-processed tables, will skip them")
        # Cache for future batches
        save_processed_ids = processed_table_ids
    else:
        # Reuse cached processed_table_ids from state (avoids 8s reload every batch)
        processed_table_ids = state.processed_table_ids
        save_processed_ids = None  # Don't update state again
        logger.debug(f"  Using cached processed_table_ids ({len(processed_table_ids)} tables)")
    
    # Calculate batch parameters
    offset = state.current_batch_index * state.batch_size
    
    # Get total count on first batch
    if state.current_batch_index == 0:
        total_tables = _get_table_count(store.db, state.table_store_name)
        
        # Apply max_tables limit (None means all)
        if state.max_tables is not None:
            total_tables = min(total_tables, state.max_tables)
        
        total_batches = (total_tables + state.batch_size - 1) // state.batch_size
        
        logger.info(f"Total tables: {total_tables}, batches: {total_batches}")
    else:
        total_tables = state.total_tables
        total_batches = state.total_batches
    
    # Calculate actual limit for this batch (respecting max_tables)
    remaining = total_tables - offset
    limit = min(state.batch_size, remaining)
    
    # Load batch and filter out already-processed tables
    tables = _load_batch(
        db=store.db,
        store_name=state.table_store_name,
        offset=offset,
        limit=limit,
        processed_table_ids=processed_table_ids
    )
    
    logger.info(f"Loaded {len(tables)} tables for batch {state.current_batch_index + 1}/{total_batches}")
    
    # Log sample table info
    if tables:
        sample = tables[0]
        logger.debug(f"Sample table: {sample.get('table_id')}, columns: {sample.get('columns', [])[:3]}...")
    
    result = {
        "current_batch_tables": tables,
        "total_tables": total_tables,
        "total_batches": total_batches,
    }
    
    # Only save processed_table_ids to state on first load (cache it)
    if save_processed_ids is not None:
        result["processed_table_ids"] = save_processed_ids
    
    return result


def _get_table_count(db, store_name: str) -> int:
    """Get total number of tables in the store."""
    try:
        table = db.open_table(store_name)
        return table.count_rows()
    except Exception as e:
        logger.error(f"Failed to get table count: {e}")
        return 0


def _get_processed_table_ids(db, dataset_name: str) -> set:
    """
    Get set of table IDs that have already been processed.
    
    Checks {dataset}_column_summaries table in LanceDB.
    Returns empty set if table doesn't exist.
    """
    try:
        table_name = f"{dataset_name}_column_summaries"
        table = db.open_table(table_name)
        
        # Get unique table IDs
        records = table.search().limit(100000).to_list()
        table_ids = set(record['table_id'] for record in records if 'table_id' in record)
        
        logger.debug(f"Found {len(table_ids)} processed tables in {table_name}")
        return table_ids
        
    except Exception as e:
        # Table doesn't exist or error - no processed tables
        logger.debug(f"No processed tables found ({dataset_name}_column_summaries): {e}")
        return set()


def _load_batch(
    db,
    store_name: str,
    offset: int,
    limit: int,
    processed_table_ids: set = None
) -> List[Dict[str, Any]]:
    """
    Load a batch of tables from LanceDB.
    
    Row data is loaded lazily from separate databases: data/lake/lancedb_rows/{dataset}/{table_id}
    
    Args:
        db: LanceDB connection
        store_name: Name of the table store
        offset: Starting offset
        limit: Number of records to load
        processed_table_ids: Set of table IDs to skip (for resume)
        
    Returns:
        List of table records with parsed rows (excluding already-processed)
    """
    if processed_table_ids is None:
        processed_table_ids = set()
    
    try:
        table = db.open_table(store_name)
        
        # Extract dataset name from store_name (e.g., "adventure_works_tables_entries" -> "adventure_works")
        dataset_name = store_name.replace("_tables_entries", "")
        
        # Use search with limit and offset
        # Note: LanceDB doesn't have native offset, we use workaround
        records = table.search().limit(offset + limit).to_list()
        batch_records = records[offset:offset + limit]
        
        # Parse rows from JSON and filter out processed tables
        tables = []
        skipped_count = 0
        for record in batch_records:
            table_data = dict(record)
            table_id = table_data.get('table_id', '')
            
            # Skip if already processed
            if table_id in processed_table_ids:
                skipped_count += 1
                continue
            
            # Load row data from separate table (lazy loading)
            parsed_rows = _load_table_rows(db, dataset_name, table_id)
            
            # Fallback to sample_rows if row table doesn't exist (backward compatibility)
            if not parsed_rows:
                rows_field = table_data.get("all_rows") or table_data.get("sample_rows")
                if rows_field:
                    try:
                        parsed_rows = json.loads(rows_field)
                    except json.JSONDecodeError:
                        parsed_rows = []
            
            table_data["parsed_rows"] = parsed_rows
            
            # Ensure columns is a list
            columns = table_data.get("columns", [])
            if isinstance(columns, str):
                try:
                    table_data["columns"] = json.loads(columns)
                except:
                    table_data["columns"] = []
            
            tables.append(table_data)
        
        if skipped_count > 0:
            logger.info(f"  Skipped {skipped_count} already-processed tables in this batch")
        
        return tables
        
    except Exception as e:
        logger.error(f"Failed to load batch: {e}")
        return []


def _get_rows_db_path(dataset_name: str):
    """Get the path to the rows database for a dataset.
    
    Structure: data/lake/lancedb_rows/{dataset}/
    """
    from pathlib import Path
    # Get project root (source/../)
    source_dir = Path(__file__).resolve().parent.parent.parent.parent
    project_root = source_dir.parent
    return project_root / "data" / "lake" / "lancedb_rows" / dataset_name


def _sanitize_table_id(table_id: str) -> str:
    """Sanitize table_id for use as LanceDB table name.
    
    LanceDB table names can only contain alphanumeric characters, 
    underscores, hyphens, and periods.
    """
    import re
    sanitized = re.sub(r'[^a-zA-Z0-9_\-.]', '_', table_id)
    return sanitized


def _load_table_rows(db, dataset_name: str, table_id: str) -> List[List[Any]]:
    """
    Load row data from separate rows database.
    
    Row data is stored in: data/lake/lancedb_rows/{dataset}/{table_id}
    with dynamic columns: row_index, col_0, col_1, ..., col_N
    
    Args:
        db: Main LanceDB connection (not used, kept for compatibility)
        dataset_name: Name of the dataset
        table_id: ID of the table
        
    Returns:
        List of rows (each row is a list of cell values)
    """
    try:
        import lancedb
        
        rows_db_path = _get_rows_db_path(dataset_name)
        if not rows_db_path.exists():
            logger.debug(f"Rows database not found: {rows_db_path}")
            return []
        
        rows_db = lancedb.connect(str(rows_db_path))
        
        # Sanitize table_id for use as table name
        safe_table_id = _sanitize_table_id(table_id)
        
        row_table = rows_db.open_table(safe_table_id)
        
        # Load all rows
        records = row_table.search().limit(1000000).to_list()
        
        if not records:
            return []
        
        # Determine number of columns from first record (col_0, col_1, ...)
        first_record = records[0]
        col_keys = sorted(
            [k for k in first_record.keys() if k.startswith("col_")],
            key=lambda k: int(k.split("_")[1])  # Sort by column index
        )
        
        # Sort by row_index and extract column values
        records.sort(key=lambda r: r.get("row_index", 0))
        rows = []
        for record in records:
            row = [record.get(col_key, "") for col_key in col_keys]
            rows.append(row)
        
        return rows
        
    except Exception as e:
        # Table doesn't exist - return empty (backward compatibility)
        logger.debug(f"Row table not found for {table_id}: {e}")
        return []

