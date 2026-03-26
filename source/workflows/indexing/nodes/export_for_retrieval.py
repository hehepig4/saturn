"""
Node: Export For Retrieval

Exports Stage 4 summaries to LanceDB for vector retrieval.
Creates a structured table with separated components:
- table_description: Table-level semantic description
- column_descriptions: Column semantics (class, name, description)
- column_stats: Column statistics/readouts (values, ranges, distributions)
- column_types (for quick TBox filtering)

This enables:
1. Flexible combination of components for embedding (td, td_cd, td_cd_cs)
2. Quick TBox filtering without SPARQL
3. Separated storage for debugging and ablation studies
4. Independent control of description vs stats for retrieval

Index Naming Convention:
- {dataset}_retrieval_td: table_description only
- {dataset}_retrieval_td_cd: table_description + column_descriptions
- {dataset}_retrieval_td_cd_cs: all three fields (full)
"""

from typing import Dict, Any, List
import re
from datetime import datetime
from loguru import logger

from workflows.common.node_decorators import graph_node


def extract_column_types(table_summary) -> List[str]:
    """
    Extract list of primitive class types from table summary.
    
    Args:
        table_summary: TableSummary object or dict
        
    Returns:
        List of primitive class names (e.g., ["YearColumn", "NameColumn"])
    """
    column_metadata = getattr(table_summary, 'column_metadata', None)
    if column_metadata is None and isinstance(table_summary, dict):
        column_metadata = table_summary.get('column_metadata', [])
    
    if not column_metadata:
        return []
    
    types = []
    for col in column_metadata:
        pclass = col.get('primitive_class', '')
        # Clean up prefix
        clean_class = pclass.replace('upo:', '')
        if clean_class and clean_class not in types:
            types.append(clean_class)
    
    return types


def has_only_fallback_columns(column_types: List[str]) -> bool:
    """
    Check if all columns are fallback (generic Column class).
    
    Args:
        column_types: List of primitive class names
        
    Returns:
        True if all columns are 'Column' (fallback), False otherwise
    """
    if not column_types:
        return True
    
    for ct in column_types:
        # Clean class name
        clean = ct.replace('upo:', '')
        if clean not in ('Column', 'upo:Column'):
            return False
    
    return True


def clean_column_name_suffix(text: str) -> str:
    """
    Remove duplicate column index suffix (e.g., 'Organization:7' -> 'Organization').
    
    The :N suffix is added during ingestion to handle duplicate column headers,
    but should not appear in the final retrieval text.
    
    Args:
        text: Text potentially containing column names with :N suffix
        
    Returns:
        Text with :N suffixes removed from column names
    """
    # Pattern: word characters followed by :digit at word boundary
    # e.g., "Organization:7:" -> "Organization:"
    # e.g., "[Name] Regular Season:3:" -> "[Name] Regular Season:"
    return re.sub(r'(\w+):(\d+):', r'\1:', text)


def build_retrieval_record(
    table_summary,
) -> Dict[str, Any]:
    """
    Build a record for the retrieval export table.
    
    Args:
        table_summary: TableSummary object or dict
        
    Returns:
        Dict with retrieval record fields
    """
    # Handle both Pydantic model and dict
    if hasattr(table_summary, 'model_dump'):
        ts = table_summary.model_dump()
    elif isinstance(table_summary, dict):
        ts = table_summary
    else:
        ts = dict(table_summary)
    
    table_id = ts.get('table_id', '')
    table_description = ts.get('table_description', '')
    
    # Get blocked column descriptions (no stats)
    blocked_views = ts.get('blocked_views', {})
    column_descriptions = ts.get('blocked_column_narrations', '')
    if not column_descriptions:
        # Fallback to old 'table' field for backward compatibility
        column_descriptions = blocked_views.get('descriptions', '') or blocked_views.get('table', '')
    
    # Clean column name suffixes (e.g., 'Organization:7' -> 'Organization')
    column_descriptions = clean_column_name_suffix(column_descriptions)
    
    # Get blocked column stats (new field)
    column_stats = ts.get('blocked_column_stats', '')
    if not column_stats:
        column_stats = blocked_views.get('stats', '')
    
    # Clean column name suffixes in stats as well
    column_stats = clean_column_name_suffix(column_stats)
    
    # Extract types
    column_types = extract_column_types(table_summary)
    
    # Check fallback
    fallback_only = has_only_fallback_columns(column_types)
    
    return {
        "table_id": table_id,
        "table_description": table_description,
        "column_descriptions": column_descriptions,
        "column_stats": column_stats,
        "column_types": column_types,
        "has_fallback_only": fallback_only,
        "num_cols": ts.get('num_cols', 0),
        "created_at": datetime.now().isoformat(),
    }


@graph_node()
def export_for_retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Export summaries to LanceDB for vector retrieval.
    
    Creates table: {dataset}_table_summaries_retrieval
    
    Each row contains (3 text fields for flexible embedding):
    - table_id: Unique identifier
    - table_description: Table-level description for embedding
    - column_descriptions: Column semantics (class, name, desc) - no stats
    - column_stats: Column statistics/readouts - no descriptions
    - column_types: List for quick TBox filtering
    - has_fallback_only: Flag for UPO filtering
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with export info
    """
    import pandas as pd
    
    dataset_name = state.dataset_name
    table_summaries = state.table_summaries or []
    
    logger.info(f"  Dataset: {dataset_name}")
    logger.info(f"  Tables: {len(table_summaries)}")
    
    # Build records
    records = []
    tables_with_fallback_only = 0
    
    for ts in table_summaries:
        record = build_retrieval_record(ts)
        records.append(record)
        
        if record['has_fallback_only']:
            tables_with_fallback_only += 1
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Store in LanceDB
    from store.store_singleton import get_store
    store = get_store()
    
    table_name = f"{dataset_name}_table_summaries_retrieval"
    
    # Check if table exists and drop it
    try:
        store.db.drop_table(table_name)
        logger.debug(f"  Dropped existing table: {table_name}")
    except Exception:
        pass  # Table doesn't exist
    
    # Create new table (handle race condition)
    try:
        tbl = store.db.create_table(table_name, df)
    except ValueError as e:
        if "already exists" in str(e):
            # Another process created the table, drop and retry
            try:
                store.db.drop_table(table_name)
            except Exception:
                pass
            tbl = store.db.create_table(table_name, df)
        else:
            raise
    
    logger.info(f"  ✓ Created LanceDB table: {table_name}")
    logger.info(f"  ✓ Total records: {len(records)}")
    logger.info(f"  ✓ Tables with fallback only: {tables_with_fallback_only}")
    
    return {
        'retrieval_export_done': True,
        'retrieval_table_name': table_name,
        'retrieval_record_count': len(records),
        'tables_with_fallback_only': tables_with_fallback_only,
    }
