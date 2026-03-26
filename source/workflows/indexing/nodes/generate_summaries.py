"""
Node: Generate Summaries

Generates table-level summaries for embedding.
Simplified version that uses a unified table view without role-based grouping.
"""

from typing import Dict, Any, List
from loguru import logger

from workflows.common.node_decorators import graph_node
from utils.data_helpers import safe_parse_json_dict, safe_parse_json_list, extract_readout_from_column_summary
from ..state import TableSummary


# ============== Serialization Utilities ==============

def format_number(value) -> str:
    """Format number for display."""
    if value is None:
        return "?"
    try:
        value = float(value)
    except:
        return str(value)
    
    if abs(value) >= 1_000_000_000:
        return f"{value/1_000_000_000:.1f}B"
    elif abs(value) >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.1f}K"
    elif isinstance(value, float) and value != int(value):
        return f"{value:.2f}"
    else:
        return str(int(value))


def get_column_stats(summary: Dict) -> str:
    """Extract statistics from column summary."""
    parts = []
    
    # Unique count
    unique_count = summary.get('unique_count', 0)
    if unique_count > 0:
        parts.append(f"{unique_count} unique")
    
    # Numeric stats
    numeric_stats = safe_parse_json_dict(summary.get('numeric_stats'))
    if numeric_stats:
        if 'min' in numeric_stats and 'max' in numeric_stats:
            parts.append(f"range: {format_number(numeric_stats['min'])}-{format_number(numeric_stats['max'])}")
        if 'mean' in numeric_stats:
            parts.append(f"mean={format_number(numeric_stats['mean'])}")
    
    # Top values
    top_values = safe_parse_json_list(summary.get('top_values'))
    if top_values:
        cats = [v.get('value', str(v)) if isinstance(v, dict) else str(v) for v in top_values[:3]]
        parts.append(f"top: {', '.join(cats)}")
    
    return ', '.join(parts) if parts else ""


def serialize_column(
    col_summary: Dict,
    col_defined: Dict,
    level: int = 2,
    include_readout: bool = True,
) -> str:
    """
    Serialize a column for LM embedding (COMBINED format).
    
    Level 1: Flat (PNEUMA compatible)
    Level 2: Class-prefixed with stats (default)
    Level 3: Natural language
    
    Args:
        col_summary: Column summary dict (from Stage 2)
        col_defined: Column defined class dict (from Stage 3)
        level: Serialization level (1, 2, or 3)
        include_readout: Whether to include statistical readout from Stage 2
        
    Returns:
        Serialized column string for embedding
    """
    # Get primitive class, removing upo: prefix and 'Column' suffix for display
    raw_pclass = col_defined.get('primitive_class', '')
    pclass = raw_pclass.replace('upo:', '').replace('Column', '') if raw_pclass else ''
    desc = col_defined.get('description', '')
    col_name = col_defined.get('column_name', '')
    
    # Get statistics
    stats = get_column_stats(col_summary)
    
    # Extract readout from column_summary's data_property_values (Stage 2 output)
    readout = ""
    if include_readout:
        readout = extract_readout_from_column_summary(col_summary)
    
    # Use leaf class name for display
    class_display = pclass
    
    if level == 1:
        # PNEUMA compatible - include readout in description
        if readout:
            return f"{col_name}: {desc} [Stats: {readout}]"
        return f"{col_name}: {desc}"
    elif level == 2:
        # Class-prefixed with stats (recommended)
        parts = [f"[{class_display}] {col_name}: {desc}"]
        if stats:
            parts.append(f"({stats})")
        if readout:
            parts.append(f"[Stats: {readout}]")
        return " ".join(parts)
    else:
        # Natural language
        base = f"{col_name} ({stats}): {desc}" if stats else f"{col_name}: {desc}"
        if readout:
            return f"{base} Statistics: {readout}."
        return base


def serialize_column_description(
    col_summary: Dict,
    col_defined: Dict,
    level: int = 2,
) -> str:
    """
    Serialize column DESCRIPTION only (no stats/readout).
    
    Used for semantic matching on column semantics.
    Stats like unique count, range, top values are NOT included here.
    
    Args:
        col_summary: Column summary dict (from Stage 2) - unused but kept for API consistency
        col_defined: Column defined class dict (from Stage 3)
        level: Serialization level
        
    Returns:
        Description-only serialized string (no stats)
    """
    raw_pclass = col_defined.get('primitive_class', '')
    pclass = raw_pclass.replace('upo:', '').replace('Column', '') if raw_pclass else ''
    desc = col_defined.get('description', '')
    col_name = col_defined.get('column_name', '')
    
    # Note: stats are intentionally NOT included in descriptions
    # They belong in column_stats field for separate retrieval control
    
    # Use leaf class name for display
    class_display = pclass
    
    if level == 1:
        return f"{col_name}: {desc}"
    elif level == 2:
        return f"[{class_display}] {col_name}: {desc}"
    else:
        return f"{col_name}: {desc}"


def serialize_column_stats(
    col_summary: Dict,
    col_defined: Dict,
) -> str:
    """
    Serialize column STATS/READOUT only (no description).
    
    Used for value-based filtering and verification.
    
    Args:
        col_summary: Column summary dict (from Stage 2)
        col_defined: Column defined class dict (from Stage 3)
        
    Returns:
        Stats-only serialized string, or empty if no stats
    """
    col_name = col_defined.get('column_name', '')
    readout = extract_readout_from_column_summary(col_summary)
    
    if readout:
        return f"{col_name}: {readout}"
    return ""


# ============== View Generation ==============

def generate_table_view(
    table_class: Dict,
    column_classes: List[Dict],
    column_summaries: Dict[str, Dict],
    level: int = 2,
) -> Dict[str, str]:
    """
    Generate a unified table view.
    
    Args:
        table_class: Table class metadata
        column_classes: Column class definitions
        column_summaries: Column summary data by column name
        level: Serialization level
        
    Returns:
        Dict with 'table' view
    """
    table_name = table_class.get('class_name', 'Table')
    table_desc = table_class.get('description', '')
    
    # Serialize all columns
    cols_text = ' || '.join([
        serialize_column(
            column_summaries.get(col.get('column_name', ''), {}),
            col,
            level=level,
        )
        for col in column_classes
    ])
    
    views = {
        'table': f"[Table] {table_name}: {table_desc}\nColumns: {cols_text}",
    }
    
    return views


def generate_blocked_view(
    table_class: Dict,
    column_classes: List[Dict],
    column_summaries: Dict[str, Dict],
    level: int = 2,
) -> Dict[str, str]:
    """
    Generate blocked format for embedding.
    
    Now generates separate description and stats views:
    - 'table': Combined (description + stats) for backward compatibility
    - 'descriptions': Description only (semantic matching)
    - 'stats': Stats/readout only (value verification)
    
    Args:
        table_class: Table class metadata
        column_classes: Column class definitions
        column_summaries: Column summary data by column name
        level: Serialization level
        
    Returns:
        Dict with 'table', 'descriptions', and 'stats' blocked views
    """
    # Combined view (backward compatible)
    combined_parts = []
    desc_parts = []
    stats_parts = []
    
    for col in column_classes:
        col_name = col.get('column_name', '')
        col_sum = column_summaries.get(col_name, {})
        
        # Combined (original behavior)
        combined_parts.append(serialize_column(col_sum, col, level=level))
        
        # Description only
        desc_parts.append(serialize_column_description(col_sum, col, level=level))
        
        # Stats only (skip if empty)
        stats_str = serialize_column_stats(col_sum, col)
        if stats_str:
            stats_parts.append(stats_str)
    
    blocked = {
        'table': ' || '.join(combined_parts),
        'descriptions': ' || '.join(desc_parts),
        'stats': ' || '.join(stats_parts) if stats_parts else '',
    }
    
    return blocked


def generate_relationship_view(
    table_id: str,
    table_class: Dict,
    relationships: List[Dict],
    level: int = 2,
) -> Dict[str, str]:
    """
    Generate relationship-centric view for a table.
    
    Creates a view that describes semantic relationships between columns.
    
    Args:
        table_id: Table identifier
        table_class: Table defined class metadata
        relationships: List of relationships for this table
        level: Serialization level
        
    Returns:
        Dict with 'relationship' view and 'relationship_blocked' format
    """
    if not relationships:
        return {}
    
    table_name = table_class.get('class_name', 'Table')
    table_desc = table_class.get('description', '')
    
    # Format each relationship
    rel_parts = []
    for rel in relationships:
        source = rel.get('source_column', '')
        target = rel.get('target_column', '')
        prop = rel.get('property_name', '')
        # Support both field names: 'reasoning' (old) and 'description' (current)
        reasoning = rel.get('reasoning', '') or rel.get('description', '')
        
        if level == 1:
            # Simple format
            rel_parts.append(f"{source} → {target} ({prop})")
        elif level == 2:
            # Property-prefixed format
            rel_parts.append(f"[{prop}] {source} → {target}: {reasoning}")
        else:
            # Natural language
            rel_parts.append(f"The '{source}' column {prop.replace('has', 'has a ')} relationship with '{target}': {reasoning}")
    
    rel_text = ' || '.join(rel_parts)
    
    views = {
        'relationship': f"[Relationship View] {table_name}: {table_desc}\nRelationships: {rel_text}",
        'relationship_blocked': rel_text,
    }
    
    return views


# ============== Main Node ==============

@graph_node()
def generate_summaries_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate table-level summaries for all tables.
    
    Creates:
        - Table view (unified column serialization)
        - Relationship view (if relationships exist)
        - Blocked views for embedding
        - PNEUMA-compatible column narrations
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with generated summaries
    """
    table_classes = state.table_classes or []
    col_summaries_by_table = state.col_summaries_by_table or {}
    col_classes_by_table = state.col_classes_by_table or {}
    relationships_by_table = getattr(state, 'relationships_by_table', {}) or {}
    serialization_level = state.serialization_level
    
    logger.info(f"Generating summaries for {len(table_classes)} tables")
    logger.info(f"  Relationships available for {len(relationships_by_table)} tables")
    
    summaries = []
    tables_with_relationships = 0
    
    for tbl_row in table_classes:
        table_id = tbl_row.get('source_table', '')
        if not table_id:
            continue
        
        columns = col_classes_by_table.get(table_id, [])
        col_sums = col_summaries_by_table.get(table_id, {})
        table_rels = relationships_by_table.get(table_id, [])
        
        if not columns:
            continue
        
        # Generate table view
        views = generate_table_view(tbl_row, columns, col_sums, level=serialization_level)
        blocked = generate_blocked_view(tbl_row, columns, col_sums, level=serialization_level)
        
        # Generate relationship view if relationships exist
        if table_rels:
            tables_with_relationships += 1
            rel_views = generate_relationship_view(table_id, tbl_row, table_rels, level=serialization_level)
            views['relationship'] = rel_views.get('relationship', '')
            blocked['relationship'] = rel_views.get('relationship_blocked', '')
        
        # Build summary
        summary = TableSummary(
            table_id=table_id,
            table_class_name=tbl_row.get('class_name', ''),
            table_description=tbl_row.get('description', ''),
            table_summary=tbl_row.get('summary', ''),
            num_cols=len(columns),
            view_summaries=views,
            blocked_views=blocked,
            column_metadata=[
                {
                    "column_name": c.get('column_name', ''),
                    "primitive_class": c.get('primitive_class', ''),
                    "description": c.get('description', ''),
                }
                for c in columns
            ],
            column_narrations=[
                {"column_name": c.get('column_name', ''), "narration": c.get('description', '')}
                for c in columns
            ],
            blocked_column_narrations=blocked.get('descriptions', ''),  # Description only
            blocked_column_stats=blocked.get('stats', ''),  # Stats only
        )
        
        summaries.append(summary)
    
    logger.info(f"  ✓ Generated {len(summaries)} table summaries")
    logger.info(f"  ✓ {tables_with_relationships} tables have relationship views")
    
    return {
        'summaries_generated': True,
        'table_summaries': summaries,
    }
