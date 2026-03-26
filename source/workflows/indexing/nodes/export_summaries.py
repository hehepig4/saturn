"""
Node: Export Summaries

Exports generated summaries to JSON and Parquet formats (PNEUMA-compatible format only).
"""

import json
from typing import Dict, Any
from datetime import datetime
from pathlib import Path
from loguru import logger

from workflows.common.node_decorators import graph_node
from ..state import SummarizationOutput


@graph_node()
def export_summaries_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Export summaries to JSON and Parquet files (PNEUMA format).
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with export paths
    """
    import pandas as pd
    
    dataset_name = state.dataset_name
    output_base = Path(state.output_base)
    table_summaries = state.table_summaries or []
    
    logger.info(f"Exporting {len(table_summaries)} summaries")
    
    # Create output directories
    output_base.mkdir(parents=True, exist_ok=True)
    pneuma_output = output_base / "pneuma_format"
    pneuma_output.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert to serializable dicts
    summaries_data = []
    for s in table_summaries:
        if hasattr(s, 'model_dump'):
            summaries_data.append(s.model_dump())
        elif isinstance(s, dict):
            summaries_data.append(s)
        else:
            summaries_data.append(dict(s))
    
    # Save PNEUMA compatible format
    pneuma_summaries = []
    tables_with_relationships = 0
    for s in summaries_data:
        blocked_views = s.get("blocked_views", {})
        has_relationship = bool(blocked_views.get("relationship", ""))
        if has_relationship:
            tables_with_relationships += 1
        
        pneuma_summaries.append({
            "table_id": s.get("table_id", ""),
            "num_cols": s.get("num_cols", 0),
            "column_narrations": s.get("column_narrations", []),
            "blocked_column_narrations": s.get("blocked_column_narrations", ""),
            # Add view-specific blocked for extended comparison
            "blocked_key": blocked_views.get("key", ""),
            "blocked_temporal": blocked_views.get("temporal", ""),
            "blocked_measure": blocked_views.get("measure", ""),
            "blocked_attribute": blocked_views.get("attribute", ""),
            "blocked_core": blocked_views.get("core", ""),
            # Add relationship view
            "blocked_relationship": blocked_views.get("relationship", ""),
        })
    
    pneuma_output_data = {
        "dataset": dataset_name,
        "timestamp": timestamp,
        "summaries": pneuma_summaries,
    }
    
    pneuma_json_path = str(pneuma_output / f"{dataset_name}_summaries.json")
    with open(pneuma_json_path, 'w') as f:
        json.dump(pneuma_output_data, f, indent=2, ensure_ascii=False)
    logger.debug(f"  ✓ Saved PNEUMA-compatible summaries: {pneuma_json_path}")
    logger.debug(f"    ({tables_with_relationships} tables have relationship views)")
    
    pneuma_df = pd.DataFrame(pneuma_summaries)
    pneuma_parquet_path = str(pneuma_output / f"{dataset_name}_summaries.parquet")
    pneuma_df.to_parquet(pneuma_parquet_path)
    logger.debug(f"  ✓ Saved PNEUMA parquet: {pneuma_parquet_path}")
    
    # Create output object
    output = SummarizationOutput(
        dataset_name=dataset_name,
        timestamp=timestamp,
        num_summaries=len(table_summaries),
        summaries=table_summaries,
        pneuma_json_path=pneuma_json_path,
        pneuma_parquet_path=pneuma_parquet_path,
    )
    
    return {
        'export_done': True,
        'output': output,
        'success': True,
    }
