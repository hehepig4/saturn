"""
Node: Aggregate Summaries

Aggregates all batch checkpoints into final output and stores to LanceDB.
"""

from typing import Dict, Any, List
from pathlib import Path
import json
from datetime import datetime
from loguru import logger

from workflows.common.node_decorators import graph_node
from workflows.population.state import (
    ColumnSummaryState, 
    ColumnSummary, 
    TableColumnSummaries,
)


@graph_node(node_type="finalization", on_error="continue")
def aggregate_summaries_node(state: ColumnSummaryState) -> Dict[str, Any]:
    """
    Aggregate all checkpoint files into final results.
    
    Loads all batch checkpoints and merges them into a single output.
    Also stores to LanceDB for downstream processing.
    
    Returns:
        Updated state with all_summaries and end_time
    """
    logger.info(f"Aggregating {len(state.checkpoint_paths)} checkpoint files...")
    
    all_summaries = {}
    
    for checkpoint_path in state.checkpoint_paths:
        try:
            with open(checkpoint_path, 'r') as f:
                batch_data = json.load(f)
            
            for table_id, table_data in batch_data.get("summaries", {}).items():
                table_summaries = _deserialize_table_summaries(table_data)
                all_summaries[table_id] = table_summaries
                
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            state.errors.append({
                "type": "checkpoint_load_error",
                "path": checkpoint_path,
                "error": str(e)
            })
    
    # Save final aggregated output
    output_dir = Path(state.output_dir)
    final_path = output_dir / f"{state.dataset_name}_column_summaries_final.json"
    
    final_output = {
        "metadata": {
            "dataset_name": state.dataset_name,
            "total_tables": len(all_summaries),
            "total_columns_analyzed": state.total_columns_analyzed,
            "total_llm_calls": state.total_llm_calls,
            "code_reuse_count": state.code_reuse_count,
            "start_time": state.start_time.isoformat() if state.start_time else None,
            "end_time": datetime.now().isoformat(),
        },
        "statistics": _compute_statistics(all_summaries),
        "summaries": {
            table_id: _serialize_table_summaries(ts)
            for table_id, ts in all_summaries.items()
        }
    }
    
    with open(final_path, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    logger.info(f"Final output saved: {final_path}")
    logger.info(f"Total: {len(all_summaries)} tables, {state.total_columns_analyzed} columns")
    logger.info(f"LLM calls: {state.total_llm_calls}, Code reused: {state.code_reuse_count}")
    
    # Store to LanceDB for downstream Layer 2 processing
    _store_summaries_to_lancedb(state.dataset_name, all_summaries)
    
    return {
        "all_summaries": all_summaries,
        "end_time": datetime.now(),
    }


def _store_summaries_to_lancedb(dataset_name: str, all_summaries: Dict[str, TableColumnSummaries]) -> None:
    """
    Store column summaries to LanceDB for downstream processing.
    
    Creates a flat table with one row per column for easy querying.
    Uses merge mode to append new results without dropping existing data.
    """
    try:
        from store.store_singleton import get_store
        import pyarrow as pa
        
        store = get_store()
        table_name = f"{dataset_name}_column_summaries"
        
        # Build records
        records = []
        for table_id, table_summaries in all_summaries.items():
            for col_name, summary in table_summaries.columns.items():
                # Serialize data property values
                data_props_json = json.dumps([
                    {
                        "data_property_name": dp.data_property_name,
                        "range_type": dp.range_type,
                        "statistics": dp.statistics,
                        "transform_success_rate": dp.transform_success_rate,
                        "readout": dp.readout
                    }
                    for dp in (summary.data_property_values or [])
                ])
                
                records.append({
                    "table_id": table_id,
                    "document_title": table_summaries.document_title,
                    "section_title": table_summaries.section_title,
                    "column_name": summary.column_name,
                    "column_index": summary.column_index,
                    "is_virtual": summary.is_virtual,
                    "primitive_class": summary.primitive_class,
                    "null_ratio": summary.null_ratio,
                    "unique_ratio": summary.unique_ratio,
                    "total_count": summary.total_count,
                    "null_count": summary.null_count,
                    "unique_count": summary.unique_count,
                    "sample_values": json.dumps(summary.sample_values[:5] if summary.sample_values else []),
                    "data_property_values": data_props_json,
                    "execution_time_ms": summary.execution_time_ms or 0.0,
                    "error_message": summary.error_message or "",
                })
        
        if not records:
            logger.warning("No records to store to LanceDB")
            return
        
        # Create PyArrow table
        pa_table = pa.Table.from_pylist(records)
        
        # Check if table exists
        try:
            existing_table = store.db.open_table(table_name)
            # Table exists - merge new data
            # First, remove records with same table_ids to avoid duplicates
            existing_records = existing_table.search().limit(100000).to_list()
            new_table_ids = set(r['table_id'] for r in records)
            filtered_existing = [r for r in existing_records if r.get('table_id') not in new_table_ids]
            
            # Combine filtered existing + new records
            all_records = filtered_existing + records
            combined_table = pa.Table.from_pylist(all_records)
            
            # Drop and recreate with combined data
            # Use try/except to handle race condition
            store.db.drop_table(table_name)
            try:
                store.db.create_table(table_name, combined_table)
            except ValueError as ve:
                if "already exists" in str(ve):
                    tbl = store.db.open_table(table_name)
                    tbl.add(combined_table)
                else:
                    raise
            logger.info(f"✓ Merged {len(records)} new summaries with {len(filtered_existing)} existing to LanceDB: {table_name}")
            
        except FileNotFoundError:
            # Table doesn't exist - create new
            # Use try/except to handle race condition
            try:
                store.db.create_table(table_name, pa_table)
            except ValueError as ve:
                if "already exists" in str(ve):
                    tbl = store.db.open_table(table_name)
                    tbl.add(pa_table)
                else:
                    raise
            logger.info(f"✓ Created new table and stored {len(records)} column summaries to LanceDB: {table_name}")
        
    except Exception as e:
        logger.error(f"Failed to store summaries to LanceDB: {e}")


def _deserialize_table_summaries(data: Dict[str, Any]) -> TableColumnSummaries:
    """Reconstruct TableColumnSummaries from JSON data."""
    from workflows.population.state import DataPropertyValue
    
    columns = {}
    
    for col_name, col_data in data.get("columns", {}).items():
        # Deserialize data_property_values
        dpv_list = []
        for dpv_data in col_data.get("data_property_values", []):
            dpv_list.append(DataPropertyValue(
                data_property_name=dpv_data.get("data_property_name", ""),
                range_type=dpv_data.get("range_type", "xsd:string"),
                statistics=dpv_data.get("statistics", {}),
                transform_pattern=dpv_data.get("transform_pattern"),
                transform_success_rate=dpv_data.get("transform_success_rate", 0.0),
                readout=dpv_data.get("readout"),
            ))
        
        columns[col_name] = ColumnSummary(
            column_name=col_data.get("column_name", col_name),
            column_index=col_data.get("column_index", 0),
            is_virtual=col_data.get("is_virtual", False),
            primitive_class=col_data.get("primitive_class", ""),  # Empty if not classified
            data_property_values=dpv_list,
            total_count=col_data.get("total_count", 0),
            null_count=col_data.get("null_count", 0),
            unique_count=col_data.get("unique_count", 0),
            null_ratio=col_data.get("null_ratio", 0.0),
            unique_ratio=col_data.get("unique_ratio", 0.0),
            sample_values=col_data.get("sample_values", []),
            execution_time_ms=col_data.get("execution_time_ms"),
            error_message=col_data.get("error_message"),
        )
    
    return TableColumnSummaries(
        table_id=data.get("table_id", ""),
        document_title=data.get("document_title", ""),
        section_title=data.get("section_title", ""),
        columns=columns,
        total_columns=data.get("total_columns", 0),
        successful_columns=data.get("successful_columns", 0),
        failed_columns=data.get("failed_columns", []),
    )


def _serialize_table_summaries(table_summaries: TableColumnSummaries) -> Dict[str, Any]:
    """Convert TableColumnSummaries to JSON-serializable dict."""
    return {
        "table_id": table_summaries.table_id,
        "document_title": table_summaries.document_title,
        "section_title": table_summaries.section_title,
        "total_columns": table_summaries.total_columns,
        "successful_columns": table_summaries.successful_columns,
        "failed_columns": table_summaries.failed_columns,
        "columns": {
            name: {
                "column_name": s.column_name,
                "column_index": s.column_index,
                "primitive_class": s.primitive_class,
                "data_property_values": [
                    {
                        "data_property_name": dpv.data_property_name,
                        "range_type": dpv.range_type,
                        "statistics": dpv.statistics,
                        "readout": dpv.readout,
                    }
                    for dpv in s.data_property_values
                ],
                "null_ratio": s.null_ratio,
                "unique_ratio": s.unique_ratio,
                "sample_values": s.sample_values[:5],  # Already sampled in analyze_columns
            }
            for name, s in table_summaries.columns.items()
        }
    }


def _compute_statistics(all_summaries: Dict[str, TableColumnSummaries]) -> Dict[str, Any]:
    """Compute aggregate statistics across all summaries."""
    class_counts = {}
    total_columns = 0
    
    for table_id, table_summaries in all_summaries.items():
        for col_name, summary in table_summaries.columns.items():
            total_columns += 1
            
            # Primitive class distribution
            class_name = summary.primitive_class
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    return {
        "total_columns": total_columns,
        "primitive_class_distribution": class_counts,
    }
