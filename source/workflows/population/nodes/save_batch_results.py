"""
Node 4: Save Batch Results

Saves the current batch results to disk for checkpointing.
"""

from typing import Dict, Any, List
from pathlib import Path
import json
from datetime import datetime
from loguru import logger

from workflows.common.node_decorators import graph_node
from workflows.population.state import ColumnSummaryState, TableColumnSummaries
from llm.statistics import get_usage_stats


@graph_node(node_type="processing", on_error="continue")
def save_batch_results_node(state: ColumnSummaryState) -> Dict[str, Any]:
    """
    Save current batch results to checkpoint file.
    
    Creates a JSON file with all column summaries for the current batch.
    Enables resumption if processing is interrupted.
    Skips saving if batch is empty (all tables already processed).
    
    Returns:
        Updated state with checkpoint_paths and completed_tables count
    """
    # Skip checkpoint if no results (all tables were already processed)
    if not state.current_batch_results:
        logger.info(f"Batch {state.current_batch_index + 1} empty (all tables skipped), no checkpoint needed")
        
        # Just increment batch index and continue
        new_batch_index = state.current_batch_index + 1
        
        return {
            "current_batch_index": new_batch_index,
            "current_batch_tables": [],  # Clear for next batch
            "current_batch_results": {},  # Clear for next batch
        }
    
    logger.info(f"Saving batch {state.current_batch_index + 1} results...")
    
    # Create output directory
    output_dir = Path(state.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate checkpoint filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"{state.dataset_name}_batch_{state.current_batch_index:04d}_{timestamp}.json"
    checkpoint_path = output_dir / checkpoint_name
    
    # Convert results to serializable format
    batch_data = {
        "metadata": {
            "dataset_name": state.dataset_name,
            "batch_index": state.current_batch_index,
            "total_batches": state.total_batches,
            "tables_in_batch": len(state.current_batch_results),
            "timestamp": timestamp,
        },
        "summaries": {}
    }
    
    for table_id, table_summaries in state.current_batch_results.items():
        batch_data["summaries"][table_id] = _serialize_table_summaries(table_summaries)
    
    # Save to file
    with open(checkpoint_path, 'w') as f:
        json.dump(batch_data, f, indent=2)
    
    logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Update progress
    new_completed = state.completed_tables + len(state.current_batch_results)
    new_checkpoint_paths = state.checkpoint_paths + [str(checkpoint_path)]
    
    # Increment batch index for next iteration
    new_batch_index = state.current_batch_index + 1
    
    # ========== Record LLM Stats Timeline Snapshot ==========
    # Capture current LLM statistics for ablation experiments
    batch_table_ids = list(state.current_batch_results.keys())
    llm_stats = get_usage_stats()
    
    snapshot = {
        "batch_idx": state.current_batch_index,
        "total_tables": new_completed,
        "batch_table_count": len(batch_table_ids),
        "table_ids": batch_table_ids,
        "cumulative_by_caller": {
            caller: dict(stats) for caller, stats in llm_stats.get("by_caller", {}).items()
        },
        "cumulative_totals": {
            "total_requests": llm_stats.get("total_requests", 0),
            "total_tokens": llm_stats.get("total_tokens", 0),
            "total_input_tokens": llm_stats.get("total_input_tokens", 0),
            "total_output_tokens": llm_stats.get("total_output_tokens", 0),
        },
        "code_reuse_count": state.code_reuse_count,
    }
    new_timeline = state.llm_stats_timeline + [snapshot]
    logger.info(f"  LLM stats snapshot: {new_completed} tables, {llm_stats.get('total_requests', 0)} requests, {llm_stats.get('total_tokens', 0)} tokens")
    
    return {
        "checkpoint_paths": new_checkpoint_paths,
        "completed_tables": new_completed,
        "current_batch_index": new_batch_index,
        "current_batch_tables": [],  # Clear for next batch
        "current_batch_results": {},  # Clear for next batch
        "llm_stats_timeline": new_timeline,
    }


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
            name: _serialize_column_summary(summary)
            for name, summary in table_summaries.columns.items()
        }
    }


def _serialize_column_summary(summary) -> Dict[str, Any]:
    """Convert ColumnSummary to JSON-serializable dict."""
    return {
        "column_name": summary.column_name,
        "column_index": summary.column_index,
        "is_virtual": summary.is_virtual,
        "primitive_class": summary.primitive_class,
        "data_property_values": [
            {
                "data_property_name": dpv.data_property_name,
                "range_type": dpv.range_type,
                "statistics": dpv.statistics,
                "transform_pattern": dpv.transform_pattern,
                "transform_success_rate": dpv.transform_success_rate,
                "readout": dpv.readout,
            }
            for dpv in summary.data_property_values
        ],
        "total_count": summary.total_count,
        "null_count": summary.null_count,
        "unique_count": summary.unique_count,
        "null_ratio": summary.null_ratio,
        "unique_ratio": summary.unique_ratio,
        "sample_values": summary.sample_values,
        "execution_time_ms": summary.execution_time_ms,
        "error_message": summary.error_message,
    }
