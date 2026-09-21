"""
Node 2: Annotate Tables

Uses LLM to enhance column annotations with semantic information.

NOTE: primitive_class is now pre-assigned by Stage 2 (Column Summary Agent).
This node only adds:
- P0: description (rdfs:comment) for each column
- table-level metadata (class_name, summary)

Implementation notes:
- Uses numeric string field names ("0", "1", ...) for minimal output tokens
- For high-column tables (>64 columns), uses batched annotation with aggregation
"""

import json
from typing import Dict, Any, List
from loguru import logger

from config.truncation_limits import TruncationLimits
from workflows.common.node_decorators import graph_node
from workflows.indexing.annotation.state import (
    TableDiscoveryLayer2State,
    TableAnnotation,
    ColumnAnnotation,
)
from utils.data_helpers import safe_parse_json_list

# Global flag for one-time prompt logging
_PROMPT_LOGGED = False
_AGG_PROMPT_LOGGED = False


# ============== Prompts ==============

# Prompt for single-batch mode (tables with <= MAX_COLUMNS_PER_LLM_CALL columns)
LAYER2_ANNOTATION_PROMPT = """Annotate table columns with brief descriptions.

## Table
**ID**: {table_id}
**Context**: {document_title} / {section_title}
**Size**: {num_rows} rows × {num_cols} columns

## Columns
{column_summaries}

## Guidelines
- Write the description content directly. Do not prefix with the column name (the name is already shown above).
- Include 2-3 representative values in the description if available.
- High null ratio does not indicate noise. Metadata columns (notes, comments, optional attributes) are often sparse but semantically important.

Output format: field "0", "1", ... maps to column index (description for that column).
Also provide table_class_name (CamelCase) and table_summary."""

# Prompt for batched mode - per batch (generates partial_summary instead of final)
LAYER2_BATCH_ANNOTATION_PROMPT = """Annotate table columns with brief descriptions.

## Table
**ID**: {table_id}
**Context**: {document_title} / {section_title}
**Size**: {num_rows} rows × {num_cols} columns
**Processing**: Batch {batch_num}/{total_batches} (columns {col_start}-{col_end} of {num_cols} total)

## Columns in This Batch
{column_summaries}

## Guidelines
- Write the description content directly. Do not prefix with the column name (the name is already shown above).
- Include 2-3 representative values in the description if available.
- High null ratio does not indicate noise. Metadata columns (notes, comments, optional attributes) are often sparse but semantically important.

Output format: field "0", "1", ... maps to column index (description for that column).
Also provide partial_summary describing what these columns represent together."""

# Prompt for aggregation step - combines all batch summaries
LAYER2_AGGREGATE_PROMPT = """Generate final table metadata by combining partial summaries.

## Table
**ID**: {table_id}
**Context**: {document_title} / {section_title}
**Size**: {num_rows} rows × {num_cols} columns

## Partial Summaries from Column Batches
{partial_summaries}

Based on the partial summaries above, provide:
1. A CamelCase class name that represents this table (e.g., SalesTransaction, CustomerAddress)
2. A comprehensive summary of what this table contains"""


@graph_node(node_type="processing", warn_threshold_seconds=120.0)
def annotate_tables_node(state: TableDiscoveryLayer2State) -> Dict[str, Any]:
    """
    Annotate current batch of tables with Layer 1 primitive classes.
    
    For each table in current batch:
        1. Format column summaries and primitives for prompt
        2. Call LLM for structured annotation
        3. Validate primitive classes (P1-c)
        4. Track corrections and statistics
    
    Uses parallel execution for efficiency.
    
    Returns:
        State updates with annotations for current batch
    """
    from llm.parallel import invoke_llm_parallel_with_func

    logger.info("=" * 70)
    logger.info(f"Node 2: Annotating Batch {state.current_batch_index + 1}/{state.total_batches}")
    logger.info("=" * 70)
    
    # Use current_batch_tables instead of all tables
    tables = state.current_batch_tables
    column_summaries = state.column_summaries
    primitive_classes = state.primitive_classes
    layer1_class_names = set(state.layer1_class_names)
    llm_max_workers = getattr(state, 'llm_max_workers', 128)
    
    logger.info(f"  Tables in batch: {len(tables)}")
    logger.info(f"  Parallel workers: {llm_max_workers}")
    
    # Prepare annotation function
    def annotate_single_table(task: Dict) -> Dict:
        """Annotate a single table (for parallel execution)."""
        table = task['table']
        summaries = task['summaries']
        table_id = task['table_id']
        
        try:
            annotation = _annotate_table_llm(table, summaries, primitive_classes)
            
            return {
                'success': True,
                'table_id': table_id,
                'annotation': annotation,
                'summaries': summaries,
            }
        except Exception as e:
            return {
                'success': False,
                'table_id': table_id,
                'error': str(e),
            }
    
    # Build task list
    tasks = []
    for i, table in enumerate(tables):
        table_id = table.get('table_id', f'table_{i}')
        summaries = column_summaries.get(table_id, [])
        tasks.append({
            'table': table,
            'summaries': summaries,
            'table_id': table_id,
            'doc_title': table.get('document_title', ''),
        })
    
    logger.info(f"  Processing {len(tasks)} tables in parallel...")
    
    # Parallel annotation
    max_workers = min(llm_max_workers, len(tasks))
    results = invoke_llm_parallel_with_func(
        annotate_single_table,
        tasks,
        max_workers=max_workers,
        show_progress=True,
    )
    
    # Process results
    annotations = []
    annotation_errors = []
    
    for i, result in enumerate(results):
        task = tasks[i]
        table_id = task['table_id']
        
        if isinstance(result, Exception):
            logger.error(f"  ✗ {table_id}: {result}")
            annotation_errors.append({'table_id': table_id, 'error': str(result)})
            continue
        
        if not result.get('success'):
            logger.error(f"  ✗ {table_id}: {result.get('error')}")
            annotation_errors.append({'table_id': table_id, 'error': result.get('error')})
            continue
        
        annotation = result['annotation']
        annotations.append(annotation)
        
        logger.debug(f"  ✓ {table_id}: {len(annotation.column_annotations)} columns → {annotation.table_class_name}")
    
    logger.info("")
    logger.info(f"  Annotation complete: {len(annotations)} succeeded, {len(annotation_errors)} failed")
    
    return {
        "annotation_done": True,
        "current_batch_annotations": annotations,  # Save to current_batch
        "annotation_errors": annotation_errors,
    }


def _create_dynamic_annotation_model(num_columns: int, is_batch: bool = False):
    """
    Create a Pydantic model with one description field per column.
    
    Uses numeric string field names: "0", "1", "2", ... for minimal output tokens.
    
    This approach leverages SGLang's guided decoding:
    - Each field is required → all columns must be annotated
    - Numeric field names → fewer output tokens
    - Simpler FSM → faster decoding
    
    Args:
        num_columns: Number of columns in the table (or batch)
        is_batch: If True, generates partial_summary instead of final table metadata
        
    Returns:
        DynamicModel class
    """
    from pydantic import create_model, Field
    
    # Build fields dict with numeric string names ("0", "1", "2", ...)
    # Note: table_id is NOT included - it's passed through from input, not generated by LLM
    if is_batch:
        # Batch mode: only partial_summary, no class_name
        fields = {
            'partial_summary': (str, ...),  # Brief summary of columns in this batch
        }
    else:
        # Single-call mode: full metadata
        fields = {
            'table_class_name': (str, ...),  # CamelCase class name for the table
            'table_summary': (str, ...),  # Brief table summary
        }
    
    # Add one description field per column with numeric names
    for idx in range(num_columns):
        field_name = str(idx)  # "0", "1", "2", ...
        fields[field_name] = (str, "")  # Brief description for column at index idx
    
    # Create dynamic model
    DynamicAnnotationModel = create_model(
        'TableAnnotation',
        **fields
    )
    
    return DynamicAnnotationModel


def _create_aggregate_model():
    """Create a Pydantic model for table metadata aggregation."""
    from pydantic import create_model
    
    return create_model(
        'TableAggregation',
        table_class_name=(str, ...),  # CamelCase class name
        table_summary=(str, ...),  # Comprehensive table summary
    )


def _compute_even_batch_sizes(total: int, num_batches: int) -> List[int]:
    """
    Compute evenly distributed batch sizes.
    
    Example: total=129, num_batches=2 → [65, 64]
    Example: total=130, num_batches=3 → [44, 43, 43]
    
    Args:
        total: Total number of items
        num_batches: Number of batches
        
    Returns:
        List of batch sizes that sum to total
    """
    base_size = total // num_batches
    remainder = total % num_batches
    
    # First 'remainder' batches get one extra item
    return [base_size + (1 if i < remainder else 0) for i in range(num_batches)]


def _annotate_table_llm(
    table: Dict,
    column_summaries: List[Dict],
    primitive_classes: List[Dict],
) -> TableAnnotation:
    """
    Call LLM to add semantic annotations to a table.
    
    For high-column tables (>MAX_COLUMNS_PER_LLM_CALL), uses batched mode:
    1. Process columns in batches, each generating partial_summary
    2. Aggregate partial summaries to get final table_class_name and table_summary
    
    Note: primitive_class is pre-assigned by Stage 2. This function only adds:
    - description (column-level)
    - table-level metadata (class_name, summary)
    """
    from llm.manager import get_llm_by_purpose
    from llm.invoke_with_stats import invoke_structured_llm_with_retry
    
    num_columns = len(column_summaries)
    max_cols_per_batch = TruncationLimits.MAX_COLUMNS_PER_LLM_CALL
    
    # Get table metadata once
    table_id = table.get('table_id', '')
    document_title = table.get('document_title', '')
    section_title = table.get('section_title', '')
    num_rows = table.get('row_count', 0)
    num_cols = table.get('column_count', num_columns)
    
    # LLM factory
    def llm_factory(temperature: float):
        return get_llm_by_purpose("default", temperature_override=temperature)
    
    global _PROMPT_LOGGED, _AGG_PROMPT_LOGGED
    
    if num_columns <= max_cols_per_batch:
        # Single-call mode: original logic
        return _annotate_table_single_call(
            table, column_summaries, llm_factory,
            table_id, document_title, section_title, num_rows, num_cols
        )
    else:
        # Batched mode for high-column tables
        return _annotate_table_batched(
            table, column_summaries, llm_factory,
            table_id, document_title, section_title, num_rows, num_cols,
            max_cols_per_batch
        )


def _annotate_table_single_call(
    table: Dict,
    column_summaries: List[Dict],
    llm_factory,
    table_id: str,
    document_title: str,
    section_title: str,
    num_rows: int,
    num_cols: int,
) -> TableAnnotation:
    """Single LLM call for tables with <= MAX_COLUMNS_PER_LLM_CALL columns."""
    from llm.invoke_with_stats import invoke_structured_llm_with_retry
    
    global _PROMPT_LOGGED
    
    num_columns = len(column_summaries)
    DynamicAnnotationModel = _create_dynamic_annotation_model(num_columns, is_batch=False)
    
    summaries_str = _format_column_summaries(column_summaries)
    
    prompt = LAYER2_ANNOTATION_PROMPT.format(
        table_id=table_id,
        document_title=document_title,
        section_title=section_title,
        num_rows=num_rows,
        num_cols=num_cols,
        column_summaries=summaries_str,
    )
    
    if not _PROMPT_LOGGED:
        logger.debug(f"=== Layer2 Annotation Prompt Sample ===\n{prompt}\n{'=' * 50}")
        _PROMPT_LOGGED = True
    
    result = invoke_structured_llm_with_retry(
        llm_factory=llm_factory,
        output_schema=DynamicAnnotationModel,
        prompt=prompt,
        max_retries=3,
    )
    
    # Extract column descriptions (numeric field names: "0", "1", ...)
    column_annotations = []
    for idx in range(num_columns):
        field_name = str(idx)  # "0", "1", "2", ...
        description = getattr(result, field_name, "")
        column_annotations.append(
            ColumnAnnotation(
                column_index=idx,
                description=description,
            )
        )
    
    return TableAnnotation(
        table_id=table_id,
        column_annotations=column_annotations,
        table_class_name=result.table_class_name,
        table_summary=result.table_summary,
    )


def _annotate_table_batched(
    table: Dict,
    column_summaries: List[Dict],
    llm_factory,
    table_id: str,
    document_title: str,
    section_title: str,
    num_rows: int,
    num_cols: int,
    max_cols_per_batch: int,
) -> TableAnnotation:
    """Batched annotation for high-column tables with aggregation step."""
    from llm.invoke_with_stats import invoke_structured_llm_with_retry
    
    global _PROMPT_LOGGED, _AGG_PROMPT_LOGGED
    
    num_columns = len(column_summaries)
    
    # Compute even batch sizes
    num_batches = (num_columns + max_cols_per_batch - 1) // max_cols_per_batch
    batch_sizes = _compute_even_batch_sizes(num_columns, num_batches)
    
    logger.debug(f"  High column count ({num_columns}), processing in {num_batches} batches: {batch_sizes}")
    
    # Collect results from all batches
    all_column_annotations = []
    partial_summaries = []
    
    batch_start = 0
    for batch_idx, batch_size in enumerate(batch_sizes):
        batch_end = batch_start + batch_size
        batch_summaries = column_summaries[batch_start:batch_end]
        
        # Create model for this batch
        DynamicBatchModel = _create_dynamic_annotation_model(batch_size, is_batch=True)
        
        # Format summaries for this batch (renumber to 0-based within batch)
        summaries_str = _format_column_summaries_batch(batch_summaries, batch_start)
        
        prompt = LAYER2_BATCH_ANNOTATION_PROMPT.format(
            table_id=table_id,
            document_title=document_title,
            section_title=section_title,
            num_rows=num_rows,
            num_cols=num_cols,
            batch_num=batch_idx + 1,
            total_batches=num_batches,
            col_start=batch_start,
            col_end=batch_end - 1,
            column_summaries=summaries_str,
        )
        
        if not _PROMPT_LOGGED:
            logger.debug(f"=== Layer2 Batch Annotation Prompt Sample ===\n{prompt}\n{'=' * 50}")
            _PROMPT_LOGGED = True
        
        result = invoke_structured_llm_with_retry(
            llm_factory=llm_factory,
            output_schema=DynamicBatchModel,
            prompt=prompt,
            max_retries=3,
        )
        
        # Extract column descriptions (numeric field names: "0", "1", ..., map back to global indices)
        for local_idx in range(batch_size):
            global_idx = batch_start + local_idx
            field_name = str(local_idx)  # "0", "1", "2", ...
            description = getattr(result, field_name, "")
            all_column_annotations.append(
                ColumnAnnotation(
                    column_index=global_idx,
                    description=description,
                )
            )
        
        # Collect partial summary
        partial_summaries.append(f"Batch {batch_idx + 1} (columns {batch_start}-{batch_end - 1}): {result.partial_summary}")
        
        batch_start = batch_end
    
    # Aggregation step: combine partial summaries
    logger.info(f"  Aggregating {len(partial_summaries)} partial summaries for table {table_id}")
    
    AggregateModel = _create_aggregate_model()
    
    agg_prompt = LAYER2_AGGREGATE_PROMPT.format(
        table_id=table_id,
        document_title=document_title,
        section_title=section_title,
        num_rows=num_rows,
        num_cols=num_cols,
        partial_summaries="\n".join(partial_summaries),
    )
    
    if not _AGG_PROMPT_LOGGED:
        logger.debug(f"=== Layer2 Aggregate Prompt Sample ===\n{agg_prompt}\n{'=' * 50}")
        _AGG_PROMPT_LOGGED = True
    
    agg_result = invoke_structured_llm_with_retry(
        llm_factory=llm_factory,
        output_schema=AggregateModel,
        prompt=agg_prompt,
        max_retries=3,
    )
    
    return TableAnnotation(
        table_id=table_id,
        column_annotations=all_column_annotations,
        table_class_name=agg_result.table_class_name,
        table_summary=agg_result.table_summary,
    )


def _format_column_summaries_batch(summaries: List[Dict], global_offset: int) -> str:
    """
    Format column summaries for a batch, showing both global and local indices.
    
    Args:
        summaries: Column summaries for this batch
        global_offset: Starting global index for this batch
        
    Returns:
        Formatted string for prompt
    """
    lines = []
    for local_idx, s in enumerate(summaries):
        global_idx = global_offset + local_idx
        col_name = s.get('column_name', 'Unknown')
        
        primitive_class = s.get('primitive_class')
        if not primitive_class:
            raise ValueError(
                f"Missing primitive_class for column '{col_name}'. "
                f"Ensure Stage 2 (Column Summary) was completed successfully."
            )
        
        # Simplified format: local index maps to output field
        # Strip "Column" suffix from class name for brevity
        short_class = primitive_class[:-6] if primitive_class.endswith('Column') else primitive_class
        line = f"- **{local_idx}.** {col_name} | Class: **{short_class}**"
        
        # Get readouts
        data_property_values = s.get('data_property_values', [])
        if isinstance(data_property_values, str):
            try:
                data_property_values = safe_parse_json_list(data_property_values)
            except:
                data_property_values = []
        
        readouts = []
        for dpv in data_property_values:
            if isinstance(dpv, dict) and dpv.get('readout'):
                readouts.append(dpv['readout'])
        
        if readouts:
            combined_readout = "; ".join(readouts)
            line += f" | Summary: {combined_readout}"
        
        # Null ratio
        null_ratio = s.get('null_ratio')
        if null_ratio is not None:
            try:
                null_ratio_float = float(null_ratio)
                if null_ratio_float > 0.1:
                    line += f" | Nulls: {null_ratio_float:.0%}"
            except (ValueError, TypeError):
                pass
        
        # Fallback
        if not readouts:
            sample_values = safe_parse_json_list(s.get('sample_values'))
            if sample_values:
                from config.truncation_limits import truncate_sample_values
                samples_str = truncate_sample_values(sample_values, n=3)
                if samples_str:
                    line += f" | e.g., {samples_str}"
        
        lines.append(line)
    
    return "\n".join(lines) if lines else "(No column summaries available)"


def _format_column_summaries(summaries: List[Dict]) -> str:
    """
    Format column summaries for the annotation prompt.
    
    Uses:
    - column_index (0-based) for LLM to reference columns
    - primitive_class (from Stage 2 classification)
    - readout from data_property_values (already formatted description)
    - unique_ratio / null_ratio (for role inference: key vs noise)
    
    Removed:
    - inferred_type (replaced by primitive_class)
    - numeric_stats (replaced by readout from DataProperty template)
    - sample_values (included in readout template)
    """
    lines = []
    for idx, s in enumerate(summaries):
        col_name = s.get('column_name', 'Unknown')
        col_index = s.get('column_index', idx)  # Use stored index or enumerate index
        
        # Use pre-assigned primitive_class from Stage 2 (required, no fallback)
        primitive_class = s.get('primitive_class')
        if not primitive_class:
            raise ValueError(
                f"Missing primitive_class for column '{col_name}'. "
                f"Ensure Stage 2 (Column Summary) was completed successfully."
            )
        
        # Simplified format: index maps to output field
        # Strip "Column" suffix from class name for brevity
        short_class = primitive_class[:-6] if primitive_class.endswith('Column') else primitive_class
        line = f"- **{col_index}.** {col_name} | Class: **{short_class}**"
        
        # Get readouts from all data_property_values (multifacet)
        data_property_values = s.get('data_property_values', [])
        if isinstance(data_property_values, str):
            try:
                data_property_values = safe_parse_json_list(data_property_values)
            except:
                data_property_values = []
        
        # Collect all readouts from multifacet DPs
        readouts = []
        for dpv in data_property_values:
            if isinstance(dpv, dict) and dpv.get('readout'):
                readouts.append(dpv['readout'])
        
        if readouts:
            # Join multiple facet readouts with semicolon
            combined_readout = "; ".join(readouts)
            line += f" | Summary: {combined_readout}"
        
        # Null ratio - show if significant
        null_ratio = s.get('null_ratio')
        if null_ratio is not None:
            try:
                null_ratio_float = float(null_ratio)
                if null_ratio_float > 0.1:  # Only show if significant
                    line += f" | Nulls: {null_ratio_float:.0%}"
            except (ValueError, TypeError):
                pass
        
        # Fallback: if no readout, show sample_values
        if not readouts:
            sample_values = safe_parse_json_list(s.get('sample_values'))
            if sample_values:
                from config.truncation_limits import truncate_sample_values
                samples_str = truncate_sample_values(sample_values, n=3)
                if samples_str:
                    line += f" | e.g., {samples_str}"
        
        lines.append(line)
    
    return "\n".join(lines) if lines else "(No column summaries available)"
