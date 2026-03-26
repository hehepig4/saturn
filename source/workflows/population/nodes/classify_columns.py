"""
Node: Classify Columns

Uses LLM to classify each column into a Primitive Class from Layer 1 TBox.
This classification happens BEFORE code-based analysis to enable better code reuse.

Design rationale:
- Stage 2 now handles primitive_class assignment (moved from Stage 3)
- Stage 3 only adds descriptions and discovers properties
- Contract matching uses primitive_class for precise code reuse

Implementation notes:
- Uses numeric string field names ("0", "1", ...) for minimal output tokens
- Uses shortened class names (without "Column" suffix) in LLM output for fewer tokens
- Each field is required, ensuring all columns are classified without fallback
- Restores full class names after LLM response
"""

from typing import Dict, Any, List, Tuple, Optional
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from workflows.common.node_decorators import graph_node
from workflows.population.state import ColumnSummaryState

# Global flag for one-time prompt logging
_PROMPT_LOGGED = False


# ============== Dynamic Pydantic Models ==============

def _strip_column_suffix(class_name: str) -> str:
    """Remove 'Column' suffix from class name for shorter output tokens.
    
    Special case: 'Column' itself is preserved as 'Column' (not stripped to empty).
    """
    if class_name == 'Column':
        return 'Column'  # Keep as-is for the root class
    if class_name.endswith('Column'):
        return class_name[:-6]  # Remove 'Column' (6 chars)
    return class_name


def _restore_column_suffix(short_name: str, original_classes: List[str]) -> str:
    """Restore 'Column' suffix to match original class names."""
    # Direct match (already has Column suffix or is 'Column' itself)
    if short_name in original_classes:
        return short_name
    # Try adding Column suffix
    with_suffix = f"{short_name}Column"
    if with_suffix in original_classes:
        return with_suffix
    # Fallback: return as-is (shouldn't happen with valid Literal constraint)
    return short_name


def _create_dynamic_classification_model(
    num_columns: int,
    primitive_class_names: List[str],
) -> type:
    """
    Create a Pydantic model with one field per column for classification.
    
    Uses numeric string field names: "0", "1", "2", ...
    Uses shortened class names (without "Column" suffix) for fewer output tokens.
    
    This approach leverages SGLang's guided decoding:
    - Each field is required → all columns must be classified
    - Literal constraint → only valid class names allowed
    - Numeric field names → fewer output tokens
    - Shorter class names → fewer output tokens
    - Simpler FSM → faster decoding
    
    Args:
        num_columns: Number of columns in the table
        primitive_class_names: List of allowed primitive class names (with Column suffix)
        
    Returns:
        DynamicModel class
    """
    from pydantic import create_model, Field
    from typing import Literal
    
    if not primitive_class_names:
        raise ValueError("primitive_class_names cannot be empty")
    
    # Strip "Column" suffix for shorter output tokens
    short_class_names = [_strip_column_suffix(c) for c in primitive_class_names]
    # Remove duplicates while preserving order
    short_class_names = list(dict.fromkeys(short_class_names))
    
    # Create Literal type for valid (shortened) class names
    ClassLiteral = Literal[tuple(short_class_names)]
    
    # Build fields dict with numeric string names ("0", "1", "2", ...)
    fields = {}
    for idx in range(num_columns):
        field_name = str(idx)  # "0", "1", "2", ...
        fields[field_name] = (ClassLiteral, ...)
    
    # Create dynamic model
    DynamicClassificationModel = create_model(
        'ColumnClassifications',
        **fields
    )
    
    return DynamicClassificationModel


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


# ============== Prompt Template ==============
# NOTE: Static content (class hierarchy, guidelines) placed FIRST to maximize prefix cache hit rate
# Dynamic content (table context, columns) placed LAST

CLASSIFY_COLUMNS_PROMPT = """Classify each column into a Primitive Class.

## Class Hierarchy
{primitive_classes}

## Table
Document: {document_title}
Section: {section_title}
Headers: {headers}
Rows: {total_rows} (showing {sample_rows})

## Columns (with sample values)
{columns_info}

## Note
Columns marked with [METADATA] contain table-level context information extracted from document/section titles. They typically have sparse values (most cells empty) - this is expected behavior, not noise.

Output format: field "0", "1", ... maps to column index, value is class name without "Column" suffix (e.g., "Year" for YearColumn)."""


def _format_columns_for_prompt(
    headers: List[str],
    sample_rows: List[List[Any]],
    max_samples: int = 5,
    virtual_column_indices: Optional[set] = None,
    batch_start: int = 0,
    batch_end: Optional[int] = None,
) -> str:
    """Format column info for the classification prompt.
    
    Supports batch processing for high column count tables.
    When batch_start/batch_end are specified, only formats those columns
    with batch-relative field names (col_0, col_1, ...) while showing
    original indices for context.
    
    Args:
        headers: All column headers (full table)
        sample_rows: Sample rows for value extraction
        max_samples: Maximum number of sample values per column
        virtual_column_indices: Set of column indices that are virtual (metadata) columns
        batch_start: Start index (inclusive) for batch processing
        batch_end: End index (exclusive) for batch processing (None = len(headers))
    """
    from workflows.population.sampling_utils import sample_values
    
    if virtual_column_indices is None:
        virtual_column_indices = set()
    
    if batch_end is None:
        batch_end = len(headers)
    
    lines = []
    
    for batch_idx, original_idx in enumerate(range(batch_start, batch_end)):
        if original_idx >= len(headers):
            break
        header = headers[original_idx]
        
        # Extract all values for this column, then sample
        all_values = []
        for row in sample_rows:
            if original_idx < len(row):
                val = row[original_idx]
                if val is not None and str(val).strip():
                    all_values.append(str(val)[:50])  # Truncate long values
        
        # Random sample from all values (instead of just first N rows)
        sampled = sample_values(all_values, max_samples)
        samples_str = ", ".join(f'"{s}"' for s in sampled)
        
        # Mark virtual columns with [METADATA] tag
        metadata_tag = " [METADATA]" if original_idx in virtual_column_indices else ""
        
        # Use batch-relative index for field name mapping
        lines.append(f"- **{batch_idx}.** {header}{metadata_tag}: [{samples_str}]")
    
    return "\n".join(lines)


def _format_primitive_classes(classes: List[str], classes_full: List[Dict[str, Any]] = None) -> str:
    """
    Format primitive classes for prompt.
    Uses the unified tree_formatter utility.
    
    If classes_full is provided (with descriptions), format as hierarchy with descriptions.
    Otherwise, format as simple list.
    """
    if not classes_full:
        # Fallback: simple list format
        from utils.tree_formatter import format_class_list
        return format_class_list(classes)
    
    from utils.tree_formatter import format_class_hierarchy
    
    # Filter out non-column classes (e.g., Table) and normalize
    normalized = []
    for c in classes_full:
        name = c.get('name', '')
        # Skip Table class - it's not a column type
        if name in ('Table', 'upo:Table'):
            continue
        # Skip the Column root itself - it's implicit
        if name in ('Column', 'upo:Column'):
            continue
        normalized.append(c)
    
    return format_class_hierarchy(normalized, include_column_root=True, max_desc_length=None)


def _classify_table_columns(
    table: Dict[str, Any],
    primitive_classes: List[str],
    primitive_classes_full: List[Dict[str, Any]] = None,
    llm_purpose: str = "default",
) -> Dict[str, str]:
    """
    Classify columns for a single table using LLM with independent field schema.
    
    Implementation:
    - Creates one required field per column (col_0, col_1, ...)
    - Each field has Literal constraint for valid class names
    - SGLang's guided decoding ensures all fields are filled with valid values
    - For high column count tables, processes in batches to avoid context overflow
    
    Args:
        table: Table data with columns and parsed_rows (from load_tables_batch_node)
        primitive_classes: List of allowed primitive class names
        primitive_classes_full: Full class info with descriptions (optional)
        llm_purpose: LLM purpose key for model selection
    
    Returns:
        Dict mapping column_name to primitive_class
    """
    from config.truncation_limits import TruncationLimits
    
    # Use 'columns' from load_tables_batch_node output (fallback to 'headers')
    headers = table.get('columns', table.get('headers', []))
    if not headers:
        return {}
    
    num_columns = len(headers)
    
    # Use 'parsed_rows' from load_tables_batch_node output (fallback to 'rows')
    rows = table.get('parsed_rows', table.get('rows', []))
    document_title = table.get('document_title', '')
    section_title = table.get('section_title', '')
    
    # Get table size info for prompt
    total_rows = len(rows)
    max_samples = 5
    sample_rows_count = min(total_rows, max_samples)
    
    # Get virtual column indices for METADATA tagging
    virtual_column_indices = set(table.get('virtual_column_indices', []))
    
    # Get max columns per batch from config
    max_cols_per_batch = TruncationLimits.MAX_COLUMNS_PER_LLM_CALL
    
    # Collect index-based results from all batches
    result_by_index: Dict[int, str] = {}
    
    # Determine if batching is needed
    if num_columns <= max_cols_per_batch:
        # Single batch - original logic
        result_by_index = _classify_columns_batch(
            headers=headers,
            rows=rows,
            batch_start=0,
            batch_end=num_columns,
            primitive_classes=primitive_classes,
            primitive_classes_full=primitive_classes_full,
            llm_purpose=llm_purpose,
            document_title=document_title,
            section_title=section_title,
            total_rows=total_rows,
            sample_rows_count=sample_rows_count,
            virtual_column_indices=virtual_column_indices,
        )
    else:
        # Multiple batches for high column count tables
        # Use even distribution instead of max_cols_per_batch, max_cols_per_batch, remainder
        # e.g., 129 columns with max 64 → 2 batches of 65, 64 instead of 64, 64, 1
        num_batches = (num_columns + max_cols_per_batch - 1) // max_cols_per_batch
        batch_sizes = _compute_even_batch_sizes(num_columns, num_batches)
        
        logger.info(f"High column count ({num_columns}), processing in {num_batches} batches: {batch_sizes}")
        batch_start = 0
        for batch_idx, batch_size in enumerate(batch_sizes):
            batch_end = batch_start + batch_size
            logger.debug(f"  Batch {batch_idx + 1}/{num_batches}: columns {batch_start}-{batch_end-1} ({batch_size} columns)")
            batch_result = _classify_columns_batch(
                headers=headers,
                rows=rows,
                batch_start=batch_start,
                batch_end=batch_end,
                primitive_classes=primitive_classes,
                primitive_classes_full=primitive_classes_full,
                llm_purpose=llm_purpose,
                document_title=document_title,
                section_title=section_title,
                total_rows=total_rows,
                sample_rows_count=sample_rows_count,
                virtual_column_indices=virtual_column_indices,
            )
            result_by_index.update(batch_result)
            batch_start = batch_end
    
    # Convert index-based result to header-based result
    # For tables with duplicate column names, use format "header:index" for disambiguation
    result: Dict[str, str] = {}
    header_counts: Dict[str, int] = {}
    
    for idx, header in enumerate(headers):
        prim_class = result_by_index.get(idx)
        if prim_class is None:
            # This should not happen with independent fields, but just in case
            logger.warning(f"Missing classification for column {idx} ({header}), using fallback")
            prim_class = 'Column' if 'Column' in primitive_classes else primitive_classes[0]
        
        # Create unique key for duplicate headers
        count = header_counts.get(header, 0)
        header_counts[header] = count + 1
        
        if count == 0:
            # First occurrence - use header as key
            key = header
        else:
            # Duplicate - use header:index format
            key = f"{header}:{idx}"
        
        result[key] = prim_class
    
    return result


def _classify_columns_batch(
    headers: List[str],
    rows: List[List[Any]],
    batch_start: int,
    batch_end: int,
    primitive_classes: List[str],
    primitive_classes_full: List[Dict[str, Any]],
    llm_purpose: str,
    document_title: str,
    section_title: str,
    total_rows: int,
    sample_rows_count: int,
    virtual_column_indices: set,
) -> Dict[int, str]:
    """
    Classify a batch of columns using LLM.
    
    Uses batch-relative field names (col_0, col_1, ...) in the Schema,
    then maps results back to original column indices.
    
    Args:
        headers: All column headers (full table)
        rows: All rows (full table)
        batch_start: Start index (inclusive) in original headers
        batch_end: End index (exclusive) in original headers
        primitive_classes: List of allowed primitive class names
        primitive_classes_full: Full class info with descriptions
        llm_purpose: LLM purpose key
        document_title: Table's document title
        section_title: Table's section title
        total_rows: Total number of rows in table
        sample_rows_count: Number of sample rows shown
        virtual_column_indices: Set of virtual column indices
        
    Returns:
        Dict[original_index, primitive_class]
    """
    from llm.manager import get_llm_by_purpose
    from llm.invoke_with_stats import invoke_structured_llm_with_retry
    
    batch_size = batch_end - batch_start
    if batch_size <= 0:
        return {}
    
    # Format only this batch's columns for prompt
    # Uses batch-relative indexing (col_0, col_1, ...) with original index shown for context
    columns_info = _format_columns_for_prompt(
        headers, rows, max_samples=5,
        virtual_column_indices=virtual_column_indices,
        batch_start=batch_start,
        batch_end=batch_end,
    )
    classes_info = _format_primitive_classes(primitive_classes, primitive_classes_full)
    
    # Show batch info in headers list for multi-batch tables
    if batch_start > 0 or batch_end < len(headers):
        batch_headers = headers[batch_start:batch_end]
        headers_display = f"{batch_headers} (columns {batch_start}-{batch_end-1} of {len(headers)})"
    else:
        headers_display = str(headers)
    
    prompt = CLASSIFY_COLUMNS_PROMPT.format(
        document_title=document_title,
        section_title=section_title,
        headers=headers_display,
        total_rows=total_rows,
        sample_rows=sample_rows_count,
        columns_info=columns_info,
        primitive_classes=classes_info,
    )
    
    # Log prompt once for debugging (use global flag)
    global _PROMPT_LOGGED
    if not _PROMPT_LOGGED:
        logger.debug(f"=== Classify Columns Prompt Sample ===\n{prompt}\n{'=' * 50}")
        _PROMPT_LOGGED = True
    
    # Create dynamic model with batch_size fields ("0", "1", ...)
    # Uses shortened class names (without "Column" suffix) for fewer output tokens
    DynamicClassificationModel = _create_dynamic_classification_model(
        batch_size, primitive_classes
    )
    
    # Call LLM with retry mechanism
    # Note: invoke_structured_llm_with_retry now handles both parsing errors (temp increase)
    # and infra errors (timeout/rate limit with exponential backoff) internally
    def llm_factory(temperature: float):
        return get_llm_by_purpose(llm_purpose, temperature_override=temperature)
    
    fallback_class = 'Column' if 'Column' in primitive_classes else (primitive_classes[0] if primitive_classes else 'Column')
    
    try:
        output = invoke_structured_llm_with_retry(
            llm_factory=llm_factory,
            output_schema=DynamicClassificationModel,
            prompt=prompt,
            max_retries=3,
        )
        
        # Map numeric fields back to original indices and restore full class names
        # "0" → batch_start, "1" → batch_start + 1, ...
        result: Dict[int, str] = {}
        for batch_idx in range(batch_size):
            field_name = str(batch_idx)  # "0", "1", "2", ...
            short_class = getattr(output, field_name)
            # Restore "Column" suffix to match original class names
            prim_class = _restore_column_suffix(short_class, primitive_classes)
            original_idx = batch_start + batch_idx
            result[original_idx] = prim_class
        
        logger.debug(f"Batch [{batch_start}-{batch_end-1}] classifications: {len(result)} columns")
        return result
        
    except Exception as e:
        # All retries exhausted (including infra retries) - use fallback
        logger.warning(f"LLM classification failed for batch [{batch_start}-{batch_end-1}]: {e}. Using fallback class.")
        result = {}
        for batch_idx in range(batch_size):
            original_idx = batch_start + batch_idx
            result[original_idx] = fallback_class
        return result


@graph_node(node_type="processing", warn_threshold_seconds=180.0)
def classify_columns_node(state: ColumnSummaryState) -> Dict[str, Any]:
    """
    Classify all columns in current batch using LLM.
    
    This node runs BEFORE analyze_columns_node to:
    1. Assign primitive_class to each column
    2. Enable contract matching based on primitive_class
    3. Improve code reuse precision
    
    Returns:
        State update with column_classifications
        
    Raises:
        LLMServiceUnavailableError: When LLM service becomes unavailable
    """
    from llm.parallel import invoke_llm_parallel_with_func, LLMServiceUnavailableError
    
    logger.info("=" * 60)
    logger.info("Classifying columns with Primitive Classes")
    logger.info("=" * 60)
    
    tables = state.current_batch_tables
    primitive_classes = getattr(state, 'primitive_classes', [])
    primitive_classes_full = getattr(state, 'primitive_classes_full', [])
    
    if not primitive_classes:
        raise ValueError(
            "No primitive classes loaded from Stage 1. "
            "Stage 2 REQUIRES Stage 1 to be run first. "
            "Run: python demos/run_upo_pipeline.py --step primitive_tbox"
        )
    
    logger.info(f"  Tables to classify: {len(tables)}")
    logger.info(f"  Primitive classes: {len(primitive_classes)} (with descriptions: {len(primitive_classes_full) > 0})")
    
    # Get llm_purpose from state (default: "default")
    llm_purpose = getattr(state, 'llm_purpose', 'default')
    logger.info(f"  LLM Purpose: {llm_purpose}")
    
    # Prepare tasks for parallel execution
    def classify_task(task: Dict) -> Dict:
        table = task['table']
        table_id = task['table_id']
        task_llm_purpose = task.get('llm_purpose', 'default')
        try:
            classifications = _classify_table_columns(table, primitive_classes, primitive_classes_full, llm_purpose=task_llm_purpose)
            return {
                'table_id': table_id,
                'classifications': classifications,
                'success': True,
            }
        except Exception as e:
            logger.warning(f"Failed to classify {table_id}: {e}")
            return {
                'table_id': table_id,
                'classifications': {},
                'success': False,
                'error': str(e),
            }
    
    tasks = [
        {'table': t, 'table_id': t.get('table_id', f'table_{i}'), 'llm_purpose': llm_purpose}
        for i, t in enumerate(tables)
    ]
    
    # Get max_workers from state (default: 128)
    table_max_workers = getattr(state, 'table_max_workers', 128)
    
    # Execute in parallel with fail-fast on service unavailability
    try:
        results = invoke_llm_parallel_with_func(
            func=classify_task,
            tasks=tasks,
            max_workers=min(table_max_workers, len(tasks)),
            show_progress=True,
            max_consecutive_errors=10,
            max_error_rate=0.3,  # Abort if >30% errors
            fail_fast=True,
        )
    except LLMServiceUnavailableError as e:
        logger.error(f"LLM service unavailable during column classification: {e}")
        raise  # Propagate to stop pipeline
    
    # Aggregate results
    column_classifications = {}
    llm_calls = 0
    failed_count = 0
    
    for result in results:
        if isinstance(result, Exception):
            failed_count += 1
            continue
        if result.get('success'):
            table_id = result['table_id']
            column_classifications[table_id] = result['classifications']
            llm_calls += 1
        else:
            failed_count += 1
    
    # Final error rate check
    total = len(results)
    if total > 0 and failed_count / total > 0.3:
        error_msg = f"Too many classification failures: {failed_count}/{total} ({failed_count/total:.1%})"
        logger.error(error_msg)
        raise LLMServiceUnavailableError(error_msg)
    
    logger.info(f"  Classified {len(column_classifications)} tables")
    logger.info(f"  LLM calls: {llm_calls}")
    
    return {
        'column_classifications': column_classifications,
        'total_llm_calls': state.total_llm_calls + llm_calls,
    }
