"""
Analyze Columns Node

Main analysis node that processes columns using the multifacet DataProperty approach:

Flow for each column:
1. Get primitive_class from classification
2. Find all DataProperties whose domain includes this primitive_class
3. For each DataProperty:
   a. Find best matching TransformContract from repository
   b. If no match, generate new TransformContract with LLM
   c. Apply transform to get typed values
   d. Compute predefined statistics based on range_type
   e. Generate readout from template
4. Build ColumnSummary with all DataPropertyValues
"""

import time
import json
from typing import Any, Dict, List, Optional
from loguru import logger

from workflows.common.node_decorators import graph_node
from workflows.population.state import (
    ColumnSummaryState, 
    ColumnSummary, 
    TableColumnSummaries,
    DataPropertyValue,
)
from workflows.population.contract import (
    TransformContract,
    DataPropertySpec,
    find_applicable_data_properties,
    get_statistics_contract,
)
from workflows.population.transform_repository import TransformRepository, MIN_SUCCESS_RATE
from workflows.population.transform_generator import (
    TransformGenerator, 
    apply_transform,
    get_sample_errors,
    FailedAttempt,
)
from workflows.population.safe_regex import validate_regex_safety
from workflows.population.readout_generator import generate_readout

# Ensure statistics are registered
import workflows.population.statistics_functions  # noqa: F401

# Fallback readout template when numeric statistics are unavailable
FALLBACK_READOUT_TEMPLATE = "Sample values: {sample_values}"


def extract_columns(table: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extract column data from table record.
    
    Note: For tables with duplicate column names, uses "header:index" format
    to create unique keys (matching classify_columns.py behavior).
    """
    columns = {}
    
    headers = table.get('columns', [])
    if isinstance(headers, str):
        try:
            headers = json.loads(headers)
        except json.JSONDecodeError:
            headers = []
    
    if not headers:
        headers = table.get('headers', []) or table.get('column_names', [])
    
    data = table.get('parsed_rows', [])
    
    if not data:
        rows_str = table.get('all_rows') or table.get('sample_rows')
        if rows_str and isinstance(rows_str, str):
            try:
                data = json.loads(rows_str)
            except json.JSONDecodeError:
                data = []
        elif isinstance(rows_str, list):
            data = rows_str
    
    if not data:
        data = table.get('data', [])
        if not data:
            data_str = table.get('table_data')
            if data_str and isinstance(data_str, str):
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    pass
    
    if not headers:
        logger.warning(f"Table {table.get('table_id', 'unknown')} has no headers")
        return columns
    
    # Allow tables with headers but no data (empty tables)
    # They will have empty value lists but can still be classified by header names
    if not data:
        logger.debug(f"Table {table.get('table_id', 'unknown')} has headers but no data rows")
    
    # Track header occurrences to handle duplicates
    header_counts: Dict[str, int] = {}
    
    for idx, header in enumerate(headers):
        if not header:
            continue
        
        # Create unique key for duplicate headers (same logic as classify_columns.py)
        count = header_counts.get(header, 0)
        header_counts[header] = count + 1
        
        if count == 0:
            key = header
        else:
            key = f"{header}:{idx}"
        
        values = []
        for row in data:
            if isinstance(row, (list, tuple)) and idx < len(row):
                values.append(str(row[idx]) if row[idx] is not None else "")
            elif isinstance(row, dict):
                values.append(str(row.get(header, "")))
        columns[key] = values
    
    return columns


def process_data_property(
    column_values: List[str],
    primitive_class: str,
    data_property_spec: DataPropertySpec,
    repository: TransformRepository,
    generator: TransformGenerator,
    class_description: Optional[str] = None,
    max_retries: int = 3,
    sibling_properties: Optional[List[DataPropertySpec]] = None,
    table_id: Optional[str] = None,
    column_name: Optional[str] = None,
    sh_max_workers: int = 8,
    budget_cap: int = 10000,
    disable_reuse: bool = False,
) -> tuple[DataPropertyValue, bool, int]:
    """
    Process a single DataProperty for a column with retry mechanism.
    
    The retry mechanism ensures better contract generation by:
    1. Trying to find existing contract from repository (skipped if disable_reuse=True)
    2. If not found or score too low, generate new with LLM
    3. If generated contract fails, retry with feedback from failed attempts
    4. Keep the best contract across all attempts
    
    Args:
        sibling_properties: Other DataProperties for the same class (for context)
        table_id: Table identifier for BF indexing (if enabled)
        column_name: Column name for BF indexing (if enabled)
        sh_max_workers: Max parallel workers for Successive Halving evaluation
        budget_cap: Upper limit on Successive Halving budget
        disable_reuse: If True, skip repository lookup and always generate with LLM
    
    Returns:
        Tuple of (DataPropertyValue, used_llm, llm_calls)
    """
    used_llm = False
    llm_calls = 0
    
    # Count non-null values for error rate calculation
    non_null_count = sum(1 for v in column_values if v and str(v).strip())
    
    # Best contract tracking across all attempts
    best_contract: Optional[TransformContract] = None
    best_score: float = 0.0
    
    # Try to find existing contract from repository (skip if reuse disabled)
    unmatched_values = None
    if not disable_reuse:
        contract, unmatched_values = repository.find_best_contract(
            primitive_class=primitive_class,
            data_property=data_property_spec.name,
            column_values=column_values,
            sh_max_workers=sh_max_workers,
            budget_cap=budget_cap,
        )
        
        if contract is not None:
            # Compute score for existing contract (unified: match + transform success)
            score = contract.success_rate(column_values)
            if score >= MIN_SUCCESS_RATE:
                # Existing contract is good enough
                best_contract = contract
                best_score = score
            else:
                # Existing contract has low score - need to regenerate
                # Still keep it as fallback
                best_contract = contract
                best_score = score
    
    # Need to generate new contract if no good existing one
    if best_score < MIN_SUCCESS_RATE:
        failed_attempts: List[FailedAttempt] = []
        retry_count = 0
        
        while retry_count < max_retries and best_score < MIN_SUCCESS_RATE:
            llm_calls += 1
            used_llm = True
            
            # Generate contract (with feedback if we have failed attempts)
            if failed_attempts:
                gen_result = generator.generate_with_feedback(
                    primitive_class=primitive_class,
                    data_property_spec=data_property_spec,
                    sample_values=column_values,
                    class_description=class_description,
                    failed_attempts=failed_attempts,
                    sibling_properties=sibling_properties,
                )
            else:
                gen_result = generator.generate(
                    primitive_class=primitive_class,
                    data_property_spec=data_property_spec,
                    sample_values=column_values,
                    class_description=class_description,
                    unmatched_values=unmatched_values if unmatched_values else None,
                    sibling_properties=sibling_properties,
                )
            
            new_contract = gen_result.contract
            
            # Validate pattern compiles with RE2 (rejects unsupported features)
            if not validate_regex_safety(new_contract.pattern):
                logger.warning(
                    f"RE2 rejected pattern for {primitive_class}.{data_property_spec.name}: "
                    f"'{new_contract.pattern[:80]}'"
                )
                failed_attempts.append(FailedAttempt(
                    pattern=new_contract.pattern,
                    transform_expr=new_contract.transform_expr,
                    success_rate=0.0,
                    error_count=len(column_values),
                    sample_errors=[
                        f"INVALID RE2 PATTERN: `{new_contract.pattern}` was rejected by the RE2 regex engine. "
                        f"RE2 does NOT support: backreferences (\\1), lookahead (?=...), lookbehind (?<=...), "
                        f"possessive quantifiers (a++), or atomic groups (?>...). "
                        f"Use simple patterns with character classes: [0-9.]+, [^,]+, [A-Za-z ]+"
                    ],
                ))
                retry_count += 1
                continue
            
            # Test the new contract
            transformed, null_count, error_count = apply_transform(new_contract, column_values)
            # Use unified success_rate (match + transform success)
            score = new_contract.success_rate(column_values)
            
            # Check if this is better than previous best
            # Use >= to ensure we always keep at least the first attempt
            if score >= best_score or best_contract is None:
                best_contract = new_contract
                best_score = score
            
            # If good enough, we're done
            if score >= MIN_SUCCESS_RATE:
                break
            
            # Record failed attempt for next iteration's feedback
            sample_errors = get_sample_errors(new_contract, column_values, max_samples=5)
            failed_attempts.append(FailedAttempt(
                pattern=new_contract.pattern,
                transform_expr=new_contract.transform_expr,
                success_rate=score,
                error_count=error_count,
                sample_errors=sample_errors,
            ))
            
            retry_count += 1
        
        # Store the best contract found (quality gate: only stores if success_rate >= 85%)
        if best_contract:
            repository.store(best_contract, column_values=column_values)
    
    # Use fallback if we still don't have a contract
    if best_contract is None:
        logger.warning(
            f"No valid contract found for {primitive_class}.{data_property_spec.name}, "
            "using fallback"
        )
        best_contract = generator._get_fallback_contract(primitive_class, data_property_spec)
    
    # Apply the best contract
    transformed_values, null_count, error_count = apply_transform(best_contract, column_values)
    
    # Filter out None values before computing statistics
    # (transform may return None for values it couldn't process)
    valid_values = [v for v in transformed_values if v is not None]
    
    # Compute statistics
    stats_contract = get_statistics_contract(data_property_spec.range_type)
    if stats_contract and valid_values:
        try:
            statistics = stats_contract.compute(valid_values)
        except Exception as e:
            # Fallback if statistics computation fails
            logger.debug(f"Statistics computation failed for {data_property_spec.name}: {e}")
            statistics = {"count": len(valid_values)}
    else:
        statistics = {"count": len(valid_values)}
    
    # Add transform metadata to statistics
    statistics["null_count"] = null_count
    statistics["error_count"] = error_count
    # Use unified success_rate as the quality metric
    success_rate = best_contract.success_rate(column_values)
    statistics["transform_success_rate"] = success_rate
    
    # Add sample_values for fallback readout (random sampling to avoid position bias)
    # Use sample_for_llm_prompt which includes truncation to prevent overflow from binary data
    from workflows.population.sampling_utils import sample_for_llm_prompt
    non_null_values = [v for v in column_values if v and str(v).strip()]
    statistics["sample_values"] = sample_for_llm_prompt(non_null_values, n=5)
    
    # Generate readout with fallback for low match rate
    readout = None
    template_used = data_property_spec.readout_template
    
    if template_used:
        # If success_rate is too low, use fallback template for numeric properties
        # This avoids "N/A to N/A" readouts when numeric conversion fails
        # Note: xsd:float and xsd:double are EL-forbidden but may appear in legacy data
        NUMERIC_TYPES = (
            "xsd:integer", "xsd:decimal", "xsd:nonNegativeInteger",
            # EL-forbidden but may appear in input (will be converted to xsd:decimal)
            "xsd:float", "xsd:double", "xsd:positiveInteger"
        )
        if success_rate < MIN_SUCCESS_RATE and data_property_spec.range_type in NUMERIC_TYPES:
            template_used = FALLBACK_READOUT_TEMPLATE
        
        readout = generate_readout(template_used, statistics)
    
    # Build DataPropertyValue
    dpv = DataPropertyValue(
        data_property_name=data_property_spec.name,
        range_type=data_property_spec.range_type,
        statistics=statistics,
        contract_id=best_contract.contract_id,
        transform_pattern=best_contract.pattern,
        transform_success_rate=success_rate,
        readout=readout,
        readout_template=data_property_spec.readout_template,
    )
    
    return dpv, used_llm, llm_calls


def build_summary(
    column_name: str,
    column_index: int,
    values: List[str],
    primitive_class: str,
    data_property_values: List[DataPropertyValue],
    execution_time_ms: float = 0.0,
    error_message: Optional[str] = None,
    is_virtual: bool = False,
) -> ColumnSummary:
    """Build ColumnSummary from analysis results."""
    total_count = len(values)
    non_null = [v for v in values if v and str(v).strip()]
    null_count = total_count - len(non_null)
    unique_count = len(set(non_null)) if non_null else 0
    
    # For virtual columns, use unique values (they represent table-level metadata)
    # For normal columns, use random sampling to avoid position bias
    from workflows.population.sampling_utils import sample_values as do_sample
    if is_virtual:
        sample_values = list(set(str(v) for v in non_null))[:5]
    else:
        sample_values = [str(v) for v in do_sample(non_null, 5)]
    
    return ColumnSummary(
        column_name=column_name,
        column_index=column_index,
        is_virtual=is_virtual,
        primitive_class=primitive_class,
        data_property_values=data_property_values,
        total_count=total_count,
        null_count=null_count,
        unique_count=unique_count,
        null_ratio=null_count / total_count if total_count > 0 else 0.0,
        unique_ratio=unique_count / len(non_null) if non_null else 0.0,
        sample_values=sample_values,
        execution_time_ms=execution_time_ms,
        error_message=error_message,
    )


@graph_node(node_type="processing", enable_timing=True)
def analyze_columns_node(state: ColumnSummaryState) -> Dict[str, Any]:
    """
    Analyze columns using multifacet DataProperty approach (parallel per table).
    
    For each table (in parallel):
    1. Get primitive_class from classification
    2. Find all applicable DataProperties
    3. For each DataProperty, find/generate TransformContract and compute statistics
    4. Build ColumnSummary with all DataPropertyValues
    5. Save to LanceDB immediately
    
    Optimization: Uses a shared TransformRepository across all threads for:
    - Better cache hit rate (contracts discovered by one thread benefit others)
    - Batch writes to LanceDB (accumulated across all tables)
    """
    from llm.parallel import invoke_llm_parallel_with_func
    
    logger.info("=" * 70)
    logger.info("Analyzing columns (Multifacet DataProperty - Parallel)")
    logger.info("=" * 70)
    
    column_classifications = getattr(state, 'column_classifications', {})
    data_properties = getattr(state, 'data_properties', [])
    class_hierarchy = getattr(state, 'class_hierarchy', {})
    data_property_hierarchy = getattr(state, 'data_property_hierarchy', {})
    dataset_name = state.dataset_name
    budget_multiplier = getattr(state, 'budget_multiplier', 1.0)
    sh_max_workers = getattr(state, 'sh_max_workers', 8)  # Used for Successive Halving parallelism
    budget_cap = getattr(state, 'budget_cap', 10000)
    
    logger.info(f"  Successive Halving max workers: {sh_max_workers}")
    logger.info(f"  Successive Halving budget cap: {budget_cap}")
    
    # Shared repository across all threads
    shared_repository = TransformRepository(
        dataset_name=dataset_name,
        budget_multiplier=budget_multiplier,
    )
    # Pre-load cache for faster access across threads
    shared_repository._load_cache()
    
    # Define task function for parallel execution
    def analyze_table_task(task: Dict) -> Dict:
        """Analyze a single table (for parallel execution)."""
        import threading
        thread_id = threading.current_thread().name
        task_start_time = time.time()
        
        table = task['table']
        table_id = table.get('table_id', '')
        if not table_id:
            return {'success': False, 'error': 'no_table_id'}
        
        logger.debug(f"[{thread_id}] Starting table: {table_id[:50]}...")
        
        # Use shared repository (thread-safe), own generator per thread
        repository = shared_repository
        task_llm_purpose = task.get('llm_purpose', 'default')
        generator = TransformGenerator(llm_purpose=task_llm_purpose)
        
        table_classifications = column_classifications.get(table_id, {})
        columns = extract_columns(table)
        
        # Get virtual column indices (set by expand_virtual_columns_node)
        virtual_column_indices = set(table.get('virtual_column_indices', []))
        
        if not columns:
            return {'success': False, 'table_id': table_id, 'error': 'no_columns'}
        
        table_summaries: Dict[str, ColumnSummary] = {}
        failed_columns: List[str] = []
        table_llm_calls = 0
        table_reuse = 0
        table_errors = []
        
        for col_idx, (col_name, col_values) in enumerate(columns.items()):
            start_time = time.time()
            
            classification = table_classifications.get(col_name)
            if classification is None or not classification:
                raise ValueError(
                    f"Missing classification for column '{col_name}' in table '{table_id}'. "
                    f"Ensure classify_columns_node ran successfully before analyze_columns_node."
                )
            prim_class = classification
            
            # Check if this column is virtual
            is_virtual = col_idx in virtual_column_indices
            
            # Handle empty columns (tables with headers but no data rows)
            # Skip DataProperty processing, create minimal ColumnSummary with only classification
            if not col_values:
                elapsed = (time.time() - start_time) * 1000
                logger.debug(f"Column '{col_name}' has no data values, creating minimal summary")
                summary = build_summary(
                    column_name=col_name,
                    column_index=col_idx,
                    values=[],
                    primitive_class=prim_class,
                    data_property_values=[],  # No DataProperty processing for empty columns
                    execution_time_ms=elapsed,
                    is_virtual=is_virtual,
                )
                table_summaries[col_name] = summary
                continue
            
            try:
                applicable_props = find_applicable_data_properties(
                    prim_class, 
                    data_properties, 
                    class_hierarchy,
                    data_property_hierarchy,
                )
                
                if not applicable_props:
                    raise ValueError(
                        f"No DataProperty found for class '{prim_class}' and its ancestors. "
                        f"Please ensure the TBox defines DataProperties for this class hierarchy."
                    )
                
                dpv_list = []
                col_llm_calls = 0
                col_reuse = 0
                
                for prop_spec in applicable_props:
                    dpv, used_llm, calls = process_data_property(
                        column_values=col_values,
                        primitive_class=prim_class,
                        data_property_spec=prop_spec,
                        repository=repository,
                        generator=generator,
                        sibling_properties=applicable_props,
                        table_id=table_id,
                        column_name=col_name,
                        sh_max_workers=sh_max_workers,
                        budget_cap=budget_cap,
                        disable_reuse=task.get('disable_reuse', False),
                    )
                    dpv_list.append(dpv)
                    col_llm_calls += calls
                    if not used_llm:
                        col_reuse += 1
                
                table_llm_calls += col_llm_calls
                table_reuse += col_reuse
                
                elapsed = (time.time() - start_time) * 1000
                
                summary = build_summary(
                    column_name=col_name,
                    column_index=col_idx,
                    values=col_values,
                    primitive_class=prim_class,
                    data_property_values=dpv_list,
                    execution_time_ms=elapsed,
                    is_virtual=is_virtual,
                )
                table_summaries[col_name] = summary
                
            except Exception as e:
                elapsed = (time.time() - start_time) * 1000
                failed_columns.append(col_name)
                table_errors.append({
                    "table_id": table_id,
                    "column_name": col_name,
                    "error": str(e),
                })
                
                summary = build_summary(
                    column_name=col_name,
                    column_index=col_idx,
                    values=col_values,
                    primitive_class=prim_class,
                    data_property_values=[],
                    execution_time_ms=elapsed,
                    error_message=str(e),
                    is_virtual=is_virtual,
                )
                table_summaries[col_name] = summary
        
        table_summary_obj = TableColumnSummaries(
            table_id=table_id,
            document_title=table.get('document_title', ''),
            section_title=table.get('section_title', ''),
            columns=table_summaries,
            total_columns=len(columns),
            successful_columns=len(table_summaries) - len(failed_columns),
            failed_columns=failed_columns,
        )
        
        # NOTE: Do NOT save to LanceDB here - we'll batch save after all parallel tasks complete
        # This avoids lock contention and improves throughput
        
        task_elapsed = time.time() - task_start_time
        logger.debug(
            f"[{thread_id}] Completed table: {table_id[:30]} "
            f"({len(table_summaries)} cols, {table_llm_calls} LLM calls, {task_elapsed:.1f}s)"
        )
        
        return {
            'success': True,
            'table_id': table_id,
            'result': table_summary_obj,
            'columns_analyzed': len(table_summaries),
            'llm_calls': table_llm_calls,
            'reuse_count': table_reuse,
            'errors': table_errors,
        }
    
    # Build task list
    llm_purpose = getattr(state, 'llm_purpose', 'default')
    disable_reuse = getattr(state, 'disable_transform_reuse', False)
    logger.info(f"  LLM Purpose: {llm_purpose}")
    if disable_reuse:
        logger.info(f"  Transform Reuse: DISABLED (ablation mode)")
    tasks = [{'table': t, 'llm_purpose': llm_purpose, 'disable_reuse': disable_reuse} for t in state.current_batch_tables]
    
    # Get analyze_max_workers from state (default: 32)
    # This is separate from table_max_workers because analyze_columns has nested
    # Successive Halving parallelism (sh_max_workers), so we need lower parallelism here
    # to avoid thread explosion: total threads ≈ analyze_max_workers × sh_max_workers
    analyze_max_workers = getattr(state, 'analyze_max_workers', 32)

    logger.info(f"  Processing {len(tasks)} tables in parallel (analyze_max_workers={analyze_max_workers})")

    # Execute in parallel
    results_list = invoke_llm_parallel_with_func(
        func=analyze_table_task,
        tasks=tasks,
        max_workers=min(analyze_max_workers, len(tasks)),
        show_progress=True,
    )
    
    # Aggregate results
    results: Dict[str, TableColumnSummaries] = {}
    llm_calls = 0
    reuse_count = 0
    columns_analyzed = 0
    errors = []
    
    for result in results_list:
        if isinstance(result, Exception):
            errors.append({'type': 'task_exception', 'error': str(result)})
            continue
        if result.get('success'):
            table_id = result['table_id']
            results[table_id] = result['result']
            columns_analyzed += result['columns_analyzed']
            llm_calls += result['llm_calls']
            reuse_count += result['reuse_count']
            errors.extend(result.get('errors', []))
    
    # Batch save all table summaries to LanceDB (single write operation)
    if results:
        saved_count = _save_all_tables_to_lancedb(dataset_name, results)
        logger.info(f"  Batch saved {saved_count} tables to LanceDB")
    
    logger.info("-" * 50)
    logger.info(f"Batch Analysis Complete:")
    logger.info(f"  Tables processed: {len(results)}")
    logger.info(f"  Columns analyzed: {columns_analyzed}")
    logger.info(f"  Transforms reused: {reuse_count}")
    logger.info(f"  LLM calls: {llm_calls}")
    logger.info("-" * 50)
    
    # Flush pending contracts to database
    flushed = shared_repository.flush()
    if flushed > 0:
        logger.info(f"  Contracts flushed to DB: {flushed}")
    
    return {
        "current_batch_results": results,
        "total_columns_analyzed": state.total_columns_analyzed + columns_analyzed,
        "total_llm_calls": state.total_llm_calls + llm_calls,
        "code_reuse_count": state.code_reuse_count + reuse_count,
        "completed_tables": state.completed_tables + len(results),
        "errors": state.errors + errors,
    }


def _save_all_tables_to_lancedb(dataset_name: str, results: Dict[str, TableColumnSummaries]) -> int:
    """
    Batch save all table column summaries to LanceDB in a single write operation.
    
    This is much faster than individual per-table writes as it:
    1. Avoids lock contention from parallel tasks
    2. Reduces I/O operations to a single batch write
    3. Minimizes index rebuilding overhead
    
    Args:
        dataset_name: Name of the dataset
        results: Dict mapping table_id to TableColumnSummaries
        
    Returns:
        Number of tables saved
    """
    try:
        from store.store_singleton import get_store
        import pyarrow as pa
        
        store = get_store()
        table_name = f"{dataset_name}_column_summaries"
        
        # Build all records at once
        all_records = []
        for table_id, table_summaries in results.items():
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
                
                all_records.append({
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
        
        if not all_records:
            return 0
        
        # Create PyArrow table for batch write
        pa_table = pa.Table.from_pylist(all_records)
        
        # Single batch write to LanceDB
        # Use try/except to handle race condition where table_names() may not reflect latest state
        try:
            if table_name in store.db.table_names(limit=1000000):
                tbl = store.db.open_table(table_name)
                tbl.add(pa_table)
                logger.debug(f"Batch appended {len(all_records)} columns ({len(results)} tables) to {table_name}")
            else:
                store.db.create_table(table_name, pa_table)
                logger.info(f"Created {table_name} with {len(all_records)} columns ({len(results)} tables)")
        except ValueError as e:
            if "already exists" in str(e):
                # Table was created between check and create - just append
                logger.debug(f"Table {table_name} was created concurrently, appending instead")
                tbl = store.db.open_table(table_name)
                tbl.add(pa_table)
            else:
                raise
        
        return len(results)
        
    except Exception as e:
        logger.error(f"Failed to batch save tables to LanceDB: {e}")
        import traceback
        traceback.print_exc()
        return 0


def _save_table_to_lancedb(dataset_name: str, table_id: str, table_summaries: TableColumnSummaries) -> None:
    """
    Save column summaries for a single table to LanceDB (incremental).
    
    Creates/appends to dataset_name_column_summaries table.
    Thread-safe: uses a global lock to prevent concurrent writes.
    
    NOTE: This function is kept for backward compatibility but the preferred
    approach is to use _save_all_tables_to_lancedb for batch saving after
    parallel processing completes.
    """
    import threading
    
    # Global lock for LanceDB writes (module-level)
    if not hasattr(_save_table_to_lancedb, '_lock'):
        _save_table_to_lancedb._lock = threading.Lock()
    
    try:
        from store.store_singleton import get_store
        import pyarrow as pa
        
        store = get_store()
        table_name = f"{dataset_name}_column_summaries"
        
        # Build records for this table
        records = []
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
            return
        
        # Create PyArrow table
        pa_table = pa.Table.from_pylist(records)
        
        # Thread-safe LanceDB write with lock
        with _save_table_to_lancedb._lock:
            if table_name in store.db.table_names(limit=1000000):
                tbl = store.db.open_table(table_name)
                tbl.add(pa_table)
                logger.debug(f"Appended {len(records)} columns to {table_name}")
            else:
                store.db.create_table(table_name, pa_table)
                logger.info(f"Created {table_name} with {len(records)} columns")
            
    except Exception as e:
        logger.error(f"Failed to save table {table_id} to LanceDB: {e}")
        import traceback
        traceback.print_exc()
