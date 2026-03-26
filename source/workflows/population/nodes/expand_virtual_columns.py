"""
Node: Expand Virtual Columns

Extracts virtual columns from table context (e.g., document_title, section_title)
using LLM. These virtual columns capture metadata information that is not present
in the actual table content but is semantically relevant.

Design:
- Virtual columns are treated identically to real columns in subsequent processing
- Only difference: marked with is_virtual=True for TBox export (hasVirtualColumn)
- LLM extracts column name + cell values from context text
- Inserted between load_tables_batch and classify_columns in Stage 2
"""

from typing import Dict, Any, List, Optional
from loguru import logger
from pydantic import BaseModel, Field

from workflows.common.node_decorators import graph_node
from workflows.population.state import ColumnSummaryState


# ============== Output Schema ==============

class VirtualColumnSpec(BaseModel):
    """Specification for a single virtual column extracted from context."""
    name: str  # Column name describing the information type
    cells: List[str]  # Cell values (usually just one for metadata)


class VirtualColumnsOutput(BaseModel):
    """Output schema for virtual column extraction."""
    columns: List[VirtualColumnSpec] = Field(default_factory=list)  # Extracted virtual columns


# ============== Prompt Template ==============

EXTRACT_VIRTUAL_COLUMNS_PROMPT = """# Virtual Column Extraction

## Task
Extract metadata columns from the table context. These columns represent table-level information (e.g., year, category, region) that applies to the entire table but is not explicitly present as a column.

## Input
- **Table Context**: {table_context}
- **Existing Columns**: {headers}

## Reference: Column Type Hierarchy
Use these types to guide your column naming:
{class_hierarchy}

## Guidelines

1. **Context-Based Extraction**
   Extract temporal markers (years, dates), geographic references (countries, regions), and categorical identifiers (leagues, organizations) from the table context.
   Each extracted piece represents a table-level attribute.

2. **Atomic Values**
   Each cell should contain a single atomic value, not compound phrases or ranges.
   For example: "2023" instead of "2022-2023 Season".

## Constraints
- **ONLY** extract information explicitly present in Table Context
- **DO NOT** invent or hallucinate column names or values
- **At most 3 columns** - prioritize the most distinctive metadata
- If no clear metadata is found, return {{"columns": []}}

## Example
Context: "Category A - Region X 2022, 2023 Report"
Output: {{"columns": [{{"name": "Category", "cells": ["A"]}}, {{"name": "Region", "cells": ["X"]}}, {{"name": "Year", "cells": ["2022", "2023"]}}]}}

## Output
Analyze "{table_context}" and extract:
"""


# ============== Helper Functions ==============

def _format_context_info(table: Dict[str, Any], context_fields: List[str]) -> str:
    """
    Build table title from context fields.
    
    Combines document_title and section_title into a natural title string.
    """
    parts = []
    for field in context_fields:
        value = table.get(field, '')
        if value:
            parts.append(str(value).strip())
    
    if not parts:
        return ""
    
    # Combine: "Document Title - Section Title" or just "Document Title"
    return " - ".join(parts)


def _extract_virtual_columns_for_table(
    table: Dict[str, Any],
    context_fields: List[str],
    primitive_classes_full: Optional[List[Dict[str, Any]]] = None,
    llm_purpose: str = "default",
) -> List[Dict[str, Any]]:
    """
    Extract virtual columns for a single table using LLM.
    
    Args:
        table: Table data with context fields
        context_fields: List of field names to use as context
        primitive_classes_full: Full primitive class info for type hierarchy guidance
        llm_purpose: LLM purpose key for model selection
        
    Returns:
        List of virtual column dicts with name, cells, is_virtual
    """
    from llm.manager import get_llm_by_purpose
    from llm.invoke_with_stats import invoke_structured_llm_with_retry
    from utils.tree_formatter import format_class_hierarchy
    
    # Build table context from context fields
    table_context = _format_context_info(table, context_fields)
    if not table_context:
        return []
    
    # Get table schema for reference
    headers = table.get('columns', table.get('headers', []))
    
    # Format class hierarchy for prompt
    class_hierarchy_str = ""
    if primitive_classes_full:
        # Filter out Table and Column root classes
        filtered_classes = [
            c for c in primitive_classes_full 
            if c.get('name', '') not in ('Table', 'upo:Table', 'Column', 'upo:Column')
        ]
        class_hierarchy_str = format_class_hierarchy(
            filtered_classes, 
            include_column_root=True, 
            max_desc_length=60
        )
    
    if not class_hierarchy_str:
        class_hierarchy_str = "(No type hierarchy available)"
    
    # Format prompt
    prompt = EXTRACT_VIRTUAL_COLUMNS_PROMPT.format(
        table_context=table_context,
        headers=headers,
        class_hierarchy=class_hierarchy_str,
    )
    
    # Call LLM
    def llm_factory(temperature: float):
        return get_llm_by_purpose(llm_purpose, temperature_override=temperature)
    
    try:
        output = invoke_structured_llm_with_retry(
            llm_factory=llm_factory,
            output_schema=VirtualColumnsOutput,
            prompt=prompt,
            max_retries=2,
        )
        
        # Convert to internal format with is_virtual flag
        virtual_columns = []
        for col_spec in output.columns:
            if col_spec.name and col_spec.cells:
                virtual_columns.append({
                    'name': col_spec.name,
                    'cells': col_spec.cells,
                    'is_virtual': True,
                })
        
        return virtual_columns
        
    except Exception as e:
        logger.warning(f"Failed to extract virtual columns: {e}")
        return []


def _expand_table_with_virtual_columns(
    table: Dict[str, Any],
    virtual_columns: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Expand table data with virtual columns.
    
    Virtual columns are appended to the columns list and their values
    are added to parsed_rows.
    
    Args:
        table: Original table data
        virtual_columns: List of virtual column specs
        
    Returns:
        Expanded table data
    """
    if not virtual_columns:
        return table
    
    # Copy table to avoid mutation
    expanded = dict(table)
    
    # Get existing columns
    existing_columns = list(expanded.get('columns', expanded.get('headers', [])))
    existing_rows = list(expanded.get('parsed_rows', expanded.get('rows', [])))
    
    # Track which columns are virtual (by index)
    virtual_column_indices = []
    
    # Add virtual columns
    num_rows = len(existing_rows)
    for vc in virtual_columns:
        col_name = vc['name']
        col_cells = vc['cells']
        
        # Add to columns list
        existing_columns.append(col_name)
        virtual_column_indices.append(len(existing_columns) - 1)
        
        # Fill cells into corresponding rows (remaining rows get empty string)
        # This handles multi-value virtual columns (e.g., Year: ["1982", "1983", "1984"])
        # Empty values are automatically filtered during sampling for LLM prompts
        for row_idx, row in enumerate(existing_rows):
            if row_idx < len(col_cells):
                row.append(col_cells[row_idx])
            else:
                row.append("")  # Empty for remaining rows
    
    expanded['columns'] = existing_columns
    expanded['parsed_rows'] = existing_rows
    expanded['virtual_column_indices'] = virtual_column_indices
    
    return expanded


# ============== Main Node ==============

@graph_node(node_type="processing", warn_threshold_seconds=300.0)
def expand_virtual_columns_node(state: ColumnSummaryState) -> Dict[str, Any]:
    """
    Expand tables with virtual columns extracted from context.
    
    This node runs BEFORE classify_columns to ensure virtual columns
    go through the same processing pipeline as real columns.
    
    Uses parallel LLM calls for efficient batch processing.
    
    Args:
        state: Current workflow state with current_batch_tables
        
    Returns:
        State update with expanded tables containing virtual columns
    """
    from llm.parallel import invoke_llm_parallel_with_func
    
    if not state.enable_virtual_columns:
        logger.info("  Virtual column expansion disabled, skipping")
        return {}
    
    tables = state.current_batch_tables
    context_fields = state.context_fields
    
    if not tables:
        logger.info("  No tables to process")
        return {}
    
    if not context_fields:
        logger.info("  No context fields specified, skipping virtual column expansion")
        return {}
    
    logger.info(f"  Extracting virtual columns from {len(tables)} tables (parallel)")
    logger.info(f"  Context fields: {context_fields}")
    
    # Get primitive_classes_full for type hierarchy guidance
    primitive_classes_full = state.primitive_classes_full
    if primitive_classes_full:
        logger.info(f"  Using {len(primitive_classes_full)} primitive classes for type guidance")
    
    # Get llm_purpose from state (default: "default")
    llm_purpose = getattr(state, 'llm_purpose', 'default')
    logger.info(f"  LLM Purpose: {llm_purpose}")
    
    # Prepare tasks for parallel execution
    def extract_task(task: Dict) -> Dict:
        table = task['table']
        table_id = task['table_id']
        idx = task['idx']
        classes_full = task.get('primitive_classes_full')
        task_llm_purpose = task.get('llm_purpose', 'default')
        
        try:
            virtual_columns = _extract_virtual_columns_for_table(
                table, context_fields, primitive_classes_full=classes_full, llm_purpose=task_llm_purpose
            )
            return {
                'idx': idx,
                'table_id': table_id,
                'virtual_columns': virtual_columns,
                'success': True,
            }
        except Exception as e:
            logger.warning(f"Failed to extract virtual columns for {table_id[:50]}: {e}")
            return {
                'idx': idx,
                'table_id': table_id,
                'virtual_columns': [],
                'success': False,
                'error': str(e),
            }
    
    tasks = [
        {
            'table': t, 
            'table_id': t.get('table_id', f'table_{i}'), 
            'idx': i,
            'primitive_classes_full': primitive_classes_full,
            'llm_purpose': llm_purpose,
        }
        for i, t in enumerate(tables)
    ]
    
    # Get max_workers from state (default: 128)
    table_max_workers = getattr(state, 'table_max_workers', 128)
    
    # Execute in parallel
    results = invoke_llm_parallel_with_func(
        func=extract_task,
        tasks=tasks,
        max_workers=min(table_max_workers, len(tasks)),
        show_progress=True,
        max_consecutive_errors=10,
        max_error_rate=0.5,  # Virtual columns are optional, allow higher error rate
    )
    
    # Build results map (idx -> virtual_columns)
    results_map = {}
    for result in results:
        if isinstance(result, Exception):
            continue
        results_map[result['idx']] = result.get('virtual_columns', [])
    
    # Expand tables with virtual columns (preserving order)
    expanded_tables = []
    total_virtual_columns = 0
    tables_with_virtual = 0
    
    for i, table in enumerate(tables):
        table_id = table.get('table_id', 'unknown')
        virtual_columns = results_map.get(i, [])
        
        if virtual_columns:
            tables_with_virtual += 1
            total_virtual_columns += len(virtual_columns)
            logger.debug(f"  Table '{table_id[:50]}': extracted {len(virtual_columns)} virtual columns")
            for vc in virtual_columns:
                logger.debug(f"    - {vc['name']}: {vc['cells']}")
        else:
            # Log why no virtual columns were extracted
            context = _format_context_info(table, context_fields)
            if context:
                logger.debug(f"  Table '{table_id[:50]}': LLM returned no virtual columns for '{context[:60]}...'")
        
        # Expand table with virtual columns
        expanded_table = _expand_table_with_virtual_columns(table, virtual_columns)
        expanded_tables.append(expanded_table)
    
    logger.info(f"  ✓ Extracted {total_virtual_columns} virtual columns from {tables_with_virtual} tables")
    
    return {
        'current_batch_tables': expanded_tables,
    }
