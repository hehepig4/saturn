"""
Node 3: Create Defined Classes

Creates Column Defined Classes and Table Defined Classes from config.
"""

import json
from typing import Dict, Any, List, Optional
from loguru import logger

from workflows.common.node_decorators import graph_node
from workflows.indexing.annotation.state import (
    TableDiscoveryLayer2State,
    TableAnnotation,
    ColumnAnnotation,
    ColumnDefinedClass,
    TableDefinedClass,
)


@graph_node(node_type="processing")
def create_defined_classes_node(state: TableDiscoveryLayer2State) -> Dict[str, Any]:
    """
    Create Column and Table Defined Classes from current batch annotations.
    
    For each annotation in current batch:
        1. Create ColumnDefinedClass for each column
        2. Create TableDefinedClass with Full EL and Core EL
        3. Track primitive class usage
    
    Returns:
        State updates with defined classes for current batch
    """
    logger.info("=" * 70)
    logger.info(f"Node 3: Creating Defined Classes (Batch {state.current_batch_index + 1})")
    logger.info("=" * 70)
    
    # Use current_batch_annotations instead of all annotations
    annotations = state.current_batch_annotations
    column_summaries = state.column_summaries
    tables = state.current_batch_tables  # Use current batch tables
    primitive_classes = state.primitive_classes
    
    # Build table lookup
    table_lookup = {t.get('table_id', ''): t for t in tables}
    
    all_column_classes = []
    all_table_classes = []
    primitive_usage = {}
    
    for annotation in annotations:
        table_id = annotation.table_id
        table = table_lookup.get(table_id, {})
        doc_title = table.get('document_title', '')
        summaries = column_summaries.get(table_id, [])
        
        # Validate summaries exist
        if not summaries:
            raise ValueError(
                f"No column summaries found for table '{table_id}'. "
                f"Ensure Stage 2 (Column Summary) was completed successfully. "
                f"Available tables in column_summaries: {len(column_summaries)}"
            )
        
        # Build index-based lookup for summaries
        summaries_by_index = {s.get('column_index', i): s for i, s in enumerate(summaries)}
        
        # Create Column Defined Classes
        column_classes = []
        for col_ann in annotation.column_annotations:
            # Use column_index to find matching summary
            col_summary = summaries_by_index.get(col_ann.column_index)
            
            if not col_summary:
                raise ValueError(
                    f"No column summary found for column index {col_ann.column_index} in table '{table_id}'. "
                    f"Available indices: {sorted(summaries_by_index.keys())}. "
                    f"Total summaries: {len(summaries)}."
                )
            
            col_class = _create_column_defined_class(
                table_id, col_ann, col_summary, primitive_classes
            )
            column_classes.append(col_class)
            
            # Track primitive usage (get from Stage 2 summary, required)
            prim = col_summary.get('primitive_class')
            if not prim:
                raise ValueError(
                    f"Missing primitive_class for column index {col_ann.column_index} in table '{table_id}'. "
                    f"Column summary exists but primitive_class is empty/None. "
                    f"Ensure Stage 2 (Column Summary) was completed successfully."
                )
            primitive_usage[prim] = primitive_usage.get(prim, 0) + 1
        
        # Create Table Defined Class
        table_class = _create_table_defined_class(
            annotation, column_classes, doc_title, summaries
        )
        
        all_column_classes.extend(column_classes)
        all_table_classes.append(table_class)
    
    logger.info(f"  ✓ Created {len(all_column_classes)} Column Defined Classes")
    logger.info(f"  ✓ Created {len(all_table_classes)} Table Defined Classes")
    
    # Log primitive usage
    logger.debug("")
    logger.debug("  Primitive Class Usage (top 10):")
    for prim, count in sorted(primitive_usage.items(), key=lambda x: -x[1])[:10]:
        logger.debug(f"    {prim}: {count}")
    
    return {
        "classes_created": True,
        "current_batch_column_classes": all_column_classes,  # Save to current_batch
        "current_batch_table_classes": all_table_classes,    # Save to current_batch
        "primitive_classes_used": primitive_usage,
    }


def _create_column_defined_class(
    table_id: str,
    col_ann: ColumnAnnotation,
    col_summary: Optional[Dict] = None,
    primitive_classes: Optional[List[Dict]] = None,
) -> ColumnDefinedClass:
    """
    Create a Column Defined Class.
    
    P0: Description from LLM (rdfs:comment)
    
    Note: primitive_class comes from Stage 2 (col_summary), NOT from LLM output.
    Uses column_index to identify columns.
    """
    # Get column_name from col_summary (Stage 2 is source of truth)
    col_name = col_summary.get('column_name', f'col_{col_ann.column_index}') if col_summary else f'col_{col_ann.column_index}'
    col_id = f"{table_id}::{col_name}"
    
    # Get primitive_class from Stage 2 summary (required, no fallback)
    primitive_class = col_summary.get('primitive_class') if col_summary else None
    if not primitive_class:
        raise ValueError(
            f"Missing primitive_class for column index {col_ann.column_index} in table '{table_id}'. "
            f"col_summary={col_summary}. "
            f"Ensure Stage 2 (Column Summary) was completed successfully."
        )
    
    # EL definition is simply the primitive class
    el_def = primitive_class
    
    # Extract insights from column summary
    insights = {}
    if col_summary:
        insights = {
            "unique_ratio": col_summary.get('unique_ratio'),
            "null_ratio": col_summary.get('null_ratio'),
        }
        
        # Get readouts from all data_property_values (multifacet)
        data_property_values = col_summary.get('data_property_values', [])
        if isinstance(data_property_values, str):
            try:
                data_property_values = json.loads(data_property_values)
            except:
                data_property_values = []
        
        # Collect all readouts from multifacet DPs
        readouts = []
        for dpv in data_property_values:
            if isinstance(dpv, dict) and dpv.get('readout'):
                readouts.append(dpv['readout'])
        
        if readouts:
            # Join multiple facet readouts
            insights['readout'] = "; ".join(readouts)
        
        # Fallback: sample_values if no readout
        if 'readout' not in insights:
            sample_values = col_summary.get('sample_values', [])
            if isinstance(sample_values, str):
                try:
                    sample_values = json.loads(sample_values)
                except:
                    sample_values = []
            if sample_values:
                from workflows.population.sampling_utils import sample_values as do_sample
                insights['sample_values'] = do_sample(sample_values, 5)
    
    # Extract contract_id from data_property_values (first one that has it)
    contract_id = None
    if col_summary:
        data_property_values = col_summary.get('data_property_values', [])
        if isinstance(data_property_values, str):
            try:
                data_property_values = json.loads(data_property_values)
            except:
                data_property_values = []
        
        for dpv in data_property_values:
            if isinstance(dpv, dict) and dpv.get('contract_id'):
                contract_id = dpv['contract_id']
                break
    
    # P0: Description from LLM
    description = col_ann.description or f"Column '{col_name}' classified as {primitive_class}"
    
    return ColumnDefinedClass(
        column_id=col_id,
        column_name=col_name,
        primitive_class=primitive_class,
        el_definition=el_def,
        label=col_name,
        description=description,
        insights=insights,
        contract_id=contract_id,
    )


def _create_table_defined_class(
    annotation: TableAnnotation,
    column_classes: List[ColumnDefinedClass],
    document_title: str,
    column_summaries: Optional[List[Dict]] = None,
) -> TableDefinedClass:
    """
    Create a Table Defined Class with EL definition.
    
    EL: Table ⊓ ∃hasColumn.Col1 ⊓ ∃hasVirtualColumn.VCol1 ...
    
    Virtual columns use hasVirtualColumn instead of hasColumn.
    """
    # Build index-based lookup for summaries to check is_virtual
    summaries_by_index = {}
    if column_summaries:
        for i, s in enumerate(column_summaries):
            summaries_by_index[s.get('column_index', i)] = s
    
    # EL Definition: Table with all columns (using appropriate predicate)
    column_constraints = []
    for cc in column_classes:
        # Check if this column is virtual
        summary = summaries_by_index.get(cc.column_name)
        if summary is None:
            # Try to find by column_index from ColumnDefinedClass
            # Parse column_id format: {table_id}::{column_name}
            for idx, s in summaries_by_index.items():
                if s.get('column_name') == cc.column_name:
                    summary = s
                    break
        
        is_virtual = summary.get('is_virtual', False) if summary else False
        predicate = "hasVirtualColumn" if is_virtual else "hasColumn"
        column_constraints.append(f"∃{predicate}.{cc.column_name}")
    
    el_def = "Table ⊓ " + " ⊓ ".join(column_constraints) if column_constraints else "Table"
    
    # Summary from LLM or generate from structure
    if annotation.table_summary:
        summary = annotation.table_summary
    else:
        summary = f"Table '{document_title}' with {len(column_classes)} columns."
    
    return TableDefinedClass(
        table_id=annotation.table_id,
        class_name=annotation.table_class_name,
        el_definition=el_def,
        label=document_title or annotation.table_class_name,
        description=summary,
        summary=summary,
        column_ids=[cc.column_id for cc in column_classes],
    )
