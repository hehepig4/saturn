"""
State Definition for Table Discovery Layer 2 Subgraph

Defines the state schema for generating Layer 2 Defined Classes from tables
using Layer 1 Primitive Classes.

Layer 2 includes:
- Column Defined Classes: Columns classified by Primitive Class with P0/P1 metadata
- Table Defined Classes: Tables with Full EL and Core EL expressions
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


# ============== Pydantic Models for Structured Output ==============

class ColumnAnnotation(BaseModel):
    """LLM output for a single column annotation.
    
    Only outputs: description.
    primitive_class comes from Stage 2 and is added later.
    Uses column_index (0-based) to match with Stage 2 data.
    """
    column_index: int = Field(description="Column index (0-based) matching the order in column summaries")
    description: str = Field(default="", description="Human-readable description for rdfs:comment (P0)")


class TableAnnotation(BaseModel):
    """LLM output for complete table annotation."""
    table_id: str
    column_annotations: List[ColumnAnnotation]
    table_class_name: str = Field(description="CamelCase class name like 'ElectionResultsTable'")
    table_summary: str = Field(
        description="Information-dense summary including: table purpose, key entities/columns, time coverage, measures, and domain context (used for both rdfs:comment and semantic search)"
    )


class ColumnDefinedClass(BaseModel):
    """Layer 2 Defined Class for a column."""
    column_id: str = Field(description="Unique column ID: table_id::column_name")
    column_name: str
    primitive_class: str
    el_definition: str = Field(description="EL++ definition")
    label: str
    description: str = Field(description="rdfs:comment content (P0)")
    insights: Dict[str, Any] = Field(default_factory=dict)
    # Transform contract reference (from Stage 2)
    contract_id: Optional[str] = Field(default=None, description="ID of TransformContract for value transformation")


class TableDefinedClass(BaseModel):
    """Layer 2 Defined Class for a table."""
    table_id: str
    class_name: str
    el_definition: str = Field(description="EL definition: Table ⊓ ∃hasColumn.Col1 ⊓ ...")
    label: str
    description: str = Field(description="rdfs:comment content")
    summary: str = Field(default="", description="Detailed summary for LM embedding")
    column_ids: List[str] = Field(default_factory=list)


# ============== Main State ==========================

class TableDiscoveryLayer2State(BaseModel):
    """
    State for Table Discovery Layer 2 workflow.
    
    Workflow:
        1. load_data: Load tables, column summaries, and Layer 1 primitives
        2. annotate_tables: Annotate tables with Layer 1 classes (parallel)
        3. create_defined_classes: Create Column and Table Defined Classes
        4. export_layer2: Store to LanceDB and export OWL
    """
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        extra='allow'
    )
    
    # ========== Input Configuration ==========
    
    dataset_name: str = Field(
        default="fetaqa",
        description="Dataset name (e.g., 'fetaqa')"
    )
    
    max_tables: Optional[int] = Field(
        default=None,
        description="Maximum tables to process (None = all)"
    )
    
    table_offset: int = Field(
        default=0,
        description="Skip first N tables (for incremental processing)"
    )
    
    llm_max_workers: int = Field(
        default=128,
        description="Max parallel workers for LLM calls (annotation, property discovery)"
    )
    
    batch_size: int = Field(
        default=50,
        description="Number of tables to process per batch for checkpointing"
    )
    
    output_dir: str = Field(
        default="data/cache/layer2_checkpoints",
        description="Directory for batch checkpoint files"
    )
    
    # ========== Batch Processing State ==========
    
    current_batch_index: int = Field(
        default=0,
        description="Current batch being processed (0-indexed)"
    )
    
    total_batches: int = Field(
        default=0,
        description="Total number of batches to process"
    )
    
    current_batch_tables: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Tables in current batch"
    )
    
    current_batch_annotations: List[TableAnnotation] = Field(
        default_factory=list,
        description="Annotations for current batch"
    )
    
    current_batch_column_classes: List[ColumnDefinedClass] = Field(
        default_factory=list,
        description="Column classes for current batch"
    )
    
    current_batch_table_classes: List[TableDefinedClass] = Field(
        default_factory=list,
        description="Table classes for current batch"
    )
    
    checkpoint_paths: List[str] = Field(
        default_factory=list,
        description="Paths to saved checkpoint files"
    )
    
    completed_tables: int = Field(
        default=0,
        description="Number of tables completed so far"
    )
    
    # ========== Node 1: load_data outputs ==========
    
    data_loaded: bool = Field(
        default=False,
        description="Flag indicating data loading is complete"
    )
    
    tables: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Loaded table records from LanceDB"
    )
    
    column_summaries: Dict[str, List[Dict]] = Field(
        default_factory=dict,
        description="Column summaries grouped by table_id"
    )
    
    primitive_classes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Layer 1 Primitive Classes"
    )
    
    layer1_class_names: List[str] = Field(
        default_factory=list,
        description="Set of Layer 1 class names for validation"
    )
    
    data_properties: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Layer 1 Data Properties with readout templates"
    )
    
    layer1_ontology_id: Optional[str] = Field(
        default=None,
        description="Layer 1 ontology ID for reference"
    )
    
    # ========== Node 2: annotate_tables outputs ==========
    
    annotation_done: bool = Field(
        default=False,
        description="Flag indicating annotation is complete"
    )
    
    annotations: List[TableAnnotation] = Field(
        default_factory=list,
        description="All table annotations"
    )
    
    annotation_errors: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Tables that failed annotation"
    )
    
    correction_stats: Dict[str, int] = Field(
        default_factory=lambda: {
            'valid': 0,
            'case_corrected': 0,
            'fuzzy_corrected': 0,
            'word_overlap_corrected': 0,
            'fallback': 0,
        },
        description="P1-c validation correction statistics"
    )
    
    all_corrections: List[Dict[str, str]] = Field(
        default_factory=list,
        description="All primitive class corrections made"
    )
    
    # ========== Node 3: create_defined_classes outputs ==========
    
    classes_created: bool = Field(
        default=False,
        description="Flag indicating defined class creation is complete"
    )
    
    column_defined_classes: List[ColumnDefinedClass] = Field(
        default_factory=list,
        description="All Column Defined Classes"
    )
    
    table_defined_classes: List[TableDefinedClass] = Field(
        default_factory=list,
        description="All Table Defined Classes"
    )
    
    primitive_classes_used: Dict[str, int] = Field(
        default_factory=dict,
        description="Primitive class usage count"
    )
    
    # ========== Node 5a: validate_layer2 outputs ==========
    
    validation_done: bool = Field(
        default=False,
        description="Flag indicating validation is complete"
    )
    
    validation_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Complete validation result"
    )
    
    valid_table_ids: List[str] = Field(
        default_factory=list,
        description="Table IDs that passed validation"
    )
    
    invalid_table_ids: List[str] = Field(
        default_factory=list,
        description="Table IDs that failed validation"
    )
    
    # ========== Node 5b: export_layer2 outputs ==========
    
    export_done: bool = Field(
        default=False,
        description="Flag indicating export is complete"
    )
    
    layer2_ontology_id: Optional[str] = Field(
        default=None,
        description="Generated Layer 2 ontology ID"
    )
    
    owl_xml_path: Optional[str] = Field(
        default=None,
        description="Path to exported OWL/XML file"
    )
    
    owl_manchester_path: Optional[str] = Field(
        default=None,
        description="Path to exported OWL Manchester file"
    )
    
    # ========== Final Status ==========
    
    success: bool = Field(
        default=False,
        description="Whether the workflow completed successfully"
    )
    
    error: Optional[str] = Field(
        default=None,
        description="Error message if workflow failed"
    )
    
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Workflow timestamp"
    )
