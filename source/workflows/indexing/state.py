"""
State Definition for Table Summarization Subgraph

Defines the state schema for generating table-level summaries from Layer 2
Defined Classes for retrieval and embedding.

Outputs:
    - Table View: Unified column serialization for embedding
    - Relationship View: Column relationships from Layer 2 property discovery
    - PNEUMA-compatible format for benchmark comparison
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict


# ============== Output Models ==============


class TableSummary(BaseModel):
    """Complete summary for a table."""
    table_id: str
    table_class_name: str
    table_description: str
    table_summary: str = Field(default="")
    num_cols: int = Field(default=0)
    
    # Multi-view summaries
    view_summaries: Dict[str, str] = Field(default_factory=dict)
    blocked_views: Dict[str, str] = Field(default_factory=dict)
    
    # Column metadata for extended output
    column_metadata: List[Dict[str, Any]] = Field(default_factory=list)
    
    # PNEUMA-compatible fields
    column_narrations: List[Dict[str, str]] = Field(default_factory=list)
    blocked_column_narrations: str = Field(default="")
    blocked_column_stats: str = Field(default="")  # Stats/readouts only (separated)


class SummarizationOutput(BaseModel):
    """Complete output of the summarization workflow."""
    dataset_name: str
    timestamp: str
    num_summaries: int = Field(default=0)
    summaries: List[TableSummary] = Field(default_factory=list)
    
    # Output paths (PNEUMA format only)
    pneuma_json_path: Optional[str] = None
    pneuma_parquet_path: Optional[str] = None


# ============== Main State ==============

class TableSummarizationState(BaseModel):
    """
    State for Table Summarization workflow.
    
    Workflow:
        1. load_layer2_data: Load column mappings, table classes, column summaries
        2. generate_summaries: Create multi-view summaries for each table
        3. export_summaries: Save to JSON and Parquet formats
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
    
    output_base: str = Field(
        default="",  # Will be set to lake_data_path('summaries') in graph.py
        description="Base output directory for summaries (absolute path)"
    )
    
    serialization_level: int = Field(
        default=2,
        description="Column serialization level: 1 (flat), 2 (class-prefixed), 3 (natural)"
    )
    
    # ========== Node 1: load_layer2_data outputs ==========
    
    data_loaded: bool = Field(
        default=False,
        description="Flag indicating data loading is complete"
    )
    
    column_summaries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Column summaries from LanceDB"
    )
    
    column_mappings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Column mappings (Layer 2 column classes)"
    )
    
    table_classes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Table defined classes from Layer 2"
    )
    
    # Lookup structures
    col_summaries_by_table: Dict[str, Dict[str, Dict]] = Field(
        default_factory=dict,
        description="Column summaries grouped by table_id -> column_name"
    )
    
    col_classes_by_table: Dict[str, List[Dict]] = Field(
        default_factory=dict,
        description="Column classes grouped by table_id"
    )
    
    # Relationships from Layer 2 (extracted from OWL or LanceDB)
    relationships: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Column relationships discovered in Layer 2"
    )
    
    relationships_by_table: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Relationships grouped by table_id"
    )
    
    # ========== Node 2: generate_summaries outputs ==========
    
    summaries_generated: bool = Field(
        default=False,
        description="Flag indicating summaries have been generated"
    )
    
    table_summaries: List[TableSummary] = Field(
        default_factory=list,
        description="Generated table summaries"
    )
    
    # ========== Node 3: export_summaries outputs ==========
    
    export_done: bool = Field(
        default=False,
        description="Flag indicating export is complete"
    )
    
    output: Optional[SummarizationOutput] = Field(
        default=None,
        description="Final output with paths"
    )
    
    # ========== Workflow Control ==========
    
    success: bool = Field(
        default=False,
        description="Whether the workflow completed successfully"
    )
    
    error: Optional[str] = Field(
        default=None,
        description="Error message if workflow failed"
    )
    
    trajectory: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Workflow trajectory for debugging"
    )
