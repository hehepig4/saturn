"""
State Definition for Column Summary Agent

Defines state schema and Pydantic models for column analysis results.
This is a simplified version focused on pure statistical analysis.

Key Design:
- A column is classified to a PrimitiveClass
- The column can have multiple DataProperty values (multifacet)
- Each DataProperty has its own statistics based on range_type
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from config.hyperparameters import SUCCESSIVE_HALVING_BUDGET_MULTIPLIER


class DataPropertyValue(BaseModel):
    """
    Value computed for a single DataProperty.
    
    A column classified to PrimitiveClass can have multiple DataProperty values,
    each computed with its own TransformContract and statistics.
    """
    data_property_name: str = Field(description="DataProperty name from TBox")
    range_type: str = Field(default="xsd:string", description="OWL 2 EL datatype")
    
    # Statistics for this DataProperty (based on range_type)
    statistics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Computed statistics (min, max, mean, etc.)"
    )
    
    # Transform metadata
    contract_id: Optional[str] = Field(
        default=None,
        description="ID of the TransformContract used for this column"
    )
    transform_pattern: Optional[str] = Field(
        default=None,
        description="Regex pattern used for transformation"
    )
    transform_success_rate: float = Field(
        default=0.0,
        description="Fraction of values that both matched pattern AND transformed successfully"
    )
    
    # Readout
    readout: Optional[str] = Field(
        default=None,
        description="Human-readable description from template"
    )
    readout_template: Optional[str] = Field(
        default=None,
        description="Template used for readout"
    )


class ColumnSummary(BaseModel):
    """
    Column statistics summary with primitive class assignment.
    
    This is the output contract between:
    - Column Summary Agent (producer)
    - Table Discovery Layer 2 (consumer)
    
    Contains:
    - primitive_class: Assigned by LLM classification
    - data_property_values: Multiple DataProperty values (multifacet)
    - Basic value statistics
    
    Note: role, description, and properties are handled by Stage 3.
    """
    
    # Basic info
    column_name: str = Field(description="Column header name")
    column_index: int = Field(description="0-based column index")
    is_virtual: bool = Field(
        default=False,
        description="True if this column was extracted from context (not actual table data)"
    )
    
    # Primitive class (assigned by classify_columns_node)
    # MUST be loaded from Stage 1 - no hardcoded defaults
    primitive_class: str = Field(
        default="",
        description="Primitive class from Layer 1 TBox (e.g., upo:YearColumn). Empty means not yet classified."
    )
    
    # Multiple DataProperty values (multifacet)
    data_property_values: List[DataPropertyValue] = Field(
        default_factory=list,
        description="Values for each applicable DataProperty"
    )
    
    # Value statistics (basic counts)
    total_count: int = Field(default=0, description="Total number of cells")
    null_count: int = Field(default=0, description="Count of null/empty cells")
    unique_count: int = Field(default=0, description="Count of unique values")
    
    # Derived ratios
    null_ratio: float = Field(default=0.0, description="null_count / total_count")
    unique_ratio: float = Field(default=0.0, description="unique_count / (total - null)")
    
    # Sample values
    sample_values: List[str] = Field(
        default_factory=list,
        description="5 representative sample values"
    )
    
    # Audit info
    execution_time_ms: Optional[float] = Field(
        default=None,
        description="Time taken to execute analysis"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if analysis failed"
    )
    

class TableColumnSummaries(BaseModel):
    """All column summaries for a single table."""
    table_id: str
    document_title: str = ""
    section_title: str = ""
    columns: Dict[str, ColumnSummary] = Field(
        default_factory=dict,
        description="Map from column_name to ColumnSummary"
    )
    total_columns: int = 0
    successful_columns: int = 0
    failed_columns: List[str] = Field(default_factory=list)


class ColumnSummaryState(BaseModel):
    """State for Column Summary Agent subgraph."""
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        extra='allow'
    )
    
    # ========== Configuration ==========
    
    dataset_name: str = Field(
        description="Dataset name (e.g., 'fetaqa')"
    )
    table_store_name: str = Field(
        description="LanceDB table name for table data"
    )
    output_dir: str = Field(
        default="data/cache/column_summaries",
        description="Directory for output and checkpoints"
    )
    primitive_tbox_path: Optional[str] = Field(
        default=None,
        description="Path to Layer 1 Primitive TBox OWL file (optional)"
    )
    
    # TBox iteration selection
    tbox_iteration: int = Field(
        default=-1,
        description="TBox iteration to use (-1 = latest iteration from Stage 1)"
    )
    
    # Primitive classes loaded from TBox (Stage 1)
    primitive_classes: List[str] = Field(
        default_factory=list,
        description="List of primitive class names from Layer 1 TBox (e.g., upo:YearColumn)"
    )
    
    # Full primitive class info (with description and hierarchy)
    primitive_classes_full: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Full primitive class info: name, description, parent_classes"
    )
    
    # Ontology ID of the loaded primitive TBox
    primitive_tbox_ontology_id: Optional[str] = Field(
        default=None,
        description="Ontology ID of the loaded primitive TBox from Stage 1"
    )
    
    # DataProperties loaded from TBox (with readout templates)
    data_properties: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of DataProperty dicts from Layer 1 TBox (name, domain, range_type, readout_template, statistics_requirements)"
    )
    
    # Class hierarchy from TBox (child -> parent mapping)
    class_hierarchy: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Class hierarchy: maps each class to its parent classes (e.g., 'CandidateColumn': ['NameColumn'])"
    )
    
    # DataProperty hierarchy from TBox (child -> parent mapping)
    data_property_hierarchy: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="DataProperty hierarchy: maps each property to its parent properties (e.g., 'hasNameValue': ['hasTextValue'])"
    )
    
    # ========== Virtual Column Configuration ==========
    
    enable_virtual_columns: bool = Field(
        default=True,
        description="If True (default), extract virtual columns from table context"
    )
    context_fields: List[str] = Field(
        default_factory=lambda: ['document_title', 'section_title'],
        description="Field names to use as context for virtual column extraction"
    )
    
    # ========== LLM Configuration ==========
    
    llm_purpose: str = Field(
        default="default",
        description="LLM purpose key for model selection (e.g., 'default', 'qwen', 'gemini')"
    )
    disable_transform_reuse: bool = Field(
        default=False,
        description="If True, skip repository lookup and always generate with LLM (for ablation)"
    )
    
    # ========== Batch Configuration ==========
    
    batch_size: int = Field(
        default=1000,
        description="Number of tables per batch (larger = more GPU utilization)"
    )
    max_tables: Optional[int] = Field(
        default=None,
        description="Maximum tables to process (None = all)"
    )
    fresh_start: bool = Field(
        default=False,
        description="If True, clear Transform Contracts before starting. If False (default), reuse existing contracts for incremental runs."
    )
    table_max_workers: int = Field(
        default=128,
        description="Max workers for parallel table processing in expand_virtual_columns and classify_columns. "
                    "These nodes have no nested parallelism, so high values are safe."
    )
    analyze_max_workers: int = Field(
        default=32,
        description="Max workers for parallel table processing in analyze_columns node. "
                    "Lower than table_max_workers because analyze_columns has nested Successive Halving parallelism. "
                    "Total threads ≈ analyze_max_workers × sh_max_workers."
    )
    sh_max_workers: int = Field(
        default=8,
        description="Max parallel workers for Successive Halving in Transform Contract evaluation. "
                    "Controls parallelism when evaluating multiple contract candidates per column. "
                    "Total threads ≈ analyze_max_workers × sh_max_workers. Keep low (4-8) to avoid thread explosion."
    )
    
    # ========== Execution Configuration ==========
    
    code_timeout_seconds: float = Field(
        default=5.0,
        description="Timeout for each code execution"
    )
    max_retries: int = Field(
        default=2,
        description="Max retries for failed code generation"
    )
    budget_multiplier: float = Field(
        default=SUCCESSIVE_HALVING_BUDGET_MULTIPLIER,
        description="Successive Halving budget = k * num_rows * n_candidates"
    )
    budget_cap: int = Field(
        default=1000,
        description="Upper limit on Successive Halving budget. Prevents excessive CPU time on large columns."
    )
    
    # ========== Runtime State ==========
    
    total_tables: int = Field(default=0)
    total_batches: int = Field(default=0)
    current_batch_index: int = Field(default=0)
    
    current_batch_tables: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Tables in current batch"
    )
    current_batch_results: Dict[str, TableColumnSummaries] = Field(
        default_factory=dict,
        description="Results for current batch"
    )
    
    # Column classifications from classify_columns_node
    column_classifications: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description="Map: table_id -> {column_name -> primitive_class}"
    )
    
    # Cache of processed table IDs (for resume optimization)
    processed_table_ids: Optional[set] = Field(
        default=None,
        description="Set of table IDs already in LanceDB (loaded once, reused across batches)"
    )
    
    # ========== Progress Tracking ==========
    
    completed_tables: int = Field(default=0)
    failed_tables: int = Field(default=0)
    total_columns_analyzed: int = Field(default=0)
    total_llm_calls: int = Field(default=0)
    code_reuse_count: int = Field(default=0, description="Number of times code was reused")
    
    # ========== LLM Stats Timeline (for ablation experiments) ==========
    
    llm_stats_timeline: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Timeline of LLM stats snapshots per batch. Each entry: "
                    "{batch_idx, total_tables, table_ids, cumulative_by_caller}"
    )
    
    checkpoint_paths: List[str] = Field(
        default_factory=list,
        description="Paths to saved checkpoint files"
    )
    
    # ========== Final Output ==========
    
    all_summaries: Dict[str, TableColumnSummaries] = Field(
        default_factory=dict,
        description="Final aggregated results"
    )
    
    # ========== Metadata ==========
    
    start_time: Optional[datetime] = Field(default=None)
    end_time: Optional[datetime] = Field(default=None)
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Error log"
    )
