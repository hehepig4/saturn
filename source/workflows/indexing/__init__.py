"""
Indexing Package — Stage 3 + Stage 4

Sub-packages:
    - annotation: Layer 2 Defined Class annotation (Stage 3)
    - (root): Multi-view summary generation for retrieval (Stage 4)

Stage 3 — Annotation (indexing.annotation):
    1. Load Data: Load tables, column summaries, and Layer 1 primitives
    2. Annotate Tables: LLM-based column annotation with P0/P1 enhancements
    3. Create Defined Classes: Build Column and Table Defined Classes
    4. Export Layer 2: Store to LanceDB and export OWL

Stage 4 — Summarization:
    1. Load Layer 2 Data: Load column mappings, table classes, column summaries
    2. Generate Summaries: Create multi-view summaries for each table
    3. Export Summaries: Save to JSON and Parquet formats
"""

from .graph import (
    create_table_summarization_graph,
    invoke_table_summarization,
)
from .state import (
    TableSummarizationState,
    TableSummary,
    SummarizationOutput,
)

__all__ = [
    # Summarization Graph (Stage 4)
    "create_table_summarization_graph",
    "invoke_table_summarization",
    # State
    "TableSummarizationState",
    # Models
    "TableSummary",
    "SummarizationOutput",
]
