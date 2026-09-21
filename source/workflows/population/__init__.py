"""
Column Summary Agent Subgraph

Analyzes table columns using multifacet DataProperty approach:

Stage 1: Transform Reuse (0 LLM calls)
- For each applicable DataProperty, search for matching TransformContract
- Select best by pattern match rate
- Apply transform and compute predefined statistics

Stage 2: LLM Transform Generation (1 LLM call per DataProperty)
- Generate new TransformContract if no suitable match found
- LLM generates: pattern + transform expression
- Contract uniquely identified by (primitive_class, data_property, pattern)

Key Design:
- One column → one PrimitiveClass → multiple DataProperties
- Each DataProperty has its own TransformContract(s) for different input formats
- Statistics computation is predefined per range_type

Main exports:
- ColumnSummaryState: Workflow state
- ColumnSummary: Output model for each column
- DataPropertyValue: Value for each DataProperty
- TransformContract: Transform specification
"""

from workflows.population.state import (
    ColumnSummaryState,
    ColumnSummary,
    TableColumnSummaries,
    DataPropertyValue,
)

from workflows.population.graph import (
    build_column_summary_graph,
    run_column_summary_workflow,
)

from workflows.population.contract import (
    TransformContract,
    DataPropertySpec,
)

__all__ = [
    # State
    "ColumnSummaryState",
    "ColumnSummary", 
    "TableColumnSummaries",
    "DataPropertyValue",
    # Graph
    "build_column_summary_graph",
    "run_column_summary_workflow",
    # Contract
    "TransformContract",
    "DataPropertySpec",
]
