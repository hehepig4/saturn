"""
Nodes package for Column Summary subgraph.

Four-stage approach:
1. expand_virtual_columns: Extract virtual columns from context (optional)
2. classify_columns: LLM assigns primitive_class to each column
3. analyze_columns (Stage 1): Try to reuse existing code from repository
4. analyze_columns (Stage 2): Generate new code with LLM if no match found
"""

from workflows.population.nodes.load_primitive_tbox import load_primitive_tbox_node
from workflows.population.nodes.load_tables_batch import load_tables_batch_node
from workflows.population.nodes.expand_virtual_columns import expand_virtual_columns_node
from workflows.population.nodes.classify_columns import classify_columns_node
from workflows.population.nodes.analyze_columns import analyze_columns_node
from workflows.population.nodes.save_batch_results import save_batch_results_node
from workflows.population.nodes.aggregate_summaries import aggregate_summaries_node

__all__ = [
    "load_primitive_tbox_node",
    "load_tables_batch_node",
    "expand_virtual_columns_node",
    "classify_columns_node",
    "analyze_columns_node",
    "save_batch_results_node",
    "aggregate_summaries_node",
]
