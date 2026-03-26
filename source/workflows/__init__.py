"""
Workflows Package

Contains LangGraph workflows for the UPO pipeline:

- common: Shared utilities, decorators, helpers
- conceptualization: Federated TBox generation (Stage 1)
- population: Column statistics analysis (Stage 2, was column_summary)
- indexing: Layer 2 annotation + summarization (Stage 3+4)
  - indexing.annotation: Layer 2 Defined Class annotation
- retrieval: Retrieval workflows and query analysis (Stage 5)
"""

from .common import graph_node
from .population import build_column_summary_graph
from .indexing.annotation import create_table_discovery_layer2_graph
from .indexing import create_table_summarization_graph

__all__ = [
    # Decorators
    "graph_node",
    # Graphs
    "build_column_summary_graph",
    "create_table_discovery_layer2_graph",
    "create_table_summarization_graph",
]
