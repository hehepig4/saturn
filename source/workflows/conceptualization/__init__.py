"""
Federated Primitive TBox Generation Subgraph

A Global-Local collaborative framework for primitive ontology generation.

Architecture:
    Phase 0: CQ Generation
    Phase 1: Global Initialization (CQ-driven)
    Phase 2: Local Proposals (parallel)
    Phase 3: Hierarchical Global Synthesis
    Phase 4: Local Voting (parallel)
    Phase 5: Aggregate & Export

Key Features:
    - CQ-Driven methodology throughout
    - Configurable agent tree with predefined bounds (K, B parameters)
    - Single inheritance for Primitive Classes
    - Binary voting (0/1) for usefulness scoring
    - Object Property toggle (only hasColumn by default)
"""

from .graph import (
    create_federated_tbox_graph,
    run_federated_tbox,
    invoke_federated_tbox,
)
from .state import FederatedPrimitiveTBoxState

__all__ = [
    "create_federated_tbox_graph",
    "run_federated_tbox",
    "invoke_federated_tbox",
    "FederatedPrimitiveTBoxState",
]
