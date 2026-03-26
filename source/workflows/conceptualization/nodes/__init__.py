"""
Nodes Package for Federated Primitive TBox

LangGraph nodes for each phase of the workflow.
"""

from .load_data import load_data_node
from .phase0_cq_gen import (
    cluster_queries_node,
    create_branch_cq_generator,
    collect_backbone_cqs_node,
)
from .phase1_init import global_init_node
from .phase2_local import create_local_proposal_node, create_local_proposal_nodes
from .phase3_intermediate import (
    create_synthesizer_node,
    build_synthesizer_hierarchy,
    get_root_children,
)
from .phase3_synthesis import global_synthesis_node
from .phase4_voting import (
    create_local_voting_node,
    create_local_voting_nodes,
    aggregate_votes_node,
)
from .phase5_review import global_review_node
from .phase5b_insights import insights_synthesizer_node
from .phase5_export import export_tbox_node

__all__ = [
    # Data loading
    "load_data_node",
    # Phase 0: Query clustering + parallel CQ generation
    "cluster_queries_node",
    "create_branch_cq_generator",
    "collect_backbone_cqs_node",
    # Phase 1
    "global_init_node",
    # Phase 2
    "create_local_proposal_node",
    "create_local_proposal_nodes",
    # Phase 3 - Intermediate (Synthesizers)
    "create_synthesizer_node",
    "build_synthesizer_hierarchy",
    "get_root_children",
    # Phase 3 - Synthesis (Root)
    "global_synthesis_node",
    # Phase 4: Voting
    "create_local_voting_node",
    "create_local_voting_nodes",
    "aggregate_votes_node",
    # Phase 5: Review
    "global_review_node",
    # Phase 5b: Insights Synthesizer
    "insights_synthesizer_node",
    # Phase 6: Export
    "export_tbox_node",
]
