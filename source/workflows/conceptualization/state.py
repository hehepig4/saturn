"""
State Definition for Federated Primitive TBox Generation

Defines FederatedPrimitiveTBoxState with:
- Agent tree configuration (K, B parameters)
- Phase-specific state fields
- Reducer-annotated fields for parallel writes
"""

from typing import Dict, Any, List, Optional, Annotated
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


# ========== Reducers for Parallel State Merging ==========

def merge_dict_reducer(existing: Dict, update: Dict) -> Dict:
    """Merge two dictionaries. Used for parallel branch results.
    
    Key collision: update value overwrites existing.
    """
    if existing is None:
        return update or {}
    if update is None:
        return existing
    return {**existing, **update}


def append_list_reducer(existing: List, update: List) -> List:
    """Append lists. Used for accumulating results from parallel branches."""
    if existing is None:
        return update or []
    if update is None:
        return existing
    return existing + update


class FederatedPrimitiveTBoxState(BaseModel):
    """
    State for Federated Primitive TBox generation workflow.
    
    Workflow Phases:
        Phase 0: CQ Generation - Generate Competency Questions
        Phase 1: Global Init - Create initial TBox from backbone CQs
        Phase 2: Local Proposals - Each cluster proposes changes
        Phase 3: Global Synthesis - Hierarchical aggregation
        Phase 4: Local Voting - Binary usefulness voting
        Phase 5: Export - Generate OWL file
    """
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        extra='allow'
    )
    
    # ========== Input Configuration ==========
    
    dataset_name: str = Field(
        ...,
        description="Dataset name (e.g., 'fetaqa')"
    )
    
    table_store_name: str = Field(
        ...,
        description="LanceDB table store name"
    )
    
    query_store_name: str = Field(
        ...,
        description="LanceDB query store name"
    )
    
    db_path: str = Field(
        default="data/lake",
        description="Path to LanceDB database"
    )
    
    # ========== Agent Tree Configuration ==========
    
    agent_cq_capacity: int = Field(
        default=30,
        description="K: Maximum CQs a single agent can handle"
    )
    
    global_agent_span: int = Field(
        default=10,
        description="B: Max branching factor (max children per aggregator). Larger = fewer LLM calls."
    )
    
    proposal_capacity: int = Field(
        default=30,
        description="P: Max class proposals for global synthesis per iteration. "
                    "Each local agent gets proposal_capacity / n_clusters proposals to submit."
    )
    
    n_iterations: int = Field(
        default=3,
        description="Number of Phase 2-3 iterations"
    )
    
    target_classes: int = Field(
        default=0,
        description="Target number of classes. 0 = no specific target (let agents decide). "
                    "Used to guide agents on the desired scale of the ontology."
    )
    
    # ========== Parallel Execution Configuration ==========
    
    cq_max_concurrent: int = Field(
        default=16,
        description="Maximum concurrent LLM calls for CQ generation (Phase 0)"
    )
    
    dp_max_concurrent: int = Field(
        default=5,
        description="Maximum concurrent LLM calls for data property generation"
    )
    
    # ========== Data Loading Configuration ==========
    
    max_tables: Optional[int] = Field(
        default=None,
        description="Maximum tables to load (None = all)"
    )
    
    max_queries: Optional[int] = Field(
        default=None,
        description="Maximum queries to load (None = all)"
    )
    
    tables: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Loaded table records"
    )
    
    queries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Loaded query records"
    )
    
    # ========== Query Embedding Config ==========
    
    query_embedding_field: str = Field(
        default="query_text_embedding",
        description="Field name for query embedding in LanceDB"
    )
    
    table_embedding_field: str = Field(
        default="table_text_embedding",
        description="Field name for table embedding in LanceDB"
    )
    
    retrieval_top_k: int = Field(
        default=3,
        description="Number of tables to retrieve per query"
    )
    
    rag_type: str = Field(
        default="hybrid",
        description="Retrieval type for hard negative sampling: 'bm25', 'vector', or 'hybrid'"
    )
    
    sampling_seed: Optional[int] = Field(
        default=42,
        description="Random seed for reproducible sampling"
    )
    
    # ========== Phase 0: CQ Generation (per-query control) ==========
    
    scq_per_query: int = Field(
        default=1,
        description="Number of Scoping CQs to generate per query (default: 1)"
    )
    
    vcq_per_query: int = Field(
        default=1,
        description="Number of Validating CQs to generate per query (default: 1)"
    )
    
    competency_questions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All generated Competency Questions (merged from branches)"
    )
    
    cq_statistics: Dict[str, Any] = Field(
        default_factory=dict,
        description="CQ generation statistics"
    )
    
    # ========== Clustering (Query-based, not CQ-based) ==========
    
    n_clusters: int = Field(
        default=0,
        description="Number of query clusters. 0 = auto-compute from (N × cq_per_query / K)"
    )
    
    min_cluster_ratio: float = Field(
        default=0.5,
        description="Min cluster size as ratio of average (0.5 = clusters can be 50%-150% of avg)"
    )
    
    cluster_labels: Optional[List[int]] = Field(
        default=None,
        description="Cluster label for each query"
    )
    
    cluster_assignments: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Cluster assignments: {group_id: {query_indices, centroid, query_table_pairs}}"
    )
    
    # Branch CQs from parallel generation (with reducer for parallel writes)
    branch_cqs: Annotated[Dict[str, List[Dict]], merge_dict_reducer] = Field(
        default_factory=dict,
        description="CQs from each parallel branch: {group_id: [cqs]}"
    )
    
    # Branch CQ caches for unified saving (with reducer for parallel writes)
    branch_cq_caches: Annotated[Dict[str, Dict[str, List[Dict]]], merge_dict_reducer] = Field(
        default_factory=dict,
        description="CQ caches from each branch: {group_id: {query_hash: [cqs]}} - merged and saved in collect_backbone_cqs"
    )
    
    # ========== Phase 1: Global Initialization ==========
    
    initial_tbox: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Initial TBox: {classes: [...], data_properties: [...]}"
    )
    
    backbone_cqs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Backbone CQs used for initialization"
    )
    
    # ========== Phase 2: Local Proposals (with reducer) ==========
    
    local_proposals: Annotated[Dict[str, List[Dict]], merge_dict_reducer] = Field(
        default_factory=dict,
        description="Local proposals by group: {group_id: [ClassProposal.dict(), ...]}"
    )
    
    # ========== Phase 3: Global Synthesis ==========
    
    current_tbox: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Current TBox state (updated each iteration)"
    )
    
    synthesized_actions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Actions from latest synthesis"
    )
    
    synthesis_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Decision history across iterations"
    )
    
    # Agent tree structure (built at runtime)
    agent_tree: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Hierarchical agent tree structure"
    )
    
    # Tree config from tree_design (computed at clustering)
    tree_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Tree structure config: {n_clusters, branching_factor, depth, levels, ...}"
    )
    
    # Intermediate proposals from hierarchical aggregation (with reducer for parallel writes)
    intermediate_proposals: Annotated[Dict[str, List[Dict]], merge_dict_reducer] = Field(
        default_factory=dict,
        description="Intermediate aggregated proposals: {layer_node_id: [proposal_dicts]}"
    )
    
    # ========== Phase 4: Voting (with reducer) ==========
    
    local_votes: Annotated[Dict[str, Dict[str, int]], merge_dict_reducer] = Field(
        default_factory=dict,
        description="Voting results: {group_id: {class_name: 0|1}}"
    )
    
    aggregated_votes: Dict[str, float] = Field(
        default_factory=dict,
        description="Aggregated class scores: {class_name: weighted_score}"
    )
    
    # ========== Phase 5: Review Output (for insights synthesizer) ==========
    
    review_output: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Latest review output: {iteration, actions, summary, n_actions}"
    )
    
    # Review history across iterations - for ablation experiment analysis
    review_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Review history with voting: [{iteration, actions, voting_summary, raw_votes, ...}]"
    )
    
    # ========== Phase 5b: Global Insights (RNN-style memory) ==========
    
    global_insights: Optional[Any] = Field(
        default=None,
        description="GlobalInsights object - compressed memory for guiding agents"
    )
    
    # ========== Phase 6: Export ==========
    
    final_tbox: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Final TBox with voting annotations"
    )
    
    tbox_owl: str = Field(
        default="",
        description="OWL Manchester Syntax output"
    )
    
    owl_path: str = Field(
        default="",
        description="Path to exported OWL file"
    )
    
    report_path: str = Field(
        default="",
        description="Path to JSON report file"
    )
    
    export_success: bool = Field(
        default=False,
        description="Whether export succeeded"
    )
    
    export_error: Optional[str] = Field(
        default=None,
        description="Error message if export failed"
    )
    
    export_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Export metadata (owl_path, statistics, etc.)"
    )
    
    # ========== Iteration State ==========
    
    current_iteration: int = Field(
        default=0,
        description="Current iteration number (0-indexed)"
    )
    
    iteration_metrics: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Metrics from each iteration"
    )
    
    # ========== Error Tracking ==========
    
    phase_errors: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Errors by phase: {'phase_2': ['group_0 failed: ...']}"
    )
    
    partial_success: bool = Field(
        default=False,
        description="True if some branches succeeded despite errors"
    )
    
    # ========== Dynamic Max Proposals per Local Agent ==========
    
    local_max_proposals: Dict[str, int] = Field(
        default_factory=dict,
        description="Dynamic max proposals per local agent: {'group_0': 5, 'group_1': 5, ...}"
    )
    
    # ========== Workflow Metadata ==========
    
    success: bool = Field(
        default=False,
        description="Overall workflow success flag"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if workflow failed"
    )
    
    started_at: Optional[datetime] = Field(
        default=None,
        description="Workflow start timestamp"
    )
    
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Workflow completion timestamp"
    )
    
    # ========== Debug & LLM Configuration ==========
    
    debug_output_dir: Optional[str] = Field(
        default=None,
        description="If set, dump intermediate JSON to this directory"
    )
    
    llm_purpose: str = Field(
        default="gemini",
        description="LLM purpose key from config"
    )
    
    llm_override_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Override LLM configuration"
    )
    
    # ========== Computed Properties ==========
    
    def get_max_proposals_for_group(self, group_id: str) -> int:
        """
        Get max proposals for a specific local agent group.
        
        Uses dynamically computed local_max_proposals if available,
        otherwise falls back to static calculation.
        
        Args:
            group_id: The group ID (e.g., "group_0")
            
        Returns:
            Max proposals this group can submit
        """
        # Use dynamic value if available
        if self.local_max_proposals and group_id in self.local_max_proposals:
            return self.local_max_proposals[group_id]
        
        # Fallback to static calculation
        span = self.global_agent_span if self.global_agent_span > 0 else 6
        return max(1, self.proposal_capacity // span)
    

