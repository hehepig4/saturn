"""
Federated Primitive TBox Generation Graph

Creates a LangGraph workflow for federated TBox generation with:
- Global Agent for initialization, synthesis, and review
- Local Agents for proposals and voting (parallel)
- Fixed iteration loop with voting + review in each iteration

Workflow (Updated with iterative voting + review):
    Phase 0: load_data → cluster_queries → [PARALLEL] cq_gen_group_0, ... → collect_backbone_cqs
    Phase 1: global_init
    [Iteration Loop x n_iterations]
        Phase 2: [PARALLEL] local_propose_group_0, local_propose_group_1, ...
        Phase 3: global_synthesis
        Phase 4: [PARALLEL] local_vote_group_0, ... → aggregate_votes → global_review
    Phase DP: final_dp_generation
    Phase Final: export_tbox → END
"""

from typing import Optional, Dict, Any, List
from loguru import logger
from langgraph.graph import StateGraph, START, END

from .state import FederatedPrimitiveTBoxState
from .nodes import (
    load_data_node,
    cluster_queries_node,
    create_branch_cq_generator,
    collect_backbone_cqs_node,
    global_init_node,
    create_local_proposal_node,
    create_local_proposal_nodes,
    global_synthesis_node,
    create_local_voting_node,
    create_local_voting_nodes,
    aggregate_votes_node,
    export_tbox_node,
    global_review_node,
    insights_synthesizer_node,
)


def create_federated_tbox_graph(n_clusters: int = 3) -> StateGraph:
    """
    Create the Federated Primitive TBox generation workflow graph.
    
    Note: This creates a SINGLE ITERATION graph. For multiple iterations,
    use run_federated_tbox() which wraps this graph in a Python loop.
    
    Graph Structure:
        load_data → cluster_queries 
                          │
              ┌───────────┼───────────┐
              ↓           ↓           ↓
         cq_gen_0     cq_gen_1     cq_gen_{n-1}
              │           │           │
              └───────────┼───────────┘
                          ↓
                   collect_backbone_cqs → global_init
                                              │
                               ┌──────────────┼──────────────┐
                               ↓              ↓              ↓
                    local_propose_0  local_propose_1  local_propose_{n-1}
                               │              │              │
                               └──────────────┼──────────────┘
                                              ↓
                                     global_synthesis
                                              │
                               ┌──────────────┼──────────────┐
                               ↓              ↓              ↓
                        local_vote_0   local_vote_1   local_vote_{n-1}
                               │              │              │
                               └──────────────┼──────────────┘
                                              ↓
                                    aggregate_votes → export_tbox → END
    
    Args:
        n_clusters: Number of parallel local agents (clusters)
        
    Returns:
        Compiled StateGraph workflow
    """
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
    
    logger.info(f"Creating Federated TBox graph with {n_clusters} local agents")
    
    workflow = StateGraph(FederatedPrimitiveTBoxState)
    
    # ========== Add Fixed Nodes ==========
    workflow.add_node("load_data", load_data_node)
    workflow.add_node("cluster_queries", cluster_queries_node)
    workflow.add_node("collect_backbone_cqs", collect_backbone_cqs_node)
    workflow.add_node("global_init", global_init_node)
    workflow.add_node("global_synthesis", global_synthesis_node)
    workflow.add_node("aggregate_votes", aggregate_votes_node)
    workflow.add_node("export_tbox", export_tbox_node)
    
    # ========== Create Parallel Nodes ==========
    group_ids = [f"group_{i}" for i in range(n_clusters)]
    
    # Phase 0b: Parallel CQ Generation Nodes
    for group_id in group_ids:
        cq_gen_node = create_branch_cq_generator(group_id)
        workflow.add_node(f"cq_gen_{group_id}", cq_gen_node)
    
    # Phase 2: Local Proposal Nodes
    for group_id in group_ids:
        proposal_node = create_local_proposal_node(group_id)
        workflow.add_node(f"local_propose_{group_id}", proposal_node)
    
    # Phase 4: Local Voting Nodes
    for group_id in group_ids:
        voting_node = create_local_voting_node(group_id)
        workflow.add_node(f"local_vote_{group_id}", voting_node)
    
    # ========== Define Flow ==========
    
    # Phase 0: Entry → Cluster Queries
    workflow.set_entry_point("load_data")
    workflow.add_edge("load_data", "cluster_queries")
    
    # Phase 0b: Cluster Queries → Fan-out to CQ Generation (PARALLEL)
    for group_id in group_ids:
        workflow.add_edge("cluster_queries", f"cq_gen_{group_id}")
    
    # Phase 0c: Fan-in from CQ Generation → Collect Backbone CQs
    for group_id in group_ids:
        workflow.add_edge(f"cq_gen_{group_id}", "collect_backbone_cqs")
    
    # Phase 1: Collect Backbone → Global Init
    workflow.add_edge("collect_backbone_cqs", "global_init")
    
    # Phase 2: Global Init → Fan-out to Local Proposals (PARALLEL)
    for group_id in group_ids:
        workflow.add_edge("global_init", f"local_propose_{group_id}")
    
    # Phase 3: Fan-in from Local Proposals → Global Synthesis
    for group_id in group_ids:
        workflow.add_edge(f"local_propose_{group_id}", "global_synthesis")
    
    # Phase 4: Global Synthesis → Fan-out to Local Voting (PARALLEL)
    for group_id in group_ids:
        workflow.add_edge("global_synthesis", f"local_vote_{group_id}")
    
    # Phase 4: Fan-in from Local Voting → Aggregate
    for group_id in group_ids:
        workflow.add_edge(f"local_vote_{group_id}", "aggregate_votes")
    
    # Phase 5: Aggregate → Export → END
    workflow.add_edge("aggregate_votes", "export_tbox")
    workflow.add_edge("export_tbox", END)
    
    logger.info(f"Graph created: {len(group_ids)} local agents")
    
    return workflow.compile()


def create_iteration_subgraph(
    n_clusters: int,
    tree_config: Optional[Dict[str, Any]] = None,
) -> StateGraph:
    """
    Create a subgraph for a single Phase 2-3 iteration with hierarchical aggregation.
    
    Used for iteration loop in run_federated_tbox().
    
    Node Types:
        - Leaf: local_propose_* (n_clusters nodes)
        - Synthesizer: synth_L*_* (intermediate aggregation, 0 or more layers)
        - Root: global_synthesis (final decision)
    
    Structure (depth=1, flat):
        dispatch → [PARALLEL] local_propose_* → global_synthesis → END
        
    Structure (depth=2, hierarchical):
        dispatch → [PARALLEL] local_propose_* → [PARALLEL] synth_L1_* → global_synthesis → END
        
    Args:
        n_clusters: Number of parallel local agents (Leaf nodes)
        tree_config: Optional tree structure config from phase0_cq_gen
        
    Returns:
        Compiled StateGraph for one iteration
    """
    from .nodes.phase3_intermediate import (
        create_synthesizer_node,
        build_synthesizer_hierarchy,
        get_root_children,
    )
    
    workflow = StateGraph(FederatedPrimitiveTBoxState)
    
    group_ids = [f"group_{i}" for i in range(n_clusters)]
    depth = tree_config.get("depth", 1) if tree_config else 1
    proposal_capacity = tree_config.get("proposal_capacity", 30) if tree_config else 30
    
    logger.info(f"  Creating iteration subgraph: {n_clusters} leaves, depth={depth}")
    
    # ========== Build Synthesizer hierarchy and compute local_max_proposals ==========
    synthesizer_hierarchy, local_max_proposals = build_synthesizer_hierarchy(
        tree_config, n_clusters, proposal_capacity
    )
    
    logger.info(f"  Local max proposals: {local_max_proposals}")
    
    # Add a dispatch node to fan-out to all local proposals in parallel
    # Also sets local_max_proposals in state for local agents to use
    def dispatch_proposals(state: FederatedPrimitiveTBoxState) -> Dict:
        """Dispatch node to trigger parallel local proposals and set max proposals."""
        return {"local_max_proposals": local_max_proposals}
    
    workflow.add_node("dispatch", dispatch_proposals)
    
    # ========== Add Leaf nodes (local_propose_*) ==========
    for group_id in group_ids:
        proposal_node = create_local_proposal_node(group_id)
        workflow.add_node(f"local_propose_{group_id}", proposal_node)
    
    # ========== Add Synthesizer nodes ==========
    for node_id, node_info in synthesizer_hierarchy.items():
        synth_node = create_synthesizer_node(
            level=node_info["level"],
            aggregator_index=node_info["index"],
            child_node_ids=node_info["children"],
            output_capacity=node_info["output_capacity"],
        )
        workflow.add_node(node_id, synth_node)
        logger.debug(f"    Added synthesizer: {node_id} <- {node_info['children']} (capacity={node_info['output_capacity']})")
    
    # ========== Add Root node (global_synthesis) ==========
    workflow.add_node("global_synthesis", global_synthesis_node)
    
    # ========== Set entry point ==========
    workflow.set_entry_point("dispatch")
    
    # ========== Build edges ==========
    # Fan-out from dispatch to all Leaf nodes (parallel execution)
    for group_id in group_ids:
        workflow.add_edge("dispatch", f"local_propose_{group_id}")
    
    # Hierarchical: Leaf → Synthesizers → Root
    _add_hierarchical_edges(
        workflow=workflow,
        synthesizer_hierarchy=synthesizer_hierarchy,
        tree_config=tree_config,
        n_clusters=n_clusters,
    )
    
    workflow.add_edge("global_synthesis", END)
    
    return workflow.compile()


def _add_hierarchical_edges(
    workflow: StateGraph,
    synthesizer_hierarchy: Dict[str, Dict],
    tree_config: Dict[str, Any],
    n_clusters: int,
) -> None:
    """
    Add edges for hierarchical aggregation.
    
    Flow: Leaf → L1 Synthesizers → L2 Synthesizers → ... → Root
    """
    from .nodes.phase3_intermediate import get_root_children
    
    # If no synthesizers, connect leaves directly to root
    if not synthesizer_hierarchy:
        for i in range(n_clusters):
            workflow.add_edge(f"local_propose_group_{i}", "global_synthesis")
        return
    
    # Add edges from children to their parent synthesizers
    for node_id, node_info in synthesizer_hierarchy.items():
        for child_id in node_info["children"]:
            workflow.add_edge(child_id, node_id)
    
    # Add edges from highest-level synthesizers to root
    root_children = get_root_children(tree_config, n_clusters)
    for child_id in root_children:
        workflow.add_edge(child_id, "global_synthesis")


def run_federated_tbox(
    dataset_name: str = "fetaqa",
    table_store_name: str = "fetaqa_tables_entries",
    query_store_name: str = "fetaqa_train_queries",
    n_clusters: int = 3,
    n_iterations: int = 2,
    target_classes: int = 0,
    scq_per_query: int = 1,
    vcq_per_query: int = 1,
    agent_cq_capacity: int = 30,
    global_agent_span: int = 10,
    proposal_capacity: int = 30,
    llm_purpose: str = "gemini",
    max_tables: Optional[int] = None,
    max_queries: Optional[int] = None,
    dp_only: bool = False,
    cq_max_concurrent: int = 16,
    dp_max_concurrent: int = 5,
    rag_type: str = "hybrid",
) -> FederatedPrimitiveTBoxState:
    """
    Run the Federated Primitive TBox generation workflow with iterations.
    
    This implements the full workflow with an external Python loop for
    Phase 2-3 iterations, which is simpler than LangGraph conditional edges.
    
    Workflow:
        1. Run initialization graph (Phase 0-1)
        2. Loop n_iterations times:
           a. Run Phase 2 (local proposals) in parallel
           b. Run Phase 3 (global synthesis)
        3. Run finalization graph (Phase 4-5)
    
    CQ Generation:
        - Per-query: Each query generates scq_per_query SCQs + vcq_per_query VCQs
        - Async parallel LLM calls for efficiency
        - Caching to avoid regenerating CQs
    
    Args:
        dataset_name: Name of the dataset
        table_store_name: LanceDB table name for tables
        query_store_name: LanceDB table name for queries
        n_clusters: Number of local agents (0 = auto-compute from capacity)
        n_iterations: Number of Phase 2-3 iterations
        target_classes: Target number of classes (0 = no specific target)
        scq_per_query: Number of Scoping CQs per query (default: 1)
        vcq_per_query: Number of Validating CQs per query (default: 1)
        agent_cq_capacity: K parameter (max CQs per agent)
        global_agent_span: B parameter (max branching factor)
        proposal_capacity: P parameter (max class proposals for global synthesis)
        llm_purpose: LLM purpose key
        max_tables: Max tables to load (None = all, not used in Stage 1)
        max_queries: Max queries to load (None = all)
        dp_only: If True, skip iterations and only run DP generation (load from LanceDB)
        cq_max_concurrent: Maximum concurrent LLM calls for CQ generation
        dp_max_concurrent: Maximum concurrent LLM calls for data property generation
        rag_type: Retrieval type for hard negative sampling ('bm25', 'vector', 'hybrid')
        
    Returns:
        Final workflow state with Federated TBox
    """
    logger.info("=" * 80)
    if dp_only:
        logger.info("Federated Primitive TBox - DP GENERATION ONLY")
    else:
        logger.info("Federated Primitive TBox Generation Workflow")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_name}")
    if not dp_only:
        logger.info(f"Initial Clusters: {n_clusters} (0 = auto)")
        logger.info(f"Iterations: {n_iterations}")
        logger.info(f"CQs per Query: SCQ={scq_per_query}, VCQ={vcq_per_query}")
        logger.info(f"Agent CQ Capacity (K): {agent_cq_capacity}")
        logger.info(f"Global Agent Span (B): {global_agent_span}")
        logger.info(f"Proposal Capacity (P): {proposal_capacity}")
        logger.info(f"CQ Max Concurrent: {cq_max_concurrent}")
    logger.info(f"DP Max Concurrent: {dp_max_concurrent}")
    logger.info(f"LLM Purpose: {llm_purpose}")
    if dp_only:
        logger.info("TBox Source: LanceDB (latest snapshot)")
    logger.info(f"RAG Type: {rag_type}")
    logger.info("=" * 80)
    
    # ========== Initialize State ==========
    initial_state = FederatedPrimitiveTBoxState(
        dataset_name=dataset_name,
        table_store_name=table_store_name,
        query_store_name=query_store_name,
        n_clusters=n_clusters,
        n_iterations=n_iterations,
        target_classes=target_classes,
        scq_per_query=scq_per_query,
        vcq_per_query=vcq_per_query,
        agent_cq_capacity=agent_cq_capacity,
        global_agent_span=global_agent_span,
        proposal_capacity=proposal_capacity,
        llm_purpose=llm_purpose,
        max_tables=max_tables,
        max_queries=max_queries,
        cq_max_concurrent=cq_max_concurrent,
        dp_max_concurrent=dp_max_concurrent,
        rag_type=rag_type,
    )
    
    # Import utilities
    from workflows.conceptualization.utils.snapshot import save_tbox_snapshot
    from workflows.conceptualization.nodes.phase_dp_final import final_dp_generation_node
    from utils.tree_formatter import format_class_hierarchy
    
    # ========== DP-Only Mode: Load existing TBox and generate DPs ==========
    if dp_only:
        return _run_dp_only_mode(
            initial_state=initial_state,
            dataset_name=dataset_name,
            final_dp_generation_node=final_dp_generation_node,
            save_tbox_snapshot=save_tbox_snapshot,
        )
    
    # ========== Phase 0a: Load Data & Cluster ==========
    logger.info("\n" + "=" * 40)
    logger.info("PHASE 0a: Load Data & Cluster Queries")
    logger.info("=" * 40)
    
    # First run: load data and cluster queries to get actual cluster count
    cluster_graph = _create_cluster_graph()
    state = cluster_graph.invoke(initial_state)
    state = FederatedPrimitiveTBoxState.model_validate(state)
    
    # Determine actual n_clusters from clustering
    actual_n_clusters = len(state.cluster_assignments)
    if actual_n_clusters == 0:
        logger.error("No clusters created, cannot proceed")
        return state
    
    logger.info(f"Actual clusters: {actual_n_clusters}")
    
    # ========== Phase 0b-1: CQ Generation & Global Init ==========
    logger.info("\n" + "=" * 40)
    logger.info("PHASE 0b-1: Parallel CQ Generation & Global Init")
    logger.info("=" * 40)
    
    # Now create CQ generation graph with actual cluster count
    cq_gen_graph = _create_cq_gen_graph(actual_n_clusters)
    state = cq_gen_graph.invoke(state)
    state = FederatedPrimitiveTBoxState.model_validate(state)
    
    # Print initial class hierarchy tree
    if state.initial_tbox and state.initial_tbox.get("classes"):
        n_init_classes = len(state.initial_tbox["classes"])
        n_init_dps = len(state.initial_tbox.get("data_properties", []))
        logger.info(f"  Initial TBox: {n_init_classes} classes, {n_init_dps} DataProperties")
        
        class_tree = format_class_hierarchy(
            state.initial_tbox["classes"],
            include_column_root=True,
            max_desc_length=None
        )
        logger.info("\n  === Initial Class Hierarchy ===")
        for line in class_tree.split("\n"):
            logger.info(f"  {line}")
        logger.info("  " + "=" * 40)
    
    # Save initial TBox snapshot (iteration 0)
    if state.initial_tbox:
        save_tbox_snapshot(
            tbox=state.initial_tbox,
            dataset_name=dataset_name,
            iteration=0,
            notes=f"Initial TBox from Phase 1 ({len(state.backbone_cqs)} backbone CQs)"
        )
        # Record iteration 0 in review_log for experiment analysis
        initial_n_classes = len(state.initial_tbox.get("classes", []))
        initial_review = {
            "iteration": 0,
            "actions": [],
            "summary": f"Initial skeleton ({initial_n_classes} classes)",
            "n_actions": 0,
            "voting_summary": [],
            "raw_votes": {},
            "n_classes": initial_n_classes,
            "n_voting_agents": 0,
        }
        state = state.model_copy(update={"review_log": [initial_review]})
        logger.info(f"  Iteration 0 (skeleton): {initial_n_classes} classes")
    
    # ========== Phase 2-5: Iteration Loop (Proposals → Synthesis → Voting → Review) ==========
    for iteration in range(n_iterations):
        logger.info("\n" + "=" * 40)
        logger.info(f"ITERATION {iteration + 1}/{n_iterations}")
        logger.info("=" * 40)
        
        # ===== Phase 2-3: Local Proposals + Global Synthesis =====
        logger.info("  Phase 2-3: Local Proposals + Global Synthesis")
        iter_graph = create_iteration_subgraph(
            n_clusters=actual_n_clusters,
            tree_config=state.tree_config,
        )
        state = iter_graph.invoke(state)
        state = FederatedPrimitiveTBoxState.model_validate(state)
        
        # Log progress after synthesis
        n_classes = len(state.current_tbox.get("classes", [])) if state.current_tbox else 0
        logger.info(f"  After synthesis: {n_classes} classes")
        
        # ===== Phase 4: Voting + Review =====
        logger.info("  Phase 4-5: Voting + Review")
        vote_review_graph = _create_voting_review_subgraph(actual_n_clusters)
        state = vote_review_graph.invoke(state)
        state = FederatedPrimitiveTBoxState.model_validate(state)
        
        # Log progress after review
        n_classes_after_review = len(state.current_tbox.get("classes", [])) if state.current_tbox else 0
        if n_classes_after_review != n_classes:
            logger.info(f"  After review: {n_classes_after_review} classes (changed from {n_classes})")
        else:
            logger.info(f"  After review: {n_classes_after_review} classes (no changes)")
        
        # Print class hierarchy tree
        if state.current_tbox and state.current_tbox.get("classes"):
            class_tree = format_class_hierarchy(
                state.current_tbox["classes"],
                include_column_root=True,
                max_desc_length=None
            )
            logger.info(f"\n  === Class Hierarchy (Iteration {iteration + 1}) ===")
            for line in class_tree.split("\n"):
                logger.info(f"  {line}")
            logger.info("  " + "=" * 40)
        
        # Save TBox snapshot after each iteration (classes only)
        if state.current_tbox:
            save_tbox_snapshot(
                tbox=state.current_tbox,
                dataset_name=dataset_name,
                iteration=iteration + 1,
                notes=f"After iteration {iteration + 1}: {n_classes_after_review} classes (with voting+review)"
            )
    
    # ========== Phase DP: Final DataProperty Generation ==========
    logger.info("\n" + "=" * 40)
    logger.info("PHASE DP: Final DataProperty Generation")
    logger.info("=" * 40)
    
    # Generate DPs for all final classes in parallel
    dp_result = final_dp_generation_node(state)
    
    # Update state with DPs
    if "current_tbox" in dp_result:
        state.current_tbox = dp_result["current_tbox"]
    
    n_classes = len(state.current_tbox.get("classes", [])) if state.current_tbox else 0
    n_props = len(state.current_tbox.get("data_properties", [])) if state.current_tbox else 0
    logger.info(f"  Final TBox: {n_classes} classes, {n_props} DataProperties")
    
    # Save final TBox snapshot with DPs
    if state.current_tbox:
        save_tbox_snapshot(
            tbox=state.current_tbox,
            dataset_name=dataset_name,
            iteration=n_iterations + 1,  # iteration N+1 = final with DPs
            notes=f"Final TBox with DPs: {n_classes} classes, {n_props} props"
        )
    
    # ========== Phase Final: Export ==========
    # Note: Voting and Review now happen in each iteration loop above
    logger.info("\n" + "=" * 40)
    logger.info("PHASE FINAL: Export TBox")
    logger.info("=" * 40)
    
    # Directly call export node (voting already done in iterations)
    export_result = export_tbox_node(state)
    
    # Update state with export results
    if "export_success" in export_result:
        state.export_success = export_result["export_success"]
    if "export_error" in export_result:
        state.export_error = export_result["export_error"]
    if "owl_path" in export_result:
        state.owl_path = export_result["owl_path"]
    if "tbox_path" in export_result:
        state.tbox_path = export_result["tbox_path"]
    
    logger.info("\n" + "=" * 80)
    logger.info("WORKFLOW COMPLETE")
    logger.info("=" * 80)
    
    return state


def _run_dp_only_mode(
    initial_state: FederatedPrimitiveTBoxState,
    dataset_name: str,
    final_dp_generation_node,
    save_tbox_snapshot,
) -> FederatedPrimitiveTBoxState:
    """
    Run DP-only mode: Load existing TBox from LanceDB and regenerate DataProperties.
    
    Args:
        initial_state: Initial state with config
        dataset_name: Dataset name for loading from LanceDB
        final_dp_generation_node: DP generation function
        save_tbox_snapshot: Snapshot save function
        
    Returns:
        State with updated TBox (classes + generated DPs)
    """
    from workflows.conceptualization.utils.snapshot import load_tbox_snapshot
    
    state = initial_state
    
    # ========== Load Existing TBox from LanceDB ==========
    logger.info("\n" + "=" * 40)
    logger.info("DP-ONLY: Loading Existing TBox from LanceDB")
    logger.info("=" * 40)
    
    logger.info(f"  Dataset: {dataset_name}")
    tbox = load_tbox_snapshot(dataset_name)
    
    if not tbox:
        logger.error(f"No TBox snapshot found for dataset '{dataset_name}' in LanceDB")
        state.export_error = f"No TBox snapshot found for dataset '{dataset_name}'"
        state.export_success = False
        return state
    
    # Normalize classes format (LanceDB uses different field names)
    classes = tbox.get("classes", [])
    normalized_classes = []
    for cls in classes:
        normalized_classes.append({
            "name": cls.get("name", cls.get("class_name", "")),
            "description": cls.get("description", cls.get("definition", "")),
            "parent_class": cls.get("parent_class", cls.get("parent", "Column")),
            "parent_classes": cls.get("parent_classes", [cls.get("parent", "Column")]),
        })
    
    if not normalized_classes:
        logger.error("TBox has no classes")
        state.export_error = "TBox has no classes"
        state.export_success = False
        return state
    
    # Set up state with loaded classes (no DPs - will be regenerated)
    state.current_tbox = {
        "classes": normalized_classes,
        "data_properties": [],
    }
    state.initial_tbox = state.current_tbox.copy()
    
    n_classes = len(normalized_classes)
    logger.info(f"  ✓ Loaded {n_classes} classes (DPs will be regenerated)")
    
    # ========== Generate DataProperties ==========
    logger.info("\n" + "=" * 40)
    logger.info("DP-ONLY: Generating DataProperties")
    logger.info("=" * 40)
    
    dp_result = final_dp_generation_node(state)
    
    # Update state with DPs
    if "current_tbox" in dp_result:
        state.current_tbox = dp_result["current_tbox"]
    
    n_props = len(state.current_tbox.get("data_properties", [])) if state.current_tbox else 0
    logger.info(f"  Generated {n_props} DataProperties for {n_classes} classes")
    
    # Save snapshot
    if state.current_tbox:
        save_tbox_snapshot(
            tbox=state.current_tbox,
            dataset_name=dataset_name,
            iteration=999,  # Special marker for DP-only
            notes=f"DP-only regeneration: {n_classes} classes, {n_props} props"
        )
    
    # ========== Export ==========
    logger.info("\n" + "=" * 40)
    logger.info("DP-ONLY: Exporting TBox")
    logger.info("=" * 40)
    
    export_result = export_tbox_node(state)
    
    if "export_success" in export_result:
        state.export_success = export_result["export_success"]
    if "export_error" in export_result:
        state.export_error = export_result["export_error"]
    if "owl_path" in export_result:
        state.owl_path = export_result["owl_path"]
    if "tbox_path" in export_result:
        state.tbox_path = export_result["tbox_path"]
    
    logger.info("\n" + "=" * 80)
    logger.info("DP-ONLY COMPLETE")
    logger.info("=" * 80)
    
    return state


def _create_cluster_graph() -> StateGraph:
    """Create graph for Phase 0a: Load Data & Cluster Queries.
    
    This runs first to determine actual cluster count.
    """
    workflow = StateGraph(FederatedPrimitiveTBoxState)
    
    workflow.add_node("load_data", load_data_node)
    workflow.add_node("cluster_queries", cluster_queries_node)
    
    workflow.set_entry_point("load_data")
    workflow.add_edge("load_data", "cluster_queries")
    workflow.add_edge("cluster_queries", END)
    
    return workflow.compile()


def _create_cq_gen_graph(n_clusters: int) -> StateGraph:
    """Create graph for Phase 0b-1: Parallel CQ Generation & Global Init.
    
    Args:
        n_clusters: Actual number of clusters from clustering step
    """
    workflow = StateGraph(FederatedPrimitiveTBoxState)
    
    # Parallel CQ generation nodes
    group_ids = [f"group_{i}" for i in range(n_clusters)]
    for group_id in group_ids:
        cq_gen_node = create_branch_cq_generator(group_id)
        workflow.add_node(f"cq_gen_{group_id}", cq_gen_node)
    
    # Collection and initialization
    workflow.add_node("collect_backbone_cqs", collect_backbone_cqs_node)
    workflow.add_node("global_init", global_init_node)
    
    # Entry point is the first CQ generation node
    workflow.set_entry_point(f"cq_gen_group_0")
    
    # All CQ generation nodes run in parallel from entry
    for group_id in group_ids[1:]:
        workflow.add_edge(START, f"cq_gen_{group_id}")
    
    # Fan-in: parallel CQ generation → collect backbone
    for group_id in group_ids:
        workflow.add_edge(f"cq_gen_{group_id}", "collect_backbone_cqs")
    
    workflow.add_edge("collect_backbone_cqs", "global_init")
    workflow.add_edge("global_init", END)
    
    return workflow.compile()


def _create_voting_review_subgraph(n_clusters: int) -> StateGraph:
    """Create voting + review subgraph for iterative TBox refinement.
    
    This subgraph performs:
    1. Parallel local voting (each group votes on current TBox)
    2. Vote aggregation
    3. Global review (uses voting stats to optimize TBox)
    
    Args:
        n_clusters: Number of clusters/groups
        
    Returns:
        Compiled StateGraph for voting + review
    """
    workflow = StateGraph(FederatedPrimitiveTBoxState)
    
    group_ids = [f"group_{i}" for i in range(n_clusters)]
    
    # Add dispatch node for parallel voting fan-out
    def dispatch_voting(state: FederatedPrimitiveTBoxState) -> Dict:
        """Dispatch node to trigger parallel local voting."""
        return {}  # No state change, just a synchronization point
    
    workflow.add_node("dispatch_voting", dispatch_voting)
    
    # Add voting nodes
    for group_id in group_ids:
        voting_node = create_local_voting_node(group_id)
        workflow.add_node(f"local_vote_{group_id}", voting_node)
    
    workflow.add_node("aggregate_votes", aggregate_votes_node)
    workflow.add_node("global_review", global_review_node)
    workflow.add_node("insights_synthesizer", insights_synthesizer_node)
    
    # Flow: dispatch_voting → [PARALLEL] local_vote_* → aggregate_votes → global_review → insights_synthesizer
    workflow.set_entry_point("dispatch_voting")
    
    # Fan-out from dispatch to all local votes (parallel execution)
    for group_id in group_ids:
        workflow.add_edge("dispatch_voting", f"local_vote_{group_id}")
    
    # Fan-in from local votes to aggregate
    for group_id in group_ids:
        workflow.add_edge(f"local_vote_{group_id}", "aggregate_votes")
    
    workflow.add_edge("aggregate_votes", "global_review")
    workflow.add_edge("global_review", "insights_synthesizer")
    workflow.add_edge("insights_synthesizer", END)
    
    return workflow.compile()


# ============== Convenience Functions ==============

def invoke_federated_tbox(
    dataset_name: str = "fetaqa",
    **kwargs
) -> FederatedPrimitiveTBoxState:
    """
    Convenience wrapper for run_federated_tbox().
    
    Args:
        dataset_name: Dataset name
        **kwargs: Additional arguments passed to run_federated_tbox()
        
    Returns:
        Final workflow state
    """
    return run_federated_tbox(dataset_name=dataset_name, **kwargs)
