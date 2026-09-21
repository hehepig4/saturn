"""
Intermediate Synthesis Nodes for Hierarchical Aggregation.

This module provides Synthesizer nodes that aggregate proposals from child nodes
in the agent tree hierarchy.

Node Types:
- Leaf: local_propose_* (defined in phase2_local.py)
- Synthesizer: intermediate_synth_L*_* (defined here)
- Root: global_synthesis (defined in phase3_synthesis.py)

Synthesizer nodes use the same one-shot synthesis logic as Root, but:
1. Output ClassProposal (not SynthesizedAction applied to TBox)
2. Output capacity is limited by proposal_capacity // n_siblings at their parent
"""

from typing import Dict, Any, List, Callable, Tuple
from loguru import logger

from workflows.conceptualization.state import FederatedPrimitiveTBoxState


def create_synthesizer_node(
    level: int,
    aggregator_index: int,
    child_node_ids: List[str],
    output_capacity: int,
) -> Callable[[FederatedPrimitiveTBoxState], Dict[str, Any]]:
    """
    Factory function to create a Synthesizer node for intermediate aggregation.
    
    Synthesizers:
    1. Collect proposals from child nodes (Leafs or lower-level Synthesizers)
    2. Use shared _synthesize_proposals_core() for one-shot synthesis
    3. Convert SynthesizedActions back to ClassProposals for next level
    4. Output at most output_capacity proposals
    
    NOTE: Unlike Root, Synthesizers do NOT apply actions to TBox.
          They pass synthesized proposals to their parent synthesizer or root.
    
    Args:
        level: The level in the tree (1, 2, ... where 0 is leaf level)
        aggregator_index: Index of this synthesizer at its level
        child_node_ids: List of child node IDs that feed into this synthesizer
        output_capacity: Max proposals this synthesizer can output
                        (calculated as proposal_capacity // n_siblings at parent)
    
    Returns:
        A node function that synthesizes and outputs proposals
    """
    node_id = f"synth_L{level}_{aggregator_index}"
    
    def synthesizer_node(state: FederatedPrimitiveTBoxState) -> Dict[str, Any]:
        """
        Collect proposals from children, synthesize, and output merged proposals.
        """
        from llm.manager import get_llm_by_purpose
        from workflows.conceptualization.schemas.proposals import ClassProposal
        from workflows.conceptualization.nodes.phase3_synthesis import (
            _synthesize_proposals_core,
            _action_to_proposal,
            _collect_proposals_from_children,
        )
        
        logger.info(f"  [{node_id}] Collecting proposals from {len(child_node_ids)} children (output_capacity={output_capacity})")
        
        # ========== Step 1: Collect all proposals from children ==========
        collected_proposals = _collect_proposals_from_children(
            state=state,
            child_node_ids=child_node_ids,
        )
        
        logger.info(f"  [{node_id}] Collected {len(collected_proposals)} total proposals")
        
        # ========== Step 2: LLM One-Shot Synthesis (always execute, no pass-through) ==========
        logger.info(f"  [{node_id}] Synthesizing {len(collected_proposals)} proposals -> {output_capacity} output capacity")
        
        # Get current TBox for context
        current_tbox = state.current_tbox or state.initial_tbox
        
        # Get LLM
        llm = get_llm_by_purpose(
            purpose=state.llm_purpose,
            override_config=state.llm_override_config
        )
        
        # Use shared synthesis core
        iteration = len(state.synthesis_log) + 1
        target_classes = getattr(state, 'target_classes', 0)
        actions, notes = _synthesize_proposals_core(
            proposals=collected_proposals,
            current_tbox=current_tbox,
            output_capacity=output_capacity,
            llm=llm,
            node_id=node_id,
            release_log="",  # Intermediate synths don't have release log context
            iteration=iteration,
            target_classes=target_classes,
        )
        
        logger.info(f"  [{node_id}] {notes}")
        
        # ========== Step 4: Convert actions back to proposals ==========
        output_proposals = []
        for action in actions:
            proposal = _action_to_proposal(action, source=node_id)
            output_proposals.append(proposal.model_dump())
        
        logger.info(f"  [{node_id}] Output {len(output_proposals)} proposals")
        
        return {
            "intermediate_proposals": {node_id: output_proposals}
        }
    
    # Set function name for better debugging
    synthesizer_node.__name__ = f"synthesizer_{node_id}"
    synthesizer_node.__doc__ = f"Synthesizer node {node_id} aggregating from {child_node_ids} (capacity={output_capacity})"
    
    return synthesizer_node


def get_child_nodes_for_synthesizer(
    level: int,
    aggregator_index: int,
    branching_factor: int,
    total_children_at_prev_level: int,
    prev_level_prefix: str,
) -> List[str]:
    """
    Compute which child nodes feed into a specific synthesizer.
    
    Args:
        level: The synthesizer's level
        aggregator_index: Index of this synthesizer at its level
        branching_factor: Max children per synthesizer (B)
        total_children_at_prev_level: Total nodes at the previous level
        prev_level_prefix: Prefix for child node names
                          "local_propose_group_" for level 1
                          "synth_L{level-1}_" for higher levels
    
    Returns:
        List of child node IDs
    
    Example:
        get_child_nodes_for_synthesizer(1, 0, 5, 9, "local_propose_group_")
        → ["local_propose_group_0", "local_propose_group_1", ..., "local_propose_group_4"]
        
        get_child_nodes_for_synthesizer(1, 1, 5, 9, "local_propose_group_")
        → ["local_propose_group_5", "local_propose_group_6", ..., "local_propose_group_8"]
    """
    start_idx = aggregator_index * branching_factor
    end_idx = min(start_idx + branching_factor, total_children_at_prev_level)
    
    return [f"{prev_level_prefix}{i}" for i in range(start_idx, end_idx)]


def build_synthesizer_hierarchy(
    tree_config: Dict[str, Any],
    n_clusters: int,
    proposal_capacity: int = 30,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """
    Build the complete synthesizer hierarchy based on tree config.
    
    Each synthesizer's output_capacity = proposal_capacity // n_siblings_at_parent.
    Each local's max_proposals = proposal_capacity (locals generate freely, synths filter).
    
    Args:
        tree_config: Tree structure from compute_tree_structure()
        n_clusters: Actual number of leaf clusters
        proposal_capacity: Max proposals flowing to root
    
    Returns:
        Tuple of:
        - hierarchy: Dict mapping node_id to {level, index, children, node_type, output_capacity}
        - local_max_proposals: Dict mapping group_id to proposal_capacity
    """
    hierarchy = {}
    local_max_proposals = {}
    
    levels = tree_config.get("levels", [])
    B = tree_config.get("branching_factor", 6)
    depth = tree_config.get("depth", 1)
    
    # ========== First pass: Build hierarchy structure ==========
    # Track number of nodes at each level
    # Level 0 = leaves (n_clusters)
    nodes_per_level = {0: n_clusters}
    
    # Build from level 1 up to (depth - 1), excluding root
    for level_info in levels:
        level = level_info["level"]
        node_type = level_info["node_type"]
        n_agents = level_info["n_agents"]
        
        if node_type == "leaf" or node_type == "root":
            continue  # Skip leaf and root levels
        
        # This is a synthesizer level
        prev_level = level - 1
        prev_level_count = nodes_per_level.get(prev_level, n_clusters)
        
        # Determine prefix for children
        if prev_level == 0:
            prev_prefix = "local_propose_group_"
        else:
            prev_prefix = f"synth_L{prev_level}_"
        
        # Create synthesizer entries (output_capacity will be calculated in second pass)
        for agg_idx in range(n_agents):
            node_id = f"synth_L{level}_{agg_idx}"
            children = get_child_nodes_for_synthesizer(
                level=level,
                aggregator_index=agg_idx,
                branching_factor=B,
                total_children_at_prev_level=prev_level_count,
                prev_level_prefix=prev_prefix,
            )
            
            hierarchy[node_id] = {
                "level": level,
                "index": agg_idx,
                "children": children,
                "node_type": "synthesizer",
                "output_capacity": proposal_capacity,  # Will be refined in second pass
            }
        
        # Update nodes_per_level for this level
        nodes_per_level[level] = n_agents
    
    # ========== Second pass: Calculate output_capacity top-down ==========
    # Root feeds into proposal_capacity, so highest synth level gets capacity // n_synths
    highest_synth_level = depth - 1
    
    # Get count of synthesizers at highest level (root's direct children)
    n_root_children = nodes_per_level.get(highest_synth_level, n_clusters)
    capacity_per_root_child = max(1, proposal_capacity // n_root_children) if n_root_children > 0 else proposal_capacity
    
    # Assign output_capacity from top (highest synth level) to bottom (level 1)
    for level in range(highest_synth_level, 0, -1):
        # Get capacity for this level (already calculated or inherited)
        if level == highest_synth_level:
            level_capacity = capacity_per_root_child
        else:
            # Capacity comes from parent synthesizer
            # All synths at this level should have the same capacity
            # (their parents at level+1 divided by their sibling count)
            pass  # Handled below
        
        # Update all synths at this level
        for node_id, info in hierarchy.items():
            if info["level"] == level:
                if level == highest_synth_level:
                    info["output_capacity"] = level_capacity
                # else: already set by parent iteration
                
                # Calculate capacity for children (if they're synthesizers at level-1)
                if level > 1:
                    n_children = len(info["children"])
                    child_capacity = max(1, info["output_capacity"] // n_children) if n_children > 0 else 1
                    for child_id in info["children"]:
                        if child_id in hierarchy:
                            hierarchy[child_id]["output_capacity"] = child_capacity
    
    # ========== Third pass: Calculate local_max_proposals ==========
    # Each local gets proposal_capacity // n_siblings (siblings under same parent synth)
    for node_id, info in hierarchy.items():
        if info["level"] == 1:
            n_children = len(info["children"])
            max_per_local = max(1, proposal_capacity // n_children) if n_children > 0 else 1
            for child_id in info["children"]:
                group_id = child_id.replace("local_propose_", "")
                local_max_proposals[group_id] = max_per_local
    
    return hierarchy, local_max_proposals


def get_root_children(
    tree_config: Dict[str, Any],
    n_clusters: int,
) -> List[str]:
    """
    Get the child node IDs for the root (global_synthesis) node.
    
    If depth > 1: children are the highest-level synthesizers
    If depth == 1: children are the leaf nodes directly
    
    Args:
        tree_config: Tree structure config
        n_clusters: Number of leaf clusters
    
    Returns:
        List of child node IDs for the root
    """
    depth = tree_config.get("depth", 1)
    levels = tree_config.get("levels", [])
    B = tree_config.get("branching_factor", 6)
    
    if depth <= 1:
        # Direct leaf -> root
        return [f"local_propose_group_{i}" for i in range(n_clusters)]
    
    # Find the highest synthesizer level (one below root)
    highest_synth_level = depth - 1
    
    # Count synthesizers at that level
    for level_info in levels:
        if level_info["level"] == highest_synth_level:
            n_synth = level_info["n_agents"]
            return [f"synth_L{highest_synth_level}_{i}" for i in range(n_synth)]
    
    # Fallback: direct leaf connection
    return [f"local_propose_group_{i}" for i in range(n_clusters)]


__all__ = [
    "create_synthesizer_node",
    "get_child_nodes_for_synthesizer",
    "build_synthesizer_hierarchy",
    "get_root_children",
]
