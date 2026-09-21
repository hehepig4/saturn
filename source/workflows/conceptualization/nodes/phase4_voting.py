"""
Phase 4: Local Voting

Each Local Agent votes on class usefulness for their cluster.
Uses factory pattern like Phase 2 for parallel voting.
"""

import threading
from typing import Dict, Any, List, Callable
from loguru import logger

from workflows.common.node_decorators import graph_node
from workflows.conceptualization.state import FederatedPrimitiveTBoxState
from workflows.conceptualization.schemas.voting import (
    create_voting_model,
    parse_voting_result,
    aggregate_votes,
)
from workflows.conceptualization.prompts.templates import LOCAL_VOTING_PROMPT
from .prompt_log_utils import log_prompt_once


# ============== Node Factory ==============

def create_local_voting_node(group_id: str) -> Callable:
    """
    Factory function to create a local voting node for a specific group.
    
    Pattern: Same as create_local_proposal_node in phase2_local.py
    
    Args:
        group_id: The cluster group ID (e.g., "group_0")
        
    Returns:
        A node function that generates votes for this group
    """
    
    @graph_node(node_type="generation", log_level="INFO")
    def local_vote(state: FederatedPrimitiveTBoxState) -> Dict[str, Any]:
        """Vote on class usefulness for group {group_id}."""
        logger.info(f"Phase 4: Local Voting for {group_id}")
        
        from llm.manager import get_llm_by_purpose
        from llm.invoke_with_stats import invoke_structured_llm
        
        # Get this group's CQs from branch_cqs (populated in Phase 0b)
        group_cqs = state.branch_cqs.get(group_id, [])
        
        if not group_cqs:
            logger.warning(f"  {group_id}: No CQs assigned, voting 0 for all")
            return {"local_votes": {group_id: {}}}
        
        # Get final TBox
        final_tbox = state.current_tbox
        if not final_tbox:
            logger.error(f"  {group_id}: No TBox available for voting")
            return {
                "local_votes": {group_id: {}},
                "phase_errors": {f"phase_4_{group_id}": ["No TBox available"]},
            }
        
        # Get all class names
        classes = final_tbox.get("classes", [])
        class_names = [c.get("name") for c in classes if c.get("name")]
        
        if not class_names:
            logger.warning(f"  {group_id}: No classes to vote on")
            return {"local_votes": {group_id: {}}}
        
        logger.info(f"  {group_id}: Voting on {len(class_names)} classes")
        
        # Format prompt inputs
        classes_summary = _format_classes_for_voting(classes)
        cqs_block = _format_cqs_block(group_cqs)
        
        # Create dynamic voting model
        VotingModel = create_voting_model(class_names)
        
        # Get LLM
        llm = get_llm_by_purpose(
            purpose=state.llm_purpose,
            override_config=state.llm_override_config
        )
        
        prompt = LOCAL_VOTING_PROMPT.format(
            group_id=group_id,
            all_classes_with_descriptions=classes_summary,
            cqs_block=cqs_block,
        )
        
        # Per-iteration prompt logging
        iteration = _get_iteration(state)
        log_prompt_once("local_voting", iteration, prompt, "Local Voting")
        
        try:
            voting_llm = llm.with_structured_output(VotingModel)
            vote_result = invoke_structured_llm(voting_llm, prompt)
            
            # Parse to dict
            votes = parse_voting_result(vote_result, class_names)
            
            useful_count = sum(1 for v in votes.values() if v == 1)
            logger.info(f"    {group_id}: {useful_count}/{len(class_names)} classes marked useful")
            
        except Exception as e:
            logger.error(f"  {group_id}: Voting failed: {e}")
            # Fallback: all zeros
            votes = {name: 0 for name in class_names}
        
        return {"local_votes": {group_id: votes}}
    
    # Set unique node name for LangGraph
    local_vote.__name__ = f"local_vote_{group_id}"
    
    return local_vote


# ============== Aggregation Node ==============

@graph_node(node_type="aggregation", log_level="INFO")
def aggregate_votes_node(state: FederatedPrimitiveTBoxState) -> Dict[str, Any]:
    """
    Aggregate votes from all Local Agents.
    
    Computes average score per class across all groups.
    Stores as TF-IDF style weight for downstream use.
    """
    logger.info("Phase 4: Aggregating votes")
    
    local_votes = state.local_votes or {}
    
    if not local_votes:
        logger.warning("  No votes to aggregate")
        return {"aggregated_votes": {}}
    
    # Get all class names from final TBox
    final_tbox = state.current_tbox
    classes = final_tbox.get("classes", []) if final_tbox else []
    class_names = [c.get("name") for c in classes if c.get("name")]
    
    if not class_names:
        logger.warning("  No classes to aggregate votes for")
        return {"aggregated_votes": {}}
    
    # Aggregate
    aggregated = aggregate_votes(local_votes, class_names)
    
    # Log summary
    high_score = [n for n, s in aggregated.items() if s >= 0.5]
    low_score = [n for n, s in aggregated.items() if s < 0.5]
    
    logger.info(f"  High score (>=0.5): {len(high_score)} classes")
    logger.info(f"  Low score (<0.5): {len(low_score)} classes")
    
    # Attach scores to TBox classes
    updated_tbox = _attach_scores_to_tbox(final_tbox, aggregated)
    
    return {
        "aggregated_votes": aggregated,
        "current_tbox": updated_tbox,
    }


# ============== Helper Functions ==============

def _format_classes_for_voting(classes: List[Dict]) -> str:
    """Format classes as tree hierarchy for voting prompt."""
    from utils.tree_formatter import format_class_hierarchy
    
    if not classes:
        return "No classes to vote on."
    
    # format_class_hierarchy expects parent_classes (list), convert if needed
    normalized = []
    for cls in classes:
        cls_copy = dict(cls)
        # Ensure parent_classes is a list
        if "parent_classes" not in cls_copy:
            parent = cls_copy.get("parent_class", cls_copy.get("parent", "Column"))
            cls_copy["parent_classes"] = [parent] if parent else ["Column"]
        normalized.append(cls_copy)
    
    return format_class_hierarchy(normalized, include_column_root=True, max_desc_length=None)


def _format_cqs_block(cqs: List[Dict]) -> str:
    """Format CQs for voting prompt - no truncation."""
    lines = []
    for i, cq in enumerate(cqs):
        question = cq.get("question", cq.get("text", ""))
        lines.append(f"{i+1}. {question}")
    return "\n".join(lines) if lines else "No CQs available."


def _attach_scores_to_tbox(
    tbox: Dict[str, Any],
    scores: Dict[str, float]
) -> Dict[str, Any]:
    """Attach aggregated scores to TBox classes."""
    if not tbox:
        return tbox
    
    classes = tbox.get("classes", [])
    updated_classes = []
    
    for cls in classes:
        cls_copy = dict(cls)
        name = cls.get("name")
        if name and name in scores:
            cls_copy["usefulness_score"] = scores[name]
        updated_classes.append(cls_copy)
    
    return {
        **tbox,
        "classes": updated_classes,
    }


def _get_iteration(state: FederatedPrimitiveTBoxState) -> int:
    """Get current iteration number from synthesis log.
    
    Phase 3 (synthesis) appends to synthesis_log before Phase 4 (voting) runs,
    so the current iteration = len(synthesis_log) (not +1).
    """
    synthesis_log = state.synthesis_log or []
    return len(synthesis_log)


# ============== Factory for All Groups ==============

def create_local_voting_nodes(
    group_ids: List[str]
) -> Dict[str, Callable]:
    """
    Create local voting nodes for all groups.
    
    Args:
        group_ids: List of group IDs from cluster_assignments
        
    Returns:
        Dict mapping node names to node functions
    """
    nodes = {}
    for group_id in group_ids:
        node = create_local_voting_node(group_id)
        nodes[node.__name__] = node
    return nodes
