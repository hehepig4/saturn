"""
Phase 2: Local Class Proposals

Each Local Agent proposes class changes based on its cluster's CQs and data.
DataProperty generation is handled separately in phase_dp_final.py.

Uses factory pattern to create parallel nodes for each group.
"""

import uuid
import threading
from typing import Dict, Any, List, Callable
from loguru import logger
from pydantic import BaseModel, Field

from workflows.common.node_decorators import graph_node
from workflows.conceptualization.state import FederatedPrimitiveTBoxState
from workflows.conceptualization.schemas.proposals import ClassProposal
from workflows.conceptualization.prompts.templates import LOCAL_CLASS_PROPOSAL_PROMPT
from workflows.conceptualization.nodes.prompt_log_utils import log_prompt_once, build_target_classes_hint


# ============== LLM Output Schema ==============

class LocalClassProposalsOutput(BaseModel):
    """LLM output for local class proposals."""
    proposals: List[ClassProposal] = Field(
        default_factory=list,
        description="List of class proposals from this local agent"
    )


# ============== Node Factory ==============

def create_local_proposal_node(group_id: str) -> Callable:
    """
    Factory function to create a local proposal node for a specific group.
    
    Pattern from branch_nodes.py:
        - Each node has unique __name__ for LangGraph
        - Returns dict with group_id key for merge_dict_reducer
    
    Args:
        group_id: The cluster group ID (e.g., "group_0")
        
    Returns:
        A node function that generates class proposals for this group
    """
    
    @graph_node(node_type="generation", log_level="INFO")
    def local_propose(state: FederatedPrimitiveTBoxState) -> Dict[str, Any]:
        """Generate class proposals for group {group_id}."""
        logger.info(f"Phase 2: Local Class Proposals for {group_id}")
        
        from llm.manager import get_llm_by_purpose
        from llm.invoke_with_stats import invoke_structured_llm
        
        # Get proposal capacity for this agent (dynamically computed)
        max_proposals = state.get_max_proposals_for_group(group_id)
        logger.info(f"  {group_id}: Max proposals allowed: {max_proposals}")
        
        # Get this group's CQs from branch_cqs (populated in Phase 0b)
        group_cqs = state.branch_cqs.get(group_id, [])
        
        if not group_cqs:
            logger.warning(f"  {group_id}: No CQs assigned, skipping")
            return {"local_proposals": {group_id: []}}
        
        logger.info(f"  {group_id}: {len(group_cqs)} CQs")
        
        # Get current TBox
        current_tbox = state.current_tbox or state.initial_tbox
        if not current_tbox:
            logger.error(f"  {group_id}: No TBox available")
            return {
                "local_proposals": {group_id: []},
                "phase_errors": {f"phase_2_{group_id}": ["No TBox available"]},
            }
        
        # DEBUG: Log current TBox size
        n_classes = len(current_tbox.get("classes", []))
        logger.info(f"  {group_id}: Current TBox has {n_classes} classes")
        
        # Format current TBox for prompt
        current_classes = _format_current_classes(current_tbox)
        cqs_block = _format_cqs_block(group_cqs)
        
        # Format release log from global_insights (if available)
        release_log = _format_release_log(state.global_insights)
        
        # Get LLM
        llm = get_llm_by_purpose(
            purpose=state.llm_purpose,
            override_config=state.llm_override_config
        )
        
        # ========== Generate Class Proposals ==========
        logger.info(f"  {group_id}: Generating class proposals (max {max_proposals})")
        
        # Build target classes hint with current count
        target_classes = getattr(state, 'target_classes', 0)
        current_n_classes = len(current_tbox.get("classes", [])) if current_tbox else 0
        target_classes_hint = build_target_classes_hint(target_classes, current_n_classes)
        
        class_prompt = LOCAL_CLASS_PROPOSAL_PROMPT.format(
            group_id=group_id,
            max_proposals=max_proposals,
            target_classes_hint=target_classes_hint,
            current_tbox_classes=current_classes,
            cqs_block=cqs_block,
            release_log=release_log,
        )
        
        # Per-iteration prompt logging
        iteration = _get_iteration(state)
        log_prompt_once("local_proposal", iteration, class_prompt, "Local Class Proposal")
        
        try:
            # Create output schema for LLM
            class_llm = llm.with_structured_output(LocalClassProposalsOutput)
            class_result: LocalClassProposalsOutput = invoke_structured_llm(
                class_llm, class_prompt
            )
            
            # Assign proper proposal IDs and source
            class_proposals = []
            for prop in class_result.proposals:
                prop.source = group_id
                prop.proposal_id = f"{group_id}_{uuid.uuid4().hex[:8]}"
                class_proposals.append(prop)
            
            # Enforce capacity limit
            if len(class_proposals) > max_proposals:
                logger.warning(f"    {group_id}: Truncating {len(class_proposals)} -> {max_proposals} proposals")
                class_proposals = class_proposals[:max_proposals]
            
            logger.info(f"    {group_id}: Generated {len(class_proposals)} class proposals")
            
            # Detailed logging for debugging
            for i, prop in enumerate(class_proposals):
                logger.info(f"      [{i+1}] {prop.operation.upper()}: {prop.class_name} (parent={prop.parent_class})")
            
        except Exception as e:
            logger.error(f"  {group_id}: Class proposal failed: {e}")
            class_proposals = []
        
        if not class_proposals:
            logger.info(f"  {group_id}: No class proposals, returning empty")
            return {"local_proposals": {group_id: []}}
        
        # Convert to dict for state storage (just class proposals, no DP)
        proposal_dicts = [p.model_dump() for p in class_proposals]
        
        return {"local_proposals": {group_id: proposal_dicts}}
    
    # Set unique node name for LangGraph
    local_propose.__name__ = f"local_propose_{group_id}"
    
    return local_propose


# ============== Helper Functions ==============

def _format_current_classes(tbox: Dict[str, Any]) -> str:
    """Format current TBox classes as tree hierarchy for prompt."""
    from utils.tree_formatter import format_class_hierarchy
    
    classes = tbox.get("classes", [])
    if not classes:
        return "No classes defined yet."
    
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
    """Format CQs for prompt - no truncation."""
    lines = []
    for i, cq in enumerate(cqs):
        question = cq.get("question", cq.get("text", ""))
        lines.append(f"{i+1}. {question}")
    return "\n".join(lines) if lines else "No CQs available."


def _format_release_log(global_insights) -> str:
    """Format global insights for the Previous Learnings section.
    
    Returns content only (no header) - the header is in the prompt template.
    """
    if global_insights is None:
        return "First iteration - no previous learnings."
    
    # Use GlobalInsights.format_release_log() if available
    if hasattr(global_insights, 'format_release_log'):
        return global_insights.format_release_log()
    
    # Fallback for dict-like structure
    if isinstance(global_insights, dict):
        changelog = global_insights.get('changelog', '')
        patterns = global_insights.get('patterns', [])
        iteration = global_insights.get('iteration', 0)
        
        if iteration == 0 or (not changelog and not patterns):
            return "First iteration - no previous learnings."
        
        lines = []
        lines.append(f"**After Iteration {iteration}:**")
        lines.append(changelog if changelog else "No changes recorded.")
        lines.append("")
        
        if patterns:
            lines.append("**Patterns:**")
            for p in patterns:
                lines.append(f"- {p}")
        
        return "\n".join(lines)
    
    return "First iteration - no previous learnings."


def _get_iteration(state: FederatedPrimitiveTBoxState) -> int:
    """Get current iteration number from synthesis log."""
    synthesis_log = state.synthesis_log or []
    return len(synthesis_log) + 1


# ============== Factory for All Groups ==============

def create_local_proposal_nodes(group_ids: List[str]) -> Dict[str, Callable]:
    """
    Create local proposal nodes for all groups.
    
    Args:
        group_ids: List of group IDs from cluster_assignments
        
    Returns:
        Dict mapping node names to node functions
    """
    nodes = {}
    for group_id in group_ids:
        node = create_local_proposal_node(group_id)
        nodes[node.__name__] = node
    return nodes
