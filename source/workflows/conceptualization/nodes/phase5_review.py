"""
Phase 5: Global Review

Reviews the TBox based on voting results and suggests conservative improvements.
Primarily focuses on merging similar classes rather than deleting them.
"""

from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

from workflows.common.node_decorators import graph_node
from workflows.conceptualization.state import FederatedPrimitiveTBoxState
from workflows.conceptualization.schemas.review import (
    ReviewAction,
    ClassVotingSummary,
    GlobalReviewOutput,
)
from workflows.conceptualization.prompts.templates import GLOBAL_REVIEW_PROMPT

from .prompt_log_utils import log_prompt_once, build_target_classes_hint
from workflows.conceptualization.utils.validation import (
    detect_class_cycle,
    would_create_cycle,
    log_cycle_detection_result,
)


# ============== Main Review Node ==============

@graph_node(node_type="generation", log_level="INFO")
def global_review_node(state: FederatedPrimitiveTBoxState) -> Dict[str, Any]:
    """
    Review the current TBox based on voting results.
    
    Philosophy: Conservative optimization - prefer merging over deletion.
    
    Returns:
        Updated current_tbox after applying review actions
    """
    from llm.manager import get_llm_by_purpose
    from llm.invoke_with_stats import invoke_structured_llm
    from utils.tree_formatter import format_class_hierarchy
    
    logger.info("Phase 5: Global Review")
    
    # Get current TBox and votes
    current_tbox = state.current_tbox
    local_votes = state.local_votes or {}
    
    if not current_tbox:
        logger.warning("  No TBox to review")
        return {}
    
    classes = current_tbox.get("classes", [])
    if not classes:
        logger.warning("  No classes to review")
        return {}
    
    # Compute voting summary
    voting_summary = _compute_voting_summary(classes, local_votes)
    logger.info(f"  Reviewing {len(classes)} classes with votes from {len(local_votes)} agents")
    
    # Create a map from class_name to voting summary
    voting_map = {s.class_name: s for s in voting_summary}
    
    # Format tree with voting info embedded
    tbox_tree_with_votes = _format_tbox_with_voting(classes, voting_map)
    
    # Build target classes hint
    target_classes = getattr(state, 'target_classes', 0)
    target_classes_hint = build_target_classes_hint(target_classes, len(classes))
    
    # Build prompt
    prompt = GLOBAL_REVIEW_PROMPT.format(
        target_classes_hint=target_classes_hint,
        current_tbox_tree=tbox_tree_with_votes,
    )
    
    # Per-iteration prompt logging
    iteration = _get_iteration(state)
    log_prompt_once("global_review", iteration, prompt, "Global Review")
    
    # Get LLM and invoke
    llm = get_llm_by_purpose(
        purpose=state.llm_purpose,
        override_config=state.llm_override_config
    )
    
    try:
        review_llm = llm.with_structured_output(GlobalReviewOutput)
        review_result: GlobalReviewOutput = invoke_structured_llm(review_llm, prompt)
        
        actions = review_result.actions
        summary = review_result.review_summary
        
        logger.info(f"  Review summary: {summary}")
        logger.info(f"  Generated {len(actions)} review actions")
        
        for action in actions:
            if action.operation == "add":
                logger.info(f"    -> ADD {action.class_name} (parent: {action.parent_class})")
            elif action.operation == "modify":
                if action.new_class_name:
                    logger.info(f"    -> MODIFY {action.class_name} → {action.new_class_name}")
                else:
                    logger.info(f"    -> MODIFY {action.class_name}")
            elif action.operation == "delete":
                logger.info(f"    -> DELETE {action.class_name}")
            elif action.operation == "merge":
                logger.info(f"    -> MERGE {action.source_classes} into {action.class_name}")
        
        # Store review output for insights synthesizer to consume
        # Also include complete voting results for experiment analysis
        review_output = {
            "iteration": _get_iteration(state),
            "actions": [a.model_dump() for a in actions],
            "summary": summary,
            "n_actions": len(actions),
            # Complete voting data for ablation experiments 
            "voting_summary": [s.model_dump() for s in voting_summary],
            "raw_votes": local_votes,  # {agent_id: {class_name: 0/1}}
            "n_classes": len(classes),
            "n_voting_agents": len(local_votes),
        }
        
        # Apply review actions to TBox
        if actions:
            # Note: _apply_review_actions now performs incremental cycle detection
            # and rejects individual actions that would create cycles
            updated_tbox, rejected_actions = _apply_review_actions(current_tbox, actions)
            
            # Add rejected actions to review output if any
            if rejected_actions:
                review_output["rejected_actions_due_to_cycle"] = rejected_actions
            
            # ========== Final Cycle Detection (Safety Check) ==========
            # This is a safety check - if incremental detection worked correctly,
            # there should be no cycles here
            cycle_path = detect_class_cycle(updated_tbox.get("classes", []))
            
            if cycle_path:
                # This shouldn't happen if incremental detection worked, but handle it
                logger.error(f"  [Phase5] UNEXPECTED cycle detected after incremental filtering: {' -> '.join(cycle_path)}")
                logger.warning(f"  [Phase5] Rejecting reviewed TBox, reverting to current_tbox")
                review_output["rejected_reason"] = f"cycle_detected: {' -> '.join(cycle_path)}"
                return {
                    "current_tbox": current_tbox,  # Keep original TBox without cycle
                    "review_output": review_output,
                    "review_log": state.review_log + [review_output],
                    "phase_errors": {"phase_5_review": [f"Cycle detected in class hierarchy: {' -> '.join(cycle_path)}"]},
                }
            
            new_class_count = len(updated_tbox.get("classes", []))
            logger.info(f"  Updated TBox: {len(classes)} → {new_class_count} classes")
            return {
                "current_tbox": updated_tbox,
                "review_output": review_output,
                "review_log": state.review_log + [review_output],
            }
        else:
            logger.info("  No changes needed")
            return {
                "review_output": review_output,
                "review_log": state.review_log + [review_output],
            }
            
    except Exception as e:
        logger.error(f"  Review failed: {e}")
        return {"phase_errors": {"phase_5_review": [str(e)]}}


# ============== Helper Functions ==============

def _get_iteration(state: FederatedPrimitiveTBoxState) -> int:
    """Get current iteration number from synthesis log.
    
    Phase 3 (synthesis) appends to synthesis_log before Phase 5 (review) runs,
    so the current iteration = len(synthesis_log) (not +1).
    """
    synthesis_log = state.synthesis_log or []
    return len(synthesis_log)


# ============== Voting Summary Computation ==============

def _compute_voting_summary(
    classes: List[Dict],
    local_votes: Dict[str, Dict[str, int]]
) -> List[ClassVotingSummary]:
    """Compute voting summary for each class."""
    summaries = []
    n_agents = len(local_votes)
    
    for cls in classes:
        class_name = cls.get("name")
        if not class_name:
            continue
        
        # Count positive votes
        positive_votes = 0
        voting_agents = []
        
        for group_id, votes in local_votes.items():
            if votes.get(class_name, 0) == 1:
                positive_votes += 1
                voting_agents.append(group_id)
        
        coverage_ratio = positive_votes / n_agents if n_agents > 0 else 0.0
        
        summaries.append(ClassVotingSummary(
            class_name=class_name,
            total_votes=n_agents,
            positive_votes=positive_votes,
            coverage_ratio=coverage_ratio,
            voting_agents=voting_agents,
        ))
    
    return summaries


# ============== Formatting Helpers ==============

def _format_tbox_with_voting(classes: List[Dict], voting_map: Dict[str, ClassVotingSummary]) -> str:
    """Format TBox as tree hierarchy with voting info embedded.
    
    Output format:
        **Column**
        ├── **TemporalColumn**: Time-related columns [3/5 votes, 60%]
        │   ├── **YearColumn**: Represents year values [4/5 votes, 80%]
        │   └── **DateColumn**: Represents date values [2/5 votes, 40%]
        └── **NameColumn**: Names [5/5 votes, 100%]
    """
    from utils.tree_formatter import format_hierarchy_tree
    
    if not classes:
        return "No classes."
    
    # Normalize classes and add voting info to description
    enriched_classes = []
    for cls in classes:
        cls_copy = dict(cls)
        
        # Normalize parent_classes
        if "parent_classes" not in cls_copy:
            parent = cls_copy.get("parent_class", cls_copy.get("parent", "Column"))
            cls_copy["parent_classes"] = [parent] if parent else ["Column"]
        
        # Get original description (keep full, no truncation)
        original_desc = cls_copy.get("description", "")
        
        # Get voting info
        class_name = cls_copy.get("name", "")
        voting_info = voting_map.get(class_name)
        
        if voting_info:
            coverage_pct = int(voting_info.coverage_ratio * 100)
            vote_str = f"[{voting_info.positive_votes}/{voting_info.total_votes} votes, {coverage_pct}%]"
        else:
            vote_str = "[no votes]"
        
        # Combine: description + voting
        if original_desc:
            cls_copy["description"] = f"{original_desc} {vote_str}"
        else:
            cls_copy["description"] = vote_str
        
        enriched_classes.append(cls_copy)
    
    return format_hierarchy_tree(
        enriched_classes,
        name_key="name",
        parent_key="parent_classes",
        description_key="description",
        root_name="Column",
        max_desc_length=None,  # Don't truncate, we already handled it
    )


# ============== Apply Review Actions ==============

def _apply_review_actions(
    tbox: Dict[str, Any],
    actions: List[ReviewAction]
) -> Tuple[Dict[str, Any], List[str]]:
    """Apply review actions to TBox.
    
    Handles:
    - add: Create new class
    - modify: Update existing class (can rename)
    - delete: Remove class (reassign children)
    - merge: Combine classes
    
    Returns:
        Tuple of (updated_tbox, rejected_actions)
        - updated_tbox: The TBox after applying valid actions
        - rejected_actions: List of action descriptions that were rejected due to cycles
    """
    classes = list(tbox.get("classes", []))
    class_by_name = {c.get("name"): c for c in classes}
    rejected_actions = []
    
    for action in actions:
        if action.operation == "add":
            result = _apply_add_with_cycle_check(classes, class_by_name, action)
            if result is None:
                rejected_actions.append(f"ADD {action.class_name} (parent={action.parent_class})")
            else:
                classes, class_by_name = result
        elif action.operation == "modify":
            result = _apply_modify_with_cycle_check(classes, class_by_name, action)
            if result is None:
                rejected_actions.append(f"MODIFY {action.class_name} (parent={action.parent_class})")
            else:
                classes, class_by_name = result
        elif action.operation == "delete":
            classes, class_by_name = _apply_delete(classes, class_by_name, action)
        elif action.operation == "merge":
            result = _apply_merge_with_cycle_check(classes, class_by_name, action)
            if result is None:
                rejected_actions.append(f"MERGE -> {action.class_name} (parent={action.parent_class})")
            else:
                classes, class_by_name = result
    
    # Log rejected actions if any
    if rejected_actions:
        logger.warning(f"  Rejected {len(rejected_actions)} review actions due to cycle detection:")
        for ra in rejected_actions:
            logger.warning(f"    - {ra}")
    
    return {
        **tbox,
        "classes": classes,
    }, rejected_actions


def _apply_add_with_cycle_check(
    classes: List[Dict],
    class_by_name: Dict[str, Dict],
    action: ReviewAction
) -> Optional[tuple]:
    """Apply add operation with cycle check. Returns None if would create cycle."""
    class_name = action.class_name
    proposed_parent = action.parent_class or "Column"
    
    if class_name in class_by_name:
        logger.warning(f"  Add target already exists: {class_name}")
        return classes, class_by_name
    
    # Check for cycle before adding
    if would_create_cycle(classes, class_name, proposed_parent):
        logger.warning(f"  ADD {class_name} (parent={proposed_parent}) rejected: would create cycle")
        return None
    
    return _apply_add(classes, class_by_name, action)


def _apply_modify_with_cycle_check(
    classes: List[Dict],
    class_by_name: Dict[str, Dict],
    action: ReviewAction
) -> Optional[tuple]:
    """Apply modify operation with cycle check. Returns None if would create cycle."""
    class_name = action.class_name
    
    if class_name not in class_by_name:
        logger.warning(f"  Modify target not found: {class_name}")
        return classes, class_by_name
    
    # Check 1: Renaming to an existing class name would cause conflicts
    if action.new_class_name and action.new_class_name != class_name:
        if action.new_class_name in class_by_name:
            # Renaming A -> B when B already exists
            # This could cause self-reference cycles when updating parent references
            logger.warning(f"  MODIFY {class_name} → {action.new_class_name} rejected: target name already exists")
            return None
    
    # Check 2: Cycle check if parent_class is being changed
    if action.parent_class:
        target_class_name = action.new_class_name if action.new_class_name else class_name
        if would_create_cycle(classes, target_class_name, action.parent_class):
            logger.warning(f"  MODIFY {class_name} (parent={action.parent_class}) rejected: would create cycle")
            return None
    
    return _apply_modify(classes, class_by_name, action)


def _apply_merge_with_cycle_check(
    classes: List[Dict],
    class_by_name: Dict[str, Dict],
    action: ReviewAction
) -> Optional[tuple]:
    """Apply merge operation with cycle check. Returns None if would create cycle."""
    target_name = action.class_name
    proposed_parent = action.parent_class or "Column"
    
    # Check if setting the parent would create a cycle
    if proposed_parent and proposed_parent != "Column":
        if would_create_cycle(classes, target_name, proposed_parent):
            logger.warning(f"  MERGE -> {target_name} (parent={proposed_parent}) rejected: would create cycle")
            return None
    
    return _apply_merge(classes, class_by_name, action)


def _apply_add(
    classes: List[Dict],
    class_by_name: Dict[str, Dict],
    action: ReviewAction
) -> tuple:
    """Apply add operation: create new class."""
    class_name = action.class_name
    
    if class_name in class_by_name:
        logger.warning(f"  Add target already exists: {class_name}")
        return classes, class_by_name
    
    new_class = {
        "name": class_name,
        "description": action.description or "",
        "parent_class": action.parent_class or "Column",
    }
    classes.append(new_class)
    class_by_name[class_name] = new_class
    
    logger.debug(f"  Applied add: {class_name}")
    return classes, class_by_name


def _apply_modify(
    classes: List[Dict],
    class_by_name: Dict[str, Dict],
    action: ReviewAction
) -> tuple:
    """Apply modify operation: update class (can rename)."""
    class_name = action.class_name
    
    if class_name not in class_by_name:
        logger.warning(f"  Modify target not found: {class_name}")
        return classes, class_by_name
    
    target_class = class_by_name[class_name]
    
    # Handle rename
    if action.new_class_name and action.new_class_name != class_name:
        new_name = action.new_class_name
        old_name = class_name
        
        # Update class name
        target_class["name"] = new_name
        
        # Update children's parent references
        for cls in classes:
            parent = cls.get("parent_class") or cls.get("parent")
            if parent == old_name:
                cls["parent_class"] = new_name
                if "parent" in cls:
                    cls["parent"] = new_name
        
        # Update lookup
        del class_by_name[old_name]
        class_by_name[new_name] = target_class
        
        logger.debug(f"  Applied rename: {old_name} → {new_name}")
    
    # Update other fields
    if action.description:
        target_class["description"] = action.description
    if action.parent_class:
        target_class["parent_class"] = action.parent_class
        if "parent" in target_class:
            target_class["parent"] = action.parent_class
    
    return classes, class_by_name


def _apply_delete(
    classes: List[Dict],
    class_by_name: Dict[str, Dict],
    action: ReviewAction
) -> tuple:
    """Apply delete operation: remove class and reassign children."""
    class_name = action.class_name
    
    if class_name not in class_by_name:
        logger.warning(f"  Delete target not found: {class_name}")
        return classes, class_by_name
    
    target_class = class_by_name[class_name]
    target_parent = target_class.get("parent_class") or target_class.get("parent", "Column")
    
    # Reassign children to target's parent
    for cls in classes:
        parent = cls.get("parent_class") or cls.get("parent")
        if parent == class_name:
            cls["parent_class"] = target_parent
            if "parent" in cls:
                cls["parent"] = target_parent
    
    # Remove target class
    classes = [c for c in classes if c.get("name") != class_name]
    del class_by_name[class_name]
    
    logger.debug(f"  Applied delete: {class_name} (children → {target_parent})")
    return classes, class_by_name


def _apply_merge(
    classes: List[Dict],
    class_by_name: Dict[str, Dict],
    action: ReviewAction
) -> tuple:
    """Apply merge operation: combine source_classes into class_name."""
    target_name = action.class_name
    source_names = action.source_classes or []
    
    if not source_names:
        logger.warning(f"  Merge action has no source_classes: {target_name}")
        return classes, class_by_name
    
    # Find or create target class
    target_class = class_by_name.get(target_name)
    
    if target_class:
        # Update existing class
        if action.description:
            target_class["description"] = action.description
        if action.parent_class:
            target_class["parent_class"] = action.parent_class
    else:
        # Target is one of the sources - use first source as base
        base_source = source_names[0] if source_names else None
        if base_source and base_source in class_by_name:
            target_class = dict(class_by_name[base_source])
            target_class["name"] = target_name
            if action.description:
                target_class["description"] = action.description
            if action.parent_class:
                target_class["parent_class"] = action.parent_class
            classes.append(target_class)
            class_by_name[target_name] = target_class
    
    # Remove source classes
    removed_names = set()
    for source_name in source_names:
        if source_name != target_name and source_name in class_by_name:
            removed_names.add(source_name)
    
    # Update children of removed classes to point to target
    for cls in classes:
        parent = cls.get("parent_class") or cls.get("parent")
        if parent in removed_names:
            cls["parent_class"] = target_name
            if "parent" in cls:
                cls["parent"] = target_name
    
    # Filter out removed classes
    classes = [c for c in classes if c.get("name") not in removed_names]
    for name in removed_names:
        if name in class_by_name:
            del class_by_name[name]
    
    logger.debug(f"  Applied merge: {source_names} → {target_name}")
    return classes, class_by_name
