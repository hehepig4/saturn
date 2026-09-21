"""
Phase 3: Global Class Synthesis (One-Shot Mode)

Global Agent synthesizes all local class proposals into final actions in one LLM call.
No batching by parent_class - all proposals processed together for conflict resolution.

DataProperty generation is handled separately in phase_dp_final.py.

This module provides:
- _synthesize_proposals_core(): Shared synthesis logic for both intermediate synth and root
- _action_to_proposal(): Convert SynthesizedAction back to ClassProposal for intermediate layers
- global_synthesis_node(): Root node that applies synthesis to TBox
"""

import uuid
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
from pydantic import BaseModel, Field

from workflows.common.node_decorators import graph_node
from workflows.conceptualization.state import FederatedPrimitiveTBoxState
from workflows.conceptualization.schemas.proposals import ClassProposal
from workflows.conceptualization.schemas.synthesis import (
    SynthesizedAction,
    GlobalSynthesisOutput,
)
from workflows.conceptualization.prompts.templates import GLOBAL_ONE_SHOT_SYNTHESIS_PROMPT
from .prompt_log_utils import log_prompt_once, build_target_classes_hint
from workflows.conceptualization.utils.validation import (
    detect_class_cycle,
    would_create_cycle,
    log_cycle_detection_result,
)


# ============== Shared Synthesis Core (used by both intermediate synth and root) ==============

def _synthesize_proposals_core(
    proposals: List[ClassProposal],
    current_tbox: Dict[str, Any],
    output_capacity: int,
    llm,
    node_id: str = "synthesis",
    release_log: str = "",
    iteration: int = 1,
    target_classes: int = 0,
) -> Tuple[List[SynthesizedAction], str]:
    """
    Core synthesis logic shared by intermediate synthesizers and root.
    
    Uses LLM one-shot synthesis to merge/dedupe proposals and resolve conflicts.
    
    Args:
        proposals: List of ClassProposal to synthesize
        current_tbox: Current TBox for context
        output_capacity: Max number of actions to output
        llm: LLM instance to use
        node_id: Identifier for logging (e.g., "synth_L1_0" or "global_synthesis")
        release_log: Previous learnings text (for root, empty for intermediate)
        iteration: Current iteration number for prompt logging
        target_classes: Target number of classes (0 = no specific target)
        
    Returns:
        Tuple of (List[SynthesizedAction], merge_notes)
    """
    from llm.invoke_with_stats import invoke_structured_llm
    
    if not proposals:
        return [], "No proposals to synthesize"
    
    # Format proposals and TBox
    all_proposals_text = _format_all_proposals(proposals)
    current_classes = _format_current_classes(current_tbox)
    
    # Build target classes hint
    current_n_classes = len(current_tbox.get("classes", [])) if current_tbox else 0
    target_classes_hint = build_target_classes_hint(target_classes, current_n_classes)
    
    # Build synthesis prompt (reuse global synthesis prompt)
    synthesis_prompt = GLOBAL_ONE_SHOT_SYNTHESIS_PROMPT.format(
        proposal_capacity=output_capacity,
        target_classes_hint=target_classes_hint,
        current_tbox_classes=current_classes,
        n_proposals=len(proposals),
        all_proposals=all_proposals_text,
        release_log=release_log or "First iteration - no previous learnings.",
    )
    
    # Log prompt for debugging
    log_prompt_once(f"synthesis_{node_id}", iteration, synthesis_prompt, f"Synthesis ({node_id})")
    
    try:
        synthesis_llm = llm.with_structured_output(GlobalSynthesisOutput)
        synthesis_result: GlobalSynthesisOutput = invoke_structured_llm(
            synthesis_llm, synthesis_prompt
        )
        
        # Assign action IDs and enforce capacity
        actions = []
        for action in synthesis_result.actions[:output_capacity]:
            action.action_id = f"{node_id}_{uuid.uuid4().hex[:8]}"
            actions.append(action)
        
        return actions, f"Synthesized {len(proposals)} -> {len(actions)}"
        
    except Exception as e:
        logger.error(f"  [{node_id}] Synthesis failed: {e}")
        return [], f"Synthesis failed: {e}"


def _action_to_proposal(action: SynthesizedAction, source: str) -> ClassProposal:
    """
    Convert a SynthesizedAction back to ClassProposal for passing to upper layers.
    
    This is used by intermediate synthesizers to output in the same format as local proposals,
    enabling seamless propagation up the hierarchy.
    
    Args:
        action: SynthesizedAction from synthesis
        source: Source identifier (e.g., "synth_L1_0")
        
    Returns:
        ClassProposal with equivalent content
    """
    return ClassProposal(
        proposal_id=f"{source}_{uuid.uuid4().hex[:8]}",
        operation=action.operation,
        class_name=action.new_class_name or action.class_name,
        parent_class=action.parent_class,
        description=action.description,
        reason=action.synthesis_reasoning or "Synthesized from child proposals",
        source_classes=action.source_classes,
        source=source,
    )


@graph_node(node_type="generation", log_level="INFO")
def global_synthesis_node(state: FederatedPrimitiveTBoxState) -> Dict[str, Any]:
    """
    Phase 3: Global One-Shot Class Synthesis (Root Node)
    
    This is the ROOT node in the agent tree hierarchy.
    It collects proposals from:
    - Synthesizer nodes (if depth > 1) via intermediate_proposals
    - Leaf nodes directly (if depth == 1) via local_proposals
    
    Workflow:
        1. Collect all class proposals from children
        2. Use shared _synthesize_proposals_core() for synthesis
        3. Apply actions to current TBox (classes only, no DPs)
    
    Unlike intermediate synthesizers:
    - Root has NO output capacity division (uses full proposal_capacity)
    - Root applies actions to TBox (synthesizers just pass proposals)
    """
    logger.info("Phase 3: Global One-Shot Class Synthesis (Root)")
    
    from llm.manager import get_llm_by_purpose
    from .phase3_intermediate import get_root_children
    
    proposal_capacity = state.proposal_capacity
    logger.info(f"  Proposal capacity: {proposal_capacity}")
    
    # ========== Step 1: Collect all proposals ==========
    tree_config = state.tree_config or {}
    depth = tree_config.get("depth", 1)
    n_clusters = state.n_clusters or len(state.cluster_assignments or {})
    
    if depth > 1 and state.intermediate_proposals:
        # Hierarchical mode: collect from highest-level synthesizers
        root_children = get_root_children(tree_config, n_clusters)
        all_proposals = _collect_proposals_from_children(
            state=state,
            child_node_ids=root_children,
        )
        logger.info(f"  [Hierarchical] Collected from {len(root_children)} synthesizers")
    else:
        # Flat mode: collect directly from local_proposals
        all_proposals = _collect_all_proposals(state.local_proposals)
        logger.info(f"  [Flat] Collected directly from {len(state.local_proposals)} groups")
    
    logger.info(f"  Total: {len(all_proposals)} class proposals")
    
    if not all_proposals:
        logger.warning("  No proposals to synthesize")
        return {
            "synthesized_actions": [],
            "synthesis_log": state.synthesis_log + [
                {"iteration": _get_iteration(state), "actions": 0, "reason": "no_proposals"}
            ],
        }
    
    # Get current TBox
    current_tbox = state.current_tbox or state.initial_tbox
    if not current_tbox:
        logger.error("  No TBox available for synthesis")
        return {
            "phase_errors": {"phase_3": ["No TBox available"]},
        }
    
    # Log input proposals
    logger.info(f"  Input proposals ({len(all_proposals)} total):")
    for cp in all_proposals:
        logger.info(f"    <- {cp.operation.upper()} {cp.class_name} from {cp.source}")
    
    # ========== Step 2: Synthesis using shared core ==========
    llm = get_llm_by_purpose(
        purpose=state.llm_purpose,
        override_config=state.llm_override_config
    )
    
    iteration = _get_iteration(state)
    release_log = _format_release_log(state.global_insights)
    target_classes = getattr(state, 'target_classes', 0)
    
    class_actions, notes = _synthesize_proposals_core(
        proposals=all_proposals,
        current_tbox=current_tbox,
        output_capacity=proposal_capacity,  # Root uses full capacity
        llm=llm,
        node_id="global_synthesis",
        release_log=release_log,
        iteration=iteration,
        target_classes=target_classes,
    )
    
    logger.info(f"  {notes}")
    
    # Log synthesized actions
    for ca in class_actions:
        reasoning = ca.synthesis_reasoning[:60] if ca.synthesis_reasoning else "N/A"
        logger.info(f"    -> {ca.operation.upper()} {ca.class_name}: {reasoning}...")
    
    # ========== Post-processing validation ==========
    class_actions = _validate_and_fix_actions(class_actions, current_tbox)
    logger.info(f"  After validation: {len(class_actions)} class actions")
    
    # ========== Step 3: Apply actions to TBox (classes only) ==========
    # Note: _apply_class_actions_to_tbox now performs incremental cycle detection
    # and rejects individual actions that would create cycles
    updated_tbox, rejected_actions = _apply_class_actions_to_tbox(
        current_tbox=current_tbox,
        class_actions=class_actions,
    )
    
    # ========== Step 3.5: Final Cycle Detection (Safety Check) ==========
    # This is a safety check - if incremental detection worked correctly,
    # there should be no cycles here
    cycle_path = detect_class_cycle(updated_tbox.get("classes", []))
    
    if cycle_path:
        # This shouldn't happen if incremental detection worked, but handle it
        logger.error(f"  [Phase3] UNEXPECTED cycle detected after incremental filtering: {' -> '.join(cycle_path)}")
        logger.warning(f"  [Phase3] Rejecting synthesized TBox, reverting to current_tbox")
        synthesis_log_entry = {
            "iteration": _get_iteration(state),
            "n_class_actions": len(class_actions),
            "class_actions": [a.model_dump() for a in class_actions],
            "rejected_reason": f"cycle_detected: {' -> '.join(cycle_path)}",
            "rejected_actions": rejected_actions,
        }
        return {
            "current_tbox": current_tbox,  # Keep original
            "synthesized_actions": [],  # No valid actions
            "synthesis_log": state.synthesis_log + [synthesis_log_entry],
            "local_proposals": {},
            "phase_errors": {"phase_3": [f"Cycle detected in class hierarchy: {' -> '.join(cycle_path)}"]},
        }
    
    # ========== Step 4: Build synthesis log ==========
    synthesis_log_entry = {
        "iteration": _get_iteration(state),
        "n_class_actions": len(class_actions),
        "class_actions": [a.model_dump() for a in class_actions],
    }
    
    # Add rejected actions info if any
    if rejected_actions:
        synthesis_log_entry["rejected_actions_due_to_cycle"] = rejected_actions
    
    return {
        "current_tbox": updated_tbox,
        "synthesized_actions": [a.model_dump() for a in class_actions],
        "synthesis_log": state.synthesis_log + [synthesis_log_entry],
        # Clear local_proposals for next iteration
        "local_proposals": {},
    }


# ============== Helper Functions ==============

def _validate_and_fix_actions(
    actions: List[SynthesizedAction],
    current_tbox: Dict[str, Any],
) -> List[SynthesizedAction]:
    """
    Post-processing validation for synthesized actions.
    
    Validations:
    1. class_name must end with "Column" (non-Column classes rejected)
    2. parent_class must exist in TBox or be "Column"
    3. ADD for existing class -> convert to MODIFY
    4. new_class_name (if provided) must end with "Column"
    
    Returns:
        List of validated/fixed actions (invalid actions removed)
    """
    # Get existing class names
    existing_classes = {c.get("name") for c in current_tbox.get("classes", [])}
    existing_classes.add("Column")  # Root is always valid
    
    validated = []
    
    for action in actions:
        # Get the target class name (may be new_class_name for rename)
        target_name = action.new_class_name or action.class_name
        
        # Validation 1: Class name must end with "Column"
        if not action.class_name.endswith("Column"):
            logger.warning(f"  Validation: Rejected {action.class_name} (not a Column class)")
            continue
            
        if target_name and not target_name.endswith("Column"):
            logger.warning(f"  Validation: Rejected rename to {target_name} (not a Column class)")
            continue
        
        # Validation 2: parent_class must be valid (skip for delete)
        if action.operation != "delete" and action.parent_class:
            if action.parent_class not in existing_classes:
                logger.warning(f"  Validation: Fixed {action.class_name} parent {action.parent_class} -> Column")
                action.parent_class = "Column"
        
        # Validation 3: ADD for existing class -> convert to MODIFY
        if action.operation == "add" and action.class_name in existing_classes:
            logger.warning(f"  Validation: Converted ADD {action.class_name} to MODIFY (already exists)")
            action.operation = "modify"
        
        # Validation 4: merge source_classes should exist
        if action.operation == "merge" and action.source_classes:
            valid_sources = [s for s in action.source_classes if s in existing_classes]
            if len(valid_sources) < len(action.source_classes):
                missing = set(action.source_classes) - set(valid_sources)
                logger.warning(f"  Validation: Merge removed non-existent sources: {missing}")
            action.source_classes = valid_sources
            
            # If no valid sources, convert to ADD
            if not valid_sources:
                logger.warning(f"  Validation: Converted MERGE {action.class_name} to ADD (no valid sources)")
                action.operation = "add"
                action.source_classes = None
        
        validated.append(action)
    
    return validated


def _collect_all_proposals(local_proposals: Dict[str, List[Dict]]) -> List[ClassProposal]:
    """
    Collect all class proposals from all local agents (flat mode).
    
    Args:
        local_proposals: Dict mapping group_id -> list of proposal dicts
        
    Returns:
        List of ClassProposal objects
    """
    all_proposals = []
    
    for group_id, proposals in local_proposals.items():
        for prop_dict in proposals:
            # Handle both old bundle format and new flat format
            if "class_proposal" in prop_dict:
                # Old bundle format
                cp_dict = prop_dict["class_proposal"]
            else:
                # New flat format (just ClassProposal)
                cp_dict = prop_dict
            
            try:
                cp = ClassProposal.model_validate(cp_dict)
                if not cp.source:
                    cp.source = group_id
                all_proposals.append(cp)
            except Exception as e:
                logger.warning(f"Failed to parse proposal from {group_id}: {e}")
    
    return all_proposals


def _collect_proposals_from_children(
    state: FederatedPrimitiveTBoxState,
    child_node_ids: List[str],
) -> List[ClassProposal]:
    """
    Collect proposals from child nodes (hierarchical mode).
    
    Children can be:
    - Synthesizer nodes: get from intermediate_proposals
    - Leaf nodes: get from local_proposals
    
    Args:
        state: Current state
        child_node_ids: List of child node IDs
        
    Returns:
        List of ClassProposal objects
    """
    all_proposals = []
    
    for child_id in child_node_ids:
        if child_id.startswith("synth_L"):
            # Child is a Synthesizer - get from intermediate_proposals
            proposals = state.intermediate_proposals.get(child_id, [])
            for prop_dict in proposals:
                if "class_proposal" in prop_dict:
                    cp_dict = prop_dict["class_proposal"]
                else:
                    cp_dict = prop_dict
                try:
                    cp = ClassProposal.model_validate(cp_dict)
                    all_proposals.append(cp)
                except Exception as e:
                    logger.warning(f"Failed to parse proposal from {child_id}: {e}")
                    
        elif child_id.startswith("local_propose_"):
            # Child is a Leaf - get from local_proposals
            group_id = child_id.replace("local_propose_", "")
            proposals = state.local_proposals.get(group_id, [])
            for prop_dict in proposals:
                if "class_proposal" in prop_dict:
                    cp_dict = prop_dict["class_proposal"]
                else:
                    cp_dict = prop_dict
                try:
                    cp = ClassProposal.model_validate(cp_dict)
                    if not cp.source:
                        cp.source = group_id
                    all_proposals.append(cp)
                except Exception as e:
                    logger.warning(f"Failed to parse proposal from {group_id}: {e}")
    
    return all_proposals


def _get_iteration(state: FederatedPrimitiveTBoxState) -> int:
    """Get current iteration number from synthesis log."""
    return len(state.synthesis_log) + 1


def _format_all_proposals(proposals: List[ClassProposal]) -> str:
    """Format all proposals for the synthesis prompt."""
    if not proposals:
        return "No proposals submitted."
    
    lines = []
    for i, cp in enumerate(proposals, 1):
        lines.append(f"### Proposal {i} (from {cp.source})")
        lines.append(f"- **Operation**: {cp.operation}")
        lines.append(f"- **Class Name**: {cp.class_name}")
        
        # Operation-specific fields
        if cp.operation == "modify" and cp.new_class_name:
            lines.append(f"- **New Class Name**: {cp.new_class_name}")
        if cp.operation == "merge" and cp.source_classes:
            lines.append(f"- **Source Classes**: {', '.join(cp.source_classes)}")
        if cp.operation != "delete":
            lines.append(f"- **Parent Class**: {cp.parent_class or 'Column'}")
            lines.append(f"- **Description**: {cp.description or 'N/A'}")
        if cp.reason:
            lines.append(f"- **Reason**: {cp.reason}")
        lines.append("")
    
    return "\n".join(lines)


def _format_current_classes(tbox: Dict[str, Any]) -> str:
    """Format current TBox classes as tree hierarchy for the synthesis prompt."""
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


def _apply_class_actions_to_tbox(
    current_tbox: Dict[str, Any],
    class_actions: List[SynthesizedAction],
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Apply synthesized class actions to create updated TBox.
    
    Supported operations:
    - add: Create new class
    - modify: Update existing class (with optional rename via new_class_name)
    - delete: Remove class and cascade to DataProperties
    - merge: Combine source_classes into target class
    
    Note: DataProperties are handled with cascade delete/update for affected classes.
    
    Returns:
        Tuple of (updated_tbox, rejected_actions)
        - updated_tbox: The TBox after applying valid actions
        - rejected_actions: List of action descriptions that were rejected due to cycles
    """
    # Copy current TBox
    classes = list(current_tbox.get("classes", []))
    data_properties = list(current_tbox.get("data_properties", []))
    
    # Build name-to-index map
    class_idx_map = {c.get("name"): i for i, c in enumerate(classes)}
    
    # Track deleted and renamed classes for DP cascade
    deleted_classes = set()
    renamed_classes = {}  # old_name -> new_name
    merged_classes = {}   # source_class -> target_class
    
    # Track rejected actions
    rejected_actions = []
    
    # Apply class actions
    for action in class_actions:
        
        if action.operation == "delete":
            # ========== DELETE Operation ==========
            if action.class_name in class_idx_map:
                idx = class_idx_map[action.class_name]
                logger.info(f"  Deleting class: {action.class_name}")
                classes[idx] = None  # Mark for removal
                deleted_classes.add(action.class_name)
                del class_idx_map[action.class_name]
            else:
                logger.warning(f"  Delete target {action.class_name} not found, skipping")
                
        elif action.operation == "merge":
            # ========== MERGE Operation ==========
            # source_classes -> class_name (target)
            source_classes = action.source_classes or []
            target_name = action.class_name
            proposed_parent = action.parent_class or "Column"
            
            # Check for cycle before creating target class with parent
            current_classes_snapshot = [c for c in classes if c is not None]
            if would_create_cycle(current_classes_snapshot, target_name, proposed_parent):
                logger.warning(f"  MERGE {target_name} (parent={proposed_parent}) rejected: would create cycle")
                rejected_actions.append(f"MERGE {target_name} (parent={proposed_parent})")
                continue
            
            # Create target class if not exists
            if target_name not in class_idx_map:
                new_class = {
                    "name": target_name,
                    "description": action.description,
                    "parent_classes": [proposed_parent],
                    "_synthesis_action_id": action.action_id,
                    "_synthesis_reasoning": action.synthesis_reasoning,
                }
                classes.append(new_class)
                class_idx_map[target_name] = len(classes) - 1
                logger.info(f"  Merge: Created target class {target_name}")
            
            # Remove source classes
            for src_cls in source_classes:
                if src_cls in class_idx_map:
                    idx = class_idx_map[src_cls]
                    logger.info(f"  Merge: Removing source class {src_cls} -> {target_name}")
                    classes[idx] = None
                    deleted_classes.add(src_cls)
                    merged_classes[src_cls] = target_name
                    del class_idx_map[src_cls]
                    
        elif action.operation == "modify":
            # ========== MODIFY Operation (with optional rename) ==========
            if action.class_name in class_idx_map:
                idx = class_idx_map[action.class_name]
                target_class_name = action.new_class_name if action.new_class_name else action.class_name
                
                # Check for cycle if parent_class is being changed
                if action.parent_class:
                    current_classes_snapshot = [c for c in classes if c is not None]
                    if would_create_cycle(current_classes_snapshot, target_class_name, action.parent_class):
                        logger.warning(f"  MODIFY {action.class_name} (parent={action.parent_class}) rejected: would create cycle")
                        rejected_actions.append(f"MODIFY {action.class_name} (parent={action.parent_class})")
                        continue
                
                # Handle rename
                if action.new_class_name and action.new_class_name != action.class_name:
                    old_name = action.class_name
                    new_name = action.new_class_name
                    logger.info(f"  Modify with rename: {old_name} -> {new_name}")
                    
                    classes[idx]["name"] = new_name
                    renamed_classes[old_name] = new_name
                    
                    # Update index map
                    del class_idx_map[old_name]
                    class_idx_map[new_name] = idx
                
                # Update other fields (only if provided)
                if action.description:
                    classes[idx]["description"] = action.description
                if action.parent_class:
                    classes[idx]["parent_classes"] = [action.parent_class]
                    
                # Metadata
                classes[idx]["_synthesis_action_id"] = action.action_id
                classes[idx]["_synthesis_reasoning"] = action.synthesis_reasoning
            else:
                # Target not found, add as new
                logger.warning(f"  Modify target {action.class_name} not found, adding as new")
                new_class_name = action.new_class_name or action.class_name
                proposed_parent = action.parent_class or "Column"
                
                # Check for cycle before adding
                current_classes_snapshot = [c for c in classes if c is not None]
                if would_create_cycle(current_classes_snapshot, new_class_name, proposed_parent):
                    logger.warning(f"  MODIFY-AS-ADD {new_class_name} (parent={proposed_parent}) rejected: would create cycle")
                    rejected_actions.append(f"MODIFY-AS-ADD {new_class_name} (parent={proposed_parent})")
                    continue
                
                new_class = {
                    "name": new_class_name,
                    "description": action.description,
                    "parent_classes": [proposed_parent],
                    "_synthesis_action_id": action.action_id,
                    "_synthesis_reasoning": action.synthesis_reasoning,
                }
                classes.append(new_class)
                class_idx_map[new_class["name"]] = len(classes) - 1
                
        else:  # add
            # ========== ADD Operation ==========
            proposed_parent = action.parent_class or "Column"
            
            # Check for cycle before adding
            current_classes_snapshot = [c for c in classes if c is not None]
            if would_create_cycle(current_classes_snapshot, action.class_name, proposed_parent):
                logger.warning(f"  ADD {action.class_name} (parent={proposed_parent}) rejected: would create cycle")
                rejected_actions.append(f"ADD {action.class_name} (parent={proposed_parent})")
                continue
            
            new_class = {
                "name": action.class_name,
                "description": action.description,
                "parent_classes": [proposed_parent],
                "_synthesis_action_id": action.action_id,
                "_synthesis_reasoning": action.synthesis_reasoning,
            }
            
            if action.class_name in class_idx_map:
                logger.warning(f"  Add class {action.class_name} skipped - already exists, merging metadata")
                idx = class_idx_map[action.class_name]
                classes[idx]["_synthesis_action_id"] = action.action_id
            else:
                classes.append(new_class)
                class_idx_map[action.class_name] = len(classes) - 1
    
    # Remove None entries (deleted classes)
    classes = [c for c in classes if c is not None]
    
    # ========== CASCADE: Update parent_classes references ==========
    for cls in classes:
        parent_classes = cls.get("parent_classes", [])
        updated_parents = []
        for p in parent_classes:
            if p in deleted_classes:
                # Parent was deleted, fall back to "Column"
                updated_parents.append("Column")
            elif p in renamed_classes:
                updated_parents.append(renamed_classes[p])
            elif p in merged_classes:
                updated_parents.append(merged_classes[p])
            else:
                updated_parents.append(p)
        cls["parent_classes"] = updated_parents
    
    # ========== CASCADE: DataProperty domain updates ==========
    updated_dps = []
    for dp in data_properties:
        domain = dp.get("domain") or dp.get("domain_class")
        
        # Check if domain class was deleted
        if domain in deleted_classes:
            logger.info(f"  Cascade delete: DP {dp.get('name')} (domain {domain} deleted)")
            continue  # Remove this DP
            
        # Check if domain class was renamed
        if domain in renamed_classes:
            logger.info(f"  Cascade rename: DP {dp.get('name')} domain {domain} -> {renamed_classes[domain]}")
            dp_copy = dict(dp)
            if "domain" in dp_copy:
                dp_copy["domain"] = renamed_classes[domain]
            if "domain_class" in dp_copy:
                dp_copy["domain_class"] = renamed_classes[domain]
            updated_dps.append(dp_copy)
            continue
            
        # Check if domain class was merged
        if domain in merged_classes:
            logger.info(f"  Cascade merge: DP {dp.get('name')} domain {domain} -> {merged_classes[domain]}")
            dp_copy = dict(dp)
            if "domain" in dp_copy:
                dp_copy["domain"] = merged_classes[domain]
            if "domain_class" in dp_copy:
                dp_copy["domain_class"] = merged_classes[domain]
            updated_dps.append(dp_copy)
            continue
            
        # No change needed
        updated_dps.append(dp)
    
    # Log rejected actions if any
    if rejected_actions:
        logger.warning(f"  Rejected {len(rejected_actions)} actions due to cycle detection:")
        for ra in rejected_actions:
            logger.warning(f"    - {ra}")
    
    return {
        "classes": classes,
        "data_properties": updated_dps,
    }, rejected_actions
