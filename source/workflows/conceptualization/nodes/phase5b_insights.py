"""
Phase 5b: Insights Synthesizer

Synthesizes a compressed GlobalInsights from:
- Previous insights (RNN-style memory)
- Synthesis actions (Phase 3)
- Review actions (Phase 5)

This creates a unified "Release Log" that guides all agents in next iteration.

Architecture follows Hindsight's 4-layer model:
- Layer 1 (World): Current TBox state - not stored, retrieved directly
- Layer 2 (Experiences): changelog - what happened
- Layer 3 (Opinions): Implicit in patterns - beliefs about what works
- Layer 4 (Observations): patterns - generalized rules
"""

from typing import Dict, Any, List, Optional
from loguru import logger

from workflows.common.node_decorators import graph_node
from workflows.conceptualization.state import FederatedPrimitiveTBoxState
from workflows.conceptualization.schemas.insights import (
    GlobalInsights,
    InsightsSynthesizerOutput,
)
from workflows.conceptualization.prompts.templates import INSIGHTS_SYNTHESIZER_PROMPT
from .prompt_log_utils import log_prompt_once


@graph_node(node_type="generation", log_level="INFO")
def insights_synthesizer_node(state: FederatedPrimitiveTBoxState) -> Dict[str, Any]:
    """
    Phase 5b: Synthesize GlobalInsights from iteration actions.
    
    RNN-like update: new_insights = f(old_insights, synthesis_actions, review_actions)
    
    Returns:
        Updated global_insights
    """
    from llm.manager import get_llm_by_purpose
    from llm.invoke_with_stats import invoke_structured_llm
    
    logger.info("Phase 5b: Insights Synthesizer")
    
    # Get current iteration
    current_iteration = _get_iteration(state)
    logger.info(f"  Iteration: {current_iteration}")
    
    # Get previous insights (may be None for first iteration)
    previous_insights = state.global_insights
    
    # Get synthesis actions (from Phase 3)
    synthesis_actions = state.synthesized_actions or []
    
    # Get review actions (from Phase 5 - stored in current_tbox's review log)
    # We need to extract them from the review output
    review_actions = _extract_review_actions(state)
    
    # Get current TBox stats
    current_tbox = state.current_tbox
    n_classes = len(current_tbox.get("classes", [])) if current_tbox else 0
    
    logger.info(f"  Synthesis actions: {len(synthesis_actions)}")
    logger.info(f"  Review actions: {len(review_actions)}")
    logger.info(f"  Current TBox: {n_classes} classes")
    
    # Format previous insights for prompt
    prev_insights_text = _format_previous_insights(previous_insights)
    
    # Format actions for prompt
    synthesis_text = _format_actions(synthesis_actions, "Synthesis")
    review_text = _format_actions(review_actions, "Review")
    
    # Compute stats
    prev_deletions = previous_insights.total_deletions if previous_insights else 0
    prev_merges = previous_insights.total_merges if previous_insights else 0
    new_deletions = _count_deletions(synthesis_actions) + _count_deletions(review_actions)
    new_merges = _count_merges(synthesis_actions) + _count_merges(review_actions)
    
    # Build prompt
    prompt = INSIGHTS_SYNTHESIZER_PROMPT.format(
        current_iteration=current_iteration,
        previous_insights=prev_insights_text,
        synthesis_actions=synthesis_text,
        review_actions=review_text,
        n_classes=n_classes,
        total_deletions=prev_deletions + new_deletions,
        total_merges=prev_merges + new_merges,
    )
    
    # Per-iteration prompt logging
    log_prompt_once("insights_synthesizer", current_iteration, prompt, "Insights Synthesizer")
    
    # Get LLM and invoke
    llm = get_llm_by_purpose(
        purpose=state.llm_purpose,
        override_config=state.llm_override_config
    )
    
    try:
        insights_llm = llm.with_structured_output(InsightsSynthesizerOutput)
        result: InsightsSynthesizerOutput = invoke_structured_llm(insights_llm, prompt)
        
        # Soft-enforce limits (truncate if needed, don't fail)
        from config.truncation_limits import TruncationLimits, truncate_value
        changelog = result.changelog or ""
        if len(changelog) > TruncationLimits.CHANGELOG_MAX_LENGTH:
            # Truncate long changelog, keep recent iterations
            original_len = len(changelog)
            changelog = truncate_value(changelog, TruncationLimits.CHANGELOG_MAX_LENGTH)
            logger.warning(f"  Changelog truncated from {original_len} to {TruncationLimits.CHANGELOG_MAX_LENGTH} chars")
        
        patterns = result.patterns[:20]  # Keep at most 20 patterns
        
        # Create new GlobalInsights
        new_insights = GlobalInsights(
            changelog=changelog,
            patterns=patterns,
            iteration=current_iteration,
            total_classes=n_classes,
            total_deletions=prev_deletions + new_deletions,
            total_merges=prev_merges + new_merges,
        )
        
        # Log the new insights
        logger.info("=" * 60)
        logger.info("[New Global Insights]")
        logger.info("=" * 60)
        logger.info(f"Iteration: {new_insights.iteration}")
        logger.info(f"Changelog:\n{new_insights.changelog}")
        logger.info(f"Patterns ({len(new_insights.patterns)}):")
        for i, p in enumerate(new_insights.patterns, 1):
            logger.info(f"  {i}. {p}")
        logger.info(f"Stats: {new_insights.total_classes} classes, {new_insights.total_deletions} deletions, {new_insights.total_merges} merges")
        logger.info("=" * 60)
        
        return {"global_insights": new_insights}
        
    except Exception as e:
        logger.error(f"  Insights synthesis failed: {e}")
        # Return empty insights on failure to not block pipeline
        return {
            "global_insights": GlobalInsights(
                iteration=current_iteration,
                total_classes=n_classes,
                changelog=f"Iter {current_iteration}: (synthesis failed)",
                patterns=[],
            )
        }


# ============== Helper Functions ==============

def _get_iteration(state: FederatedPrimitiveTBoxState) -> int:
    """Get current iteration number from synthesis log."""
    synthesis_log = state.synthesis_log or []
    return len(synthesis_log)


def _extract_review_actions(state: FederatedPrimitiveTBoxState) -> List[Dict]:
    """Extract review actions from state.review_output.
    
    review_output is set by phase5_review and contains:
    - iteration: int
    - actions: List[Dict] (serialized ReviewAction objects)
    - summary: str
    - n_actions: int
    """
    review_output = state.review_output
    if not review_output:
        return []
    
    # Return the actions from review_output
    actions = review_output.get("actions", [])
    summary = review_output.get("summary", "")
    n_actions = review_output.get("n_actions", 0)
    
    # If we have actual actions, return them
    if actions:
        return actions
    
    # Fallback: return summary-based entry for context
    if summary:
        return [{"summary": summary, "n_actions": n_actions}]
    
    return []


def _format_previous_insights(insights: Optional[GlobalInsights]) -> str:
    """Format previous insights for prompt."""
    if not insights or insights.is_empty():
        return "No previous insights (first iteration)."
    
    lines = [
        f"### Previous Changelog (Iter 1-{insights.iteration})",
        insights.changelog,
        "",
        f"### Previous Patterns ({len(insights.patterns)}):",
    ]
    for p in insights.patterns:
        lines.append(f"- {p}")
    
    return "\n".join(lines)


def _format_actions(actions: List[Dict], source: str) -> str:
    """Format actions list for prompt."""
    if not actions:
        return f"No {source} actions this iteration."
    
    lines = [f"### {source} Actions:"]
    
    for action in actions:
        if "summary" in action:
            # Review log format (summary-only)
            lines.append(f"- Summary: {action['summary'][:200]}...")
            lines.append(f"  ({action.get('n_actions', '?')} actions)")
        else:
            # Synthesis action format
            op = action.get("operation", "?")
            name = action.get("class_name", "?")
            
            if op == "add":
                parent = action.get("parent_class", "Column")
                lines.append(f"- ADD {name} (parent: {parent})")
            elif op == "modify":
                new_name = action.get("new_class_name")
                if new_name:
                    lines.append(f"- MODIFY {name} → {new_name}")
                else:
                    lines.append(f"- MODIFY {name}")
            elif op == "delete":
                lines.append(f"- DELETE {name}")
            elif op == "merge":
                sources = action.get("source_classes", [])
                lines.append(f"- MERGE {sources} → {name}")
            else:
                lines.append(f"- {op.upper()} {name}")
    
    return "\n".join(lines)


def _count_deletions(actions: List[Dict]) -> int:
    """Count delete operations in actions."""
    count = 0
    for action in actions:
        if action.get("operation") == "delete":
            count += 1
        elif "n_actions" in action:
            # Rough estimate from review summary
            # Assume ~50% of review actions are deletions
            count += action.get("n_actions", 0) // 2
    return count


def _count_merges(actions: List[Dict]) -> int:
    """Count merge operations in actions."""
    count = 0
    for action in actions:
        if action.get("operation") == "merge":
            count += 1
    return count
