"""
Phase 1: Global Initialization

Creates initial TBox classes from backbone CQs and cluster center samples.
DataProperty generation is deferred to final phase (phase_dp_final.py).
"""

from typing import Dict, Any, List
from loguru import logger
from pydantic import BaseModel, Field

from workflows.common.node_decorators import graph_node
from workflows.conceptualization.state import FederatedPrimitiveTBoxState
from workflows.conceptualization.prompts.templates import (
    GLOBAL_INIT_CLASSES_PROMPT,
)
from workflows.conceptualization.nodes.prompt_log_utils import build_target_classes_hint


# ============== Global for one-time prompt logging ==============

_INIT_CLASS_PROMPT_LOGGED = False


# ============== LLM Output Schemas ==============

class InitialClass(BaseModel):
    """Schema for initial class output."""
    name: str  # CamelCase class name
    description: str  # Class description
    parent_class: str  # Single parent class name


class InitialClassesOutput(BaseModel):
    """LLM output for Phase 1 class generation."""
    classes: List[InitialClass] = Field(default_factory=list)


@graph_node(node_type="generation", log_level="INFO")
def global_init_node(state: FederatedPrimitiveTBoxState) -> Dict[str, Any]:
    """
    Phase 1: Generate initial TBox classes from backbone CQs.
    
    Only generates classes. DataProperties are generated in phase_dp_final.py
    after all class iterations are complete.
    """
    logger.info("Phase 1: Global Initialization")
    
    from llm.manager import get_llm_by_purpose
    from llm.invoke_with_stats import invoke_structured_llm
    
    backbone_cqs = state.backbone_cqs
    if not backbone_cqs:
        logger.warning("  No backbone CQs, using all CQs")
        backbone_cqs = state.competency_questions[:30]  # Fallback
    
    logger.info(f"  Backbone CQs: {len(backbone_cqs)}")
    
    # Format CQs for prompt
    cqs_block = _format_cqs(backbone_cqs)
    
    # Get LLM
    llm = get_llm_by_purpose(
        purpose=state.llm_purpose,
        override_config=state.llm_override_config
    )
    
    # ========== Generate Classes ==========
    logger.info("  Generating initial classes...")
    
    # Build target classes hint
    target_classes = getattr(state, 'target_classes', 0)
    target_classes_hint = build_target_classes_hint(target_classes)
    
    class_prompt = GLOBAL_INIT_CLASSES_PROMPT.format(
        domain=state.dataset_name,
        target_classes_hint=target_classes_hint,
        cqs_block=cqs_block,
    )
    
    # One-time prompt logging for debug
    global _INIT_CLASS_PROMPT_LOGGED
    if not _INIT_CLASS_PROMPT_LOGGED:
        logger.debug("=" * 80)
        logger.debug("[Global Init Classes] FIRST PROMPT (one-time log):")
        logger.debug("=" * 80)
        logger.debug(class_prompt)
        logger.debug("=" * 80)
        _INIT_CLASS_PROMPT_LOGGED = True
    
    try:
        class_llm = llm.with_structured_output(InitialClassesOutput)
        class_result: InitialClassesOutput = invoke_structured_llm(class_llm, class_prompt)
        
        # Convert parent_class (str) to parent_classes (List[str]) for compatibility
        classes = []
        for c in class_result.classes:
            cls_dict = c.model_dump()
            # Convert parent_class to parent_classes list for tree formatter
            parent = cls_dict.pop("parent_class", "Column")
            cls_dict["parent_classes"] = [parent] if parent else ["Column"]
            classes.append(cls_dict)
        
        logger.info(f"    Generated {len(classes)} classes")
    except Exception as e:
        logger.error(f"  Class generation failed: {e}")
        classes = []
    
    if not classes:
        logger.error("  No classes generated, cannot proceed")
        return {
            "initial_tbox": {"classes": [], "data_properties": []},
            "phase_errors": {"phase_1": ["No classes generated"]},
        }
    
    # Build initial TBox (no DataProperties - deferred to final phase)
    initial_tbox = {
        "classes": classes,
        "data_properties": [],  # DP generation deferred to phase_dp_final
    }
    
    logger.info(f"  Initial TBox: {len(classes)} classes (DP generation deferred)")
    
    return {
        "initial_tbox": initial_tbox,
        "current_tbox": initial_tbox,  # Set as current for iteration
    }


def _format_cqs(cqs: List[Dict]) -> str:
    """Format CQs for prompt - no Related columns since backbone CQs are aggregated."""
    lines = []
    for i, cq in enumerate(cqs):
        cq_type = cq.get("type", "CQ")
        question = cq.get("question", cq.get("text", ""))
        lines.append(f"{i+1}. [{cq_type}] {question}")
    return "\n".join(lines)


