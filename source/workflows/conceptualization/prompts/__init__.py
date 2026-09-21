"""
Prompts Package for Federated Primitive TBox
"""

from .templates import (
    # Phase 1
    GLOBAL_INIT_CLASSES_PROMPT,
    # Phase 2
    LOCAL_CLASS_PROPOSAL_PROMPT,
    # Phase 3 (One-Shot Synthesis)
    GLOBAL_ONE_SHOT_SYNTHESIS_PROMPT,
    # Phase 4
    LOCAL_VOTING_PROMPT,
    # Phase 5
    GLOBAL_REVIEW_PROMPT,
)

__all__ = [
    "GLOBAL_INIT_CLASSES_PROMPT",
    "LOCAL_CLASS_PROPOSAL_PROMPT",
    "GLOBAL_ONE_SHOT_SYNTHESIS_PROMPT",
    "LOCAL_VOTING_PROMPT",
    "GLOBAL_REVIEW_PROMPT",
]
