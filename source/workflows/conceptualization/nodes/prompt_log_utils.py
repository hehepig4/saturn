"""
Prompt Logging Utilities

Centralized management for per-iteration prompt logging.
Each node type logs its prompt once per iteration, enabling debugging
while avoiding excessive log output.
"""

import threading
from typing import Set, Dict
from loguru import logger


# Thread-safe storage for logged iterations per node type
_logged_iterations: Dict[str, Set[int]] = {}
_lock = threading.Lock()


def build_target_classes_hint(target_classes: int, current_n_classes: int = 0) -> str:
    """
    Build a target classes hint string for prompts.
    
    Args:
        target_classes: Target number of classes (0 = no specific target)
        current_n_classes: Current number of classes in TBox
        
    Returns:
        Formatted hint string for inclusion in prompts
    """
    if target_classes <= 0:
        return "No specific target. Let the CQs and domain guide the number of classes."
    
    hint = f"Target approximately {target_classes} classes."
    if current_n_classes > 0:
        diff = target_classes - current_n_classes
        if diff > 0:
            hint += f" Currently {current_n_classes} classes (need ~{diff} more)."
        elif diff < 0:
            hint += f" Currently {current_n_classes} classes (consider consolidating ~{-diff})."
        else:
            hint += f" Currently at target ({current_n_classes} classes)."
    
    return hint


def should_log_prompt(node_type: str, iteration: int) -> bool:
    """
    Check if prompt should be logged for this node type at this iteration.
    
    Returns True if this is the first call for this (node_type, iteration) pair.
    Thread-safe.
    
    Args:
        node_type: Identifier for the node type (e.g., "local_proposal", "synthesis")
        iteration: Current iteration number
        
    Returns:
        True if prompt should be logged, False if already logged this iteration
    """
    with _lock:
        if node_type not in _logged_iterations:
            _logged_iterations[node_type] = set()
        
        if iteration in _logged_iterations[node_type]:
            return False
        
        _logged_iterations[node_type].add(iteration)
        return True


def log_prompt_once(node_type: str, iteration: int, prompt: str, label: str = None):
    """
    Log a prompt if it hasn't been logged for this iteration yet.
    
    Args:
        node_type: Identifier for the node type
        iteration: Current iteration number
        prompt: The prompt text to log
        label: Optional label for the log output
    """
    if should_log_prompt(node_type, iteration):
        label = label or node_type.replace("_", " ").title()
        logger.debug("=" * 80)
        logger.debug(f"[{label}] PROMPT (Iteration {iteration}):")
        logger.debug("=" * 80)
        logger.debug(prompt)
        logger.debug("=" * 80)

