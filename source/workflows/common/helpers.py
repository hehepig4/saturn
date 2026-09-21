"""
Common Helper Functions for Subgraphs

Shared utilities including:
- State update helpers
- Cache summary formatting
- Message handling
"""

from loguru import logger
from typing import Any

def update_state_safe(state: Any, **updates) -> Any:
    """
    Safely update state with new values.
    
    Handles messages separately to maintain list consistency.
    Trajectory tracking is done at node level via decorators.
    
    Args:
        state: State object to update
        **updates: Fields to update
        
    Returns:
        Updated state object
    """
    # Handle messages separately
    new_messages_list = updates.pop('messages', None)
    
    # Create new state with non-message updates
    state_dict = state.model_dump()
    state_dict.update(updates)
    new_state = state.__class__(**state_dict)
    
    # If messages were provided, add new ones
    if new_messages_list is not None:
        original_count = len(state.messages)
        new_state.messages = list(state.messages)
        
        # Preserve trajectory_log if exists
        if hasattr(state, 'trajectory_log'):
            new_state.trajectory_log = list(state.trajectory_log)
        
        # Add only new messages
        for msg in new_messages_list[original_count:]:
            new_state.messages.append(msg)
    
    return new_state
