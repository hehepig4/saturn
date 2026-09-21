"""
Synthesis Data Structures for Federated Primitive TBox

Defines the output structures for Global Synthesis (Phase 3).

Design: Global Agent outputs final actions directly with operations: add, modify, delete, merge.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class SynthesizedAction(BaseModel):
    """A single action output by Global Agent after synthesizing proposals.
    
    Operations:
    - add: Create new class
    - modify: Update existing class (can rename via new_class_name)
    - delete: Remove class
    - merge: Combine multiple classes into one
    """
    action_id: str = Field(
        ...,
        description="Unique action identifier"
    )
    operation: Literal["add", "modify", "delete", "merge"] = Field(
        ...,
        description="Operation type: add, modify, delete, or merge"
    )
    
    # Target class name
    class_name: str = Field(
        ...,
        description="add: new name; modify/delete: target name; merge: merged result name"
    )
    
    # Rename field (modify only)
    new_class_name: Optional[str] = Field(
        default=None,
        description="New name when renaming (modify operation only)"
    )
    
    # Class definition fields (optional for delete)
    description: Optional[str] = Field(
        default=None,
        description="Final description (not required for delete)"
    )
    parent_class: Optional[str] = Field(
        default=None,
        description="Final parent class name (not required for delete)"
    )
    
    # Merge field
    source_classes: Optional[List[str]] = Field(
        default=None,
        description="Classes to merge into class_name (merge operation only)"
    )
    
    synthesis_reasoning: str = Field(
        ...,
        description="Global Agent's reasoning for this decision"
    )


class GlobalSynthesisOutput(BaseModel):
    """Output schema for Phase 3 Class Synthesis stage."""
    actions: List[SynthesizedAction] = Field(
        default_factory=list,
        description="List of class actions to execute"
    )


# NOTE: SynthesisActionBundle was removed as it was never used
