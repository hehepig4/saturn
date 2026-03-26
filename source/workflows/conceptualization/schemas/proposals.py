"""
Proposal Data Structures for Federated Primitive TBox

Defines ClassProposal for class change proposals.

Design Principles:
    - Operations: add, modify (with rename), delete, merge
    - Modify uses None for unchanged fields
    - Single inheritance (parent_class is str, not List[str])
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ClassProposal(BaseModel):
    """Class change proposal, used by both Local and Global agents.
    
    Operations:
    - add: Create new class (class_name = new name, fill all fields)
    - modify: Update existing class (class_name = target, fill fields to change)
              Can also rename via new_class_name
    - delete: Remove class (class_name = target, reason required)
    - merge: Combine multiple classes (source_classes → class_name)
    
    Note: parent_class is a single string (not List) to enforce single inheritance.
    """
    proposal_id: str = Field(
        ...,
        description="Unique identifier, format: '{source}_{uuid}'"
    )
    operation: Literal["add", "modify", "delete", "merge"] = Field(
        ...,
        description="Operation type: add, modify, delete, or merge"
    )
    source: str = Field(
        ...,
        description="Proposal source, e.g., 'local_group_1', 'global'"
    )
    reason: str = Field(
        ...,
        description="Brief explanation for this proposal"
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
    
    # Class definition fields (add/modify/merge)
    description: Optional[str] = Field(
        default=None,
        description="Class description (not for delete)"
    )
    parent_class: Optional[str] = Field(
        default=None,
        description="Single parent class name (not for delete)"
    )
    
    # Merge field
    source_classes: Optional[List[str]] = Field(
        default=None,
        description="Classes to merge into class_name (merge operation only)"
    )


