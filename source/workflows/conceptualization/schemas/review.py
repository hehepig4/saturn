"""
Review Data Structures for Federated Primitive TBox

Defines the output structures for Global Review Agent (Phase 5).

Design: Review Agent uses voting results to optimize the TBox with
the same operations as Synthesis Agent: add, modify, delete, merge.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ReviewAction(BaseModel):
    """A single review action output by Global Review Agent.
    
    Uses same operations as Local/Synthesis Agents: add, modify, delete, merge.
    """
    action_id: str = Field(
        ...,
        description="Unique action identifier (e.g., 'review_001')"
    )
    operation: Literal["add", "modify", "delete", "merge"] = Field(
        ...,
        description="Review operation: add, modify, delete, or merge"
    )
    
    # Target class name
    class_name: str = Field(
        ...,
        description="Target class name for modify/delete; result class name for merge; new class name for add"
    )
    
    # Rename (modify only)
    new_class_name: Optional[str] = Field(
        default=None,
        description="New name for the class (modify operation with rename)"
    )
    
    # Merge operation fields
    source_classes: Optional[List[str]] = Field(
        default=None,
        description="Classes to merge into class_name (merge operation)"
    )
    
    # Parent class (add/modify/merge)
    parent_class: Optional[str] = Field(
        default=None,
        description="Parent class (add/modify/merge operations)"
    )
    
    # Common fields
    description: Optional[str] = Field(
        default=None,
        description="Class description (add/modify/merge operations)"
    )
    
    # Evidence fields
    voting_evidence: str = Field(
        ...,
        description="Voting statistics that justify this action"
    )
    reasoning: str = Field(
        ...,
        description="Why this action improves the TBox quality"
    )


class ClassVotingSummary(BaseModel):
    """Voting summary for a single class across all Local Agents."""
    class_name: str = Field(
        ...,
        description="Name of the class"
    )
    total_votes: int = Field(
        ...,
        description="Total votes from all agents"
    )
    positive_votes: int = Field(
        ...,
        description="Number of agents that voted 'useful'"
    )
    coverage_ratio: float = Field(
        ...,
        description="Fraction of clusters that find this class useful (0.0-1.0)"
    )
    voting_agents: List[str] = Field(
        default_factory=list,
        description="List of agent IDs that voted for this class"
    )


class GlobalReviewOutput(BaseModel):
    """Complete output from Global Review Agent."""
    actions: List[ReviewAction] = Field(
        default_factory=list,
        description="List of review actions to apply (empty if no changes needed)"
    )
    review_summary: str = Field(
        ...,
        description="Comprehensive summary with class-specific information and guidance for future iterations"
    )
