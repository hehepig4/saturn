"""
Insights Schema for Federated Primitive TBox

Defines GlobalInsights - a compressed memory structure following Hindsight's architecture:
- Layer 1 (World/Facts): Current TBox state - NOT stored, retrieved from TBox directly
- Layer 2 (Experiences): changelog - compressed summary of past actions
- Layer 3 (Opinions): Implicit in patterns - high confidence beliefs
- Layer 4 (Observations): patterns - generalized rules from experience

This provides a unified "Release Log" view for all agents.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class GlobalInsights(BaseModel):
    """
    Compressed, bounded memory for guiding all agents.
    
    Inspired by Hindsight's 4-layer architecture, simplified to Release Log format.
    Fields are soft-bounded in code (truncated if needed) rather than hard-validated.
    
    Total size: ~300 tokens (bounded)
    """
    
    # === Layer 2: Experiences (compressed changelog) ===
    changelog: str = Field(
        default="",
        description=(
            "Compressed changelog of all iterations. "
            "Format: 'Iter N: [action summary]' per iteration. "
            "Older iterations get progressively compressed. "
            "E.g., 'Iter 1-2: Major identifier cleanup. Iter 3: Removed Sports* classes.'"
        )
    )
    
    # === Layer 3 & 4: Opinions + Observations (patterns) ===
    patterns: List[str] = Field(
        default_factory=list,
        description=(
            "Up to 20 generalized patterns/rules learned from experience. "
            "E.g., 'Domain-specific subclasses (Sports*, Election*) rarely survive review'"
        )
    )
    
    # === Meta ===
    iteration: int = Field(
        default=0,
        description="Current iteration number"
    )
    
    total_classes: int = Field(
        default=0,
        description="Current number of classes in TBox"
    )
    
    total_deletions: int = Field(
        default=0,
        description="Cumulative count of deletions across all iterations"
    )
    
    total_merges: int = Field(
        default=0,
        description="Cumulative count of merges across all iterations"
    )
    
    def format_release_log(self) -> str:
        """
        Format insights as content for the Previous Learnings section.
        
        Returns:
            Formatted string (without section header) for inclusion in prompts.
        """
        if self.iteration == 0:
            return "First iteration - no previous learnings."
        
        lines = []
        
        # What Changed
        lines.append(f"**After Iteration {self.iteration}:**")
        lines.append(self.changelog if self.changelog else "No changes recorded.")
        lines.append("")
        
        # Patterns Learned
        if self.patterns:
            lines.append("**Patterns:**")
            for pattern in self.patterns:
                lines.append(f"- {pattern}")
        
        # Stats
        lines.append("")
        lines.append(f"*Stats: {self.total_classes} classes, {self.total_deletions} deletions, {self.total_merges} merges*")
        
        return "\n".join(lines)
    
    def is_empty(self) -> bool:
        """Check if insights are empty (first iteration)."""
        return self.iteration == 0 and not self.changelog and not self.patterns


class InsightsSynthesizerOutput(BaseModel):
    """
    Output schema for the Insights Synthesizer LLM call.
    
    The LLM will generate this structured output to update GlobalInsights.
    No hard validation constraints - soft limits enforced in code.
    """
    
    changelog: str = Field(
        ...,
        description=(
            "Updated changelog incorporating new actions. "
            "Compress older entries while preserving recent details. "
            "Target: ~1000 characters."
        )
    )
    
    patterns: List[str] = Field(
        ...,
        description=(
            "Up to 20 most important patterns. "
            "Can update existing patterns, add new ones, or remove obsolete ones. "
            "Each pattern should be actionable guidance."
        )
    )
