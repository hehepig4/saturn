"""
Constraint Data Types for Path Matching.

This module defines the constraint types used in path-based table retrieval.
Constraints represent requirements extracted from user queries.

Design (v2.1):
- TBox constraints: Required primitive classes (column types)
- ABox constraints: Entity values that must exist in matched tables
- Combined into PathConstraint for unified processing
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class MatchType(Enum):
    """Match type for ABox constraints."""
    EXACT = "exact"       # Exact string match (for numbers, IDs)
    CONTAINS = "contains" # Substring match (for text)


@dataclass
class TBoxConstraint:
    """
    TBox constraint representing a required primitive class.
    
    Attributes:
        class_name: Primitive class name (e.g., 'YearColumn')
        required: Whether this constraint is mandatory
        path: Full path from root to this class (computed lazily)
    """
    class_name: str
    required: bool = True
    path: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash(self.class_name)
    
    def __eq__(self, other):
        if isinstance(other, TBoxConstraint):
            return self.class_name == other.class_name
        return False


@dataclass
class ABoxConstraint:
    """
    ABox constraint representing a required value.
    
    In v2.1 design, ABox only serves as existence constraint.
    No IDF is computed for values.
    
    Attributes:
        value: The value to match
        class_name: Associated primitive class (None if unspecified)
        match_type: How to match the value
        normalized_value: Normalized string form for BF query
    """
    value: str
    class_name: Optional[str] = None
    match_type: MatchType = MatchType.EXACT
    normalized_value: Optional[str] = None
    
    def __post_init__(self):
        # Default normalized value is the original value as string
        if self.normalized_value is None:
            self.normalized_value = str(self.value).strip().lower()


@dataclass
class PathConstraint:
    """
    Combined TBox + ABox constraint for path matching.
    
    Represents a single constraint in the form:
    TBox Path + Optional ABox Value
    
    Example:
        Column → TemporalColumn → YearColumn → [2024]
        
    Attributes:
        tbox: TBox constraint (required primitive class)
        abox: Optional ABox constraint (value to match)
    """
    tbox: TBoxConstraint
    abox: Optional[ABoxConstraint] = None
    
    @property
    def class_name(self) -> str:
        """Shortcut to TBox class name."""
        return self.tbox.class_name
    
    @property
    def has_value(self) -> bool:
        """Whether this constraint has an ABox value."""
        return self.abox is not None
    
    @property
    def value(self) -> Optional[str]:
        """Shortcut to ABox normalized value."""
        return self.abox.normalized_value if self.abox else None
    
    def __repr__(self):
        if self.abox:
            return f"PathConstraint({self.class_name} + [{self.abox.value}])"
        return f"PathConstraint({self.class_name})"


@dataclass
class ConstraintSet:
    """
    Set of constraints extracted from a query.
    
    Contains multiple PathConstraints to be matched against tables.
    """
    constraints: List[PathConstraint] = field(default_factory=list)
    same_row_required: bool = True  # Whether all constraints must match in same row
    
    def __len__(self):
        return len(self.constraints)
    
    def __iter__(self):
        return iter(self.constraints)
    
    @property
    def tbox_classes(self) -> List[str]:
        """Get all TBox class names."""
        return [c.class_name for c in self.constraints]
    
    @property
    def has_abox_constraints(self) -> bool:
        """Whether any constraint has ABox value."""
        return any(c.has_value for c in self.constraints)


@dataclass 
class MatchResult:
    """
    Result of matching a constraint against a table.
    
    Attributes:
        table_id: Matched table identifier
        score: Match score in [0, 1]
        matched_class: The class where value was found (if ABox constraint)
        matched_node_idf: IDF of the matched node
        required_leaf_idf: IDF of the required leaf (denominator)
        value_exists: Whether the ABox value was found
    """
    table_id: str
    score: float
    matched_class: Optional[str] = None
    matched_node_idf: float = 0.0
    required_leaf_idf: float = 0.0
    value_exists: bool = False
    
    def __repr__(self):
        return f"MatchResult({self.table_id}, score={self.score:.3f})"
