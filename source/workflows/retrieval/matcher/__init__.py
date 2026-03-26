"""
Path Matcher Module for TBox + ABox Constraint-based Table Retrieval.

This module provides the v2.1 implementation of path-based table matching
with integrated ABox value existence checking.

Components:
- PathMatcher: Core matching engine
- PathConstraint: Combined TBox + ABox constraint
- ConstraintSet: Collection of constraints
- MatchResult: Matching result with details

Usage:
    from workflows.retrieval.matcher import get_path_matcher, PathConstraint, ConstraintSet
    
    # Get matcher
    matcher = get_path_matcher('fetaqa')
    
    # Single constraint retrieval
    results = matcher.retrieve_with_value('YearColumn', '2020', top_k=10)
    
    # Multiple constraints
    constraints = ConstraintSet(constraints=[
        PathConstraint.from_class_value('YearColumn', '2020'),
        PathConstraint.from_class_value('PersonColumn', 'John'),
    ])
    results = matcher.retrieve(constraints, top_k=10)
"""

from workflows.retrieval.matcher.constraints import (
    TBoxConstraint,
    ABoxConstraint,
    PathConstraint,
    ConstraintSet,
    MatchResult,
    MatchType,
)
from workflows.retrieval.matcher.path_matcher import (
    PathMatcher,
    get_path_matcher,
)
from workflows.retrieval.matcher.utils import (
    ClassHierarchy,
    IDFCalculator,
    get_hierarchy,
    get_idf_calculator,
    normalize_value,
)
from workflows.retrieval.matcher.scorer_v3 import (
    ScorerV3,
    get_scorer_v3,
    PrecomputedConstraint,
    AncestorInfo,
)

__all__ = [
    # Core
    'PathMatcher',
    'get_path_matcher',
    # V3 Scorer
    'ScorerV3',
    'get_scorer_v3',
    'PrecomputedConstraint',
    'AncestorInfo',
    # Constraints
    'TBoxConstraint',
    'ABoxConstraint', 
    'PathConstraint',
    'ConstraintSet',
    'MatchResult',
    'MatchType',
    # Utils
    'ClassHierarchy',
    'IDFCalculator',
    'get_hierarchy',
    'get_idf_calculator',
    'normalize_value',
]
