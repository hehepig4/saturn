"""
Retrieval Nodes Package

Dual-modal retrieval pipeline nodes:
1. extract_constraints - LLM-based constraint extraction  
2. semantic_search - Vector + BM25 hybrid search
3. matcher - PathMatcher with TBox + ABox constraints (v2.1)

Note: fusion and other deprecated nodes have been moved to deprecated/ folder.
The new architecture uses RRF fusion directly in graph.py.
"""

from workflows.retrieval.nodes.extract_constraints import (
    extract_constraints_node,
    convert_to_path_constraints,
)
from workflows.retrieval.nodes.semantic_search import semantic_search_node

# v2.1 Path Matcher
from workflows.retrieval.matcher import (
    PathMatcher,
    get_path_matcher,
    PathConstraint,
    ConstraintSet,
    TBoxConstraint,
    ABoxConstraint,
    MatchResult,
)

__all__ = [
    'extract_constraints_node',
    'convert_to_path_constraints',
    'semantic_search_node',
    # v2.1 Path Matcher
    'PathMatcher',
    'get_path_matcher',
    'PathConstraint',
    'ConstraintSet',
    'TBoxConstraint',
    'ABoxConstraint',
    'MatchResult',
]
