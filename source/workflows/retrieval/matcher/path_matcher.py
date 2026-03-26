"""
Path Matcher for TBox + ABox Constraint-based Table Retrieval.

This module implements the v2.1 path matching design:
- TBox: Class hierarchy-based matching with IDF weighting
- ABox: Value existence constraint using Table-Class CBF

Scoring Formula:
    Score = max(IDF(node) × exists) / IDF(required_leaf)
    
Where:
- node ∈ intersection(required_path, annotation_path)
- exists = 1 if value in Table-Class CBF, else 0
- required_leaf = deepest class in required path

Key Design Principle (v2.1):
- NO candidate filtering - all tables are scored
- Since all classes inherit from Column, every path intersection is non-empty
- Score differences come from intersection depth (deeper = higher IDF = higher score)
- Inverted index is used for fast class-to-class score computation, not filtering
"""

from __future__ import annotations

import time
from collections import defaultdict

from core.paths import get_db_path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import lancedb
from loguru import logger

from core.datatypes.el_datatypes import TargetType
from workflows.retrieval.matcher.constraints import (
    PathConstraint,
    ConstraintSet,
    MatchResult,
    TBoxConstraint,
    ABoxConstraint,
)
from workflows.retrieval.matcher.utils import (
    ClassHierarchy,
    IDFCalculator,
    get_hierarchy,
    get_idf_calculator,
    normalize_value,
    get_deepest_node,
    get_target_type_for_class,
)


@dataclass
class PathMatcher:
    """
    Path-based table matcher using TBox + ABox constraints.
    
    Implements the v2.1 design with:
    - Full table scan (no candidate filtering)
    - Table-Class CBF: (table_id, class_name) → CBF for value existence
    - IDF computation: TBox classes only, with DF propagation
    
    Usage:
        matcher = PathMatcher(dataset='fetaqa')
        
        # Create constraints
        constraints = ConstraintSet(constraints=[
            PathConstraint(
                tbox=TBoxConstraint(class_name='YearColumn'),
                abox=ABoxConstraint(value='2020'),
            ),
        ])
        
        # Retrieve matching tables
        results = matcher.retrieve(constraints)
    """
    dataset: str
    
    # Core components
    _hierarchy: ClassHierarchy = field(default=None, repr=False)
    _idf_calc: IDFCalculator = field(default=None, repr=False)
    
    # Indexes
    _table_classes: Dict[str, List[str]] = field(default_factory=dict, repr=False)
    
    _loaded: bool = field(default=False, repr=False)
    
    def __post_init__(self):
        """Initialize hierarchy and IDF calculator."""
        self._hierarchy = get_hierarchy(self.dataset)
        self._idf_calc = get_idf_calculator(self.dataset)
    
    @property
    def hierarchy(self) -> ClassHierarchy:
        """Get class hierarchy."""
        return self._hierarchy
    
    @property
    def idf_calculator(self) -> IDFCalculator:
        """Get IDF calculator."""
        return self._idf_calc
    
    @property
    def all_tables(self) -> List[str]:
        """Get all table IDs."""
        self._load()
        return list(self._table_classes.keys())
    
    def _load(self) -> None:
        """Load table-class mappings."""
        if self._loaded:
            return
        
        start = time.time()
        
        try:
            db = lancedb.connect(str(get_db_path()))
            
            # Load column mappings to build table → classes mapping
            cm_df = db.open_table(f'{self.dataset}_column_mappings').to_pandas()
            
            tbl_cls: Dict[str, Set[str]] = defaultdict(set)
            
            for _, row in cm_df.iterrows():
                table_id = row['source_table']
                pclass = str(row.get('primitive_class', '')).replace('upo:', '').replace('ont:', '')
                if pclass:
                    tbl_cls[table_id].add(pclass)
            
            self._table_classes = {k: list(v) for k, v in tbl_cls.items()}
            
            self._loaded = True
            elapsed = time.time() - start
            logger.info(
                f"PathMatcher loaded in {elapsed:.2f}s: "
                f"{len(self._table_classes)} tables"
            )
            
        except Exception as e:
            logger.error(f"Failed to load PathMatcher: {e}")
            raise
    

    def compute_score(
        self,
        table_id: str,
        constraint: PathConstraint,
    ) -> MatchResult:
        """
        Compute match score for a single constraint against a table.
        
        Formula (v2.1):
            Score = max(IDF(node) × exists) / IDF(required_leaf)
            
        Where node ∈ intersection(required_path, annotation_path)
        
        Args:
            table_id: Table identifier
            constraint: PathConstraint with TBox and optional ABox
            
        Returns:
            MatchResult with score and details
        """
        self._load()
        
        # Get paths
        required_class = constraint.class_name
        required_path = self._hierarchy.get_path(required_class)
        required_leaf_idf = self._idf_calc.get_idf(required_class)
        
        # Get table's annotation classes
        table_classes = self._table_classes.get(table_id, [])
        if not table_classes:
            return MatchResult(
                table_id=table_id,
                score=0.0,
                required_leaf_idf=required_leaf_idf,
            )
        
        # Compute annotation path (union of all class paths)
        annotation_nodes = set()
        for cls in table_classes:
            annotation_nodes.update(self._hierarchy.get_path(cls))
        
        # Compute path intersection
        intersection = set(required_path) & annotation_nodes
        if not intersection:
            return MatchResult(
                table_id=table_id,
                score=0.0,
                required_leaf_idf=required_leaf_idf,
            )
        
        # Find best matching node
        best_score = 0.0
        matched_class = None
        value_exists = False
        matched_node_idf = 0.0
        
        if constraint.has_value:
            # ABox constraint: check value existence in each intersection node
            normalized_value = constraint.value
            
            # IMPORTANT: Use the required_class's TargetType for BF queries
            # Values are stored with the original class's format and propagated
            # to ancestors WITHOUT re-normalization. So we must query with the
            # same format that was used during storage.
            required_target_type = get_target_type_for_class(required_class)
            
            for node in intersection:
                node_idf = self._idf_calc.get_idf(node)
                
                # Check value existence in Table-Class CBF
                # Use required_class's target_type, not node's
                exists = True  # Conservative: always assume exists
                
                if exists:
                    # Compute score: node_idf / required_leaf_idf
                    if required_leaf_idf > 0:
                        score = node_idf / required_leaf_idf
                    else:
                        score = 1.0
                    
                    if score > best_score:
                        best_score = score
                        matched_class = node
                        matched_node_idf = node_idf
                        value_exists = True
        else:
            # TBox-only: use deepest node in intersection
            # This is the "partial match" case where we score by how much of the path matched
            deepest_node = get_deepest_node(intersection, self._hierarchy)
            if deepest_node:
                node_idf = self._idf_calc.get_idf(deepest_node)
                if required_leaf_idf > 0:
                    best_score = node_idf / required_leaf_idf
                else:
                    best_score = 1.0
                matched_class = deepest_node
                matched_node_idf = node_idf
        
        # Clamp score to [0, 1]
        best_score = min(max(best_score, 0.0), 1.0)
        
        return MatchResult(
            table_id=table_id,
            score=best_score,
            matched_class=matched_class,
            matched_node_idf=matched_node_idf,
            required_leaf_idf=required_leaf_idf,
            value_exists=value_exists,
        )
    
    def compute_table_score(
        self,
        table_id: str,
        constraints: ConstraintSet,
        abox_weight_boost: float = 1.5,
    ) -> Tuple[float, List[MatchResult]]:
        """
        Compute overall score for a table against all constraints.
        
        Aggregation: IDF-weighted average of per-constraint scores
            Score = Σ weight(c) × score(c) / Σ weight(c)
            
        Where weight(c) = IDF(c) × abox_weight_boost if constraint has ABox
                        = IDF(c) otherwise
        
        Args:
            table_id: Table identifier
            constraints: ConstraintSet with multiple constraints
            abox_weight_boost: Weight multiplier for constraints with ABox (default: 1.5)
            
        Returns:
            Tuple of (overall_score, per_constraint_results)
        """
        if len(constraints) == 0:
            return 1.0, []
        
        results = []
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for constraint in constraints:
            result = self.compute_score(table_id, constraint)
            results.append(result)
            
            # Base weight is IDF
            base_idf = result.required_leaf_idf
            
            # Apply ABox boost if constraint has ABox value
            if constraint.has_value:
                weight = base_idf * abox_weight_boost
            else:
                weight = base_idf
            
            total_weighted_score += weight * result.score
            total_weight += weight
        
        if total_weight > 0:
            overall_score = total_weighted_score / total_weight
        else:
            # Fallback: simple average
            overall_score = sum(r.score for r in results) / len(results)
        
        return overall_score, results
    
    def retrieve(
        self,
        constraints: ConstraintSet,
        score_threshold: float = 0.0,
    ) -> List[Tuple[str, float, List[MatchResult]]]:
        """
        Retrieve all tables by scoring against constraints.
        
        Since all classes inherit from Column, every path intersection is non-empty.
        Score differences come from intersection depth.
        
        Args:
            constraints: ConstraintSet with TBox + optional ABox constraints
            score_threshold: Minimum score to include (default: 0.0, include all)
            
        Returns:
            List of (table_id, score, per_constraint_results) tuples, sorted by score desc
        """
        self._load()
        
        if len(constraints) == 0:
            logger.warning("No constraints provided for retrieval")
            return []
        
        start = time.time()
        total_tables = len(self._table_classes)
        
        tables_to_score = set(self._table_classes.keys())
        
        # Score candidate tables
        scored_results = []
        for table_id in tables_to_score:
            score, details = self.compute_table_score(table_id, constraints)
            if score > score_threshold:
                scored_results.append((table_id, score, details))
        
        # Sort by score descending
        scored_results.sort(key=lambda x: -x[1])
        
        elapsed = time.time() - start
        logger.debug(
            f"Retrieved {len(scored_results)} from {total_tables} tables "
            f"(scanned {len(tables_to_score)}, {elapsed*1000:.1f}ms total)"
        )
        
        return scored_results


# Singleton cache
_matcher_cache: Dict[str, PathMatcher] = {}


def get_path_matcher(dataset: str) -> PathMatcher:
    """Get cached PathMatcher instance."""
    if dataset not in _matcher_cache:
        _matcher_cache[dataset] = PathMatcher(dataset=dataset)
    return _matcher_cache[dataset]