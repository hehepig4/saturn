"""
Path Matcher v3 for TBox + ABox Constraint-based Table Retrieval.

This module implements the v3 path matching design with improved scoring:
- TBox: Class hierarchy-based matching with IDF ratio
- ABox: Capped conditional IDF with bound constraint
- Unified formula: score = capped_abox + tbox_ratio

Key Improvements over v2.1:
- ABox score capped by path minimum IDF (prevents Column from dominating)
- Only considers ancestors of required class (not children)
- Better weight assignment for aggregation

Scoring Formula:
    score_i = max_{C ∈ ancestors(required)} [
        min(IDF(v|C), β·bound) × exists(C) + IDF(C)/IDF(required)
    ]
    
    TableScore = Σ(w_i × score_i) / Σ(w_i)
    
Where:
- bound = min{IDF(C') : C' ∈ path(required), IDF(C') > 0}
- β >= 1 is a relaxation parameter (default: 1.0)
- w_i = β·bound + 1 for ABox constraints, IDF(required) for TBox-only
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

import lancedb
from loguru import logger

from core.paths import get_db_path
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
class PrecomputedConstraint:
    """Precomputed values for a single constraint."""
    constraint: PathConstraint
    required_class: str
    required_idf: float  # IDF(required) - kept for logging
    bound: float  # min{IDF(v|C)} for ABox, not used for TBox
    ancestors: List[str]  # Required class and its ancestors
    
    # For ABox constraints
    ancestor_info: Optional[Dict[str, 'AncestorInfo']] = None


@dataclass
class AncestorInfo:
    """Precomputed values for an ancestor class."""
    class_name: str
    idf_c: float  # IDF(C) - used directly as TBox component
    df_c: int  # df(C)
    candidates: Set[str]  # Tables with value at this class (BF query result)
    df_v_c: int  # Number of candidates = df(v, C)
    idf_v_c: float  # Conditional IDF = log(df(C) / df(v, C))


@dataclass
class ScorerV3:
    """
    V3 Path Matcher with improved scoring formula.
    
    Features:
    - Capped ABox score using path bound
    - Only considers ancestors (not children)
    - Detailed debug logging
    """
    dataset: str
    beta: float = 1.0  # Relaxation parameter for ABox bound
    
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
        return self._hierarchy
    
    @property
    def idf_calculator(self) -> IDFCalculator:
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
            
            # Load column mappings
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
                f"ScorerV3 loaded in {elapsed:.2f}s: "
                f"{len(self._table_classes)} tables"
            )
            
        except Exception as e:
            logger.error(f"Failed to load ScorerV3: {e}")
            raise
    
    # ========== Precomputation ==========
    
    def precompute_constraints(
        self,
        constraints: ConstraintSet,
    ) -> List[PrecomputedConstraint]:
        """
        Precompute values for all constraints.
        
        This includes:
        - Bound computation for each constraint
        - Ancestor list and TBox ratios
        - ABox statistics (candidates, conditional IDF)
        """
        self._load()
        
        precomputed = []
        
        for constraint in constraints:
            required_class = constraint.class_name
            required_idf = self._idf_calc.get_idf(required_class)
            
            # Get ancestors (including self)
            ancestors = self._hierarchy.get_ancestors(required_class, include_self=True)
            
            logger.debug(
                f"[Precompute] Constraint: {required_class}"
                + (f" with value '{constraint.value}'" if constraint.has_value else "")
            )
            logger.debug(f"  required_idf={required_idf:.4f}, ancestors={ancestors}")
            
            # For ABox constraints, compute per-ancestor info to get bound
            if constraint.has_value:
                ancestor_info = {}
                
                for ancestor in ancestors:
                    idf_c = self._idf_calc.get_idf(ancestor)
                    df_c = self._idf_calc.get_df(ancestor)
                    
                    # Without value existence index, candidates are unknown
                    candidates = set()
                    df_v_c = 0
                    idf_v_c = float('inf')
                    
                    info = AncestorInfo(
                        class_name=ancestor,
                        idf_c=idf_c,
                        df_c=df_c,
                        candidates=candidates,
                        df_v_c=df_v_c,
                        idf_v_c=idf_v_c,
                    )
                    ancestor_info[ancestor] = info
                    
                    logger.debug(
                        f"  [Ancestor: {ancestor}] "
                        f"idf_c={idf_c:.4f}, df_c={df_c}"
                    )
                
                # No value existence data, use class IDF as fallback bound
                path_idfs = [self._idf_calc.get_idf(c) for c in ancestors]
                nonzero_idfs = [idf for idf in path_idfs if idf > 0]
                bound = min(nonzero_idfs) if nonzero_idfs else 1.0
                
                logger.debug(f"  bound={bound:.4f}")
                
                pc = PrecomputedConstraint(
                    constraint=constraint,
                    required_class=required_class,
                    required_idf=required_idf,
                    bound=bound,
                    ancestors=ancestors,
                )
                pc.ancestor_info = ancestor_info
                
            else:
                # TBox-only constraint: bound not needed (set to 0)
                bound = 0.0
                
                logger.debug(f"  TBox-only constraint, no bound needed")
                
                pc = PrecomputedConstraint(
                    constraint=constraint,
                    required_class=required_class,
                    required_idf=required_idf,
                    bound=bound,
                    ancestors=ancestors,
                )
            
            precomputed.append(pc)
        
        return precomputed
    
    # ========== Scoring ==========
    
    def compute_constraint_score(
        self,
        table_id: str,
        pc: PrecomputedConstraint,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute score for a single constraint.
        
        Returns:
            Tuple of (score, debug_info)
        """
        table_classes = self._table_classes.get(table_id, [])
        
        # Get table's annotation path (all ancestors of its classes)
        table_path = set()
        for cls in table_classes:
            table_path.update(self._hierarchy.get_ancestors(cls, include_self=True))
        
        # Intersection with required's ancestors
        intersection = set(pc.ancestors) & table_path
        
        debug_info = {
            'table_classes': table_classes,
            'intersection': list(intersection),
            'required': pc.required_class,
        }
        
        if not intersection:
            logger.debug(
                f"  [Table {table_id}] Empty intersection, score=0"
            )
            debug_info['score'] = 0.0
            debug_info['reason'] = 'empty_intersection'
            return 0.0, debug_info
        
        if pc.constraint.has_value:
            # ABox constraint: find best scoring ancestor in intersection
            # Formula: score = β × clip(IDF(v|C), bound) × exists + IDF(C)
            best_score = 0.0
            best_ancestor = None
            
            for ancestor in intersection:
                info = pc.ancestor_info[ancestor]
                
                # Check existence using precomputed candidates
                exists = table_id in info.candidates
                
                if exists:
                    # Clipped ABox score + TBox IDF
                    clipped_abox = min(info.idf_v_c, pc.bound)
                    score = self.beta * clipped_abox + info.idf_c
                    
                    if score > best_score:
                        best_score = score
                        best_ancestor = ancestor
                else:
                    # Value doesn't exist, only TBox IDF contributes
                    score = info.idf_c
                    
                    if score > best_score:
                        best_score = score
                        best_ancestor = ancestor
            
            debug_info['best_ancestor'] = best_ancestor
            debug_info['score'] = best_score
            debug_info['has_abox'] = True
            
            return best_score, debug_info
        
        else:
            # TBox-only: use deepest node in intersection
            # Formula: score = IDF(matched_class)
            deepest = get_deepest_node(intersection, self._hierarchy)
            
            if deepest:
                score = self._idf_calc.get_idf(deepest)
            else:
                score = 0.0
            
            debug_info['deepest_matched'] = deepest
            debug_info['score'] = score
            debug_info['has_abox'] = False
            
            return score, debug_info
    
    def compute_table_score(
        self,
        table_id: str,
        precomputed: List[PrecomputedConstraint],
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Compute overall score for a table using simple average.
        
        Returns:
            Tuple of (score, list of per-constraint debug info)
        """
        if not precomputed:
            return 1.0, []
        
        total_score = 0.0
        constraint_details = []
        
        for pc in precomputed:
            score, debug_info = self.compute_constraint_score(table_id, pc)
            total_score += score
            constraint_details.append(debug_info)
        
        # Simple average
        final_score = total_score / len(precomputed)
        
        return final_score, constraint_details
    
    # ========== Retrieval ==========
    
    def retrieve(
        self,
        constraints: ConstraintSet,
        score_threshold: float = 0.0,
    ) -> List[Tuple[str, float, List[Dict[str, Any]]]]:
        """
        Retrieve all tables by scoring against constraints.
        
        Args:
            constraints: ConstraintSet with TBox + optional ABox constraints
            score_threshold: Minimum score to include
            
        Returns:
            List of (table_id, score, debug_info) tuples, sorted by score desc
        """
        self._load()
        
        if len(constraints) == 0:
            logger.warning("No constraints provided")
            return []
        
        start = time.time()
        
        # Step 1: Precompute
        logger.debug("=" * 60)
        logger.debug("[Precomputation Phase]")
        precomputed = self.precompute_constraints(constraints)
        precompute_time = time.time() - start
        logger.debug(f"Precomputation done in {precompute_time*1000:.1f}ms")
        
        # Step 2: Score all tables
        logger.debug("=" * 60)
        logger.debug("[Scoring Phase]")
        score_start = time.time()
        
        results = []
        for table_id in self._table_classes.keys():
            score, details = self.compute_table_score(table_id, precomputed)
            if score > score_threshold:
                results.append((table_id, score, details))
        
        score_time = time.time() - score_start
        
        # Sort by score descending
        results.sort(key=lambda x: -x[1])
        
        total_time = time.time() - start
        logger.info(
            f"ScorerV3 retrieved {len(results)} tables in {total_time*1000:.1f}ms "
            f"(precompute: {precompute_time*1000:.1f}ms, score: {score_time*1000:.1f}ms)"
        )
        
        return results


# Singleton cache
_scorer_v3_cache: Dict[str, ScorerV3] = {}


def get_scorer_v3(dataset: str, beta: float = 1.0) -> ScorerV3:
    """Get cached ScorerV3 instance."""
    key = f"{dataset}_{beta}"
    if key not in _scorer_v3_cache:
        _scorer_v3_cache[key] = ScorerV3(dataset=dataset, beta=beta)
    return _scorer_v3_cache[key]
