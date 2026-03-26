"""
Utility Functions for Path Matching.

This module provides helper functions for:
- Class hierarchy operations (path, ancestors, descendants)
- IDF computation
- Value normalization for BF queries

Design (v2.1):
- IDF computed only for TBox classes, not ABox values
- IDF formula: log((N + 1) / (DF(C) + 1))
- DF propagation: parent DF >= child DF (monotonicity guaranteed)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from core.datatypes.el_datatypes import TargetType
from core.paths import get_db_path

import lancedb
from loguru import logger


@dataclass
class ClassHierarchy:
    """
    Class hierarchy manager for primitive classes.
    
    Provides:
    - Parent/child relationships
    - Path computation (root to leaf)
    - Ancestor/descendant lookup
    - Depth calculation
    
    Attributes:
        dataset: Dataset name
        _parent_map: child -> parent mapping
        _children_map: parent -> children mapping
        _all_classes: set of all class names
    """
    dataset: str
    _parent_map: Dict[str, str] = field(default_factory=dict, repr=False)
    _children_map: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set), repr=False)
    _all_classes: Set[str] = field(default_factory=set, repr=False)
    _loaded: bool = field(default=False, repr=False)
    
    def _load(self) -> None:
        """Load class hierarchy from LanceDB."""
        if self._loaded:
            return
        
        try:
            db = lancedb.connect(str(get_db_path()))
            
            # Get latest ontology for dataset
            meta_df = db.open_table('ontology_metadata').to_pandas()
            dataset_meta = meta_df[
                (meta_df['dataset_name'] == self.dataset) & 
                (meta_df['ontology_type'] == 'federated_primitive_tbox')
            ].sort_values('created_at', ascending=False)
            
            if len(dataset_meta) == 0:
                logger.warning(f"No ontology found for dataset {self.dataset}")
                self._loaded = True
                return
            
            ontology_id = dataset_meta.iloc[0]['ontology_id']
            
            # Load classes
            class_df = db.open_table('ontology_classes').to_pandas()
            classes = class_df[class_df['ontology_id'] == ontology_id]
            
            for _, row in classes.iterrows():
                cname = row['class_name']
                self._all_classes.add(cname)
                
                parents = row['parent_classes']
                if parents is not None:
                    plist = list(parents) if hasattr(parents, '__iter__') else [parents]
                    plist = [p for p in plist if p and p != 'null']
                    if plist:
                        # Take first parent for path computation
                        self._parent_map[cname] = plist[0]
                        self._children_map[plist[0]].add(cname)
            
            self._loaded = True
            logger.debug(f"ClassHierarchy loaded: {len(self._all_classes)} classes")
            
        except Exception as e:
            logger.error(f"Failed to load class hierarchy: {e}")
            self._loaded = True
    
    def get_path(self, class_name: str) -> List[str]:
        """
        Get path from root (Column) to the given class.
        
        Example: get_path("YearColumn") -> ["Column", "TemporalColumn", "YearColumn"]
        """
        self._load()
        
        path = [class_name]
        current = class_name
        while current in self._parent_map:
            current = self._parent_map[current]
            path.append(current)
        return list(reversed(path))
    
    def get_ancestors(self, class_name: str, include_self: bool = True) -> Set[str]:
        """Get all ancestors of a class."""
        self._load()
        
        result = {class_name} if include_self else set()
        current = class_name
        while current in self._parent_map:
            current = self._parent_map[current]
            result.add(current)
        return result
    
    def get_descendants(self, class_name: str, include_self: bool = True) -> Set[str]:
        """Get all descendants of a class."""
        self._load()
        
        result = {class_name} if include_self else set()
        
        def collect(cls: str):
            for child in self._children_map.get(cls, set()):
                if child not in result:
                    result.add(child)
                    collect(child)
        
        collect(class_name)
        return result
    
    def get_depth(self, class_name: str) -> int:
        """Get depth of a class (root = 1)."""
        return len(self.get_path(class_name))
    
    def get_parent(self, class_name: str) -> Optional[str]:
        """Get direct parent of a class."""
        self._load()
        return self._parent_map.get(class_name)
    
    def get_children(self, class_name: str) -> Set[str]:
        """Get direct children of a class."""
        self._load()
        return self._children_map.get(class_name, set())
    
    @property
    def all_classes(self) -> Set[str]:
        """Get all class names."""
        self._load()
        return self._all_classes.copy()


@dataclass
class IDFCalculator:
    """
    IDF calculator for TBox classes.
    
    Uses propagated DF to ensure monotonicity:
    - DF(parent) >= DF(child)
    - IDF(parent) <= IDF(child)
    
    Formula: IDF(C) = log((N + 1) / (DF(C) + 1))
    
    Attributes:
        dataset: Dataset name
        hierarchy: Class hierarchy (for path computation)
    """
    dataset: str
    hierarchy: ClassHierarchy = field(default=None, repr=False)
    _df: Dict[str, int] = field(default_factory=dict, repr=False)
    _idf: Dict[str, float] = field(default_factory=dict, repr=False)
    _total_tables: int = field(default=0, repr=False)
    _loaded: bool = field(default=False, repr=False)
    
    def __post_init__(self):
        if self.hierarchy is None:
            self.hierarchy = ClassHierarchy(dataset=self.dataset)
    
    def _load(self) -> None:
        """Load DF values and compute IDF."""
        if self._loaded:
            return
        
        try:
            db = lancedb.connect(str(get_db_path()))
            
            # Load column mappings
            cm_df = db.open_table(f'{self.dataset}_column_mappings').to_pandas()
            self._total_tables = cm_df['source_table'].nunique()
            
            # Compute propagated DF
            # For each table, collect its classes and propagate to ancestors
            df_tables: Dict[str, Set[str]] = defaultdict(set)
            
            for _, row in cm_df.iterrows():
                table_id = row['source_table']
                pclass = str(row.get('primitive_class', '')).replace('upo:', '').replace('ont:', '')
                if pclass:
                    # Propagate to all ancestors
                    for ancestor in self.hierarchy.get_ancestors(pclass, include_self=True):
                        df_tables[ancestor].add(table_id)
            
            self._df = {c: len(tables) for c, tables in df_tables.items()}
            
            # Compute IDF
            N = self._total_tables
            for class_name, df in self._df.items():
                self._idf[class_name] = math.log((N + 1) / (df + 1))
            
            self._loaded = True
            logger.debug(f"IDFCalculator loaded: {len(self._idf)} classes, {N} tables")
            
        except Exception as e:
            logger.error(f"Failed to load IDF data: {e}")
            self._loaded = True
    
    def get_idf(self, class_name: str) -> float:
        """Get IDF value for a class."""
        self._load()
        return self._idf.get(class_name, 0.0)
    
    def get_df(self, class_name: str) -> int:
        """Get document frequency for a class."""
        self._load()
        return self._df.get(class_name, 0)
    
    @property
    def total_tables(self) -> int:
        """Get total number of tables."""
        self._load()
        return self._total_tables


def normalize_value(value: str) -> str:
    """
    Normalize a value for BF query.
    
    Standard form:
    - Lowercase
    - Strip whitespace
    - Single spaces between words
    """
    if not value:
        return ""
    return ' '.join(str(value).lower().split())


def compute_path_intersection(
    required_path: List[str],
    annotation_path: List[str],
) -> Set[str]:
    """
    Compute intersection of two paths.
    
    Args:
        required_path: Required TBox path (from constraint)
        annotation_path: Annotation path (from table)
        
    Returns:
        Set of common class names
    """
    return set(required_path) & set(annotation_path)


def get_deepest_node(
    nodes: Set[str],
    hierarchy: ClassHierarchy,
) -> Optional[str]:
    """
    Get the deepest node (highest depth) from a set of nodes.
    
    Args:
        nodes: Set of class names
        hierarchy: Class hierarchy for depth lookup
        
    Returns:
        The deepest node, or None if empty
    """
    if not nodes:
        return None
    
    return max(nodes, key=lambda n: hierarchy.get_depth(n))


# Singleton instances for caching
_hierarchy_cache: Dict[str, ClassHierarchy] = {}
_idf_cache: Dict[str, IDFCalculator] = {}


def get_hierarchy(dataset: str) -> ClassHierarchy:
    """Get cached ClassHierarchy instance."""
    if dataset not in _hierarchy_cache:
        _hierarchy_cache[dataset] = ClassHierarchy(dataset=dataset)
    return _hierarchy_cache[dataset]


def get_idf_calculator(dataset: str) -> IDFCalculator:
    """Get cached IDFCalculator instance."""
    if dataset not in _idf_cache:
        _idf_cache[dataset] = IDFCalculator(
            dataset=dataset,
            hierarchy=get_hierarchy(dataset),
        )
    return _idf_cache[dataset]


def get_target_type_for_class(class_name: str) -> TargetType:
    """
    Get TargetType from DataProperty's range definition in ontology.
    
    Maps XSD types from ontology to TargetType:
    - xsd:string → STR
    - xsd:integer → INT  
    - xsd:decimal, xsd:float, xsd:double → FLOAT
    - xsd:dateTime, xsd:date, xsd:time → DATETIME
    
    Args:
        class_name: Primitive class name (e.g., 'YearColumn')
        
    Returns:
        TargetType enum value
    """
    from core.datatypes.el_datatypes import TargetType
    
    # Get range from ontology DataProperty
    range_type = _get_property_range_for_class(class_name)
    
    if range_type is None:
        return TargetType.STR  # Default fallback
    
    # Map XSD types to TargetType
    range_lower = range_type.lower()
    
    if 'integer' in range_lower or 'int' in range_lower:
        return TargetType.INT
    elif 'decimal' in range_lower or 'float' in range_lower or 'double' in range_lower:
        return TargetType.FLOAT
    elif 'datetime' in range_lower or 'date' in range_lower or 'time' in range_lower:
        return TargetType.DATETIME
    else:
        return TargetType.STR


# Cache for property ranges
_property_range_cache: Dict[str, Optional[str]] = {}


def _get_property_range_for_class(class_name: str) -> Optional[str]:
    """
    Get the DataProperty range type for a primitive class from ontology.
    
    Looks up the DataProperty that has this class as domain and returns its range.
    
    Args:
        class_name: Primitive class name
        
    Returns:
        XSD type string (e.g., 'xsd:integer') or None if not found
    """
    global _property_range_cache
    
    if class_name in _property_range_cache:
        return _property_range_cache[class_name]
    
    try:
        import lancedb
        import numpy as np
        db = lancedb.connect(str(get_db_path()))
        
        if 'ontology_properties' not in db.table_names(limit=1000000):
            _property_range_cache[class_name] = None
            return None
        
        df = db.open_table('ontology_properties').to_pandas()
        
        # Find DataProperty with this class as domain
        for _, row in df.iterrows():
            if row.get('property_type') != 'data':
                continue
            
            domain = row.get('domain', [])
            # Handle numpy array
            if isinstance(domain, np.ndarray):
                domain = domain.tolist()
            elif isinstance(domain, str):
                domain = [domain]
            
            # Check if class_name is in domain
            if class_name in domain or f'upo:{class_name}' in domain:
                range_val = row.get('range', [])
                # Handle numpy array
                if isinstance(range_val, np.ndarray):
                    range_val = range_val.tolist()
                
                if isinstance(range_val, list) and range_val:
                    range_type = range_val[0]
                elif isinstance(range_val, str):
                    range_type = range_val
                else:
                    range_type = None
                
                _property_range_cache[class_name] = range_type
                return range_type
        
        # Not found - cache None
        _property_range_cache[class_name] = None
        return None
        
    except Exception as e:
        logger.debug(f"Failed to get property range for {class_name}: {e}")
        _property_range_cache[class_name] = None
        return None