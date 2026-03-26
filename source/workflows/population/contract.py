"""
Contracts for Column Summary Pipeline

Design Principles:
1. TransformContract is uniquely identified by (primitive_class, data_property, pattern)
2. A column classified to PrimitiveClass will try ALL DataProperties with matching domain
3. For each DataProperty, find best matching TransformContract by pattern match rate
4. If no match, LLM generates new TransformContract for that (class, property) pair
5. Statistics computation is PREDEFINED per range_type (no LLM generation)

Flow:
    Column → classify → PrimitiveClass
                            ↓
                    Query TBox: find all DataProperties where domain ⊇ PrimitiveClass
                            ↓
                    For each DataProperty:
                        - Find TransformContracts where (class, prop) matches
                        - Select best by pattern match rate
                        - Or generate new if none match
                        - Apply transform + predefined statistics
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import re2
import hashlib

import pandas as pd
from loguru import logger

# Import safe regex utilities for compilation and protected matching
from workflows.population.safe_regex import (
    safe_compile,
    safe_pandas_match,
)

# Import TargetType from unified datatype module
from core.datatypes.el_datatypes import TargetType, XSD_TO_TARGET_TYPE

# Import namespace builder for transform evaluation
from workflows.population.transform_namespace import build_namespace

# Cached namespace for transform evaluation
_TRANSFORM_NAMESPACE = None

def _get_transform_namespace():
    """Get the cached transform namespace."""
    global _TRANSFORM_NAMESPACE
    if _TRANSFORM_NAMESPACE is None:
        _TRANSFORM_NAMESPACE = build_namespace()
    return _TRANSFORM_NAMESPACE


@dataclass
class TransformContract:
    """
    Contract for parsing and transforming raw column values.
    
    Uniquely identified by: (primitive_class, data_property, pattern)
    
    This is what the LLM generates for a specific (class, property) pair.
    It describes:
    - The input pattern this transform handles
    - How to convert raw strings to target type
    
    Attributes:
        primitive_class: The primitive TBox class (e.g., "YearColumn")
        data_property: The DataProperty this transform populates (e.g., "hasYearValue")
        pattern: Regex pattern this transform handles (e.g., "\\d{4}$" for 4-digit years)
        target_type: The Python type to convert to (REQUIRED)
        transform_expr: Python expression using 'x' to transform (REQUIRED)
        sample_values: Sample values used when generating (optional)
        hit_count: Number of times this contract has been successfully used (for LRU)
    """
    primitive_class: str
    data_property: str
    pattern: str
    target_type: TargetType  # No default - must be explicit
    transform_expr: str  # No default - must be explicit
    sample_values: List[str] = field(default_factory=list)  # Only this has default
    hit_count: int = 0  # LRU counter: incremented when contract is selected for use
    
    def __post_init__(self):
        """Validate required fields are not empty."""
        if not self.primitive_class:
            raise ValueError("primitive_class is required")
        if not self.data_property:
            raise ValueError("data_property is required")
        if not self.pattern:
            raise ValueError("pattern is required")
        if not self.transform_expr:
            raise ValueError("transform_expr is required")
    
    @property
    def contract_key(self) -> str:
        """Unique key for this contract."""
        return f"{self.primitive_class}::{self.data_property}::{self.pattern}"
    
    @property
    def contract_id(self) -> str:
        """Short hash ID for storage."""
        return hashlib.md5(self.contract_key.encode()).hexdigest()[:12]
    
    @property
    def compiled_pattern(self) -> Optional[re2._Regexp]:
        """Cached compiled regex pattern (lazy initialization, uses RE2 safe_compile)."""
        if not hasattr(self, '_compiled_pattern'):
            self._compiled_pattern = safe_compile(self.pattern)
        return self._compiled_pattern
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "primitive_class": self.primitive_class,
            "data_property": self.data_property,
            "pattern": self.pattern,
            "target_type": self.target_type.value if isinstance(self.target_type, TargetType) else self.target_type,
            "transform_expr": self.transform_expr,
            "sample_values": self.sample_values,
            "hit_count": self.hit_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformContract":
        """
        Create from dictionary.
        
        Required fields: primitive_class, data_property, pattern, target_type, 
                         transform_expr
        
        Raises:
            KeyError: If required field is missing
            ValueError: If target_type is invalid
        """
        # Check required fields
        required_fields = ["primitive_class", "data_property", "pattern", "target_type", 
                          "transform_expr"]
        missing = [f for f in required_fields if f not in data or data[f] is None]
        if missing:
            raise KeyError(f"Missing required fields in TransformContract: {missing}")
        
        # Parse target_type
        target_type = data["target_type"]
        if isinstance(target_type, str):
            try:
                target_type = TargetType(target_type)
            except ValueError:
                raise ValueError(f"Invalid target_type: {target_type}. Must be one of {[t.value for t in TargetType]}")
        
        return cls(
            primitive_class=data["primitive_class"],
            data_property=data["data_property"],
            pattern=data["pattern"],
            target_type=target_type,
            transform_expr=data["transform_expr"],
            sample_values=data.get("sample_values", []),
            hit_count=data.get("hit_count", 0),  # Default 0 for backward compatibility
        )
    
    def success_rate(self, values: List[str], max_samples: Optional[int] = None) -> float:
        """
        Calculate what fraction of non-null values can be successfully processed.
        
        Uses pandas vectorized matching for performance (~15x faster than per-value).
        
        A value is considered successful if:
        1. It matches the regex pattern, AND
        2. The transform expression executes without error and returns non-None
        
        This is the unified score metric - combining pattern matching and
        transformation success into a single meaningful indicator.
        
        Args:
            values: Column values to test
            max_samples: Optional maximum number of samples to test for performance
            
        Returns:
            Score from 0.0 to 1.0 representing the fraction of processable values
            Returns 0.0 if pattern matching times out (potential ReDoS)
        """
        if not values:
            return 0.0
        
        # Filter out empty values (simple null check)
        non_null = [v for v in values if v and str(v).strip()]
        if not non_null:
            return 0.0
        
        # Sample if max_samples specified
        test_values = non_null
        if max_samples and len(non_null) > max_samples:
            # Deterministic sampling: evenly spaced
            step = len(non_null) // max_samples
            test_values = non_null[::step][:max_samples]
        
        # Use cached compiled pattern - check validity
        if self.compiled_pattern is None:
            return 0.0
        
        # Convert to pandas Series for vectorized matching
        s = pd.Series([str(v).strip() for v in test_values], dtype="string")
        
        # Vectorized pattern matching (direct call, no subprocess overhead)
        matched, has_error = safe_pandas_match(self.pattern, s)
        
        if has_error:
            # Error indicates invalid pattern - treat as failure
            return 0.0
        
        # Get matched values for transform testing
        matched_values = s[matched].tolist()
        if not matched_values:
            return 0.0
        
        # Get unified namespace for transform evaluation
        namespace = _get_transform_namespace()
        
        # Count successful transforms
        success_count = 0
        for v_str in matched_values:
            try:
                import warnings
                local_ns = {"x": v_str, **namespace}
                # Convert SyntaxWarning to error to detect invalid escape sequences
                with warnings.catch_warnings():
                    warnings.simplefilter("error", SyntaxWarning)
                    result = eval(self.transform_expr, local_ns)
                if result is not None:
                    success_count += 1
            except Exception:
                pass
        
        return success_count / len(test_values)


@dataclass
class DataPropertySpec:
    """
    Specification for a DataProperty from TBox.
    
    Used to pass DataProperty info to transform generation.
    """
    name: str                    # e.g., "hasYearValue"
    range_type: str              # e.g., "xsd:gYear"
    comment: Optional[str] = None  # LLM-generated description
    readout_template: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "range_type": self.range_type,
            "comment": self.comment,
            "readout_template": self.readout_template,
        }


@dataclass
class StatisticsContract:
    """
    Contract for computing statistics based on range_type.
    
    This is PREDEFINED (not LLM-generated). Each range_type has a fixed
    set of statistics and a corresponding compute function.
    """
    range_type: str
    statistics: List[str]
    compute_func: Callable[[List[Any]], Dict[str, Any]]
    
    def compute(self, values: List[Any]) -> Dict[str, Any]:
        """Compute statistics for the given values."""
        return self.compute_func(values)


# ============== Statistics Registry ==============

STATISTICS_REGISTRY: Dict[str, StatisticsContract] = {}


def register_statistics(
    range_type: str, 
    statistics: List[str], 
    compute_func: Callable[[List[Any]], Dict[str, Any]]
) -> None:
    """Register a statistics contract for a range_type."""
    STATISTICS_REGISTRY[range_type] = StatisticsContract(
        range_type=range_type,
        statistics=statistics,
        compute_func=compute_func,
    )


def get_statistics_contract(range_type: str) -> Optional[StatisticsContract]:
    """Get the statistics contract for a range_type."""
    if range_type in STATISTICS_REGISTRY:
        return STATISTICS_REGISTRY[range_type]
    return STATISTICS_REGISTRY.get("xsd:string")


# ============== DataProperty Lookup Helpers ==============

def find_applicable_data_properties(
    primitive_class: str,
    data_properties: List[Dict[str, Any]],
    class_hierarchy: Optional[Dict[str, List[str]]] = None,
    data_property_hierarchy: Optional[Dict[str, List[str]]] = None,
) -> List[DataPropertySpec]:
    """
    Find the MOST SPECIFIC DataProperties applicable to the given primitive class.
    
    Two key filtering principles (both applied):
    
    1. DOMAIN SPECIFICITY: Prefer properties whose domain is closer in the class hierarchy.
       E.g., if VotesColumn → QuantitativeColumn → Column, prefer properties with domain
       QuantitativeColumn over those with domain Column.
    
    2. PROPERTY HIERARCHY: Child properties override parent properties.
       E.g., if hasRank is a subproperty of hasNumericValue, only return hasRank.
    
    Args:
        primitive_class: The primitive class name (e.g., "upo:YearColumn" or "YearColumn")
        data_properties: List of DataProperty dicts from TBox
        class_hierarchy: Optional dict mapping class -> list of parent classes
        data_property_hierarchy: Optional dict mapping property -> list of parent properties
        
    Returns:
        List of DataPropertySpec for the most specific applicable properties
    """
    # Normalize primitive class (remove upo: prefix for matching)
    prim_class_normalized = primitive_class.replace('upo:', '')
    
    # Build ordered list of classes by distance from primitive class
    # Index 0 = the class itself, higher index = further ancestor
    class_distance: Dict[str, int] = {prim_class_normalized: 0}
    
    if class_hierarchy:
        distance = 0
        current_level = [prim_class_normalized]
        visited = {prim_class_normalized}
        
        while current_level:
            next_level = []
            distance += 1
            for cls in current_level:
                parents = class_hierarchy.get(cls, [])
                for parent in parents:
                    parent_normalized = parent.replace('upo:', '')
                    if parent_normalized not in visited:
                        visited.add(parent_normalized)
                        class_distance[parent_normalized] = distance
                        next_level.append(parent_normalized)
            current_level = next_level
    
    # Find all properties with their domain distance
    # Property -> (distance, property_info)
    prop_matches: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    
    for dp in data_properties:
        domains = dp.get("domain", [])
        normalized_domains = [d.replace('upo:', '') for d in domains]
        
        # Find the minimum distance for any matching domain
        min_distance = None
        for domain in normalized_domains:
            if domain in class_distance:
                dist = class_distance[domain]
                if min_distance is None or dist < min_distance:
                    min_distance = dist
        
        if min_distance is not None:
            prop_name = dp.get("name", "")
            range_type = dp.get("range_type", "xsd:string")
            if isinstance(range_type, list):
                range_type = range_type[0] if range_type else "xsd:string"
            
            prop_matches[prop_name] = (min_distance, {
                "name": prop_name,
                "range_type": range_type,
                "comment": dp.get("comment"),
                "readout_template": dp.get("readout_template"),
                "parent_properties": dp.get("parent_properties", []),
            })
    
    if not prop_matches:
        return []
    
    # Find the minimum distance across all matched properties
    min_domain_distance = min(dist for dist, _ in prop_matches.values())
    
    # Filter to keep only properties at minimum distance
    closest_props = {
        name: info for name, (dist, info) in prop_matches.items()
        if dist == min_domain_distance
    }
    
    # Apply property hierarchy filtering within the closest properties
    # Find all ancestors of matched properties
    ancestor_props = set()
    if data_property_hierarchy:
        for prop_name in closest_props:
            visited = set()
            to_visit = [prop_name]
            while to_visit:
                current = to_visit.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                parents = data_property_hierarchy.get(current, [])
                for parent in parents:
                    parent_normalized = parent.replace('upo:', '')
                    if parent_normalized not in visited:
                        ancestor_props.add(parent_normalized)
                        to_visit.append(parent_normalized)
    
    # Build final result: properties that are NOT ancestors of other matched properties
    applicable = []
    for prop_name, info in closest_props.items():
        prop_normalized = prop_name.replace('upo:', '')
        if prop_normalized not in ancestor_props:
            applicable.append(DataPropertySpec(
                name=info["name"],
                range_type=info["range_type"],
                comment=info.get("comment"),
                readout_template=info.get("readout_template"),
            ))
    
    return applicable


def get_target_type_from_range(range_type: str) -> TargetType:
    """
    Map XSD range type to Python target type.
    
    Note: Uses unified XSD_TO_TARGET_TYPE mapping from core.datatypes.el_datatypes.
    """
    return XSD_TO_TARGET_TYPE.get(range_type, TargetType.STR)
