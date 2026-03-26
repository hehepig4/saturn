"""
Utility Functions for Table Discovery Layer 2

Includes:
- P1-c: Primitive Class Validation (prevent LLM hallucinations)
"""

from typing import Set, Tuple
import re
from loguru import logger


# ============== Primitive Class Validation (P1-c) ==============


def validate_primitive_class(
    primitive_class: str,
    layer1_classes: Set[str],
) -> Tuple[str, str]:
    """
    Validate and correct primitive_class to ensure it's in Layer 1.
    
    Prevents LLM hallucination by checking against the ACTUAL Layer 1 classes
    loaded from LanceDB/OWL (NOT a hardcoded list).
    
    Args:
        primitive_class: The primitive class assigned by LLM
        layer1_classes: Set of valid Layer 1 class names (from LanceDB)
        
    Returns:
        Tuple of (corrected_class, correction_type)
        correction_type: "valid" | "case_corrected" | "fuzzy_corrected" | 
                        "word_overlap_corrected" | "fallback" | "unvalidated"
    """
    if not layer1_classes:
        logger.warning("No Layer 1 classes provided for validation, skipping")
        return primitive_class, "unvalidated"
    
    # 1. Exact match
    if primitive_class in layer1_classes:
        return primitive_class, "valid"
    
    # 2. Case-insensitive match
    lower_map = {c.lower(): c for c in layer1_classes}
    if primitive_class.lower() in lower_map:
        corrected = lower_map[primitive_class.lower()]
        logger.debug(f"Case correction: {primitive_class} → {corrected}")
        return corrected, "case_corrected"
    
    # 3. Fuzzy match (substring matching)
    # Find a Layer 1 class that contains or is contained in the given class name
    for l1_class in layer1_classes:
        # "PersonColumn" matches "Person", "Organization" matches "CompanyColumn" etc.
        if primitive_class.lower() in l1_class.lower():
            logger.warning(f"Fuzzy match (substring): {primitive_class} → {l1_class}")
            return l1_class, "fuzzy_corrected"
        if l1_class.lower() in primitive_class.lower():
            logger.warning(f"Fuzzy match (superstring): {primitive_class} → {l1_class}")
            return l1_class, "fuzzy_corrected"
    
    # 4. Semantic similarity fallback - find best match by word overlap
    primitive_words = set(re.findall(r'[A-Z][a-z]+|[a-z]+', primitive_class))
    best_match = None
    best_score = 0
    
    for l1_class in layer1_classes:
        l1_words = set(re.findall(r'[A-Z][a-z]+|[a-z]+', l1_class))
        overlap = len(primitive_words & l1_words)
        if overlap > best_score:
            best_score = overlap
            best_match = l1_class
    
    if best_match and best_score > 0:
        logger.warning(f"Word overlap match: {primitive_class} → {best_match} (score={best_score})")
        return best_match, "word_overlap_corrected"
    
    # 5. Fallback to TextValue (or whatever is the default in Layer 1)
    fallback = "TextValue" if "TextValue" in layer1_classes else next(iter(layer1_classes))
    logger.warning(f"Unknown primitive class: {primitive_class} → {fallback} (fallback)")
    return fallback, "fallback"
