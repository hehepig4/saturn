"""
OWL 2 EL Datatype Management Module

Provides unified datatype definitions, mappings, and utilities for OWL 2 EL compliance.

Based on W3C OWL 2 Profiles Recommendation (Section 2.2.1):
https://www.w3.org/TR/owl2-profiles/#OWL_2_EL

CRITICAL: This is the SINGLE SOURCE OF TRUTH for datatype handling.
All other modules should import from here instead of defining their own.

DESIGN PRINCIPLE:
    For each EL-compliant datatype, we define:
    1. STANDARD FORMAT: The canonical string representation
    2. VALIDATION PATTERN: Regex to validate standard format
    3. PYTHON CONVERSION: How to convert Python values to standard format
    4. LLM INSTRUCTION: Prompt snippet telling LLM how to transform to standard format
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set
import re


# ============================================================================
#                        TARGET TYPE ENUM
# ============================================================================

class TargetType(str, Enum):
    """
    Target Python types after transformation.
    
    Maps to EL-compliant XSD types:
    - INT → xsd:integer
    - FLOAT → xsd:decimal  
    - STR → xsd:string
    - DATETIME → xsd:dateTime
    """
    INT = "int"         # → xsd:integer, xsd:nonNegativeInteger
    FLOAT = "float"     # → xsd:decimal
    STR = "str"         # → xsd:string and variants
    DATETIME = "datetime"  # → xsd:dateTime, xsd:dateTimeStamp


# ============================================================================
#                        STANDARD FORMAT SPECIFICATIONS
# ============================================================================

@dataclass(frozen=True)
class DatatypeSpec:
    """
    Complete specification for an EL-compliant datatype.
    
    Defines the standard format and how to convert/validate values.
    """
    xsd_type: str                    # e.g., "xsd:integer"
    target_type: TargetType          # Python target type
    standard_format: str             # Human-readable format description
    validation_pattern: str          # Regex to validate standard format
    python_conversion: str           # Python expr to convert to standard format (var: v)
    llm_instruction: str             # Instruction for LLM to generate transform
    example_values: tuple            # Example values in standard format
    statistics: tuple                # Available statistics for this type


# Core EL-compliant datatypes with full specifications
DATATYPE_SPECS: Dict[str, DatatypeSpec] = {
    # ==================== INTEGER TYPES ====================
    "xsd:integer": DatatypeSpec(
        xsd_type="xsd:integer",
        target_type=TargetType.INT,
        standard_format="Integer without thousands separator: -?[0-9]+",
        validation_pattern=r"^-?[0-9]+$",
        python_conversion="int(v)",
        llm_instruction=(
            "Convert to integer. Remove any formatting (commas, spaces). "
            "The transform_expr should output a Python int. "
            "Example: '1,234' → int(x.replace(',', '')) → 1234"
        ),
        example_values=("-123", "0", "456", "1000000"),
        statistics=("min", "max", "mean", "median", "sum", "count", "std"),
    ),
    
    "xsd:nonNegativeInteger": DatatypeSpec(
        xsd_type="xsd:nonNegativeInteger",
        target_type=TargetType.INT,
        standard_format="Non-negative integer: [0-9]+",
        validation_pattern=r"^[0-9]+$",
        python_conversion="max(0, int(v))",
        llm_instruction=(
            "Convert to non-negative integer (≥0). Remove formatting. "
            "The transform_expr should output a Python int ≥ 0. "
            "Example: '1,234' → int(x.replace(',', '')) → 1234"
        ),
        example_values=("0", "123", "456789"),
        statistics=("min", "max", "sum", "count", "mean"),
    ),
    
    # ==================== DECIMAL TYPE ====================
    "xsd:decimal": DatatypeSpec(
        xsd_type="xsd:decimal",
        target_type=TargetType.FLOAT,
        standard_format="Decimal number: -?[0-9]+(\\.[0-9]+)?",
        validation_pattern=r"^-?[0-9]+(\.[0-9]+)?$",
        python_conversion="float(v)",
        llm_instruction=(
            "Convert to decimal number. Remove formatting (commas, currency symbols, %). "
            "For percentages like '45%', divide by 100: float(x.rstrip('%')) / 100 "
            "The transform_expr should output a Python float."
        ),
        example_values=("-123.45", "0.0", "3.14159", "1000000.00"),
        statistics=("min", "max", "mean", "median", "std", "count"),
    ),
    
    # ==================== DATETIME TYPES ====================
    "xsd:dateTime": DatatypeSpec(
        xsd_type="xsd:dateTime",
        target_type=TargetType.DATETIME,
        standard_format="ISO 8601 datetime: YYYY-MM-DDTHH:MM:SS",
        validation_pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
        python_conversion="datetime.fromisoformat(v).isoformat()",
        llm_instruction=(
            "Convert to ISO 8601 datetime format: YYYY-MM-DDTHH:MM:SS. "
            "Use datetime.fromisoformat() or parse common formats. "
            "Example: '2024-01-15' → datetime.fromisoformat(x + 'T00:00:00') "
            "Example: 'Jan 15, 2024' → parse and convert to ISO format"
        ),
        example_values=("2024-01-15T10:30:00", "2023-12-31T23:59:59"),
        statistics=("min", "max", "range_days", "count"),
    ),
    
    "xsd:dateTimeStamp": DatatypeSpec(
        xsd_type="xsd:dateTimeStamp",
        target_type=TargetType.DATETIME,
        standard_format="ISO 8601 datetime with timezone: YYYY-MM-DDTHH:MM:SS+HH:MM",
        validation_pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$",
        python_conversion="datetime.fromisoformat(v).isoformat()",
        llm_instruction=(
            "Convert to ISO 8601 datetime with timezone: YYYY-MM-DDTHH:MM:SS+HH:MM. "
            "Include timezone offset. Default to UTC (+00:00) if not specified."
        ),
        example_values=("2024-01-15T10:30:00+00:00", "2023-12-31T23:59:59-05:00"),
        statistics=("min", "max", "range_days", "count"),
    ),
    
    # ==================== STRING TYPES ====================
    "xsd:string": DatatypeSpec(
        xsd_type="xsd:string",
        target_type=TargetType.STR,
        standard_format="Unicode string, whitespace preserved",
        validation_pattern=r"^.*$",
        python_conversion="str(v)",
        llm_instruction=(
            "String type. Consider the semantic meaning of this property "
            "when choosing an appropriate transform (regex, split, keyword extraction, etc.)."
        ),
        example_values=("Hello World", "John Doe", "Category A"),
        statistics=("distinct_count", "top_values", "avg_length", "count"),
    ),
    
    "xsd:normalizedString": DatatypeSpec(
        xsd_type="xsd:normalizedString",
        target_type=TargetType.STR,
        standard_format="String with normalized whitespace (no newlines/tabs)",
        validation_pattern=r"^[^\r\n\t]*$",
        python_conversion="' '.join(str(v).split())",
        llm_instruction=(
            "Normalize whitespace: replace tabs/newlines with spaces, collapse multiple spaces. "
            "Example: ' '.join(x.split())"
        ),
        example_values=("Hello World", "Single line text"),
        statistics=("distinct_count", "top_values", "avg_length", "count"),
    ),
    
    "xsd:token": DatatypeSpec(
        xsd_type="xsd:token",
        target_type=TargetType.STR,
        standard_format="Token: no leading/trailing/consecutive whitespace",
        validation_pattern=r"^\S+(\s\S+)*$",
        python_conversion="' '.join(str(v).split())",
        llm_instruction=(
            "Convert to token format: strip and collapse whitespace. "
            "Example: ' '.join(x.split())"
        ),
        example_values=("hello", "hello world"),
        statistics=("distinct_count", "top_values", "count"),
    ),
    
    "xsd:anyURI": DatatypeSpec(
        xsd_type="xsd:anyURI",
        target_type=TargetType.STR,
        standard_format="Valid URI/IRI string",
        validation_pattern=r"^[a-zA-Z][a-zA-Z0-9+.-]*:.*$",
        python_conversion="str(v).strip()",
        llm_instruction=(
            "Convert to URI format. Ensure proper scheme (http://, https://, etc.). "
            "Example: keep as 'x' if already a URL, or prepend scheme if needed."
        ),
        example_values=("https://example.com", "urn:isbn:0451450523"),
        statistics=("distinct_count", "domain_distribution", "count"),
    ),
    
    # ==================== BINARY TYPES ====================
    "xsd:hexBinary": DatatypeSpec(
        xsd_type="xsd:hexBinary",
        target_type=TargetType.STR,
        standard_format="Hexadecimal string: [0-9A-Fa-f]*",
        validation_pattern=r"^[0-9A-Fa-f]*$",
        python_conversion="v.encode().hex()",
        llm_instruction=(
            "Convert to hexadecimal string representation. "
            "Example: bytes(x, 'utf-8').hex()"
        ),
        example_values=("48656C6C6F", "0F1E2D3C"),
        statistics=("count",),
    ),
    
    "xsd:base64Binary": DatatypeSpec(
        xsd_type="xsd:base64Binary",
        target_type=TargetType.STR,
        standard_format="Base64 encoded string",
        validation_pattern=r"^[A-Za-z0-9+/]*={0,2}$",
        python_conversion="__import__('base64').b64encode(v.encode()).decode()",
        llm_instruction=(
            "Convert to base64 string representation. "
            "Example: __import__('base64').b64encode(x.encode()).decode()"
        ),
        example_values=("SGVsbG8gV29ybGQ=",),
        statistics=("count",),
    ),
}


# ============================================================================
#                        EL-SUPPORTED DATATYPES (W3C SPEC)
# ============================================================================

# OWL 2 EL Profile supports ONLY these datatypes (W3C Recommendation §2.2.1)
EL_SUPPORTED_DATATYPES: Set[str] = {
    # Literal types
    "rdf:PlainLiteral",
    "rdf:XMLLiteral",
    "rdfs:Literal",
    
    # OWL-specific numeric types
    "owl:real",
    "owl:rational",
    
    # XSD Numeric types (ONLY these 3!)
    "xsd:decimal",
    "xsd:integer",
    "xsd:nonNegativeInteger",
    
    # XSD String types (NO xsd:language!)
    "xsd:string",
    "xsd:normalizedString",
    "xsd:token",
    "xsd:Name",
    "xsd:NCName",
    "xsd:NMTOKEN",
    
    # XSD Binary types
    "xsd:hexBinary",
    "xsd:base64Binary",
    
    # XSD URI
    "xsd:anyURI",
    
    # XSD Temporal types (ONLY dateTime and dateTimeStamp!)
    "xsd:dateTime",
    "xsd:dateTimeStamp",
}


# ============================================================================
#                        DATATYPE MAPPING (FORBIDDEN → COMPLIANT)
# ============================================================================

# Maps EL-forbidden datatypes to their closest EL-compliant equivalents
EL_DATATYPE_MAPPING: Dict[str, str] = {
    # Floating point → decimal
    "xsd:float": "xsd:decimal",
    "xsd:double": "xsd:decimal",
    
    # Integer variants → integer or nonNegativeInteger
    "xsd:positiveInteger": "xsd:nonNegativeInteger",
    "xsd:negativeInteger": "xsd:integer",
    "xsd:nonPositiveInteger": "xsd:integer",
    "xsd:long": "xsd:integer",
    "xsd:int": "xsd:integer",
    "xsd:short": "xsd:integer",
    "xsd:byte": "xsd:integer",
    "xsd:unsignedLong": "xsd:nonNegativeInteger",
    "xsd:unsignedInt": "xsd:nonNegativeInteger",
    "xsd:unsignedShort": "xsd:nonNegativeInteger",
    "xsd:unsignedByte": "xsd:nonNegativeInteger",
    
    # String variants → string
    "xsd:language": "xsd:string",
    
    # Boolean → string (with values "true"/"false")
    "xsd:boolean": "xsd:string",
    
    # Date/time variants → dateTime, integer, or string
    "xsd:date": "xsd:dateTime",
    "xsd:time": "xsd:string",
    "xsd:gYear": "xsd:integer",
    "xsd:gYearMonth": "xsd:string",
    "xsd:gMonth": "xsd:integer",
    "xsd:gMonthDay": "xsd:string",
    "xsd:gDay": "xsd:integer",
    
    # Duration → string
    "xsd:duration": "xsd:string",
    "xsd:dayTimeDuration": "xsd:string",
    "xsd:yearMonthDuration": "xsd:string",
}


# ============================================================================
#                        PYTHON ↔ XSD TYPE MAPPINGS
# ============================================================================

# XSD type → Python type (EL-compliant types only)
XSD_TO_PYTHON: Dict[str, type] = {
    # String types
    "xsd:string": str,
    "xsd:normalizedString": str,
    "xsd:token": str,
    "xsd:Name": str,
    "xsd:NCName": str,
    "xsd:NMTOKEN": str,
    
    # Numeric types
    "xsd:integer": int,
    "xsd:nonNegativeInteger": int,
    "xsd:decimal": float,
    
    # Date/time (as strings - owlready2 stores them as strings)
    "xsd:dateTime": str,
    "xsd:dateTimeStamp": str,
    
    # URI
    "xsd:anyURI": str,
    
    # Binary
    "xsd:hexBinary": str,
    "xsd:base64Binary": str,
    
    # Literal types
    "rdf:PlainLiteral": str,
    "rdf:XMLLiteral": str,
    "rdfs:Literal": str,
    
    # OWL numeric
    "owl:real": float,
    "owl:rational": float,
}

# XSD type → TargetType (for TransformContract)
XSD_TO_TARGET_TYPE: Dict[str, TargetType] = {
    # Integer types
    "xsd:integer": TargetType.INT,
    "xsd:nonNegativeInteger": TargetType.INT,
    
    # Decimal types
    "xsd:decimal": TargetType.FLOAT,
    
    # DateTime types
    "xsd:dateTime": TargetType.DATETIME,
    "xsd:dateTimeStamp": TargetType.DATETIME,
    
    # String types
    "xsd:string": TargetType.STR,
    "xsd:normalizedString": TargetType.STR,
    "xsd:token": TargetType.STR,
    "xsd:Name": TargetType.STR,
    "xsd:NCName": TargetType.STR,
    "xsd:NMTOKEN": TargetType.STR,
    "xsd:anyURI": TargetType.STR,
    "xsd:hexBinary": TargetType.STR,
    "xsd:base64Binary": TargetType.STR,
    
    # Literal types
    "rdf:PlainLiteral": TargetType.STR,
    "rdfs:Literal": TargetType.STR,
}


# ============================================================================
#                        STATISTICS PER DATATYPE
# ============================================================================

# Statistics available for each datatype
# NOTE: This is the SINGLE SOURCE OF TRUTH for available statistics.
# Prompt templates should dynamically read from here.
DATATYPE_STATISTICS: Dict[str, List[str]] = {
    # Integer types - supports mode (discrete values)
    "xsd:integer": [
        "min", "max", "mean", "median", "sum", "count", "std",
        "percentile_25", "percentile_75", "mode"
    ],
    "xsd:nonNegativeInteger": [
        "min", "max", "sum", "count", "mean",
        "percentile_25", "percentile_75", "mode"
    ],
    
    # Decimal/Float types - no mode (continuous values)
    "xsd:decimal": [
        "min", "max", "mean", "median", "std", "count",
        "percentile_25", "percentile_75"
    ],
    
    # Date/Time types
    "xsd:dateTime": ["min", "max", "range_days", "count"],
    "xsd:dateTimeStamp": ["min", "max", "range_days", "count"],
    
    # String types - enhanced with text analysis statistics
    "xsd:string": [
        "distinct_count", "top_values", "sample_values", "avg_length", "count",
        "tfidf_keywords", "word_count_avg", "max_length"
    ],
    "xsd:normalizedString": [
        "distinct_count", "top_values", "sample_values", "avg_length", "count",
        "word_count_avg", "max_length"
    ],
    "xsd:token": ["distinct_count", "top_values", "sample_values", "count"],
    "xsd:Name": ["distinct_count", "top_values", "sample_values", "count"],
    "xsd:NCName": ["distinct_count", "top_values", "sample_values", "count"],
    "xsd:NMTOKEN": ["distinct_count", "top_values", "sample_values", "count"],
    
    # URI
    "xsd:anyURI": ["distinct_count", "domain_distribution", "count"],
    
    # Binary
    "xsd:hexBinary": ["count"],
    "xsd:base64Binary": ["count"],
    
    # Literal types
    "rdf:PlainLiteral": ["distinct_count", "top_values", "sample_values"],
    "rdf:XMLLiteral": ["count"],
    "rdfs:Literal": ["distinct_count", "sample_values"],
    
    # OWL numeric - same as xsd:decimal
    "owl:real": [
        "min", "max", "mean", "median", "std", "count",
        "percentile_25", "percentile_75"
    ],
    "owl:rational": ["min", "max", "mean", "count"],
}


# ============================================================================
#                        HELPER FUNCTIONS FOR STANDARD FORMAT
# ============================================================================

def get_datatype_spec(xsd_type: str) -> Optional[DatatypeSpec]:
    """
    Get the full specification for an EL-compliant datatype.
    
    Args:
        xsd_type: XSD datatype (e.g., "xsd:integer")
        
    Returns:
        DatatypeSpec or None if not found
    """
    return DATATYPE_SPECS.get(xsd_type)


def get_llm_instruction(xsd_type: str) -> str:
    """
    Get the LLM instruction for converting to a datatype's standard format.
    
    Args:
        xsd_type: XSD datatype (e.g., "xsd:integer")
        
    Returns:
        Instruction string for LLM prompt
    """
    spec = DATATYPE_SPECS.get(xsd_type)
    if spec:
        return spec.llm_instruction
    
    # Default for unknown types
    return "Convert to string format. Output type: str."
