# Unified OWL 2 EL Datatype Management
"""
Unified datatype management for OWL 2 EL compliance.

This module provides a single source of truth for:
1. EL-supported datatypes and their specifications
2. Standard format definitions per datatype
3. Python type mappings
4. Statistics definitions per datatype
"""

from core.datatypes.el_datatypes import (
    # Constants
    EL_SUPPORTED_DATATYPES,
    EL_DATATYPE_MAPPING,
    DATATYPE_STATISTICS,
    DATATYPE_SPECS,
    XSD_TO_PYTHON,
    XSD_TO_TARGET_TYPE,
    # Types
    TargetType,
    DatatypeSpec,
    # Functions
    get_datatype_spec,
    get_llm_instruction,
)

__all__ = [
    # Constants
    "EL_SUPPORTED_DATATYPES",
    "EL_DATATYPE_MAPPING",
    "DATATYPE_STATISTICS",
    "DATATYPE_SPECS",
    "XSD_TO_PYTHON",
    "XSD_TO_TARGET_TYPE",
    # Types
    "TargetType",
    "DatatypeSpec",
    # Functions
    "get_datatype_spec",
    "get_llm_instruction",
]