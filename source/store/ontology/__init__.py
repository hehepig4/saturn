"""
Ontology Storage Module

This module provides ontology management capabilities:
- TBox storage (classes, properties, axioms) via OntologyManager
- OWL conversion for validation

Architecture:
    store/ontology/
    ├── __init__.py          # This file - public API
    ├── ontology_table.py    # LanceDB table managers
    ├── ontology_manager.py  # High-level TBox API
    └── owlready_converter.py   # OWL export for validation

Example:
    >>> from store.ontology import OntologyManager
    >>> manager = OntologyManager(db)
    >>> ontology_id = manager.store_tbox(tbox, name="my_ontology")
"""

# Table managers and high-level API
from .ontology_table import (
    OntologyTableManager,
    OntologyClassTableManager,
    OntologyPropertyTableManager,
    OntologyAxiomTableManager,
)

from .ontology_manager import OntologyManager

# OWL conversion for validation
from .owlready_converter import OwlreadyConverter, ConversionResult


__all__ = [
    # Managers
    "OntologyManager",
    "OntologyTableManager",
    "OntologyClassTableManager",
    "OntologyPropertyTableManager",
    "OntologyAxiomTableManager",
    # OWL conversion
    "OwlreadyConverter",
    "ConversionResult",
]
