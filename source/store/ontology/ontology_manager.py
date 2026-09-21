"""
Ontology Manager - High-level API for TBox Storage

Provides unified interface for storing and retrieving complete ontologies.
Manages coordination between the four underlying tables.

Note:
    For TBox snapshot persistence, use workflows.conceptualization.utils.snapshot
    which provides save_tbox_snapshot() and load_tbox_snapshot() functions that work
    directly with dictionary-based TBox representations.

Example Usage:
    >>> from store.ontology import OntologyManager
    >>> 
    >>> # Initialize manager
    >>> manager = OntologyManager(db)
    >>> 
    >>> # List ontologies
    >>> ontologies = manager.list_ontologies(ontology_type="federated_primitive_tbox")
    >>> 
    >>> # Get latest metadata
    >>> latest = manager.get_latest_ontology(
    ...     ontology_type="federated_primitive_tbox",
    ...     dataset_name="adventure_works"
    ... )
"""

from typing import Any, Dict, List, Optional
import lancedb

from .ontology_table import (
    OntologyTableManager,
    OntologyClassTableManager,
    OntologyPropertyTableManager,
    OntologyAxiomTableManager,
)


class OntologyManager:
    """
    High-level manager for complete ontology storage and retrieval.
    
    Provides access to the underlying table managers for direct operations,
    and convenience methods for listing and querying ontologies.
    
    Attributes:
        metadata_mgr: OntologyTableManager for metadata operations
        class_mgr: OntologyClassTableManager for class operations
        property_mgr: OntologyPropertyTableManager for property operations
        axiom_mgr: OntologyAxiomTableManager for axiom operations
    """
    
    def __init__(self, db: lancedb.DBConnection):
        """
        Initialize ontology manager.
        
        Args:
            db: LanceDB connection
        """
        self.db = db
        
        # TBox managers - exposed for direct access
        self.metadata_mgr = OntologyTableManager(db)
        self.class_mgr = OntologyClassTableManager(db)
        self.property_mgr = OntologyPropertyTableManager(db)
        self.axiom_mgr = OntologyAxiomTableManager(db)
    
    def list_ontologies(
        self,
        ontology_type: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all ontologies, optionally filtered.
        
        Args:
            ontology_type: Filter by type
            dataset_name: Filter by dataset
        
        Returns:
            List of metadata dicts sorted by created_at (newest first)
        """
        if ontology_type:
            results = self.metadata_mgr.get_ontologies_by_type(ontology_type)
        elif dataset_name:
            results = self.metadata_mgr.get_ontologies_by_dataset(dataset_name)
        else:
            results = self.metadata_mgr.query()
        
        # Sort by created_at descending
        results.sort(key=lambda x: x["created_at"], reverse=True)
        
        return results
    
    def get_latest_ontology(
        self,
        ontology_type: str,
        dataset_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the most recent ontology of a given type.
        
        Args:
            ontology_type: Type of ontology
            dataset_name: Optional dataset filter
        
        Returns:
            Metadata dict or None
        """
        return self.metadata_mgr.get_latest_version(ontology_type, dataset_name)
