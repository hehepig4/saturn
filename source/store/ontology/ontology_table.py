"""
Ontology Table Managers

Four-table design for ontology storage:
1. ontology_metadata - Top-level ontology info (version, creation time, metrics)
2. ontology_classes - Class definitions with hierarchy
3. ontology_properties - Object and data properties  
4. ontology_axioms - Logical constraints and axioms

Design Rationale:
- Separate tables for flexibility in querying (e.g., "find all classes in ontology X")
- Denormalized for performance (ontology_id repeated in each table)
- Supports multiple ontology versions simultaneously
- Enables incremental updates (add/remove classes without full reload)

Example Usage:
    >>> # Create ontology metadata
    >>> ontology_mgr = OntologyTableManager(db)
    >>> ontology_id = ontology_mgr.create_ontology(
    ...     name="user_preference_v1",
    ...     ontology_type="user_preference",
    ...     namespace="http://example.org/upo#",
    ...     version="1.0",
    ...     dataset_name="inquire_rerank",
    ...     query_corpus="validation_queries",
    ...     odp_sources=["Participation", "Temporal"]
    ... )
    >>> 
    >>> # Add classes
    >>> class_mgr = OntologyClassTableManager(db)
    >>> class_mgr.add_class(
    ...     ontology_id=ontology_id,
    ...     class_name="Animal",
    ...     label="Animal",
    ...     parent_classes=["NaturalWorldEntity"],
    ...     description="Living organism"
    ... )
    >>> 
    >>> # Query all classes in an ontology
    >>> classes = class_mgr.get_classes_by_ontology(ontology_id)
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from loguru import logger
import lancedb
import pyarrow as pa

from store.core.base_table import BaseTableManager


# ==================== Table 1: Ontology Metadata ====================

class OntologyTableManager(BaseTableManager):
    """
    Manages top-level ontology metadata.
    
    Each row = one ontology version
    
    Schema:
        - ontology_id: str (PK) - Unique identifier
        - name: str - Human-readable ontology name
        - ontology_type: str - Type (e.g., "user_preference_ontology")
        - namespace: str - OWL namespace URI
        - version: str - Version string
        - domain: list<str> - Domain scope (dataset names where ontology applies)
        - dataset_name: str - Associated dataset (optional, deprecated, use domain)
        - query_corpus: str - Query corpus used (optional)
        - odp_sources: list<str> - ODP patterns used
        - metrics: str (JSON) - Verification metrics
        - notes: str - Additional notes
        - num_classes: int - Count of classes
        - num_object_properties: int - Count of object properties
        - num_data_properties: int - Count of data properties
        - num_axioms: int - Count of axioms
        - created_at: timestamp
        - updated_at: timestamp
    """
    
    def __init__(self, db: lancedb.DBConnection):
        super().__init__(db, "ontology_metadata")
    
    def get_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("ontology_id", pa.string(), nullable=False),
            pa.field("name", pa.string(), nullable=False),
            pa.field("ontology_type", pa.string(), nullable=False),
            pa.field("namespace", pa.string(), nullable=False),
            pa.field("version", pa.string(), nullable=False),
            pa.field("domain", pa.list_(pa.string()), nullable=True),  # Domain scope
            pa.field("created_at", pa.timestamp('us'), nullable=False),
            pa.field("updated_at", pa.timestamp('us'), nullable=False),
            pa.field("dataset_name", pa.string(), nullable=True),  # Deprecated
            pa.field("query_corpus", pa.string(), nullable=True),
            pa.field("num_classes", pa.int32(), nullable=False),
            pa.field("num_object_properties", pa.int32(), nullable=False),
            pa.field("num_data_properties", pa.int32(), nullable=False),
            pa.field("num_axioms", pa.int32(), nullable=False),
            pa.field("num_external_alignments", pa.int32(), nullable=True),  # Wikidata/Schema.org alignments
            pa.field("alignment_coverage", pa.float64(), nullable=True),  # Percentage of aligned concepts
            pa.field("odp_sources", pa.list_(pa.string()), nullable=True),
            pa.field("metrics", pa.string(), nullable=True),  # JSON string
            pa.field("notes", pa.string(), nullable=True),
        ])
    
    def _create_indices(self) -> None:
        """Create indices for fast lookup."""
        # Index on ontology_type for filtering by type
        # Index on dataset_name for finding ontologies for a dataset
        pass
    
    def create_ontology(
        self,
        name: str,
        ontology_type: str,
        namespace: str,
        version: str,
        domain: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        query_corpus: Optional[str] = None,
        odp_sources: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
        ontology_id: Optional[str] = None,
    ) -> str:
        """
        Create a new ontology metadata record.
        
        Args:
            name: Human-readable name
            ontology_type: Type of ontology
            namespace: OWL namespace URI
            version: Version string
            dataset_name: Associated dataset
            query_corpus: Query corpus identifier
            odp_sources: List of ODP names used
            metrics: Dict with verification metrics
            notes: Additional notes
            ontology_id: Optional custom ID (auto-generated if not provided)
        
        Returns:
            ontology_id: Unique identifier for this ontology
        """
        import json
        
        # Generate ID if not provided
        if ontology_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = name.replace(" ", "_").lower()
            ontology_id = f"{safe_name}_{timestamp}"
        
        now = datetime.now()
        
        # Auto-derive domain from dataset_name if not provided
        if domain is None and dataset_name is not None:
            domain = [dataset_name]
        elif domain is None:
            domain = []
        
        data = {
            "ontology_id": ontology_id,
            "name": name,
            "ontology_type": ontology_type,
            "namespace": namespace,
            "version": version,
            "domain": domain,  # NEW: Domain scope
            "created_at": now,
            "updated_at": now,
            "dataset_name": dataset_name,  # Keep for backward compatibility
            "query_corpus": query_corpus,
            "num_classes": 0,  # Will be updated as classes are added
            "num_object_properties": 0,
            "num_data_properties": 0,
            "num_axioms": 0,
            "num_external_alignments": 0,  # Wikidata/Schema.org alignments
            "alignment_coverage": 0.0,  # Percentage of aligned concepts
            "odp_sources": odp_sources or [],
            "metrics": json.dumps(metrics) if metrics else None,
            "notes": notes,
        }
        
        self.insert(data)
        logger.info(f"✓ Created ontology metadata: {ontology_id}")
        
        return ontology_id
    
    def update_counts(
        self,
        ontology_id: str,
        num_classes: int,
        num_object_properties: int,
        num_data_properties: int,
        num_axioms: int,
    ) -> bool:
        """Update counts after adding classes/properties/axioms."""
        # Use delete + insert pattern since update() is disabled in LanceDB
        
        # 1. Get existing record
        existing = self.get_ontology(ontology_id)
        if not existing:
            logger.warning(f"Cannot update counts: ontology {ontology_id} not found")
            return False
        
        # 2. Update counts in the record
        now = datetime.now()
        existing["num_classes"] = num_classes
        existing["num_object_properties"] = num_object_properties
        existing["num_data_properties"] = num_data_properties
        existing["num_axioms"] = num_axioms
        existing["updated_at"] = now
        
        # 3. Delete old record
        self.delete("ontology_id", ontology_id)
        
        # 4. Insert updated record
        self.insert([existing])
        
        return True
    
    def get_ontology(self, ontology_id: str) -> Optional[Dict[str, Any]]:
        """Get ontology metadata by ID."""
        return self.get_by_id("ontology_id", ontology_id)
    
    def get_ontologies_by_type(self, ontology_type: str) -> List[Dict[str, Any]]:
        """Get all ontologies of a specific type."""
        return self.query(filter_expr=f"ontology_type = '{ontology_type}'")
    
    def get_ontologies_by_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Get all ontologies associated with a dataset."""
        return self.query(filter_expr=f"dataset_name = '{dataset_name}'")
    
    def get_latest_version(self, ontology_type: str, dataset_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get the most recent ontology version."""
        filter_expr = f"ontology_type = '{ontology_type}'"
        if dataset_name:
            filter_expr += f" AND dataset_name = '{dataset_name}'"
        
        results = self.query(filter_expr=filter_expr)
        
        if not results:
            return None
        
        # Sort by created_at descending
        results.sort(key=lambda x: x["created_at"], reverse=True)
        return results[0]
    
    def get_version_by_iteration(
        self,
        ontology_type: str,
        dataset_name: str,
        iteration: int = -1,
    ) -> Optional[Dict[str, Any]]:
        """
        Get ontology version by iteration number.
        
        The version field stores iteration info as "iter_{n}" format.
        
        Args:
            ontology_type: Type of ontology (e.g., "federated_primitive_tbox")
            dataset_name: Dataset name
            iteration: Iteration number to retrieve. -1 means the latest iteration.
        
        Returns:
            Ontology metadata dict or None if not found
        """
        filter_expr = f"ontology_type = '{ontology_type}'"
        if dataset_name:
            filter_expr += f" AND dataset_name = '{dataset_name}'"
        
        results = self.query(filter_expr=filter_expr)
        
        if not results:
            return None
        
        # Parse iteration from version field (format: "iter_{n}")
        def extract_iteration(record: Dict[str, Any]) -> int:
            version = record.get("version", "")
            if version.startswith("iter_"):
                try:
                    return int(version.split("_")[1])
                except (IndexError, ValueError):
                    pass
            return -1
        
        # Sort by created_at descending within each iteration
        results.sort(key=lambda x: x["created_at"], reverse=True)
        
        if iteration == -1:
            # Find the max iteration number
            max_iter = max(extract_iteration(r) for r in results)
            # Get the latest record with max iteration
            for r in results:
                if extract_iteration(r) == max_iter:
                    return r
            return results[0]  # Fallback
        else:
            # Find specific iteration
            for r in results:
                if extract_iteration(r) == iteration:
                    return r
            return None
    
    def list_iterations(
        self,
        ontology_type: str,
        dataset_name: str,
    ) -> List[int]:
        """
        List all available iterations for a dataset.
        
        Args:
            ontology_type: Type of ontology
            dataset_name: Dataset name
        
        Returns:
            Sorted list of iteration numbers
        """
        filter_expr = f"ontology_type = '{ontology_type}' AND dataset_name = '{dataset_name}'"
        results = self.query(filter_expr=filter_expr)
        
        iterations = set()
        for r in results:
            version = r.get("version", "")
            if version.startswith("iter_"):
                try:
                    iterations.add(int(version.split("_")[1]))
                except (IndexError, ValueError):
                    pass
        
        return sorted(iterations)
    


# ==================== Table 2: Ontology Classes ====================

class OntologyClassTableManager(BaseTableManager):
    """
    Manages OWL class definitions.
    
    Each row = one class in one ontology
    
    Schema:
        - class_id: str (PK) - Unique identifier (ontology_id + class_name)
        - ontology_id: str (FK) - Reference to ontology_metadata
        - class_name: str - Class name (e.g., "Animal", "VisualObject")
        - label: str - Human-readable label
        - description: str - Class description
        - parent_classes: list<str> - Superclasses
        - equivalent_classes: list<str> - Equivalent classes
        - disjoint_with: list<str> - Disjoint classes
        - created_at: timestamp
    """
    
    def __init__(self, db: lancedb.DBConnection):
        super().__init__(db, "ontology_classes")
    
    def get_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("class_id", pa.string(), nullable=False),
            pa.field("ontology_id", pa.string(), nullable=False),
            pa.field("class_name", pa.string(), nullable=False),
            pa.field("label", pa.string(), nullable=False),
            pa.field("description", pa.string(), nullable=True),
            pa.field("parent_classes", pa.list_(pa.string()), nullable=True),
            pa.field("equivalent_classes", pa.list_(pa.string()), nullable=True),
            pa.field("disjoint_with", pa.list_(pa.string()), nullable=True),
            pa.field("annotations", pa.string(), nullable=True),  # JSON serialized AnnotationCollection
            pa.field("created_at", pa.timestamp('us'), nullable=False),
        ])
    
    def _create_indices(self) -> None:
        """Create indices for fast lookup."""
        # Index on ontology_id for filtering by ontology
        # Index on class_name for finding specific classes
        pass
    
    def add_class(
        self,
        ontology_id: str,
        class_name: str,
        label: str,
        parent_classes: Optional[List[str]] = None,
        equivalent_classes: Optional[List[str]] = None,
        disjoint_with: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> str:
        """
        Add a class to an ontology.
        
        Returns:
            class_id: Unique identifier for this class
        """
        class_id = f"{ontology_id}:{class_name}"
        
        data = {
            "class_id": class_id,
            "ontology_id": ontology_id,
            "class_name": class_name,
            "label": label,
            "description": description,
            "parent_classes": parent_classes or [],
            "equivalent_classes": equivalent_classes or [],
            "disjoint_with": disjoint_with or [],
            "created_at": datetime.now(),
        }
        
        self.insert(data)
        logger.debug(f"Added class: {class_name} to {ontology_id}")
        
        return class_id
    
    def get_classes_by_ontology(self, ontology_id: str) -> List[Dict[str, Any]]:
        """Get all classes in an ontology."""
        return self.query(filter_expr=f"ontology_id = '{ontology_id}'")
    
    def get_class(self, ontology_id: str, class_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific class."""
        class_id = f"{ontology_id}:{class_name}"
        return self.get_by_id("class_id", class_id)
    
    def add_classes_batch(
        self,
        ontology_id: str,
        classes: List[Dict[str, Any]]
    ) -> int:
        """
        Batch add multiple classes (much faster than one-by-one).
        
        Args:
            ontology_id: Ontology ID
            classes: List of class dicts with keys:
                - class_name, label
                - parent_classes, equivalent_classes, disjoint_with (optional lists)
                - description (optional str)
        
        Returns:
            Number of classes inserted
        """
        if not classes:
            return 0
        
        now = datetime.now()
        records = []
        for cls_data in classes:
            class_name = cls_data['class_name']
            class_id = f"{ontology_id}:{class_name}"
            records.append({
                "class_id": class_id,
                "ontology_id": ontology_id,
                "class_name": class_name,
                "label": cls_data.get('label', class_name),
                "description": cls_data.get('description'),
                "parent_classes": cls_data.get('parent_classes') or [],
                "equivalent_classes": cls_data.get('equivalent_classes') or [],
                "disjoint_with": cls_data.get('disjoint_with') or [],
                "created_at": now,
            })
        
        self.table.add(records)
        logger.debug(f"Batch added {len(records)} classes to {ontology_id}")
        return len(records)
    
    def delete_classes_by_ontology(self, ontology_id: str) -> int:
        """Delete all classes for an ontology."""
        classes = self.get_classes_by_ontology(ontology_id)
        for cls in classes:
            self.delete("class_id", cls["class_id"])
        return len(classes)


# ==================== Table 3: Ontology Properties ====================

class OntologyPropertyTableManager(BaseTableManager):
    """
    Manages OWL properties (both object and data properties).
    
    Each row = one property in one ontology
    
    Schema:
        - property_id: str (PK) - Unique identifier
        - ontology_id: str (FK) - Reference to ontology_metadata
        - property_name: str - Property name
        - property_type: str - "object" or "data"
        - label: str - Human-readable label
        - description: str - Property description
        - domain: list<str> - Domain classes
        - range: list<str> - Range (classes for object, datatypes for data)
        - characteristics: list<str> - ["functional", "transitive", etc.]
        - inverse_of: str - Inverse property name (for object properties)
        - readout_template: str - Template for generating human-readable descriptions (for data properties)
        - created_at: timestamp
    """
    
    def __init__(self, db: lancedb.DBConnection):
        super().__init__(db, "ontology_properties")
    
    def get_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("property_id", pa.string(), nullable=False),
            pa.field("ontology_id", pa.string(), nullable=False),
            pa.field("property_name", pa.string(), nullable=False),
            pa.field("property_type", pa.string(), nullable=False),  # "object" or "data"
            pa.field("label", pa.string(), nullable=False),
            pa.field("description", pa.string(), nullable=True),
            pa.field("domain", pa.list_(pa.string()), nullable=True),
            pa.field("range", pa.list_(pa.string()), nullable=True),
            pa.field("characteristics", pa.list_(pa.string()), nullable=True),
            pa.field("inverse_of", pa.string(), nullable=True),
            pa.field("parent_properties", pa.list_(pa.string()), nullable=True),
            pa.field("annotations", pa.string(), nullable=True),  # JSON serialized AnnotationCollection
            pa.field("readout_template", pa.string(), nullable=True),  # Template for human-readable descriptions
            pa.field("created_at", pa.timestamp('us'), nullable=False),
        ])
    
    def _create_indices(self) -> None:
        """Create indices for fast lookup."""
        pass
    
    def add_property(
        self,
        ontology_id: str,
        property_name: str,
        property_type: str,  # "object" or "data"
        label: str,
        domain: Optional[List[str]] = None,
        range: Optional[List[str]] = None,
        characteristics: Optional[List[str]] = None,
        inverse_of: Optional[str] = None,
        parent_properties: Optional[List[str]] = None,
        description: Optional[str] = None,
        readout_template: Optional[str] = None,
    ) -> str:
        """
        Add a property to an ontology.
        
        Returns:
            property_id: Unique identifier for this property
        """
        property_id = f"{ontology_id}:{property_name}"
        
        data = {
            "property_id": property_id,
            "ontology_id": ontology_id,
            "property_name": property_name,
            "property_type": property_type,
            "label": label,
            "description": description,
            "domain": domain or [],
            "range": range or [],
            "characteristics": characteristics or [],
            "inverse_of": inverse_of,
            "parent_properties": parent_properties or [],
            "readout_template": readout_template,
            "created_at": datetime.now(),
        }
        
        self.insert(data)
        logger.debug(f"Added {property_type} property: {property_name} to {ontology_id}")
        
        return property_id
    
    def get_properties_by_ontology(
        self,
        ontology_id: str,
        property_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all properties in an ontology, optionally filtered by type."""
        filter_expr = f"ontology_id = '{ontology_id}'"
        if property_type:
            filter_expr += f" AND property_type = '{property_type}'"
        
        return self.query(filter_expr=filter_expr)
    
    def get_property(self, ontology_id: str, property_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific property."""
        property_id = f"{ontology_id}:{property_name}"
        return self.get_by_id("property_id", property_id)
    
    def add_properties_batch(
        self,
        ontology_id: str,
        properties: List[Dict[str, Any]]
    ) -> int:
        """
        Batch add multiple properties (much faster than one-by-one).
        
        Args:
            ontology_id: Ontology ID
            properties: List of property dicts with keys:
                - property_name, property_type ('object' or 'data'), label
                - domain, range, characteristics (optional lists)
                - inverse_of, parent_properties, description, readout_template (optional)
        
        Returns:
            Number of properties inserted
        """
        if not properties:
            return 0
        
        now = datetime.now()
        records = []
        for prop_data in properties:
            property_name = prop_data['property_name']
            property_id = f"{ontology_id}:{property_name}"
            records.append({
                "property_id": property_id,
                "ontology_id": ontology_id,
                "property_name": property_name,
                "property_type": prop_data.get('property_type', 'object'),
                "label": prop_data.get('label', property_name),
                "description": prop_data.get('description'),
                "domain": prop_data.get('domain') or [],
                "range": prop_data.get('range') or [],
                "characteristics": prop_data.get('characteristics') or [],
                "inverse_of": prop_data.get('inverse_of'),
                "parent_properties": prop_data.get('parent_properties') or [],
                "readout_template": prop_data.get('readout_template'),
                "created_at": now,
            })
        
        self.table.add(records)
        logger.debug(f"Batch added {len(records)} properties to {ontology_id}")
        return len(records)
    
    def delete_properties_by_ontology(self, ontology_id: str) -> int:
        """Delete all properties for an ontology."""
        properties = self.get_properties_by_ontology(ontology_id)
        for prop in properties:
            self.delete("property_id", prop["property_id"])
        return len(properties)


# ==================== Table 4: Ontology Axioms ====================

class OntologyAxiomTableManager(BaseTableManager):
    """
    Manages OWL axioms and logical constraints.
    
    Each row = one axiom in one ontology
    
    Schema:
        - axiom_id: str (PK) - Unique identifier
        - ontology_id: str (FK) - Reference to ontology_metadata
        - axiom_type: str - "subclass", "disjoint", "domain", "range", etc.
        - subject: str - Subject class/property
        - predicate: str - Axiom predicate
        - object: str - Object class/property/value
        - description: str - Human-readable explanation
        - created_at: timestamp
    """
    
    def __init__(self, db: lancedb.DBConnection):
        super().__init__(db, "ontology_axioms")
    
    def get_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("axiom_id", pa.string(), nullable=False),
            pa.field("ontology_id", pa.string(), nullable=False),
            pa.field("axiom_type", pa.string(), nullable=False),
            pa.field("subject", pa.string(), nullable=False),
            pa.field("predicate", pa.string(), nullable=False),
            pa.field("object", pa.string(), nullable=False),
            pa.field("description", pa.string(), nullable=True),
            pa.field("annotations", pa.string(), nullable=True),  # JSON serialized AnnotationCollection
            pa.field("created_at", pa.timestamp('us'), nullable=False),
        ])
    
    def _create_indices(self) -> None:
        """Create indices for fast lookup."""
        pass
    
    def add_axiom(
        self,
        ontology_id: str,
        axiom_type: str,
        subject: str,
        predicate: str,
        object: str,
        description: Optional[str] = None,
    ) -> str:
        """
        Add an axiom to an ontology.
        
        Returns:
            axiom_id: Unique identifier for this axiom
        """
        # Generate unique ID based on content
        import hashlib
        content = f"{axiom_type}:{subject}:{predicate}:{object}"
        hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
        axiom_id = f"{ontology_id}:axiom_{hash_suffix}"
        
        data = {
            "axiom_id": axiom_id,
            "ontology_id": ontology_id,
            "axiom_type": axiom_type,
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "description": description,
            "created_at": datetime.now(),
        }
        
        self.insert(data)
        logger.debug(f"Added axiom: {axiom_type} to {ontology_id}")
        
        return axiom_id
    
    def get_axioms_by_ontology(
        self,
        ontology_id: str,
        axiom_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all axioms in an ontology, optionally filtered by type."""
        filter_expr = f"ontology_id = '{ontology_id}'"
        if axiom_type:
            filter_expr += f" AND axiom_type = '{axiom_type}'"
        
        return self.query(filter_expr=filter_expr)
    
    def add_axioms_batch(
        self,
        ontology_id: str,
        axioms: List[Dict[str, Any]]
    ) -> int:
        """
        Batch add multiple axioms (much faster than one-by-one).
        
        Args:
            ontology_id: Ontology ID
            axioms: List of axiom dicts with keys:
                - axiom_type, subject, predicate, object
                - description (optional)
        
        Returns:
            Number of axioms inserted
        """
        if not axioms:
            return 0
        
        import hashlib
        now = datetime.now()
        records = []
        for axiom_data in axioms:
            # Generate unique ID based on content
            content = f"{axiom_data['axiom_type']}:{axiom_data['subject']}:{axiom_data['predicate']}:{axiom_data['object']}"
            hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
            axiom_id = f"{ontology_id}:axiom_{hash_suffix}"
            
            records.append({
                "axiom_id": axiom_id,
                "ontology_id": ontology_id,
                "axiom_type": axiom_data['axiom_type'],
                "subject": axiom_data['subject'],
                "predicate": axiom_data['predicate'],
                "object": axiom_data['object'],
                "description": axiom_data.get('description'),
                "created_at": now,
            })
        
        self.table.add(records)
        logger.debug(f"Batch added {len(records)} axioms to {ontology_id}")
        return len(records)
    
    def delete_axioms_by_ontology(self, ontology_id: str) -> int:
        """Delete all axioms for an ontology."""
        axioms = self.get_axioms_by_ontology(ontology_id)
        for axiom in axioms:
            self.delete("axiom_id", axiom["axiom_id"])
        return len(axioms)
