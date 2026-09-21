"""
TBox Snapshot Management

Saves iteration snapshots to LanceDB for:
1. State persistence across sessions
2. Iteration comparison and debugging
3. Future resumption of interrupted runs
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger

from store.store_singleton import get_store


def save_tbox_snapshot(
    tbox: Dict[str, Any],
    dataset_name: str,
    iteration: int,
    ontology_type: str = "federated_primitive_tbox",
    namespace: str = "http://example.org/federated_tbox#",
    notes: Optional[str] = None,
) -> Optional[str]:
    """
    Save a TBox snapshot to LanceDB ontology tables.
    
    Args:
        tbox: TBox dictionary with 'classes' and 'data_properties'
        dataset_name: Dataset identifier for this ontology
        iteration: Iteration number (0 for initial, 1+ for iterations)
        ontology_type: Type of ontology
        namespace: OWL namespace
        notes: Additional notes
    
    Returns:
        ontology_id if successful, None otherwise
    """
    if not tbox:
        logger.warning("Cannot save snapshot: TBox is empty")
        return None
    
    try:
        store = get_store()
        ontology_mgr = store.ontology_manager
        class_mgr = ontology_mgr.class_mgr
        property_mgr = ontology_mgr.property_mgr
        
        # Generate ontology ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ontology_name = f"{dataset_name}_fedtbox_iter{iteration}"
        ontology_id = f"{ontology_name}_{timestamp}"
        
        # Extract counts
        classes = tbox.get("classes", [])
        data_properties = tbox.get("data_properties", [])
        
        # Create ontology metadata
        ontology_mgr.metadata_mgr.create_ontology(
            name=ontology_name,
            ontology_type=ontology_type,
            namespace=namespace,
            version=f"iter_{iteration}",
            domain=[dataset_name],
            dataset_name=dataset_name,
            notes=notes or f"Federated TBox iteration {iteration} snapshot",
            ontology_id=ontology_id,
        )
        
        # Save classes
        if classes:
            class_records = []
            for cls in classes:
                # Get parent from either parent_classes (list) or parent_class (str)
                parent_classes = cls.get("parent_classes", [])
                if not parent_classes:
                    parent = cls.get("parent_class", cls.get("parent", "owl:Thing"))
                    parent_classes = [parent] if parent else ["owl:Thing"]
                
                class_record = {
                    "class_name": cls.get("name", "Unknown"),
                    "label": cls.get("name", "Unknown"),  # Use name as label
                    "description": cls.get("description", cls.get("definition", "")),
                    "parent_classes": parent_classes,
                }
                class_records.append(class_record)
            
            class_mgr.add_classes_batch(ontology_id, class_records)
            logger.debug(f"  Saved {len(classes)} classes to snapshot")
        
        # Save data properties
        if data_properties:
            for dp in data_properties:
                # Support both "property_name" (from InitialDataProperty) and "name" legacy format
                prop_name = dp.get("property_name", dp.get("name", "unknown_property"))
                # Support both "domain" (list) and "domains" legacy format
                domain = dp.get("domain", dp.get("domains", []))
                # Support both "range_type" (from InitialDataProperty) and "range" legacy format
                range_val = dp.get("range_type", dp.get("range", "xsd:string"))
                # Support both "description" and "definition" legacy format
                description = dp.get("description", dp.get("definition", ""))
                # Get readout_template for human-readable summaries
                readout_template = dp.get("readout_template", "")
                # Get parent_properties for hierarchy
                parent_properties = dp.get("parent_properties", [])
                
                property_mgr.add_property(
                    ontology_id=ontology_id,
                    property_name=prop_name,
                    property_type="data",
                    label=prop_name,  # Use property name as label
                    domain=domain,
                    range=[range_val],  # range expects a list
                    description=description,
                    readout_template=readout_template,
                    parent_properties=parent_properties,
                )
            logger.debug(f"  Saved {len(data_properties)} data properties to snapshot")
        
        # Update counts
        ontology_mgr.metadata_mgr.update_counts(
            ontology_id=ontology_id,
            num_classes=len(classes),
            num_object_properties=0,
            num_data_properties=len(data_properties),
            num_axioms=0,
        )
        
        logger.info(f"✓ Saved TBox snapshot: {ontology_id} ({len(classes)} classes, {len(data_properties)} props)")
        return ontology_id
        
    except Exception as e:
        logger.error(f"Failed to save TBox snapshot: {e}")
        return None


def load_tbox_snapshot(
    dataset_name: str,
    iteration: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load the latest TBox snapshot for a dataset.
    
    Args:
        dataset_name: Dataset identifier
        iteration: Specific iteration to load (None = latest)
    
    Returns:
        TBox dictionary or None if not found
    """
    try:
        store = get_store()
        ontology_mgr = store.ontology_manager
        class_mgr = ontology_mgr.class_mgr
        property_mgr = ontology_mgr.property_mgr
        
        # Find matching ontologies
        # Query by domain containing dataset_name
        all_ontologies = ontology_mgr.list_ontologies(
            ontology_type="federated_primitive_tbox"
        )
        
        # Filter by dataset
        matching = [
            o for o in all_ontologies
            if dataset_name in (o.get("domain") or []) or o.get("dataset_name") == dataset_name
        ]
        
        if not matching:
            logger.info(f"No TBox snapshots found for dataset: {dataset_name}")
            return None
        
        # Filter by iteration if specified
        if iteration is not None:
            matching = [
                o for o in matching
                if o.get("version", "").endswith(f"iter_{iteration}")
            ]
        
        if not matching:
            logger.info(f"No TBox snapshot found for iteration {iteration}")
            return None
        
        # Get latest by created_at
        latest = max(matching, key=lambda o: o.get("created_at", datetime.min))
        ontology_id = latest["ontology_id"]
        
        logger.info(f"Loading TBox snapshot: {ontology_id}")
        
        # Load classes
        classes_data = class_mgr.get_classes_by_ontology(ontology_id)
        classes = []
        for cls in classes_data:
            classes.append({
                "name": cls.get("class_name"),
                "parent": cls.get("parent_classes", ["owl:Thing"])[0] if cls.get("parent_classes") else "owl:Thing",
                "definition": cls.get("description", ""),
                "external_alignments": cls.get("external_alignments", {}),
            })
        
        # Load data properties
        props_data = property_mgr.get_properties_by_ontology(ontology_id)
        data_properties = []
        for prop in props_data:
            if prop.get("property_type") == "data":
                data_properties.append({
                    "name": prop.get("property_name"),
                    "domains": prop.get("domain_classes", []),
                    "range": prop.get("range_type", "xsd:string"),
                    "definition": prop.get("description", ""),
                })
        
        tbox = {
            "classes": classes,
            "data_properties": data_properties,
        }
        
        logger.info(f"✓ Loaded TBox snapshot: {len(classes)} classes, {len(data_properties)} props")
        return tbox
        
    except Exception as e:
        logger.error(f"Failed to load TBox snapshot: {e}")
        return None

