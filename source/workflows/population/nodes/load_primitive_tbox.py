"""
Node 1: Load Primitive TBox

Loads Layer 1 Primitive TBox classes and DataProperties from LanceDB (Stage 1 output).
"""

from typing import Dict, Any, List
from loguru import logger
import numpy as np

from workflows.common.node_decorators import graph_node
from workflows.population.state import ColumnSummaryState


def _normalize_value(val, default=''):
    """Normalize values from LanceDB (handles numpy arrays, lists, etc.)."""
    if val is None:
        return default
    if isinstance(val, np.ndarray):
        if len(val) > 0:
            return str(val[0])
        return default
    if isinstance(val, (list, tuple)):
        return val[0] if val else default
    return str(val)


def _normalize_list(val):
    """Normalize list values from LanceDB, filtering out null/empty values."""
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        # Filter out null/empty values
        return [str(x) for x in val if x and str(x).lower() != 'null']
    if isinstance(val, str):
        # Filter out null/empty values
        return [d.strip() for d in val.split(',') if d.strip() and d.strip().lower() != 'null']
    if isinstance(val, (list, tuple)):
        # Filter out null/empty values
        return [str(x) for x in val if x and str(x).lower() != 'null']
    if val and str(val).lower() != 'null':
        return [str(val)]
    return []


def _load_from_lancedb(dataset_name: str, iteration: int = -1) -> tuple[List[str], List[Dict[str, Any]], Dict, Dict, str, List[Dict]]:
    """
    Load primitive classes and DataProperties from LanceDB (Stage 1 output).
    
    Supports both legacy "primitive_tbox" and new "federated_primitive_tbox" types.
    Prioritizes federated_primitive_tbox if both exist.
    
    Args:
        dataset_name: Name of the dataset
        iteration: Iteration number to load (-1 = latest)
    
    Returns:
        Tuple of (class_names, data_properties, class_hierarchy, dp_hierarchy, ontology_id, primitive_classes_full)
        
    Raises:
        RuntimeError if no primitive TBox is found for the dataset
    """
    from store.store_singleton import get_store
    from store.ontology.ontology_table import (
        OntologyTableManager,
        OntologyClassTableManager,
        OntologyPropertyTableManager,
    )
    
    store = get_store()
    
    # Find the latest primitive_tbox ontology for this dataset
    # Try federated_primitive_tbox first (new mode), then legacy primitive_tbox
    meta_mgr = OntologyTableManager(store.db)
    
    # Use iteration-aware query
    if iteration == -1:
        # Get latest iteration
        ontology = meta_mgr.get_version_by_iteration("federated_primitive_tbox", dataset_name, -1)
    else:
        # Get specific iteration
        ontology = meta_mgr.get_version_by_iteration("federated_primitive_tbox", dataset_name, iteration)
    
    if not ontology:
        # Fallback to legacy primitive_tbox (no iteration support for legacy)
        ontology = meta_mgr.get_latest_version("primitive_tbox", dataset_name)
    
    if not ontology:
        raise RuntimeError(
            f"No primitive_tbox found for dataset '{dataset_name}'. "
            f"Please run Stage 1 (federated_primitive_tbox) first: "
            f"python demos/run_upo_pipeline.py --step federated_primitive_tbox"
        )
    
    ontology_id = ontology['ontology_id']
    version = ontology.get('version', 'unknown')
    logger.info(f"Loading TBox: {ontology_id} (version={version}, requested_iteration={iteration})")
    
    # Load classes from ontology_classes table
    class_mgr = OntologyClassTableManager(store.db)
    class_records = class_mgr.get_classes_by_ontology(ontology_id)
    
    class_names = [rec['class_name'] for rec in class_records if rec.get('class_name')]
    
    # Build full class info (with description)
    primitive_classes_full = []
    for rec in class_records:
        if rec.get('class_name'):
            primitive_classes_full.append({
                'name': rec['class_name'],
                'description': rec.get('description', '') or '',
                'parent_classes': _normalize_list(rec.get('parent_classes', [])),
            })
    
    # Load DataProperties from ontology_properties table
    prop_mgr = OntologyPropertyTableManager(store.db)
    prop_records = prop_mgr.get_properties_by_ontology(ontology_id)
    
    # Build class hierarchy (child -> parent mapping)
    class_hierarchy = {}
    for rec in class_records:
        class_name = rec.get('class_name', '')
        parent_classes = _normalize_list(rec.get('parent_classes', []))
        # Normalize class name (remove upo: prefix for consistency)
        class_name_normalized = class_name.replace('upo:', '')
        if parent_classes:
            class_hierarchy[class_name_normalized] = parent_classes
    
    # Build DataProperty hierarchy (child -> parent mapping)
    data_property_hierarchy = {}
    
    # Filter for data properties and build lookup structure
    data_properties = []
    for prop in prop_records:
        if prop.get('property_type') == 'data':
            # Normalize domain
            domain = _normalize_list(prop.get('domain', []))
            
            # Normalize range_type
            range_type = _normalize_value(prop.get('range'), 'xsd:string')
            
            # Normalize readout_template
            readout_template = _normalize_value(prop.get('readout_template'), '')
            
            # Get parent properties
            parent_properties = _normalize_list(prop.get('parent_properties', []))
            
            prop_name = prop.get('property_name', '')
            
            data_properties.append({
                'name': prop_name,
                'domain': domain,
                'range_type': range_type,
                'comment': prop.get('description', ''),
                'readout_template': readout_template,
                'parent_properties': parent_properties,
            })
            
            # Build hierarchy
            if parent_properties:
                data_property_hierarchy[prop_name] = parent_properties
    
    logger.info(f"Loaded {len(class_names)} primitive classes, {len(data_properties)} DataProperties from LanceDB: {ontology_id}")
    return class_names, data_properties, class_hierarchy, data_property_hierarchy, ontology_id, primitive_classes_full


@graph_node(node_type="processing")
def load_primitive_tbox_node(state: ColumnSummaryState) -> Dict[str, Any]:
    """
    Load primitive classes and DataProperties from Stage 1 output (LanceDB).
    
    Requires Stage 1 to have been run first.
    Optionally clears the transform contract repository if state.fresh_start is True.
    
    Returns:
        Updated state with primitive_classes list and data_properties
    """
    from datetime import datetime
    from workflows.population.transform_repository import TransformRepository
    
    logger.info("Loading Primitive TBox classes and DataProperties...")
    
    start_time = datetime.now()
    
    # Only clear existing transform contracts if fresh_start is requested
    # This allows incremental runs to reuse previously generated contracts
    repo = TransformRepository(dataset_name=state.dataset_name)
    if state.fresh_start:
        repo.clear_all()
        logger.info("  Fresh start: cleared Transform Contracts repository")
    else:
        logger.debug("  Incremental mode: reusing existing Transform Contracts")
    
    # Get iteration parameter from state (default -1 for latest)
    tbox_iteration = getattr(state, 'tbox_iteration', -1)
    
    # Load from LanceDB (Stage 1 output) - required (do this first to get class_hierarchy)
    primitive_classes, data_properties, class_hierarchy, data_property_hierarchy, ontology_id, primitive_classes_full = _load_from_lancedb(
        state.dataset_name, iteration=tbox_iteration
    )
    
    logger.info(f"✓ Loaded from Stage 1: {len(primitive_classes)} classes, {len(data_properties)} DataProperties")
    logger.info(f"Primitive classes: {primitive_classes[:10]}{'...' if len(primitive_classes) > 10 else ''}")
    
    return {
        "primitive_classes": primitive_classes,
        "primitive_classes_full": primitive_classes_full,
        "data_properties": data_properties,
        "class_hierarchy": class_hierarchy,
        "data_property_hierarchy": data_property_hierarchy,
        "primitive_tbox_ontology_id": ontology_id,
        "start_time": start_time,
    }
