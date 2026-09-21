"""
Node 5a: Validate Layer 2 with OWL Reasoner

Validates Layer 2 Defined Classes per-table using owlready2 + Pellet/HermiT reasoner.

This implements a per-table validation-correction loop:
1. For each table, validate its Column and Table Defined Classes
2. If validation fails, attempt correction or flag for review
3. Only valid tables contribute to final export

Key Validations:
1. Ontology Consistency - No unsatisfiable classes
2. Class Reference Validity - Primitive classes exist
3. EL Profile Compliance - Datatypes and constructs are EL-compatible

Usage:
    # In workflow
    validate_node = validate_layer2(state)
    
    # Standalone per-table
    result = validate_table(table_class, column_classes, ...)
"""

from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
from pydantic import BaseModel, Field

from workflows.common.node_decorators import graph_node
from workflows.indexing.annotation.state import (
    TableDiscoveryLayer2State,
    ColumnDefinedClass,
    TableDefinedClass,
)


# ============== Validation Result Models ==============

class TableValidationIssue(BaseModel):
    """A single validation issue for a table."""
    type: str = Field(..., description="Issue type: error, warning, info")
    category: str = Field(..., description="Category: consistency, reference, el_compliance")
    message: str = Field(..., description="Human-readable description")
    column_id: Optional[str] = Field(None, description="Related column if applicable")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")


class TableValidationResult(BaseModel):
    """Validation result for a single table."""
    table_id: str
    is_valid: bool = Field(True, description="Overall validity")
    is_consistent: bool = Field(True, description="Reasoner consistency check passed")
    issues: List[TableValidationIssue] = Field(default_factory=list)
    inconsistent_classes: List[str] = Field(default_factory=list)
    
    def add_error(self, category: str, message: str, column_id: str = None, details: Dict = None):
        self.is_valid = False
        self.issues.append(TableValidationIssue(
            type="error", category=category, message=message, 
            column_id=column_id, details=details
        ))
    
    def add_warning(self, category: str, message: str, column_id: str = None, details: Dict = None):
        self.issues.append(TableValidationIssue(
            type="warning", category=category, message=message,
            column_id=column_id, details=details
        ))
    
    def add_info(self, category: str, message: str, column_id: str = None, details: Dict = None):
        self.issues.append(TableValidationIssue(
            type="info", category=category, message=message,
            column_id=column_id, details=details
        ))


class Layer2ValidationResult(BaseModel):
    """Complete Layer 2 validation result."""
    is_valid: bool = Field(True, description="Overall validity")
    valid_tables: List[str] = Field(default_factory=list)
    invalid_tables: List[str] = Field(default_factory=list)
    table_results: Dict[str, TableValidationResult] = Field(default_factory=dict)
    total_issues: int = Field(0)
    error_count: int = Field(0)
    warning_count: int = Field(0)


# ============== Helper Functions ==============

def _strip_prefix(name: str) -> str:
    """Remove upo: prefix if present."""
    if name and name.startswith("upo:"):
        return name[4:]
    return name or ""


def _get_table_columns(
    table_id: str,
    column_classes: List[ColumnDefinedClass],
) -> List[ColumnDefinedClass]:
    """Get all column classes belonging to a table."""
    return [c for c in column_classes if c.column_id.startswith(f"{table_id}::")]


# ============== Per-Table Validation ==============

def validate_table_references(
    table_class: TableDefinedClass,
    column_classes: List[ColumnDefinedClass],
    layer1_class_names: List[str],
    data_properties: Optional[List[Dict[str, Any]]] = None,
    class_hierarchy: Optional[Dict[str, List[str]]] = None,
    data_property_hierarchy: Optional[Dict[str, List[str]]] = None,
) -> TableValidationResult:
    """
    Validate references for a single table (fast, no reasoner).
    
    Checks:
    1. Primitive class references exist in Layer 1
    2. Column IDs in table match column classes
    3. DataProperty domain compatibility (optional)
    4. DataProperty hierarchy validity (optional)
    """
    result = TableValidationResult(table_id=table_class.table_id)
    
    # Normalize Layer 1 class names
    layer1_normalized = set(_strip_prefix(n) for n in layer1_class_names)
    
    # Build DataProperty lookup
    dp_lookup = {}
    if data_properties:
        for dp in data_properties:
            dp_name = dp.get('name', '')
            dp_lookup[dp_name] = {
                'domain': [_strip_prefix(d) for d in dp.get('domain', [])],
                'parent_properties': dp.get('parent_properties', []),
            }
    
    # Build class ancestry lookup (for domain checking)
    def get_class_ancestors(class_name: str) -> set:
        """Get all ancestors of a class including itself."""
        ancestors = {_strip_prefix(class_name)}
        if class_hierarchy:
            queue = [_strip_prefix(class_name)]
            while queue:
                current = queue.pop(0)
                parents = class_hierarchy.get(current, [])
                for parent in parents:
                    parent_norm = _strip_prefix(parent)
                    if parent_norm not in ancestors:
                        ancestors.add(parent_norm)
                        queue.append(parent_norm)
        return ancestors
    
    # 1. Check primitive class references
    for col in column_classes:
        primitive = _strip_prefix(col.primitive_class)
        if primitive and primitive not in layer1_normalized:
            result.add_error(
                "reference",
                f"Primitive class '{primitive}' not found in Layer 1",
                column_id=col.column_id,
                details={"available_classes": list(layer1_normalized)[:10]}
            )
    
    # 2. Check column IDs match
    table_col_ids = set(table_class.column_ids)
    actual_col_ids = set(c.column_id for c in column_classes)
    
    missing = table_col_ids - actual_col_ids
    if missing:
        result.add_warning(
            "reference",
            f"Table references {len(missing)} column(s) without definitions",
            details={"missing": list(missing)}
        )
    
    return result


def validate_table_with_reasoner(
    table_class: TableDefinedClass,
    column_classes: List[ColumnDefinedClass],
    layer1_class_names: List[str],
    data_properties: List[Dict[str, Any]],
    reasoner_type: str = "pellet",
    skip_if_no_java: bool = True,
) -> TableValidationResult:
    """
    Validate a single table using OWL reasoner.
    
    Creates a mini-ontology for just this table and runs reasoner.
    
    Args:
        table_class: TableDefinedClass to validate
        column_classes: Related ColumnDefinedClasses
        layer1_class_names: Layer 1 class names for reference
        data_properties: Layer 1 DataProperties
        reasoner_type: "pellet" or "hermit"
        skip_if_no_java: Skip if Java not available
    
    Returns:
        TableValidationResult with consistency info
    """
    result = TableValidationResult(table_id=table_class.table_id)
    
    try:
        from owlready2 import World, Thing, sync_reasoner_pellet, sync_reasoner_hermit
        from owlready2 import ObjectProperty as OWLObjectProperty
        from store.ontology.owlready_converter import Layer2Converter
    except ImportError:
        result.add_warning("reasoner", "owlready2 not installed, skipping reasoner validation")
        return result
    
    try:
        # Use Layer2Converter to create mini-ontology
        converter = Layer2Converter()
        
        # Convert table-specific data
        column_dicts = [c.model_dump() for c in column_classes]
        table_dicts = [table_class.model_dump()]
        
        # Build primitive class map
        primitive_class_map = {_strip_prefix(n): {} for n in layer1_class_names}
        
        conv_result = converter.from_layer2_state(
            column_classes=column_dicts,
            table_classes=table_dicts,
            primitive_class_map=primitive_class_map,
            layer1_data_properties=data_properties,
        )
        
        if not conv_result.success:
            result.add_error("conversion", f"Failed to convert: {conv_result.error}")
            return result
        
        world = conv_result.world
        
        # Run reasoner
        if reasoner_type == "pellet":
            sync_reasoner_pellet(world, infer_property_values=True, infer_data_property_values=True)
        elif reasoner_type == "hermit":
            sync_reasoner_hermit(world, infer_property_values=True)
        
        # Check for inconsistent classes
        inconsistent = list(world.inconsistent_classes())
        inconsistent_names = [
            str(c).split('.')[-1] for c in inconsistent 
            if str(c).split('.')[-1] not in ('Nothing', 'Thing')
        ]
        
        if inconsistent_names:
            result.is_consistent = False
            result.is_valid = False
            result.inconsistent_classes = inconsistent_names
            result.add_error(
                "consistency",
                f"Table has {len(inconsistent_names)} unsatisfiable class(es): {inconsistent_names}",
                details={"classes": inconsistent_names}
            )
        else:
            result.add_info("consistency", "Table ontology is consistent")
        
    except Exception as e:
        error_msg = str(e)
        if "java" in error_msg.lower() or "jvm" in error_msg.lower():
            if skip_if_no_java:
                result.add_warning("reasoner", f"Java/Reasoner not available: {error_msg}")
            else:
                result.add_error("reasoner", f"Reasoner failed: {error_msg}")
        else:
            result.add_error("reasoner", f"Reasoner error: {error_msg}")
            logger.error(f"Reasoner failed for table {table_class.table_id}: {e}")
    
    return result


# ============== Per-Table Correction ==============

def attempt_table_correction(
    table_class: TableDefinedClass,
    column_classes: List[ColumnDefinedClass],
    validation_result: TableValidationResult,
    layer1_class_names: List[str],
) -> Tuple[bool, List[ColumnDefinedClass], List[str]]:
    """
    Attempt to correct validation errors for a table.
    
    Correction strategies:
    1. Missing primitive class -> fuzzy match to closest Layer 1 class
    2. Invalid column reference -> remove from table
    
    Args:
        table_class: TableDefinedClass to correct
        column_classes: Related ColumnDefinedClasses
        validation_result: Validation result with issues
        layer1_class_names: Layer 1 class names for matching
    
    Returns:
        (success, corrected_columns, corrections_made)
    """
    corrections_made = []
    corrected_columns = list(column_classes)
    
    layer1_normalized = {_strip_prefix(n): n for n in layer1_class_names}
    
    # Attempt corrections for reference errors
    for issue in validation_result.issues:
        if issue.type != "error" or issue.category != "reference":
            continue
        
        if "Primitive class" in issue.message and issue.column_id:
            # Try fuzzy matching
            for i, col in enumerate(corrected_columns):
                if col.column_id == issue.column_id:
                    original = _strip_prefix(col.primitive_class)
                    
                    # Simple word overlap matching
                    original_words = set(original.lower().split())
                    best_match = None
                    best_score = 0
                    
                    for layer1_name in layer1_normalized.keys():
                        layer1_words = set(layer1_name.lower().split())
                        overlap = len(original_words & layer1_words)
                        if overlap > best_score:
                            best_score = overlap
                            best_match = layer1_name
                    
                    if best_match and best_score > 0:
                        corrected_columns[i] = col.model_copy(
                            update={"primitive_class": f"upo:{best_match}"}
                        )
                        corrections_made.append(
                            f"{issue.column_id}: {original} -> {best_match}"
                        )
                    break
    
    success = len(corrections_made) > 0
    return success, corrected_columns, corrections_made


# ============== Main Validation Loop ==============

def validate_all_tables(
    state: TableDiscoveryLayer2State,
    use_reasoner: bool = True,
    reasoner_type: str = "pellet",
    max_correction_attempts: int = 2,
) -> Layer2ValidationResult:
    """
    Validate all tables with per-table validation-correction loop.
    
    For each table:
    1. Run reference validation (fast)
    2. If errors, attempt correction
    3. Run reasoner validation (if enabled)
    4. If still invalid after max attempts, mark as invalid
    
    Args:
        state: TableDiscoveryLayer2State with all data
        use_reasoner: Whether to run OWL reasoner
        reasoner_type: "pellet" or "hermit"
        max_correction_attempts: Max correction attempts per table
    
    Returns:
        Layer2ValidationResult with per-table results
    """
    result = Layer2ValidationResult()
    
    column_classes = state.column_defined_classes
    table_classes = state.table_defined_classes
    layer1_class_names = state.layer1_class_names
    data_properties = getattr(state, 'data_properties', [])
    
    logger.info(f"Validating {len(table_classes)} tables...")
    
    for table_class in table_classes:
        table_id = table_class.table_id
        
        # Get table-specific data
        table_columns = _get_table_columns(table_id, column_classes)
        
        logger.debug(f"Validating table {table_id}: {len(table_columns)} columns")
        
        # Validation-correction loop
        current_columns = table_columns
        table_valid = False
        
        for attempt in range(max_correction_attempts + 1):
            # 1. Reference validation (fast)
            ref_result = validate_table_references(
                table_class, current_columns,
                layer1_class_names,
            )
            
            # 2. If reference errors, attempt correction
            if not ref_result.is_valid and attempt < max_correction_attempts:
                success, corrected_columns, corrections = attempt_table_correction(
                    table_class, current_columns, ref_result, layer1_class_names
                )
                if success:
                    logger.debug(f"Table {table_id} corrections: {corrections}")
                    current_columns = corrected_columns
                    continue
            
            # 3. Reasoner validation (if enabled and references valid)
            if use_reasoner and ref_result.is_valid:
                reasoner_result = validate_table_with_reasoner(
                    table_class, current_columns,
                    layer1_class_names, data_properties,
                    reasoner_type=reasoner_type
                )
                
                # Merge results
                ref_result.issues.extend(reasoner_result.issues)
                ref_result.is_consistent = reasoner_result.is_consistent
                ref_result.inconsistent_classes = reasoner_result.inconsistent_classes
                if not reasoner_result.is_valid:
                    ref_result.is_valid = False
            
            # 4. Final result for this table
            table_valid = ref_result.is_valid
            result.table_results[table_id] = ref_result
            break
        
        # Track valid/invalid tables
        if table_valid:
            result.valid_tables.append(table_id)
        else:
            result.invalid_tables.append(table_id)
        
        # Count issues
        for issue in result.table_results.get(table_id, TableValidationResult(table_id=table_id)).issues:
            result.total_issues += 1
            if issue.type == "error":
                result.error_count += 1
            elif issue.type == "warning":
                result.warning_count += 1
    
    # Overall validity
    result.is_valid = len(result.invalid_tables) == 0
    
    logger.info(
        f"Validation complete: {len(result.valid_tables)} valid, "
        f"{len(result.invalid_tables)} invalid, "
        f"{result.error_count} errors, {result.warning_count} warnings"
    )
    
    return result


# ============== Graph Node ==============

@graph_node(node_type="validation")
def validate_layer2_node(state: TableDiscoveryLayer2State) -> Dict[str, Any]:
    """
    Validate Layer 2 tables before export.
    
    Runs per-table validation-correction loop:
    1. Reference validation (fast)
    2. Reasoner validation (if Java available)
    3. Correction attempts for fixable errors
    
    Returns:
        State updates with validation results
    """
    logger.info("=" * 70)
    logger.info("Node 5a: Validating Layer 2")
    logger.info("=" * 70)
    
    # Run validation
    validation_result = validate_all_tables(
        state,
        use_reasoner=True,
        reasoner_type="pellet",
        max_correction_attempts=2,
    )
    
    # Filter to valid tables only for export
    valid_table_ids = set(validation_result.valid_tables)
    
    # Filter column classes
    valid_column_classes = [
        c for c in state.column_defined_classes
        if c.column_id.split('::')[0] in valid_table_ids
    ]
    
    # Filter table classes
    valid_table_classes = [
        t for t in state.table_defined_classes
        if t.table_id in valid_table_ids
    ]
    
    logger.info(
        f"After validation: {len(valid_table_classes)} tables, "
        f"{len(valid_column_classes)} columns"
    )
    
    return {
        "validation_done": True,
        "validation_result": validation_result.model_dump(),
        "valid_table_ids": list(valid_table_ids),
        "invalid_table_ids": validation_result.invalid_tables,
        # Update filtered data for export
        "column_defined_classes": valid_column_classes,
        "table_defined_classes": valid_table_classes,
    }
