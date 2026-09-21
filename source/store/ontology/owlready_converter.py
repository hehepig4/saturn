"""
OWLReady2 Converter - Convert PrimitiveTBoxState to owlready2 ontology.

This module provides a reusable converter that transforms the state representation
of a Primitive TBox (Dict-based) into an owlready2 ontology for:
1. Validation with Pellet/HermiT reasoner
2. Export to OWL/RDF formats
3. Consistency checking

Key Features:
- Complete field support (annotations, comments, labels, descriptions)
- Cycle-safe SubClassOf handling
- Disjoint axiom support
- Property characteristics (functional, transitive, reflexive, etc.)
- Isolated World per conversion (thread-safe)

Usage:
    from store.ontology.owlready_converter import OwlreadyConverter
    
    converter = OwlreadyConverter()
    result = converter.from_primitive_state(state)
    
    if result.success:
        # Use result.world, result.onto, result.class_map
        sync_reasoner_pellet(result.world)
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger

from owlready2 import (
    World, Thing, Nothing,
    ObjectProperty as OWLObjectProperty,
    DataProperty as OWLDataProperty,
    FunctionalProperty,
    AllDisjoint,
    comment, label,
)


def _strip_prefix(name: str) -> str:
    """Remove common prefixes (upo:, owl:, etc.)."""
    if not name:
        return ""
    for prefix in ["upo:", "owl:", "rdfs:", "rdf:", "xsd:"]:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


@dataclass
class ConversionResult:
    """Result of converting state to owlready2 ontology."""
    
    success: bool = False
    error: Optional[str] = None
    
    # owlready2 objects
    world: Optional[World] = None
    onto: Any = None  # Ontology object
    
    # Mappings for external use
    class_map: Dict[str, Any] = field(default_factory=dict)  # name -> OWL class
    object_property_map: Dict[str, Any] = field(default_factory=dict)  # name -> OWL ObjectProperty
    data_property_map: Dict[str, Any] = field(default_factory=dict)  # name -> OWL DataProperty
    
    # Conversion warnings/info
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    
    # Statistics
    class_count: int = 0
    object_property_count: int = 0
    data_property_count: int = 0
    axiom_count: int = 0


class OwlreadyConverter:
    """
    Converts PrimitiveTBoxState to owlready2 ontology.
    
    Thread-safe: Each conversion creates an isolated World.
    
    Attributes:
        namespace: Default namespace for the ontology
    """
    
    DEFAULT_NAMESPACE = "http://example.org/saturn/upo#"
    
    def __init__(self, namespace: str = None):
        """
        Initialize converter.
        
        Args:
            namespace: OWL namespace URI (default: http://example.org/saturn/upo#)
        """
        self.namespace = namespace or self.DEFAULT_NAMESPACE
    
    def from_primitive_state(
        self,
        state: Any,  # PrimitiveTBoxState or dict-like
        include_annotations: bool = True,
        include_descriptions: bool = True,
    ) -> ConversionResult:
        """
        Convert PrimitiveTBoxState to owlready2 ontology.
        
        Args:
            state: PrimitiveTBoxState with primitive_classes, object_properties, etc.
            include_annotations: Include rdfs:comment, rdfs:label, etc.
            include_descriptions: Include description/comment annotations
        
        Returns:
            ConversionResult with world, onto, and mappings
        """
        result = ConversionResult()
        
        try:
            # Create isolated world
            result.world = World()
            result.onto = result.world.get_ontology(self.namespace)
            
            with result.onto:
                # Step 1: Create base classes (Column, Table)
                self._create_base_classes(result)
                
                # Step 2: Create all primitive classes (without parents first)
                primitive_classes = getattr(state, 'primitive_classes', None) or []
                if isinstance(state, dict):
                    primitive_classes = state.get('primitive_classes', [])
                
                self._create_primitive_classes(
                    result, 
                    primitive_classes,
                    include_annotations,
                    include_descriptions,
                )
                
                # Step 3: Set up class hierarchy (SubClassOf)
                self._setup_class_hierarchy(result, primitive_classes)
                
                # Step 4: Set up disjoint axioms
                disjoint_axioms = getattr(state, 'disjoint_axioms', None) or []
                if isinstance(state, dict):
                    disjoint_axioms = state.get('disjoint_axioms', [])
                
                self._setup_disjoint_axioms(result, disjoint_axioms)
                
                # Step 5: Create hasColumn object property
                self._create_has_column_property_for_layer1(result)
                
                # Step 6: Create data properties
                data_properties = getattr(state, 'data_properties', None) or []
                if isinstance(state, dict):
                    data_properties = state.get('data_properties', [])
                
                self._create_data_properties(
                    result,
                    data_properties,
                    include_annotations,
                    include_descriptions,
                )
            
            # Update statistics
            result.class_count = len(result.class_map)
            result.object_property_count = len(result.object_property_map)
            result.data_property_count = len(result.data_property_map)
            result.success = True
            
            logger.debug(
                f"Converted to owlready2: {result.class_count} classes, "
                f"{result.object_property_count} object props, "
                f"{result.data_property_count} data props"
            )
            
        except Exception as e:
            import traceback
            result.success = False
            result.error = str(e)
            logger.error(f"Failed to convert to owlready2: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        
        return result
    
    def _create_base_classes(self, result: ConversionResult) -> None:
        """Create base classes: Column, Table, and base property hasValue."""
        onto = result.onto
        
        # Column - base class for all column types
        Column = type('Column', (Thing,), {
            "namespace": onto,
        })
        Column.comment = ["A column in a tabular data structure."]
        Column.label = ["Column"]
        result.class_map['Column'] = Column
        
        # Table - represents a table
        Table = type('Table', (Thing,), {
            "namespace": onto,
        })
        Table.comment = ["A tabular data structure with columns and rows."]
        Table.label = ["Table"]
        result.class_map['Table'] = Table
        
        # hasValue - base data property for all column values
        hasValue = type('hasValue', (OWLDataProperty,), {
            "namespace": onto,
        })
        hasValue.comment = ["Abstract base property for all column values."]
        hasValue.label = ["has value"]
        hasValue.domain = [Column]
        hasValue.range = [str]  # rdfs:Literal maps to str in owlready2
        result.data_property_map['hasValue'] = hasValue
    
    def _create_primitive_classes(
        self,
        result: ConversionResult,
        primitive_classes: List[Dict[str, Any]],
        include_annotations: bool,
        include_descriptions: bool,
    ) -> None:
        """Create all primitive classes (first pass, no parents yet)."""
        onto = result.onto
        
        for cls_data in primitive_classes:
            cls_name = _strip_prefix(cls_data.get('name', ''))
            if not cls_name or cls_name in result.class_map:
                continue
            
            # Create class with Thing as parent (will set real parents later)
            owl_cls = type(cls_name, (Thing,), {"namespace": onto})
            
            # Add annotations
            if include_annotations:
                # rdfs:label
                cls_label = cls_data.get('label', cls_name)
                if cls_label:
                    owl_cls.label = [cls_label]
                
                # rdfs:comment (description)
                if include_descriptions:
                    description = cls_data.get('description', '')
                    if description:
                        owl_cls.comment = [description]
                
                # Additional annotations from examples
                examples = cls_data.get('examples', [])
                if examples:
                    # Store as custom annotation (owlready2 supports this)
                    # We'll use comment for now since custom annotations are complex
                    if owl_cls.comment:
                        owl_cls.comment.append(f"Examples: {', '.join(examples[:5])}")
                    else:
                        owl_cls.comment = [f"Examples: {', '.join(examples[:5])}"]
            
            result.class_map[cls_name] = owl_cls
    
    def _setup_class_hierarchy(
        self,
        result: ConversionResult,
        primitive_classes: List[Dict[str, Any]],
    ) -> None:
        """Set up SubClassOf relationships using is_a.append()."""
        Column = result.class_map.get('Column', Thing)
        
        for cls_data in primitive_classes:
            cls_name = _strip_prefix(cls_data.get('name', ''))
            if not cls_name or cls_name not in result.class_map:
                continue
            
            owl_cls = result.class_map[cls_name]
            parent_names = [_strip_prefix(p) for p in cls_data.get('parent_classes', [])]
            
            has_valid_parent = False
            for p_name in parent_names:
                if p_name == cls_name:
                    result.warnings.append(f"Class '{cls_name}' has self-reference, skipped")
                    continue
                
                if p_name in result.class_map:
                    try:
                        owl_cls.is_a.append(result.class_map[p_name])
                        has_valid_parent = True
                    except TypeError as e:
                        # Cyclic SubClassOf - OWL allows this (implies equivalence)
                        result.info.append(
                            f"Cyclic SubClassOf: {cls_name} ⊑ {p_name} (treated as equivalent by reasoner)"
                        )
            
            # If no valid parent, default to Column
            if not has_valid_parent and cls_name not in ('Column', 'Table'):
                try:
                    owl_cls.is_a.append(Column)
                except TypeError:
                    pass
    
    def _setup_disjoint_axioms(
        self,
        result: ConversionResult,
        disjoint_axioms: List[Dict[str, Any]],
    ) -> None:
        """Set up DisjointClasses axioms."""
        for axiom in disjoint_axioms:
            classes = axiom.get('classes', [])
            owl_classes = []
            
            for c in classes:
                c_name = _strip_prefix(c)
                if c_name in result.class_map:
                    owl_classes.append(result.class_map[c_name])
            
            if len(owl_classes) >= 2:
                try:
                    AllDisjoint(owl_classes)
                    result.axiom_count += 1
                except Exception as e:
                    result.warnings.append(f"Failed to create DisjointClasses: {e}")
    
    def _create_has_column_property_for_layer1(self, result: ConversionResult) -> None:
        """Create hasColumn object property for Layer 1 ontology."""
        onto = result.onto
        hasColumn = type('hasColumn', (OWLObjectProperty,), {"namespace": onto})
        hasColumn.domain = [result.class_map.get('Table', Thing)]
        hasColumn.range = [result.class_map.get('Column', Thing)]
        hasColumn.label = ["has column"]
        hasColumn.comment = ["Relates a Table to its Columns."]
        result.object_property_map['hasColumn'] = hasColumn

    def _create_data_properties(
        self,
        result: ConversionResult,
        data_properties: List[Dict[str, Any]],
        include_annotations: bool,
        include_descriptions: bool,
    ) -> None:
        """Create all data properties."""
        from core.datatypes.el_datatypes import XSD_TO_PYTHON
        
        onto = result.onto
        
        # Use unified XSD type mapping from core.datatypes
        xsd_type_map = XSD_TO_PYTHON
        
        # DEBUG: Check for duplicate property names
        prop_names = [_strip_prefix(p.get('name', '')) for p in data_properties]
        unique_names = set(prop_names)
        if len(prop_names) != len(unique_names):
            duplicates = [n for n in prop_names if prop_names.count(n) > 1]
            logger.warning(f"Found {len(prop_names) - len(unique_names)} duplicate property names: {set(duplicates)}")
        
        created_count = 0
        for prop_data in data_properties:
            prop_name = _strip_prefix(prop_data.get('name', ''))
            if not prop_name or prop_name in result.data_property_map:
                continue
            
            # Check for functional characteristic
            base_classes = [OWLDataProperty]
            is_functional = prop_data.get('functional', False)
            if is_functional:
                base_classes.append(FunctionalProperty)
            
            # Create property with error handling
            try:
                # Check if property already exists in this world (from another ontology)
                existing_in_world = onto.world.search(iri=f"*{prop_name}")
                if existing_in_world:
                    logger.warning(f"Property '{prop_name}' already exists in world: {existing_in_world}")
                    # Reuse existing property instead of creating new one
                    owl_prop = existing_in_world[0]
                    logger.warning(f"  Reusing existing property: {owl_prop}")
                else:
                    owl_prop = type(prop_name, tuple(base_classes), {"namespace": onto})
                created_count += 1
            except TypeError as e:
                # Log detailed info for debugging
                logger.error(f"Failed to create data property '{prop_name}' (#{created_count}): {e}")
                logger.error(f"  base_classes: {base_classes}")
                logger.error(f"  prop_data: {prop_data}")
                logger.error(f"  is_functional: {is_functional}")
                logger.error(f"  onto name: {onto.name}")
                logger.error(f"  onto world: {onto.world}")
                # DEBUG: Check previous properties in result.data_property_map
                logger.error(f"  Previously created {len(result.data_property_map)} data properties")
                # DEBUG: Check if property already exists
                existing = getattr(onto, prop_name, None)
                logger.error(f"  Existing property with same name: {existing}")
                # DEBUG: Check all owlready2 internal state
                import traceback
                logger.error(f"  Full traceback:\n{traceback.format_exc()}")
                # Try without FunctionalProperty
                logger.warning(f"  Attempting fallback: creating without FunctionalProperty...")
                try:
                    owl_prop = type(prop_name, (OWLDataProperty,), {"namespace": onto})
                    logger.warning(f"  Fallback succeeded: {owl_prop}")
                    created_count += 1
                except Exception as e2:
                    logger.error(f"  Fallback also failed: {e2}")
                    raise e  # Re-raise original error
            
            # Set domain
            domains = prop_data.get('domain', [])
            if domains:
                owl_prop.domain = [
                    result.class_map.get(_strip_prefix(d), Thing) for d in domains
                ]
            
            # Set range (datatype)
            range_type = prop_data.get('range_type', 'xsd:string')
            python_type = xsd_type_map.get(range_type, str)
            owl_prop.range = [python_type]
            
            # Add annotations
            if include_annotations:
                prop_label = prop_data.get('label', prop_name)
                if prop_label:
                    owl_prop.label = [prop_label]
                
                if include_descriptions:
                    description = prop_data.get('description', '')
                    if description:
                        owl_prop.comment = [description]
                    
                    # Add readout_template info
                    readout = prop_data.get('readout_template', '')
                    if readout:
                        if owl_prop.comment:
                            owl_prop.comment.append(f"Readout: {readout}")
                        else:
                            owl_prop.comment = [f"Readout: {readout}"]
                    
                    # Add statistics_requirements
                    stats_req = prop_data.get('statistics_requirements', [])
                    if stats_req:
                        stats_str = ', '.join(stats_req[:5])
                        if owl_prop.comment:
                            owl_prop.comment.append(f"Stats: {stats_str}")
                        else:
                            owl_prop.comment = [f"Stats: {stats_str}"]
            
            result.data_property_map[prop_name] = owl_prop
        
        # Setup property hierarchy (SubPropertyOf) after all properties created
        self._setup_data_property_hierarchy(result, data_properties)
    
    def _setup_data_property_hierarchy(
        self,
        result: ConversionResult,
        data_properties: List[Dict[str, Any]],
    ) -> None:
        """Set up SubPropertyOf relationships for data properties.
        
        Root data properties (those without parent_properties) are automatically
        set as SubPropertyOf hasValue.
        """
        hasValue = result.data_property_map.get('hasValue')
        
        for prop_data in data_properties:
            prop_name = _strip_prefix(prop_data.get('name', ''))
            parent_props = prop_data.get('parent_properties', [])
            
            if prop_name not in result.data_property_map or prop_name == 'hasValue':
                continue
            
            owl_prop = result.data_property_map[prop_name]
            has_valid_parent = False
            
            for parent_name in parent_props:
                parent_name = _strip_prefix(parent_name)
                parent_owl = result.data_property_map.get(parent_name)
                
                if parent_owl and parent_owl != owl_prop:
                    try:
                        # Add parent to is_a list
                        if parent_owl not in owl_prop.is_a:
                            owl_prop.is_a.append(parent_owl)
                            has_valid_parent = True
                    except Exception as e:
                        logger.warning(f"Failed to set SubPropertyOf {prop_name} -> {parent_name}: {e}")
            
            # If no valid parent, default to hasValue
            if not has_valid_parent and hasValue:
                try:
                    if hasValue not in owl_prop.is_a:
                        owl_prop.is_a.append(hasValue)
                except Exception as e:
                    logger.warning(f"Failed to set SubPropertyOf {prop_name} -> hasValue: {e}")


# ============== Layer 2 Support ==============

class Layer2Converter(OwlreadyConverter):
    """
    Extends OwlreadyConverter for Layer 2 Defined Classes export.
    
    Layer 2 includes:
    - Column Defined Classes (subclasses of Primitive Classes)
    - Table Defined Classes (complex class definitions)
    - Discovered ObjectProperty relationships
    """
    
    LAYER2_ONTOLOGY_IRI = "http://example.org/saturn/upo/layer2"
    LAYER1_ONTOLOGY_IRI = "http://example.org/saturn/upo/layer1"
    
    def __init__(self, namespace: str = None):
        super().__init__(namespace or self.DEFAULT_NAMESPACE)
    
    def from_layer2_state(
        self,
        column_classes: List[Dict],
        table_classes: List[Dict],
        primitive_class_map: Dict[str, Any] = None,
        layer1_data_properties: List[Dict] = None,
        timestamp: str = None,
    ) -> ConversionResult:
        """
        Convert Layer 2 state to owlready2 ontology.
        
        Args:
            column_classes: List of ColumnDefinedClass dicts/objects
            table_classes: List of TableDefinedClass dicts/objects
            primitive_class_map: Map of primitive class names to their descriptions (from Layer 1)
            layer1_data_properties: List of Layer 1 DataProperty dicts for readout templates
            timestamp: Ontology creation timestamp
            
        Returns:
            ConversionResult with ontology ready for export
        """
        from datetime import datetime
        
        result = ConversionResult()
        
        # Build data property index by primitive class domain
        self._layer1_data_property_by_class = {}
        if layer1_data_properties:
            for dp in layer1_data_properties:
                for domain_class in dp.get('domain', []):
                    self._layer1_data_property_by_class[domain_class] = dp
        
        try:
            result.world = World()
            
            # Create Layer 1 import ontology BEFORE entering 'with onto:' block
            layer1_onto = result.world.get_ontology(self.LAYER1_ONTOLOGY_IRI)
            
            # Create Layer 2 ontology
            result.onto = result.world.get_ontology(self.LAYER2_ONTOLOGY_IRI)
            
            # Add import for Layer 1 (outside of 'with' block)
            result.onto.imported_ontologies.append(layer1_onto)
            
            with result.onto:
                # Reference base classes from Layer 1 (Column, Table)
                # These are defined in Layer 1, we only create placeholders for internal use
                self._create_layer2_base_class_refs(result, layer1_onto)
                
                # Create hasColumn object property
                self._create_has_column_property(result)
                
                # Create Column Defined Classes with readout template support
                self._create_column_defined_classes(
                    result, column_classes, primitive_class_map
                )
                
                # Create Table Defined Classes
                self._create_table_defined_classes(result, table_classes)
            
            result.class_count = len(result.class_map)
            result.object_property_count = len(result.object_property_map)
            result.success = True
            
            logger.info(
                f"Layer 2 converted: {result.class_count} classes, "
                f"{result.object_property_count} object properties"
            )
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error(f"Failed to convert Layer 2: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def _create_layer2_base_class_refs(self, result: ConversionResult, layer1_onto) -> None:
        """
        Create references to base classes from Layer 1.
        
        In Layer 2, we don't redefine Column and Table - they come from Layer 1
        via owl:imports. We just create local references for internal use.
        
        The actual import relationship is handled by OWL semantics.
        """
        onto = result.onto
        
        # Create Column class reference - this will be used as parent for primitive classes
        # Note: We mark it clearly as imported from Layer 1
        Column = type('Column', (Thing,), {"namespace": onto})
        Column.comment = ["(Imported from Layer 1) A column in a tabular data structure."]
        Column.label = ["Column"]
        result.class_map['Column'] = Column
        
        # Create Table class reference
        Table = type('Table', (Thing,), {"namespace": onto})
        Table.comment = ["(Imported from Layer 1) A tabular data structure with columns and rows."]
        Table.label = ["Table"]
        result.class_map['Table'] = Table
    
    def _create_has_column_property(self, result: ConversionResult) -> None:
        """Create hasColumn object property to relate Tables to Columns."""
        onto = result.onto
        Table = result.class_map.get('Table', Thing)
        Column = result.class_map.get('Column', Thing)
        
        hasColumn = type('hasColumn', (OWLObjectProperty,), {"namespace": onto})
        hasColumn.domain = [Table]
        hasColumn.range = [Column]
        hasColumn.label = ["has column"]
        hasColumn.comment = ["Relates a Table to any of its Columns."]
        result.object_property_map['hasColumn'] = hasColumn
    
    def _create_column_defined_classes(
        self,
        result: ConversionResult,
        column_classes: List,
        primitive_class_map: Dict[str, Any] = None,
    ) -> None:
        """Create Column Defined Classes as subclasses of Primitive Classes."""
        onto = result.onto
        Column = result.class_map.get('Column', Thing)
        
        for col_class in column_classes:
            # Handle both dict and pydantic model
            if hasattr(col_class, 'column_id'):
                col_data = {
                    'column_id': col_class.column_id,
                    'column_name': col_class.column_name,
                    'primitive_class': col_class.primitive_class,
                    'label': col_class.label,
                    'description': col_class.description,
                    'insights': getattr(col_class, 'insights', {}),
                }
            else:
                col_data = col_class
            
            # Generate safe class name
            safe_name = col_data['column_id'].replace("::", "_").replace(" ", "_").replace("-", "_")
            
            # Get primitive class name (strip prefix)
            primitive_name = _strip_prefix(col_data.get('primitive_class', ''))
            
            # Determine parent class
            if primitive_name and primitive_name in result.class_map:
                parent_cls = result.class_map[primitive_name]
            else:
                # Create placeholder for primitive class
                parent_cls = type(primitive_name, (Column,), {"namespace": onto})
                result.class_map[primitive_name] = parent_cls
            
            # Create Column Defined Class
            col_owl_cls = type(safe_name, (parent_cls,), {"namespace": onto})
            
            # Set label
            col_owl_cls.label = [col_data.get('label', col_data.get('column_name', safe_name))]
            
            # Build rich description including insights
            description = self._build_rich_column_description(col_data)
            col_owl_cls.comment = [description]
            
            result.class_map[safe_name] = col_owl_cls
    
    def _build_rich_column_description(self, col_data: Dict) -> str:
        """
        Build rich description using Layer 1 DataProperty readout templates.
        
        Priority:
        1. Use readout_template from Layer 1 DataProperty if matched by primitive class
        2. Fall back to generic stats formatting
        3. Include LLM-generated description
        """
        parts = []
        
        # Base description from LLM
        if col_data.get('description'):
            parts.append(col_data['description'])
        
        # Get insights from Stage 2
        insights = col_data.get('insights', {})
        primitive_class = col_data.get('primitive_class', '').replace('upo:', '')
        
        # Priority 1: Use pre-computed readout from insights (from data_property_values)
        if insights.get('readout'):
            parts.append(f"Summary: {insights['readout']}")
        else:
            # Priority 2: Try to use readout_template from Layer 1 DataProperty
            readout_str = None
            if hasattr(self, '_layer1_data_property_by_class') and primitive_class:
                dp = self._layer1_data_property_by_class.get(primitive_class)
                if dp and dp.get('readout_template') and insights:
                    template = dp['readout_template']
                    try:
                        # Build template context from insights
                        template_context = {}
                        
                        # Count stats
                        if insights.get('unique_ratio') is not None:
                            template_context['distinct_count'] = f"{insights['unique_ratio']:.0%}"
                        else:
                            template_context['distinct_count'] = 'N/A'
                        
                        # Sample values for top_values - use random sampling
                        sample_values = insights.get('sample_values', [])
                        if sample_values:
                            from workflows.population.sampling_utils import sample_values as do_sample
                            sampled = do_sample(sample_values, 3)
                            template_context['top_values'] = ', '.join(str(v) for v in sampled)
                            template_context['sample_values'] = ', '.join(str(v) for v in sampled)
                        else:
                            template_context['top_values'] = 'N/A'
                            template_context['sample_values'] = 'N/A'
                        
                        # Try to format template
                        readout_str = template.format(**template_context)
                    except (KeyError, ValueError):
                        # Template formatting failed, fall back
                        pass
            
            if readout_str:
                parts.append(f"Summary: {readout_str}")
            elif insights:
                # Priority 3: Minimal fallback
                insight_parts = []
                
                if insights.get('unique_ratio') is not None:
                    insight_parts.append(f"Uniqueness: {insights['unique_ratio']:.0%}")
                
                if insights.get('sample_values'):
                    from workflows.population.sampling_utils import sample_values as do_sample
                    sampled = do_sample(insights['sample_values'], 3)
                    samples = ', '.join(str(v) for v in sampled)
                    insight_parts.append(f"e.g., {samples}")
                
                if insight_parts:
                    parts.append("; ".join(insight_parts))
        
        return " | ".join(parts) if parts else f"Column defined class for {col_data.get('column_name', 'unknown')}"
    
    def _create_table_defined_classes(
        self,
        result: ConversionResult,
        table_classes: List,
    ) -> None:
        """
        Create Table Defined Classes with existential restrictions.
        """
        onto = result.onto
        Table = result.class_map.get('Table', Thing)
        
        for tbl_class in table_classes:
            # Handle both dict and pydantic model
            if hasattr(tbl_class, 'class_name'):
                tbl_data = {
                    'class_name': tbl_class.class_name,
                    'table_id': getattr(tbl_class, 'table_id', ''),
                    'label': tbl_class.label,
                    'description': tbl_class.description,
                    'core_el_definition': getattr(tbl_class, 'core_el_definition', ''),
                    'column_ids': getattr(tbl_class, 'column_ids', []),
                }
            else:
                tbl_data = tbl_class
            
            class_name = tbl_data['class_name']
            table_id = tbl_data.get('table_id', '')
            
            # Create Table Defined Class
            tbl_owl_cls = type(class_name, (Table,), {"namespace": onto})
            
            # Set label and description
            tbl_owl_cls.label = [tbl_data.get('label', class_name)]
            if tbl_data.get('description'):
                tbl_owl_cls.comment = [tbl_data['description']]
            
            # Add existential restrictions from core_el_definition
            core_el = tbl_data.get('core_el_definition', '')
            if core_el:
                import re
                pattern = r'∃(\w+)\.(\w+)'
                for match in re.finditer(pattern, core_el):
                    prop_name, filler_name = match.groups()
                    prop = result.object_property_map.get(prop_name)
                    filler = result.class_map.get(filler_name)
                    
                    if prop and filler:
                        tbl_owl_cls.is_a.append(prop.some(filler))
            
            result.class_map[class_name] = tbl_owl_cls
