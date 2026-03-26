"""
Table Discovery Layer 2 Subgraph

Generates Layer 2 Defined Classes from tables using Layer 1 Primitive Classes.
This subgraph focuses on annotating tables and columns with semantic metadata.

Workflow:
    1. Load Data: Load tables, column summaries, and Layer 1 primitives
    2. Annotate Tables: LLM-based column annotation with P0/P1 enhancements
    3. Create Defined Classes: Build Column and Table Defined Classes
    4. Discover Properties: Identify semantic relationships (Phase 3)
    5. Export Layer 2: Store to LanceDB and export OWL

Output:
    - Column Defined Classes with P0/P1 metadata:
        - P0: rdfs:comment descriptions
        - P1-a: Match Level (exact, semantic, fallback)
        - P1-b: Column Role (key, temporal, measure, attribute, noise)
        - P1-c: Validated primitive class
    - Table Defined Classes with:
        - Full EL: All column constraints
        - Core EL: Role-based constraints (excludes NOISE)
        - Semantic summary by role
"""

from .graph import (
    create_table_discovery_layer2_graph,
    invoke_table_discovery_layer2,
)
from .state import (
    TableDiscoveryLayer2State,
    ColumnAnnotation,
    TableAnnotation,
    ColumnDefinedClass,
    TableDefinedClass,
)

__all__ = [
    # Graph
    "create_table_discovery_layer2_graph",
    "invoke_table_discovery_layer2",
    # State
    "TableDiscoveryLayer2State",
    # Models
    "ColumnAnnotation",
    "TableAnnotation",
    "ColumnDefinedClass",
    "TableDefinedClass",
]
