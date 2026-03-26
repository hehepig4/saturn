"""
Retrieval State Definition (v3 - Parallel Dual-Path Design)

State for the table retrieval workflow that implements:
1. Semantic Retrieval: Vector + BM25 hybrid search on Table Summaries
2. Structural Retrieval: SPARQL/Index-based search on TBox constraints
3. RRF Fusion: Reciprocal Rank Fusion of parallel recall paths

Architecture:
┌──────────────────────────────────────────────────────────┐
│            extract_constraints                            │
│                    │                                      │
│    ┌───────────────┴───────────────┐                     │
│    ▼ (Parallel)                    ▼                     │
│ semantic_search              structural_search            │
│ (Vector + BM25)              (SPARQL/Index)              │
│    │                               │                      │
│    └───────────────┬───────────────┘                     │
│                    ▼                                      │
│               rrf_fusion                                  │
│                    │                                      │
│               abox_verify (optional)                      │
└──────────────────────────────────────────────────────────┘

Design Decisions:
- Dual-path parallel recall (Semantic || Structural) instead of serial filtering
- LLM outputs structured constraints, programs assemble SPARQL
- RRF fusion combines both paths for robust ranking
- Fallback columns (class=upo:Column) indicate low-quality matches
"""

from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict
from config.hyperparameters import HYBRID_VECTOR_WEIGHT, HYBRID_BM25_WEIGHT


# ==================== Constraint Models ====================

class TBoxClassConstraint(BaseModel):
    """A primitive class constraint from TBox."""
    class_name: str = Field(description="Primitive class name (e.g., YearColumn)")


class TBoxConstraints(BaseModel):
    """
    Structured TBox constraints extracted by LLM.
    
    Used for Stage 1: Schema-level filtering.
    """
    primitive_class_constraints: List[TBoxClassConstraint] = Field(
        default_factory=list,
        description="Required column types (primitive classes)"
    )
    
    def get_required_classes(self) -> List[str]:
        """Get list of required class names."""
        return [c.class_name for c in self.primitive_class_constraints]


class EntityConstraint(BaseModel):
    """
    An entity value constraint for ABox filtering.
    
    Value must be in standardized format based on column type's data range:
    - xsd:integer: "-?[0-9]+" (e.g., "42", "-7", "2020")
    - xsd:decimal: "-?[0-9]+(\\.[0-9]+)?" (e.g., "3.14", "-0.5")
    - xsd:dateTime: "YYYY-MM-DDTHH:MM:SS" (e.g., "2024-01-15T00:00:00")
    - xsd:string: lowercase, collapsed whitespace (e.g., "john doe")
    """
    value: str = Field(description="Entity value in standardized format")
    column_type: str = Field(description="Column type where this value should exist")


class ABoxConstraints(BaseModel):
    """
    Structured ABox constraints extracted by LLM.
    
    Used for Stage 2: Cell-level verification.
    """
    entity_constraints: List[EntityConstraint] = Field(
        default_factory=list,
        description="Entity values to match in ABox"
    )
    same_row_required: bool = Field(
        default=True,
        description="Whether all entities must appear in the same row"
    )


# ==================== Result Models ====================

class ColumnTypeMatch(BaseModel):
    """Match between query intent and table column types."""
    column_type: str = Field(description="Matched column type (e.g., YearColumn)")
    confidence: float = Field(default=1.0, description="Match confidence")
    is_fallback: bool = Field(default=False, description="Whether this is a fallback (Column) match")


class RetrievalResult(BaseModel):
    """A single retrieval result."""
    table_id: str = Field(description="Table identifier")
    
    # Scores
    final_score: float = Field(default=0.0, description="Final combined score")
    semantic_score: float = Field(default=0.0, description="Semantic (vector) similarity score")
    structural_score: float = Field(default=0.0, description="Structural (SPARQL) match score")
    
    # Match details
    source: str = Field(
        default="hybrid",
        description="Result source: 'semantic', 'structural', or 'hybrid'"
    )
    tbox_matches: List[ColumnTypeMatch] = Field(
        default_factory=list,
        description="TBox-level column type matches"
    )
    abox_matches: List[str] = Field(
        default_factory=list,
        description="ABox-level entity matches"
    )
    has_fallback_only: bool = Field(
        default=False,
        description="True if all columns are fallback (low quality)"
    )
    
    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (table_description, etc.)"
    )


# ==================== Main State ====================

class RetrievalState(BaseModel):
    """
    State for Parallel Dual-Path Table Retrieval workflow.
    
    Workflow:
        1. extract_constraints: LLM extracts TBox/ABox constraints
        2. parallel_recall: Execute semantic + structural in parallel
           - semantic_search: Vector + BM25 on Table Summaries
           - structural_search: SPARQL/Index on TBox constraints
        3. rrf_fusion: Reciprocal Rank Fusion of both paths
        4. abox_verify: Precise cell-level verification (optional)
    
    Design Principles:
        - Semantic retrieval provides recall breadth (vector similarity)
        - Structural retrieval provides precision (hard constraints)
        - RRF fusion combines complementary strengths
        - Fallback columns indicate low-quality matches
    """
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        extra='allow'
    )
    
    # ========== Input Configuration ==========
    
    query: str = Field(
        default="",
        description="Natural language query"
    )
    
    dataset_name: str = Field(
        default="fetaqa",
        description="Dataset to search in"
    )
    
    top_k: int = Field(
        default=20,
        description="Number of final results to return"
    )
    
    semantic_top_k: int = Field(
        default=100,
        description="Number of candidates from semantic search"
    )
    
    # ========== Feature Toggles ==========
    
    enable_semantic: bool = Field(
        default=True,
        description="Enable semantic (vector) search"
    )
    
    enable_structural: bool = Field(
        default=True,
        description="Enable structural (SPARQL/Index) search path"
    )
    
    structural_top_k: int = Field(
        default=100,
        description="Number of candidates from structural search"
    )
    
    enable_abox_verify: bool = Field(
        default=True,
        description="Enable ABox-level entity verification (lazy transform)"
    )
    
    skip_fallback_tables: bool = Field(
        default=True,
        description="Skip tables where all columns are fallback (upo:Column)"
    )
    
    # ========== Index Configuration ==========
    
    index_key: Optional[str] = Field(
        default=None,
        description="Index configuration key: 'td', 'td_cd', or 'td_cd_cs' (default)"
    )
    
    rag_type: str = Field(
        default="hybrid",
        description="Retrieval type: 'bm25', 'vector', or 'hybrid' (default)"
    )
    
    # ========== Hybrid Search Configuration ==========
    
    enable_bm25: bool = Field(
        default=True,
        description="Enable BM25 (lexical) search in addition to vector search"
    )
    
    search_fields: List[str] = Field(
        default_factory=lambda: ['table_description', 'column_descriptions', 'column_stats'],
        description="Fields to search: table_description, column_descriptions, column_stats"
    )
    
    vector_weight: float = Field(
        default=HYBRID_VECTOR_WEIGHT,
        description="Weight for vector search in hybrid RRF fusion"
    )
    
    bm25_weight: float = Field(
        default=HYBRID_BM25_WEIGHT,
        description="Weight for BM25 search in hybrid RRF fusion"
    )
    
    # ========== Fusion Configuration ==========
    
    fusion_strategy: str = Field(
        default="intersection_first",
        description="Fusion strategy: 'intersection_first', 'union', 'semantic_only', 'structural_only'"
    )
    
    semantic_weight: float = Field(
        default=0.5,
        description="Weight for semantic score in final ranking"
    )
    
    structural_weight: float = Field(
        default=0.5,
        description="Weight for structural score in final ranking"
    )
    
    # ========== Step 1: Constraint Extraction ==========
    
    tbox_constraints: Optional[TBoxConstraints] = Field(
        default=None,
        description="TBox constraints extracted by LLM"
    )
    
    abox_constraints: Optional[ABoxConstraints] = Field(
        default=None,
        description="ABox constraints extracted by LLM"
    )
    
    constraint_extraction_time: float = Field(
        default=0.0,
        description="Time spent on constraint extraction (seconds)"
    )
    
    # ========== Step 2a: Semantic Search Results ==========
    
    semantic_results: List[RetrievalResult] = Field(
        default_factory=list,
        description="Results from semantic (vector+BM25) search path"
    )
    
    semantic_search_time: float = Field(
        default=0.0,
        description="Time spent on semantic search (seconds)"
    )
    
    # ========== Step 2b: Structural Search Results ==========
    
    structural_results: List[RetrievalResult] = Field(
        default_factory=list,
        description="Results from structural (SPARQL/Index) search path"
    )
    
    structural_search_time: float = Field(
        default=0.0,
        description="Time spent on structural search (seconds)"
    )
    
    # ========== Step 4: ABox Verification Results ==========
    
    abox_verified_results: List[RetrievalResult] = Field(
        default_factory=list,
        description="Results that pass ABox verification (lazy transform)"
    )
    
    abox_verify_time: float = Field(
        default=0.0,
        description="Time spent on ABox verification (seconds)"
    )
    
    # ========== Step 5: Fusion Results ==========
    
    fused_results: List[RetrievalResult] = Field(
        default_factory=list,
        description="Results after RRF fusion of semantic + structural paths"
    )
    
    fusion_time: float = Field(
        default=0.0,
        description="Time spent on RRF fusion (seconds)"
    )
    
    # ========== Final Results ==========
    
    final_results: List[RetrievalResult] = Field(
        default_factory=list,
        description="Final ranked retrieval results (after optional ABox verification)"
    )
    
    # ========== Execution Metadata ==========
    
    success: bool = Field(
        default=False,
        description="Whether retrieval completed successfully"
    )
    
    error: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )
    
    total_time: float = Field(
        default=0.0,
        description="Total retrieval time (seconds)"
    )
    
    trajectory: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Execution trajectory for debugging"
    )
