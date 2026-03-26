"""
Evaluation Results Export Utilities

Provides unified data models and export functions for retrieval evaluation results.

Usage:
    from evaluation.export_utils import EvaluationResults, EvaluationRun, QueryResult
    
    # Create result container
    run = EvaluationRun(
        run_id="20260204_120000",
        dataset="fetaqa",
        method="hyde",
        hyde_mode="combined",
        retriever_type="hybrid"
    )
    results = EvaluationResults(run)
    
    # Add query results
    results.add_query_result(QueryResult(
        query_id=0,
        query_text="Which team won the championship?",
        gt_tables=["table_123"],
        rank=3,
        top_k_table_ids=["table_456", "table_789", "table_123"],
        top_k_scores=[0.89, 0.85, 0.82]
    ))
    
    # Compute and export
    results.compute_metrics()
    results.save_json("results.json")
    results.save_parquet("results.parquet")
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available, some export features disabled")


@dataclass
class EvaluationRun:
    """Metadata for a single evaluation run"""
    
    run_id: str                    # Unique identifier (timestamp-based)
    timestamp: str                 # ISO format timestamp
    dataset: str                   # Dataset name
    split: str = "test"            # Data split (test/train)
    num_queries: int = -1          # Number of queries (-1 for all)
    
    # Method-specific metadata
    method: str = ""               # Evaluation method (hyde/structural/hybrid)
    llm: str = "local"             # LLM type
    
    # HyDE-specific
    hyde_mode: Optional[str] = None      # raw/table_desc/column_desc/combined
    retriever_type: Optional[str] = None # bm25/vector/hybrid
    
    # Structural-specific
    tbox_iteration: Optional[int] = None # TBox iteration version
    
    # Additional parameters
    top_k: int = 100               # Maximum retrieval candidates
    rag_top_k: Optional[int] = None     # RAG top-k for analysis
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class QueryResult:
    """Result for a single query's retrieval"""
    
    query_id: int
    query_text: str
    gt_tables: List[str]           # Ground truth table IDs
    
    # Retrieval results
    rank: Optional[int] = None     # Best GT rank (1-indexed, None if not found)
    top_k_table_ids: List[str] = field(default_factory=list)  # Top-K retrieved table IDs
    top_k_scores: List[float] = field(default_factory=list)   # Corresponding scores
    
    # HyDE-specific (optional)
    hyde_table_desc: Optional[str] = None
    hyde_column_desc: Optional[str] = None
    
    # Structural-specific (optional)
    tbox_constraints: Optional[List[str]] = None
    abox_constraints: Optional[List[str]] = None
    
    # Error tracking
    error: Optional[str] = None    # Error message if retrieval failed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class AggregatedMetrics:
    """Aggregated retrieval metrics"""
    
    hit_at_1: float
    hit_at_5: float
    hit_at_10: float
    hit_at_50: float
    hit_at_100: float
    mrr: float
    
    # Optional extended metrics
    precision_at_10: Optional[float] = None
    ndcg_at_10: Optional[float] = None
    
    # Statistics
    num_queries: int = 0
    num_found: int = 0            # Queries where GT was found in top-K
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}


class EvaluationResults:
    """Container for complete evaluation results"""
    
    def __init__(self, run_metadata: EvaluationRun):
        self.metadata = run_metadata
        self.query_results: List[QueryResult] = []
        self.aggregated_metrics: Optional[AggregatedMetrics] = None
    
    def set_aggregated_metrics(
        self,
        hit_at_1: float = 0.0,
        hit_at_5: float = 0.0,
        hit_at_10: float = 0.0,
        hit_at_50: float = 0.0,
        hit_at_100: float = 0.0,
        mrr: float = 0.0,
        num_queries: int = 0,
        num_found: int = 0
    ):
        """Manually set aggregated metrics (for parsed results)"""
        self.aggregated_metrics = AggregatedMetrics(
            hit_at_1=hit_at_1,
            hit_at_5=hit_at_5,
            hit_at_10=hit_at_10,
            hit_at_50=hit_at_50,
            hit_at_100=hit_at_100,
            mrr=mrr,
            num_queries=num_queries,
            num_found=num_found
        )
    
    def add_query_result(self, result: QueryResult):
        """Add a single query result"""
        self.query_results.append(result)
    
    def compute_metrics(self) -> AggregatedMetrics:
        """Compute aggregated metrics from query results"""
        total = len(self.query_results)
        if total == 0:
            raise ValueError("No query results to compute metrics from")
        
        def hit_at_k(k: int) -> float:
            """Calculate Hit@K"""
            hits = sum(1 for r in self.query_results 
                      if r.rank is not None and r.rank <= k)
            return hits / total
        
        def mrr() -> float:
            """Calculate Mean Reciprocal Rank"""
            rrs = [1.0 / r.rank for r in self.query_results 
                  if r.rank is not None]
            return sum(rrs) / total if rrs else 0.0
        
        num_found = sum(1 for r in self.query_results if r.rank is not None)
        
        self.aggregated_metrics = AggregatedMetrics(
            hit_at_1=hit_at_k(1),
            hit_at_5=hit_at_k(5),
            hit_at_10=hit_at_k(10),
            hit_at_50=hit_at_k(50),
            hit_at_100=hit_at_k(100),
            mrr=mrr(),
            num_queries=total,
            num_found=num_found
        )
        return self.aggregated_metrics
    
    def to_pandas(self) -> 'pd.DataFrame':
        """Convert to pandas DataFrame (per-query level)"""
        if not HAS_PANDAS:
            raise RuntimeError("pandas is required for to_pandas()")
        
        records = []
        for qr in self.query_results:
            record = {
                **self.metadata.to_dict(),
                **qr.to_dict()
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def to_summary_pandas(self) -> 'pd.DataFrame':
        """Convert to pandas DataFrame (aggregated summary - one row per run)
        
        This is useful for visualization where each run should be one row.
        """
        if not HAS_PANDAS:
            raise RuntimeError("pandas is required for to_summary_pandas()")
        
        if self.aggregated_metrics is None:
            self.compute_metrics()
        
        # Create single row with metadata + metrics
        record = {
            **self.metadata.to_dict(),
            **self.aggregated_metrics.to_dict()
        }
        
        return pd.DataFrame([record])
    
    def save_parquet(self, filepath: str, summary_only: bool = True):
        """Save results to Parquet file (requires pandas and pyarrow)
        
        Args:
            filepath: Output file path
            summary_only: If True, exports only aggregated metrics (one row per run).
                         If False, exports per-query results.
        """
        if not HAS_PANDAS:
            raise RuntimeError("pandas is required for save_parquet()")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if summary_only:
            df = self.to_summary_pandas()
        else:
            df = self.to_pandas()
        
        df.to_parquet(filepath, engine='pyarrow', compression='snappy', index=False)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Export as summary dictionary (metadata + aggregated metrics)"""
        if self.aggregated_metrics is None:
            self.compute_metrics()
        
        return {
            "metadata": self.metadata.to_dict(),
            "metrics": self.aggregated_metrics.to_dict(),
            "timestamp_computed": datetime.now().isoformat()
        }
    
    def save_json(self, filepath: str, include_per_query: bool = True):
        """Save results to JSON file
        
        Args:
            filepath: Output file path
            include_per_query: Include per-query details (default: True)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_summary_dict()
        
        if include_per_query:
            data["query_results"] = [qr.to_dict() for qr in self.query_results]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_json(cls, filepath: str) -> 'EvaluationResults':
        """Load results from JSON file
        
        Args:
            filepath: Input file path
        
        Returns:
            EvaluationResults instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruct metadata
        metadata = EvaluationRun(**data['metadata'])
        results = cls(metadata)
        
        # Reconstruct query results
        if 'query_results' in data:
            for qr_dict in data['query_results']:
                results.add_query_result(QueryResult(**qr_dict))
        
        # Reconstruct metrics if available
        if 'metrics' in data:
            results.aggregated_metrics = AggregatedMetrics(**data['metrics'])
        
        return results
    

def create_run_id(dataset: str, method: str, **kwargs) -> str:
    """Generate a unique run ID
    
    Args:
        dataset: Dataset name
        method: Evaluation method
        **kwargs: Additional parameters for ID
    
    Returns:
        Unique run ID string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [dataset, method]
    
    # Add optional parameters
    for key in ['retriever_type', 'hyde_mode']:
        if key in kwargs and kwargs[key]:
            parts.append(kwargs[key])
    
    parts.append(timestamp)
    return "_".join(parts)

