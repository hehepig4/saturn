#!/usr/bin/env python3
"""
Analyze BM25 Term Contribution for HyDE Retrieval

This script analyzes how individual terms in HyDE-generated text contribute to 
BM25 retrieval scores, with special focus on primitive class tokens (e.g., [Identifier > Name]).

Key features:
1. Term contribution decomposition: Break down BM25 score by individual terms
2. Primitive class token analysis: Identify and quantify contribution from [Parent > Child] tokens
3. Ablation experiments: Remove/keep only primitive class tokens and measure recall changes

Usage:
    # Basic term contribution analysis
    python -m evaluation.runners.bm25_analysis -d fetaqa --mode contribution

    # Ablation: remove primitive class tokens
    python -m evaluation.runners.bm25_analysis -d fetaqa --mode ablation-remove

    # Ablation: keep only primitive class tokens
    python -m evaluation.runners.bm25_analysis -d fetaqa --mode ablation-keep

    # Full analysis with case-by-case details
    python -m evaluation.runners.bm25_analysis -d fetaqa --mode full --show-cases 10
"""

import argparse
import json
import pickle
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

import numpy as np
import bm25s
import Stemmer

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
import _path_setup  # noqa: F401

from loguru import logger
from core.paths import get_db_path, get_data_root
from workflows.retrieval.unified_search import load_unified_indexes
from workflows.retrieval.config import INDEX_KEY_TD_CD_CS


# ==================== Data Classes ====================

@dataclass
class TermContribution:
    """Contribution of a single term to BM25 score."""
    term: str
    original_token: str  # Before stemming
    idf: float
    tf_component: float  # TF part of score
    contribution: float  # Final contribution to score
    is_primitive_class: bool = False
    occurrences: int = 1  # Number of times this token appeared in the query


@dataclass
class DocumentScore:
    """BM25 score breakdown for a document."""
    doc_id: str
    total_score: float
    term_contributions: List[TermContribution] = field(default_factory=list)
    primitive_class_contribution: float = 0.0
    other_contribution: float = 0.0


# ==================== BM25 Term Contribution Calculator ====================

class BM25TermAnalyzer:
    """
    Analyze term contributions in BM25 retrieval.
    
    Implements BM25 score decomposition:
    score(q, d) = Σ IDF(t) × TF_component(t, d)
    
    Where TF_component = (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × |d|/avgdl))
    """
    
    def __init__(
        self,
        bm25_index_path: str,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Initialize analyzer with BM25 index.
        
        Args:
            bm25_index_path: Path to bm25s index directory
            k1: BM25 term frequency saturation parameter
            b: BM25 document length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.stemmer = Stemmer.Stemmer("english")
        
        # Load BM25 index components
        index_path = Path(bm25_index_path) / "index"
        
        # Load vocabulary
        with open(index_path / "vocab.index.json", "r") as f:
            self.vocab = json.load(f)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Load sparse matrix (CSC format)
        self.data = np.load(index_path / "data.csc.index.npy")
        self.indices = np.load(index_path / "indices.csc.index.npy")
        self.indptr = np.load(index_path / "indptr.csc.index.npy")
        
        # Load params
        with open(index_path / "params.index.json", "r") as f:
            params = json.load(f)
        self.num_docs = params["num_docs"]
        
        # Calculate DF and IDF for each term
        self.df = np.diff(self.indptr)
        self.idf = self._compute_idf(self.df, self.num_docs)
        
        # Load corpus for document length calculation
        self.corpus = []
        corpus_path = index_path / "corpus.jsonl"
        if corpus_path.exists():
            with open(corpus_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('{'):
                        doc = json.loads(line)
                        self.corpus.append(doc.get("text", ""))
                    else:
                        self.corpus.append(line)
        
        # Calculate document lengths (in tokens)
        self.doc_lengths = self._compute_doc_lengths()
        self.avgdl = np.mean(self.doc_lengths) if len(self.doc_lengths) > 0 else 1.0
        
        # Load table_ids mapping
        table_ids_path = Path(bm25_index_path) / "table_ids.pkl"
        if table_ids_path.exists():
            with open(table_ids_path, "rb") as f:
                self.table_ids = pickle.load(f)
        else:
            self.table_ids = [f"doc_{i}" for i in range(self.num_docs)]
        
        logger.info(f"BM25TermAnalyzer initialized:")
        logger.info(f"  Vocab size: {len(self.vocab)}")
        logger.info(f"  Num docs: {self.num_docs}")
        logger.info(f"  Avg doc length: {self.avgdl:.1f}")
    
    def _compute_idf(self, df: np.ndarray, N: int) -> np.ndarray:
        """Compute IDF using Lucene formula."""
        # IDF = log(1 + (N - df + 0.5) / (df + 0.5))
        return np.log(1 + (N - df + 0.5) / (df + 0.5))
    
    def _compute_doc_lengths(self) -> np.ndarray:
        """Compute document lengths from corpus."""
        lengths = []
        for doc_text in self.corpus:
            tokens = bm25s.tokenize([doc_text], stemmer=self.stemmer, show_progress=False)[0]
            lengths.append(len(tokens))
        return np.array(lengths) if lengths else np.ones(self.num_docs) * 100
    
    def _get_term_freq(self, term_id: int, doc_id: int) -> float:
        """Get term frequency for a term in a document."""
        start = self.indptr[term_id]
        end = self.indptr[term_id + 1]
        
        # Search for doc_id in indices[start:end]
        doc_positions = self.indices[start:end]
        tf_values = self.data[start:end]
        
        mask = doc_positions == doc_id
        if np.any(mask):
            return float(tf_values[mask][0])
        return 0.0
    
    def tokenize_with_tracking(self, text: str) -> List[Tuple[str, str]]:
        """
        Tokenize text and track original tokens.
        
        Returns:
            List of (stemmed_token, original_token) tuples
        """
        # Get stemmed tokens using the same method as BM25 index
        tokenized = bm25s.tokenize([text], stemmer=self.stemmer, stopwords="en", show_progress=False)
        
        # Convert tokenized result to list of strings
        id_to_token = {v: k for k, v in tokenized.vocab.items()}
        stemmed_tokens = [id_to_token[i] for i in tokenized.ids[0]]
        
        # For original tokens, tokenize without stemming
        raw_tokenized = bm25s.tokenize([text], stopwords="en", show_progress=False)
        raw_id_to_token = {v: k for k, v in raw_tokenized.vocab.items()}
        raw_tokens = [raw_id_to_token[i] for i in raw_tokenized.ids[0]]
        
        # Match up (they should be same length after stopword removal)
        result = []
        for i, stemmed in enumerate(stemmed_tokens):
            original = raw_tokens[i] if i < len(raw_tokens) else stemmed
            result.append((stemmed, original))
        
        return result
    
    def compute_term_contributions(
        self,
        query_text: str,
        doc_id: int,
        primitive_class_tokens: Optional[Set[str]] = None,
    ) -> DocumentScore:
        """
        Compute contribution of each query term to the BM25 score for a document.
        
        Args:
            query_text: Query or HyDE text
            doc_id: Document index
            primitive_class_tokens: Set of tokens that are primitive class names (stemmed)
        
        Returns:
            DocumentScore with term-by-term breakdown
        """
        if primitive_class_tokens is None:
            primitive_class_tokens = set()
        
        token_pairs = self.tokenize_with_tracking(query_text)
        doc_length = self.doc_lengths[doc_id] if doc_id < len(self.doc_lengths) else self.avgdl
        
        # Aggregate contributions by unique stemmed token
        token_aggregator: Dict[str, TermContribution] = {}
        total_score = 0.0
        primitive_total = 0.0
        other_total = 0.0
        
        for stemmed, original in token_pairs:
            if stemmed not in self.vocab:
                continue
            
            term_id = self.vocab[stemmed]
            tf = self._get_term_freq(term_id, doc_id)
            
            if tf == 0:
                continue
            
            idf = self.idf[term_id]
            
            # BM25 TF component
            tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_length / self.avgdl))
            
            contribution = idf * tf_component
            total_score += contribution
            
            # Check if this is a primitive class token
            is_primitive = stemmed in primitive_class_tokens or original.lower() in primitive_class_tokens
            
            if is_primitive:
                primitive_total += contribution
            else:
                other_total += contribution
            
            # Aggregate: merge duplicate stemmed tokens into one entry
            if stemmed in token_aggregator:
                existing = token_aggregator[stemmed]
                existing.contribution += contribution
                existing.tf_component += tf_component
                existing.occurrences += 1
            else:
                token_aggregator[stemmed] = TermContribution(
                    term=stemmed,
                    original_token=original,
                    idf=idf,
                    tf_component=tf_component,
                    contribution=contribution,
                    is_primitive_class=is_primitive,
                )
        
        contributions = list(token_aggregator.values())
        
        # Sort by contribution descending
        contributions.sort(key=lambda x: x.contribution, reverse=True)
        
        table_id = self.table_ids[doc_id] if doc_id < len(self.table_ids) else f"doc_{doc_id}"
        
        return DocumentScore(
            doc_id=table_id,
            total_score=total_score,
            term_contributions=contributions,
            primitive_class_contribution=primitive_total,
            other_contribution=other_total,
        )
    

# ==================== Primitive Class Token Extractor ====================

def extract_primitive_class_tokens(hyde_text: str) -> Set[str]:
    """
    Extract primitive class tokens from HyDE column descriptions.
    
    Pattern: [Parent > Child] chain format at the start of each column description
    Examples: [Identifier > Name], [Numeric > Financial > CostMetric]
    
    Returns:
        Set of stemmed primitive class tokens
    """
    stemmer = Stemmer.Stemmer("english")
    
    # Pattern: [Content] - matches text inside square brackets (chain format)
    pattern = r'\[([^\]]+)\]'
    matches = re.findall(pattern, hyde_text)
    
    # Stem the class names
    stemmed = set()
    for match in matches:
        # Split by ' > ' to get individual class names in the chain
        parts = [p.strip() for p in match.split('>')]
        for part in parts:
            # Split CamelCase: PersonName -> person name
            words = re.sub(r'([a-z])([A-Z])', r'\1 \2', part).lower().split()
            for word in words:
                stemmed_word = stemmer.stemWord(word)
                stemmed.add(stemmed_word)
            # Also add the full part (lowered)
            stemmed.add(part.lower())
    
    return stemmed


def remove_primitive_class_markers(hyde_text: str) -> str:
    """Remove [Parent > Child] chain markers from HyDE text.

    The HyDE column description format is: [Parent > Child] ColumnName: description
    The bracket markers with chain format need to be removed for proper ablation.
    """
    # Remove [Content] markers (chain format)
    result = re.sub(r'\[[^\]]+\]\s*', '', hyde_text)
    return result


def remove_primitive_class_brackets(text: str) -> str:
    """Remove only [Parent > Child] bracket markers, preserving the actual column name.

    For indexed documents the format is: [Parent > Child] ActualColumnName: description
    Only the bracket marker (with chain content) should be removed,
    keeping the real column schema name intact.
    """
    return re.sub(r'\[[^\]]+\]\s*', '', text)


def keep_only_primitive_class_tokens(hyde_text: str) -> str:
    """
    Extract only the primitive class tokens from HyDE text.
    
    The HyDE format is: [Parent > Child] ColumnName: description
    For chain format like [Identifier > Name], we extract the chain components.
    
    Returns a string containing the expanded class names from the chains.
    """
    # Match [Content] pattern (chain format)
    pattern = r'\[([^\]]+)\]'
    matches = re.findall(pattern, hyde_text)

    result = []
    for match in matches:
        # Split by ' > ' to get individual class names
        parts = [p.strip() for p in match.split('>')]
        for part in parts:
            # Split CamelCase: PersonName -> Person Name
            words = re.sub(r'([a-z])([A-Z])', r'\1 \2', part)
            result.append(words)

    return ' '.join(result)


# ==================== Analysis Functions ====================

def load_unified_analysis(
    dataset: str,
    rag_type: str = None,
    no_primitive_classes: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load unified analysis results.
    
    Args:
        dataset: Dataset name
        rag_type: RAG retrieval type ("bm25", "vector", "hybrid") or None for any
        no_primitive_classes: If True, look for _no_pc suffix files
    """
    eval_dir = get_db_path() / "eval_results"
    
    # Build search patterns based on options
    patterns = []
    
    # New format patterns with rag_type
    if rag_type:
        pc_suffix = "_no_pc" if no_primitive_classes else ""
        patterns.extend([
            f"{dataset}_test_unified_analysis_all_local_rag3_{rag_type}{pc_suffix}.json",
            f"{dataset}_test_unified_analysis_all_local_rag5_{rag_type}{pc_suffix}.json",
        ])
    
    # Legacy patterns (backward compatible)
    patterns.extend([
        f"{dataset}_test_unified_analysis_all_local_rag3.json",
        f"{dataset}_unified_analysis_all_local_rag3.json",
        f"{dataset}_test_unified_analysis_all_local.json",
    ])
    
    for pattern in patterns:
        filepath = eval_dir / pattern
        if filepath.exists():
            logger.info(f"Loading: {filepath}")
            with open(filepath, 'r') as f:
                return json.load(f)
    
    # Glob search for matching files
    glob_patterns = [
        f"{dataset}_test_unified_analysis_*_local_rag*_{rag_type or '*'}.json",
        f"{dataset}_test_unified_analysis_*_local_rag*.json",
    ]
    
    for pattern in glob_patterns:
        files = list(eval_dir.glob(pattern))
        # Filter by no_pc if specified
        if no_primitive_classes:
            files = [f for f in files if "_no_pc" in f.name]
        elif not no_primitive_classes and rag_type:
            files = [f for f in files if "_no_pc" not in f.name]
        
        if files:
            files.sort(key=lambda f: f.stat().st_size, reverse=True)
            filepath = files[0]
            logger.info(f"Loading: {filepath}")
            with open(filepath, 'r') as f:
                return json.load(f)
    
    raise FileNotFoundError(f"No unified analysis file found for {dataset}")


def analyze_term_contributions(
    analysis_data: List[Dict[str, Any]],
    analyzer: BM25TermAnalyzer,
    metadata_list: List[Dict],
    num_queries: int = -1,
    show_cases: int = 5,
    hyde_mode: str = "combined",
) -> Dict[str, Any]:
    """
    Analyze term contributions across all queries.
    
    Returns statistics on primitive class token contributions.
    """
    if num_queries > 0:
        analysis_data = analysis_data[:num_queries]
    
    # Build table_id -> doc_id mapping
    table_to_doc = {}
    for i, meta in enumerate(metadata_list):
        table_to_doc[meta.get('table_id', '')] = i
    
    results = {
        'total_queries': len(analysis_data),
        'queries_with_gt_in_index': 0,
        'primitive_class_contribution_ratios': [],
        'top_contributing_classes': defaultdict(float),
        'case_analyses': [],
    }
    
    for i, item in enumerate(analysis_data):
        query = item['query']
        gt_tables = item.get('gt_tables', [item['gt_table']] if item.get('gt_table') else [])
        analysis = item.get('analysis', {})
        
        # Raw mode doesn't require analysis
        if hyde_mode != "raw" and (not analysis or item.get('error')):
            continue
        
        # Get query/HyDE text based on mode
        if hyde_mode == "raw":
            hyde_text = query  # Use raw query as input
        elif hyde_mode == "column_desc":
            hyde_text = analysis.get('hypothetical_column_descriptions', '')
        elif hyde_mode == "table_desc":
            hyde_text = analysis.get('hypothetical_table_description', '')
        elif hyde_mode == "combined":
            table_desc = analysis.get('hypothetical_table_description', '')
            col_desc = analysis.get('hypothetical_column_descriptions', '')
            hyde_text = f"{table_desc}\n{col_desc}"
        else:
            hyde_text = query  # Fallback to raw query
        
        if not hyde_text.strip():
            continue
        
        # Find GT documents
        gt_doc_ids = []
        for gt in gt_tables:
            if gt in table_to_doc:
                gt_doc_ids.append((table_to_doc[gt], gt))
        
        if not gt_doc_ids:
            continue
        
        results['queries_with_gt_in_index'] += 1
        
        # Extract primitive class tokens
        primitive_tokens = extract_primitive_class_tokens(hyde_text)
        
        # Compute contributions for ALL GT documents and average
        all_doc_scores = []
        all_ratios = []
        for doc_id, gt_table in gt_doc_ids:
            doc_score = analyzer.compute_term_contributions(hyde_text, doc_id, primitive_tokens)
            all_doc_scores.append((doc_score, gt_table))
            if doc_score.total_score > 0:
                ratio = doc_score.primitive_class_contribution / doc_score.total_score
                all_ratios.append(ratio)
        
        # Use average ratio across all GTs
        if all_ratios:
            avg_ratio = np.mean(all_ratios)
            results['primitive_class_contribution_ratios'].append(avg_ratio)
            
            # Track top contributing classes (aggregate across all GTs)
            for doc_score, _ in all_doc_scores:
                for tc in doc_score.term_contributions:
                    if tc.is_primitive_class:
                        # Divide by number of GTs to avoid overcounting
                        results['top_contributing_classes'][tc.term] += tc.contribution / len(gt_doc_ids)
        
        # For case analysis, use the GT with highest score
        best_doc_score, best_gt_table = max(all_doc_scores, key=lambda x: x[0].total_score)
        ratio = avg_ratio if all_ratios else 0
        
        # Store case analysis for top cases
        if len(results['case_analyses']) < show_cases:
            results['case_analyses'].append({
                'index': i + 1,
                'query': query,
                'gt_table': best_gt_table,  # Best GT for display
                'gt_tables': gt_tables,  # All GTs for reference
                'num_gt_tables': len(gt_doc_ids),
                'avg_ratio': avg_ratio if all_ratios else 0,
                'hyde_text': hyde_text[:500] + '...' if len(hyde_text) > 500 else hyde_text,
                'primitive_tokens': list(primitive_tokens),
                'total_score': best_doc_score.total_score,
                'primitive_contribution': best_doc_score.primitive_class_contribution,
                'other_contribution': best_doc_score.other_contribution,
                'contribution_ratio': ratio,
                'top_terms': [
                    {
                        'term': tc.term,
                        'original': tc.original_token,
                        'contribution': tc.contribution,
                        'idf': tc.idf,
                        'is_primitive': tc.is_primitive_class,
                        'occurrences': tc.occurrences,
                    }
                    for tc in best_doc_score.term_contributions[:10]
                ],
            })
        
        # Progress
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(analysis_data)}")
    
    # Compute statistics
    ratios = results['primitive_class_contribution_ratios']
    if ratios:
        results['stats'] = {
            'mean_primitive_ratio': np.mean(ratios),
            'median_primitive_ratio': np.median(ratios),
            'min_primitive_ratio': np.min(ratios),
            'max_primitive_ratio': np.max(ratios),
            'std_primitive_ratio': np.std(ratios),
        }
    
    # Sort top contributing classes
    results['top_contributing_classes'] = dict(
        sorted(results['top_contributing_classes'].items(), key=lambda x: x[1], reverse=True)[:20]
    )
    
    return results


def _build_clean_bm25_index(
    bm25_index_path: str,
    clean_fn,
) -> Tuple[bm25s.BM25, List[str]]:
    """
    Build a BM25 index with primitive class markers removed from documents.

    Args:
        bm25_index_path: Path to the original BM25 index directory
        clean_fn: Function to apply to each document text (e.g., remove_primitive_class_brackets)

    Returns:
        Tuple of (clean_retriever, table_ids)
    """
    stemmer = Stemmer.Stemmer("english")
    index_path = Path(bm25_index_path) / "index"

    # Load original corpus text
    corpus_texts = []
    corpus_path = index_path / "corpus.jsonl"
    if corpus_path.exists():
        with open(corpus_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith('{'):
                    doc = json.loads(line)
                    corpus_texts.append(doc.get("text", ""))
                else:
                    corpus_texts.append(line)
    else:
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    # Apply cleaning function to each document
    clean_corpus = [clean_fn(text) for text in corpus_texts]

    # Build new BM25 index from clean corpus
    clean_tokens = bm25s.tokenize(clean_corpus, stemmer=stemmer, stopwords="en", show_progress=False)
    clean_retriever = bm25s.BM25()
    clean_retriever.index(clean_tokens, show_progress=False)

    # Load table_ids
    table_ids_path = Path(bm25_index_path) / "table_ids.pkl"
    if table_ids_path.exists():
        with open(table_ids_path, "rb") as f:
            table_ids = pickle.load(f)
    else:
        table_ids = [f"doc_{i}" for i in range(len(corpus_texts))]

    logger.info(f"Built clean BM25 index: {len(clean_corpus)} docs, "
                f"avg removed chars: {np.mean([len(a)-len(b) for a,b in zip(corpus_texts, clean_corpus)]):.0f}")

    return clean_retriever, table_ids


def run_ablation_experiment(
    analysis_data: List[Dict[str, Any]],
    bm25_retriever: bm25s.BM25,
    table_ids: List[str],
    metadata_list: List[Dict],
    num_queries: int = -1,
    hyde_mode: str = "column_desc",
    ablation_type: str = "remove",  # "remove" or "keep"
    bm25_index_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run ablation experiment: remove or keep only primitive class tokens.
    
    For fair comparison, the "remove" mode also builds a clean BM25 index
    with [ClassName] markers removed from document texts, so both query
    and document sides are consistent.

    Args:
        ablation_type: 
            - "remove": Remove [ClassName] markers and measure recall
            - "keep": Keep only [ClassName] tokens for search
        bm25_index_path: Path to BM25 index dir (required for fair ablation)
    """
    if num_queries > 0:
        analysis_data = analysis_data[:num_queries]
    
    stemmer = Stemmer.Stemmer("english")
    
    results = {
        'ablation_type': ablation_type,
        'hyde_mode': hyde_mode,
        'total_queries': len(analysis_data),
        'original_ranks': [],
        'ablated_ranks': [],
    }
    
    # For "remove" mode, build a clean BM25 index with markers removed from docs
    ablated_retriever = bm25_retriever
    ablated_table_ids = table_ids
    if ablation_type == "remove" and bm25_index_path:
        logger.info("Building clean BM25 index (removing [ClassName] from documents)...")
        ablated_retriever, ablated_table_ids = _build_clean_bm25_index(
            bm25_index_path, remove_primitive_class_brackets
        )
    elif ablation_type == "remove" and not bm25_index_path:
        logger.warning("bm25_index_path not provided; ablation will only modify query side (unfair)")
    
    for i, item in enumerate(analysis_data):
        query = item['query']
        gt_tables = item.get('gt_tables', [item['gt_table']] if item.get('gt_table') else [])
        analysis = item.get('analysis', {})
        
        # Raw mode doesn't require analysis
        if hyde_mode != "raw" and (not analysis or item.get('error')):
            results['original_ranks'].append(None)
            results['ablated_ranks'].append(None)
            continue
        
        # Get query/HyDE text
        if hyde_mode == "raw":
            hyde_text = query  # Use raw query as input
        elif hyde_mode == "column_desc":
            hyde_text = analysis.get('hypothetical_column_descriptions', '')
        elif hyde_mode == "table_desc":
            hyde_text = analysis.get('hypothetical_table_description', '')
        elif hyde_mode == "combined":
            table_desc = analysis.get('hypothetical_table_description', '')
            col_desc = analysis.get('hypothetical_column_descriptions', '')
            hyde_text = f"{table_desc}\n{col_desc}"
        else:
            hyde_text = query  # Fallback to raw query
        
        if not hyde_text.strip():
            results['original_ranks'].append(None)
            results['ablated_ranks'].append(None)
            continue
        
        # Create ablated text
        if ablation_type == "remove":
            ablated_text = remove_primitive_class_markers(hyde_text)
        else:  # keep
            ablated_text = keep_only_primitive_class_tokens(hyde_text)
        
        # Search with original text
        # Use min(100, num_docs) to handle small datasets
        num_docs = bm25_retriever.corpus_tokens.vocab.vocab_size if hasattr(bm25_retriever, 'corpus_tokens') else 100
        num_docs = bm25_retriever.scores['num_docs'] if 'num_docs' in bm25_retriever.scores else num_docs
        k = min(100, num_docs)
        
        original_tokens = bm25s.tokenize([hyde_text], stemmer=stemmer, stopwords="en", show_progress=False)
        original_results, original_scores = bm25_retriever.retrieve(original_tokens, k=k, show_progress=False)
        
        # Search with ablated text using the appropriate index
        # For "remove" mode with fair ablation, use the clean index
        if ablated_text.strip():
            ablated_tokens = bm25s.tokenize([ablated_text], stemmer=stemmer, stopwords="en", show_progress=False)
            ablated_k = min(100, ablated_retriever.scores.get('num_docs', k))
            ablated_results, ablated_scores = ablated_retriever.retrieve(ablated_tokens, k=ablated_k, show_progress=False)
        else:
            ablated_results = np.array([[-1] * 100])  # Dummy result with invalid indices
        
        # Find GT ranks
        def find_rank(result_items, gt_tables, tids):
            """Find GT rank in results (handles both index and dict formats)."""
            for rank, item in enumerate(result_items, 1):
                # Handle dict format (when load_corpus=True)
                if isinstance(item, dict):
                    idx = item.get('id')
                else:
                    idx = item
                
                if isinstance(idx, (int, np.integer)) and idx >= 0 and idx < len(tids):
                    if tids[idx] in gt_tables:
                        return rank
            return None
        
        original_rank = find_rank(original_results[0], gt_tables, table_ids)
        ablated_rank = find_rank(ablated_results[0], gt_tables, ablated_table_ids)
        
        results['original_ranks'].append(original_rank)
        results['ablated_ranks'].append(ablated_rank)
        
        # Progress
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(analysis_data)}")
    
    # Compute metrics
    def compute_recall_at_k(ranks, k):
        valid = [r for r in ranks if r is not None]
        return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)
    
    def compute_mrr(ranks):
        rrs = [1.0 / r for r in ranks if r is not None]
        return sum(rrs) / len(ranks) if ranks else 0.0
    
    results['original_metrics'] = {
        'recall@1': compute_recall_at_k(results['original_ranks'], 1),
        'recall@5': compute_recall_at_k(results['original_ranks'], 5),
        'recall@10': compute_recall_at_k(results['original_ranks'], 10),
        'recall@50': compute_recall_at_k(results['original_ranks'], 50),
        'mrr': compute_mrr(results['original_ranks']),
    }
    
    results['ablated_metrics'] = {
        'recall@1': compute_recall_at_k(results['ablated_ranks'], 1),
        'recall@5': compute_recall_at_k(results['ablated_ranks'], 5),
        'recall@10': compute_recall_at_k(results['ablated_ranks'], 10),
        'recall@50': compute_recall_at_k(results['ablated_ranks'], 50),
        'mrr': compute_mrr(results['ablated_ranks']),
    }
    
    # Compute deltas
    results['delta'] = {
        k: results['ablated_metrics'][k] - results['original_metrics'][k]
        for k in results['original_metrics']
    }
    
    return results


def print_contribution_results(results: Dict[str, Any]):
    """Pretty print contribution analysis results."""
    print("\n" + "=" * 80)
    print("📊 BM25 Term Contribution Analysis")
    print("=" * 80)
    
    print(f"\nTotal queries: {results['total_queries']}")
    print(f"Queries with GT in index: {results['queries_with_gt_in_index']}")
    
    if 'stats' in results:
        stats = results['stats']
        print(f"\n📈 Primitive Class Token Contribution Statistics:")
        print(f"   Mean ratio:   {stats['mean_primitive_ratio']*100:.1f}%")
        print(f"   Median ratio: {stats['median_primitive_ratio']*100:.1f}%")
        print(f"   Min ratio:    {stats['min_primitive_ratio']*100:.1f}%")
        print(f"   Max ratio:    {stats['max_primitive_ratio']*100:.1f}%")
        print(f"   Std:          {stats['std_primitive_ratio']*100:.1f}%")
    
    if results.get('top_contributing_classes'):
        print(f"\n🏷️  Top Contributing Primitive Classes:")
        for cls, contrib in list(results['top_contributing_classes'].items())[:10]:
            print(f"   {cls:20s}: {contrib:.2f}")
    
    if results.get('case_analyses'):
        print(f"\n📝 Case Analysis Examples:")
        for case in results['case_analyses'][:3]:
            print(f"\n   ─── Case {case['index']} ───")
            print(f"   Query: {case['query'][:80]}...")
            print(f"   GT: {case['gt_table'][:60]}...")
            print(f"   Primitive tokens: {case['primitive_tokens'][:5]}")
            print(f"   Total score: {case['total_score']:.3f}")
            print(f"   Primitive contribution: {case['primitive_contribution']:.3f} ({case['contribution_ratio']*100:.1f}%)")
            print(f"   Other contribution: {case['other_contribution']:.3f}")
            print(f"   Top terms:")
            for term in case['top_terms'][:5]:
                marker = "🔹" if term['is_primitive'] else "  "
                occ = f" (×{term['occurrences']})" if term.get('occurrences', 1) > 1 else ""
                print(f"      {marker} {term['term']:15s} contrib={term['contribution']:.3f} idf={term['idf']:.2f}{occ}")


def print_ablation_results(results: Dict[str, Any]):
    """Pretty print ablation experiment results."""
    print("\n" + "=" * 80)
    print(f"🔬 Ablation Experiment: {results['ablation_type'].upper()}")
    print("=" * 80)
    
    print(f"\nHyDE mode: {results['hyde_mode']}")
    print(f"Total queries: {results['total_queries']}")
    
    print(f"\n📊 Metrics Comparison:")
    print(f"{'Metric':<15} {'Original':>12} {'Ablated':>12} {'Delta':>12}")
    print("-" * 55)
    
    for metric in ['recall@1', 'recall@5', 'recall@10', 'recall@50', 'mrr']:
        orig = results['original_metrics'][metric]
        ablated = results['ablated_metrics'][metric]
        delta = results['delta'][metric]
        
        if metric == 'mrr':
            print(f"{metric:<15} {orig:>11.4f} {ablated:>11.4f} {delta:>+11.4f}")
        else:
            sign = "+" if delta >= 0 else ""
            print(f"{metric:<15} {orig*100:>10.1f}% {ablated*100:>10.1f}% {sign}{delta*100:>10.1f}pp")


# ==================== Main Entry ====================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze BM25 term contributions for HyDE retrieval"
    )
    parser.add_argument("-d", "--dataset", default="fetaqa", help="Dataset name")
    parser.add_argument("-n", "--num-queries", type=int, default=-1,
                        help="Number of queries to analyze (-1 for all)")
    parser.add_argument("--mode", type=str, default="contribution",
                        choices=["contribution", "ablation-remove", "ablation-keep", "full"],
                        help="Analysis mode")
    parser.add_argument("--hyde-mode", type=str, default="combined",
                        choices=["raw", "table_desc", "column_desc", "combined"],
                        help="HyDE text mode for analysis")
    parser.add_argument("--show-cases", type=int, default=5,
                        help="Number of case analyses to show")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON filename")
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data for dataset: {args.dataset}")
    analysis_data = load_unified_analysis(args.dataset)
    
    # Load indexes using unified interface
    logger.info("Loading indexes...")
    faiss_index, metadata_list, bm25_retriever, table_ids = load_unified_indexes(args.dataset, INDEX_KEY_TD_CD_CS)
    
    if bm25_retriever is None:
        logger.error("BM25 index not found!")
        return
    
    # Get BM25 index path
    from pathlib import Path
    bm25_path = str(get_data_root() / "lake" / "indexes" / args.dataset / INDEX_KEY_TD_CD_CS / "bm25")
    
    # Run analysis based on mode
    if args.mode == "contribution":
        analyzer = BM25TermAnalyzer(bm25_path)
        results = analyze_term_contributions(
            analysis_data, analyzer, metadata_list,
            num_queries=args.num_queries,
            show_cases=args.show_cases,
            hyde_mode=args.hyde_mode,
        )
        print_contribution_results(results)
        
    elif args.mode == "ablation-remove":
        results = run_ablation_experiment(
            analysis_data, bm25_retriever, table_ids, metadata_list,
            num_queries=args.num_queries,
            hyde_mode=args.hyde_mode,
            ablation_type="remove",
            bm25_index_path=bm25_path,
        )
        print_ablation_results(results)
        
    elif args.mode == "ablation-keep":
        results = run_ablation_experiment(
            analysis_data, bm25_retriever, table_ids, metadata_list,
            num_queries=args.num_queries,
            hyde_mode=args.hyde_mode,
            ablation_type="keep",
            bm25_index_path=bm25_path,
        )
        print_ablation_results(results)
        
    elif args.mode == "full":
        # Run all analyses
        analyzer = BM25TermAnalyzer(bm25_path)
        
        print("\n" + "=" * 80)
        print("🔬 FULL BM25 TERM CONTRIBUTION ANALYSIS")
        print("=" * 80)
        
        # 1. Term contribution analysis
        contrib_results = analyze_term_contributions(
            analysis_data, analyzer, metadata_list,
            num_queries=args.num_queries,
            show_cases=args.show_cases,
            hyde_mode=args.hyde_mode,
        )
        print_contribution_results(contrib_results)
        
        # 2. Ablation: remove primitive class markers (fair: clean index)
        remove_results = run_ablation_experiment(
            analysis_data, bm25_retriever, table_ids, metadata_list,
            num_queries=args.num_queries,
            hyde_mode=args.hyde_mode,
            ablation_type="remove",
            bm25_index_path=bm25_path,
        )
        print_ablation_results(remove_results)
        
        # 3. Ablation: keep only primitive class tokens
        keep_results = run_ablation_experiment(
            analysis_data, bm25_retriever, table_ids, metadata_list,
            num_queries=args.num_queries,
            hyde_mode=args.hyde_mode,
            ablation_type="keep",
            bm25_index_path=bm25_path,
        )
        print_ablation_results(keep_results)
        
        results = {
            'contribution': contrib_results,
            'ablation_remove': remove_results,
            'ablation_keep': keep_results,
        }
    
    # Save results
    if args.output:
        output_path = get_db_path() / "eval_results" / args.output
        with open(output_path, 'w') as f:
            # Remove numpy arrays for JSON serialization
            def clean_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, dict):
                    return {k: clean_for_json(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [clean_for_json(v) for v in obj]
                return obj
            
            json.dump(clean_for_json(results), f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
