#!/usr/bin/env python3
"""
Analyze Retrieval Quality - Structural, Semantic, and Hybrid

This script evaluates retrieval performance using multiple approaches:
1. Structural: TBox/ABox constraint-based retrieval (ScorerV3)
2. Semantic: Vector + BM25 hybrid search (FAISS + BM25 + RRF)
3. Hybrid: Combined structural + semantic retrieval

Features:
1. Constraint Quality Analysis:
   - Compare extracted classes vs GT table's actual primitive classes
   - Strict matching and hierarchy-aware matching
   - Precision/recall metrics with detailed breakdown

2. Retrieval Performance Testing:
   - ScorerV3-based structural retrieval
   - Semantic retrieval (FAISS + BM25)
   - Hybrid retrieval with fusion
   - Recall@k metrics (k=1,5,10,50,100)
   - MRR (Mean Reciprocal Rank)
   - Detailed failure case analysis

3. Comparative Analysis:
   - Case-by-case comparison of different methods
   - Identify which method works better for which queries

Usage:
    # Structural retrieval test
    python scripts/eval/analyze_structural_retrieval.py -d fetaqa -n 100 --test-retrieval

    # Semantic retrieval test
    python scripts/eval/analyze_structural_retrieval.py -d fetaqa -n 100 --test-semantic

    # Hybrid retrieval test
    python scripts/eval/analyze_structural_retrieval.py -d fetaqa -n 100 --test-hybrid

    # Compare all methods
    python scripts/eval/analyze_structural_retrieval.py -d fetaqa -n 100 --compare
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, Any, List, Optional, Set, Tuple

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
import _path_setup  # noqa: F401

from loguru import logger
from core.paths import get_db_path
from store.store_singleton import get_store


# ==================== Multi-GT Helper ====================

def find_best_rank_multi_gt(result_ids: List[str], gt_tables: List[str]) -> Optional[int]:
    """Find the best (minimum) rank among all ground truth tables.
    
    Args:
        result_ids: List of table IDs in retrieval order
        gt_tables: List of ground truth table IDs (may be single or multiple)
        
    Returns:
        Best rank (1-indexed) if any GT found, None if none found
    """
    best_rank = None
    for gt in gt_tables:
        try:
            rank = result_ids.index(gt) + 1
            if best_rank is None or rank < best_rank:
                best_rank = rank
        except ValueError:
            continue
    return best_rank


# ==================== Data Loading ====================

def load_constraints_file(
    dataset: str, 
    num_queries: int, 
    source: str = "auto",
    llm: str = "local",
    use_rag: bool = True,
    rag_top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Load constraints from JSON file.
    
    Supports two formats:
    1. Legacy: {dataset}_constraints_{num_queries}.json
    2. Unified analysis: {dataset}_unified_analysis_{num_queries}_{llm}[_rag{k}].json
    
    Args:
        dataset: Dataset name (e.g., 'fetaqa', 'adventure_works')
        num_queries: Number of queries (or 'all')
        source: 'auto' (try unified first, then legacy), 'unified', or 'legacy'
        llm: LLM used for analysis ('local', 'gemini')
        use_rag: Whether RAG-enhanced analysis was used
        rag_top_k: RAG top-k parameter
        
    Returns:
        List of constraint records, normalized to a common format.
    """
    eval_dir = get_db_path() / "eval_results"
    
    # Build possible file paths (in priority order)
    candidates = []
    query_str = "all" if num_queries <= 0 else str(num_queries)
    
    if source in ("auto", "unified"):
        # Unified analysis format (newer, preferred)
        # Support multiple naming patterns:
        # - {dataset}_unified_analysis_{n}_{llm}.json (standard)
        # - {dataset}_test_unified_analysis_{n}_{llm}.json (with split)
        # - {dataset}_{split}_unified_analysis_{n}_{llm}_rag{k}.json (with RAG)
        # - {dataset}_{split}_unified_analysis_{n}_{llm}_rag{k}_{rag_type}.json (with RAG + type)
        
        for split_prefix in ["", "test_", "train_"]:
            if use_rag:
                # Try with rag_type suffix (e.g., _rag3_vector)
                for rag_type in ["vector", "hybrid", "bm25", ""]:
                    suffix = f"_rag{rag_top_k}_{rag_type}" if rag_type else f"_rag{rag_top_k}"
                    candidates.append(eval_dir / f"{dataset}_{split_prefix}unified_analysis_{query_str}_{llm}{suffix}.json")
            candidates.append(eval_dir / f"{dataset}_{split_prefix}unified_analysis_{query_str}_{llm}.json")
            # Try without query count suffix (all queries)
            if use_rag:
                for rag_type in ["vector", "hybrid", "bm25", ""]:
                    suffix = f"_rag{rag_top_k}_{rag_type}" if rag_type else f"_rag{rag_top_k}"
                    candidates.append(eval_dir / f"{dataset}_{split_prefix}unified_analysis_all_{llm}{suffix}.json")
            candidates.append(eval_dir / f"{dataset}_{split_prefix}unified_analysis_all_{llm}.json")
    
    if source in ("auto", "legacy"):
        # Legacy format
        candidates.append(eval_dir / f"{dataset}_constraints_{num_queries}.json")
    
    # Find first existing file
    filepath = None
    for candidate in candidates:
        if candidate.exists():
            filepath = candidate
            break
    
    if filepath is None:
        raise FileNotFoundError(
            f"No constraints file found. Tried:\n" +
            "\n".join(f"  - {p}" for p in candidates) +
            f"\n\nGenerate with:\n"
            f"  python demos/retrieval.py --analyze-queries -d {dataset} -n {num_queries} --llm {llm}" +
            (f" --use-rag --rag-top-k {rag_top_k}" if use_rag else "")
        )
    
    logger.info(f"Loading constraints from: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Normalize to common format
    return _normalize_constraints_data(raw_data)


def _normalize_constraints_data(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize constraints data from different formats to a common format.
    
    Handles two formats:
    1. Legacy format:
       {
         "query": "...",
         "gt_table": "...",
         "tbox_constraints": ["Class1", "Class2"],
         "abox_constraints": {
           "entity_constraints": [{"column_type": "X", "value": "v"}],
           "same_row_required": true
         }
       }
    
    2. Unified analysis format:
       {
         "query": "...",
         "gt_table": "...",
         "analysis": {
           "tbox_constraints": ["Class1Column", "Class2Column"],
           "abox_constraints": [{"column_type": "X", "value": "v"}],
           ...
         }
       }
    
    Returns normalized format matching legacy structure.
    """
    normalized = []
    
    for record in raw_data:
        # Skip records with errors
        if record.get('error'):
            continue
        
        # Parse gt_tables (support both single and multi-GT)
        gt_tables = record.get('gt_tables')
        if not gt_tables:
            # Fall back to single gt_table
            gt_tables = [record['gt_table']] if record.get('gt_table') else []
        
        # Check if it's unified analysis format (has 'analysis' key)
        if 'analysis' in record and record['analysis'] is not None:
            analysis = record['analysis']
            norm_record = {
                'query': record['query'],
                'gt_table': gt_tables[0] if gt_tables else None,  # Backward compatible
                'gt_tables': gt_tables,  # Multi-GT support
                'tbox_constraints': analysis.get('tbox_constraints', []),
                'abox_constraints': _normalize_abox_constraints(
                    analysis.get('abox_constraints', [])
                ),
            }
        else:
            # Legacy format - already normalized
            norm_record = {
                'query': record['query'],
                'gt_table': gt_tables[0] if gt_tables else record.get('gt_table'),
                'gt_tables': gt_tables,
                'tbox_constraints': record.get('tbox_constraints', []),
                'abox_constraints': record.get('abox_constraints', {
                    'entity_constraints': [],
                    'same_row_required': False
                }),
            }
        
        normalized.append(norm_record)
    
    return normalized


def _normalize_abox_constraints(abox_constraints: Any) -> Dict[str, Any]:
    """Normalize ABox constraints to legacy format.
    
    Unified analysis format: [{"column_type": "X", "value": "v"}, ...]
    Legacy format: {"entity_constraints": [...], "same_row_required": bool}
    """
    # Already in legacy format
    if isinstance(abox_constraints, dict):
        return abox_constraints
    
    # Convert from unified analysis format (list)
    if isinstance(abox_constraints, list):
        # Filter out entries without values
        entity_constraints = [
            {"column_type": c.get('column_type'), "value": c.get('value')}
            for c in abox_constraints
            if c.get('value') is not None and c.get('value') != ""
        ]
        return {
            'entity_constraints': entity_constraints,
            'same_row_required': False,  # Default assumption
        }
    
    # Fallback
    return {'entity_constraints': [], 'same_row_required': False}


def load_class_hierarchy(dataset: str) -> Tuple[Dict[str, List[str]], Dict[str, Set[str]]]:
    """
    Load class hierarchy from ontology_classes table for the given dataset.
    
    Returns:
        Tuple of:
        - child_to_parents: {child_class: [parent_classes]}
        - child_to_ancestors: {child_class: {all_ancestor_classes}}
    """
    store = get_store()
    
    try:
        # Get latest ontology_id for this dataset
        meta_tbl = store.db.open_table('ontology_metadata')
        meta_df = meta_tbl.to_pandas()
        
        # Filter for federated_primitive_tbox of this dataset
        dataset_df = meta_df[
            (meta_df['ontology_type'] == 'federated_primitive_tbox') & 
            (meta_df['dataset_name'] == dataset)
        ]
        
        if len(dataset_df) == 0:
            logger.warning(f"No ontology found for dataset {dataset}")
            return {}, {}
        
        # Get latest version
        latest = dataset_df.sort_values('created_at', ascending=False).iloc[0]
        ontology_id = latest['ontology_id']
        logger.info(f"Using ontology: {ontology_id}")
        
        # Get classes for this ontology
        class_tbl = store.db.open_table('ontology_classes')
        class_df = class_tbl.to_pandas()
        classes_df = class_df[class_df['ontology_id'] == ontology_id]
        
        # Build child -> parents mapping
        child_to_parents = {}
        for _, row in classes_df.iterrows():
            class_name = row['class_name'].replace('upo:', '')
            parents = row['parent_classes']
            
            if parents is not None:
                parent_list = list(parents) if hasattr(parents, '__iter__') else [parents]
                parent_list = [p.replace('upo:', '') for p in parent_list if p and p != 'null']
                if parent_list:
                    child_to_parents[class_name] = parent_list
        
        # Build child -> all ancestors mapping (transitive closure)
        child_to_ancestors = {}
        
        def get_ancestors(cls: str, visited: Set[str] = None) -> Set[str]:
            if visited is None:
                visited = set()
            if cls in visited:
                return set()
            visited.add(cls)
            
            ancestors = set()
            if cls in child_to_parents:
                for parent in child_to_parents[cls]:
                    ancestors.add(parent)
                    ancestors.update(get_ancestors(parent, visited.copy()))
            return ancestors
        
        for class_name in classes_df['class_name']:
            clean_name = class_name.replace('upo:', '')
            child_to_ancestors[clean_name] = get_ancestors(clean_name)
        
        logger.info(f"Loaded class hierarchy: {len(child_to_parents)} classes with parents")
        return child_to_parents, child_to_ancestors
        
    except Exception as e:
        logger.warning(f"Failed to load class hierarchy: {e}")
        return {}, {}


def load_column_mappings(dataset: str, table_id: str) -> List[Dict[str, Any]]:
    """Load column mappings for a specific table."""
    store = get_store()
    table_name = f"{dataset}_column_mappings"
    
    try:
        tbl = store.db.open_table(table_name)
        escaped_table_id = table_id.replace("'", "''")
        result = tbl.search().where(f"source_table = '{escaped_table_id}'", prefilter=True).to_pandas()
        return result.to_dict('records')
    except Exception as e:
        logger.warning(f"Failed to load column mappings: {e}")
        return []


def load_table_raw_data(dataset: str, table_id: str, max_rows: int = 3) -> Optional[Dict[str, Any]]:
    """Load raw table data (title and sample rows) from tables_entries."""
    import json as json_module
    
    store = get_store()
    table_name = f"{dataset}_tables_entries"
    
    try:
        tbl = store.db.open_table(table_name)
        escaped_table_id = table_id.replace("'", "''")
        result = tbl.search().where(f"table_id = '{escaped_table_id}'", prefilter=True).limit(1).to_pandas()
        
        if len(result) == 0:
            return None
        
        row = result.iloc[0]
        
        doc_title = row.get('document_title', '')
        sec_title = row.get('section_title', '')
        title = f"{doc_title} - {sec_title}" if sec_title else doc_title
        
        table_info = {
            'title': title,
            'headers': [],
            'rows': [],
            'row_count': row.get('row_count', 0),
            'column_count': row.get('column_count', 0),
        }
        
        # Get columns (headers)
        columns = row.get('columns')
        if columns is not None:
            if isinstance(columns, str):
                try:
                    table_info['headers'] = json_module.loads(columns)
                except:
                    table_info['headers'] = [columns]
            elif hasattr(columns, 'tolist'):
                table_info['headers'] = columns.tolist()
            elif isinstance(columns, list):
                table_info['headers'] = columns
        
        # Get sample rows
        sample_rows = row.get('sample_rows')
        if sample_rows is not None:
            if isinstance(sample_rows, str):
                try:
                    table_info['rows'] = json_module.loads(sample_rows)[:max_rows]
                except:
                    pass
            elif hasattr(sample_rows, 'tolist'):
                table_info['rows'] = sample_rows.tolist()[:max_rows]
            elif isinstance(sample_rows, list):
                table_info['rows'] = sample_rows[:max_rows]
        
        return table_info
    except Exception as e:
        logger.warning(f"Failed to load table raw data: {e}")
        return None


# ==================== Analysis Functions ====================

def check_class_match_with_hierarchy(
    required_class: str,
    actual_classes: Set[str],
    child_to_ancestors: Dict[str, Set[str]],
) -> Tuple[bool, str]:
    """
    Check if required_class matches any actual_class considering inheritance.
    
    Returns:
        Tuple of (is_match, match_type)
        - match_type: 'exact', 'subclass', 'superclass', or 'none'
    """
    # Exact match
    if required_class in actual_classes:
        return True, 'exact'
    
    # Check if any actual_class is a subclass of required_class
    # (actual_class is more specific than required)
    for actual in actual_classes:
        if actual in child_to_ancestors and required_class in child_to_ancestors[actual]:
            return True, 'subclass'
    
    # Check if required_class is a subclass of any actual_class
    # (required is more specific than actual - less ideal but still a match)
    if required_class in child_to_ancestors:
        for ancestor in child_to_ancestors[required_class]:
            if ancestor in actual_classes:
                return True, 'superclass'
    
    return False, 'none'


def analyze_single_query(
    query_data: Dict[str, Any],
    dataset: str,
    child_to_ancestors: Dict[str, Set[str]],
    valid_classes: Set[str],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Analyze a single query's constraints against its GT table.
    
    Computes both strict matching and hierarchy-aware matching.
    """
    query = query_data['query']
    gt_table = query_data['gt_table']
    
    # New format: tbox_constraints is List[str]
    tbox_constraints = query_data.get('tbox_constraints', [])
    abox_constraints = query_data.get('abox_constraints', {})
    
    analysis = {
        'query': query,
        'gt_table': gt_table,
        'tbox_analysis': None,
        'abox_analysis': None,
        'insights': [],
    }
    
    # Load column mappings for the table
    column_mappings = load_column_mappings(dataset, gt_table)
    
    if not column_mappings:
        analysis['insights'].append(f"Could not find column mappings for GT table")
        return analysis
    
    # Extract actual primitive classes
    actual_classes_list = []  # With repetition for TF
    actual_classes_set = set()  # Unique
    for cm in column_mappings:
        if cm.get('primitive_class'):
            class_name = cm['primitive_class'].replace('upo:', '').replace('ont:', '')
            actual_classes_list.append(class_name)
            actual_classes_set.add(class_name)
    
    actual_classes = actual_classes_set
    
    # Analyze TBox constraints
    if tbox_constraints:
        required_classes = [c.replace('upo:', '').replace('ont:', '') for c in tbox_constraints]
        
        # Check for invalid constraints (not in ontology)
        invalid_constraints = [c for c in required_classes if c not in valid_classes]
        valid_required = [c for c in required_classes if c in valid_classes]
        
        # Strict matching (no inheritance)
        strict_matched = [c for c in valid_required if c in actual_classes]
        strict_not_matched = [c for c in valid_required if c not in actual_classes]
        
        # Hierarchy-aware matching
        hierarchy_matches = []
        hierarchy_not_matched = []
        match_details = []
        
        for req_class in valid_required:
            is_match, match_type = check_class_match_with_hierarchy(
                req_class, actual_classes, child_to_ancestors
            )
            match_details.append({
                'class': req_class,
                'matched': is_match,
                'match_type': match_type,
            })
            if is_match:
                hierarchy_matches.append(req_class)
            else:
                hierarchy_not_matched.append(req_class)
        
        analysis['tbox_analysis'] = {
            'required_classes': required_classes,
            'valid_required': valid_required,
            'invalid_constraints': invalid_constraints,
            'actual_classes': list(actual_classes),
            'actual_classes_with_tf': actual_classes_list,
            # Strict matching
            'strict_matched': strict_matched,
            'strict_not_matched': strict_not_matched,
            'strict_precision': len(strict_matched) / len(valid_required) if valid_required else 1.0,
            'strict_recall': len(strict_matched) / len(actual_classes) if actual_classes else 1.0,
            # Hierarchy-aware matching
            'hierarchy_matched': hierarchy_matches,
            'hierarchy_not_matched': hierarchy_not_matched,
            'hierarchy_precision': len(hierarchy_matches) / len(valid_required) if valid_required else 1.0,
            'hierarchy_recall': len(hierarchy_matches) / len(actual_classes) if actual_classes else 1.0,
            'match_details': match_details,
        }
        
        if verbose:
            # Show table preview
            table_raw = load_table_raw_data(dataset, gt_table, max_rows=3)
            if table_raw:
                print(f"\n📋 GT Table Preview:")
                print(f"   Title: {table_raw['title']}")
                if table_raw['headers']:
                    print(f"   Headers: {table_raw['headers']}")
            
            # Column type mapping
            print(f"\n🏷️  Column-to-Class Mapping:")
            for cm in column_mappings[:10]:
                col_name = cm.get('column_name', 'Unknown')
                prim_class = cm.get('primitive_class', 'Unknown')
                print(f"     {col_name}: {prim_class}")
            if len(column_mappings) > 10:
                print(f"     ... ({len(column_mappings) - 10} more columns)")
            
            print(f"\n📦 TBox Constraint Analysis:")
            print(f"   Required: {required_classes}")
            print(f"   GT has: {list(actual_classes)}")
            print(f"   Strict Match: {strict_matched}")
            print(f"   Hierarchy Match: {hierarchy_matches}")
            print(f"   Not Matched: {hierarchy_not_matched}")
            if invalid_constraints:
                print(f"   ⚠️  Invalid: {invalid_constraints}")
    
    # Analyze ABox constraints
    if abox_constraints:
        entity_constraints = abox_constraints.get('entity_constraints', [])
        
        analysis['abox_analysis'] = {
            'entities': [e.get('value') for e in entity_constraints],
            'count': len(entity_constraints),
            'same_row_required': abox_constraints.get('same_row_required', True),
        }
        
        if verbose and entity_constraints:
            print(f"\n📄 ABox Constraints:")
            for ec in entity_constraints:
                print(f"     '{ec.get('value')}' in {ec.get('column_type')}")
    
    return analysis


def analyze_all_queries(
    dataset: str,
    num_queries: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Analyze all queries and generate comprehensive statistics."""
    
    # Load data
    constraints_data = load_constraints_file(dataset, num_queries)
    child_to_parents, child_to_ancestors = load_class_hierarchy(dataset)
    
    # Get set of valid classes
    valid_classes = set(child_to_ancestors.keys())
    valid_classes.add('Column')  # Root is always valid
    
    print(f"\n{'='*80}")
    print(f"Analyzing {len(constraints_data)} queries for dataset: {dataset}")
    print(f"Class hierarchy loaded: {len(valid_classes)} classes")
    print(f"{'='*80}")
    
    all_analyses = []
    
    for i, query_data in enumerate(constraints_data):
        analysis = analyze_single_query(
            query_data, dataset, child_to_ancestors, valid_classes, verbose=verbose
        )
        all_analyses.append(analysis)
        
        if not verbose and (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(constraints_data)} queries...")
    
    # Compute aggregate statistics
    strict_precisions = [a['tbox_analysis']['strict_precision'] for a in all_analyses if a.get('tbox_analysis')]
    strict_recalls = [a['tbox_analysis']['strict_recall'] for a in all_analyses if a.get('tbox_analysis')]
    hierarchy_precisions = [a['tbox_analysis']['hierarchy_precision'] for a in all_analyses if a.get('tbox_analysis')]
    hierarchy_recalls = [a['tbox_analysis']['hierarchy_recall'] for a in all_analyses if a.get('tbox_analysis')]
    
    # Collect mismatch patterns
    strict_mismatches = []
    hierarchy_mismatches = []
    match_type_counts = Counter()
    invalid_constraint_counts = Counter()
    extracted_class_counts = Counter()
    
    for a in all_analyses:
        if a.get('tbox_analysis'):
            tbox = a['tbox_analysis']
            strict_mismatches.extend(tbox['strict_not_matched'])
            hierarchy_mismatches.extend(tbox['hierarchy_not_matched'])
            for detail in tbox.get('match_details', []):
                match_type_counts[detail['match_type']] += 1
            invalid_constraint_counts.update(tbox.get('invalid_constraints', []))
            extracted_class_counts.update(tbox['required_classes'])
    
    summary = {
        'total_queries': len(constraints_data),
        'class_hierarchy_size': len(valid_classes),
        'tbox_strict': {
            'avg_precision': mean(strict_precisions) if strict_precisions else 0,
            'avg_recall': mean(strict_recalls) if strict_recalls else 0,
            'mismatch_count': len([a for a in all_analyses if a.get('tbox_analysis') and a['tbox_analysis']['strict_not_matched']]),
        },
        'tbox_hierarchy': {
            'avg_precision': mean(hierarchy_precisions) if hierarchy_precisions else 0,
            'avg_recall': mean(hierarchy_recalls) if hierarchy_recalls else 0,
            'mismatch_count': len([a for a in all_analyses if a.get('tbox_analysis') and a['tbox_analysis']['hierarchy_not_matched']]),
        },
        'match_type_distribution': dict(match_type_counts),
    }
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n📊 TBox Constraint Quality (Strict Matching):")
    print(f"   Average Precision: {summary['tbox_strict']['avg_precision']:.2%}")
    print(f"   Average Recall: {summary['tbox_strict']['avg_recall']:.2%}")
    print(f"   Queries with mismatched classes: {summary['tbox_strict']['mismatch_count']}/{len(constraints_data)}")
    
    print(f"\n📊 TBox Constraint Quality (Hierarchy-Aware Matching):")
    print(f"   Average Precision: {summary['tbox_hierarchy']['avg_precision']:.2%}")
    print(f"   Average Recall: {summary['tbox_hierarchy']['avg_recall']:.2%}")
    print(f"   Queries with mismatched classes: {summary['tbox_hierarchy']['mismatch_count']}/{len(constraints_data)}")
    
    print(f"\n📊 Match Type Distribution:")
    total_matches = sum(match_type_counts.values())
    for match_type, count in sorted(match_type_counts.items(), key=lambda x: -x[1]):
        pct = count / total_matches * 100 if total_matches else 0
        print(f"   {match_type}: {count} ({pct:.1f}%)")
    
    # Top extracted classes
    print(f"\n📊 Top 15 Extracted Classes:")
    for cls, cnt in extracted_class_counts.most_common(15):
        print(f"   {cls}: {cnt}")
    
    # Most common mismatches
    if strict_mismatches:
        print(f"\n❌ Most Common Strict Mismatches (LLM required but GT doesn't have):")
        for class_name, count in Counter(strict_mismatches).most_common(10):
            print(f"   {class_name}: {count} times")
    
    if hierarchy_mismatches:
        print(f"\n❌ Most Common Hierarchy Mismatches:")
        for class_name, count in Counter(hierarchy_mismatches).most_common(10):
            print(f"   {class_name}: {count} times")
    
    if invalid_constraint_counts:
        print(f"\n⚠️  Invalid Constraints (LLM generated but not in ontology):")
        for class_name, count in invalid_constraint_counts.most_common(10):
            print(f"   {class_name}: {count} times")
    
    # Precision distribution
    print(f"\n📊 Strict Precision Distribution:")
    print(f"   100%: {sum(1 for p in strict_precisions if p == 1.0)} queries")
    print(f"   75-99%: {sum(1 for p in strict_precisions if 0.75 <= p < 1.0)} queries")
    print(f"   50-74%: {sum(1 for p in strict_precisions if 0.5 <= p < 0.75)} queries")
    print(f"   25-49%: {sum(1 for p in strict_precisions if 0.25 <= p < 0.5)} queries")
    print(f"   <25%: {sum(1 for p in strict_precisions if p < 0.25)} queries")
    
    print(f"\n📊 Hierarchy-Aware Precision Distribution:")
    print(f"   100%: {sum(1 for p in hierarchy_precisions if p == 1.0)} queries")
    print(f"   75-99%: {sum(1 for p in hierarchy_precisions if 0.75 <= p < 1.0)} queries")
    print(f"   50-74%: {sum(1 for p in hierarchy_precisions if 0.5 <= p < 0.75)} queries")
    print(f"   25-49%: {sum(1 for p in hierarchy_precisions if 0.25 <= p < 0.5)} queries")
    print(f"   <25%: {sum(1 for p in hierarchy_precisions if p < 0.25)} queries")
    
    # Save detailed results
    output_dir = get_db_path() / "eval_results"
    output_file = output_dir / f"{dataset}_constraints_analysis_{num_queries}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'analyses': all_analyses,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed analysis saved to: {output_file}")
    
    return {
        'summary': summary,
        'analyses': all_analyses,
    }


# ==================== Retrieval Testing ====================

def test_retrieval_recall(
    constraints_data: List[Dict[str, Any]],
    dataset: str,
    top_k: int = 100,
    show_failures: int = 3,
) -> Dict[str, Any]:
    """
    Test retrieval recall using ScorerV3.
    
    For each query, extract constraints from LLM output and retrieve top-k tables.
    Check if GT table appears in results and at what rank.
    """
    from workflows.retrieval.matcher import ScorerV3
    from workflows.retrieval.matcher.constraints import ConstraintSet, PathConstraint, TBoxConstraint, ABoxConstraint
    
    print(f"\n{'='*80}")
    print(f"🔍 Retrieval Test (ScorerV3): {len(constraints_data)} queries, top_k={top_k}")
    print(f"{'='*80}")
    
    # Create scorer
    print(f"\n📦 Loading ScorerV3...")
    import time
    start = time.time()
    scorer = ScorerV3(dataset=dataset)
    
    load_time = time.time() - start
    print(f"   Scorer loaded in {load_time:.2f}s")
    
    # Track metrics
    gt_found_at_rank: List[Optional[int]] = []
    failures: List[Dict[str, Any]] = []
    abox_used_count = 0
    
    # Test each query
    print(f"\n🚀 Testing retrieval...")
    for i, query_data in enumerate(constraints_data):
        # New format: tbox_constraints is List[str]
        tbox_list = query_data.get('tbox_constraints', [])
        abox_constraints = query_data.get('abox_constraints', {})
        entity_constraints = abox_constraints.get('entity_constraints', [])
        
        if not tbox_list:
            gt_found_at_rank.append(None)
            failures.append({
                'index': i + 1,
                'query': query_data['query'],
                'gt_table': query_data['gt_table'],
                'reason': 'No constraints extracted',
            })
            continue
        
        # Build PathConstraints with optional ABox
        # Create a map from column_type to entity value for efficient lookup
        entity_value_map = {}
        for entity in entity_constraints:
            col_type = entity.get('column_type', '')
            value = entity.get('value')
            if col_type and value:
                entity_value_map[col_type] = str(value)
        
        path_constraints = []
        for tbox_class in tbox_list:
            tbox = TBoxConstraint(class_name=tbox_class)
            
            # Match ABox entity to TBox class by column_type
            abox = None
            if tbox_class in entity_value_map:
                abox = ABoxConstraint(value=entity_value_map[tbox_class])
                abox_used_count += 1
            
            path_constraints.append(PathConstraint(tbox=tbox, abox=abox))
        
        constraints = ConstraintSet(constraints=path_constraints)
        
        # Retrieve tables using ScorerV3
        gt_tables = query_data.get('gt_tables', [query_data['gt_table']])
        all_results = scorer.retrieve(constraints, score_threshold=0.0)
        results = all_results[:top_k]
        
        # Find best GT table rank (any GT match counts)
        result_ids = [r[0] for r in results]
        rank = find_best_rank_multi_gt(result_ids, gt_tables)
        
        gt_found_at_rank.append(rank)
        
        if rank is None:
            # Build detailed ABox constraints info
            abox_details = []
            for pc in path_constraints:
                if pc.abox:
                    abox_details.append({
                        'column_type': pc.tbox.class_name,
                        'value': pc.abox.value
                    })
            
            # Find GT table score details from all_results (check all GTs)
            gt_score_details = None
            for tid, score, details in all_results:
                if tid in gt_tables:
                    gt_score_details = {
                        'table_id': tid,
                        'score': score,
                        'constraint_details': details,
                    }
                    break
            
            failures.append({
                'index': i + 1,
                'query': query_data['query'],
                'gt_table': gt_tables[0],  # Primary GT for display
                'gt_tables': gt_tables,    # All GT tables
                'reason': f'GT table not in top-{top_k}',
                'tbox_constraints': tbox_list,
                'abox_constraints': abox_details,
                'top_5': [(r[0], r[1], r[2]) for r in results[:5]],  # Include details
                'gt_score_details': gt_score_details,
            })
        
        # Progress
        if (i + 1) % 20 == 0:
            found_so_far = sum(1 for r in gt_found_at_rank if r is not None)
            print(f"   Processed {i+1}/{len(constraints_data)}, recall@{top_k}: {found_so_far}/{i+1} ({found_so_far/(i+1)*100:.1f}%)")
    
    print(f"\n   ABox constraints used: {abox_used_count}")
    
    # Count multi-GT queries
    multi_gt_count = sum(1 for q in constraints_data if len(q.get('gt_tables', [q['gt_table']])) > 1)
    
    # Calculate metrics
    total = len(constraints_data)
    
    def recall_at_k(k: int) -> float:
        return sum(1 for r in gt_found_at_rank if r is not None and r <= k) / total
    
    def mrr() -> float:
        rrs = [1.0/r for r in gt_found_at_rank if r is not None]
        return sum(rrs) / total if rrs else 0.0
    
    metrics = {
        'method': 'structural',
        'total_queries': total,
        'multi_gt_queries': multi_gt_count,
        'top_k': top_k,
        'recall@1': recall_at_k(1),
        'recall@5': recall_at_k(5),
        'recall@10': recall_at_k(10),
        'recall@50': recall_at_k(50),
        'recall@100': recall_at_k(100),
        'mrr': mrr(),
        'not_found': sum(1 for r in gt_found_at_rank if r is None),
        'ranks': gt_found_at_rank,
    }
    
    # Print metrics
    print(f"\n{'='*80}")
    print(f"📊 Retrieval Metrics (n={total}, multi-GT: {multi_gt_count})")
    print(f"{'='*80}")
    print(f"   Recall@1:   {metrics['recall@1']*100:.1f}%")
    print(f"   Recall@5:   {metrics['recall@5']*100:.1f}%")
    print(f"   Recall@10:  {metrics['recall@10']*100:.1f}%")
    print(f"   Recall@50:  {metrics['recall@50']*100:.1f}%")
    print(f"   Recall@100: {metrics['recall@100']*100:.1f}%")
    print(f"   MRR:        {metrics['mrr']:.4f}")
    print(f"   Not Found:  {metrics['not_found']} ({metrics['not_found']/total*100:.1f}%)")
    
    # Rank distribution
    ranks = [r for r in gt_found_at_rank if r is not None]
    if ranks:
        print(f"\n   Rank Distribution:")
        print(f"     Rank 1:      {sum(1 for r in ranks if r == 1)}")
        print(f"     Rank 2-5:    {sum(1 for r in ranks if 2 <= r <= 5)}")
        print(f"     Rank 6-10:   {sum(1 for r in ranks if 6 <= r <= 10)}")
        print(f"     Rank 11-50:  {sum(1 for r in ranks if 11 <= r <= 50)}")
        print(f"     Rank 51-100: {sum(1 for r in ranks if 51 <= r <= 100)}")
    
    # Show failure cases with detailed info
    if show_failures > 0 and failures:
        print(f"\n{'='*80}")
        print(f"❌ Failed Cases (showing {min(show_failures, len(failures))} of {len(failures)})")
        print(f"{'='*80}")
        
        for case in failures[:show_failures]:
            print(f"\n{'─'*80}")
            print(f"  📌 Case {case['index']}")
            print(f"{'─'*80}")
            
            # Query
            print(f"\n  📝 Query:")
            print(f"     {case['query']}")
            
            # Constraints
            print(f"\n  🎯 TBox Constraints: {case.get('tbox_constraints', [])}")
            abox_list = case.get('abox_constraints', [])
            if abox_list:
                print(f"  🏷️  ABox Constraints:")
                for ab in abox_list:
                    print(f"       {ab['column_type']}: \"{ab['value']}\"")
            else:
                print(f"  🏷️  ABox Constraints: (none)")
            
            # GT Table info
            gt_table = case['gt_table']
            print(f"\n  📋 Ground Truth Table: {gt_table}")
            gt_raw = load_table_raw_data(dataset, gt_table, max_rows=3)
            if gt_raw:
                print(f"     Title: {gt_raw['title']}")
                if gt_raw.get('headers'):
                    print(f"     Headers: {gt_raw['headers']}")
                if gt_raw.get('rows'):
                    for ri, row in enumerate(gt_raw['rows'][:2], 1):
                        row_str = str(row)[:120] + "..." if len(str(row)) > 120 else str(row)
                        print(f"     Row {ri}: {row_str}")
            
            # GT Table column mappings
            gt_mappings = load_column_mappings(dataset, gt_table)
            if gt_mappings:
                print(f"     Primitive Classes: ", end="")
                classes = [cm.get('primitive_class', '').replace('upo:', '') for cm in gt_mappings if cm.get('primitive_class')]
                print(f"{classes}")
            
            # GT Table constraint score details
            gt_details = case.get('gt_score_details')
            if gt_details:
                print(f"\n  📊 GT Table Score Breakdown (total={gt_details['score']:.4f}):")
                for cd in gt_details.get('constraint_details', []):
                    req = cd.get('required', 'Unknown')
                    score = cd.get('score', 0.0)
                    weight = cd.get('weight', 1.0)
                    has_abox = cd.get('has_abox', False)
                    best_anc = cd.get('best_ancestor', cd.get('deepest_matched', '-'))
                    inter = cd.get('intersection', [])
                    reason = cd.get('reason', '')
                    
                    abox_mark = "(+ABox)" if has_abox else "(TBox)"
                    inter_str = ','.join(inter[:3]) + ('...' if len(inter) > 3 else '') if inter else 'empty'
                    
                    if reason == 'empty_intersection':
                        print(f"       • {req} {abox_mark}: score={score:.4f}, weight={weight:.2f}, ❌ NO MATCH (intersection empty)")
                    else:
                        print(f"       • {req} {abox_mark}: score={score:.4f}, weight={weight:.2f}, best={best_anc}, inter=[{inter_str}]")
            else:
                print(f"\n  📊 GT Table Score: (not found in results)")
            
            # Top retrieved (false recalls)
            if 'top_5' in case:
                print(f"\n  ❌ Top Retrieved (False Recalls):")
                for j, item in enumerate(case['top_5'][:3], 1):
                    tid, score = item[0], item[1]
                    print(f"\n     {j}. {tid} (score={score:.4f})")
                    tr_raw = load_table_raw_data(dataset, tid, max_rows=2)
                    if tr_raw:
                        print(f"        Title: {tr_raw['title']}")
                        if tr_raw.get('headers'):
                            hdrs = tr_raw['headers'][:8]
                            hdrs_str = str(hdrs) if len(hdrs) <= 6 else str(hdrs[:6]) + "..."
                            print(f"        Headers: {hdrs_str}")
                        if tr_raw.get('rows') and tr_raw['rows']:
                            row0 = tr_raw['rows'][0]
                            row_str = str(row0)[:100] + "..." if len(str(row0)) > 100 else str(row0)
                            print(f"        Row 1: {row_str}")
    
    return metrics


def print_detailed_case(
    query_data: Dict[str, Any],
    analysis: Dict[str, Any],
    case_idx: int,
    dataset: str,
) -> None:
    """Print a detailed analysis case for human inspection."""
    print(f"\n{'='*80}")
    print(f"📍 CASE {case_idx}")
    print(f"{'='*80}")
    
    print(f"\n📝 Query:")
    print(f"   {query_data['query']}")
    
    print(f"\n📋 GT Table ID: {query_data['gt_table']}")
    
    # Load and show table preview
    table_raw = load_table_raw_data(dataset, query_data['gt_table'])
    if table_raw:
        print(f"\n📊 Table Preview:")
        print(f"   Title: {table_raw['title']}")
        if table_raw.get('headers'):
            print(f"   Headers: {table_raw['headers']}")
        if table_raw.get('rows'):
            for i, row in enumerate(table_raw['rows'][:3], 1):
                row_str = str(row)[:150] + "..." if len(str(row)) > 150 else str(row)
                print(f"   Row {i}: {row_str}")
    
    # Column mappings
    column_mappings = load_column_mappings(dataset, query_data['gt_table'])
    if column_mappings:
        print(f"\n🏷️  Column → Primitive Class Mapping (GT):")
        for cm in column_mappings:
            col_name = cm.get('column_name', 'Unknown')
            pclass = cm.get('primitive_class', 'None')
            if pclass:
                pclass = pclass.replace('upo:', '').replace('ont:', '')
            print(f"     {col_name}: {pclass}")
    
    # TBox analysis
    tbox = analysis.get('tbox_analysis')
    if tbox:
        print(f"\n📦 TBox Constraint Analysis:")
        print(f"   LLM Required: {tbox['required_classes']}")
        print(f"   GT Actual: {tbox['actual_classes']}")
        
        if tbox.get('invalid_constraints'):
            print(f"   ⚠️  Invalid: {tbox['invalid_constraints']}")
        
        print(f"\n   Strict Matching:")
        print(f"     Matched: {tbox['strict_matched']}")
        print(f"     Not Matched: {tbox['strict_not_matched']}")
        print(f"     Precision: {tbox['strict_precision']*100:.1f}%")
        
        print(f"\n   Hierarchy-Aware Matching:")
        print(f"     Matched: {tbox['hierarchy_matched']}")
        print(f"     Not Matched: {tbox['hierarchy_not_matched']}")
        print(f"     Precision: {tbox['hierarchy_precision']*100:.1f}%")
        
        print(f"\n   Match Details:")
        for detail in tbox.get('match_details', []):
            status = "✅" if detail['matched'] else "❌"
            print(f"     {status} {detail['class']} -> {detail['match_type']}")
    
    # ABox analysis  
    abox = analysis.get('abox_analysis')
    if abox:
        print(f"\n📄 ABox Constraints:")
        print(f"   Entities: {abox.get('entities', [])}")
        print(f"   Same Row Required: {abox.get('same_row_required', True)}")


# ==================== Semantic Retrieval Testing ====================

def test_semantic_retrieval(
    dataset: str,
    num_queries: int,
    top_k: int = 100,
    show_failures: int = 3,
    split: str = "test",
) -> Dict[str, Any]:
    """
    Test semantic retrieval recall using FAISS + BM25 hybrid search.
    
    This mirrors the semantic path in demos/retrieval.py.
    
    Args:
        dataset: Dataset name
        num_queries: Number of queries to test
        top_k: Top-K for retrieval
        show_failures: Number of failures to show
        split: Query split to use: 'test', 'train', or 'entries'
    """
    import numpy as np
    from workflows.retrieval.unified_search import (
        load_unified_indexes, vector_search, bm25_search, reciprocal_rank_fusion, get_text_embedder
    )
    from workflows.retrieval.config import INDEX_KEY_TD_CD_CS
    from store.store_singleton import get_store
    
    print(f"\n{'='*80}")
    print(f"🔍 Semantic Retrieval Test (FAISS + BM25): top_k={top_k}")
    print(f"{'='*80}")
    
    # Load indexes using unified interface
    print(f"\n📦 Loading indexes...")
    import time
    start = time.time()
    faiss_index, metadata_list, bm25_retriever, table_ids = load_unified_indexes(dataset, INDEX_KEY_TD_CD_CS)
    load_time = time.time() - start
    
    if faiss_index is None and bm25_retriever is None:
        print(f"❌ No indexes found. Run: python demos/retrieval.py --generate-index -d {dataset}")
        return {}
    
    print(f"   FAISS: {faiss_index.ntotal if faiss_index else 0} vectors")
    print(f"   BM25: {len(table_ids)} documents")
    print(f"   Loaded in {load_time:.2f}s")
    
    # Load embedder
    embedder = get_text_embedder()
    
    # Load queries - use the correct table based on split
    store = get_store()
    if split == "test":
        query_table = f"{dataset}_test_queries"
    else:
        # Default to train queries
        query_table = f"{dataset}_train_queries"
    tbl = store.db.open_table(query_table)
    df = tbl.to_pandas()
    
    # Handle num_queries: -1 or 0 means all queries
    if num_queries > 0:
        queries_df = df.head(num_queries)
    else:
        queries_df = df  # Use all queries
    
    print(f"\n🚀 Testing {len(queries_df)} queries...")
    
    # Helper to parse multi-GT
    import json as json_module
    def get_gt_tables(row) -> List[str]:
        """Get ground truth table(s) from row."""
        if 'ground_truth_table_ids' in row and row['ground_truth_table_ids']:
            try:
                gt_list = json_module.loads(row['ground_truth_table_ids'])
                if isinstance(gt_list, list) and gt_list:
                    return gt_list
            except (json_module.JSONDecodeError, TypeError):
                pass
        return [row['ground_truth_table_id']]
    
    # Track metrics
    gt_found_at_rank: List[Optional[int]] = []
    failures: List[Dict[str, Any]] = []
    multi_gt_count = 0
    
    for i, row in queries_df.iterrows():
        query = row['query_text']
        gt_tables = get_gt_tables(row)
        if len(gt_tables) > 1:
            multi_gt_count += 1
        
        # Embed query
        query_emb_list = embedder.compute_query_embeddings(query)
        query_emb = np.array(query_emb_list[0], dtype=np.float32)
        
        # Search
        vec_results = vector_search(query_emb, faiss_index, metadata_list, top_k * 2) if faiss_index else []
        bm25_results = bm25_search(query, bm25_retriever, table_ids, metadata_list, top_k * 2) if bm25_retriever else []
        
        # Fuse
        if vec_results and bm25_results:
            fused = reciprocal_rank_fusion(vec_results, bm25_results)
        elif vec_results:
            fused = vec_results
        else:
            fused = bm25_results
        
        # Find best GT rank (any GT match counts)
        result_ids = [r[0] for r in fused[:top_k]]
        rank = find_best_rank_multi_gt(result_ids, gt_tables)
        
        gt_found_at_rank.append(rank)
        
        if rank is None:
            failures.append({
                'index': i + 1,
                'query': query,
                'gt_table': gt_tables[0],  # Primary GT for display
                'gt_tables': gt_tables,    # All GT tables
                'top_3': [(tid, score) for tid, score, _ in fused[:3]],
            })
        
        # Progress
        if (i + 1) % 20 == 0:
            found_so_far = sum(1 for r in gt_found_at_rank if r is not None)
            print(f"   Processed {i+1}/{len(queries_df)}, recall@{top_k}: {found_so_far}/{i+1}")
    
    # Calculate metrics
    total = len(queries_df)
    
    def recall_at_k(k: int) -> float:
        return sum(1 for r in gt_found_at_rank if r is not None and r <= k) / total
    
    def mrr() -> float:
        rrs = [1.0/r for r in gt_found_at_rank if r is not None]
        return sum(rrs) / total if rrs else 0.0
    
    metrics = {
        'method': 'semantic',
        'total_queries': total,
        'multi_gt_queries': multi_gt_count,  # Track queries with multiple GTs
        'top_k': top_k,
        'recall@1': recall_at_k(1),
        'recall@5': recall_at_k(5),
        'recall@10': recall_at_k(10),
        'recall@50': recall_at_k(50),
        'recall@100': recall_at_k(100),
        'mrr': mrr(),
        'not_found': sum(1 for r in gt_found_at_rank if r is None),
        'ranks': gt_found_at_rank,
    }
    
    # Print metrics
    print(f"\n{'='*80}")
    print(f"📊 Semantic Retrieval Metrics (n={total}, multi-GT: {multi_gt_count})")
    print(f"{'='*80}")
    print(f"   Recall@1:   {metrics['recall@1']*100:.1f}%")
    print(f"   Recall@5:   {metrics['recall@5']*100:.1f}%")
    print(f"   Recall@10:  {metrics['recall@10']*100:.1f}%")
    print(f"   Recall@50:  {metrics['recall@50']*100:.1f}%")
    print(f"   Recall@100: {metrics['recall@100']*100:.1f}%")
    print(f"   MRR:        {metrics['mrr']:.4f}")
    print(f"   Not Found:  {metrics['not_found']} ({metrics['not_found']/total*100:.1f}%)")
    
    # Show failure cases
    if show_failures > 0 and failures:
        print(f"\n❌ Failed Cases (showing {min(show_failures, len(failures))} of {len(failures)})")
        for case in failures[:show_failures]:
            print(f"\n   Query: {case['query'][:80]}...")
            print(f"   GT: {case['gt_table'][:60]}...")
            if case['top_3']:
                print(f"   Top1: {case['top_3'][0][0][:60]}... (score={case['top_3'][0][1]:.4f})")
    
    return metrics


# ==================== Hybrid Retrieval Testing ====================

def test_hybrid_retrieval(
    constraints_data: List[Dict[str, Any]],
    dataset: str,
    top_k: int = 100,
    structural_weight: float = 0.5,
    semantic_weight: float = 0.5,
    show_failures: int = 3,
) -> Dict[str, Any]:
    """
    Test hybrid retrieval combining structural + semantic.
    
    Uses RRF to fuse structural and semantic results.
    """
    import numpy as np
    from workflows.retrieval.unified_search import (
        load_unified_indexes, vector_search, bm25_search, reciprocal_rank_fusion, get_text_embedder
    )
    from workflows.retrieval.config import INDEX_KEY_TD_CD_CS
    from workflows.retrieval.matcher import ScorerV3
    from workflows.retrieval.matcher.constraints import ConstraintSet, PathConstraint, TBoxConstraint, ABoxConstraint
    
    print(f"\n{'='*80}")
    print(f"🔍 Hybrid Retrieval Test (Structural + Semantic)")
    print(f"   Weights: structural={structural_weight}, semantic={semantic_weight}")
    print(f"{'='*80}")
    
    # Load indexes using unified interface
    print(f"\n📦 Loading indexes...")
    import time
    start = time.time()
    
    # Semantic
    faiss_index, metadata_list, bm25_retriever, table_ids = load_unified_indexes(dataset, INDEX_KEY_TD_CD_CS)
    embedder = get_text_embedder()
    
    # Structural
    scorer = ScorerV3(dataset=dataset)
    
    load_time = time.time() - start
    print(f"   Loaded in {load_time:.2f}s")
    
    print(f"\n🚀 Testing {len(constraints_data)} queries...")
    
    # Track metrics
    gt_found_at_rank: List[Optional[int]] = []
    failures: List[Dict[str, Any]] = []
    
    for i, query_data in enumerate(constraints_data):
        query = query_data['query']
        gt_table = query_data['gt_table']
        tbox_list = query_data.get('tbox_constraints', [])
        abox_constraints = query_data.get('abox_constraints', {})
        entity_constraints = abox_constraints.get('entity_constraints', [])
        
        # ===== Semantic Search =====
        query_emb_list = embedder.compute_query_embeddings(query)
        query_emb = np.array(query_emb_list[0], dtype=np.float32)
        
        vec_results = vector_search(query_emb, faiss_index, metadata_list, top_k * 2) if faiss_index else []
        bm25_results = bm25_search(query, bm25_retriever, table_ids, metadata_list, top_k * 2) if bm25_retriever else []
        
        if vec_results and bm25_results:
            semantic_fused = reciprocal_rank_fusion(vec_results, bm25_results)
        elif vec_results:
            semantic_fused = vec_results
        else:
            semantic_fused = bm25_results
        
        # ===== Structural Search =====
        structural_results = []
        if tbox_list:
            # Build constraints
            entity_value_map = {}
            for entity in entity_constraints:
                col_type = entity.get('column_type', '')
                value = entity.get('value')
                if col_type and value:
                    entity_value_map[col_type] = str(value)
            
            path_constraints = []
            for tbox_class in tbox_list:
                tbox = TBoxConstraint(class_name=tbox_class)
                abox = ABoxConstraint(value=entity_value_map[tbox_class]) if tbox_class in entity_value_map else None
                path_constraints.append(PathConstraint(tbox=tbox, abox=abox))
            
            constraints = ConstraintSet(constraints=path_constraints)
            structural_results = scorer.retrieve(constraints, score_threshold=0.0)[:top_k * 2]
        
        # ===== Hybrid Fusion (RRF) =====
        # Convert to common format: (table_id, score, meta)
        semantic_list = [(tid, score, meta) for tid, score, meta in semantic_fused[:top_k * 2]]
        structural_list = [(tid, score, {}) for tid, score, _ in structural_results]
        
        # Build rank maps
        semantic_ranks = {tid: rank + 1 for rank, (tid, _, _) in enumerate(semantic_list)}
        structural_ranks = {tid: rank + 1 for rank, (tid, _, _) in enumerate(structural_list)}
        
        all_tables = set(semantic_ranks.keys()) | set(structural_ranks.keys())
        
        k_rrf = 60
        hybrid_fused = []
        for tid in all_tables:
            score = 0.0
            if tid in semantic_ranks:
                score += semantic_weight / (k_rrf + semantic_ranks[tid])
            if tid in structural_ranks:
                score += structural_weight / (k_rrf + structural_ranks[tid])
            hybrid_fused.append((tid, score))
        
        hybrid_fused.sort(key=lambda x: x[1], reverse=True)
        
        # Find GT rank
        result_ids = [r[0] for r in hybrid_fused[:top_k]]
        try:
            rank = result_ids.index(gt_table) + 1
        except ValueError:
            rank = None
        
        gt_found_at_rank.append(rank)
        
        if rank is None:
            failures.append({
                'index': i + 1,
                'query': query,
                'gt_table': gt_table,
                'semantic_rank': semantic_ranks.get(gt_table),
                'structural_rank': structural_ranks.get(gt_table),
            })
        
        # Progress
        if (i + 1) % 20 == 0:
            found_so_far = sum(1 for r in gt_found_at_rank if r is not None)
            print(f"   Processed {i+1}/{len(constraints_data)}, recall@{top_k}: {found_so_far}/{i+1}")
    
    # Calculate metrics
    total = len(constraints_data)
    
    def recall_at_k(k: int) -> float:
        return sum(1 for r in gt_found_at_rank if r is not None and r <= k) / total
    
    def mrr() -> float:
        rrs = [1.0/r for r in gt_found_at_rank if r is not None]
        return sum(rrs) / total if rrs else 0.0
    
    metrics = {
        'method': 'hybrid',
        'total_queries': total,
        'top_k': top_k,
        'structural_weight': structural_weight,
        'semantic_weight': semantic_weight,
        'recall@1': recall_at_k(1),
        'recall@5': recall_at_k(5),
        'recall@10': recall_at_k(10),
        'recall@50': recall_at_k(50),
        'recall@100': recall_at_k(100),
        'mrr': mrr(),
        'not_found': sum(1 for r in gt_found_at_rank if r is None),
        'ranks': gt_found_at_rank,
    }
    
    # Print metrics
    print(f"\n{'='*80}")
    print(f"📊 Hybrid Retrieval Metrics (n={total})")
    print(f"{'='*80}")
    print(f"   Recall@1:   {metrics['recall@1']*100:.1f}%")
    print(f"   Recall@5:   {metrics['recall@5']*100:.1f}%")
    print(f"   Recall@10:  {metrics['recall@10']*100:.1f}%")
    print(f"   Recall@50:  {metrics['recall@50']*100:.1f}%")
    print(f"   Recall@100: {metrics['recall@100']*100:.1f}%")
    print(f"   MRR:        {metrics['mrr']:.4f}")
    print(f"   Not Found:  {metrics['not_found']} ({metrics['not_found']/total*100:.1f}%)")
    
    return metrics


# ==================== Comparative Analysis ====================

def compare_all_methods(
    constraints_data: List[Dict[str, Any]],
    dataset: str,
    num_queries: int,
    top_k: int = 100,
    split: str = "test",
) -> Dict[str, Any]:
    """
    Compare all retrieval methods and analyze case-by-case differences.
    """
    print(f"\n{'='*80}")
    print(f"📊 COMPARATIVE ANALYSIS: All Methods")
    print(f"   Dataset: {dataset}, Queries: {len(constraints_data)}")
    print(f"{'='*80}")
    
    # Run all methods
    print("\n" + "="*40)
    print("1️⃣  STRUCTURAL RETRIEVAL")
    print("="*40)
    structural_metrics = test_retrieval_recall(
        constraints_data, dataset, top_k, show_failures=0
    )
    structural_ranks = structural_metrics.get('ranks', [None] * len(constraints_data))
    
    print("\n" + "="*40)
    print("2️⃣  SEMANTIC RETRIEVAL")
    print("="*40)
    semantic_metrics = test_semantic_retrieval(dataset, num_queries, top_k, show_failures=0, split=split)
    semantic_ranks = semantic_metrics.get('ranks', [None] * len(constraints_data))
    
    print("\n" + "="*40)
    print("3️⃣  HYBRID RETRIEVAL")
    print("="*40)
    hybrid_metrics = test_hybrid_retrieval(
        constraints_data, dataset, top_k, show_failures=0
    )
    hybrid_ranks = hybrid_metrics.get('ranks', [None] * len(constraints_data))
    
    # Comparative analysis
    print(f"\n{'='*80}")
    print("📊 COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    headers = ["Metric", "Structural", "Semantic", "Hybrid"]
    rows = [
        ["Recall@1", f"{structural_metrics.get('recall@1', 0)*100:.1f}%", 
         f"{semantic_metrics.get('recall@1', 0)*100:.1f}%",
         f"{hybrid_metrics.get('recall@1', 0)*100:.1f}%"],
        ["Recall@5", f"{structural_metrics.get('recall@5', 0)*100:.1f}%",
         f"{semantic_metrics.get('recall@5', 0)*100:.1f}%",
         f"{hybrid_metrics.get('recall@5', 0)*100:.1f}%"],
        ["Recall@10", f"{structural_metrics.get('recall@10', 0)*100:.1f}%",
         f"{semantic_metrics.get('recall@10', 0)*100:.1f}%",
         f"{hybrid_metrics.get('recall@10', 0)*100:.1f}%"],
        ["MRR", f"{structural_metrics.get('mrr', 0):.4f}",
         f"{semantic_metrics.get('mrr', 0):.4f}",
         f"{hybrid_metrics.get('mrr', 0):.4f}"],
    ]
    
    # Print table
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(4)]
    
    print("\n   " + "".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)))
    print("   " + "-" * sum(col_widths))
    for row in rows:
        print("   " + "".join(str(row[i]).ljust(col_widths[i]) for i in range(4)))
    
    # Case-by-case analysis
    print(f"\n📊 Case-by-Case Analysis:")
    
    # Cases where structural wins
    structural_wins = []
    semantic_wins = []
    both_fail = []
    both_succeed = []
    
    for i in range(min(len(structural_ranks), len(semantic_ranks))):
        s_rank = structural_ranks[i] if i < len(structural_ranks) else None
        m_rank = semantic_ranks[i] if i < len(semantic_ranks) else None
        
        s_hit = s_rank is not None and s_rank <= 10
        m_hit = m_rank is not None and m_rank <= 10
        
        if s_hit and not m_hit:
            structural_wins.append(i)
        elif m_hit and not s_hit:
            semantic_wins.append(i)
        elif not s_hit and not m_hit:
            both_fail.append(i)
        else:
            both_succeed.append(i)
    
    print(f"   Structural wins (hit@10 when semantic misses): {len(structural_wins)}")
    print(f"   Semantic wins (hit@10 when structural misses): {len(semantic_wins)}")
    print(f"   Both succeed: {len(both_succeed)}")
    print(f"   Both fail: {len(both_fail)}")
    
    # Show example cases
    if structural_wins:
        print(f"\n   📌 Example: Structural wins")
        idx = structural_wins[0]
        query_data = constraints_data[idx]
        print(f"      Query: {query_data['query'][:70]}...")
        print(f"      Structural rank: {structural_ranks[idx]}, Semantic rank: {semantic_ranks[idx] or '>100'}")
        print(f"      TBox: {query_data.get('tbox_constraints', [])[:3]}")
    
    if semantic_wins:
        print(f"\n   📌 Example: Semantic wins")
        idx = semantic_wins[0]
        query_data = constraints_data[idx]
        print(f"      Query: {query_data['query'][:70]}...")
        print(f"      Structural rank: {structural_ranks[idx] or '>100'}, Semantic rank: {semantic_ranks[idx]}")
    
    return {
        'structural': structural_metrics,
        'semantic': semantic_metrics,
        'hybrid': hybrid_metrics,
        'analysis': {
            'structural_wins': len(structural_wins),
            'semantic_wins': len(semantic_wins),
            'both_succeed': len(both_succeed),
            'both_fail': len(both_fail),
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze retrieval quality - structural, semantic, and hybrid"
    )
    parser.add_argument("-d", "--dataset", default="fetaqa", help="Dataset name")
    parser.add_argument("-n", "--num-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show each query analysis")
    parser.add_argument("--sample", type=int, default=None, help="Analyze first N queries in detail")
    parser.add_argument("--case", type=int, nargs='+', default=None, help="Show specific case(s) by index (1-based)")
    parser.add_argument("--mismatch", action="store_true", help="Show only mismatched cases")
    
    # Retrieval testing modes
    parser.add_argument("--test-retrieval", action="store_true", help="Test structural retrieval using ScorerV3")
    parser.add_argument("--test-semantic", action="store_true", help="Test semantic retrieval (FAISS + BM25)")
    parser.add_argument("--test-hybrid", action="store_true", help="Test hybrid retrieval (structural + semantic)")
    parser.add_argument("--compare", action="store_true", help="Compare all retrieval methods")
    
    # Retrieval options
    parser.add_argument("--top-k", type=int, default=100, help="Top-k for retrieval test (default: 100)")
    parser.add_argument("--show-failures", type=int, default=3, help="Show N failed cases in detail (default: 3)")
    parser.add_argument("--structural-weight", type=float, default=0.5, help="Structural weight for hybrid (default: 0.5)")
    parser.add_argument("--semantic-weight", type=float, default=0.5, help="Semantic weight for hybrid (default: 0.5)")
    
    # Constraint loading options (for unified analysis format)
    parser.add_argument("--llm", type=str, default="local", help="LLM used for query analysis (default: local)")
    parser.add_argument("--use-rag", action="store_true", default=True, help="Use RAG-enhanced analysis files (default: True)")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG (use non-RAG analysis files)")
    parser.add_argument("--rag-top-k", type=int, default=3, help="RAG top-k parameter (default: 3)")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train", "entries"],
                        help="Query split to use (default: test)")
    parser.add_argument("--constraint-source", type=str, default="auto", 
                        choices=["auto", "unified", "legacy"],
                        help="Constraint source: auto (try unified then legacy), unified, or legacy")
    
    args = parser.parse_args()
    
    # Handle --no-rag flag
    use_rag = args.use_rag and not args.no_rag
    
    # Load data
    constraints_data = load_constraints_file(
        args.dataset, 
        args.num_queries,
        source=args.constraint_source,
        llm=args.llm,
        use_rag=use_rag,
        rag_top_k=args.rag_top_k,
    )
    _, child_to_ancestors = load_class_hierarchy(args.dataset)
    valid_classes = set(child_to_ancestors.keys())
    valid_classes.add('Column')
    
    # Comparison mode - runs all methods
    if args.compare:
        compare_all_methods(
            constraints_data,
            dataset=args.dataset,
            num_queries=args.num_queries,
            top_k=args.top_k,
            split=args.split,
        )
        return
    
    # Semantic retrieval test mode
    if args.test_semantic:
        test_semantic_retrieval(
            dataset=args.dataset,
            num_queries=args.num_queries,
            top_k=args.top_k,
            show_failures=args.show_failures,
            split=args.split,
        )
        return
    
    # Hybrid retrieval test mode
    if args.test_hybrid:
        test_hybrid_retrieval(
            constraints_data,
            dataset=args.dataset,
            top_k=args.top_k,
            structural_weight=args.structural_weight,
            semantic_weight=args.semantic_weight,
            show_failures=args.show_failures,
        )
        return
    
    # Structural retrieval test mode (original)
    if args.test_retrieval:
        test_retrieval_recall(
            constraints_data,
            dataset=args.dataset,
            top_k=args.top_k,
            show_failures=args.show_failures,
        )
        return
    
    if args.case:
        # Analyze specific cases
        print(f"\n{'='*80}")
        print(f"Detailed analysis of specified cases: {args.case}")
        print(f"{'='*80}")
        
        for case_idx in args.case:
            if 1 <= case_idx <= len(constraints_data):
                query_data = constraints_data[case_idx - 1]
                analysis = analyze_single_query(
                    query_data, args.dataset, child_to_ancestors, valid_classes, verbose=False
                )
                print_detailed_case(query_data, analysis, case_idx, args.dataset)
            else:
                print(f"Invalid case index: {case_idx} (valid: 1-{len(constraints_data)})")
    elif args.sample:
        # Analyze a sample in detail
        print(f"\n{'='*80}")
        print(f"Detailed analysis of first {args.sample} queries")
        print(f"{'='*80}")
        
        shown = 0
        for i, query_data in enumerate(constraints_data[:args.sample]):
            analysis = analyze_single_query(
                query_data, args.dataset, child_to_ancestors, valid_classes, verbose=False
            )
            
            # Filter mismatched if requested
            if args.mismatch:
                tbox = analysis.get('tbox_analysis')
                if tbox and tbox.get('hierarchy_precision', 1.0) >= 1.0:
                    continue
            
            print_detailed_case(query_data, analysis, i + 1, args.dataset)
            shown += 1
        
        if args.mismatch:
            print(f"\n(Showed {shown} mismatched cases out of {args.sample})")
    else:
        # Full analysis
        analyze_all_queries(args.dataset, args.num_queries, verbose=args.verbose)


if __name__ == "__main__":
    main()
