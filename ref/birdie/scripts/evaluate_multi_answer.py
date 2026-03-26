#!/usr/bin/env python3
"""
Multi-Answer Table-Level Hit@K Evaluation for Birdie.

This script evaluates Birdie predictions with proper support for:
1. Table-level Hit@K (not document-level)
2. Multi-answer queries (one query can have multiple correct tables)

The Hit@K metric is:
- Hit@K = 1 if ANY of the top-K predictions matches ANY of the correct answer tables
- Hit@K = 0 otherwise

This is different from document-level metrics where a document might contain multiple tables.

Usage:
    python evaluate_multi_answer.py \
        --predictions /path/to/predictions.json \
        --ground_truth /path/to/test_queries.json \
        --semantic_id_mapping /path/to/id_mapping.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


def load_predictions(pred_file: str) -> Dict[str, List[str]]:
    """
    Load predictions from file.
    
    Expected format (one of):
    - List of {"query": "...", "predictions": ["id1", "id2", ...]}
    - Dict mapping query_id -> list of predicted IDs
    """
    with open(pred_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        # Convert list format to dict
        predictions = {}
        for item in data:
            query_id = item.get('query_id', item.get('query', ''))
            preds = item.get('predictions', item.get('predicted_ids', []))
            predictions[query_id] = preds
        return predictions
    
    return data


def load_ground_truth(gt_file: str) -> Dict[str, List[str]]:
    """
    Load ground truth from unified format test queries.
    
    Returns: Dict mapping query_id -> list of correct table_ids
    """
    with open(gt_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ground_truth = {}
    for item in data:
        query_id = item.get('query_id', item.get('question', ''))
        
        # Support both formats:
        # - "all_answer_tables": [...] (our extended format)
        # - "answer_tables": [...] (original unified format)
        # - "table_id": "..." (original Birdie format)
        answer_tables = item.get('all_answer_tables', 
                                 item.get('answer_tables', 
                                          [item.get('table_id')]))
        
        if answer_tables:
            ground_truth[query_id] = answer_tables
    
    return ground_truth


def load_semantic_id_mapping(mapping_file: str) -> Dict[str, str]:
    """
    Load mapping from table_id to semantic_id.
    
    This is needed because Birdie predictions use semantic IDs,
    but ground truth uses original table IDs.
    
    Returns: Dict mapping table_id -> semantic_id
    """
    if not mapping_file or not Path(mapping_file).exists():
        return {}
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different formats
    if 'table_to_semantic' in data:
        return data['table_to_semantic']
    
    return data


def compute_hit_at_k(predictions: List[str], 
                     ground_truth: Set[str], 
                     k: int) -> bool:
    """
    Compute Hit@K for a single query.
    
    Hit@K = 1 if any of the top-K predictions matches any ground truth table.
    
    Args:
        predictions: List of predicted table IDs (or semantic IDs)
        ground_truth: Set of correct table IDs (or semantic IDs)
        k: Number of top predictions to consider
    
    Returns:
        True if there's at least one match in top-K, False otherwise
    """
    top_k_preds = set(predictions[:k])
    return len(top_k_preds & ground_truth) > 0


def evaluate_multi_answer(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    semantic_id_mapping: Dict[str, str] = None,
    k_values: List[int] = [1, 5, 10, 20]
) -> Tuple[Dict[str, float], Dict]:
    """
    Evaluate predictions with multi-answer support.
    
    Args:
        predictions: Dict mapping query_id -> list of predicted IDs
        ground_truth: Dict mapping query_id -> list of correct table_ids
        semantic_id_mapping: Optional mapping from table_id to semantic_id
        k_values: List of K values for Hit@K
    
    Returns:
        Tuple of (metrics_dict, detailed_results)
    """
    # Convert ground truth to semantic IDs if mapping provided
    if semantic_id_mapping:
        gt_converted = {}
        for query_id, table_ids in ground_truth.items():
            semantic_ids = []
            for tid in table_ids:
                if tid in semantic_id_mapping:
                    semantic_ids.append(semantic_id_mapping[tid])
                else:
                    # Keep original ID if no mapping
                    semantic_ids.append(tid)
            gt_converted[query_id] = semantic_ids
        ground_truth = gt_converted
    
    # Initialize counters
    hits = {k: 0 for k in k_values}
    total = 0
    
    # Per-query results
    detailed_results = []
    
    # Track statistics
    multi_answer_count = 0
    missing_predictions = 0
    
    for query_id, gt_tables in ground_truth.items():
        if query_id not in predictions:
            missing_predictions += 1
            continue
        
        pred_list = predictions[query_id]
        gt_set = set(gt_tables)
        
        if len(gt_tables) > 1:
            multi_answer_count += 1
        
        # Compute Hit@K for each K
        query_result = {
            'query_id': query_id,
            'ground_truth': gt_tables,
            'predictions_top10': pred_list[:10],
            'num_answers': len(gt_tables)
        }
        
        for k in k_values:
            hit = compute_hit_at_k(pred_list, gt_set, k)
            if hit:
                hits[k] += 1
            query_result[f'hit@{k}'] = hit
        
        detailed_results.append(query_result)
        total += 1
    
    # Compute metrics
    metrics = {}
    for k in k_values:
        metrics[f'Hit@{k}'] = hits[k] / total if total > 0 else 0.0
    
    metrics['total_queries'] = total
    metrics['multi_answer_queries'] = multi_answer_count
    metrics['multi_answer_ratio'] = multi_answer_count / total if total > 0 else 0.0
    metrics['missing_predictions'] = missing_predictions
    
    return metrics, detailed_results


def print_results(metrics: Dict[str, float], detailed: bool = False, detailed_results: List = None):
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("Multi-Answer Table-Level Hit@K Evaluation Results")
    print("=" * 60)
    
    print(f"\nTotal queries evaluated: {metrics['total_queries']}")
    print(f"Multi-answer queries: {metrics['multi_answer_queries']} ({metrics['multi_answer_ratio']*100:.1f}%)")
    
    if metrics.get('missing_predictions', 0) > 0:
        print(f"Missing predictions: {metrics['missing_predictions']}")
    
    print("\nHit@K Metrics:")
    for k in [1, 5, 10, 20]:
        key = f'Hit@{k}'
        if key in metrics:
            print(f"  {key}: {metrics[key]*100:.2f}%")
    
    print("=" * 60)
    
    if detailed and detailed_results:
        print("\nDetailed Results (first 10):")
        for result in detailed_results[:10]:
            print(f"\nQuery: {result['query_id'][:50]}...")
            print(f"  Ground Truth ({result['num_answers']}): {result['ground_truth']}")
            print(f"  Predictions: {result['predictions_top10'][:5]}")
            print(f"  Hit@1: {result.get('hit@1', 'N/A')}, Hit@5: {result.get('hit@5', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Answer Table-Level Hit@K Evaluation"
    )
    parser.add_argument("--predictions", required=True, type=str,
                        help="Path to predictions file (JSON)")
    parser.add_argument("--ground_truth", required=True, type=str,
                        help="Path to ground truth file (test_queries.json)")
    parser.add_argument("--semantic_id_mapping", type=str, default=None,
                        help="Path to table_id -> semantic_id mapping (optional)")
    parser.add_argument("--k_values", nargs='+', type=int, default=[1, 5, 10, 20],
                        help="K values for Hit@K (default: 1 5 10 20)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for detailed results (JSON)")
    parser.add_argument("--detailed", action="store_true",
                        help="Print detailed per-query results")
    
    args = parser.parse_args()
    
    print(f"Loading predictions from {args.predictions}")
    predictions = load_predictions(args.predictions)
    print(f"Loaded {len(predictions)} predictions")
    
    print(f"Loading ground truth from {args.ground_truth}")
    ground_truth = load_ground_truth(args.ground_truth)
    print(f"Loaded {len(ground_truth)} ground truth entries")
    
    semantic_mapping = None
    if args.semantic_id_mapping:
        print(f"Loading semantic ID mapping from {args.semantic_id_mapping}")
        semantic_mapping = load_semantic_id_mapping(args.semantic_id_mapping)
        print(f"Loaded {len(semantic_mapping)} mappings")
    
    metrics, detailed_results = evaluate_multi_answer(
        predictions=predictions,
        ground_truth=ground_truth,
        semantic_id_mapping=semantic_mapping,
        k_values=args.k_values
    )
    
    print_results(metrics, args.detailed, detailed_results)
    
    if args.output:
        output_data = {
            'metrics': metrics,
            'detailed_results': detailed_results
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
