"""
Test script for ScorerV3.

This script tests the v3 scoring formula with detailed logging.
"""

import sys
from pathlib import Path

# Add source to path - test file is at source/subgraph/retrieval/matcher/test_scorer_v3.py
# So source is 4 levels up
source_dir = Path(__file__).parent.parent.parent.parent
if str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir))

from loguru import logger

# Configure logging to show debug
logger.remove()
logger.add(
    sys.stdout,
    level="DEBUG",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
)

from workflows.retrieval.matcher import (
    get_scorer_v3,
    PathConstraint,
    ConstraintSet,
    TBoxConstraint,
    ABoxConstraint,
)


def test_single_abox_constraint():
    """Test with a single ABox constraint."""
    logger.info("=" * 70)
    logger.info("Test: Single ABox Constraint (YearColumn, '1982')")
    logger.info("=" * 70)
    
    scorer = get_scorer_v3('fetaqa', beta=1.0)
    
    # Create constraint
    constraint = PathConstraint(
        tbox=TBoxConstraint(class_name='YearColumn'),
        abox=ABoxConstraint(value='1982'),
    )
    constraints = ConstraintSet(constraints=[constraint])
    
    # Retrieve
    results = scorer.retrieve(constraints, score_threshold=0.0)
    
    # Print top results
    logger.info(f"\nTop 10 results:")
    for i, (table_id, score, details) in enumerate(results[:10]):
        logger.info(f"  {i+1}. {table_id}: {score:.4f}")
        for d in details:
            if d.get('has_abox'):
                logger.info(f"      ABox match at: {d.get('best_ancestor')}")
    
    return results


def test_single_tbox_constraint():
    """Test with a single TBox-only constraint."""
    logger.info("=" * 70)
    logger.info("Test: Single TBox Constraint (PersonColumn)")
    logger.info("=" * 70)
    
    scorer = get_scorer_v3('fetaqa', beta=1.0)
    
    # Create constraint (no ABox)
    constraint = PathConstraint(
        tbox=TBoxConstraint(class_name='PersonColumn'),
    )
    constraints = ConstraintSet(constraints=[constraint])
    
    # Retrieve
    results = scorer.retrieve(constraints, score_threshold=0.0)
    
    # Print top results
    logger.info(f"\nTop 10 results:")
    for i, (table_id, score, details) in enumerate(results[:10]):
        logger.info(f"  {i+1}. {table_id}: {score:.4f}")
        for d in details:
            logger.info(f"      TBox match at: {d.get('deepest_matched')}")
    
    return results


def test_multiple_constraints():
    """Test with multiple constraints."""
    logger.info("=" * 70)
    logger.info("Test: Multiple Constraints")
    logger.info("=" * 70)
    
    scorer = get_scorer_v3('fetaqa', beta=1.0)
    
    # Create constraints
    constraints = ConstraintSet(constraints=[
        PathConstraint(
            tbox=TBoxConstraint(class_name='YearColumn'),
            abox=ABoxConstraint(value='1982'),
        ),
        PathConstraint(
            tbox=TBoxConstraint(class_name='PersonColumn'),
        ),
    ])
    
    # Retrieve
    results = scorer.retrieve(constraints, score_threshold=0.0)
    
    # Print top results
    logger.info(f"\nTop 10 results:")
    for i, (table_id, score, details) in enumerate(results[:10]):
        logger.info(f"  {i+1}. {table_id}: {score:.4f}")
    
    return results


def test_compare_v2_v3():
    """Compare v2.1 and v3 scores for specific tables."""
    logger.info("=" * 70)
    logger.info("Test: Compare V2.1 vs V3 Scoring")
    logger.info("=" * 70)
    
    from workflows.retrieval.matcher import get_path_matcher
    
    # Get both matchers
    matcher_v2 = get_path_matcher('fetaqa')
    scorer_v3 = get_scorer_v3('fetaqa', beta=1.0)
    
    # Create constraint
    constraint = PathConstraint(
        tbox=TBoxConstraint(class_name='YearColumn'),
        abox=ABoxConstraint(value='1982'),
    )
    constraints = ConstraintSet(constraints=[constraint])
    
    # Get results from both
    results_v2 = matcher_v2.retrieve(constraints, score_threshold=0.0)
    results_v3 = scorer_v3.retrieve(constraints, score_threshold=0.0)
    
    # Build score maps
    scores_v2 = {r[0]: r[1] for r in results_v2}
    scores_v3 = {r[0]: r[1] for r in results_v3}
    
    # Compare top tables from each
    logger.info("\nTop 10 V2.1 vs V3:")
    for i, (table_id, score_v2, _) in enumerate(results_v2[:10]):
        score_v3 = scores_v3.get(table_id, 0)
        logger.info(f"  {i+1}. {table_id}: V2={score_v2:.4f}, V3={score_v3:.4f}")
    
    logger.info("\nTables with biggest score difference:")
    diffs = []
    for table_id in set(scores_v2.keys()) | set(scores_v3.keys()):
        v2 = scores_v2.get(table_id, 0)
        v3 = scores_v3.get(table_id, 0)
        diffs.append((table_id, v2, v3, abs(v2 - v3)))
    
    diffs.sort(key=lambda x: -x[3])
    for table_id, v2, v3, diff in diffs[:10]:
        logger.info(f"  {table_id}: V2={v2:.4f}, V3={v3:.4f}, diff={diff:.4f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ScorerV3')
    parser.add_argument('--test', choices=['abox', 'tbox', 'multi', 'compare', 'all'], 
                       default='abox', help='Which test to run')
    args = parser.parse_args()
    
    if args.test == 'abox' or args.test == 'all':
        test_single_abox_constraint()
    
    if args.test == 'tbox' or args.test == 'all':
        test_single_tbox_constraint()
    
    if args.test == 'multi' or args.test == 'all':
        test_multiple_constraints()
    
    if args.test == 'compare' or args.test == 'all':
        test_compare_v2_v3()
