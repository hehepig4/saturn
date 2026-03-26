"""
Retrieval Demo - Query Analysis and Evaluation

Usage:
  # Run single query
  python demos/retrieval.py -q "Which Illinois election had the most votes?"
  
  # Batch evaluation (semantic path only)
  python demos/retrieval.py --eval -d fetaqa -n 100
  
  # Batch evaluation with specific index configuration
  python demos/retrieval.py --eval -d fetaqa -n 100 --index-key td_cd
  
  # Batch evaluation (full dual-path: semantic + structural)
  python demos/retrieval.py --eval-full -d fetaqa -n 100
  
  # Analyze queries: unified LLM analysis for HyDE + TBox/ABox constraints
  python demos/retrieval.py --analyze-queries -d fetaqa -n 100 --llm gemini
  
  # RAG-enhanced query analysis (retrieve similar tables as format reference)
  python demos/retrieval.py --analyze-queries -d fetaqa --use-rag --rag-top-k 5

Index Keys (used with --index-key):
  - td: table_description only (smallest, fast)
  - td_cd: table_description + column_descriptions (balanced)
  - td_cd_cs: all three fields including column_stats (default, most comprehensive)

Note: 
  - Index generation is handled by run_upo_pipeline.py (Stage 5).
    Example: python demos/run_upo_pipeline.py -d fetaqa -s retrieval_index --index-key td_cd_cs
  - For HyDE retrieval evaluation, use scripts/eval/analyze_hyde_retrieval.py
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional
# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
import _path_setup  # noqa: F401

from loguru import logger


def cmd_search(args):
    """Run a single search query."""
    from workflows.retrieval.unified_search import unified_search
    from workflows.retrieval.config import INDEX_KEY_TD_CD_CS
    
    index_key = args.index_key or INDEX_KEY_TD_CD_CS
    rag_type = getattr(args, 'rag_type', 'hybrid')
    
    logger.info("=" * 60)
    logger.info("Unified Retrieval")
    logger.info("=" * 60)
    logger.info(f"  Query: {args.query}")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Top-K: {args.top_k}")
    logger.info(f"  Index Key: {index_key}")
    logger.info(f"  RAG Type: {rag_type}")
    
    # Use unified search
    results = unified_search(
        query=args.query,
        dataset_name=args.dataset,
        top_k=args.top_k,
        rag_type=rag_type,
        index_key=index_key,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print(f"Results (RAG Type: {rag_type})")
    print("=" * 60)
    
    if results:
        print(f"\nRetrieved {len(results)} tables:")
        for i, (table_id, score, meta) in enumerate(results[:10]):
            title = meta.get("document_title", "")[:30] if meta else ""
            print(f"  {i+1}. {table_id[:60]} (score={score:.4f}) {title}")
    else:
        print("No results found.")
    
    return results


def cmd_eval_full(args):
    """Batch evaluation with full dual-path retrieval (semantic + structural)."""
    from store.store_singleton import get_store
    from workflows.retrieval.graph import run_retrieval
    import numpy as np
    
    logger.info("=" * 60)
    logger.info(f"Full Pipeline Evaluation: {args.dataset}")
    logger.info("=" * 60)
    
    # Load queries - support different splits: test, train
    store = get_store()
    if args.split == "train":
        query_table = f"{args.dataset}_train_queries"
    else:
        query_table = f"{args.dataset}_test_queries"
    tbl = store.db.open_table(query_table)
    df = tbl.to_pandas()
    
    queries = df.head(args.num_queries) if args.num_queries > 0 else df
    logger.info(f"  Evaluating {len(queries)} queries from {query_table}")
    
    # Evaluate
    ranks = []
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    miss_samples = []
    
    for i, row in queries.iterrows():
        query = row['query_text']
        gt_table = row['ground_truth_table_id']
        
        logger.debug(f"\n{'='*60}")
        logger.debug(f"[Query {i+1}] {query[:60]}...")
        logger.debug(f"  GT: {gt_table[:50]}...")
        
        # Run full retrieval pipeline
        result_state = run_retrieval(
            query=query,
            dataset_name=args.dataset,
            top_k=100,
            semantic_top_k=100,
            enable_semantic=True,
            enable_structural=True,  # This enables structural path
            enable_bm25=True,
        )
        
        # Get results
        final_results = result_state.final_results or []
        result_ids = [r.table_id for r in final_results]
        
        # Find GT rank
        try:
            rank = result_ids.index(gt_table) + 1
        except ValueError:
            rank = len(result_ids) + 1 if result_ids else 101
        
        # Log details
        logger.debug(f"  Final rank: {rank}")
        if rank <= 5:
            logger.debug(f"  ✓ HIT!")
        else:
            logger.debug(f"  ❌ MISS")
            top3 = [(r.table_id, r.final_score, r.source) for r in final_results[:3]]
            logger.debug(f"  Top 3: {top3}")
            miss_samples.append({
                'query': query,
                'gt': gt_table,
                'rank': rank,
                'top3': top3,
            })
        
        ranks.append(rank)
        if rank == 1:
            hits_at_1 += 1
        if rank <= 5:
            hits_at_5 += 1
        if rank <= 10:
            hits_at_10 += 1
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i+1}/{len(queries)} queries")
    
    # Compute metrics
    mrr = np.mean([1.0 / r for r in ranks])
    
    print("\n" + "=" * 60)
    print("Full Pipeline Evaluation Results")
    print("=" * 60)
    print(f"  Queries: {len(queries)}")
    print(f"  MRR: {mrr:.4f}")
    print(f"  Hits@1: {hits_at_1}/{len(queries)} ({100*hits_at_1/len(queries):.1f}%)")
    print(f"  Hits@5: {hits_at_5}/{len(queries)} ({100*hits_at_5/len(queries):.1f}%)")
    print(f"  Hits@10: {hits_at_10}/{len(queries)} ({100*hits_at_10/len(queries):.1f}%)")
    print(f"  Median Rank: {np.median(ranks):.0f}")
    
    # Print miss analysis
    if miss_samples:
        print(f"\n  ❌ Missed cases ({len(miss_samples)}):")
        for sample in miss_samples[:5]:
            print(f"     Query: {sample['query'][:60]}...")
            print(f"     GT: {sample['gt'][:50]}... (rank={sample['rank']})")
            if sample['top3']:
                tid, score, src = sample['top3'][0]
                print(f"     Top1: {tid[:50]}... ({src}, score={score:.4f})")
            print()


def cmd_eval(args):
    """Batch evaluation on dataset queries."""
    from store.store_singleton import get_store
    from workflows.retrieval.unified_search import unified_search
    from workflows.retrieval.config import RAG_TYPE_BM25, RAG_TYPE_VECTOR, RAG_TYPE_HYBRID, INDEX_KEY_TD_CD_CS
    from store.embedding.embedding_registry import get_registry
    import numpy as np
    
    # Determine index_key
    index_key = args.index_key or INDEX_KEY_TD_CD_CS
    rag_type = getattr(args, 'rag_type', 'hybrid')
    
    logger.info("=" * 60)
    logger.info(f"Batch Evaluation: {args.dataset}")
    logger.info("=" * 60)
    logger.info(f"  Index Key: {index_key}")
    logger.info(f"  RAG Type: {rag_type}")
    
    # Load queries - support different splits: test, train
    store = get_store()
    if args.split == "train":
        query_table = f"{args.dataset}_train_queries"
    else:
        query_table = f"{args.dataset}_test_queries"
    tbl = store.db.open_table(query_table)
    df = tbl.to_pandas()
    
    queries = df.head(args.num_queries) if args.num_queries > 0 else df
    logger.info(f"  Evaluating {len(queries)} queries from {query_table}")
    
    # Evaluate
    ranks = []
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    miss_samples = []  # Track failed cases for analysis
    
    for i, row in queries.iterrows():
        query = row['query_text']
        gt_table = row['ground_truth_table_id']
        
        # Use unified search
        results = unified_search(
            query=query,
            dataset_name=args.dataset,
            top_k=100,
            rag_type=rag_type,
            index_key=index_key,
        )
        
        # Extract result IDs
        result_ids = [r[0] for r in results]
        
        # Find GT rank
        try:
            rank = result_ids.index(gt_table) + 1
        except ValueError:
            rank = len(result_ids) + 1
        
        logger.debug(f"\n  [{i+1}] Query: {query[:60]}...")
        logger.debug(f"      GT: {gt_table[:50]}...")
        logger.debug(f"      Rank: {rank}")
        
        if rank > 10:
            # Log top-3 results for failed cases
            logger.debug(f"      ❌ MISS - Top 3 results:")
            for j, (tid, score, _) in enumerate(results[:3]):
                logger.debug(f"         {j+1}. {tid[:50]}... (score={score:.4f})")
            miss_samples.append({
                'query': query,
                'gt': gt_table,
                'rank': rank,
                'top3': [(tid, score) for tid, score, _ in results[:3]],
            })
        
        ranks.append(rank)
        if rank == 1:
            hits_at_1 += 1
        if rank <= 5:
            hits_at_5 += 1
        if rank <= 10:
            hits_at_10 += 1
        
        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i+1}/{len(queries)} queries")
    
    # Compute metrics
    mrr = np.mean([1.0 / r for r in ranks])
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"  Queries: {len(queries)}")
    print(f"  RAG Type: {rag_type}")
    print(f"  MRR: {mrr:.4f}")
    print(f"  Hits@1: {hits_at_1}/{len(queries)} ({100*hits_at_1/len(queries):.1f}%)")
    print(f"  Hits@5: {hits_at_5}/{len(queries)} ({100*hits_at_5/len(queries):.1f}%)")
    print(f"  Hits@10: {hits_at_10}/{len(queries)} ({100*hits_at_10/len(queries):.1f}%)")
    print(f"  Median Rank: {np.median(ranks):.0f}")
    
    # Print miss analysis
    if miss_samples:
        print(f"\n  ❌ Missed cases ({len(miss_samples)}):")
        for sample in miss_samples[:5]:  # Show top 5
            print(f"     Query: {sample['query'][:60]}...")
            print(f"     GT: {sample['gt'][:50]}... (rank={sample['rank']})")
            print(f"     Top1: {sample['top3'][0][0][:50]}... (score={sample['top3'][0][1]:.4f})")
            print()


def cmd_analyze_queries(args):
    """Unified query analysis: extract HyDE hypothetical descriptions and TBox/ABox constraints.
    
    Uses a single LLM call to generate:
    1. Hypothetical table/column descriptions (for semantic HyDE matching)
    2. TBox constraints (column types for structural matching)
    3. ABox constraints (explicit values in query for structural matching)
    
    Supports parallel processing with Gemini or other LLMs.
    Results are cached to JSON for subsequent experiments.
    
    With --use-rag: Uses RAG-enhanced analysis that retrieves similar tables first
    as style reference for the LLM.
    """
    import json
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from store.store_singleton import get_store
    from core.paths import get_db_path
    from workflows.retrieval.nodes.unified_query_analysis import (
        _unified_query_analysis_impl,
        serialize_unified_result,
    )
    import asyncio
    
    # Import RAG version if needed
    if args.use_rag:
        from workflows.retrieval.nodes.rag_unified_query_analysis import (
            _rag_unified_query_analysis_impl,
        )
    
    mode_str = "RAG-enhanced" if args.use_rag else "Standard"
    logger.info("=" * 60)
    logger.info(f"Unified Query Analysis ({mode_str})")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  LLM: {args.llm}")
    logger.info(f"  Parallel workers: {args.parallel}")
    if args.use_rag:
        logger.info(f"  RAG top-k: {args.rag_top_k}")
    logger.info("=" * 60)

    # Load queries - support different splits: test, train
    store = get_store()
    if args.split == "train":
        query_table = f"{args.dataset}_train_queries"
    else:
        query_table = f"{args.dataset}_test_queries"
    tbl = store.db.open_table(query_table)
    df = tbl.to_pandas()
    
    # -1 or 0 means all queries
    if args.num_queries <= 0:
        queries_df = df
    else:
        queries_df = df.head(args.num_queries)
    
    # Support multi-GT: parse ground_truth_table_ids (JSON array) if available
    import json as json_module
    def parse_gt_tables(row) -> List[str]:
        """Parse ground truth table(s) from row."""
        # Try multi-GT field first (JSON array string)
        if 'ground_truth_table_ids' in row and row['ground_truth_table_ids']:
            try:
                gt_list = json_module.loads(row['ground_truth_table_ids'])
                if isinstance(gt_list, list) and gt_list:
                    return gt_list
            except (json_module.JSONDecodeError, TypeError):
                pass
        # Fall back to single GT field
        return [row['ground_truth_table_id']]
    
    queries = [
        (row['query_text'], parse_gt_tables(row)) 
        for _, row in queries_df.iterrows()
    ]
    logger.info(f"  Processing {len(queries)} queries")
    
    # Pre-warm embedding model to avoid concurrent initialization issues
    # Only load if using vector or hybrid mode
    if args.use_rag and args.rag_type in ("vector", "hybrid"):
        logger.info("  Pre-warming embedding model...")
        from workflows.retrieval.unified_search import get_text_embedder, load_unified_indexes
        from workflows.retrieval.config import INDEX_KEY_TD_CD_CS
        import numpy as np
        # Force model initialization in main thread
        embedder = get_text_embedder()
        _ = embedder.compute_query_embeddings("warmup query")
        # Also pre-load indexes
        _, _, _, _ = load_unified_indexes(args.dataset, INDEX_KEY_TD_CD_CS)
        logger.info("  Embedding model ready")
    elif args.use_rag and args.rag_type == "bm25":
        # Pre-load BM25 indexes only
        logger.info("  Pre-loading BM25 indexes...")
        from workflows.retrieval.unified_search import load_unified_indexes
        from workflows.retrieval.config import INDEX_KEY_TD_CD_CS
        _, _, _, _ = load_unified_indexes(args.dataset, INDEX_KEY_TD_CD_CS)
        logger.info("  BM25 indexes ready")
    
    async def analyze_single_query_async(query_info, idx, progress_tracker):
        """Async version: Analyze a single query using unified LLM call."""
        query, gt_tables = query_info  # gt_tables is now a list
        gt_table = gt_tables[0]  # Use first GT for analysis (backward compatible)
        
        result_record = {
            'query': query,
            'gt_table': gt_table,        # Single GT for backward compatibility
            'gt_tables': gt_tables,       # All GT tables (new field)
            'analysis': None,
            'error': None,
        }
        
        try:
            if args.use_rag:
                # RAG-enhanced version (returns UnifiedQueryAnalysis Pydantic object)
                from workflows.retrieval.nodes.rag_unified_query_analysis import process_unified_analysis as rag_process
                # Get index_base_path from args (may be None)
                index_base_path = Path(args.index_base_path) if args.index_base_path else None
                raw_analysis = await _rag_unified_query_analysis_impl(
                    query=query,
                    dataset_name=args.dataset,
                    llm_purpose=args.llm,
                    rag_top_k=args.rag_top_k,
                    rag_type=args.rag_type,
                    use_primitive_classes=not args.no_primitive_classes,
                    index_base_path=index_base_path,
                )
                # Process Pydantic object to serializable dict format
                analysis = rag_process(raw_analysis, dataset_name=args.dataset)
                result_record['analysis'] = analysis
            else:
                # Standard version (returns processed dict with ConstraintSet)
                analysis = await _unified_query_analysis_impl(
                    query=query,
                    dataset_name=args.dataset,
                    llm_purpose=args.llm,
                )
                # Serialize to storable format (converts ConstraintSet)
                result_record['analysis'] = serialize_unified_result(analysis)
            
        except Exception as e:
            result_record['error'] = str(e)
            logger.warning(f"  [{idx+1}] Failed: {e}")
        
        # Update progress
        progress_tracker['completed'] += 1
        completed = progress_tracker['completed']
        total = progress_tracker['total']
        
        # Progress indicator
        if result_record.get('error'):
            status = "✗"
            detail = f"Error: {result_record['error'][:50]}"
        else:
            status = "✓"
            analysis = result_record.get('analysis', {})
            tbox_cnt = len(analysis.get('tbox_constraints', []))
            abox_cnt = len(analysis.get('abox_constraints', []))
            detail = f"TBox={tbox_cnt}, ABox={abox_cnt}"
        
        print(f"[{completed}/{total}] {status} Query {idx+1}: {detail}", flush=True)
        
        return result_record, idx
    
    async def run_parallel_analysis():
        """Run all queries in parallel with semaphore-controlled concurrency."""
        semaphore = asyncio.Semaphore(args.parallel)
        progress_tracker = {'completed': 0, 'total': len(queries)}
        
        async def analyze_with_semaphore(query_info, idx):
            async with semaphore:
                return await analyze_single_query_async(query_info, idx, progress_tracker)
        
        # Create all tasks
        tasks = [analyze_with_semaphore(q, i) for i, q in enumerate(queries)]
        
        # Run all tasks concurrently
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    # Run parallel analysis with single event loop (no thread leak)
    raw_results = asyncio.run(run_parallel_analysis())
    
    # Process results (handle any exceptions from gather)
    results = [None] * len(queries)
    for item in raw_results:
        if isinstance(item, Exception):
            # This shouldn't happen as we handle exceptions inside, but just in case
            continue
        result, idx = item
        results[idx] = result
    
    # Save results
    output_dir = get_db_path() / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    query_count_str = "all" if args.num_queries <= 0 else str(args.num_queries)
    split_str = f"_{args.split}" if args.split != "entries" else ""
    
    # Use custom output directory if provided
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build filename based on options
    # Format: {dataset}_{split}_unified_analysis_{count}_{llm}[_rag{k}_{type}][_{index_key}][_no_pc].json
    if args.use_rag:
        rag_suffix = f"_rag{args.rag_top_k}_{args.rag_type}"
        # Include index_key in filename if not default
        index_key_suffix = f"_{args.index_key}" if args.index_key and args.index_key != "td_cd_cs" else ""
        pc_suffix = "_no_pc" if args.no_primitive_classes else ""
        output_file = output_dir / f"{args.dataset}{split_str}_unified_analysis_{query_count_str}_{args.llm}{rag_suffix}{index_key_suffix}{pc_suffix}.json"
    else:
        output_file = output_dir / f"{args.dataset}{split_str}_unified_analysis_{query_count_str}_{args.llm}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print(f"Summary: Processed {len(results)} queries")
    
    success_count = sum(1 for r in results if r and r.get('analysis') is not None)
    error_count = sum(1 for r in results if r and r.get('error') is not None)
    
    # Aggregate stats
    total_tbox = sum(len(r.get('analysis', {}).get('tbox_constraints', [])) for r in results if r.get('analysis'))
    total_abox = sum(len(r.get('analysis', {}).get('abox_constraints', [])) for r in results if r.get('analysis'))
    
    print(f"  ✓ Success: {success_count}/{len(results)}")
    print(f"  ✗ Errors: {error_count}/{len(results)}")
    if success_count > 0:
        print(f"  📊 Total TBox constraints: {total_tbox} (avg {total_tbox/success_count:.1f}/query)")
        print(f"  📊 Total ABox constraints: {total_abox} (avg {total_abox/success_count:.1f}/query)")
    
    # Sample output
    if success_count > 0:
        print(f"\n{'='*60}")
        print("Sample Output (first successful analysis):")
        for r in results:
            if r.get('analysis'):
                analysis = r['analysis']
                print(f"  Query: {r['query'][:70]}...")
                print(f"  Table desc: {analysis.get('hypothetical_table_description', '')[:100]}...")
                print(f"  TBox: {analysis.get('tbox_constraints', [])}")
                if analysis.get('abox_constraints'):
                    print(f"  ABox: {analysis.get('abox_constraints', [])}")
                break
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Retrieval Demo - Query Analysis and Evaluation")
    parser.add_argument("-d", "--dataset", default="fetaqa", help="Dataset name")
    
    # Modes
    parser.add_argument("--eval", action="store_true", help="Batch evaluation (semantic only)")
    parser.add_argument("--eval-full", action="store_true", help="Batch evaluation (semantic + structural)")
    parser.add_argument("--analyze-queries", action="store_true", 
                        help="Unified query analysis: HyDE descriptions + TBox/ABox constraints (single LLM call)")
    parser.add_argument("-q", "--query", type=str, help="Search query")
    
    # Index configuration
    parser.add_argument("--index-key", type=str, choices=["td", "td_cd", "td_cd_cs"], default=None,
                        help="Index configuration: td (table_desc), td_cd (+column_desc), td_cd_cs (full, default)")
    
    # Options
    parser.add_argument("-k", "--top-k", type=int, default=10, help="Top K results")
    parser.add_argument("-n", "--num-queries", type=int, default=100, help="Number of queries for eval (-1 for all)")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"],
                        help="Query split to use: test or train (default: test)")
    parser.add_argument("--semantic-only", action="store_true", help="Disable structural search")
    
    # LLM and parallel options
    parser.add_argument("--llm", "--llm-purpose", type=str, default="gemini", help="LLM to use (default, gemini, local)")
    parser.add_argument("--parallel", type=int, default=10, help="Number of parallel workers")
    
    # RAG-enhanced query analysis options
    parser.add_argument("--use-rag", action="store_true", 
                        help="Use RAG-enhanced query analysis (retrieve similar tables as style reference)")
    parser.add_argument("--rag-top-k", type=int, default=3, 
                        help="Number of similar tables to retrieve for RAG context (default: 3)")
    parser.add_argument("--rag-type", type=str, default="hybrid", choices=["bm25", "vector", "hybrid"],
                        help="RAG retrieval type: bm25, vector, or hybrid (default: hybrid)")
    parser.add_argument("--no-primitive-classes", action="store_true",
                        help="Ablation: disable primitive class types in query analysis")
    
    # Output directory for query analysis results
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for query analysis results (default: lancedb/eval_results/)")
    
    # Index base path for experiment isolation
    parser.add_argument("--index-base-path", type=str, default=None,
                        help="Base path for indexes (for experiment isolation)")
    
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set log level from --debug or LOGURU_LEVEL env var
    import sys
    logger.remove()
    if args.debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        log_level = os.environ.get("LOGURU_LEVEL", "DEBUG")
        logger.add(sys.stderr, level=log_level)
    
    if args.analyze_queries:
        cmd_analyze_queries(args)
    elif args.eval_full:
        cmd_eval_full(args)
    elif args.eval:
        cmd_eval(args)
    elif args.query:
        cmd_search(args)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python demos/retrieval.py -q 'Which team won the championship?'")
        print("  python demos/retrieval.py --eval -d fetaqa -n 100  # Semantic only (default index)")
        print("  python demos/retrieval.py --eval -d fetaqa -n 100 --index-key td_cd  # Use td_cd index")
        print("  python demos/retrieval.py --eval-full -d fetaqa -n 10  # Full pipeline")
        print("  python demos/retrieval.py --analyze-queries -d fetaqa -n 100 --llm gemini  # HyDE + constraints")
        print("  python demos/retrieval.py --analyze-queries -d fetaqa --use-rag --rag-top-k 5  # RAG-enhanced")
        print("\nIndex Keys (--index-key):")
        print("  td: table_description only (smallest)")
        print("  td_cd: table_description + column_descriptions")
        print("  td_cd_cs: all three fields (default, most comprehensive)")
        print("\nRelated Tools:")
        print("  Index Generation:    python demos/run_upo_pipeline.py -d fetaqa -s retrieval_index")
        print("  HyDE Evaluation:     python scripts/eval/analyze_hyde_retrieval.py -d fetaqa --full-compare")


if __name__ == "__main__":
    main()