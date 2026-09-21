"""UPO-aligned query transformation and table retrieval.

Examples:
    python source/cli/retrieval.py -d fetaqa -q "Which team won the championship?" --use-rag
    python source/cli/retrieval.py --analyze-queries -d fetaqa -n -1 --use-rag
    python source/cli/retrieval.py --eval -d fetaqa --rag-type bm25 -n 100
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

    search_text = args.query
    if not args.raw_query:
        import asyncio

        async def transform_query():
            if args.use_rag:
                from workflows.retrieval.nodes.rag_unified_query_analysis import (
                    _rag_unified_query_analysis_impl, process_unified_analysis,
                )
                raw = await _rag_unified_query_analysis_impl(
                    query=args.query, dataset_name=args.dataset, llm_purpose=args.llm,
                    rag_top_k=args.rag_top_k, rag_type=args.rag_type,
                    use_primitive_classes=not args.no_primitive_classes,
                    index_base_path=Path(args.index_base_path) if args.index_base_path else None,
                )
                return process_unified_analysis(raw, dataset_name=args.dataset)
            from workflows.retrieval.nodes.unified_query_analysis import _unified_query_analysis_impl
            return await _unified_query_analysis_impl(
                query=args.query, dataset_name=args.dataset, llm_purpose=args.llm,
            )

        analysis = asyncio.run(transform_query())
        search_text = "\n".join([
            analysis.get("hypothetical_table_description", ""),
            analysis.get("hypothetical_column_descriptions", ""),
        ]).strip() or args.query

    results = unified_search(
        query=search_text,
        dataset_name=args.dataset,
        top_k=args.top_k,
        rag_type=rag_type,
        index_key=index_key,
        index_base_path=Path(args.index_base_path) if args.index_base_path else None,
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


def cmd_eval(args):
    """Evaluate transformed queries against their ground-truth answer tables."""
    from evaluation.runners.hyde_retrieval import (
        load_unified_analysis, analyze_hyde_mode, print_metrics,
    )
    from workflows.retrieval.unified_search import load_unified_indexes, get_text_embedder

    data = load_unified_analysis(args.dataset, llm_suffix=args.llm)
    if not data:
        raise ValueError("The query-analysis file is empty.")
    faiss_index, metadata, bm25, table_ids = load_unified_indexes(
        args.dataset, args.index_key,
        index_base_path=Path(args.index_base_path) if args.index_base_path else None,
    )
    if args.rag_type in ('vector', 'hybrid') and faiss_index is None:
        raise ValueError('Build the vector index before evaluating this retriever.')
    if args.rag_type in ('bm25', 'hybrid') and bm25 is None:
        raise ValueError('Build the BM25 index before evaluating this retriever.')
    embedder = get_text_embedder() if args.rag_type != "bm25" and faiss_index is not None else None
    metrics = analyze_hyde_mode(
        data, faiss_index, metadata, bm25, table_ids, embedder,
        mode="combined", top_k=args.top_k, num_queries=args.num_queries,
        retriever_type=args.rag_type,
    )
    print_metrics(metrics, "Saturn", metrics["total"])
    return metrics


def cmd_analyze_queries(args):
    """Generate UPO-aligned table and column descriptions for benchmark queries.

    Uses a single LLM call to generate:
    1. Table/column descriptions for profile retrieval
    2. TBox constraints (column types for query annotation)
    3. ABox constraints (explicit values in query for query annotation)

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
    from llm.statistics import (
        get_usage_stats,
        reset_usage_stats,
        set_current_caller,
        set_current_phase,
    )

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

    # Reset and attribute LLM usage to the inference-phase query-analysis caller.
    # ContextVars set here are inherited by the asyncio tasks spawned below,
    # so every analysis LLM call is recorded under the 'query_analysis' caller.
    reset_usage_stats()
    set_current_caller("query_analysis")
    set_current_phase("inference")

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
            'gt_tables': gt_tables,       # All valid answer tables
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

    # Persist LLM token usage for the query-analysis (inference) phase so the
    # cost can be compared against baselines. Schema matches the main pipeline's
    # _collect_llm_stats() output.
    usage = get_usage_stats()
    llm_stats = {
        "total_requests": usage.get("total_requests", 0),
        "total_input_tokens": usage.get("total_input_tokens", 0),
        "total_output_tokens": usage.get("total_output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "by_model": usage.get("by_model", {}),
        "by_caller": usage.get("by_caller", {}),
        "cache": usage.get("cache", {}),
    }
    stats_file = output_file.with_name(f"{output_file.stem}_llm_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(llm_stats, f, indent=2, ensure_ascii=False)
    print(f"💾 LLM stats saved to: {stats_file}")
    print(
        f"   LLM usage -> requests: {llm_stats['total_requests']}, "
        f"input: {llm_stats['total_input_tokens']}, "
        f"output: {llm_stats['total_output_tokens']}, "
        f"total: {llm_stats['total_tokens']}"
    )

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
    parser = argparse.ArgumentParser(description="Saturn query transformation and retrieval")
    parser.add_argument("-d", "--dataset", default="fetaqa", help="Dataset name")

    # Modes
    parser.add_argument("--eval", action="store_true", help="Evaluate cached UPO-aligned queries")
    parser.add_argument("--analyze-queries", action="store_true",
                        help="Unified query analysis: HyDE descriptions + TBox/ABox constraints (single LLM call)")
    parser.add_argument("-q", "--query", type=str, help="Search query")

    # Index configuration
    parser.add_argument("--index-key", type=str, choices=["td", "td_cd", "td_cd_cs"], default=None,
                        help="Index configuration: td (table_desc), td_cd (+column_desc), td_cd_cs (full, default)")

    # Options
    parser.add_argument("-k", "--top-k", type=int, default=100, help="Top K results")
    parser.add_argument("-n", "--num-queries", type=int, default=100, help="Number of queries for eval (-1 for all)")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"],
                        help="Query split to use: test or train (default: test)")
    parser.add_argument("--raw-query", action="store_true", help="Search without UPO-aligned query transformation")

    # LLM and parallel options
    parser.add_argument("--llm", "--llm-purpose", type=str, default="local", help="LLM to use (default, gemini, local)")
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
    elif args.eval:
        cmd_eval(args)
    elif args.query:
        cmd_search(args)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python source/cli/retrieval.py -q 'Which team won the championship?'")
        print("  python source/cli/retrieval.py --eval -d fetaqa -n 100  # Semantic only (default index)")
        print("  python source/cli/retrieval.py --eval -d fetaqa -n 100 --index-key td_cd  # Use td_cd index")
        print("  python source/cli/retrieval.py --analyze-queries -d fetaqa -n 100 --llm gemini  # HyDE + constraints")
        print("  python source/cli/retrieval.py --analyze-queries -d fetaqa --use-rag --rag-top-k 5  # RAG-enhanced")
        print("\nIndex Keys (--index-key):")
        print("  td: table_description only (smallest)")
        print("  td_cd: table_description + column_descriptions")
        print("  td_cd_cs: all three fields (default, most comprehensive)")
        print("\nRelated Tools:")
        print("  Index Generation:    python source/cli/run_pipeline.py -d fetaqa -s retrieval_index")
        print("  HyDE Evaluation:     python source/evaluation/runners/hyde_retrieval.py -d fetaqa --full-compare")


if __name__ == "__main__":
    main()
