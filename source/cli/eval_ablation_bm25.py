#!/usr/bin/env python3
"""
Unified BM25 Evaluation for Ablation Experiments

Loads all 16 ablation variants, filters to the intersection of queries
where HyDE succeeded (no error) across ALL variants, then evaluates
BM25 retrieval for both 'raw' and 'combined' (HyDE) modes on this
common query set for fair comparison.

Results are exported to source/draw/data/.

Usage:
    python source/cli/eval_ablation_bm25.py
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
import _path_setup  # noqa: F401

import bm25s
import Stemmer
from loguru import logger

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_BASE = _PROJECT_ROOT / "data" / "lake" / "experiments"
OUTPUT_DIR = _PROJECT_ROOT / "source" / "draw" / "data"

VARIANTS = [
    # (label, variant_dir_path)
    ("iter1",  EXP_BASE / "iter_ablation_fetaqa_20260306_153225" / "stage2_iter1"),
    ("iter3",  EXP_BASE / "iter_ablation_fetaqa_20260306_153225" / "stage2_iter3"),
    ("iter5",  EXP_BASE / "iter_ablation_fetaqa_20260306_153225" / "stage2_iter5"),
    ("iter7",  EXP_BASE / "iter_ablation_fetaqa_20260306_153225" / "stage2_iter7"),
    ("iter10", EXP_BASE / "iter_ablation_fetaqa_20260306_153225" / "stage2_iter10"),
    ("q50",    EXP_BASE / "query_ablation_fetaqa_20260306_154600" / "stage2_q50"),
    ("q100",   EXP_BASE / "query_ablation_fetaqa_20260306_154600" / "stage2_q100"),
    ("q200",   EXP_BASE / "query_ablation_fetaqa_20260306_154600" / "stage2_q200"),
    ("q400",   EXP_BASE / "query_ablation_fetaqa_20260306_154600" / "stage2_q400"),
    ("q700",   EXP_BASE / "query_ablation_fetaqa_20260306_162558" / "stage2_q700"),
    ("q1000",  EXP_BASE / "query_ablation_fetaqa_20260306_162558" / "stage2_q1000"),
    ("tc10",   EXP_BASE / "concept_ablation_fetaqa_20260306_164425" / "stage2_tc10"),
    ("tc25",   EXP_BASE / "concept_ablation_fetaqa_20260306_164425" / "stage2_tc25"),
    ("tc50",   EXP_BASE / "concept_ablation_fetaqa_20260306_164425" / "stage2_tc50"),
    ("tc75",   EXP_BASE / "concept_ablation_fetaqa_20260306_164425" / "stage2_tc75"),
    ("tc100",  EXP_BASE / "concept_ablation_fetaqa_20260306_164425" / "stage2_tc100"),
]

ANALYSIS_FILENAME = "fetaqa_test_unified_analysis_all_local_rag3_vector.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_analysis(variant_dir: Path) -> list:
    """Load unified analysis JSON for one variant."""
    filepath = variant_dir / "eval_results" / ANALYSIS_FILENAME
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_bm25_index(variant_dir: Path):
    """Load BM25 retriever and table_ids for one variant."""
    from workflows.retrieval.config import INDEX_KEY_TD_CD_CS

    index_path = variant_dir / "indexes" / INDEX_KEY_TD_CD_CS / "bm25"
    bm25_retriever = bm25s.BM25.load(str(index_path / "index"), load_corpus=False)
    with open(index_path / "table_ids.pkl", "rb") as f:
        table_ids = pickle.load(f)
    return bm25_retriever, table_ids


# ---------------------------------------------------------------------------
# BM25 search helpers
# ---------------------------------------------------------------------------

def make_searcher(bm25_retriever, table_ids, stemmer):
    """Return a closure that maps query text -> list[(table_id, score)]."""
    def search(query_text: str, top_k: int = 100):
        tokens = bm25s.tokenize([query_text], stemmer=stemmer, show_progress=False)
        res = bm25_retriever.retrieve(tokens, k=min(top_k, len(table_ids)), show_progress=False)
        scores = res.scores[0] if len(res.scores) > 0 else []
        doc_indices = res.documents[0] if len(res.documents) > 0 else []
        return [
            (table_ids[int(doc_indices[i])], float(scores[i]))
            for i in range(len(scores))
            if int(doc_indices[i]) < len(table_ids)
        ]
    return search


def get_rank(results_list, gt_table):
    for rank, (tid, _) in enumerate(results_list, 1):
        if tid == gt_table:
            return rank
    return len(results_list) + 1


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stemmer = Stemmer.Stemmer("english")

    # ------------------------------------------------------------------
    # Phase 1: Load all analysis data and compute common success indices
    # ------------------------------------------------------------------
    logger.info("Phase 1: Loading analysis data from all 16 variants ...")
    all_data: dict[str, list] = {}
    for label, vdir in VARIANTS:
        data = load_analysis(vdir)
        all_data[label] = data
        n_err = sum(1 for item in data if item.get("error"))
        logger.info(f"  {label}: {len(data)} items, {n_err} errors")

    # Determine common success indices (no error in ANY variant)
    n_items = len(next(iter(all_data.values())))
    common_ok = set(range(n_items))
    for label, data in all_data.items():
        ok = {i for i, item in enumerate(data) if not item.get("error")}
        common_ok &= ok

    common_ok = sorted(common_ok)
    logger.info(f"Common success set: {len(common_ok)} / {n_items} queries")

    # ------------------------------------------------------------------
    # Phase 2: Evaluate each variant on the common query set
    # ------------------------------------------------------------------
    logger.info("Phase 2: Running BM25 evaluation on common query set ...")
    all_results: dict[str, dict] = {}

    for label, vdir in VARIANTS:
        logger.info(f"Evaluating {label} ...")
        bm25_retriever, table_ids = load_bm25_index(vdir)
        search = make_searcher(bm25_retriever, table_ids, stemmer)
        data = all_data[label]

        metrics = {
            "raw":      {"hits_1": 0, "hits_5": 0, "hits_10": 0, "ranks": []},
            "combined": {"hits_1": 0, "hits_5": 0, "hits_10": 0, "ranks": []},
        }

        for idx in common_ok:
            item = data[idx]
            query = item["query"]
            gt_table = item["gt_table"]
            analysis = item.get("analysis") or {}

            table_desc = analysis.get("hypothetical_table_description", "")
            col_descs = analysis.get("hypothetical_column_descriptions", [])
            combined_text = f"{table_desc} {' '.join(col_descs)}".strip()

            for mode, text in [("raw", query), ("combined", combined_text or query)]:
                rank = get_rank(search(text), gt_table)
                metrics[mode]["ranks"].append(rank)
                if rank == 1:
                    metrics[mode]["hits_1"] += 1
                if rank <= 5:
                    metrics[mode]["hits_5"] += 1
                if rank <= 10:
                    metrics[mode]["hits_10"] += 1

        # Compute aggregate metrics
        n = len(common_ok)
        for mode in metrics:
            r = metrics[mode]
            r["recall_1"] = r["hits_1"] / n
            r["recall_5"] = r["hits_5"] / n
            r["recall_10"] = r["hits_10"] / n
            r["mrr"] = sum(1.0 / rk for rk in r["ranks"]) / n
            r["n_queries"] = n
            del r["ranks"]  # no need to export

        all_results[label] = metrics
        logger.info(
            f"  {label}: raw R@1={metrics['raw']['recall_1']*100:.1f}%  "
            f"combined R@1={metrics['combined']['recall_1']*100:.1f}%"
        )

    # ------------------------------------------------------------------
    # Phase 3: Export
    # ------------------------------------------------------------------
    output = {
        "common_query_count": len(common_ok),
        "total_query_count": n_items,
        "variants": all_results,
    }
    out_path = OUTPUT_DIR / "ablation_bm25_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results exported to {out_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"BM25 Ablation Results  (common queries: {len(common_ok)}/{n_items})")
    print(f"{'='*70}")
    print(f"{'Variant':<10} {'raw R@1':>10} {'comb R@1':>10} {'raw R@5':>10} {'comb R@5':>10} {'raw MRR':>10} {'comb MRR':>10}")
    print("-" * 70)
    for label, m in all_results.items():
        print(
            f"{label:<10} "
            f"{m['raw']['recall_1']*100:>9.1f}% "
            f"{m['combined']['recall_1']*100:>9.1f}% "
            f"{m['raw']['recall_5']*100:>9.1f}% "
            f"{m['combined']['recall_5']*100:>9.1f}% "
            f"{m['raw']['mrr']:>10.4f} "
            f"{m['combined']['mrr']:>10.4f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
