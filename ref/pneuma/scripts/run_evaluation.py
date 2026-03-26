#!/usr/bin/env python3
"""
Pneuma Evaluation Runner

This script wraps the original Pneuma evaluation code with minimal patches.
It handles:
1. Setting up paths and environment
2. Converting our summaries to Pneuma's expected format
3. Building indices using original Pneuma code
4. Running evaluation with table-level aggregation

Usage:
    python run_evaluation.py --dataset fetaqapn --mode full
    python run_evaluation.py --dataset fetaqapn --mode evaluate-only
"""

import argparse
import json
import os
import sys
import time
from enum import Enum
from pathlib import Path

import bm25s
import chromadb
import numpy as np
import Stemmer
import torch
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm


# ============================================================================
# Reranking Mode (Matching Pneuma's RerankingMode)
# ============================================================================

class RerankingMode(Enum):
    NONE = 0
    COSINE = 1      # Use embedding model to re-score
    LLM = 2         # Use LLM with yes/no relevance prompts
    DIRECT_SCORE = 3  # Use cross-encoder reranker


# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.absolute()
INTERFACE_DIR = SCRIPT_DIR.parent
REF_DIR = INTERFACE_DIR.parent.parent
SATURN_ROOT = REF_DIR.parent

# Default paths
DEFAULT_WORK_DIR = REF_DIR / "pneuma_work"
DEFAULT_MODEL_DIR = SATURN_ROOT / "model"
DEFAULT_EMBED_MODEL = DEFAULT_MODEL_DIR / "bge-m3"

# Hit@K values to evaluate
DEFAULT_K_VALUES = [1, 3, 5, 10, 20, 100]


# ============================================================================
# Utility Functions
# ============================================================================

def read_jsonl(file_path: str | Path) -> list:
    """Read JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def write_jsonl(data: list, file_path: str | Path):
    """Write JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ============================================================================
# Summary Conversion (Our format -> Pneuma's _splitted/_merged format)
# ============================================================================

def convert_summaries_to_pneuma_format(
    summaries_dir: Path,
    output_dir: Path,
    embed_model: SentenceTransformer,
    max_tokens: int = 512,
):
    """
    Convert our summaries to Pneuma's expected format.
    
    Input format (our format):
        schema_narrations.jsonl: {"table": "...", "column": "...", "summary": "..."}
        sample_rows.jsonl: {"table": "...", "row_idx": N, "summary": "col1: val1 | col2: val2"}
    
    Output format (Pneuma's format):
        {dataset}_splitted.jsonl: Multiple columns merged up to 512 tokens, separated by " | "
        {dataset}_merged.jsonl: Multiple rows merged up to 512 tokens, separated by " || "
    """
    tokenizer = embed_model.tokenizer
    
    # Process schema narrations -> _splitted.jsonl
    schema_file = summaries_dir / "schema_narrations.jsonl"
    if schema_file.exists():
        print(f"Converting schema narrations...")
        schema_data = read_jsonl(schema_file)
        
        # Group by table
        table_columns = {}
        for item in schema_data:
            table = item["table"]
            col = item.get("column", "")
            summary = item.get("summary", "")
            if table not in table_columns:
                table_columns[table] = []
            # Format: "colname: description"
            table_columns[table].append(f"{col}: {summary}")
        
        # Split/merge columns up to max_tokens
        splitted_data = []
        for table in sorted(table_columns.keys()):
            columns = table_columns[table]
            col_idx = 0
            block_idx = 0
            
            while col_idx < len(columns):
                processed = columns[col_idx]
                
                # Try to merge more columns
                while (col_idx + 1) < len(columns):
                    temp = processed + " | " + columns[col_idx + 1]
                    if len(tokenizer.encode(temp)) < max_tokens:
                        processed = temp
                        col_idx += 1
                    else:
                        break
                
                col_idx += 1
                splitted_data.append({
                    "source_ids": [f"{table}_SEP_contents_SEP_schema"],
                    "table": table,
                    "summary": processed,
                })
                block_idx += 1
        
        # Write output
        output_file = output_dir / "schema_narrations_splitted.jsonl"
        write_jsonl(splitted_data, output_file)
        print(f"  Written {len(splitted_data)} schema blocks to {output_file}")
    
    # Process sample rows -> _merged.jsonl
    row_file = summaries_dir / "sample_rows.jsonl"
    if row_file.exists():
        print(f"Converting sample rows...")
        row_data = read_jsonl(row_file)
        
        # Group by table
        table_rows = {}
        for item in row_data:
            table = item["table"]
            summary = item.get("summary", "")
            row_id = item.get("id", f"{table}_SEP_contents_SEP_row-{len(table_rows.get(table, []))}")
            if summary:
                if table not in table_rows:
                    table_rows[table] = []
                table_rows[table].append({"id": row_id, "summary": summary})
        
        # Merge rows up to max_tokens
        merged_data = []
        for table in sorted(table_rows.keys()):
            rows = table_rows[table]
            row_idx = 0
            
            while row_idx < len(rows):
                processed = rows[row_idx]["summary"]
                source_ids = [rows[row_idx]["id"]]
                
                # Try to merge more rows
                while (row_idx + 1) < len(rows):
                    temp = processed + " || " + rows[row_idx + 1]["summary"]
                    if len(tokenizer.encode(temp)) < max_tokens:
                        source_ids.append(rows[row_idx + 1]["id"])
                        processed = temp
                        row_idx += 1
                    else:
                        break
                
                row_idx += 1
                merged_data.append({
                    "source_ids": source_ids,
                    "table": table,
                    "summary": processed,
                })
        
        # Write output
        output_file = output_dir / "sample_rows_merged.jsonl"
        write_jsonl(merged_data, output_file)
        print(f"  Written {len(merged_data)} row blocks to {output_file}")


# ============================================================================
# Index Building (Using Pneuma's logic)
# ============================================================================

def build_indices(
    summaries_dir: Path,
    indices_dir: Path,
    embed_model: SentenceTransformer,
    dataset_name: str,
):
    """
    Build ChromaDB and BM25 indices matching Pneuma's format.
    
    Reads from:
        schema_narrations_splitted.jsonl
        sample_rows_merged.jsonl
    """
    print(f"\nBuilding indices for {dataset_name}...")
    
    # Load summaries
    schema_file = summaries_dir / "schema_narrations_splitted.jsonl"
    row_file = summaries_dir / "sample_rows_merged.jsonl"
    
    schema_contents = read_jsonl(schema_file) if schema_file.exists() else []
    row_contents = read_jsonl(row_file) if row_file.exists() else []
    
    if not schema_contents and not row_contents:
        raise ValueError(f"No summaries found in {summaries_dir}")
    
    # Determine tables
    if row_contents:
        tables = sorted(set(content["table"] for content in row_contents))
    else:
        tables = sorted(set(content["table"] for content in schema_contents))
    
    print(f"  Found {len(tables)} tables")
    print(f"  Schema blocks: {len(schema_contents)}")
    print(f"  Row blocks: {len(row_contents)}")
    
    # Prepare documents
    documents = []
    ids = []
    corpus_json = []  # For BM25
    
    for table in tables:
        # Schema documents
        table_schemas = [c for c in schema_contents if c["table"] == table]
        for idx, schema in enumerate(table_schemas):
            doc_id = f"{table}_SEP_contents_SEP_schema-{idx}"
            documents.append(schema["summary"])
            ids.append(doc_id)
            corpus_json.append({
                "text": schema["summary"],
                "metadata": {"table": doc_id}
            })
        
        # Row documents
        table_rows = [c for c in row_contents if c["table"] == table]
        for idx, row in enumerate(table_rows):
            doc_id = f"{table}_SEP_contents_SEP_row-{idx}"
            documents.append(row["summary"])
            ids.append(doc_id)
            corpus_json.append({
                "text": row["summary"],
                "metadata": {"table": doc_id}
            })
    
    print(f"  Total documents: {len(documents)}")
    
    # Build ChromaDB index
    print("  Building ChromaDB index...")
    chroma_path = indices_dir / "chroma_index"
    if chroma_path.exists():
        import shutil
        shutil.rmtree(chroma_path)
    
    client = chromadb.PersistentClient(str(chroma_path))
    collection = client.create_collection(
        name="benchmark",
        metadata={
            "hnsw:space": "cosine",
        },
    )
    
    # Batch embedding - use smaller batch size to avoid OOM
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for i in tqdm(range(0, len(documents), batch_size), desc="    Embedding"):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        
        embeddings = embed_model.encode(
            batch_docs,
            batch_size=batch_size,
            show_progress_bar=False,
            device=device,
        )
        
        collection.add(
            embeddings=[embed.tolist() for embed in embeddings],
            documents=batch_docs,
            ids=batch_ids,
        )
    
    print(f"    ChromaDB index: {collection.count()} documents")
    
    # Build BM25 index
    print("  Building BM25 index...")
    stemmer = Stemmer.Stemmer("english")
    
    corpus_text = [doc["text"] for doc in corpus_json]
    corpus_tokens = bm25s.tokenize(
        corpus_text,
        stopwords="en",
        stemmer=stemmer,
        show_progress=False,
    )
    
    retriever = bm25s.BM25(corpus=corpus_json)
    retriever.index(corpus_tokens, show_progress=True)
    
    bm25_path = indices_dir / "bm25_index"
    retriever.save(str(bm25_path))
    print(f"    BM25 index saved to {bm25_path}")
    
    return collection, retriever


# ============================================================================
# Hybrid Retrieval (Matching Pneuma's logic exactly)
# ============================================================================

class HybridRetriever:
    """
    Pneuma-style hybrid retrieval (vector + BM25).
    
    This matches the original Pneuma HybridRetriever logic exactly,
    including reranking support.
    """
    
    def __init__(
        self,
        reranker=None,
        reranking_mode: RerankingMode = RerankingMode.NONE,
        embed_model: SentenceTransformer = None,
        llm_client=None,  # OpenAI-compatible client for LLM rerank
        llm_model: str = None,  # Model name for LLM rerank
    ):
        self.stemmer = Stemmer.Stemmer("english")
        self.reranker = reranker  # CrossEncoder for DIRECT_SCORE mode
        self.reranking_mode = reranking_mode
        self.embed_model = embed_model  # For COSINE rerank
        self.llm_client = llm_client  # OpenAI-compatible client
        self.llm_model = llm_model  # Model name
    
    def _process_nodes_bm25(
        self,
        bm25_res,
        missing_ids: list,
        dictionary_id_bm25: dict,
        bm25_retriever,
        query_tokens,
    ) -> dict:
        """Process BM25 results and augment with missing IDs."""
        from bm25s.tokenization import convert_tokenized_to_string_list
        
        results = [node for node in bm25_res[0][0]]
        scores = [float(s) for s in bm25_res[1][0]]
        
        # Add missing documents from vector search
        for one_id in missing_ids:
            if one_id in dictionary_id_bm25:
                results.append(bm25_retriever.corpus[dictionary_id_bm25[one_id]])
                query_tokens_str = convert_tokenized_to_string_list(query_tokens)[0]
                score = bm25_retriever.get_scores(query_tokens_str)[dictionary_id_bm25[one_id]]
                scores.append(float(score))
        
        # Min-max normalize
        max_score = max(scores) if scores else 1
        min_score = min(scores) if scores else 0
        
        processed = {}
        for i, node in enumerate(results):
            doc_id = node["metadata"]["table"]
            norm_score = 1.0 if min_score == max_score else (scores[i] - min_score) / (max_score - min_score)
            processed[doc_id] = (norm_score, node["text"])
        
        return processed
    
    def _process_nodes_vec(
        self,
        vec_res,
        missing_ids: list,
        collection,
        question_embedding: list,
    ) -> dict:
        """Process vector results and augment with missing IDs."""
        # Get additional info for missing IDs
        if missing_ids:
            try:
                extra_info = collection.get(
                    ids=missing_ids,
                    include=["documents", "embeddings"]
                )
                vec_res["ids"][0].extend(extra_info["ids"])
                vec_res["documents"][0].extend(extra_info["documents"])
                for i in range(len(extra_info["ids"])):
                    dist = cosine(question_embedding, extra_info["embeddings"][i])
                    vec_res["distances"][0].append(dist)
            except Exception:
                pass  # Some IDs might not exist
        
        # Convert distances to scores: score = 1 - distance
        scores = [1.0 - dist for dist in vec_res["distances"][0]]
        documents = vec_res["documents"][0]
        ids = vec_res["ids"][0]
        
        # Min-max normalize
        max_score = max(scores) if scores else 1
        min_score = min(scores) if scores else 0
        
        processed = {}
        for idx in range(len(scores)):
            norm_score = 1.0 if min_score == max_score else (scores[idx] - min_score) / (max_score - min_score)
            processed[ids[idx]] = (norm_score, documents[idx])
        
        return processed
    
    def retrieve(
        self,
        bm25_retriever,
        collection,
        bm25_res,
        vec_res,
        k: int,
        question: str,
        alpha: float,
        query_tokens,
        question_embedding: list,
        dictionary_id_bm25: dict,
    ) -> list:
        """
        Perform hybrid retrieval.
        
        Returns: List of (passage_id, score, doc) tuples
        """
        # Convert vec_res to mutable
        vec_res = {
            "ids": [list(vec_res["ids"][0])],
            "documents": [list(vec_res["documents"][0])],
            "distances": [list(vec_res["distances"][0])],
        }
        
        vec_ids = set(vec_res["ids"][0])
        bm25_ids = {node["metadata"]["table"] for node in bm25_res[0][0]}
        
        # Process and augment
        processed_bm25 = self._process_nodes_bm25(
            bm25_res,
            list(vec_ids - bm25_ids),
            dictionary_id_bm25,
            bm25_retriever,
            query_tokens,
        )
        processed_vec = self._process_nodes_vec(
            vec_res,
            list(bm25_ids - vec_ids),
            collection,
            question_embedding,
        )
        
        # Combine scores: alpha * bm25 + (1-alpha) * vec
        all_nodes = []
        for node_id in sorted(vec_ids | bm25_ids):
            bm25_score_doc = processed_bm25.get(node_id, (0.0, None))
            vec_score_doc = processed_vec.get(node_id, (0.0, None))
            combined_score = alpha * bm25_score_doc[0] + (1 - alpha) * vec_score_doc[0]
            doc = bm25_score_doc[1] if bm25_score_doc[1] is not None else vec_score_doc[1]
            all_nodes.append((node_id, combined_score, doc))
        
        # Sort by score (descending), then by ID for tie-breaking
        sorted_nodes = sorted(all_nodes, key=lambda x: (-x[1], x[0]))[:k]
        
        # Apply reranking if enabled
        if self.reranking_mode == RerankingMode.COSINE:
            sorted_nodes = self._cosine_rerank(sorted_nodes, question)
        elif self.reranking_mode == RerankingMode.DIRECT_SCORE:
            sorted_nodes = self._direct_rerank(sorted_nodes, question)
        elif self.reranking_mode == RerankingMode.LLM:
            sorted_nodes = self._llm_rerank(sorted_nodes, question)
        
        return sorted_nodes
    
    def _cosine_rerank(
        self,
        nodes: list,
        question: str,
    ) -> list:
        """
        Rerank using embedding model cosine similarity.
        Matches Pneuma's _cosine_rerank.
        """
        if not self.embed_model or not nodes:
            return nodes
        
        names = [node[0] for node in nodes]
        docs = [node[2] for node in nodes]
        
        # Encode documents and question
        docs_embeddings = self.embed_model.encode(
            docs,
            batch_size=100,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        question_embedding = self.embed_model.encode(
            question,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        # Compute cosine similarities
        similarities = [
            1 - cosine(question_embedding, doc_embedding)
            for doc_embedding in docs_embeddings
        ]
        
        # Sort by new similarity
        reranked_nodes = sorted(
            zip(names, similarities, docs),
            key=lambda x: (-x[1], x[0])
        )
        return list(reranked_nodes)
    
    def _direct_rerank(
        self,
        nodes: list,
        question: str,
    ) -> list:
        """
        Rerank using cross-encoder.
        Matches Pneuma's _direct_rerank.
        """
        if not self.reranker or not nodes:
            return nodes
        
        names = [node[0] for node in nodes]
        docs = [node[2] for node in nodes]
        
        # Compute cross-encoder scores
        pairs = [(question, doc) for doc in docs]
        scores = self.reranker.predict(pairs)
        
        # Sort by new scores
        reranked_nodes = sorted(
            zip(names, scores, docs),
            key=lambda x: (-x[1], x[0])
        )
        return list(reranked_nodes)
    
    def _get_relevance_prompt(self, desc: str, desc_type: str, question: str) -> str:
        """Generate relevance prompt for LLM reranking."""
        if desc_type == "content":
            return f"""Given a table with the following columns:
*/
{desc}
*/
and this question:
/*
{question}
*/
Is the table relevant to answer the question? Begin your answer with yes/no."""
        else:  # context
            return f"""Given this context describing a table:
*/
{desc}
*/
and this question:
/*
{question}
*/
Is the table relevant to answer the question? Begin your answer with yes/no."""
    
    def _llm_rerank(
        self,
        nodes: list,
        question: str,
    ) -> list:
        """
        Rerank using LLM with yes/no relevance prompts.
        Uses OpenAI-compatible API for inference with batch/concurrent calls.
        """
        if not self.llm_client or not self.llm_model or not nodes:
            return nodes
        
        node_tables = [node[0] for node in nodes]
        
        # Generate relevance prompts (matches Pneuma's format)
        relevance_prompts = []
        for node in nodes:
            desc_type = "content" if node[0].split("_SEP_")[1].startswith("contents") else "context"
            prompt = self._get_relevance_prompt(node[2], desc_type, question)
            relevance_prompts.append(prompt)
        
        # Batch LLM inference using concurrent calls
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            tables_relevance = {}
            
            def call_llm(idx_prompt):
                idx, prompt = idx_prompt
                try:
                    response = self.llm_client.chat.completions.create(
                        model=self.llm_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=10,
                        temperature=0,
                    )
                    answer = response.choices[0].message.content.strip().lower()
                    return idx, answer.startswith("yes")
                except Exception as e:
                    return idx, False
            
            # Use ThreadPoolExecutor for concurrent calls (workers = num prompts for max parallelism)
            num_workers = len(relevance_prompts)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(call_llm, (i, p)): i for i, p in enumerate(relevance_prompts)}
                for future in as_completed(futures):
                    idx, is_relevant = future.result()
                    tables_relevance[node_tables[idx]] = is_relevant
            
            # Reorder: relevant first, then non-relevant (matches Pneuma's logic)
            new_nodes = [
                (table_name, score, doc)
                for table_name, score, doc in nodes
                if tables_relevance.get(table_name, False)
            ] + [
                (table_name, score, doc)
                for table_name, score, doc in nodes
                if not tables_relevance.get(table_name, False)
            ]
            return new_nodes
            
        except Exception as e:
            print(f"Warning: LLM rerank failed: {e}")
            import traceback
            traceback.print_exc()
            return nodes


# ============================================================================
# Evaluation
# ============================================================================

def aggregate_to_tables(all_nodes: list, k: int, total_tables: int) -> list:
    """
    Aggregate passage-level results to table-level.
    
    Returns: List of (table_id, max_score) tuples
    """
    table_scores = {}
    for node_id, score, _ in all_nodes:
        table_id = node_id.split("_SEP_")[0]
        if table_id not in table_scores or score > table_scores[table_id]:
            table_scores[table_id] = score
    
    sorted_tables = sorted(table_scores.items(), key=lambda x: (-x[1], x[0]))
    # Safety: don't return more tables than exist
    safe_k = min(k, total_tables, len(sorted_tables))
    return sorted_tables[:safe_k]


def evaluate(
    benchmark: list,
    collection,
    bm25_retriever,
    embed_model: SentenceTransformer = None,
    question_key: str = "question",
    k_values: list = None,
    n: int = 5,
    alpha: float = 0.5,
    reranking_mode: RerankingMode = RerankingMode.NONE,
    reranker=None,
    llm_client=None,
    llm_model: str = None,
    rerank_top_k: int = None,
):
    """
    Evaluate hybrid retrieval on benchmark.
    
    Args:
        benchmark: List of query dicts with "question" and "answer_tables"
        collection: ChromaDB collection (can be None for BM25-only)
        bm25_retriever: BM25 retriever (can be None for Vector-only)
        embed_model: Sentence transformer model (can be None for BM25-only)
        question_key: Key for question field
        k_values: List of K values for Hit@K (e.g., [1, 3, 5, 10, 20, 100])
        n: Multiplier for retrieval pool
        alpha: BM25 weight (0-1). alpha=1.0 for BM25-only, alpha=0.0 for Vector-only
        reranking_mode: Reranking mode (NONE, COSINE, DIRECT_SCORE, LLM)
        reranker: CrossEncoder for DIRECT_SCORE mode
        llm_client: OpenAI-compatible client for LLM mode
        llm_model: Model name for LLM mode
        rerank_top_k: Number of tables to rerank (default: use max k_values)
    
    Returns:
        Dict with results
    """
    if k_values is None:
        k_values = DEFAULT_K_VALUES
    
    # Determine retrieval mode
    bm25_only = (alpha == 1.0)
    vector_only = (alpha == 0.0)
    
    if bm25_only:
        print("Mode: BM25-only (alpha=1.0)")
    elif vector_only:
        print("Mode: Vector-only (alpha=0.0)")
    else:
        print(f"Mode: Hybrid (alpha={alpha})")
    
    # Setup
    stemmer = Stemmer.Stemmer("english")
    hybrid_retriever = HybridRetriever(
        reranker=reranker,
        reranking_mode=reranking_mode,
        embed_model=embed_model,
        llm_client=llm_client,
        llm_model=llm_model,
    )
    
    # Build doc_id -> index mapping (for BM25)
    dictionary_id_bm25 = {}
    if bm25_retriever:
        dictionary_id_bm25 = {
            datum["metadata"]["table"]: idx
            for idx, datum in enumerate(bm25_retriever.corpus)
        }
    
    # Count unique tables
    if bm25_retriever:
        total_tables = len(set(
            node["metadata"]["table"].split("_SEP_")[0]
            for node in bm25_retriever.corpus
        ))
    else:
        total_tables = collection.count()
    print(f"Total unique tables: {total_tables}")
    print(f"Reranking mode: {reranking_mode.name}")
    if rerank_top_k:
        print(f"Rerank top-K: {rerank_top_k}")
    
    # Compute question embeddings (skip for BM25-only)
    questions = [q.get(question_key, "") for q in benchmark]
    question_embeddings = None
    
    if not bm25_only and embed_model is not None:
        print("Computing question embeddings...")
        question_embeddings = embed_model.encode(
            questions,
            batch_size=32,
            show_progress_bar=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        print("Skipping question embeddings (BM25-only mode)")
    
    # Evaluate
    max_k = max(k_values)
    # If rerank is enabled, we need to retrieve more for reranking pool
    retrieval_k = rerank_top_k if rerank_top_k else max_k
    increased_k = retrieval_k * n
    
    # Get total document count for safety
    if collection is not None:
        total_docs = collection.count()
    elif bm25_retriever is not None:
        total_docs = len(bm25_retriever.corpus)
    else:
        total_docs = 10000  # fallback
    # Safety: limit k to available documents
    safe_increased_k = min(increased_k, total_docs)
    
    hit_counts = {k: 0 for k in k_values}
    mrr_sum = 0.0
    total = 0
    wrong_questions = []
    
    start_time = time.time()
    
    for idx, datum in enumerate(tqdm(benchmark, desc="Evaluating")):
        answer_tables = datum.get("answer_tables", [])
        if not answer_tables or not questions[idx]:
            continue
        
        total += 1
        
        # Get question embedding (None for BM25-only)
        question_embedding = None
        if question_embeddings is not None:
            question_embedding = question_embeddings[idx].tolist()
        
        # Tokenize query for BM25
        query_tokens = bm25s.tokenize(
            questions[idx],
            stemmer=stemmer,
            show_progress=False,
        )
        
        # BM25-only mode: skip vector retrieval
        if bm25_only:
            # BM25 retrieval only
            results, scores = bm25_retriever.retrieve(
                query_tokens,
                k=safe_increased_k,
                show_progress=False,
            )
            # Convert to nodes format: (node_id, score, doc)
            all_nodes = []
            for i, node in enumerate(results[0]):
                node_id = node["metadata"]["table"]
                score = float(scores[0][i])
                doc = node["text"]
                all_nodes.append((node_id, score, doc))
            
            # Apply reranking if enabled
            if reranking_mode == RerankingMode.LLM:
                all_nodes = hybrid_retriever._llm_rerank(all_nodes[:increased_k], questions[idx])
            elif reranking_mode == RerankingMode.DIRECT_SCORE:
                all_nodes = hybrid_retriever._direct_rerank(all_nodes[:increased_k], questions[idx])
        
        # Vector-only mode: skip BM25
        elif vector_only:
            # Vector retrieval only
            vec_res = collection.query(
                query_embeddings=[question_embedding],
                n_results=safe_increased_k,
            )
            # Convert to nodes format
            all_nodes = []
            for i in range(len(vec_res["ids"][0])):
                node_id = vec_res["ids"][0][i]
                score = 1.0 - vec_res["distances"][0][i]  # Convert distance to score
                doc = vec_res["documents"][0][i]
                all_nodes.append((node_id, score, doc))
            
            # Apply reranking if enabled
            if reranking_mode == RerankingMode.LLM:
                all_nodes = hybrid_retriever._llm_rerank(all_nodes[:increased_k], questions[idx])
            elif reranking_mode == RerankingMode.DIRECT_SCORE:
                all_nodes = hybrid_retriever._direct_rerank(all_nodes[:increased_k], questions[idx])
        
        # Hybrid mode
        else:
            # BM25 retrieval
            results, scores = bm25_retriever.retrieve(
                query_tokens,
                k=safe_increased_k,
                show_progress=False,
            )
            bm25_res = (results, scores)
            
            # Vector retrieval
            vec_res = collection.query(
                query_embeddings=[question_embedding],
                n_results=safe_increased_k,
            )
            
            # Hybrid fusion
            all_nodes = hybrid_retriever.retrieve(
                bm25_retriever,
                collection,
                bm25_res,
                vec_res,
                increased_k,
                questions[idx],
                alpha,
                query_tokens,
                question_embedding,
                dictionary_id_bm25,
            )
        
        # Aggregate to table level
        top_tables = aggregate_to_tables(all_nodes, max_k, total_tables)
        
        # Find first hit for MRR
        first_hit_rank = None
        for rank, (table_id, _) in enumerate(top_tables):
            if table_id in answer_tables:
                first_hit_rank = rank + 1
                break
        
        # MRR
        if first_hit_rank is not None:
            mrr_sum += 1.0 / first_hit_rank
        
        # Hit@K
        for k in k_values:
            safe_k = min(k, len(top_tables))
            top_k_tables = set(t for t, _ in top_tables[:safe_k])
            if any(t in top_k_tables for t in answer_tables):
                hit_counts[k] += 1
        
        if first_hit_rank is None:
            wrong_questions.append(idx)
        
        # Progress checkpoint
        if (idx + 1) % 100 == 0:
            print(f"  Progress: {idx+1}/{len(benchmark)}, Hit@1: {hit_counts[1]}/{total}")
    
    elapsed = time.time() - start_time
    
    # Calculate metrics
    results = {
        "total_queries": total,
        "hit_rates": {},
        "mrr": round(mrr_sum / total, 4) if total > 0 else 0,
        "elapsed_seconds": round(elapsed, 2),
    }
    
    for k in k_values:
        rate = round(hit_counts[k] / total * 100, 2) if total > 0 else 0
        results["hit_rates"][k] = rate
    
    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Total queries: {total}")
    print(f"Total tables: {total_tables}")
    print()
    for k in sorted(k_values):
        print(f"Hit@{k}: {results['hit_rates'][k]:.2f}%")
    print()
    print(f"MRR: {results['mrr']:.4f}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Wrong questions (first 20): {wrong_questions[:20]}")
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pneuma Evaluation Runner")
    parser.add_argument("--dataset", "-d", required=True, help="Dataset name")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR),
                        help="Working directory")
    parser.add_argument("--embed-model", default=str(DEFAULT_EMBED_MODEL),
                        help="Path to embedding model")
    parser.add_argument("--mode", default="full",
                        choices=["full", "convert-only", "index-only", "evaluate-only"],
                        help="Execution mode")
    parser.add_argument("--question-key", default="question",
                        help="Key for question field in benchmark")
    parser.add_argument("--top-k", default="1,3,5,10,20,100",
                        help="Comma-separated K values for Hit@K")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="BM25 weight (0-1)")
    parser.add_argument("--output-file", help="Output JSON file")
    
    # Reranking arguments
    parser.add_argument("--rerank", default="none",
                        choices=["none", "cosine", "direct", "llm"],
                        help="Reranking mode: none, cosine (embedding), direct (cross-encoder), llm")
    parser.add_argument("--reranker-model", default=None,
                        help="Path to reranker model (for direct mode)")
    parser.add_argument("--openai-url", default="http://10.120.47.91:8000/v1",
                        help="OpenAI-compatible API URL (for llm mode)")
    parser.add_argument("--llm-model", default="Qwen3-Next-80B-A3B-Instruct",
                        help="LLM model name (for llm mode)")
    parser.add_argument("--rerank-top-k", type=int, default=None,
                        help="Number of tables to rerank (default: max of top-k values)")
    
    args = parser.parse_args()
    
    # Parse K values
    k_values = [int(k) for k in args.top_k.split(",")]
    
    # Parse reranking mode
    rerank_mode_map = {
        "none": RerankingMode.NONE,
        "cosine": RerankingMode.COSINE,
        "direct": RerankingMode.DIRECT_SCORE,
        "llm": RerankingMode.LLM,
    }
    reranking_mode = rerank_mode_map[args.rerank]
    
    # Setup paths
    work_dir = Path(args.work_dir) / args.dataset
    summaries_dir = work_dir / "summaries"
    indices_dir = work_dir / "indices"
    queries_file = work_dir / "converted" / "queries" / "test.jsonl"
    
    # Ensure directories exist
    summaries_dir.mkdir(parents=True, exist_ok=True)
    indices_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Dataset: {args.dataset}")
    print(f"Work directory: {work_dir}")
    print(f"Summaries: {summaries_dir}")
    print(f"Indices: {indices_dir}")
    print(f"Queries: {queries_file}")
    print(f"K values: {k_values}")
    print(f"Alpha: {args.alpha}")
    print(f"Reranking mode: {reranking_mode.name}")
    print()
    
    # Determine if we need embedding model
    bm25_only = (args.alpha == 1.0)
    need_embed_model = (
        args.mode in ["full", "convert-only", "index-only"]
        or not bm25_only
        or reranking_mode == RerankingMode.COSINE  # Cosine rerank needs embedding
    )
    
    # Load embedding model (skip for BM25-only evaluate-only)
    embed_model = None
    if need_embed_model:
        print(f"Loading embedding model: {args.embed_model}")
        embed_model = SentenceTransformer(args.embed_model)
    else:
        print("Skipping embedding model (BM25-only mode)")
    
    # Step 1: Convert summaries
    if args.mode in ["full", "convert-only"]:
        print("\n[Step 1] Converting summaries to Pneuma format...")
        convert_summaries_to_pneuma_format(
            summaries_dir=summaries_dir,
            output_dir=summaries_dir,
            embed_model=embed_model,
        )
    
    # Step 2: Build indices
    collection = None
    bm25_retriever = None
    
    if args.mode in ["full", "index-only"]:
        print("\n[Step 2] Building indices...")
        collection, bm25_retriever = build_indices(
            summaries_dir=summaries_dir,
            indices_dir=indices_dir,
            embed_model=embed_model,
            dataset_name=args.dataset,
        )
    else:
        # Load existing indices
        print("\n[Step 2] Loading existing indices...")
        chroma_path = indices_dir / "chroma_index"
        bm25_path = indices_dir / "bm25_index"
        
        # Load ChromaDB (skip for BM25-only)
        if not bm25_only:
            client = chromadb.PersistentClient(str(chroma_path))
            collection = client.get_collection("benchmark")
            print(f"  ChromaDB: {collection.count()} documents")
        else:
            print("  ChromaDB: skipped (BM25-only mode)")
        
        # Always load BM25 index
        bm25_retriever = bm25s.BM25.load(str(bm25_path), load_corpus=True)
        print(f"  BM25: {len(bm25_retriever.corpus)} documents")
    
    # Step 3: Evaluate
    if args.mode in ["full", "evaluate-only"]:
        print(f"\n[Step 3] Evaluating on {queries_file}...")
        
        if not queries_file.exists():
            print(f"ERROR: Queries file not found: {queries_file}")
            sys.exit(1)
        
        benchmark = read_jsonl(queries_file)
        print(f"Loaded {len(benchmark)} queries")
        
        # Load reranker if needed
        reranker = None
        llm_client = None
        llm_model = None
        
        if reranking_mode == RerankingMode.DIRECT_SCORE:
            if not args.reranker_model:
                print("ERROR: --reranker-model required for direct reranking mode")
                sys.exit(1)
            print(f"Loading cross-encoder reranker: {args.reranker_model}")
            reranker = CrossEncoder(args.reranker_model)
        
        elif reranking_mode == RerankingMode.LLM:
            print(f"Using OpenAI API for LLM reranking: {args.openai_url}")
            print(f"LLM model: {args.llm_model}")
            from openai import OpenAI
            llm_client = OpenAI(
                base_url=args.openai_url,
                api_key=os.environ.get("OPENAI_API_KEY", "token-abc123"),
            )
            llm_model = args.llm_model
        
        results = evaluate(
            benchmark=benchmark,
            collection=collection,
            bm25_retriever=bm25_retriever,
            embed_model=embed_model,
            question_key=args.question_key,
            k_values=k_values,
            alpha=args.alpha,
            reranking_mode=reranking_mode,
            reranker=reranker,
            llm_client=llm_client,
            llm_model=llm_model,
            rerank_top_k=args.rerank_top_k,
        )
        
        # Add metadata
        results["dataset"] = args.dataset
        results["alpha"] = args.alpha
        results["question_key"] = args.question_key
        results["reranking_mode"] = reranking_mode.name
        results["rerank_top_k"] = args.rerank_top_k
        
        # Save results
        if args.output_file:
            output_path = Path(args.output_file)
        else:
            output_path = INTERFACE_DIR / "output" / f"{args.dataset}_results.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
