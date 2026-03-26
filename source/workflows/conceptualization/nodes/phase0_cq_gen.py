"""
Phase 0: Query Clustering and CQ Generation

CORRECT DESIGN (following primitive_tbox):
1. Cluster Queries (not CQs) using K-means on query embeddings
2. For each cluster, retrieve relevant tables
3. Generate CQs per-query (in parallel using async)

Per-query CQ Generation:
- Each Query generates scq_per_query SCQs + vcq_per_query VCQs
- Uses async parallel LLM calls for efficiency
- Supports caching to avoid regenerating CQs
"""

import math
import asyncio
import hashlib
import json
import threading
from typing import Dict, Any, List, Callable, Optional, Literal
import numpy as np
from loguru import logger
from pathlib import Path
from pydantic import BaseModel, Field, create_model

from workflows.common.node_decorators import graph_node
from workflows.conceptualization.state import FederatedPrimitiveTBoxState


# ============== Global for one-time prompt logging (thread-safe) ==============

_CQ_GEN_PROMPT_LOGGED = False
_CQ_GEN_PROMPT_LOCK = threading.Lock()


# ========== Per-Query CQ Generation Template (Dynamic Schema) ==========

CQ_PER_QUERY_TEMPLATE = """Generate Competency Questions for ontology design based on this user query.

## User Query:
{query_text}

## Retrieved Tables (Schema + Sample Data):
{table_context}

## Instructions:
- SCQ (Scoping CQ): Define what column types/patterns the ontology should cover.
  Use actual column names and data patterns from the tables above.
- VCQ (Validating CQ): Define what features distinguish relevant tables.
  Reference specific data characteristics observed in the samples.
- Each CQ should be a concise question (10-50 words)
"""


# ========== Dynamic Schema Creation ==========

def create_cq_output_model(scq_count: int, vcq_count: int) -> type:
    """Create a dynamic Pydantic model with explicit SCQ and VCQ fields.
    
    Instead of using List[CQ], we create explicit fields like:
        SCQ_1: str
        SCQ_2: str
        VCQ_1: str
        ...
    
    This ensures LLM outputs ALL required CQs with no omissions.
    
    Args:
        scq_count: Number of Scoping CQs
        vcq_count: Number of Validating CQs
        
    Returns:
        Dynamically created Pydantic model class
        
    Example:
        >>> Model = create_cq_output_model(2, 1)
        >>> result = Model(SCQ_1="...", SCQ_2="...", VCQ_1="...")
    """
    fields = {}
    
    # Add SCQ fields
    for i in range(1, scq_count + 1):
        field_name = f"SCQ_{i}"
        fields[field_name] = (
            str,
            Field(
                ...,
                description=f"Scoping CQ #{i}: Define what column types/patterns the ontology should cover"
            )
        )
    
    # Add VCQ fields
    for i in range(1, vcq_count + 1):
        field_name = f"VCQ_{i}"
        fields[field_name] = (
            str,
            Field(
                ...,
                description=f"Validating CQ #{i}: Define what features distinguish relevant tables"
            )
        )
    
    return create_model('DynamicCQOutput', **fields)


def parse_cq_output(model_instance: BaseModel, scq_count: int, vcq_count: int) -> List[Dict[str, Any]]:
    """Parse dynamic CQ model output into list of CQ dicts.
    
    Args:
        model_instance: Instance of dynamically created model
        scq_count: Number of SCQs
        vcq_count: Number of VCQs
        
    Returns:
        List of CQ dicts with cq_id, cq_type, question
    """
    result = []
    model_dict = model_instance.model_dump()
    
    # Parse SCQs
    for i in range(1, scq_count + 1):
        field_name = f"SCQ_{i}"
        if field_name in model_dict and model_dict[field_name]:
            result.append({
                "cq_id": f"SCQ-{i}",
                "cq_type": "SCQ",
                "question": model_dict[field_name],
            })
    
    # Parse VCQs
    for i in range(1, vcq_count + 1):
        field_name = f"VCQ_{i}"
        if field_name in model_dict and model_dict[field_name]:
            result.append({
                "cq_id": f"VCQ-{i}",
                "cq_type": "VCQ",
                "question": model_dict[field_name],
            })
    
    return result


# ========== CQ Cache Utilities ==========

def _get_cq_cache_path(state: FederatedPrimitiveTBoxState) -> Path:
    """Get the CQ cache file path."""
    from store.store_singleton import get_store
    store = get_store()
    cache_dir = Path(store.db.uri) / "cache" / "cqs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Cache key based on dataset and generation params
    cache_key = f"{state.query_store_name}_scq{state.scq_per_query}_vcq{state.vcq_per_query}"
    return cache_dir / f"{cache_key}.json"


def _compute_query_hash(query: Dict) -> str:
    """Compute a hash for a query to use as cache key."""
    query_text = query.get("query_text") or query.get("question") or query.get("query") or ""
    return hashlib.md5(query_text.encode()).hexdigest()[:12]


def _load_cq_cache(cache_path: Path) -> Dict[str, List[Dict]]:
    """Load CQ cache from file."""
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load CQ cache: {e}")
    return {}


def _save_cq_cache(cache_path: Path, cache: Dict[str, List[Dict]]):
    """Save CQ cache to file."""
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save CQ cache: {e}")


# ========== Async Per-Query CQ Generation ==========

async def _generate_cqs_for_query_async(
    query: Dict,
    tables: List[Dict],
    scq_count: int,
    vcq_count: int,
    llm_purpose: str,
    llm_override_config: Optional[Dict] = None,
    query_idx: int = 0,
) -> Dict[str, Any]:
    """
    Generate CQs for a single query using async LLM call.
    
    Args:
        query: Query dict with query_text
        tables: List of tables for context
        scq_count: Number of SCQs to generate
        vcq_count: Number of VCQs to generate
        llm_purpose: LLM purpose for model selection
        llm_override_config: Optional LLM config override
        query_idx: Query index for logging
        
    Returns:
        Dict with query_hash and generated cqs
    """
    from llm.manager import get_llm_by_purpose
    from llm.async_client import invoke_llm_async
    
    query_text = query.get("query_text") or query.get("question") or query.get("query") or ""
    query_hash = _compute_query_hash(query)
    
    # Format table context with sample rows
    table_context_parts = []
    for table in tables[:3]:  # Limit to 3 tables (more detail each)
        title = table.get("document_title", "Unknown")
        columns = table.get("columns", [])
        # Handle both string list ["col1", "col2"] and dict list [{"name": "col1"}, ...]
        if columns and isinstance(columns[0], dict):
            col_names = [c.get("name", "") for c in columns[:10]]
        else:
            col_names = [str(c) for c in columns[:10]] if columns else []
        
        # Format sample rows (up to 3 rows)
        sample_rows = table.get("sample_rows", [])
        if isinstance(sample_rows, str):
            try:
                import json
                sample_rows = json.loads(sample_rows)
            except:
                sample_rows = []
        
        rows_text = ""
        if sample_rows and isinstance(sample_rows, list):
            rows_text = "\n  Sample rows:\n"
            for i, row in enumerate(sample_rows[:3]):
                if isinstance(row, list):
                    row_str = " | ".join(str(cell)[:30] for cell in row[:6])  # Limit cell width
                    rows_text += f"    [{i+1}] {row_str}\n"
        
        table_context_parts.append(
            f"### Table: {title}\n"
            f"  Columns: {', '.join(col_names)}"
            f"{rows_text}"
        )
    
    table_context = "\n".join(table_context_parts) if table_context_parts else "No table context available"
    
    # Create dynamic schema with explicit fields for each CQ
    # Schema constraint is passed to LLM via structured output (guided decoding)
    DynamicCQModel = create_cq_output_model(scq_count, vcq_count)
    
    # Build prompt (no field descriptions needed - schema handles that)
    prompt = CQ_PER_QUERY_TEMPLATE.format(
        query_text=query_text,
        table_context=table_context,
    )
    
    # One-time prompt logging for debug (thread-safe)
    global _CQ_GEN_PROMPT_LOGGED
    with _CQ_GEN_PROMPT_LOCK:
        if not _CQ_GEN_PROMPT_LOGGED:
            logger.debug("=" * 80)
            logger.debug("[CQ Generation] FIRST PROMPT (one-time log):")
            logger.debug("=" * 80)
            logger.debug(prompt)
            logger.debug("=" * 80)
            _CQ_GEN_PROMPT_LOGGED = True
    
    # Get LLM factory for EBNF-based invocation
    def llm_factory(temperature: float):
        return get_llm_by_purpose(
            purpose=llm_purpose, 
            override_config=llm_override_config,
            temperature_override=temperature
        )
    
    try:
        # Use EBNF-based async invocation (default use_ebnf=True)
        from llm.async_client import invoke_structured_llm_with_retry_async
        result = await invoke_structured_llm_with_retry_async(
            llm_factory=llm_factory,
            output_schema=DynamicCQModel,
            prompt=prompt,
            max_retries=2,
            timeout=120.0,
        )
        
        # Parse dynamic model output into CQ list
        cqs = parse_cq_output(result, scq_count, vcq_count)
        
        # Add source metadata
        for cq in cqs:
            cq["source_query"] = query_text
            cq["source_query_hash"] = query_hash
        
        logger.debug(f"[Query {query_idx}] Generated {len(cqs)} CQs")
        return {
            "query_hash": query_hash,
            "cqs": cqs,
            "success": True,
        }
        
    except Exception as e:
        logger.warning(f"[Query {query_idx}] CQ generation failed: {e}")
        return {
            "query_hash": query_hash,
            "cqs": [],
            "success": False,
            "error": str(e),
        }


async def _generate_cqs_for_queries_batch_async(
    queries: List[Dict],
    tables_per_query: Dict[str, List[Dict]],
    scq_count: int,
    vcq_count: int,
    llm_purpose: str,
    llm_override_config: Optional[Dict] = None,
    max_concurrent: int = 16,
    cq_cache: Optional[Dict[str, List[Dict]]] = None,
) -> Dict[str, Any]:
    """
    Generate CQs for multiple queries in parallel using async.
    
    Args:
        queries: List of queries
        tables_per_query: Dict mapping query_hash to relevant tables
        scq_count: SCQs per query
        vcq_count: VCQs per query
        llm_purpose: LLM purpose
        llm_override_config: Optional LLM config
        max_concurrent: Max concurrent LLM calls
        cq_cache: Existing CQ cache for reuse
        
    Returns:
        Dict with all_cqs, cache_hits, new_generations
    """
    cq_cache = cq_cache or {}
    semaphore = asyncio.Semaphore(max_concurrent)
    
    all_cqs = []
    cache_hits = 0
    new_generations = 0
    errors = []
    
    async def process_query(query: Dict, idx: int):
        nonlocal cache_hits, new_generations
        
        query_hash = _compute_query_hash(query)
        
        # Check cache first
        if query_hash in cq_cache:
            cached_cqs = cq_cache[query_hash]
            if cached_cqs:
                cache_hits += 1
                return cached_cqs
        
        # Generate new CQs
        async with semaphore:
            tables = tables_per_query.get(query_hash, [])
            result = await _generate_cqs_for_query_async(
                query=query,
                tables=tables,
                scq_count=scq_count,
                vcq_count=vcq_count,
                llm_purpose=llm_purpose,
                llm_override_config=llm_override_config,
                query_idx=idx,
            )
            
            if result["success"]:
                new_generations += 1
                # Update cache
                cq_cache[query_hash] = result["cqs"]
                return result["cqs"]
            else:
                errors.append(result.get("error", "Unknown error"))
                return []
    
    # Process all queries in parallel
    logger.info(f"Generating CQs for {len(queries)} queries (max_concurrent={max_concurrent})...")
    tasks = [process_query(q, i) for i, q in enumerate(queries)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Collect results
    for result in results:
        if isinstance(result, Exception):
            errors.append(str(result))
        elif isinstance(result, list):
            all_cqs.extend(result)
    
    logger.info(f"✓ CQ generation complete: {cache_hits} cached, {new_generations} new, {len(errors)} errors")
    
    return {
        "all_cqs": all_cqs,
        "cache_hits": cache_hits,
        "new_generations": new_generations,
        "errors": errors,
        "updated_cache": cq_cache,
    }


def _print_agent_tree_structure(tree_config, n_clusters: int) -> None:
    """
    Print a visual representation of the Agent Tree structure.
    
    Three node types:
    - Leaf: Local proposal agents (one per cluster)
    - Synthesizer: Intermediate aggregation agents
    - Root: Global synthesis agent
    """
    B = tree_config.branching_factor
    depth = tree_config.depth
    levels = tree_config.levels
    
    logger.info("")
    logger.info("  ┌" + "─" * 50 + "┐")
    logger.info("  │          Agent Tree Structure                   │")
    logger.info("  ├" + "─" * 50 + "┤")
    
    # Print each level
    for level_info in levels:
        level = level_info.level
        n_agents = level_info.n_agents
        node_type = level_info.node_type
        
        # Determine node type label
        if node_type == "leaf":
            type_label = "Leaf (Local Propose)"
            node_names = f"group_0..{n_agents-1}"
        elif node_type == "root":
            type_label = "Root (Global Synthesis)"
            node_names = "global_synthesis"
        else:
            type_label = f"Synthesizer L{level}"
            node_names = f"synth_L{level}_0..{n_agents-1}"
        
        logger.info(f"  │  Level {level}: {n_agents:>2} x {type_label:<25} │")
        logger.info(f"  │           └─ [{node_names}]" + " " * (30 - len(node_names)) + "│")
    
    logger.info("  ├" + "─" * 50 + "┤")
    logger.info(f"  │  Branching Factor: {B}  |  Depth: {depth}  |  Total: {tree_config.total_agents:<3} │")
    logger.info("  └" + "─" * 50 + "┘")
    logger.info("")


@graph_node(node_type="preprocessing", log_level="INFO")
def cluster_queries_node(state: FederatedPrimitiveTBoxState) -> Dict[str, Any]:
    """
    Phase 0a: Cluster queries for parallel CQ generation.
    
    Reuses clustering logic from primitive_tbox.cluster_and_sample.
    
    Output:
        - n_clusters: Number of computed clusters
        - cluster_assignments: {group_id: {query_indices, centroid, query_table_pairs}}
    """
    logger.info("=" * 60)
    logger.info("Phase 0a: Cluster Queries for Federated CQ Generation")
    logger.info("=" * 60)
    
    # Skip if cluster_assignments already exist (cache reuse)
    if state.cluster_assignments:
        n_existing = len(state.cluster_assignments)
        logger.info(f"  Cluster assignments exist ({n_existing} groups), skipping clustering")
        return {"n_clusters": n_existing}
    
    # Import utilities for table retrieval
    from workflows.conceptualization.utils.table_retrieval import (
        _retrieve_tables_for_cluster,
    )
    from store.store_singleton import get_store
    
    store = get_store()
    
    queries = state.queries
    if not queries:
        logger.error("  No queries loaded")
        return {"n_clusters": 0, "cluster_assignments": {}}
    
    # ========== Auto-compute tree structure using tree_design ==========
    # See docs2/AGENT_TREE_DESIGN.md for derivation
    from workflows.retrieval.samplers.tree_design import compute_tree_structure
    
    K = state.agent_cq_capacity  # Max CQs per leaf agent
    B = state.global_agent_span  # Max branching factor
    cq_per_query = state.scq_per_query + state.vcq_per_query
    user_n_clusters = state.n_clusters  # 0 = auto-compute
    min_ratio = getattr(state, 'min_cluster_ratio', 0.5)
    
    # Compute optimal tree structure
    tree_config = compute_tree_structure(
        num_queries=len(queries),
        max_cq_capacity=K,
        max_subagent_capacity=B,
        num_cq_per_query=cq_per_query,
        min_cluster_ratio=min_ratio,
    )
    
    # Use user-specified if provided (> 0), otherwise use auto-computed
    if user_n_clusters > 0:
        n_clusters = max(user_n_clusters, tree_config.n_clusters)
    else:
        n_clusters = tree_config.n_clusters
    
    queries_per_group = max(1, len(queries) // n_clusters)
    
    logger.info(f"  Total queries: {len(queries)}")
    logger.info(f"  CQs per query: {cq_per_query} ({state.scq_per_query} SCQ + {state.vcq_per_query} VCQ)")
    logger.info(f"  Agent CQ Capacity (K): {K}")
    logger.info(f"  Max Branching Factor (B): {B}")
    logger.info(f"  Tree Config: {tree_config.n_clusters} clusters, depth={tree_config.depth}, "
                f"total_agents={tree_config.total_agents}")
    logger.info(f"  User-specified clusters: {user_n_clusters} (0 = auto)")
    logger.info(f"  Actual clusters: {n_clusters}")
    logger.info(f"  Queries per cluster: ~{queries_per_group} "
                f"(range: [{tree_config.queries_per_cluster[0]}, {tree_config.queries_per_cluster[1]}])")
    
    # Print agent tree structure visualization
    _print_agent_tree_structure(tree_config, n_clusters)
    
    # Step 1: Extract query texts for unified similarity computation
    logger.info("Step 1: Extracting query texts for similarity computation...")
    query_texts = []
    for query in queries:
        text = query.get("query_text") or query.get("question") or query.get("query") or ""
        query_texts.append(text)
    logger.info(f"  Extracted {len(query_texts)} query texts")
    
    # Step 2: Compute similarity matrix and cluster using Spectral Clustering
    # This works with all rag_types: vector, bm25, hybrid
    from workflows.retrieval.unified_similarity import (
        compute_similarity_matrix,
        spectral_cluster_balanced,
    )
    
    sim_mode = state.rag_type if state.rag_type in ("vector", "bm25", "hybrid") else "vector"
    
    if n_clusters <= 1:
        logger.info("  Single group mode - skipping clustering")
        labels = [0] * len(queries)
    else:
        logger.info(f"Step 2: Spectral clustering into {n_clusters} groups (mode={sim_mode})...")
        
        # Compute similarity matrix based on rag_type
        similarity_matrix = compute_similarity_matrix(
            texts=query_texts,
            mode=sim_mode,
            cache_key=f"{state.dataset_name}_{sim_mode}_similarity_clustering",
        )
        
        # Run balanced spectral clustering
        labels = spectral_cluster_balanced(
            similarity_matrix=similarity_matrix,
            n_clusters=n_clusters,
            seed=state.sampling_seed,
            min_cluster_ratio=getattr(state, 'min_cluster_ratio', 0.5),
        )
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
    
    # Group indices by cluster
    cluster_indices_map = {}
    for idx, label in enumerate(labels):
        label_str = f"group_{label}"
        if label_str not in cluster_indices_map:
            cluster_indices_map[label_str] = []
        cluster_indices_map[label_str].append(idx)
    
    logger.info(f"  Cluster sizes: {[len(v) for v in cluster_indices_map.values()]}")
    
    # Step 3: Assign all queries in each cluster (spectral clustering ensures balanced sizes)
    logger.info("Step 3: Using ALL queries in each cluster (spectral clustering)")
    
    cluster_assignments = {}
    
    for group_id, indices in cluster_indices_map.items():
        # Note: Spectral clustering doesn't produce centroids like K-means
        # centroid is kept as None for interface compatibility
        cluster_assignments[group_id] = {
            "query_indices": indices,  # Use all queries in cluster
            "centroid": None,  # Not available in spectral clustering
            "cluster_size": len(indices),
            "sampled_size": len(indices),
        }
        
        logger.info(f"  {group_id}: {len(indices)} queries")
    
    # Step 4: Retrieve tables for each cluster using unified search
    logger.info(f"Step 4: Retrieving tables for each cluster (rag_type={state.rag_type})...")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def _retrieve_for_group(group_id: str, assignment: Dict) -> tuple:
        """Retrieve tables for a specific group."""
        pairs = _retrieve_tables_for_cluster(
            db=store.db,
            query_indices=assignment["query_indices"],
            queries=queries,
            tables=state.tables,
            table_store_name=state.table_store_name,
            table_embedding_field=state.table_embedding_field,
            retrieval_top_k=state.retrieval_top_k,
            dataset_name=state.dataset_name,  # For unified search
            rag_type=state.rag_type,  # Support BM25/Vector/Hybrid
        )
        return group_id, pairs
    
    max_workers = min(len(cluster_assignments), 10)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_retrieve_for_group, gid, assign): gid
            for gid, assign in cluster_assignments.items()
        }
        
        for future in as_completed(futures):
            group_id, query_table_pairs = future.result()
            cluster_assignments[group_id]["query_table_pairs"] = query_table_pairs
            cluster_assignments[group_id]["retrieval_count"] = len(query_table_pairs)
            logger.info(f"  {group_id}: Retrieved {len(query_table_pairs)} query-table pairs")
    
    logger.info("=" * 60)
    logger.info(f"✓ Query clustering complete. {len(cluster_assignments)} groups ready")
    
    # Convert tree_config to dict for state storage
    tree_config_dict = {
        "n_clusters": tree_config.n_clusters,
        "branching_factor": tree_config.branching_factor,
        "depth": tree_config.depth,
        "total_agents": tree_config.total_agents,
        "queries_per_cluster": tree_config.queries_per_cluster,
        "proposal_capacity": state.proposal_capacity,  # For dynamic max_proposals calculation
        "levels": [
            {
                "level": lvl.level,
                "n_agents": lvl.n_agents,
                "node_type": lvl.node_type,
            }
            for lvl in tree_config.levels
        ],
    }
    
    return {
        "n_clusters": n_clusters,
        "cluster_assignments": cluster_assignments,
        "cluster_labels": labels,
        "tree_config": tree_config_dict,
    }


def create_branch_cq_generator(group_id: str) -> Callable:
    """
    Create a CQ generation node for a specific group.
    
    This is Phase 0b: Generate CQs for one cluster's queries (per-query, async parallel).
    
    Args:
        group_id: The group identifier (e.g., 'group_0')
        
    Returns:
        A node function that generates CQs for this group using async parallel calls
    """
    
    @graph_node(node_type="generation", log_level="INFO")
    def branch_generate_cqs(state: FederatedPrimitiveTBoxState) -> Dict[str, Any]:
        """
        Generate Competency Questions for each query in this group (per-query, async parallel).
        
        Uses:
        - scq_per_query: Number of SCQs per query
        - vcq_per_query: Number of VCQs per query
        - Async parallel LLM calls
        - CQ caching for reuse
        """
        logger.info(f"[{group_id}] Phase 0b: Generating per-query CQs...")
        
        # Check for cached CQs in state
        if state.branch_cqs and group_id in state.branch_cqs:
            existing = state.branch_cqs[group_id]
            if existing:
                logger.info(f"[{group_id}] CQs already in state ({len(existing)}), skipping")
                return {}
        
        # Get this group's assignment
        if group_id not in state.cluster_assignments:
            logger.error(f"[{group_id}] Not found in cluster_assignments")
            return {"branch_cqs": {group_id: []}}
        
        assignment = state.cluster_assignments[group_id]
        query_indices = assignment.get("query_indices", [])
        query_table_pairs = assignment.get("query_table_pairs", [])
        
        if not query_indices:
            logger.warning(f"[{group_id}] No queries assigned, skipping CQ generation")
            return {"branch_cqs": {group_id: []}}
        
        # Get queries for this group
        group_queries = [state.queries[idx] for idx in query_indices if idx < len(state.queries)]
        logger.info(f"[{group_id}] Processing {len(group_queries)} queries (scq={state.scq_per_query}, vcq={state.vcq_per_query} per query)")
        
        # Build tables_per_query mapping
        # Data structure from _retrieve_tables_for_cluster:
        # {
        #     "query_id": "...",
        #     "query_text": "...",
        #     "retrieved_tables": [{"document_title": "...", "columns": [...], "sample_rows": "..."}],
        # }
        tables_per_query = {}
        for pair in query_table_pairs:
            query_text = pair.get("query_text", "")  # Correct field name
            # Find the query to get its hash
            for q in group_queries:
                q_text = q.get("query_text") or q.get("question") or q.get("query") or ""
                if q_text == query_text:
                    query_hash = _compute_query_hash(q)
                    if query_hash not in tables_per_query:
                        tables_per_query[query_hash] = []
                    # Extract tables from retrieved_tables list
                    retrieved_tables = pair.get("retrieved_tables", [])
                    for table in retrieved_tables:
                        tables_per_query[query_hash].append({
                            "document_title": table.get("document_title", ""),
                            "columns": table.get("columns", []),
                            "sample_rows": table.get("sample_rows", ""),
                        })
                    break
        
        # Load CQ cache
        cache_path = _get_cq_cache_path(state)
        cq_cache = _load_cq_cache(cache_path)
        logger.info(f"[{group_id}] Loaded CQ cache with {len(cq_cache)} entries")
        
        # Get LLM config
        llm_purpose = getattr(state, 'llm_purpose', 'default')
        llm_override = getattr(state, 'llm_override_config', None)
        
        # Get max concurrent from state (default: 16)
        cq_max_concurrent = getattr(state, 'cq_max_concurrent', 16)
        
        # Run async CQ generation
        import asyncio
        
        # Capture caller context before spawning threads/event loops
        # (ThreadPoolExecutor.submit does NOT propagate ContextVars)
        from llm.statistics import get_current_caller, set_current_caller
        _outer_caller = get_current_caller()
        
        async def _async_generate():
            # Restore caller context lost during ThreadPoolExecutor.submit
            set_current_caller(_outer_caller)
            return await _generate_cqs_for_queries_batch_async(
                queries=group_queries,
                tables_per_query=tables_per_query,
                scq_count=state.scq_per_query,
                vcq_count=state.vcq_per_query,
                llm_purpose=llm_purpose,
                llm_override_config=llm_override,
                max_concurrent=cq_max_concurrent,
                cq_cache=cq_cache,
            )
        
        # Run async in event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _async_generate())
                    result = future.result(timeout=600)
            else:
                result = loop.run_until_complete(_async_generate())
        except RuntimeError:
            # No event loop, create new one
            result = asyncio.run(_async_generate())
        
        all_cqs = result["all_cqs"]
        cache_hits = result["cache_hits"]
        new_gens = result["new_generations"]
        
        logger.info(f"[{group_id}] Generated {len(all_cqs)} CQs ({cache_hits} cached, {new_gens} new)")
        
        # Return updated cache for unified saving in collect_backbone_cqs_node
        # (Don't save here to avoid race condition from parallel writes)
        updated_cache = result.get("updated_cache", {})
        
        # Compute statistics
        scq_count = sum(1 for cq in all_cqs if cq.get("cq_type") == "SCQ")
        vcq_count = sum(1 for cq in all_cqs if cq.get("cq_type") == "VCQ")
        
        logger.info(f"[{group_id}]   SCQ: {scq_count}, VCQ: {vcq_count}")
        
        return {
            "branch_cqs": {group_id: all_cqs},
            "branch_cq_caches": {group_id: updated_cache},  # Collected and saved in collect_backbone_cqs_node
        }
    
    # Set function name for debugging
    branch_generate_cqs.__name__ = f"generate_cqs_{group_id}"
    return branch_generate_cqs


@graph_node(node_type="processing", log_level="INFO")
def collect_backbone_cqs_node(state: FederatedPrimitiveTBoxState) -> Dict[str, Any]:
    """
    Phase 0c: Collect backbone CQs from all branches for Phase 1 initialization.
    
    Uses DPP (Determinantal Point Process) sampling at Query level to select
    diverse representative queries, then collects their CQs as backbone.
    
    Strategy:
    1. DPP sample queries for maximum diversity (using query embeddings)
    2. Collect CQs from the sampled queries via cache
    3. Use these diverse CQs as backbone for Global Agent
    """
    logger.info("Phase 0c: Collecting backbone CQs from all branches")
    
    branch_cqs = state.branch_cqs or {}
    if not branch_cqs:
        logger.warning("  No branch CQs found")
        return {"backbone_cqs": [], "competency_questions": []}
    
    # Collect all CQs
    all_cqs = []
    for group_id in sorted(branch_cqs.keys()):
        cqs = branch_cqs[group_id]
        all_cqs.extend(cqs)
        logger.info(f"  {group_id}: {len(cqs)} CQs")
    
    logger.info(f"  Total CQs collected: {len(all_cqs)}")
    
    # ========== Merge and save CQ caches from all branches ==========
    # This unified save avoids race condition from parallel writes
    branch_caches = getattr(state, 'branch_cq_caches', {}) or {}
    if branch_caches:
        # Load existing cache (if any)
        cache_path = _get_cq_cache_path(state)
        merged_cache = _load_cq_cache(cache_path)
        original_size = len(merged_cache)
        
        # Merge all branch caches
        for group_id, group_cache in branch_caches.items():
            if group_cache:
                for query_hash, cqs in group_cache.items():
                    if query_hash not in merged_cache:
                        merged_cache[query_hash] = cqs
                    # If already exists, keep the existing (first write wins)
        
        # Save merged cache
        _save_cq_cache(cache_path, merged_cache)
        logger.info(f"  Merged CQ cache: {original_size} → {len(merged_cache)} entries (from {len(branch_caches)} branches)")
    else:
        merged_cache = {}
    
    # ========== Random Sampling at Query Level ==========
    # Calculate how many queries we need to sample
    cqs_per_query = state.scq_per_query + state.vcq_per_query
    target_query_count = max(1, state.agent_cq_capacity // cqs_per_query)
    
    logger.info(f"  Random sampling: target {target_query_count} queries (capacity={state.agent_cq_capacity}, cqs_per_query={cqs_per_query})")
    
    # Get all queries
    queries = state.queries or []
    if not queries:
        logger.warning("  No queries found for backbone sampling")
        # Fallback to simple selection
        backbone_cqs = all_cqs[:state.agent_cq_capacity]
        return {
            "backbone_cqs": backbone_cqs,
            "competency_questions": all_cqs,
            "cq_statistics": {"total_cqs": len(all_cqs), "backbone_cqs": len(backbone_cqs), "groups": len(branch_cqs)},
            "branch_cq_caches": {},
        }
    
    # Conditional DPP or Random sampling based on rag_type
    # Use DPP when rag_type is 'vector' (requires embeddings for similarity computation)
    # Use random sampling for 'bm25' or other modes
    use_dpp = state.rag_type == "vector"
    
    if use_dpp:
        logger.info(f"  Using DPP sampling (rag_type={state.rag_type})")
        # Use unified similarity for DPP sampling
        from workflows.retrieval.unified_similarity import (
            compute_similarity_matrix,
            dpp_sample_from_similarity,
        )
        
        try:
            # Extract query texts for similarity computation
            query_texts = []
            for query in queries:
                text = query.get("query_text") or query.get("question") or query.get("query") or ""
                query_texts.append(text)
            
            # Compute similarity matrix based on rag_type
            sim_mode = state.rag_type if state.rag_type in ("vector", "bm25", "hybrid") else "vector"
            logger.info(f"  Computing {sim_mode} similarity matrix for DPP...")
            
            similarity_matrix = compute_similarity_matrix(
                texts=query_texts,
                mode=sim_mode,
                cache_key=f"{state.dataset_name}_{sim_mode}_similarity",
            )
            
            # Global DPP sampling on all queries (no stratification)
            selected_indices = dpp_sample_from_similarity(
                similarity_matrix=similarity_matrix,
                k=min(target_query_count, len(queries)),
                seed=state.sampling_seed,
            )
            
            logger.info(f"  DPP selected {len(selected_indices)} diverse queries from {len(queries)}")
            
        except Exception as e:
            logger.warning(f"  DPP sampling failed: {e}, falling back to random sampling")
            # Fallback to random sampling
            np.random.seed(state.sampling_seed)
            selected_indices = np.random.choice(
                len(queries),
                size=min(target_query_count, len(queries)),
                replace=False
            ).tolist()
    else:
        # Random sampling for non-vector modes
        logger.info(f"  Using random sampling (rag_type={state.rag_type})")
        np.random.seed(state.sampling_seed)
        selected_indices = np.random.choice(
            len(queries),
            size=min(target_query_count, len(queries)),
            replace=False
        ).tolist()
        
        logger.info(f"  Random selected {len(selected_indices)} queries from {len(queries)}")
    
    # ========== Collect CQs from selected queries ==========
    # Build query hash -> CQs mapping from merged cache or branch_cqs
    backbone_cqs = []
    selected_query_hashes = set()
    
    # Get hashes of selected queries
    for idx in selected_indices:
        if idx < len(queries):
            query = queries[idx]
            query_hash = _compute_query_hash(query)
            selected_query_hashes.add(query_hash)
    
    logger.info(f"  Selected {len(selected_query_hashes)} unique query hashes")
    
    # Load cache for lookup
    if not merged_cache:
        cache_path = _get_cq_cache_path(state)
        merged_cache = _load_cq_cache(cache_path)
    
    # Collect CQs from selected queries
    for query_hash in selected_query_hashes:
        if query_hash in merged_cache:
            cqs = merged_cache[query_hash]
            backbone_cqs.extend(cqs)
    
    # If still under capacity, add more from remaining queries
    if len(backbone_cqs) < state.agent_cq_capacity:
        remaining_needed = state.agent_cq_capacity - len(backbone_cqs)
        for query_hash, cqs in merged_cache.items():
            if query_hash not in selected_query_hashes:
                backbone_cqs.extend(cqs[:remaining_needed])
                remaining_needed -= len(cqs)
                if remaining_needed <= 0:
                    break
    
    # Cap to agent capacity
    backbone_cqs = backbone_cqs[:state.agent_cq_capacity]
    
    logger.info(f"  Backbone CQs: {len(backbone_cqs)} (from {len(selected_query_hashes)} DPP-selected queries)")
    
    return {
        "backbone_cqs": backbone_cqs,
        "competency_questions": all_cqs,  # Store all CQs for reference
        "cq_statistics": {
            "total_cqs": len(all_cqs),
            "backbone_cqs": len(backbone_cqs),
            "groups": len(branch_cqs),
            "dpp_selected_queries": len(selected_query_hashes),
            "similarity_mode": state.rag_type,
        },
        "branch_cq_caches": {},  # Clear caches after saving to free memory
    }


__all__ = [
    "cluster_queries_node",
    "create_branch_cq_generator",
    "collect_backbone_cqs_node",
]
