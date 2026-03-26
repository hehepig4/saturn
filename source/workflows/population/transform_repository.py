"""
Transform Repository

Stores and retrieves TransformContracts for code reuse.

Key Design:
- TransformContract is uniquely identified by (primitive_class, data_property, pattern)
- Multiple contracts can exist for same (class, property) with different patterns
- Matching: filter by (class, property), then select best by success_rate using Successive Halving
- Successive Halving: progressively evaluate with more samples, eliminate bottom 50% each round
- LRU Eviction: when contracts per (class, property) exceed limit, evict least-used contracts
- Quality Gate: only store contracts that are BETTER than existing best

Simplified Storage Model:
- DB and memory cache are equal: both are subject to LRU eviction
- flush() completely overwrites DB with current cache state
- No separate "pending" tracking - cache is the single source of truth
"""

import json
import threading
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

import pandas as pd

from workflows.population.contract import TransformContract
from workflows.population.safe_regex import safe_pandas_match
from config.hyperparameters import (
    SUCCESSIVE_HALVING_MIN_SUCCESS_RATE as MIN_SUCCESS_RATE,
    SUCCESSIVE_HALVING_MIN_SAMPLES_PER_ARM,
    SUCCESSIVE_HALVING_ETA,
    SUCCESSIVE_HALVING_BUDGET_CAP,
)

# Contract storage limits
MAX_CONTRACTS_PER_KEY = 20  # Maximum contracts per (class, property) pair


class TransformRepository:
    """
    Repository for storing and retrieving TransformContracts.
    
    Storage structure:
    - Each record has: contract_id, primitive_class, data_property, pattern, contract_json
    - Query by (primitive_class, data_property) returns all matching contracts
    - Select best by Successive Halving based on success_rate
    
    Thread Safety:
    - Uses RLock for thread-safe operations
    - Cache is the single source of truth, DB is a persistence layer
    - flush() completely overwrites DB with cache contents
    """
    
    TABLE_NAME = "transform_contracts"
    
    def __init__(self, dataset_name: str = "default", budget_multiplier: float = 1.0):
        self.dataset_name = dataset_name
        self._table_name = f"{dataset_name}_{self.TABLE_NAME}"
        self._table = None
        self.budget_multiplier = budget_multiplier
        
        # Thread safety
        self._lock = threading.RLock()
        
        # In-memory cache for contracts (single source of truth)
        self._contracts_cache: Dict[str, List[TransformContract]] = {}  # key: "class::property"
        self._pattern_keys: set = set()  # O(1) lookup for (class, property, pattern) tuples
        self._cache_loaded = False
        self._cache_dirty = False  # Track if cache has changes not yet flushed
    
    def _load_cache(self) -> None:
        """Load all contracts from DB into memory cache."""
        if self._cache_loaded:
            return
        
        with self._lock:
            if self._cache_loaded:
                return
            
            table = self._get_table()
            if table is None:
                self._cache_loaded = True
                return
            
            try:
                all_rows = table.to_pandas().to_dict('records')
                for row in all_rows:
                    try:
                        contract_data = json.loads(row["contract_json"])
                        contract = TransformContract.from_dict(contract_data)
                        key = f"{contract.primitive_class}::{contract.data_property}"
                        pattern_key = (contract.primitive_class, contract.data_property, contract.pattern)
                        
                        if key not in self._contracts_cache:
                            self._contracts_cache[key] = []
                        self._contracts_cache[key].append(contract)
                        self._pattern_keys.add(pattern_key)
                    except Exception:
                        continue
                
                # Apply LRU limit on load: keep only top MAX_CONTRACTS_PER_KEY by hit_count
                for key, contracts in self._contracts_cache.items():
                    if len(contracts) > MAX_CONTRACTS_PER_KEY:
                        # Sort by hit_count descending, keep top N
                        contracts.sort(key=lambda c: c.hit_count, reverse=True)
                        evicted = contracts[MAX_CONTRACTS_PER_KEY:]
                        self._contracts_cache[key] = contracts[:MAX_CONTRACTS_PER_KEY]
                        # Remove evicted from pattern_keys
                        for c in evicted:
                            self._pattern_keys.discard((c.primitive_class, c.data_property, c.pattern))
                        logger.info(f"LRU on load: evicted {len(evicted)} contracts for key={key}")
                        self._cache_dirty = True  # Need to flush to persist eviction
                
                self._cache_loaded = True
                logger.debug(f"Loaded {sum(len(v) for v in self._contracts_cache.values())} contracts into cache")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self._cache_loaded = True
    
    def flush(self) -> int:
        """
        Flush cache to DB using complete overwrite.
        
        This ensures DB always mirrors cache state, including LRU evictions.
        
        Returns:
            Number of contracts written
        """
        with self._lock:
            if not self._cache_dirty and self._cache_loaded:
                # No changes to flush
                return 0
            
            # Collect all contracts from cache
            all_contracts = []
            for contracts in self._contracts_cache.values():
                all_contracts.extend(contracts)
            
            if not all_contracts and not self._cache_loaded:
                return 0
        
        # Perform DB write outside the lock
        try:
            flushed = self._overwrite_db(all_contracts)
            with self._lock:
                self._cache_dirty = False
            logger.debug(f"Flushed {flushed} contracts to DB (complete overwrite)")
            return flushed
        except Exception as e:
            logger.error(f"Failed to flush contracts: {e}")
            return 0
    
    def _overwrite_db(self, contracts: List[TransformContract]) -> int:
        """
        Completely overwrite DB with given contracts.
        
        Args:
            contracts: All contracts to write
            
        Returns:
            Number of contracts written
        """
        from store.store_singleton import get_store
        store = get_store()
        
        # Prepare all records
        records = []
        for contract in contracts:
            records.append({
                "contract_id": contract.contract_id,
                "primitive_class": contract.primitive_class,
                "data_property": contract.data_property,
                "pattern": contract.pattern,
                "contract_json": json.dumps(contract.to_dict()),
            })
        
        # Drop existing table and recreate (atomic overwrite)
        try:
            store.db.drop_table(self._table_name)
            self._table = None
        except Exception:
            pass  # Table may not exist
        
        if not records:
            return 0
        
        import pyarrow as pa
        schema = pa.schema([
            pa.field("contract_id", pa.string()),
            pa.field("primitive_class", pa.string()),
            pa.field("data_property", pa.string()),
            pa.field("pattern", pa.string()),
            pa.field("contract_json", pa.string()),
        ])
        
        # Handle race condition: another process may have created the table
        try:
            self._table = store.db.create_table(self._table_name, records, schema=schema)
        except ValueError as e:
            if "already exists" in str(e):
                # Another process created the table, drop and retry
                try:
                    store.db.drop_table(self._table_name)
                except Exception:
                    pass
                self._table = store.db.create_table(self._table_name, records, schema=schema)
            else:
                raise
        
        logger.info(f"Overwrote DB with {len(records)} contracts")
        return len(records)
    
    def clear_all(self) -> bool:
        """
        Clear all contracts from the repository.
        
        Returns:
            True if successful, False otherwise
        """
        from store.store_singleton import get_store
        store = get_store()
        
        with self._lock:
            self._contracts_cache.clear()
            self._pattern_keys.clear()
            self._cache_dirty = True
        
        try:
            store.db.drop_table(self._table_name)
            self._table = None
            logger.debug(f"Cleared transform repository: {self._table_name}")
            return True
        except Exception as e:
            logger.debug(f"No table to clear or error: {e}")
            return False
    
    def _get_table(self):
        """Get or create the LanceDB table."""
        if self._table is not None:
            return self._table
        
        from store.store_singleton import get_store
        store = get_store()
        
        try:
            self._table = store.db.open_table(self._table_name)
            logger.debug(f"Opened transform repository: {self._table_name}")
        except Exception:
            self._table = None
            logger.debug(f"Transform repository not found: {self._table_name}")
        
        return self._table

    def find_best_contract(
        self,
        primitive_class: str,
        data_property: str,
        column_values: List[str],
        min_success_rate: float = MIN_SUCCESS_RATE,
        budget: Optional[int] = None,
        budget_cap: int = SUCCESSIVE_HALVING_BUDGET_CAP,
        eta: int = SUCCESSIVE_HALVING_ETA,
        sh_max_workers: int = 8,
    ) -> Tuple[Optional[TransformContract], List[str]]:
        """
        Find the best matching TransformContract using Successive Halving.
        
        Algorithm (Successive Halving - Jamieson & Talwalkar, 2015):
        Given n candidates and budget B, run s_max = ⌊log_η(n)⌋ rounds.
        In round s:
          - Each candidate gets r_s = B / (n * s_max) * η^s samples
          - Eliminate bottom 1 - 1/η fraction (keep top 1/η)
        
        Default: η=2 (halving), budget = n_values * budget_multiplier (fixed)
        
        Args:
            primitive_class: The primitive class name (exact match, no parent fallback)
            data_property: The target DataProperty name
            column_values: Values from the column
            min_success_rate: Minimum success_rate to consider valid (default 0.85)
            budget: Total sample budget (default: n_values * budget_multiplier)
            eta: Elimination ratio (default: 2, meaning halving)
            sh_max_workers: Max parallel workers for Successive Halving evaluation (default: 8)
            
        Returns:
            Tuple of (Best matching TransformContract or None, List of unmatched values)
        """
        import math
        
        self._load_cache()
        
        # Get candidates from cache (exact class match only)
        key = f"{primitive_class}::{data_property}"
        with self._lock:
            candidates = list(self._contracts_cache.get(key, []))
        
        n = len(candidates)
        if n == 0:
            return None, []
        
        n_values = len(column_values)
        
        # Single candidate: just evaluate it
        if n == 1:
            contract = candidates[0]
            score = contract.success_rate(column_values)
            if score >= min_success_rate:
                self.increment_hit_count(contract)
                logger.debug(f"Single contract {contract.contract_id} (success_rate={score:.1%}, hit_count={contract.hit_count})")
                return contract, []
            return None, self._collect_unmatched(contract, column_values)
        
        # Successive Halving parameters
        s_max = max(1, int(math.floor(math.log(n) / math.log(eta))))
        
        if budget is None:
            budget = int(self.budget_multiplier * n_values)
        budget = min(budget, budget_cap)
        
        r_0 = max(SUCCESSIVE_HALVING_MIN_SAMPLES_PER_ARM, budget // (n * s_max))
        
        logger.debug(
            f"Successive Halving: n={n} candidates, budget={budget} (k={self.budget_multiplier}), "
            f"η={eta}, s_max={s_max}, r_0={r_0}"
        )
        
        scored_candidates: List[Tuple[TransformContract, float]] = [(c, 0.0) for c in candidates]
        
        for s in range(s_max + 1):
            n_s = len(scored_candidates)
            if n_s <= 1:
                break
            
            r_s = min(n_values, r_0 * (eta ** s))
            max_samples = int(r_s) if r_s < n_values else None
            
            new_scored = []
            for contract, _ in scored_candidates:
                try:
                    score = contract.success_rate(column_values, max_samples=max_samples)
                    new_scored.append((contract, score))
                except Exception as e:
                    logger.warning(f"Error evaluating contract: {e}")
                    new_scored.append((contract, 0.0))
            
            new_scored.sort(key=lambda x: x[1], reverse=True)
            keep_count = max(1, n_s // eta)
            
            top_scores = [f"{c.pattern[:20]}:{sc:.1%}" for c, sc in new_scored[:3]]
            logger.debug(
                f"  Round {s}: {n_s} candidates × {max_samples or 'all'} samples → "
                f"keep {keep_count} | top: {', '.join(top_scores)}"
            )
            
            scored_candidates = new_scored[:keep_count]
        
        if not scored_candidates:
            return None, []
        
        best_contract, _ = scored_candidates[0]
        best_score = best_contract.success_rate(column_values)
        
        if best_score >= min_success_rate:
            self.increment_hit_count(best_contract)
            logger.debug(
                f"Selected contract {best_contract.contract_id} "
                f"(success_rate={best_score:.1%}, hit_count={best_contract.hit_count})"
            )
            return best_contract, []
        
        return None, self._collect_unmatched(best_contract, column_values)
    
    def _collect_unmatched(
        self, contract: TransformContract, column_values: List[str], limit: int = 15
    ) -> List[str]:
        """
        Collect values that don't match the contract pattern using pandas vectorization.
        
        Returns up to `limit` non-matching values for LLM feedback.
        """
        if contract.compiled_pattern is None:
            return []
        
        # Convert to pandas Series for vectorized matching
        s = pd.Series(
            [str(v).strip() if v else "" for v in column_values], 
            dtype="string"
        )
        
        # Filter out empty values
        non_empty_mask = s != ""
        
        # Vectorized pattern matching (direct call, no subprocess overhead)
        matched, has_error = safe_pandas_match(contract.pattern, s)
        
        if has_error:
            # On error, return first few non-empty values as unmatched
            return s[non_empty_mask].head(limit).tolist()
        
        # Find unmatched non-empty values
        unmatched_mask = (~matched) & non_empty_mask

    
    def store(
        self, 
        contract: TransformContract, 
        column_values: Optional[List[str]] = None,
        bypass_quality_check: bool = False,
    ) -> Optional[str]:
        """
        Store a new TransformContract.
        
        Quality Gate: Only contracts that are BETTER than existing best are stored.
        This prevents accumulation of redundant contracts.
        
        LRU Eviction: When contracts for a (class, property) exceed MAX_CONTRACTS_PER_KEY,
        the least-used contract (lowest hit_count) is evicted.
        
        Args:
            contract: The transform contract to store
            column_values: Column values to evaluate success_rate (required for quality check)
            bypass_quality_check: If True, skip quality check (for testing/migration)
            
        Returns:
            Contract ID if stored, None if rejected
        """
        self._load_cache()
        
        key = f"{contract.primitive_class}::{contract.data_property}"
        pattern_key = (contract.primitive_class, contract.data_property, contract.pattern)
        
        # Compute success rate outside lock (expensive operation)
        new_success_rate = None
        if not bypass_quality_check:
            if column_values is None:
                logger.warning(
                    f"Cannot evaluate contract quality without column_values, "
                    f"rejecting contract {contract.contract_id}"
                )
                return None
            new_success_rate = contract.success_rate(column_values)
        
        with self._lock:
            # Check for exact pattern duplicate
            if pattern_key in self._pattern_keys:
                logger.debug(f"Contract already exists: {contract.contract_id}")
                return contract.contract_id
            
            existing = self._contracts_cache.get(key, [])
            
            # Quality gate: new contract must be better than existing best
            if not bypass_quality_check and new_success_rate is not None and existing:
                best_existing_rate = max(
                    ex.success_rate(column_values) for ex in existing
                )
                
                if new_success_rate <= best_existing_rate:
                    logger.debug(
                        f"Contract {contract.contract_id} rejected: success_rate={new_success_rate:.1%} "
                        f"<= best_existing={best_existing_rate:.1%}"
                    )
                    return None
                
                logger.debug(
                    f"Contract {contract.contract_id} accepted: success_rate={new_success_rate:.1%} "
                    f"> best_existing={best_existing_rate:.1%}"
                )
            elif not bypass_quality_check and new_success_rate is not None:
                logger.debug(
                    f"Contract {contract.contract_id} accepted (first contract): "
                    f"success_rate={new_success_rate:.1%}"
                )
            
            # LRU eviction: if at limit, remove lowest hit_count
            if len(existing) >= MAX_CONTRACTS_PER_KEY:
                min_hit_contract = min(existing, key=lambda c: c.hit_count)
                existing.remove(min_hit_contract)
                self._pattern_keys.discard(
                    (min_hit_contract.primitive_class, 
                     min_hit_contract.data_property, 
                     min_hit_contract.pattern)
                )
                logger.info(
                    f"LRU eviction: removing contract {min_hit_contract.contract_id} "
                    f"(hit_count={min_hit_contract.hit_count}) for key={key}"
                )
            
            # Add to cache
            if key not in self._contracts_cache:
                self._contracts_cache[key] = []
            self._contracts_cache[key].append(contract)
            self._pattern_keys.add(pattern_key)
            self._cache_dirty = True
            
            logger.debug(f"Stored contract in cache: {contract.contract_id}")
            return contract.contract_id
    
    def increment_hit_count(self, contract: TransformContract) -> None:
        """
        Increment the hit_count of a contract when it is selected for use.
        
        Thread-safe: Uses RLock to protect hit_count modification.
        
        Args:
            contract: The contract that was used
        """
        with self._lock:
            contract.hit_count += 1
            self._cache_dirty = True
            logger.debug(f"Contract {contract.contract_id} hit_count -> {contract.hit_count}")
