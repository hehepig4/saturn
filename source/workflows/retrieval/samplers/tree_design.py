"""
Agent Tree Structure Designer

Automatically computes optimal n-ary tree structure for multi-agent CQ generation
based on capacity constraints and LLM call minimization.

Key insight: For fixed leaf count k, larger branching factor n yields fewer LLM calls.
Total agents ≈ k × n/(n-1), so overhead_ratio = 1/(n-1).

See docs2/AGENT_TREE_DESIGN.md for detailed documentation.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class LevelInfo:
    """Information about a single level in the agent tree."""

    level: int
    node_type: str  # 'leaf', 'aggregator', or 'root'
    n_agents: int
    children_per_agent: Optional[int] = None
    queries_range: Optional[Tuple[int, int]] = None
    cqs_range: Optional[Tuple[int, int]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "level": self.level,
            "type": self.node_type,
            "n_agents": self.n_agents,
            "children_per_agent": self.children_per_agent,
            "queries_range": self.queries_range,
            "cqs_range": self.cqs_range,
        }


@dataclass
class TreeConfig:
    """Configuration for the agent tree structure."""

    # Core structure
    n_clusters: int  # Number of leaf agents (clusters)
    branching_factor: int  # n-ary tree branching factor
    depth: int  # Tree depth (0 = only leaves + root)
    levels: List[LevelInfo] = field(default_factory=list)

    # Capacity constraints
    max_queries_per_leaf: int = 0
    queries_per_cluster: Tuple[int, int] = (0, 0)  # (min, max)
    cqs_per_cluster: Tuple[int, int] = (0, 0)  # (min, max)

    # Input parameters (for reference)
    num_queries: int = 0
    max_cq_capacity: int = 0
    max_subagent_capacity: int = 0
    num_cq_per_query: int = 0

    @property
    def total_agents(self) -> int:
        """Total number of agents in the tree."""
        return sum(lvl.n_agents for lvl in self.levels)

    @property
    def total_llm_calls(self) -> int:
        """Total LLM calls needed (one per agent)."""
        return self.total_agents

    @property
    def overhead_ratio(self) -> float:
        """Overhead ratio compared to leaf count."""
        if self.n_clusters == 0:
            return 0.0
        return (self.total_agents - self.n_clusters) / self.n_clusters

    def summary(self) -> str:
        """Return a concise summary string."""
        lines = [
            f"TreeConfig(n_clusters={self.n_clusters}, "
            f"branching={self.branching_factor}, depth={self.depth})",
            f"  Total agents: {self.total_agents}, "
            f"LLM calls: {self.total_llm_calls}, "
            f"overhead: {self.overhead_ratio:.1%}",
            "  Levels:",
        ]
        for lvl in self.levels:
            if lvl.node_type == "leaf":
                lines.append(
                    f"    L{lvl.level} ({lvl.node_type}): {lvl.n_agents} agents, "
                    f"queries={lvl.queries_range}, CQs={lvl.cqs_range}"
                )
            else:
                lines.append(
                    f"    L{lvl.level} ({lvl.node_type}): {lvl.n_agents} agents, "
                    f"children={lvl.children_per_agent}"
                )
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "n_clusters": self.n_clusters,
            "branching_factor": self.branching_factor,
            "depth": self.depth,
            "total_agents": self.total_agents,
            "total_llm_calls": self.total_llm_calls,
            "overhead_ratio": self.overhead_ratio,
            "max_queries_per_leaf": self.max_queries_per_leaf,
            "queries_per_cluster": self.queries_per_cluster,
            "cqs_per_cluster": self.cqs_per_cluster,
            "levels": [lvl.to_dict() for lvl in self.levels],
            "input_params": {
                "num_queries": self.num_queries,
                "max_cq_capacity": self.max_cq_capacity,
                "max_subagent_capacity": self.max_subagent_capacity,
                "num_cq_per_query": self.num_cq_per_query,
            },
        }


class AgentTreeDesigner:
    """
    Designer for optimal agent tree structure.

    The tree structure minimizes LLM calls while respecting capacity constraints:
    - Each leaf agent handles at most `max_cq_capacity // num_cq_per_query` queries
    - Each aggregator has at most `max_subagent_capacity` children

    Key insight: Optimal branching factor = max_subagent_capacity
    (larger n means smaller overhead = k/(n-1))
    """

    @staticmethod
    def compute(
        num_queries: int,
        max_cq_capacity: int,
        max_subagent_capacity: int,
        num_cq_per_query: int,
        min_cluster_ratio: float = 0.5,
    ) -> TreeConfig:
        """
        Compute optimal agent tree structure.

        Args:
            num_queries: Total number of queries (N)
            max_cq_capacity: Max CQs per leaf agent (LLM context limit)
            max_subagent_capacity: Max children per aggregator (branching factor upper bound)
            num_cq_per_query: CQs generated per query (n_vcq + n_scq)
            min_cluster_ratio: Min cluster size as ratio of average (for soft balance)

        Returns:
            TreeConfig with optimal tree structure

        Constraints:
            1. max_queries_per_leaf = max_cq_capacity // num_cq_per_query
            2. min_clusters = ceil(N / max_queries_per_leaf)
            3. branching_factor = max_subagent_capacity (optimal choice)
            4. depth = ceil(log_n(min_clusters))
        """
        # Validate inputs
        if num_queries <= 0:
            raise ValueError(f"num_queries must be positive, got {num_queries}")
        if max_cq_capacity <= 0:
            raise ValueError(f"max_cq_capacity must be positive, got {max_cq_capacity}")
        if max_subagent_capacity < 2:
            raise ValueError(
                f"max_subagent_capacity must be >= 2, got {max_subagent_capacity}"
            )
        if num_cq_per_query <= 0:
            raise ValueError(
                f"num_cq_per_query must be positive, got {num_cq_per_query}"
            )
        if not 0 < min_cluster_ratio <= 1:
            raise ValueError(
                f"min_cluster_ratio must be in (0, 1], got {min_cluster_ratio}"
            )

        # Step 1: Compute hard constraints
        max_queries_per_leaf = max_cq_capacity // num_cq_per_query
        if max_queries_per_leaf == 0:
            raise ValueError(
                f"max_cq_capacity ({max_cq_capacity}) < num_cq_per_query ({num_cq_per_query}), "
                "cannot fit even one query per leaf"
            )

        # Step 2: Compute n_clusters using balanced target
        # Target avg = (size_min + size_max) / 2 = max_queries × (1 + min_ratio) / 2
        # This ensures avg is centered in the valid range [max×min_ratio, max]
        target_avg = max_queries_per_leaf * (1 + min_cluster_ratio) / 2
        min_clusters = math.ceil(num_queries / target_avg)

        # Step 3: Optimal branching factor = max_subagent_capacity
        # Reasoning: overhead = k/(n-1), larger n means smaller overhead
        branching_factor = max_subagent_capacity

        # Step 4: Compute tree depth
        if min_clusters <= 1:
            depth = 0
            n_clusters = 1
        else:
            depth = math.ceil(math.log(min_clusters) / math.log(branching_factor))
            n_clusters = min_clusters

        # Step 4: Build level structure
        levels = AgentTreeDesigner._build_levels(
            n_clusters=n_clusters,
            branching_factor=branching_factor,
            num_queries=num_queries,
            max_queries_per_leaf=max_queries_per_leaf,
            num_cq_per_query=num_cq_per_query,
            min_cluster_ratio=min_cluster_ratio,
        )

        # Step 5: Extract leaf constraints
        leaf_level = levels[0]
        queries_per_cluster = leaf_level.queries_range or (0, 0)
        cqs_per_cluster = leaf_level.cqs_range or (0, 0)

        return TreeConfig(
            n_clusters=n_clusters,
            branching_factor=branching_factor,
            depth=depth,
            levels=levels,
            max_queries_per_leaf=max_queries_per_leaf,
            queries_per_cluster=queries_per_cluster,
            cqs_per_cluster=cqs_per_cluster,
            num_queries=num_queries,
            max_cq_capacity=max_cq_capacity,
            max_subagent_capacity=max_subagent_capacity,
            num_cq_per_query=num_cq_per_query,
        )

    @staticmethod
    def _build_levels(
        n_clusters: int,
        branching_factor: int,
        num_queries: int,
        max_queries_per_leaf: int,
        num_cq_per_query: int,
        min_cluster_ratio: float,
    ) -> List[LevelInfo]:
        """Build the level structure from leaves to root."""
        levels = []
        current_nodes = n_clusters

        level = 0
        while True:
            if level == 0:
                # Leaf level
                # Range: [max × min_ratio, max] where max = max_queries_per_leaf
                avg_queries = num_queries / current_nodes if current_nodes > 0 else 0
                min_queries = max(1, int(max_queries_per_leaf * min_cluster_ratio))
                max_queries = max_queries_per_leaf  # Hard constraint

                levels.append(
                    LevelInfo(
                        level=level,
                        node_type="leaf",
                        n_agents=current_nodes,
                        children_per_agent=None,
                        queries_range=(min_queries, max_queries),
                        cqs_range=(
                            min_queries * num_cq_per_query,
                            max_queries * num_cq_per_query,
                        ),
                    )
                )
            elif current_nodes > 1:
                # Aggregator level
                parent_nodes = math.ceil(current_nodes / branching_factor)
                children_per_parent = math.ceil(current_nodes / parent_nodes)

                levels.append(
                    LevelInfo(
                        level=level,
                        node_type="aggregator",
                        n_agents=parent_nodes,
                        children_per_agent=children_per_parent,
                    )
                )
                current_nodes = parent_nodes
            else:
                # Root level (current_nodes == 1)
                # Check if we already added an aggregator with 1 agent
                if levels[-1].n_agents > 1:
                    levels.append(
                        LevelInfo(
                            level=level,
                            node_type="root",
                            n_agents=1,
                            children_per_agent=levels[-1].n_agents,
                        )
                    )
                else:
                    # Relabel the last level as root
                    levels[-1].node_type = "root"
                break

            level += 1

            # Safety check to prevent infinite loops
            if level > 100:
                raise RuntimeError("Tree construction exceeded maximum depth")

        return levels

    @staticmethod
    def estimate_llm_calls(n_leaves: int, branching_factor: int) -> int:
        """
        Estimate total LLM calls for given leaf count and branching factor.

        Args:
            n_leaves: Number of leaf agents
            branching_factor: n-ary tree branching factor

        Returns:
            Estimated total LLM calls (total agents in tree)
        """
        if n_leaves <= 0:
            return 0

        total = n_leaves
        current = n_leaves

        while current > 1:
            parent_count = math.ceil(current / branching_factor)
            total += parent_count
            current = parent_count

        return total

    @staticmethod
    def theoretical_overhead(branching_factor: int) -> float:
        """
        Theoretical overhead ratio for a given branching factor.

        For large k: overhead ≈ 1/(n-1)

        Args:
            branching_factor: n-ary tree branching factor

        Returns:
            Theoretical overhead ratio
        """
        if branching_factor <= 1:
            return float("inf")
        return 1.0 / (branching_factor - 1)


# Convenience function for quick computation
def compute_tree_structure(
    num_queries: int,
    max_cq_capacity: int,
    max_subagent_capacity: int,
    num_cq_per_query: int,
    min_cluster_ratio: float = 0.5,
) -> TreeConfig:
    """
    Convenience function to compute optimal agent tree structure.

    See AgentTreeDesigner.compute() for details.
    """
    return AgentTreeDesigner.compute(
        num_queries=num_queries,
        max_cq_capacity=max_cq_capacity,
        max_subagent_capacity=max_subagent_capacity,
        num_cq_per_query=num_cq_per_query,
        min_cluster_ratio=min_cluster_ratio,
    )
