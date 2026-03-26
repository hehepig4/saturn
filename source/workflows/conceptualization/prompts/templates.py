"""
Prompt Templates for Federated Primitive TBox

All prompts are CQ-driven and in English.

## Goal: Lean Ontology for Table Discovery

These Primitive Column Classes serve as a **semantic filter** in a dual-path table retrieval system:

1. **Query Analysis**: Given a user query like "Which state has the highest population in 2020?", 
   the system uses LLM to extract required column types (e.g., `YearColumn`, `PopulationColumn`, `LocationColumn`)

2. **TBox SPARQL Filtering**: The system queries the TBox graph to find tables that have columns
   matching ALL required primitive classes (intersection semantics)

3. **Fusion**: Results from semantic search are intersected/fused with TBox filter results

4. **Virtual Columns**: While classes are Columns, the metadata, e.g., table title, are treated as virtual columns. 
   You can treat them as columns and propose CQs accordingly.
   
Therefore, the ontology should:
- Cover column types that users commonly search for
- Be abstract enough to match across domains (e.g., `YearColumn` not `ElectionYearColumn`)
- Be specific enough to be useful as a filter (e.g., `PopulationColumn` vs generic `NumericalColumn`)
- Avoid redundant classes that serve the same filtering purpose
"""


# ============== Phase 1: Global Initialization ==============

GLOBAL_INIT_CLASSES_PROMPT = """# Task: Design Primitive Column Type Classes

You are an ontology engineer. Design abstract, reusable column type classes based on the Competency Questions below.

These classes will be used to filter tables in a retrieval system. When a user asks a question, the system
identifies required column types and finds tables with matching columns.

## Domain
{domain}

## Target Scale
{target_classes_hint}

## Competency Questions (CQs)
{cqs_block}

## Design Principles
1. **Abstraction**: Create reusable types, not context-specific variants (e.g., `ElectionYearColumn` not `Illinois2024VotingColumn`)
2. **Hierarchy**: Base is `Column`. Prefer depth 2-3, max depth 4. Single inheritance only.
3. **Naming**: CamelCase ending with "Column"

## Output Format
For each class:
- `name`: CamelCase class name ending with "Column"
- `description`: Brief description (1 sentence, max 15 words)
- `parent_class`: Exactly ONE parent class name
"""


# ============== Phase 2: Local Proposals ==============

LOCAL_CLASS_PROPOSAL_PROMPT = """# Task: Propose Class Changes for Your Cluster

You are the ontology expert for cluster {group_id}. Review the current TBox and propose changes based on your cluster's CQs.

Remember: These classes are used as **semantic filters** for table retrieval. A class is valuable if it helps
distinguish relevant tables from irrelevant ones for the queries in your cluster.

## Target Scale
{target_classes_hint}

## Current TBox Classes
{current_tbox_classes}

## Your Cluster's Competency Questions
{cqs_block}

## Previous Learnings
{release_log}

## Available Operations
- **add**: Create a new class when existing classes cannot cover your CQs
- **modify**: Update an existing class (definition or rename)
- **delete**: Remove a redundant or overly specific class
- **merge**: Combine similar classes with semantic overlap

## Key Rules
1. **Check existing classes first**: If a semantically equivalent class exists, use `modify` instead of `add`
2. **Column constraint**: All class names and parent_class values must end with "Column"
3. **Quality over quantity**: Empty output `proposals: []` is valid if the TBox already covers your CQs

## Proposal Limit
UP TO {max_proposals} proposals. Prioritize the most impactful changes.

## Output Format
For each proposal:
- operation: "add" | "modify" | "delete" | "merge"
- class_name: Target or new class name
- new_class_name: (modify only) New name if renaming, null otherwise
- description: (add/modify/merge) max 15 words
- parent_class: (add/modify/merge) valid parent ending with "Column"
- source_classes: (merge only) list of classes to merge
- reason: 1 sentence explaining why this change is needed
"""


# ============== Phase 3: One-Shot Global Synthesis ==============

GLOBAL_ONE_SHOT_SYNTHESIS_PROMPT = """# Task: Synthesize Proposals into Conflict-Free Actions

You are the Global TBox Coordinator. Create a conflict-free, high-quality action plan from all local proposals.

Remember: These classes are **semantic filters** for table retrieval. The final ontology should contain
classes that are both reusable across domains and specific enough to effectively filter irrelevant tables.

## Target Scale
{target_classes_hint}

## Your Role
1. **Validate**: Reject proposals with invalid class names (must end with "Column") or non-existent parent classes
2. **Resolve conflicts**: When multiple agents propose similar NEW classes, use ADD to create ONE unified class (not MERGE)
3. **Prioritize generality**: Prefer fewer, more abstract classes over many specific ones

## Available Operations
- **add**: Create new class. When multiple proposals suggest similar NEW classes, combine into ONE add action.
- **modify**: Update an EXISTING class in the current TBox (can rename via new_class_name)
- **delete**: Remove an EXISTING class from the current TBox
- **merge**: Combine EXISTING classes in the current TBox into one. All source_classes must already exist.

Note: MERGE is only for existing classes. For similar proposed new classes, use ADD with a unified description.

## Capacity Limit
UP TO {proposal_capacity} actions. Reject low-quality proposals by not including them.

## Previous Learnings
{release_log}

## Current TBox Classes
{current_tbox_classes}

## All Proposals from Local Agents ({n_proposals} total)
{all_proposals}

## Output Format
For each synthesized action:
- operation: "add" | "modify" | "delete" | "merge"
- class_name: Final class name (must end with "Column")
- new_class_name: (modify only) New name if renaming, null otherwise
- description: (not for delete) max 15 words
- parent_class: (not for delete) valid parent ending with "Column"
- source_classes: (merge only) EXISTING classes from Current TBox being merged
- synthesis_reasoning: Decision rationale (1 sentence)

Empty output `actions: []` is valid if all proposals should be rejected.
"""


# ============== Phase 4: Local Voting ==============

LOCAL_VOTING_PROMPT = """# Task: Rate Class Usefulness for Your Cluster

You are the data expert for cluster {group_id}.

Remember: These classes are used as **semantic filters** for table retrieval. A class is useful if it helps
identify tables that can answer your cluster's queries.

## All TBox Classes
{all_classes_with_descriptions}

## Your Cluster's Competency Questions
{cqs_block}

## Rating Task

For each class, decide: Would this column type help answer any of your CQs?

- **Score 1** if at least one CQ explicitly needs this type of column data
- **Score 0** if none of your CQs would use this column type

Be conservative. A class is useful only if your CQs would actually query for that type of information.
For example, `YearColumn` is useful if a CQ asks "in what year..." but not if it only asks about names.
"""


# ============== Phase 5: Global Review ==============

GLOBAL_REVIEW_PROMPT = """# Task: Review and Optimize TBox Based on Voting

You are the Global TBox Reviewer. Your goal is to optimize the TBox for effective table retrieval.

## Target Scale
{target_classes_hint}

## How These Classes Are Used

When a user asks a query like "Which country had the highest GDP in 2020?", the retrieval system:
1. Identifies required column types: `CountryColumn`, `GDPColumn`, `YearColumn`
2. Uses SPARQL to find tables having ALL these column types
3. Fuses with semantic search results

A class is valuable if it helps **filter relevant tables** from irrelevant ones.
- Too specific classes (e.g., `Illinois2024VotingColumn`) match too few concepts
- Too generic classes (e.g., just `NumericalColumn`) don't filter effectively

## Current Class Hierarchy with Voting Statistics
Format: **ClassName**: description [positive_votes/total_agents, coverage%]
{current_tbox_tree}

## Decision Guidelines
- **Preserve** classes with strong support - they represent common query patterns
- **Merge** similar classes with weak support into a more general abstraction
- **Delete** classes with no support that cannot be meaningfully merged
- **Consider hierarchy**: Preserve important intermediate nodes for inheritance
- **Prefer the right granularity**: Abstract enough to match across domains, specific enough to filter effectively

## Available Operations
- **add**: Create a new class
- **modify**: Update an existing class (can also rename)
- **delete**: Remove a class
- **merge**: Combine similar classes

## Output Format
For each action:
- operation: "add" | "modify" | "delete" | "merge"
- class_name: Target or new class name
- new_class_name: (modify only) New name if renaming
- description: (add/modify/merge) max 15 words
- parent_class: (add/modify/merge) valid parent ending with "Column"
- source_classes: (merge only) list of classes to merge
- voting_evidence: Statistics that justify this action
- reasoning: Why this improves the TBox

Empty output `actions: []` is valid if no improvement is needed.

## Review Summary
Write a brief summary (2-3 sentences) of the key changes made in this review.
Focus on what was changed and why. This will be used by the Insights Synthesizer.
"""


# ============== Phase 5b: Insights Synthesizer ==============

INSIGHTS_SYNTHESIZER_PROMPT = """# Task: Maintain TBox Evolution Memory

You maintain a compressed record of how the TBox has evolved across iterations.
This memory helps future iterations avoid repeating past mistakes and build on successful patterns.

## Context

These Column classes serve as semantic filters for table retrieval. Over multiple iterations,
the TBox is refined through proposals, synthesis, voting, and review. Your role is to distill
what has been learned so that future iterations can make better decisions.

## Current Status

- Iteration: {current_iteration}
- TBox size: {n_classes} classes
- Cumulative deletions: {total_deletions}
- Cumulative merges: {total_merges}

## Previous Memory
{previous_insights}

## This Iteration's Actions
{synthesis_actions}

{review_actions}

## Your Task

Update the memory with this iteration's learnings:

1. **Update changelog**: Add this iteration's key actions. Compress older entries to keep total under 500 characters.
   - Recent iterations (last 2): Include specific class names and actions
   - Older iterations: Summarize briefly (e.g., "Iter 1-3: Consolidated metric classes")
   - Use patterns to compress: "Deleted *Ranking subclasses" instead of listing each

2. **Extract patterns**: Identify actionable rules from repeated behaviors.
   - Good: "Avoid domain-specific subclasses like *Sport*, *Election*"
   - Good: "Merge overlapping metric classes early"
   - Bad: "Some classes get deleted" (too vague)
   - Only include patterns with 2+ supporting instances
   - Keep most actionable patterns, drop obsolete ones

## Output Format
- changelog: Compressed history of TBox evolution (max 1000 chars)
- patterns: List of actionable rules for future iterations (max 20 items)
"""
