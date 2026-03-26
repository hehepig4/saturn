"""
Predefined Statistics Functions

This module contains all the predefined statistics computation functions
for different data types. These are registered with the STATISTICS_REGISTRY
and called based on the DataProperty's range_type.

Key principle: LLM does NOT generate statistics code - it's all predefined here.
"""

import statistics as stats_module
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Union

from workflows.population.contract import register_statistics


def compute_integer_statistics(values: List[int]) -> Dict[str, Any]:
    """
    Compute statistics for integer values.
    
    Statistics: min, max, mean, median, sum, count, std (if n > 1),
                percentile_25, percentile_75, mode
    """
    if not values:
        return {"count": 0}
    
    result = {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
        "mean": stats_module.mean(values),
        "median": stats_module.median(values),
    }
    
    if len(values) > 1:
        result["std"] = stats_module.stdev(values)
        # Percentiles using quantiles
        sorted_values = sorted(values)
        result["percentile_25"] = stats_module.quantiles(sorted_values, n=4)[0]
        result["percentile_75"] = stats_module.quantiles(sorted_values, n=4)[2]
    
    # Mode - most common value
    try:
        result["mode"] = stats_module.mode(values)
    except stats_module.StatisticsError:
        # No unique mode (multimodal), skip
        pass
    
    return result


def compute_decimal_statistics(values: List[float]) -> Dict[str, Any]:
    """
    Compute statistics for decimal/float values.
    
    Statistics: min, max, mean, median, std, count, percentile_25, percentile_75
    """
    if not values:
        return {"count": 0}
    
    result = {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": stats_module.mean(values),
        "median": stats_module.median(values),
    }
    
    if len(values) > 1:
        result["std"] = stats_module.stdev(values)
        # Percentiles using quantiles
        sorted_values = sorted(values)
        result["percentile_25"] = stats_module.quantiles(sorted_values, n=4)[0]
        result["percentile_75"] = stats_module.quantiles(sorted_values, n=4)[2]
    
    return result


def compute_nonnegative_integer_statistics(values: List[int]) -> Dict[str, Any]:
    """
    Compute statistics for non-negative integers.
    
    Statistics: min, max, sum, count, mean
    """
    # Filter to only non-negative values
    valid = [v for v in values if v >= 0]
    if not valid:
        return {"count": 0}
    
    return {
        "count": len(valid),
        "min": min(valid),
        "max": max(valid),
        "sum": sum(valid),
        "mean": stats_module.mean(valid),
    }


def compute_datetime_statistics(values: List[datetime]) -> Dict[str, Any]:
    """
    Compute statistics for datetime values.
    
    Statistics: min, max, range_days, count
    """
    if not values:
        return {"count": 0}
    
    min_dt = min(values)
    max_dt = max(values)
    
    return {
        "count": len(values),
        "min": min_dt.isoformat(),
        "max": max_dt.isoformat(),
        "range_days": (max_dt - min_dt).days,
    }


def compute_date_statistics(values: List[datetime]) -> Dict[str, Any]:
    """
    Compute statistics for date values (datetime without time component).
    
    Statistics: min, max, range_days, count
    """
    return compute_datetime_statistics(values)


def compute_year_statistics(values: List[int]) -> Dict[str, Any]:
    """
    Compute statistics for year values.
    
    Statistics: min, max, range_years, count
    """
    if not values:
        return {"count": 0}
    
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "range_years": max(values) - min(values),
    }


def _compute_common_prefix(values: List[str]) -> str:
    """Find the longest common prefix among values."""
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    
    # Find common prefix
    prefix = values[0]
    for v in values[1:]:
        while not v.startswith(prefix) and prefix:
            prefix = prefix[:-1]
        if not prefix:
            break
    
    # Only return if prefix is meaningful (at least 2 chars, not just whitespace)
    if len(prefix) >= 2 and prefix.strip():
        return prefix
    return ""


def _compute_common_suffix(values: List[str]) -> str:
    """Find the longest common suffix among values."""
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    
    # Reverse strings and find common prefix
    reversed_values = [v[::-1] for v in values]
    suffix = _compute_common_prefix(reversed_values)
    return suffix[::-1] if suffix else ""


def _compute_category_entropy(counter: Counter) -> float:
    """Compute Shannon entropy of category distribution."""
    import math
    total = sum(counter.values())
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for count in counter.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return round(entropy, 3)


def _extract_tfidf_keywords(values: List[str], top_k: int = 5) -> List[str]:
    """Extract top TF-IDF keywords from text values."""
    import re
    from collections import defaultdict
    import math
    
    # Simple tokenization
    def tokenize(text: str) -> List[str]:
        # Split on non-alphanumeric, lowercase, filter short words
        words = re.findall(r'[a-zA-Z]{3,}', text.lower())
        return words
    
    # Stopwords (common English words to filter)
    stopwords = {
        'the', 'and', 'for', 'with', 'that', 'this', 'from', 'are', 'was', 'were',
        'has', 'have', 'had', 'not', 'but', 'they', 'you', 'all', 'can', 'her',
        'his', 'she', 'him', 'its', 'who', 'how', 'what', 'when', 'where', 'which'
    }
    
    # Document frequency
    doc_freq = defaultdict(int)
    # Term frequency per document
    term_freqs = []
    
    for v in values:
        tokens = tokenize(str(v))
        tokens = [t for t in tokens if t not in stopwords]
        tf = Counter(tokens)
        term_freqs.append(tf)
        for term in set(tokens):
            doc_freq[term] += 1
    
    if not doc_freq:
        return []
    
    # Compute TF-IDF scores
    n_docs = len(values)
    tfidf_scores = defaultdict(float)
    
    for tf in term_freqs:
        for term, freq in tf.items():
            # TF-IDF = tf * log(N / df)
            idf = math.log(n_docs / (doc_freq[term] + 1)) + 1
            tfidf_scores[term] += freq * idf
    
    # Get top-k keywords
    sorted_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    return [term for term, _ in sorted_terms[:top_k]]


def compute_string_statistics(values: List[str]) -> Dict[str, Any]:
    """
    Compute statistics for string/text values.
    
    Statistics: 
    - Basic: count, distinct_count, unique_ratio, avg_length, min_length, max_length
    - Word stats: word_count_avg (average word count per value)
    - Top values: top_values, sample_values
    - Pattern analysis: common_prefix, common_suffix
    - Category analysis: is_categorical, category_entropy, dominant_category_ratio
    - Text analysis: tfidf_keywords (for descriptive text)
    """
    if not values:
        return {"count": 0, "distinct_count": 0}
    
    # Import truncation limit for sample values
    from config.truncation_limits import TruncationLimits
    max_sample_len = TruncationLimits.SAMPLE_VALUE_MAX_LENGTH
    
    lengths = [len(str(v)) for v in values]
    counter = Counter(values)
    top_items = counter.most_common(10)
    
    # top_values: structured list with counts (truncate values for safety)
    top_values = [
        {"value": str(v)[:max_sample_len] + ("..." if len(str(v)) > max_sample_len else ""), "count": c} 
        for v, c in top_items
    ]
    
    # sample_values: simple list of top value strings (for readout templates)
    # Truncate each value to prevent binary/long data from overflowing prompts
    sample_values = [
        str(v)[:max_sample_len] + ("..." if len(str(v)) > max_sample_len else "")
        for v, c in top_items[:5]
    ]
    
    # Word count statistics (split by whitespace)
    word_counts = [len(str(v).split()) for v in values]
    word_count_avg = round(sum(word_counts) / len(word_counts), 1) if word_counts else 0
    
    # Basic stats
    result = {
        "count": len(values),
        "distinct_count": len(counter),
        "unique_ratio": round(len(counter) / len(values), 3),
        "avg_length": round(sum(lengths) / len(lengths), 1),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "word_count_avg": word_count_avg,
        "top_values": top_values,
        "sample_values": ", ".join(sample_values),
    }
    
    # Pattern analysis: common prefix/suffix (useful for ID columns)
    common_prefix = _compute_common_prefix(values)
    common_suffix = _compute_common_suffix(values)
    if common_prefix:
        result["common_prefix"] = common_prefix
    if common_suffix:
        result["common_suffix"] = common_suffix
    
    # Category analysis
    unique_ratio = len(counter) / len(values)
    is_categorical = unique_ratio < 0.2 and len(counter) <= 50  # Low unique ratio = categorical
    result["is_categorical"] = is_categorical
    
    if is_categorical:
        result["category_entropy"] = _compute_category_entropy(counter)
        # Dominant category ratio
        if top_items:
            result["dominant_category_ratio"] = round(top_items[0][1] / len(values), 3)
    
    # TF-IDF keywords (only for descriptive text with high unique ratio)
    if unique_ratio > 0.3 and result["avg_length"] > 20:
        keywords = _extract_tfidf_keywords(values, top_k=5)
        if keywords:
            result["tfidf_keywords"] = keywords
    
    return result


def compute_boolean_statistics(values: List[bool]) -> Dict[str, Any]:
    """
    Compute statistics for boolean values.
    
    Statistics: count, true_count, false_count, true_ratio
    """
    if not values:
        return {"count": 0}
    
    true_count = sum(1 for v in values if v)
    false_count = len(values) - true_count
    
    return {
        "count": len(values),
        "true_count": true_count,
        "false_count": false_count,
        "true_ratio": true_count / len(values),
    }


def compute_anyuri_statistics(values: List[str]) -> Dict[str, Any]:
    """
    Compute statistics for URI values.
    
    Statistics: count, distinct_count, domain_distribution
    """
    if not values:
        return {"count": 0, "distinct_count": 0}
    
    from urllib.parse import urlparse
    
    domains = []
    for v in values:
        try:
            parsed = urlparse(str(v))
            if parsed.netloc:
                domains.append(parsed.netloc)
        except Exception:
            pass
    
    domain_counter = Counter(domains)
    
    return {
        "count": len(values),
        "distinct_count": len(set(values)),
        "domain_distribution": [{"domain": d, "count": c} for d, c in domain_counter.most_common(10)],
    }


def compute_percentage_statistics(values: List[float]) -> Dict[str, Any]:
    """
    Compute statistics for percentage values (already converted to 0-100 scale).
    
    Statistics: min, max, mean, median, count
    """
    if not values:
        return {"count": 0}
    
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": stats_module.mean(values),
        "median": stats_module.median(values),
    }


def compute_currency_statistics(values: List[float]) -> Dict[str, Any]:
    """
    Compute statistics for currency values.
    
    Statistics: min, max, mean, median, sum, count
    """
    if not values:
        return {"count": 0}
    
    result = {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
        "mean": stats_module.mean(values),
        "median": stats_module.median(values),
    }
    
    if len(values) > 1:
        result["std"] = stats_module.stdev(values)
    
    return result


# ============== Register All Statistics ==============

def register_all_statistics() -> None:
    """Register all predefined statistics functions."""
    
    # Integer types
    register_statistics(
        "xsd:integer",
        ["min", "max", "mean", "median", "sum", "count", "std"],
        compute_integer_statistics
    )
    
    register_statistics(
        "xsd:nonNegativeInteger", 
        ["min", "max", "sum", "count", "mean"],
        compute_nonnegative_integer_statistics
    )
    
    register_statistics(
        "xsd:positiveInteger",
        ["min", "max", "sum", "count", "mean"],
        compute_nonnegative_integer_statistics  # Same logic
    )
    
    # Decimal/Float types
    register_statistics(
        "xsd:decimal",
        ["min", "max", "mean", "median", "std", "count"],
        compute_decimal_statistics
    )
    
    register_statistics(
        "xsd:float",
        ["min", "max", "mean", "median", "std", "count"],
        compute_decimal_statistics
    )
    
    register_statistics(
        "xsd:double",
        ["min", "max", "mean", "median", "std", "count"],
        compute_decimal_statistics
    )
    
    # Date/Time types
    register_statistics(
        "xsd:dateTime",
        ["min", "max", "range_days", "count"],
        compute_datetime_statistics
    )
    
    register_statistics(
        "xsd:date",
        ["min", "max", "range_days", "count"],
        compute_date_statistics
    )
    
    register_statistics(
        "xsd:gYear",
        ["min", "max", "range_years", "count"],
        compute_year_statistics
    )
    
    # String types
    register_statistics(
        "xsd:string",
        ["distinct_count", "top_values", "avg_length", "count"],
        compute_string_statistics
    )
    
    register_statistics(
        "xsd:normalizedString",
        ["distinct_count", "top_values", "avg_length", "count"],
        compute_string_statistics
    )
    
    # Boolean
    register_statistics(
        "xsd:boolean",
        ["true_count", "false_count", "true_ratio", "count"],
        compute_boolean_statistics
    )
    
    # URI
    register_statistics(
        "xsd:anyURI",
        ["distinct_count", "domain_distribution", "count"],
        compute_anyuri_statistics
    )


# Auto-register on module import
register_all_statistics()
