"""Thread-safe API token accounting and cache statistics."""

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime
from threading import RLock
from typing import Any, Dict, Optional

_current_caller = ContextVar('llm_caller', default='unknown')
_current_phase = ContextVar('llm_phase', default='unknown')


def set_current_caller(caller: str):
    return _current_caller.set(caller)


def get_current_caller() -> str:
    return _current_caller.get()


def set_current_phase(phase: str):
    return _current_phase.set(phase)


def reset_current_phase(token):
    _current_phase.reset(token)


def get_current_phase() -> str:
    return _current_phase.get()


@contextmanager
def usage_phase(phase: str):
    """Attribute calls to a pipeline phase and restore the caller's context."""
    token = set_current_phase(phase)
    try:
        yield
    finally:
        reset_current_phase(token)


def _usage_bucket():
    return dict(requests=0, input_tokens=0, output_tokens=0, total_tokens=0)


def _cache_bucket():
    return dict(hits=0, misses=0)


class UsageStats:
    """Accumulate provider-reported usage, separately from cache reuse."""

    def __init__(self):
        self._lock = RLock()
        self.start_time = datetime.now()
        self.reset()

    def reset(self):
        with self._lock:
            self.total_requests = 0
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.total_tokens = 0
            self.by_model = defaultdict(_usage_bucket)
            self.by_caller = defaultdict(_usage_bucket)
            self.by_phase = defaultdict(_usage_bucket)
            self.cache_by_kind = defaultdict(_cache_bucket)
            self.cache_by_kind_phase = defaultdict(lambda: defaultdict(_cache_bucket))
            self.async_calls = 0
            self.async_latencies = []
            self.last_reset_time = datetime.now()

    def record(self, model_name: str, input_tokens: int, output_tokens: int,
               caller: Optional[str] = None):
        input_tokens = max(0, int(input_tokens or 0))
        output_tokens = max(0, int(output_tokens or 0))
        total = input_tokens + output_tokens
        with self._lock:
            self.total_requests += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_tokens += total
            for bucket in (self.by_model[model_name],
                           self.by_caller[caller or get_current_caller()],
                           self.by_phase[get_current_phase()]):
                bucket['requests'] += 1
                bucket['input_tokens'] += input_tokens
                bucket['output_tokens'] += output_tokens
                bucket['total_tokens'] += total

    def record_cache(self, kind: str, hit: bool, phase: Optional[str] = None):
        key = 'hits' if hit else 'misses'
        with self._lock:
            self.cache_by_kind[kind][key] += 1
            self.cache_by_kind_phase[kind][phase or get_current_phase()][key] += 1

    def record_async(self, latency_ms: float):
        with self._lock:
            self.async_calls += 1
            self.async_latencies.append(latency_ms)
            self.async_latencies = self.async_latencies[-1000:]

    def get_summary(self) -> Dict[str, Any]:
        def cache_summary(cell):
            total = cell['hits'] + cell['misses']
            return {**cell, 'total': total,
                    'hit_rate': cell['hits'] / total if total else 0.0}

        with self._lock:
            now = datetime.now()
            return {
                'total_requests': self.total_requests,
                'total_input_tokens': self.total_input_tokens,
                'total_output_tokens': self.total_output_tokens,
                'total_tokens': self.total_tokens,
                'elapsed_seconds': (now - self.start_time).total_seconds(),
                'since_reset_seconds': (now - self.last_reset_time).total_seconds(),
                'by_model': deepcopy(dict(self.by_model)),
                'by_caller': deepcopy(dict(self.by_caller)),
                'by_phase': deepcopy(dict(self.by_phase)),
                'async_calls': self.async_calls,
                'async_latencies': list(self.async_latencies),
                'cache': {
                    'by_kind': {k: cache_summary(v) for k, v in self.cache_by_kind.items()},
                    'by_kind_phase': {
                        k: {p: cache_summary(v) for p, v in phases.items()}
                        for k, phases in self.cache_by_kind_phase.items()
                    },
                },
            }


_usage_stats = UsageStats()


def get_usage_stats() -> Dict[str, Any]:
    return _usage_stats.get_summary()


def record_usage(model_name: str, input_tokens: int, output_tokens: int,
                 caller: Optional[str] = None):
    _usage_stats.record(model_name, input_tokens, output_tokens, caller)


def record_async_call(latency_ms: float):
    _usage_stats.record_async(latency_ms)


def record_cache_hit(kind: str, phase: Optional[str] = None):
    _usage_stats.record_cache(kind, True, phase)


def record_cache_miss(kind: str, phase: Optional[str] = None):
    _usage_stats.record_cache(kind, False, phase)


def reset_usage_stats():
    _usage_stats.reset()
