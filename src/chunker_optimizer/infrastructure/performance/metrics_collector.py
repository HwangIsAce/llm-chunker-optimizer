"""Performance metrics collector"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from contextlib import contextmanager
from collections import defaultdict


@dataclass
class PerformanceMetrics:
    """Performance metrics data class"""
    
    iteration: int
    evaluation_time: float
    optimization_time: float
    chunking_time: float
    llm_call_time: float
    total_time: float
    token_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


@dataclass
class OptimizationPerformance:
    """Overall optimization process performance data"""
    
    total_iterations: int
    total_time: float
    metrics_per_iteration: List[PerformanceMetrics] = field(default_factory=list)
    average_evaluation_time: float = 0.0
    average_optimization_time: float = 0.0
    average_chunking_time: float = 0.0
    average_llm_call_time: float = 0.0
    total_llm_calls: int = 0
    total_tokens: int = 0
    
    def calculate_averages(self):
        """Calculate average values"""
        if not self.metrics_per_iteration:
            return
        
        self.average_evaluation_time = sum(
            m.evaluation_time for m in self.metrics_per_iteration
        ) / len(self.metrics_per_iteration)
        
        self.average_optimization_time = sum(
            m.optimization_time for m in self.metrics_per_iteration
        ) / len(self.metrics_per_iteration)
        
        self.average_chunking_time = sum(
            m.chunking_time for m in self.metrics_per_iteration
        ) / len(self.metrics_per_iteration)
        
        self.average_llm_call_time = sum(
            m.llm_call_time for m in self.metrics_per_iteration
        ) / len(self.metrics_per_iteration)
        
        self.total_llm_calls = len(self.metrics_per_iteration)
        self.total_tokens = sum(m.token_count for m in self.metrics_per_iteration)


class PerformanceMetricsCollector:
    """Performance metrics collector"""
    
    def __init__(self):
        """Initialize collector"""
        self.metrics: List[PerformanceMetrics] = []
        self.timers: Dict[str, float] = {}
        self.counters: Dict[str, int] = defaultdict(int)
    
    @contextmanager
    def time_operation(self, operation_name: str):
        """
        Context manager for timing operations
        
        Args:
            operation_name: Name of the operation to time
        
        Example:
            with collector.time_operation("evaluation"):
                # do work
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timers[operation_name] = elapsed
    
    def record_metrics(
        self,
        iteration: int,
        evaluation_time: float,
        optimization_time: float,
        chunking_time: float,
        llm_call_time: float,
        token_count: int = 0,
        cache_hits: int = 0,
        cache_misses: int = 0
    ):
        """
        Record metrics for an iteration
        
        Args:
            iteration: Iteration number
            evaluation_time: Time spent on evaluation
            optimization_time: Time spent on optimization
            chunking_time: Time spent on chunking
            llm_call_time: Time spent on LLM calls
            token_count: Number of tokens used
            cache_hits: Number of cache hits
            cache_misses: Number of cache misses
        """
        total_time = evaluation_time + optimization_time + chunking_time + llm_call_time
        
        metric = PerformanceMetrics(
            iteration=iteration,
            evaluation_time=evaluation_time,
            optimization_time=optimization_time,
            chunking_time=chunking_time,
            llm_call_time=llm_call_time,
            total_time=total_time,
            token_count=token_count,
            cache_hits=cache_hits,
            cache_misses=cache_misses
        )
        
        self.metrics.append(metric)
    
    def get_summary(self) -> OptimizationPerformance:
        """
        Get performance summary
        
        Returns:
            OptimizationPerformance with aggregated metrics
        """
        if not self.metrics:
            return OptimizationPerformance(
                total_iterations=0,
                total_time=0.0
            )
        
        total_time = sum(m.total_time for m in self.metrics)
        
        performance = OptimizationPerformance(
            total_iterations=len(self.metrics),
            total_time=total_time,
            metrics_per_iteration=self.metrics.copy()
        )
        performance.calculate_averages()
        
        return performance
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.timers.clear()
        self.counters.clear()
    
    def increment_counter(self, counter_name: str, value: int = 1):
        """Increment a counter"""
        self.counters[counter_name] += value
    
    def get_counter(self, counter_name: str) -> int:
        """Get counter value"""
        return self.counters.get(counter_name, 0)
