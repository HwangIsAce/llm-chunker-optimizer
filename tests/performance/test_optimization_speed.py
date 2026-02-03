"""Optimization speed tests"""
import pytest
import time
from chunker_optimizer.infrastructure.performance.metrics_collector import (
    PerformanceMetricsCollector,
    OptimizationPerformance
)


class TestOptimizationSpeed:
    """Optimization speed tests"""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector"""
        return PerformanceMetricsCollector()
    
    def test_iteration_speed(self, metrics_collector):
        """Test iteration speed measurement"""
        # Simulate multiple iterations
        for i in range(5):
            with metrics_collector.time_operation(f"iteration_{i}"):
                time.sleep(0.01)  # Simulate work
        
        summary = metrics_collector.get_summary()
        assert summary.total_iterations == 0  # No recorded metrics yet
        assert len(metrics_collector.timers) == 5
    
    def test_metrics_recording(self, metrics_collector):
        """Test metrics recording"""
        metrics_collector.record_metrics(
            iteration=0,
            evaluation_time=0.5,
            optimization_time=1.0,
            chunking_time=0.2,
            llm_call_time=0.8,
            token_count=100
        )
        
        summary = metrics_collector.get_summary()
        assert summary.total_iterations == 1
        assert summary.average_evaluation_time == 0.5
        assert summary.average_optimization_time == 1.0
    
    def test_convergence_speed(self):
        """Test convergence speed comparison"""
        # Fast convergence scenario
        fast_convergence = {
            "iterations": 3,
            "time": 5.0
        }
        
        # Slow convergence scenario
        slow_convergence = {
            "iterations": 10,
            "time": 20.0
        }
        
        fast_speed = fast_convergence["time"] / fast_convergence["iterations"]
        slow_speed = slow_convergence["time"] / slow_convergence["iterations"]
        
        assert fast_speed < slow_speed
    
    def test_cache_hit_rate(self, metrics_collector):
        """Test cache hit rate calculation"""
        metrics_collector.record_metrics(
            iteration=0,
            evaluation_time=0.5,
            optimization_time=1.0,
            chunking_time=0.2,
            llm_call_time=0.8,
            cache_hits=7,
            cache_misses=3
        )
        
        metric = metrics_collector.metrics[0]
        assert metric.cache_hit_rate == 0.7  # 7 / (7 + 3)
    
    def test_performance_summary(self, metrics_collector):
        """Test performance summary generation"""
        # Record multiple iterations
        for i in range(3):
            metrics_collector.record_metrics(
                iteration=i,
                evaluation_time=0.5 + i * 0.1,
                optimization_time=1.0,
                chunking_time=0.2,
                llm_call_time=0.8,
                token_count=100 + i * 10
            )
        
        summary = metrics_collector.get_summary()
        
        assert summary.total_iterations == 3
        assert summary.total_time > 0
        assert summary.average_evaluation_time > 0
        assert summary.total_tokens == 330  # 100 + 110 + 120
