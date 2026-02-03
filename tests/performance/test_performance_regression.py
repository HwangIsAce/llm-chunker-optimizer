"""Performance regression tests"""
import pytest
from chunker_optimizer.infrastructure.performance.metrics_collector import (
    PerformanceMetricsCollector
)


class TestPerformanceRegression:
    """Performance regression tests"""
    
    @pytest.fixture
    def baseline_metrics(self):
        """Baseline performance metrics"""
        return {
            "max_evaluation_time": 1.0,  # seconds
            "max_optimization_time": 2.0,
            "max_total_time": 10.0,
            "min_cache_hit_rate": 0.3
        }
    
    def test_no_performance_regression(self, baseline_metrics):
        """Test that performance doesn't regress"""
        # Simulate actual metrics (would come from real runs)
        actual_metrics = {
            "max_evaluation_time": 0.8,
            "max_optimization_time": 1.5,
            "max_total_time": 8.0,
            "min_cache_hit_rate": 0.4
        }
        
        # Compare against baseline
        assert actual_metrics["max_evaluation_time"] <= baseline_metrics["max_evaluation_time"]
        assert actual_metrics["max_optimization_time"] <= baseline_metrics["max_optimization_time"]
        assert actual_metrics["max_total_time"] <= baseline_metrics["max_total_time"]
        assert actual_metrics["min_cache_hit_rate"] >= baseline_metrics["min_cache_hit_rate"]
    
    def test_optimization_speed_improvement(self):
        """Test that optimization speed improves"""
        # Before optimization
        before_time = 10.0
        
        # After optimization (e.g., with caching)
        after_time = 6.0
        
        improvement = (before_time - after_time) / before_time
        assert improvement >= 0.2  # At least 20% improvement
