"""Performance benchmark tests"""
import pytest
from chunker_optimizer.infrastructure.performance.benchmark_runner import (
    BenchmarkRunner,
    BenchmarkResult
)
from chunker_optimizer.infrastructure.performance.metrics_collector import (
    PerformanceMetricsCollector
)
from chunker_optimizer.domain.entities.evaluation_metrics import EvaluationMetrics
from chunker_optimizer.application.use_cases.run_optimization_loop import (
    OptimizationResult,
    OptimizationConfig
)
from chunker_optimizer.domain.entities.prompt import Prompt


class TestPerformanceBenchmark:
    """Performance benchmark tests"""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector"""
        return PerformanceMetricsCollector()
    
    @pytest.fixture
    def benchmark_runner(self, metrics_collector):
        """Create benchmark runner"""
        return BenchmarkRunner(metrics_collector)
    
    def test_benchmark_runner_initialization(self, benchmark_runner):
        """Test benchmark runner initialization"""
        assert benchmark_runner is not None
        assert benchmark_runner.metrics_collector is not None
    
    def test_run_benchmark(self, benchmark_runner):
        """Test running a benchmark"""
        def mock_optimization_function(**kwargs):
            """Mock optimization function"""
            return OptimizationResult(
                final_prompt=Prompt(id="test", content="Test", version=1),
                final_metrics=EvaluationMetrics(
                    0.8, 0.8, 0.8, {}, {}, 0.8
                ),
                iterations=2,
                history=[{"iteration": 0}, {"iteration": 1}],  # Match iterations
                converged=True
            )
        
        test_cases = [
            {
                "name": "test_case_1",
                "text": "Sample text"
            }
        ]
        
        results = benchmark_runner.run_benchmark(
            name="test_benchmark",
            optimization_function=mock_optimization_function,
            test_cases=test_cases,
            iterations=2
        )
        
        assert len(results) == 2  # 2 iterations
        assert all(isinstance(r, BenchmarkResult) for r in results)
        assert all(r.total_time > 0 for r in results)
    
    def test_compare_strategies(self, benchmark_runner):
        """Test comparing multiple strategies"""
        def strategy_a(**kwargs):
            return OptimizationResult(
                final_prompt=Prompt(id="test", content="A", version=1),
                final_metrics=EvaluationMetrics(0.7, 0.7, 0.7, {}, {}, 0.7),
                iterations=3,
                history=[{"iteration": 0}, {"iteration": 1}, {"iteration": 2}],  # Match iterations
                converged=False
            )
        
        def strategy_b(**kwargs):
            return OptimizationResult(
                final_prompt=Prompt(id="test", content="B", version=1),
                final_metrics=EvaluationMetrics(0.8, 0.8, 0.8, {}, {}, 0.8),
                iterations=2,
                history=[{"iteration": 0}, {"iteration": 1}],  # Match iterations
                converged=True
            )
        
        strategies = {
            "strategy_a": strategy_a,
            "strategy_b": strategy_b
        }
        
        test_cases = [{"name": "test", "text": "sample"}]
        
        comparison = benchmark_runner.compare_strategies(strategies, test_cases)
        
        assert "strategy_a" in comparison
        assert "strategy_b" in comparison
        assert len(comparison["strategy_a"]) > 0
        assert len(comparison["strategy_b"]) > 0
