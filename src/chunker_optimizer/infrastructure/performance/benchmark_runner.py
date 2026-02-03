"""Benchmark runner for optimization strategies"""
import time
from typing import List, Dict, Callable, Any
from dataclasses import dataclass
from ...domain.entities.evaluation_metrics import EvaluationMetrics
from ...application.use_cases.run_optimization_loop import OptimizationResult
from .metrics_collector import (
    PerformanceMetricsCollector,
    OptimizationPerformance
)


@dataclass
class BenchmarkResult:
    """Benchmark result"""
    
    name: str
    iterations: int
    total_time: float
    average_time_per_iteration: float
    final_metrics: EvaluationMetrics
    performance: OptimizationPerformance
    converged: bool


class BenchmarkRunner:
    """Benchmark runner for comparing optimization strategies"""
    
    def __init__(self, metrics_collector: PerformanceMetricsCollector):
        """
        Initialize benchmark runner
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics_collector = metrics_collector
    
    def run_benchmark(
        self,
        name: str,
        optimization_function: Callable,
        test_cases: List[Dict[str, Any]],
        iterations: int = 3
    ) -> List[BenchmarkResult]:
        """
        Run benchmark on optimization function
        
        Args:
            name: Name of the benchmark
            optimization_function: Function that runs optimization
            test_cases: List of test case dictionaries
            iterations: Number of iterations to run for each test case
        
        Returns:
            List of benchmark results
        """
        results = []
        
        for test_case in test_cases:
            case_results = []
            
            for i in range(iterations):
                self.metrics_collector.reset()
                
                start_time = time.perf_counter()
                result = optimization_function(**test_case)
                end_time = time.perf_counter()
                
                performance = self.metrics_collector.get_summary()
                
                benchmark_result = BenchmarkResult(
                    name=f"{name}_{test_case.get('name', 'unknown')}_run_{i+1}",
                    iterations=result.iterations,
                    total_time=end_time - start_time,
                    average_time_per_iteration=performance.average_evaluation_time,
                    final_metrics=result.final_metrics,
                    performance=performance,
                    converged=result.converged
                )
                
                case_results.append(benchmark_result)
            
            results.extend(case_results)
        
        return results
    
    def compare_strategies(
        self,
        strategies: Dict[str, Callable],
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, List[BenchmarkResult]]:
        """
        Compare multiple optimization strategies
        
        Args:
            strategies: Dictionary mapping strategy names to functions
            test_cases: List of test case dictionaries
        
        Returns:
            Dictionary mapping strategy names to benchmark results
        """
        comparison = {}
        
        for strategy_name, strategy_func in strategies.items():
            results = self.run_benchmark(
                name=strategy_name,
                optimization_function=strategy_func,
                test_cases=test_cases
            )
            comparison[strategy_name] = results
        
        return comparison
