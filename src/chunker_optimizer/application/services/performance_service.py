"""Service for performance analysis"""
from typing import List, Dict, Optional
from dataclasses import dataclass
from ..use_cases.run_optimization_loop import OptimizationResult


@dataclass
class PerformanceAnalysis:
    """Performance analysis result"""
    
    total_time: float
    average_time_per_iteration: float
    iterations: int
    convergence_rate: float
    improvement_rate: float
    metrics_by_iteration: List[Dict]


class PerformanceService:
    """Service for analyzing optimization performance"""
    
    def analyze_optimization_result(
        self,
        result: OptimizationResult,
        performance_metrics: Optional[Dict] = None
    ) -> PerformanceAnalysis:
        """
        Analyze optimization result performance
        
        Args:
            result: Optimization result
            performance_metrics: Optional performance metrics (timing, etc.)
        
        Returns:
            PerformanceAnalysis
        """
        iterations = result.iterations
        converged = result.converged
        
        # Extract metrics by iteration
        metrics_by_iteration = [
            entry["metrics"] for entry in result.history
        ]
        
        # Calculate convergence rate
        convergence_rate = 1.0 if converged else 0.0
        
        # Calculate improvement rate
        if len(metrics_by_iteration) >= 2:
            initial_score = metrics_by_iteration[0]["overall_score"]
            final_score = metrics_by_iteration[-1]["overall_score"]
            improvement_rate = (final_score - initial_score) / initial_score if initial_score > 0 else 0.0
        else:
            improvement_rate = 0.0
        
        # Extract timing information if available
        if performance_metrics:
            total_time = performance_metrics.get("total_time", 0.0)
            average_time_per_iteration = total_time / iterations if iterations > 0 else 0.0
        else:
            total_time = 0.0
            average_time_per_iteration = 0.0
        
        return PerformanceAnalysis(
            total_time=total_time,
            average_time_per_iteration=average_time_per_iteration,
            iterations=iterations,
            convergence_rate=convergence_rate,
            improvement_rate=improvement_rate,
            metrics_by_iteration=metrics_by_iteration
        )
    
    def compare_results(
        self,
        result1: OptimizationResult,
        result2: OptimizationResult
    ) -> Dict:
        """
        Compare two optimization results
        
        Args:
            result1: First result
            result2: Second result
        
        Returns:
            Comparison dictionary
        """
        analysis1 = self.analyze_optimization_result(result1)
        analysis2 = self.analyze_optimization_result(result2)
        
        return {
            "iterations_diff": result2.iterations - result1.iterations,
            "converged_diff": result2.converged - result1.converged,
            "final_score_diff": result2.final_metrics.overall_score - result1.final_metrics.overall_score,
            "improvement_rate_diff": analysis2.improvement_rate - analysis1.improvement_rate,
            "convergence_rate_diff": analysis2.convergence_rate - analysis1.convergence_rate
        }
