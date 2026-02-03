"""Performance report generator"""
from typing import List, Dict
from dataclasses import asdict
import json
from pathlib import Path
from .benchmark_runner import BenchmarkResult


class PerformanceReporter:
    """Performance report generator"""
    
    def generate_report(
        self,
        results: List[BenchmarkResult],
        output_path: Path
    ):
        """
        Generate performance report
        
        Args:
            results: List of benchmark results
            output_path: Path to save report
        """
        report = {
            "summary": self._generate_summary(results),
            "detailed_results": [asdict(r) for r in results],
            "comparisons": self._generate_comparisons(results)
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    def _generate_summary(self, results: List[BenchmarkResult]) -> Dict:
        """Generate summary statistics"""
        if not results:
            return {}
        
        return {
            "total_runs": len(results),
            "average_time": sum(r.total_time for r in results) / len(results),
            "average_iterations": sum(r.iterations for r in results) / len(results),
            "convergence_rate": sum(1 for r in results if r.converged) / len(results),
            "fastest_run": min(results, key=lambda r: r.total_time).name if results else None,
            "slowest_run": max(results, key=lambda r: r.total_time).name if results else None
        }
    
    def _generate_comparisons(self, results: List[BenchmarkResult]) -> Dict:
        """Generate comparison analysis"""
        # Group by strategy name (extract from result name)
        strategies = {}
        for result in results:
            # Extract strategy name (before first underscore)
            strategy = result.name.split("_")[0]
            if strategy not in strategies:
                strategies[strategy] = []
            strategies[strategy].append(result)
        
        comparisons = {}
        for strategy, strategy_results in strategies.items():
            comparisons[strategy] = {
                "average_time": sum(r.total_time for r in strategy_results) / len(strategy_results),
                "average_iterations": sum(r.iterations for r in strategy_results) / len(strategy_results),
                "convergence_rate": sum(1 for r in strategy_results if r.converged) / len(strategy_results)
            }
        
        return comparisons
