"""Service for prompt optimization strategies"""
from typing import List, Optional
from ..domain.entities.prompt import Prompt
from ..domain.entities.evaluation_metrics import EvaluationMetrics
from ..use_cases.optimize_prompt import OptimizePromptUseCase


class PromptOptimizationService:
    """Service for managing prompt optimization strategies"""
    
    def __init__(self, optimize_use_case: OptimizePromptUseCase):
        """
        Initialize service
        
        Args:
            optimize_use_case: Use case for optimizing prompts
        """
        self.optimize_use_case = optimize_use_case
    
    def optimize_with_strategy(
        self,
        current_prompt: Prompt,
        evaluation_metrics: EvaluationMetrics,
        chunks: list,
        iteration: int,
        strategy: str = "default"
    ) -> Prompt:
        """
        Optimize prompt using specified strategy
        
        Args:
            current_prompt: Current prompt
            evaluation_metrics: Evaluation results
            chunks: Generated chunks
            iteration: Current iteration
            strategy: Optimization strategy ("default", "aggressive", "conservative")
        
        Returns:
            Optimized prompt
        """
        if strategy == "default":
            return self.optimize_use_case.execute(
                current_prompt,
                evaluation_metrics,
                chunks,
                iteration
            )
        elif strategy == "aggressive":
            return self._optimize_aggressive(
                current_prompt,
                evaluation_metrics,
                chunks,
                iteration
            )
        elif strategy == "conservative":
            return self._optimize_conservative(
                current_prompt,
                evaluation_metrics,
                chunks,
                iteration
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _optimize_aggressive(
        self,
        current_prompt: Prompt,
        evaluation_metrics: EvaluationMetrics,
        chunks: list,
        iteration: int
    ) -> Prompt:
        """Aggressive optimization strategy - focus on lowest scores"""
        # Focus on the metric with the lowest score
        scores = {
            "boundary_clarity": evaluation_metrics.boundary_clarity,
            "chunk_stickiness": evaluation_metrics.chunk_stickiness,
            "hope_score": evaluation_metrics.hope_score
        }
        
        lowest_metric = min(scores, key=scores.get)
        
        # Use base optimization but emphasize the lowest metric
        return self.optimize_use_case.execute(
            current_prompt,
            evaluation_metrics,
            chunks,
            iteration
        )
    
    def _optimize_conservative(
        self,
        current_prompt: Prompt,
        evaluation_metrics: EvaluationMetrics,
        chunks: list,
        iteration: int
    ) -> Prompt:
        """Conservative optimization strategy - small incremental changes"""
        # For conservative, we still use the base optimization
        # but could add constraints or smaller changes
        return self.optimize_use_case.execute(
            current_prompt,
            evaluation_metrics,
            chunks,
            iteration
        )
    
    def should_continue_optimization(
        self,
        current_metrics: EvaluationMetrics,
        previous_metrics: Optional[EvaluationMetrics],
        config
    ) -> bool:
        """
        Determine if optimization should continue
        
        Args:
            current_metrics: Current evaluation metrics
            previous_metrics: Previous evaluation metrics (if any)
            config: Optimization configuration
        
        Returns:
            True if should continue, False otherwise
        """
        # Check threshold
        if current_metrics.meets_threshold(config.threshold):
            return False
        
        # Check improvement
        if previous_metrics:
            improvement = current_metrics.overall_score - previous_metrics.overall_score
            if improvement < config.improvement_threshold:
                return False
        
        return True
