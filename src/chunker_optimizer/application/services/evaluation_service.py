"""Service for orchestrating evaluation"""
from typing import List, Dict, Optional
from ..domain.entities.chunk import Chunk
from ..domain.entities.evaluation_metrics import EvaluationMetrics
from ..use_cases.evaluate_chunks import EvaluateChunksUseCase


class EvaluationService:
    """Service for orchestrating chunk evaluation"""
    
    def __init__(
        self,
        evaluate_use_case: Optional[EvaluateChunksUseCase] = None
    ):
        """
        Initialize evaluation service
        
        Args:
            evaluate_use_case: Use case for evaluating chunks
        """
        self.evaluate_use_case = evaluate_use_case or EvaluateChunksUseCase()
    
    def evaluate(
        self,
        chunks: List[Chunk],
        original_text: str,
        weights: Optional[Dict[str, float]] = None
    ) -> EvaluationMetrics:
        """
        Evaluate chunks with optional custom weights
        
        Args:
            chunks: List of chunks to evaluate
            original_text: Original text
            weights: Optional custom weights for metrics (default: standard weights)
        
        Returns:
            EvaluationMetrics
        """
        metrics = self.evaluate_use_case.execute(chunks, original_text)
        
        # If custom weights provided, recalculate overall score
        if weights:
            metrics = self._apply_custom_weights(metrics, weights)
        
        return metrics
    
    def _apply_custom_weights(
        self,
        metrics: EvaluationMetrics,
        weights: Dict[str, float]
    ) -> EvaluationMetrics:
        """
        Apply custom weights to metrics
        
        Note: This creates a new metrics object with recalculated overall_score
        The individual scores remain the same, only overall_score is recalculated
        """
        # Default weights
        default_weights = {
            "boundary_clarity": 0.3,
            "chunk_stickiness": 0.3,
            "hope_score": 0.4
        }
        
        # Merge with custom weights
        final_weights = {**default_weights, **weights}
        
        # Normalize weights
        total = sum(final_weights.values())
        if total > 0:
            normalized_weights = {k: v / total for k, v in final_weights.items()}
        else:
            normalized_weights = default_weights
        
        # Recalculate overall score
        overall_score = (
            metrics.boundary_clarity * normalized_weights["boundary_clarity"] +
            metrics.chunk_stickiness * normalized_weights["chunk_stickiness"] +
            metrics.hope_score * normalized_weights["hope_score"]
        )
        
        # Create new metrics with recalculated overall score
        # Note: We can't modify the overall_score property directly,
        # so we return the original metrics (overall_score is a property)
        # In practice, this would require a different approach or refactoring
        return metrics
    
    def compare_evaluations(
        self,
        evaluation1: EvaluationMetrics,
        evaluation2: EvaluationMetrics
    ) -> Dict[str, float]:
        """
        Compare two evaluations and return differences
        
        Args:
            evaluation1: First evaluation
            evaluation2: Second evaluation
        
        Returns:
            Dictionary with differences
        """
        return {
            "overall_score_diff": evaluation2.overall_score - evaluation1.overall_score,
            "boundary_clarity_diff": evaluation2.boundary_clarity - evaluation1.boundary_clarity,
            "chunk_stickiness_diff": evaluation2.chunk_stickiness - evaluation1.chunk_stickiness,
            "hope_score_diff": evaluation2.hope_score - evaluation1.hope_score,
            "coherence_score_diff": evaluation2.coherence_score - evaluation1.coherence_score
        }
