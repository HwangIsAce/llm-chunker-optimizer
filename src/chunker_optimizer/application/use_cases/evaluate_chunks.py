"""Use case for evaluating chunk quality"""
from typing import List
from ...domain.entities.chunk import Chunk
from ...domain.entities.evaluation_metrics import EvaluationMetrics
from ...domain.value_objects.boundary_clarity import BoundaryClarity
from ...domain.value_objects.chunk_stickiness import ChunkStickiness
from ...domain.value_objects.hope_metrics import HOPEMetrics


class EvaluateChunksUseCase:
    """Use case for evaluating chunk quality using all metrics"""
    
    def execute(
        self,
        chunks: List[Chunk],
        original_text: str
    ) -> EvaluationMetrics:
        """
        Evaluate chunks using all evaluation metrics
        
        Args:
            chunks: List of chunks to evaluate
            original_text: Original text from which chunks were extracted
        
        Returns:
            EvaluationMetrics with all calculated scores
        """
        # Calculate Boundary Clarity
        boundary_clarity = BoundaryClarity.calculate(chunks, original_text)
        
        # Calculate Chunk Stickiness
        chunk_stickiness = ChunkStickiness.calculate(chunks)
        
        # Calculate HOPE Metrics
        hope_metrics = HOPEMetrics.calculate(chunks, original_text)
        
        # Create and return aggregated metrics
        return EvaluationMetrics(
            boundary_clarity=boundary_clarity.score,
            chunk_stickiness=chunk_stickiness.score,
            hope_score=hope_metrics.overall_score,
            intrinsic_properties={
                "score": hope_metrics.intrinsic_score
            },
            extrinsic_properties={
                "score": hope_metrics.extrinsic_score,
                "semantic_independence": hope_metrics.extrinsic_score
            },
            coherence_score=hope_metrics.coherence_score
        )
