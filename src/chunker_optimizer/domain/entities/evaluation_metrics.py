"""Evaluation metrics entity for chunk quality assessment"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics for chunks"""
    
    boundary_clarity: float
    chunk_stickiness: float
    hope_score: float
    intrinsic_properties: Dict[str, float]
    extrinsic_properties: Dict[str, float]
    coherence_score: float
    
    def __post_init__(self):
        """Validate metrics values"""
        # Ensure all scores are between 0 and 1
        for attr_name in ['boundary_clarity', 'chunk_stickiness', 'hope_score', 'coherence_score']:
            value = getattr(self, attr_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{attr_name} must be between 0.0 and 1.0, got {value}")
        
        # Ensure dictionaries are not None
        if self.intrinsic_properties is None:
            self.intrinsic_properties = {}
        if self.extrinsic_properties is None:
            self.extrinsic_properties = {}
    
    @property
    def overall_score(self) -> float:
        """
        Calculate weighted overall score
        
        Weights:
        - Boundary Clarity: 0.3
        - Chunk Stickiness: 0.3
        - HOPE Score: 0.4
        """
        return (
            self.boundary_clarity * 0.3 +
            self.chunk_stickiness * 0.3 +
            self.hope_score * 0.4
        )
    
    def meets_threshold(self, threshold: float) -> bool:
        """
        Check if metrics meet the threshold
        
        Args:
            threshold: Minimum overall score required (0.0 to 1.0)
        
        Returns:
            True if overall_score >= threshold
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}")
        
        return self.overall_score >= threshold
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary"""
        return {
            "boundary_clarity": self.boundary_clarity,
            "chunk_stickiness": self.chunk_stickiness,
            "hope_score": self.hope_score,
            "intrinsic_properties": self.intrinsic_properties,
            "extrinsic_properties": self.extrinsic_properties,
            "coherence_score": self.coherence_score,
            "overall_score": self.overall_score
        }
