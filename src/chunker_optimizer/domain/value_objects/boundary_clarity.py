"""Boundary Clarity value object from MoC paper"""
from dataclasses import dataclass
from typing import List
from ..entities.chunk import Chunk


@dataclass(frozen=True)
class BoundaryClarity:
    """
    Boundary Clarity metric from MoC paper
    
    Measures semantic coherence at chunk boundaries.
    Higher scores indicate clearer boundaries between chunks.
    """
    
    score: float  # 0.0 to 1.0
    
    def __post_init__(self):
        """Validate boundary clarity score"""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be between 0.0 and 1.0, got {self.score}")
    
    @classmethod
    def calculate(cls, chunks: List[Chunk], original_text: str) -> "BoundaryClarity":
        """
        Calculate boundary clarity based on semantic coherence at chunk boundaries
        
        Args:
            chunks: List of chunks to evaluate
            original_text: Original text from which chunks were extracted
        
        Returns:
            BoundaryClarity value object
        """
        if len(chunks) < 2:
            # Single chunk or no chunks - perfect boundary clarity
            return cls(score=1.0)
        
        # TODO: Implement actual algorithm from MoC paper
        # This should check semantic coherence at boundaries using embeddings
        # For now, return a placeholder score
        # The actual implementation will be in the infrastructure layer
        
        # Placeholder: simple heuristic based on chunk count
        # More chunks might indicate better boundaries (but this is simplified)
        avg_chunk_length = sum(len(chunk.content) for chunk in chunks) / len(chunks)
        text_length = len(original_text)
        
        # Simple heuristic: if chunks are reasonably sized, assume good boundaries
        if avg_chunk_length > 0 and text_length > 0:
            chunk_ratio = avg_chunk_length / text_length
            # Normalize to 0-1 range (this is a placeholder)
            placeholder_score = min(1.0, max(0.0, chunk_ratio * 10))
        else:
            placeholder_score = 0.5
        
        return cls(score=placeholder_score)
