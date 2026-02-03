"""Chunk Stickiness value object from MoC paper (2503.09600)"""
from dataclasses import dataclass
from typing import List
from ..entities.chunk import Chunk


@dataclass(frozen=True)
class ChunkStickiness:
    """
    Chunk Stickiness metric from MoC paper (2503.09600)
    
    According to the MoC paper:
    - Measures how well content within each chunk sticks together semantically
    - Higher scores indicate better internal coherence within chunks
    - Essential for ensuring chunks are semantically cohesive units
    
    The actual calculation is implemented in ChunkStickinessEvaluator
    in the infrastructure layer using semantic similarity between segments.
    """
    
    score: float  # 0.0 to 1.0
    
    def __post_init__(self):
        """Validate chunk stickiness score"""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be between 0.0 and 1.0, got {self.score}")
    
    @classmethod
    def calculate(cls, chunks: List[Chunk]) -> "ChunkStickiness":
        """
        Calculate chunk stickiness - how well content within each chunk sticks together
        
        Note: This is a placeholder implementation. The actual algorithm from
        the MoC paper is implemented in ChunkStickinessEvaluator in the
        infrastructure layer, which uses semantic similarity between segments
        within each chunk.
        
        Args:
            chunks: List of chunks to evaluate
        
        Returns:
            ChunkStickiness value object
        """
        if not chunks:
            return cls(score=0.0)
        
        # Placeholder: simple heuristic
        # The actual implementation using semantic similarity is in
        # ChunkStickinessEvaluator in the infrastructure layer
        
        # Placeholder: simple heuristic based on chunk content
        # Longer chunks might have better internal coherence (but this is simplified)
        total_length = sum(len(chunk.content) for chunk in chunks)
        avg_length = total_length / len(chunks) if chunks else 0
        
        # Simple heuristic: if chunks have reasonable length, assume good stickiness
        # This is a placeholder - actual implementation will use semantic similarity
        if avg_length > 50:  # Reasonable minimum length
            placeholder_score = min(1.0, 0.5 + (avg_length / 1000) * 0.5)
        else:
            placeholder_score = 0.3
        
        return cls(score=placeholder_score)
