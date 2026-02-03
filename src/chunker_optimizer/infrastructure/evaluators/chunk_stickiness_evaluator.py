"""Chunk Stickiness evaluator implementation"""
from typing import List, Optional
import numpy as np
from ...domain.entities.chunk import Chunk
from ...domain.value_objects.chunk_stickiness import ChunkStickiness


class ChunkStickinessEvaluator:
    """Evaluator for Chunk Stickiness metric"""
    
    def __init__(self, embedding_model: Optional[object] = None):
        """
        Initialize evaluator
        
        Args:
            embedding_model: Embedding model for semantic similarity calculation
        """
        self.embedding_model = embedding_model
    
    def evaluate(self, chunks: List[Chunk]) -> ChunkStickiness:
        """
        Evaluate chunk stickiness - internal coherence of chunks
        
        Args:
            chunks: List of chunks to evaluate
        
        Returns:
            ChunkStickiness value object
        """
        if not chunks:
            return ChunkStickiness(score=0.0)
        
        if self.embedding_model is None:
            # Fallback to simple heuristic
            return self._evaluate_with_heuristic(chunks)
        
        # Calculate stickiness for each chunk
        stickiness_scores = []
        for chunk in chunks:
            stickiness = self._calculate_chunk_stickiness(chunk)
            stickiness_scores.append(stickiness)
        
        # Average stickiness across all chunks
        avg_score = np.mean(stickiness_scores) if stickiness_scores else 0.0
        return ChunkStickiness(score=float(avg_score))
    
    def _calculate_chunk_stickiness(self, chunk: Chunk) -> float:
        """
        Calculate stickiness for a single chunk
        
        Measures how well the content within the chunk sticks together
        by comparing semantic similarity of different parts of the chunk
        
        Args:
            chunk: Chunk to evaluate
        
        Returns:
            Stickiness score (0.0 to 1.0)
        """
        if len(chunk.content) < 10:
            # Very short chunks are considered cohesive
            return 1.0
        
        # Split chunk into segments
        segments = self._split_chunk_into_segments(chunk)
        
        if len(segments) < 2:
            return 1.0
        
        # Get embeddings for segments
        segment_embeddings = [
            self.embedding_model.encode(segment) for segment in segments
        ]
        
        # Calculate pairwise similarities between segments
        similarities = []
        for i in range(len(segment_embeddings)):
            for j in range(i + 1, len(segment_embeddings)):
                similarity = self._cosine_similarity(
                    segment_embeddings[i],
                    segment_embeddings[j]
                )
                similarities.append(similarity)
        
        # Higher average similarity = higher stickiness
        avg_similarity = np.mean(similarities) if similarities else 0.0
        return float(avg_similarity)
    
    def _split_chunk_into_segments(self, chunk: Chunk, num_segments: int = 3) -> List[str]:
        """
        Split chunk into segments for analysis
        
        Args:
            chunk: Chunk to split
            num_segments: Number of segments to create
        
        Returns:
            List of segment texts
        """
        content = chunk.content
        segment_length = len(content) // num_segments
        
        segments = []
        for i in range(num_segments):
            start = i * segment_length
            if i == num_segments - 1:
                # Last segment takes remaining content
                end = len(content)
            else:
                end = (i + 1) * segment_length
            
            segment = content[start:end].strip()
            if segment:
                segments.append(segment)
        
        return segments
    
    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def _evaluate_with_heuristic(self, chunks: List[Chunk]) -> ChunkStickiness:
        """
        Fallback evaluation using simple heuristics
        """
        if not chunks:
            return ChunkStickiness(score=0.0)
        
        # Simple heuristic: longer chunks might have better internal coherence
        avg_length = np.mean([len(chunk.content) for chunk in chunks])
        
        # Normalize to 0-1 range
        if avg_length > 100:
            score = min(1.0, 0.5 + (avg_length / 2000) * 0.5)
        else:
            score = 0.3
        
        return ChunkStickiness(score=float(score))
