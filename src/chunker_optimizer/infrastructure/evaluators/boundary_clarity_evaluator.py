"""Boundary Clarity evaluator implementation"""
from typing import List, Optional
import numpy as np
from ...domain.entities.chunk import Chunk
from ...domain.value_objects.boundary_clarity import BoundaryClarity


class BoundaryClarityEvaluator:
    """Evaluator for Boundary Clarity metric"""
    
    def __init__(self, embedding_model: Optional[object] = None):
        """
        Initialize evaluator
        
        Args:
            embedding_model: Embedding model for semantic similarity calculation
                            (e.g., sentence-transformers model)
        """
        self.embedding_model = embedding_model
    
    def evaluate(
        self,
        chunks: List[Chunk],
        original_text: str
    ) -> BoundaryClarity:
        """
        Evaluate boundary clarity using semantic similarity
        
        Args:
            chunks: List of chunks to evaluate
            original_text: Original text from which chunks were extracted
        
        Returns:
            BoundaryClarity value object
        """
        if len(chunks) < 2:
            return BoundaryClarity(score=1.0)
        
        if self.embedding_model is None:
            # Fallback to simple heuristic if no embedding model
            return self._evaluate_with_heuristic(chunks, original_text)
        
        # Calculate embeddings for chunks
        chunk_embeddings = self._get_chunk_embeddings(chunks)
        
        # Calculate boundary scores
        boundary_scores = []
        for i in range(len(chunks) - 1):
            prev_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # Get embeddings for boundary regions
            prev_end_embedding = self._get_boundary_embedding(
                prev_chunk, chunk_embeddings[i], "end"
            )
            next_start_embedding = self._get_boundary_embedding(
                next_chunk, chunk_embeddings[i + 1], "start"
            )
            
            # Calculate semantic similarity at boundary
            similarity = self._cosine_similarity(
                prev_end_embedding,
                next_start_embedding
            )
            
            # Lower similarity at boundary = clearer boundary
            # Convert similarity (0-1) to clarity score
            # High similarity (similar content) = low clarity (unclear boundary)
            # Low similarity (different content) = high clarity (clear boundary)
            clarity_score = 1.0 - similarity
            boundary_scores.append(clarity_score)
        
        # Average boundary clarity
        avg_score = np.mean(boundary_scores) if boundary_scores else 1.0
        return BoundaryClarity(score=float(avg_score))
    
    def _get_chunk_embeddings(self, chunks: List[Chunk]) -> List[np.ndarray]:
        """Get embeddings for all chunks"""
        embeddings = []
        for chunk in chunks:
            embedding = self.embedding_model.encode(chunk.content)
            embeddings.append(embedding)
        return embeddings
    
    def _get_boundary_embedding(
        self,
        chunk: Chunk,
        chunk_embedding: np.ndarray,
        position: str
    ) -> np.ndarray:
        """
        Get embedding for boundary region of chunk
        
        Args:
            chunk: Chunk to get boundary from
            chunk_embedding: Full chunk embedding
            position: "start" or "end"
        
        Returns:
            Embedding for boundary region
        """
        # For simplicity, use a portion of the chunk near the boundary
        # In practice, could use a sliding window or specific boundary tokens
        if position == "start":
            # Use first portion of chunk
            boundary_text = chunk.content[:len(chunk.content) // 3]
        else:  # "end"
            # Use last portion of chunk
            boundary_text = chunk.content[-len(chunk.content) // 3:]
        
        if boundary_text:
            return self.embedding_model.encode(boundary_text)
        else:
            return chunk_embedding
    
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
    
    def _evaluate_with_heuristic(
        self,
        chunks: List[Chunk],
        original_text: str
    ) -> BoundaryClarity:
        """
        Fallback evaluation using simple heuristics
        
        This is used when no embedding model is available
        """
        if len(chunks) < 2:
            return BoundaryClarity(score=1.0)
        
        # Simple heuristic: check if chunks have reasonable sizes
        # More uniform chunk sizes might indicate better boundaries
        chunk_lengths = [len(chunk.content) for chunk in chunks]
        avg_length = np.mean(chunk_lengths)
        std_length = np.std(chunk_lengths)
        
        # Lower std relative to mean = more uniform = better boundaries
        if avg_length > 0:
            coefficient_of_variation = std_length / avg_length
            # Convert to score (lower CV = higher score)
            score = max(0.0, min(1.0, 1.0 - coefficient_of_variation))
        else:
            score = 0.5
        
        return BoundaryClarity(score=float(score))
