"""HOPE (Holistic Passage Evaluation) evaluator implementation"""
from typing import List, Optional
import numpy as np
from ...domain.entities.chunk import Chunk
from ...domain.value_objects.hope_metrics import HOPEMetrics


class HOPEEvaluator:
    """Evaluator for HOPE metrics"""
    
    def __init__(self, embedding_model: Optional[object] = None):
        """
        Initialize evaluator
        
        Args:
            embedding_model: Embedding model for semantic similarity calculation
        """
        self.embedding_model = embedding_model
    
    def evaluate(
        self,
        chunks: List[Chunk],
        original_document: str
    ) -> HOPEMetrics:
        """
        Evaluate chunks using HOPE metrics
        
        Args:
            chunks: List of chunks to evaluate
            original_document: Original document text
        
        Returns:
            HOPEMetrics value object
        """
        if not chunks:
            return HOPEMetrics(
                intrinsic_score=0.0,
                extrinsic_score=0.0,
                coherence_score=0.0,
                overall_score=0.0
            )
        
        if self.embedding_model is None:
            # Fallback to heuristics
            return self._evaluate_with_heuristic(chunks, original_document)
        
        # Calculate each component
        intrinsic = self._calculate_intrinsic(chunks)
        extrinsic = self._calculate_extrinsic(chunks)
        coherence = self._calculate_coherence(chunks, original_document)
        
        # Overall score: weighted average
        # Extrinsic is most important (56.2% impact according to HOPE paper)
        overall = (intrinsic * 0.2 + extrinsic * 0.5 + coherence * 0.3)
        
        return HOPEMetrics(
            intrinsic_score=intrinsic,
            extrinsic_score=extrinsic,
            coherence_score=coherence,
            overall_score=overall
        )
    
    def _calculate_intrinsic(self, chunks: List[Chunk]) -> float:
        """
        Calculate intrinsic passage properties
        
        Measures internal properties of each passage (coherence, completeness)
        """
        if not chunks:
            return 0.0
        
        # Get embeddings for chunks
        chunk_embeddings = [
            self.embedding_model.encode(chunk.content) for chunk in chunks
        ]
        
        # For each chunk, measure internal coherence
        intrinsic_scores = []
        for chunk, embedding in zip(chunks, chunk_embeddings):
            # Split chunk into parts and measure similarity
            if len(chunk.content) > 20:
                parts = [
                    chunk.content[:len(chunk.content) // 2],
                    chunk.content[len(chunk.content) // 2:]
                ]
                part_embeddings = [
                    self.embedding_model.encode(part) for part in parts
                ]
                similarity = self._cosine_similarity(
                    part_embeddings[0],
                    part_embeddings[1]
                )
                intrinsic_scores.append(similarity)
            else:
                # Short chunks are considered coherent
                intrinsic_scores.append(1.0)
        
        return float(np.mean(intrinsic_scores)) if intrinsic_scores else 0.0
    
    def _calculate_extrinsic(self, chunks: List[Chunk]) -> float:
        """
        Calculate extrinsic passage properties (semantic independence)
        
        This is critical - semantic independence between passages is essential
        (up to 56.2% impact on factual correctness according to HOPE paper)
        """
        if len(chunks) < 2:
            return 1.0  # Single chunk is perfectly independent
        
        # Get embeddings for all chunks
        chunk_embeddings = [
            self.embedding_model.encode(chunk.content) for chunk in chunks
        ]
        
        # Calculate pairwise similarities between chunks
        similarities = []
        for i in range(len(chunk_embeddings)):
            for j in range(i + 1, len(chunk_embeddings)):
                similarity = self._cosine_similarity(
                    chunk_embeddings[i],
                    chunk_embeddings[j]
                )
                similarities.append(similarity)
        
        # Lower average similarity = higher independence = better extrinsic score
        avg_similarity = np.mean(similarities) if similarities else 0.0
        # Convert similarity to independence score
        independence_score = 1.0 - avg_similarity
        
        return float(max(0.0, min(1.0, independence_score)))
    
    def _calculate_coherence(
        self,
        chunks: List[Chunk],
        document: str
    ) -> float:
        """
        Calculate passages-document coherence
        
        Measures how well chunks maintain coherence with the original document
        """
        if not chunks or not document:
            return 0.0
        
        # Get document embedding
        doc_embedding = self.embedding_model.encode(document)
        
        # Get chunk embeddings
        chunk_embeddings = [
            self.embedding_model.encode(chunk.content) for chunk in chunks
        ]
        
        # Calculate similarity between each chunk and document
        similarities = []
        for chunk_embedding in chunk_embeddings:
            similarity = self._cosine_similarity(chunk_embedding, doc_embedding)
            similarities.append(similarity)
        
        # Higher average similarity = higher coherence
        avg_similarity = np.mean(similarities) if similarities else 0.0
        return float(avg_similarity)
    
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
        original_document: str
    ) -> HOPEMetrics:
        """
        Fallback evaluation using simple heuristics
        """
        if not chunks:
            return HOPEMetrics(
                intrinsic_score=0.0,
                extrinsic_score=0.0,
                coherence_score=0.0,
                overall_score=0.0
            )
        
        # Simple heuristics
        intrinsic = self._heuristic_intrinsic(chunks)
        extrinsic = self._heuristic_extrinsic(chunks)
        coherence = self._heuristic_coherence(chunks, original_document)
        
        overall = (intrinsic * 0.2 + extrinsic * 0.5 + coherence * 0.3)
        
        return HOPEMetrics(
            intrinsic_score=intrinsic,
            extrinsic_score=extrinsic,
            coherence_score=coherence,
            overall_score=overall
        )
    
    def _heuristic_intrinsic(self, chunks: List[Chunk]) -> float:
        """Heuristic intrinsic score"""
        if not chunks:
            return 0.0
        
        avg_length = np.mean([len(chunk.content) for chunk in chunks])
        if avg_length > 100:
            return min(1.0, 0.6 + (avg_length / 2000) * 0.4)
        else:
            return 0.4
    
    def _heuristic_extrinsic(self, chunks: List[Chunk]) -> float:
        """Heuristic extrinsic score"""
        if len(chunks) < 2:
            return 1.0
        
        # More chunks might indicate better separation
        chunk_count = len(chunks)
        if chunk_count >= 3:
            return min(1.0, 0.5 + (chunk_count / 20) * 0.5)
        else:
            return 0.5
    
    def _heuristic_coherence(self, chunks: List[Chunk], document: str) -> float:
        """Heuristic coherence score"""
        if not chunks or not document:
            return 0.0
        
        total_chunk_length = sum(len(chunk.content) for chunk in chunks)
        document_length = len(document)
        
        if document_length > 0:
            coverage_ratio = total_chunk_length / document_length
            return min(1.0, coverage_ratio * 1.2)
        else:
            return 0.0
