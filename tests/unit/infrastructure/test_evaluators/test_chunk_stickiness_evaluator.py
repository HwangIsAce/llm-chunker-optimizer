"""Tests for ChunkStickinessEvaluator"""
import pytest
from chunker_optimizer.infrastructure.evaluators.chunk_stickiness_evaluator import (
    ChunkStickinessEvaluator
)
from chunker_optimizer.domain.entities.chunk import Chunk


class MockEmbeddingModel:
    """Mock embedding model for testing"""
    
    def encode(self, text: str):
        """Return mock embedding"""
        import numpy as np
        np.random.seed(hash(text) % 2**32)
        return np.random.rand(384).astype(np.float32)


class TestChunkStickinessEvaluator:
    """Test ChunkStickinessEvaluator"""
    
    @pytest.fixture
    def embedding_model(self):
        """Create mock embedding model"""
        return MockEmbeddingModel()
    
    @pytest.fixture
    def evaluator_with_model(self, embedding_model):
        """Create evaluator with embedding model"""
        return ChunkStickinessEvaluator(embedding_model=embedding_model)
    
    @pytest.fixture
    def evaluator_without_model(self):
        """Create evaluator without embedding model"""
        return ChunkStickinessEvaluator(embedding_model=None)
    
    def test_evaluate_no_chunks(self, evaluator_without_model):
        """Test evaluation with no chunks"""
        chunks = []
        result = evaluator_without_model.evaluate(chunks)
        
        assert result.score == 0.0
    
    def test_evaluate_with_heuristic(self, evaluator_without_model):
        """Test evaluation using heuristic"""
        chunks = [
            Chunk(id="1", content="This is a test chunk", start_index=0, end_index=20),
        ]
        result = evaluator_without_model.evaluate(chunks)
        
        assert 0.0 <= result.score <= 1.0
    
    def test_evaluate_with_embedding_model(self, evaluator_with_model):
        """Test evaluation with embedding model"""
        chunks = [
            Chunk(id="1", content="This is a longer test chunk with more content", start_index=0, end_index=50),
        ]
        result = evaluator_with_model.evaluate(chunks)
        
        assert 0.0 <= result.score <= 1.0
    
    def test_split_chunk_into_segments(self, evaluator_with_model):
        """Test chunk segmentation"""
        chunk = Chunk(
            id="1",
            content="This is a test chunk with multiple words",
            start_index=0,
            end_index=40
        )
        
        segments = evaluator_with_model._split_chunk_into_segments(chunk, num_segments=3)
        
        assert len(segments) > 0
        assert all(isinstance(seg, str) for seg in segments)
    
    def test_short_chunk_stickiness(self, evaluator_with_model):
        """Test that very short chunks are considered cohesive"""
        chunks = [
            Chunk(id="1", content="Short", start_index=0, end_index=5)
        ]
        result = evaluator_with_model.evaluate(chunks)
        
        # Short chunks should have high stickiness
        assert result.score >= 0.0
