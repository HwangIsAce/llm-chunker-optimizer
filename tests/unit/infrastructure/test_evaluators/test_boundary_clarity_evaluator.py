"""Tests for BoundaryClarityEvaluator"""
import pytest
import numpy as np
from chunker_optimizer.infrastructure.evaluators.boundary_clarity_evaluator import (
    BoundaryClarityEvaluator
)
from chunker_optimizer.domain.entities.chunk import Chunk


class MockEmbeddingModel:
    """Mock embedding model for testing"""
    
    def encode(self, text: str):
        """Return mock embedding"""
        # Return deterministic embedding based on text length
        # This is a simple mock - real embeddings would be more complex
        np.random.seed(hash(text) % 2**32)
        return np.random.rand(384).astype(np.float32)


class TestBoundaryClarityEvaluator:
    """Test BoundaryClarityEvaluator"""
    
    @pytest.fixture
    def embedding_model(self):
        """Create mock embedding model"""
        return MockEmbeddingModel()
    
    @pytest.fixture
    def evaluator_with_model(self, embedding_model):
        """Create evaluator with embedding model"""
        return BoundaryClarityEvaluator(embedding_model=embedding_model)
    
    @pytest.fixture
    def evaluator_without_model(self):
        """Create evaluator without embedding model"""
        return BoundaryClarityEvaluator(embedding_model=None)
    
    def test_evaluate_single_chunk(self, evaluator_without_model):
        """Test evaluation with single chunk"""
        chunks = [
            Chunk(id="1", content="test", start_index=0, end_index=4)
        ]
        result = evaluator_without_model.evaluate(chunks, "test")
        
        assert result.score == 1.0
    
    def test_evaluate_no_chunks(self, evaluator_without_model):
        """Test evaluation with no chunks"""
        chunks = []
        result = evaluator_without_model.evaluate(chunks, "test")
        
        assert result.score == 1.0
    
    def test_evaluate_with_heuristic(self, evaluator_without_model):
        """Test evaluation using heuristic (no embedding model)"""
        chunks = [
            Chunk(id="1", content="First chunk", start_index=0, end_index=11),
            Chunk(id="2", content="Second chunk", start_index=11, end_index=23),
        ]
        result = evaluator_without_model.evaluate(chunks, "First chunk Second chunk")
        
        assert 0.0 <= result.score <= 1.0
    
    def test_evaluate_with_embedding_model(self, evaluator_with_model):
        """Test evaluation with embedding model"""
        chunks = [
            Chunk(id="1", content="First chunk", start_index=0, end_index=11),
            Chunk(id="2", content="Second chunk", start_index=11, end_index=23),
        ]
        result = evaluator_with_model.evaluate(chunks, "First chunk Second chunk")
        
        assert 0.0 <= result.score <= 1.0
    
    def test_cosine_similarity(self, evaluator_with_model):
        """Test cosine similarity calculation"""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        
        similarity = evaluator_with_model._cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(1.0, rel=1e-3)
        
        vec3 = np.array([0.0, 1.0, 0.0])
        similarity = evaluator_with_model._cosine_similarity(vec1, vec3)
        assert similarity == pytest.approx(0.0, rel=1e-3)
