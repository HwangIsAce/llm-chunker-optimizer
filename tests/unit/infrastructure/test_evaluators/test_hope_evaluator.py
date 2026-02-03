"""Tests for HOPEEvaluator"""
import pytest
from chunker_optimizer.infrastructure.evaluators.hope_evaluator import HOPEEvaluator
from chunker_optimizer.domain.entities.chunk import Chunk


class MockEmbeddingModel:
    """Mock embedding model for testing"""
    
    def encode(self, text: str):
        """Return mock embedding"""
        import numpy as np
        np.random.seed(hash(text) % 2**32)
        return np.random.rand(384).astype(np.float32)


class TestHOPEEvaluator:
    """Test HOPEEvaluator"""
    
    @pytest.fixture
    def embedding_model(self):
        """Create mock embedding model"""
        return MockEmbeddingModel()
    
    @pytest.fixture
    def evaluator_with_model(self, embedding_model):
        """Create evaluator with embedding model"""
        return HOPEEvaluator(embedding_model=embedding_model)
    
    @pytest.fixture
    def evaluator_without_model(self):
        """Create evaluator without embedding model"""
        return HOPEEvaluator(embedding_model=None)
    
    def test_evaluate_no_chunks(self, evaluator_without_model):
        """Test evaluation with no chunks"""
        chunks = []
        result = evaluator_without_model.evaluate(chunks, "")
        
        assert result.intrinsic_score == 0.0
        assert result.extrinsic_score == 0.0
        assert result.coherence_score == 0.0
        assert result.overall_score == 0.0
    
    def test_evaluate_single_chunk(self, evaluator_without_model):
        """Test evaluation with single chunk"""
        chunks = [
            Chunk(id="1", content="test chunk", start_index=0, end_index=10)
        ]
        result = evaluator_without_model.evaluate(chunks, "test chunk")
        
        assert result.extrinsic_score == 1.0  # Single chunk is independent
        assert 0.0 <= result.intrinsic_score <= 1.0
        assert 0.0 <= result.coherence_score <= 1.0
    
    def test_evaluate_with_heuristic(self, evaluator_without_model):
        """Test evaluation using heuristic"""
        chunks = [
            Chunk(id="1", content="First chunk", start_index=0, end_index=11),
            Chunk(id="2", content="Second chunk", start_index=11, end_index=23),
        ]
        result = evaluator_without_model.evaluate(
            chunks,
            "First chunk Second chunk"
        )
        
        assert 0.0 <= result.intrinsic_score <= 1.0
        assert 0.0 <= result.extrinsic_score <= 1.0
        assert 0.0 <= result.coherence_score <= 1.0
        assert 0.0 <= result.overall_score <= 1.0
    
    def test_evaluate_with_embedding_model(self, evaluator_with_model):
        """Test evaluation with embedding model"""
        chunks = [
            Chunk(id="1", content="First chunk", start_index=0, end_index=11),
            Chunk(id="2", content="Second chunk", start_index=11, end_index=23),
        ]
        result = evaluator_with_model.evaluate(
            chunks,
            "First chunk Second chunk"
        )
        
        assert 0.0 <= result.intrinsic_score <= 1.0
        assert 0.0 <= result.extrinsic_score <= 1.0
        assert 0.0 <= result.coherence_score <= 1.0
        assert 0.0 <= result.overall_score <= 1.0
    
    def test_overall_score_weighting(self, evaluator_without_model):
        """Test that overall score uses correct weighting"""
        chunks = [
            Chunk(id="1", content="Test" * 50, start_index=0, end_index=200),
        ]
        result = evaluator_without_model.evaluate(chunks, "Test" * 50)
        
        # Overall should be weighted: intrinsic*0.2 + extrinsic*0.5 + coherence*0.3
        expected = (
            result.intrinsic_score * 0.2 +
            result.extrinsic_score * 0.5 +
            result.coherence_score * 0.3
        )
        
        assert result.overall_score == pytest.approx(expected, rel=1e-3)
    
    def test_extrinsic_independence(self, evaluator_without_model):
        """Test that extrinsic score measures independence"""
        # Multiple chunks should have lower extrinsic score than single chunk
        single_chunk = [
            Chunk(id="1", content="test", start_index=0, end_index=4)
        ]
        multiple_chunks = [
            Chunk(id="1", content="test1", start_index=0, end_index=5),
            Chunk(id="2", content="test2", start_index=5, end_index=10),
        ]
        
        single_result = evaluator_without_model.evaluate(single_chunk, "test")
        multiple_result = evaluator_without_model.evaluate(multiple_chunks, "test1 test2")
        
        # Single chunk should have perfect independence
        assert single_result.extrinsic_score == 1.0
        # Multiple chunks might have lower independence (but still valid)
        assert 0.0 <= multiple_result.extrinsic_score <= 1.0
