"""Tests for EvaluateChunksUseCase"""
import pytest
from chunker_optimizer.application.use_cases.evaluate_chunks import EvaluateChunksUseCase
from chunker_optimizer.domain.entities.chunk import Chunk


class TestEvaluateChunksUseCase:
    """Test EvaluateChunksUseCase"""
    
    def test_execute_returns_evaluation_metrics(self):
        """Test that execute returns evaluation metrics"""
        use_case = EvaluateChunksUseCase()
        
        chunks = [
            Chunk(id="1", content="test chunk", start_index=0, end_index=10)
        ]
        original_text = "test chunk"
        
        result = use_case.execute(chunks, original_text)
        
        assert result.boundary_clarity >= 0.0
        assert result.chunk_stickiness >= 0.0
        assert result.hope_score >= 0.0
        assert result.overall_score >= 0.0
        assert result.coherence_score >= 0.0
    
    def test_execute_with_multiple_chunks(self):
        """Test evaluation with multiple chunks"""
        use_case = EvaluateChunksUseCase()
        
        chunks = [
            Chunk(id="1", content="First chunk", start_index=0, end_index=11),
            Chunk(id="2", content="Second chunk", start_index=11, end_index=23),
        ]
        original_text = "First chunk Second chunk"
        
        result = use_case.execute(chunks, original_text)
        
        assert result.boundary_clarity >= 0.0
        assert result.chunk_stickiness >= 0.0
        assert result.hope_score >= 0.0
        assert isinstance(result.intrinsic_properties, dict)
        assert isinstance(result.extrinsic_properties, dict)
        assert "score" in result.intrinsic_properties
        assert "score" in result.extrinsic_properties
    
    def test_execute_with_empty_chunks(self):
        """Test evaluation with empty chunks"""
        use_case = EvaluateChunksUseCase()
        
        chunks = []
        original_text = "test"
        
        result = use_case.execute(chunks, original_text)
        
        # Should still return valid metrics (all zeros)
        assert result.boundary_clarity == 1.0  # Single/no chunk case
        assert result.chunk_stickiness == 0.0
        assert result.hope_score == 0.0
