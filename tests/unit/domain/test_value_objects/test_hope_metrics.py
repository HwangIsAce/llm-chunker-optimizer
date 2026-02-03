"""Tests for HOPEMetrics value object"""
import pytest
from chunker_optimizer.domain.value_objects.hope_metrics import HOPEMetrics
from chunker_optimizer.domain.entities.chunk import Chunk


class TestHOPEMetrics:
    """Test HOPEMetrics value object"""
    
    def test_create_hope_metrics(self):
        """Test creating valid HOPE metrics"""
        metrics = HOPEMetrics(
            intrinsic_score=0.8,
            extrinsic_score=0.7,
            coherence_score=0.9,
            overall_score=0.8
        )
        
        assert metrics.intrinsic_score == 0.8
        assert metrics.extrinsic_score == 0.7
        assert metrics.coherence_score == 0.9
        assert metrics.overall_score == 0.8
    
    def test_validation_score_out_of_range(self):
        """Test validation - score out of range"""
        with pytest.raises(ValueError, match="must be between"):
            HOPEMetrics(
                intrinsic_score=1.5,  # Invalid
                extrinsic_score=0.7,
                coherence_score=0.9,
                overall_score=0.8
            )
    
    def test_calculate_with_no_chunks(self):
        """Test calculation with no chunks"""
        chunks = []
        result = HOPEMetrics.calculate(chunks, "")
        
        assert result.intrinsic_score == 0.0
        assert result.extrinsic_score == 0.0
        assert result.coherence_score == 0.0
        assert result.overall_score == 0.0
    
    def test_calculate_with_chunks(self):
        """Test calculation with chunks"""
        chunks = [
            Chunk(id="1", content="First chunk content", start_index=0, end_index=20),
            Chunk(id="2", content="Second chunk content", start_index=20, end_index=40),
        ]
        result = HOPEMetrics.calculate(chunks, "First chunk content Second chunk content")
        
        assert 0.0 <= result.intrinsic_score <= 1.0
        assert 0.0 <= result.extrinsic_score <= 1.0
        assert 0.0 <= result.coherence_score <= 1.0
        assert 0.0 <= result.overall_score <= 1.0
    
    def test_overall_score_weighting(self):
        """Test that overall score uses correct weighting"""
        chunks = [
            Chunk(id="1", content="Test" * 100, start_index=0, end_index=400),
        ]
        result = HOPEMetrics.calculate(chunks, "Test" * 100)
        
        # Overall should be weighted: intrinsic*0.2 + extrinsic*0.5 + coherence*0.3
        expected = (
            result.intrinsic_score * 0.2 +
            result.extrinsic_score * 0.5 +
            result.coherence_score * 0.3
        )
        
        assert result.overall_score == pytest.approx(expected, rel=1e-3)
    
    def test_immutability(self):
        """Test that HOPEMetrics is immutable (frozen dataclass)"""
        metrics = HOPEMetrics(
            intrinsic_score=0.8,
            extrinsic_score=0.7,
            coherence_score=0.9,
            overall_score=0.8
        )
        
        with pytest.raises(Exception):  # FrozenInstanceError
            metrics.intrinsic_score = 0.9
