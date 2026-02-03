"""Tests for EvaluationMetrics entity"""
import pytest
from chunker_optimizer.domain.entities.evaluation_metrics import EvaluationMetrics


class TestEvaluationMetrics:
    """Test EvaluationMetrics entity"""
    
    def test_create_evaluation_metrics(self):
        """Test creating valid evaluation metrics"""
        metrics = EvaluationMetrics(
            boundary_clarity=0.8,
            chunk_stickiness=0.7,
            hope_score=0.9,
            intrinsic_properties={"score": 0.8},
            extrinsic_properties={"score": 0.7},
            coherence_score=0.85
        )
        
        assert metrics.boundary_clarity == 0.8
        assert metrics.chunk_stickiness == 0.7
        assert metrics.hope_score == 0.9
        assert metrics.coherence_score == 0.85
    
    def test_overall_score_calculation(self):
        """Test overall score calculation"""
        metrics = EvaluationMetrics(
            boundary_clarity=0.8,
            chunk_stickiness=0.7,
            hope_score=0.9,
            intrinsic_properties={},
            extrinsic_properties={},
            coherence_score=0.85
        )
        
        expected = 0.8 * 0.3 + 0.7 * 0.3 + 0.9 * 0.4
        assert metrics.overall_score == pytest.approx(expected, rel=1e-3)
    
    def test_meets_threshold_true(self):
        """Test meets_threshold - threshold met"""
        metrics = EvaluationMetrics(
            boundary_clarity=0.9,
            chunk_stickiness=0.9,
            hope_score=0.9,
            intrinsic_properties={},
            extrinsic_properties={},
            coherence_score=0.9
        )
        
        assert metrics.meets_threshold(0.8) is True
        assert metrics.meets_threshold(0.9) is True
    
    def test_meets_threshold_false(self):
        """Test meets_threshold - threshold not met"""
        metrics = EvaluationMetrics(
            boundary_clarity=0.5,
            chunk_stickiness=0.5,
            hope_score=0.5,
            intrinsic_properties={},
            extrinsic_properties={},
            coherence_score=0.5
        )
        
        assert metrics.meets_threshold(0.8) is False
    
    def test_validation_score_out_of_range(self):
        """Test validation - score out of range"""
        with pytest.raises(ValueError):
            EvaluationMetrics(
                boundary_clarity=1.5,  # Invalid
                chunk_stickiness=0.7,
                hope_score=0.9,
                intrinsic_properties={},
                extrinsic_properties={},
                coherence_score=0.85
            )
    
    def test_metadata_defaults(self):
        """Test metadata defaults to empty dict"""
        metrics = EvaluationMetrics(
            boundary_clarity=0.8,
            chunk_stickiness=0.7,
            hope_score=0.9,
            intrinsic_properties=None,
            extrinsic_properties=None,
            coherence_score=0.85
        )
        
        assert metrics.intrinsic_properties == {}
        assert metrics.extrinsic_properties == {}
    
    def test_to_dict(self):
        """Test converting metrics to dictionary"""
        metrics = EvaluationMetrics(
            boundary_clarity=0.8,
            chunk_stickiness=0.7,
            hope_score=0.9,
            intrinsic_properties={"key": 0.5},
            extrinsic_properties={"key": 0.6},
            coherence_score=0.85
        )
        
        result = metrics.to_dict()
        
        assert result["boundary_clarity"] == 0.8
        assert result["chunk_stickiness"] == 0.7
        assert result["hope_score"] == 0.9
        assert result["coherence_score"] == 0.85
        assert "overall_score" in result
        assert result["overall_score"] == metrics.overall_score
    
    def test_threshold_validation(self):
        """Test threshold validation"""
        metrics = EvaluationMetrics(
            boundary_clarity=0.8,
            chunk_stickiness=0.7,
            hope_score=0.9,
            intrinsic_properties={},
            extrinsic_properties={},
            coherence_score=0.85
        )
        
        with pytest.raises(ValueError, match="threshold must be between"):
            metrics.meets_threshold(1.5)
