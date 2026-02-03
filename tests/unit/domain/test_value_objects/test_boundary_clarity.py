"""Tests for BoundaryClarity value object"""
import pytest
from chunker_optimizer.domain.value_objects.boundary_clarity import BoundaryClarity
from chunker_optimizer.domain.entities.chunk import Chunk


class TestBoundaryClarity:
    """Test BoundaryClarity value object"""
    
    def test_create_boundary_clarity(self):
        """Test creating valid boundary clarity"""
        clarity = BoundaryClarity(score=0.8)
        
        assert clarity.score == 0.8
    
    def test_validation_score_out_of_range(self):
        """Test validation - score out of range"""
        with pytest.raises(ValueError, match="score must be between"):
            BoundaryClarity(score=1.5)
        
        with pytest.raises(ValueError, match="score must be between"):
            BoundaryClarity(score=-0.1)
    
    def test_calculate_with_single_chunk(self):
        """Test calculation with single chunk - should return perfect score"""
        chunks = [
            Chunk(id="1", content="test", start_index=0, end_index=4)
        ]
        result = BoundaryClarity.calculate(chunks, "test")
        
        assert result.score == 1.0
    
    def test_calculate_with_no_chunks(self):
        """Test calculation with no chunks - should return perfect score"""
        chunks = []
        result = BoundaryClarity.calculate(chunks, "test")
        
        assert result.score == 1.0
    
    def test_calculate_with_multiple_chunks(self):
        """Test calculation with multiple chunks"""
        chunks = [
            Chunk(id="1", content="First", start_index=0, end_index=5),
            Chunk(id="2", content="Second", start_index=5, end_index=11),
        ]
        result = BoundaryClarity.calculate(chunks, "FirstSecond")
        
        assert 0.0 <= result.score <= 1.0
    
    def test_immutability(self):
        """Test that BoundaryClarity is immutable (frozen dataclass)"""
        clarity = BoundaryClarity(score=0.8)
        
        with pytest.raises(Exception):  # FrozenInstanceError
            clarity.score = 0.9
