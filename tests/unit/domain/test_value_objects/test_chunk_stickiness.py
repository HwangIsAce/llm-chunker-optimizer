"""Tests for ChunkStickiness value object"""
import pytest
from chunker_optimizer.domain.value_objects.chunk_stickiness import ChunkStickiness
from chunker_optimizer.domain.entities.chunk import Chunk


class TestChunkStickiness:
    """Test ChunkStickiness value object"""
    
    def test_create_chunk_stickiness(self):
        """Test creating valid chunk stickiness"""
        stickiness = ChunkStickiness(score=0.8)
        
        assert stickiness.score == 0.8
    
    def test_validation_score_out_of_range(self):
        """Test validation - score out of range"""
        with pytest.raises(ValueError, match="score must be between"):
            ChunkStickiness(score=1.5)
    
    def test_calculate_with_no_chunks(self):
        """Test calculation with no chunks - should return zero"""
        chunks = []
        result = ChunkStickiness.calculate(chunks)
        
        assert result.score == 0.0
    
    def test_calculate_with_chunks(self):
        """Test calculation with chunks"""
        chunks = [
            Chunk(id="1", content="This is a test chunk", start_index=0, end_index=20),
            Chunk(id="2", content="Another chunk here", start_index=20, end_index=38),
        ]
        result = ChunkStickiness.calculate(chunks)
        
        assert 0.0 <= result.score <= 1.0
    
    def test_immutability(self):
        """Test that ChunkStickiness is immutable (frozen dataclass)"""
        stickiness = ChunkStickiness(score=0.8)
        
        with pytest.raises(Exception):  # FrozenInstanceError
            stickiness.score = 0.9
