"""Tests for Chunk entity"""
import pytest
from chunker_optimizer.domain.entities.chunk import Chunk


class TestChunk:
    """Test Chunk entity"""
    
    def test_create_chunk(self):
        """Test creating a valid chunk"""
        chunk = Chunk(
            id="chunk_1",
            content="This is a test chunk",
            start_index=0,
            end_index=20
        )
        
        assert chunk.id == "chunk_1"
        assert chunk.content == "This is a test chunk"
        assert chunk.start_index == 0
        assert chunk.end_index == 20
        assert chunk.length == 20
        assert len(chunk) == 20
    
    def test_chunk_length_property(self):
        """Test chunk length property"""
        chunk = Chunk(
            id="chunk_1",
            content="Test",
            start_index=0,
            end_index=4
        )
        
        assert chunk.length == 4
    
    def test_chunk_metadata_default(self):
        """Test chunk metadata defaults to empty dict"""
        chunk = Chunk(
            id="chunk_1",
            content="Test",
            start_index=0,
            end_index=4
        )
        
        assert chunk.metadata == {}
    
    def test_chunk_validation_empty_id(self):
        """Test chunk validation - empty id"""
        with pytest.raises(ValueError, match="id cannot be empty"):
            Chunk(id="", content="Test", start_index=0, end_index=4)
    
    def test_chunk_validation_empty_content(self):
        """Test chunk validation - empty content"""
        with pytest.raises(ValueError, match="content cannot be empty"):
            Chunk(id="chunk_1", content="", start_index=0, end_index=4)
    
    def test_chunk_validation_negative_start_index(self):
        """Test chunk validation - negative start_index"""
        with pytest.raises(ValueError, match="start_index must be non-negative"):
            Chunk(id="chunk_1", content="Test", start_index=-1, end_index=4)
    
    def test_chunk_validation_end_index_not_greater(self):
        """Test chunk validation - end_index not greater than start_index"""
        with pytest.raises(ValueError, match="end_index must be greater than start_index"):
            Chunk(id="chunk_1", content="Test", start_index=5, end_index=5)
        
        with pytest.raises(ValueError, match="end_index must be greater than start_index"):
            Chunk(id="chunk_1", content="Test", start_index=5, end_index=3)
