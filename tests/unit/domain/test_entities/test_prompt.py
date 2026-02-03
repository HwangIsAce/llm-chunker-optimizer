"""Tests for Prompt entity"""
import pytest
from chunker_optimizer.domain.entities.prompt import Prompt


class TestPrompt:
    """Test Prompt entity"""
    
    def test_create_prompt(self):
        """Test creating a valid prompt"""
        prompt = Prompt(
            id="prompt_1",
            content="Split text into meaningful chunks",
            version=1
        )
        
        assert prompt.id == "prompt_1"
        assert prompt.content == "Split text into meaningful chunks"
        assert prompt.version == 1
    
    def test_prompt_metadata_default(self):
        """Test prompt metadata defaults to empty dict"""
        prompt = Prompt(
            id="prompt_1",
            content="Test prompt",
            version=1
        )
        
        assert prompt.metadata == {}
    
    def test_prompt_validation_empty_id(self):
        """Test prompt validation - empty id"""
        with pytest.raises(ValueError, match="id cannot be empty"):
            Prompt(id="", content="Test", version=1)
    
    def test_prompt_validation_empty_content(self):
        """Test prompt validation - empty content"""
        with pytest.raises(ValueError, match="content cannot be empty"):
            Prompt(id="prompt_1", content="", version=1)
    
    def test_prompt_validation_invalid_version(self):
        """Test prompt validation - invalid version"""
        with pytest.raises(ValueError, match="version must be at least 1"):
            Prompt(id="prompt_1", content="Test", version=0)
    
    def test_increment_version(self):
        """Test incrementing prompt version"""
        prompt = Prompt(
            id="prompt_1",
            content="Test prompt",
            version=1,
            metadata={"key": "value"}
        )
        
        new_prompt = prompt.increment_version()
        
        assert new_prompt.id == prompt.id
        assert new_prompt.content == prompt.content
        assert new_prompt.version == 2
        assert new_prompt.metadata == {"key": "value"}
        # Ensure metadata is copied, not shared
        assert new_prompt.metadata is not prompt.metadata
