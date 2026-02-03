"""Tests for OpenAIClient"""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from chunker_optimizer.infrastructure.llm.openai_client import OpenAIClient


class TestOpenAIClient:
    """Test OpenAIClient"""
    
    @pytest.fixture
    def mock_api_key(self):
        """Mock API key"""
        return "test-api-key-12345"
    
    @pytest.fixture
    def client(self, mock_api_key):
        """Create OpenAI client with mock API key"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": mock_api_key}):
            with patch("chunker_optimizer.infrastructure.llm.openai_client.OpenAI"):
                client = OpenAIClient(api_key=mock_api_key)
                return client
    
    def test_init_with_api_key(self, mock_api_key):
        """Test initialization with API key"""
        with patch("chunker_optimizer.infrastructure.llm.openai_client.OpenAI"):
            client = OpenAIClient(api_key=mock_api_key)
            assert client.api_key == mock_api_key
            assert client.model == "gpt-4"
    
    def test_init_with_env_var(self, mock_api_key):
        """Test initialization with environment variable"""
        with patch.dict(os.environ, {"OPENAI_API_KEY": mock_api_key}):
            with patch("chunker_optimizer.infrastructure.llm.openai_client.OpenAI"):
                client = OpenAIClient()
                assert client.api_key == mock_api_key
    
    def test_init_without_api_key(self):
        """Test initialization without API key raises error"""
        with patch.dict(os.environ, {}, clear=True):
            with patch("chunker_optimizer.infrastructure.llm.openai_client.OpenAI"):
                with pytest.raises(ValueError, match="OpenAI API key is required"):
                    OpenAIClient()
    
    def test_optimize_prompt(self, client, mock_api_key):
        """Test prompt optimization"""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Optimized prompt text"
        
        client.client = MagicMock()
        client.client.chat.completions.create = MagicMock(return_value=mock_response)
        
        result = client.optimize_prompt(
            current_prompt="Original prompt",
            optimization_instruction="Improve this prompt"
        )
        
        assert result == "Optimized prompt text"
        client.client.chat.completions.create.assert_called_once()
    
    def test_optimize_prompt_error_handling(self, client):
        """Test error handling in prompt optimization"""
        client.client = MagicMock()
        client.client.chat.completions.create.side_effect = Exception("API Error")
        
        with pytest.raises(RuntimeError, match="Failed to optimize prompt"):
            client.optimize_prompt(
                current_prompt="Original",
                optimization_instruction="Improve"
            )
    
    def test_get_token_count(self, client):
        """Test token count estimation"""
        # Mock tiktoken module
        with patch("tiktoken.encoding_for_model") as mock_encoding:
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = [1, 2, 3, 4, 5]
            mock_encoding.return_value = mock_encoder
            
            count = client.get_token_count("test text")
            assert count == 5
    
    def test_get_token_count_fallback(self, client):
        """Test token count fallback when tiktoken fails"""
        # Mock tiktoken to raise exception
        with patch("tiktoken.encoding_for_model", side_effect=Exception("tiktoken error")):
            # Should use fallback estimation
            count = client.get_token_count("test" * 20)  # 80 characters
            # Fallback: 1 token ≈ 4 characters, so 80 / 4 = 20
            assert count == 20
