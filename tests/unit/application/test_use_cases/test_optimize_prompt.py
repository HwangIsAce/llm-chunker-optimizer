"""Tests for OptimizePromptUseCase"""
import pytest
from unittest.mock import Mock, MagicMock
from chunker_optimizer.application.use_cases.optimize_prompt import OptimizePromptUseCase
from chunker_optimizer.domain.entities.prompt import Prompt
from chunker_optimizer.domain.entities.evaluation_metrics import EvaluationMetrics
from chunker_optimizer.infrastructure.llm.llm_client import LLMClient


class MockLLMClient(LLMClient):
    """Mock LLM client for testing"""
    
    def __init__(self, return_value: str = "Optimized prompt"):
        self.return_value = return_value
        self.call_count = 0
        self.last_prompt = None
        self.last_instruction = None
    
    def optimize_prompt(self, current_prompt: str, optimization_instruction: str) -> str:
        self.call_count += 1
        self.last_prompt = current_prompt
        self.last_instruction = optimization_instruction
        return self.return_value


class TestOptimizePromptUseCase:
    """Test OptimizePromptUseCase"""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client"""
        return MockLLMClient()
    
    @pytest.fixture
    def use_case(self, mock_llm_client):
        return OptimizePromptUseCase(llm_client=mock_llm_client)
    
    def test_execute_calls_llm_client(self, use_case, mock_llm_client):
        """Test that execute calls LLM client"""
        prompt = Prompt(id="1", content="Original prompt", version=1)
        metrics = EvaluationMetrics(
            boundary_clarity=0.5,
            chunk_stickiness=0.5,
            hope_score=0.5,
            intrinsic_properties={},
            extrinsic_properties={},
            coherence_score=0.5
        )
        
        result = use_case.execute(prompt, metrics, [], 0)
        
        # LLM client should be called
        assert mock_llm_client.call_count == 1
        assert mock_llm_client.last_prompt == "Original prompt"
        assert "Optimization Results" in mock_llm_client.last_instruction or "Iteration" in mock_llm_client.last_instruction
        
        # Result should have incremented version
        assert result.version == 2
        assert result.content == "Optimized prompt"
        assert result.id == prompt.id
    
    def test_execute_increments_version(self, use_case):
        """Test that version is incremented"""
        prompt = Prompt(id="1", content="Original", version=5)
        metrics = EvaluationMetrics(
            boundary_clarity=0.8,
            chunk_stickiness=0.8,
            hope_score=0.8,
            intrinsic_properties={},
            extrinsic_properties={},
            coherence_score=0.8
        )
        
        result = use_case.execute(prompt, metrics, [], 0)
        
        assert result.version == 6
    
    def test_execute_preserves_metadata(self, use_case):
        """Test that metadata is preserved and extended"""
        prompt = Prompt(
            id="1",
            content="Original",
            version=1,
            metadata={"key": "value"}
        )
        metrics = EvaluationMetrics(
            boundary_clarity=0.8,
            chunk_stickiness=0.8,
            hope_score=0.8,
            intrinsic_properties={},
            extrinsic_properties={},
            coherence_score=0.8
        )
        
        result = use_case.execute(prompt, metrics, [], 0)
        
        assert result.metadata["key"] == "value"
        assert "iteration" in result.metadata
        assert "previous_metrics" in result.metadata
        assert result.metadata["iteration"] == 0
