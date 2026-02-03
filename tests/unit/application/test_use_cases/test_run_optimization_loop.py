"""Tests for RunOptimizationLoopUseCase"""
import pytest
from unittest.mock import Mock, MagicMock
from chunker_optimizer.application.use_cases.run_optimization_loop import (
    RunOptimizationLoopUseCase,
    OptimizationConfig,
    OptimizationResult
)
from chunker_optimizer.application.use_cases.evaluate_chunks import EvaluateChunksUseCase
from chunker_optimizer.application.use_cases.optimize_prompt import OptimizePromptUseCase
from chunker_optimizer.domain.entities.chunk import Chunk
from chunker_optimizer.domain.entities.prompt import Prompt
from chunker_optimizer.domain.entities.evaluation_metrics import EvaluationMetrics
from chunker_optimizer.domain.entities.chunking_context import ChunkingContext
from chunker_optimizer.domain.entities.document_context import (
    DocumentEnrichment,
    ParsedDocument
)
from chunker_optimizer.infrastructure.llm.llm_client import LLMClient


class MockLLMClient(LLMClient):
    """Mock LLM client"""
    
    def optimize_prompt(self, current_prompt: str, optimization_instruction: str) -> str:
        return f"Improved {current_prompt}"


class TestRunOptimizationLoopUseCase:
    """Test RunOptimizationLoopUseCase"""
    
    @pytest.fixture
    def mock_evaluate_use_case(self):
        """Mock evaluate use case"""
        use_case = Mock(spec=EvaluateChunksUseCase)
        use_case.execute = MagicMock(
            side_effect=[
                EvaluationMetrics(0.5, 0.5, 0.5, {}, {}, 0.5),  # Iteration 0
                EvaluationMetrics(0.7, 0.7, 0.7, {}, {}, 0.7),  # Iteration 1
                EvaluationMetrics(0.9, 0.9, 0.9, {}, {}, 0.9),  # Iteration 2
            ]
        )
        return use_case
    
    @pytest.fixture
    def mock_optimize_use_case(self):
        """Mock optimize use case"""
        use_case = Mock(spec=OptimizePromptUseCase)
        use_case.execute = MagicMock(
            side_effect=lambda p, m, c, i: Prompt(
                id=p.id,
                content=f"Improved {i}",
                version=p.version + 1
            )
        )
        return use_case
    
    @pytest.fixture
    def mock_chunking_function(self):
        """Mock chunking function"""
        return MagicMock(return_value=[
            Chunk(id="1", content="chunk", start_index=0, end_index=5)
        ])
    
    @pytest.fixture
    def chunking_context(self):
        """Create chunking context"""
        parsed_doc = ParsedDocument(
            raw_content="test document",
            parsed_elements=[]
        )
        enrichment = DocumentEnrichment(parsed_document=parsed_doc)
        
        return ChunkingContext(
            document_enrichment=enrichment,
            prompt=Prompt(id="test", content="Initial", version=1),
            chunk_unit="page"
        )
    
    def test_loop_stops_when_threshold_met(
        self,
        mock_evaluate_use_case,
        mock_optimize_use_case,
        mock_chunking_function,
        chunking_context
    ):
        """Test that loop stops when threshold is met"""
        use_case = RunOptimizationLoopUseCase(
            evaluate_use_case=mock_evaluate_use_case,
            optimize_use_case=mock_optimize_use_case,
            chunking_function=mock_chunking_function
        )
        
        prompt = chunking_context.prompt
        config = OptimizationConfig(
            max_iterations=10,
            threshold=0.8
        )
        
        result = use_case.execute(prompt, chunking_context, config)
        
        assert result.converged is True
        assert result.iterations <= 3
        assert result.final_metrics.overall_score >= 0.8
    
    def test_loop_stops_at_max_iterations(
        self,
        mock_evaluate_use_case,
        mock_optimize_use_case,
        mock_chunking_function,
        chunking_context
    ):
        """Test that loop stops at max iterations"""
        # Always return low score
        mock_evaluate_use_case.execute = MagicMock(
            return_value=EvaluationMetrics(
                0.5, 0.5, 0.5, {}, {}, 0.5
            )
        )
        
        use_case = RunOptimizationLoopUseCase(
            evaluate_use_case=mock_evaluate_use_case,
            optimize_use_case=mock_optimize_use_case,
            chunking_function=mock_chunking_function
        )
        
        prompt = chunking_context.prompt
        config = OptimizationConfig(
            max_iterations=3,
            threshold=0.8
        )
        
        result = use_case.execute(prompt, chunking_context, config)
        
        assert result.converged is False
        assert result.iterations == 3
    
    def test_early_stopping(
        self,
        mock_evaluate_use_case,
        mock_optimize_use_case,
        mock_chunking_function,
        chunking_context
    ):
        """Test early stopping when no improvement"""
        # Return same score repeatedly
        mock_evaluate_use_case.execute = MagicMock(
            return_value=EvaluationMetrics(
                0.5, 0.5, 0.5, {}, {}, 0.5
            )
        )
        
        use_case = RunOptimizationLoopUseCase(
            evaluate_use_case=mock_evaluate_use_case,
            optimize_use_case=mock_optimize_use_case,
            chunking_function=mock_chunking_function
        )
        
        prompt = chunking_context.prompt
        config = OptimizationConfig(
            max_iterations=10,
            threshold=0.8,
            early_stopping_patience=2
        )
        
        result = use_case.execute(prompt, chunking_context, config)
        
        # Should stop early due to no improvement
        assert result.converged is False
        assert result.iterations <= config.early_stopping_patience + 1
    
    def test_history_recording(
        self,
        mock_evaluate_use_case,
        mock_optimize_use_case,
        mock_chunking_function,
        chunking_context
    ):
        """Test that history is recorded correctly"""
        use_case = RunOptimizationLoopUseCase(
            evaluate_use_case=mock_evaluate_use_case,
            optimize_use_case=mock_optimize_use_case,
            chunking_function=mock_chunking_function
        )
        
        prompt = chunking_context.prompt
        config = OptimizationConfig(max_iterations=2, threshold=0.8)
        
        result = use_case.execute(prompt, chunking_context, config)
        
        assert len(result.history) == result.iterations
        assert result.history[0]["iteration"] == 0
        assert "metrics" in result.history[0]
        assert "chunk_count" in result.history[0]


class TestOptimizationConfig:
    """Test OptimizationConfig"""
    
    def test_valid_config(self):
        """Test creating valid config"""
        config = OptimizationConfig(
            max_iterations=10,
            threshold=0.8,
            improvement_threshold=0.01,
            early_stopping_patience=3
        )
        
        assert config.max_iterations == 10
        assert config.threshold == 0.8
    
    def test_validation_max_iterations(self):
        """Test validation - max_iterations"""
        with pytest.raises(ValueError, match="max_iterations must be at least 1"):
            OptimizationConfig(max_iterations=0)
    
    def test_validation_threshold(self):
        """Test validation - threshold"""
        with pytest.raises(ValueError, match="threshold must be between"):
            OptimizationConfig(threshold=1.5)
    
    def test_validation_improvement_threshold(self):
        """Test validation - improvement_threshold"""
        with pytest.raises(ValueError, match="improvement_threshold must be between"):
            OptimizationConfig(improvement_threshold=1.5)
    
    def test_validation_early_stopping_patience(self):
        """Test validation - early_stopping_patience"""
        with pytest.raises(ValueError, match="early_stopping_patience must be at least 1"):
            OptimizationConfig(early_stopping_patience=0)
