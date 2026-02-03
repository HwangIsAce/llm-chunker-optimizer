"""Integration tests for optimization loop"""
import pytest
from unittest.mock import Mock, MagicMock
from chunker_optimizer.application.use_cases.run_optimization_loop import (
    RunOptimizationLoopUseCase,
    OptimizationConfig
)
from chunker_optimizer.application.use_cases.evaluate_chunks import EvaluateChunksUseCase
from chunker_optimizer.application.use_cases.optimize_prompt import OptimizePromptUseCase
from chunker_optimizer.domain.entities.prompt import Prompt
from chunker_optimizer.domain.entities.chunk import Chunk
from chunker_optimizer.domain.entities.chunking_context import ChunkingContext
from chunker_optimizer.domain.entities.document_context import (
    DocumentEnrichment,
    ParsedDocument
)
from chunker_optimizer.domain.entities.evaluation_metrics import EvaluationMetrics
from chunker_optimizer.infrastructure.llm.llm_client import LLMClient
from chunker_optimizer.infrastructure.performance.metrics_collector import PerformanceMetricsCollector


class MockLLMClient(LLMClient):
    """Mock LLM client for testing"""
    
    def optimize_prompt(self, current_prompt: str, optimization_instruction: str) -> str:
        return f"Optimized: {current_prompt}"


class TestOptimizationLoopIntegration:
    """Integration tests for optimization loop"""
    
    @pytest.fixture
    def chunking_context(self):
        """Create chunking context"""
        parsed_doc = ParsedDocument(
            raw_content="This is a test document with multiple paragraphs.\n\nSecond paragraph here.\n\nThird paragraph.",
            parsed_elements=[]
        )
        enrichment = DocumentEnrichment(parsed_document=parsed_doc)
        
        return ChunkingContext(
            document_enrichment=enrichment,
            prompt=Prompt(id="test", content="Initial prompt", version=1),
            chunk_unit="page"
        )
    
    @pytest.fixture
    def mock_chunking_function(self):
        """Mock chunking function"""
        def chunk_func(context: ChunkingContext) -> list[Chunk]:
            text = context.document_enrichment.parsed_document.raw_content
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            
            chunks = []
            current_index = 0
            for i, para in enumerate(paragraphs):
                start_index = current_index
                end_index = start_index + len(para)
                chunks.append(Chunk(
                    id=f"chunk_{i}",
                    content=para,
                    start_index=start_index,
                    end_index=end_index
                ))
                current_index = end_index + 2
            
            return chunks
        
        return chunk_func
    
    @pytest.fixture
    def evaluate_use_case(self):
        """Create evaluate use case"""
        return EvaluateChunksUseCase()
    
    @pytest.fixture
    def optimize_use_case(self):
        """Create optimize use case"""
        return OptimizePromptUseCase(llm_client=MockLLMClient())
    
    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector"""
        return PerformanceMetricsCollector()
    
    def test_full_optimization_loop(
        self,
        chunking_context,
        mock_chunking_function,
        evaluate_use_case,
        optimize_use_case,
        metrics_collector
    ):
        """Test full optimization loop execution"""
        use_case = RunOptimizationLoopUseCase(
            evaluate_use_case=evaluate_use_case,
            optimize_use_case=optimize_use_case,
            chunking_function=mock_chunking_function,
            metrics_collector=metrics_collector
        )
        
        prompt = chunking_context.prompt
        config = OptimizationConfig(
            max_iterations=3,
            threshold=0.9  # High threshold to test multiple iterations
        )
        
        result = use_case.execute(prompt, chunking_context, config)
        
        # Verify results
        assert result.iterations > 0
        assert result.iterations <= config.max_iterations
        assert len(result.history) == result.iterations
        assert result.final_prompt.version >= prompt.version
        
        # Verify performance metrics
        performance = metrics_collector.get_summary()
        assert performance.total_iterations == result.iterations
        assert performance.total_time > 0
    
    def test_optimization_with_performance_tracking(
        self,
        chunking_context,
        mock_chunking_function,
        evaluate_use_case,
        optimize_use_case,
        metrics_collector
    ):
        """Test optimization with performance tracking"""
        use_case = RunOptimizationLoopUseCase(
            evaluate_use_case=evaluate_use_case,
            optimize_use_case=optimize_use_case,
            chunking_function=mock_chunking_function,
            metrics_collector=metrics_collector
        )
        
        prompt = chunking_context.prompt
        config = OptimizationConfig(max_iterations=2, threshold=0.9)
        
        result = use_case.execute(prompt, chunking_context, config)
        
        # Check performance metrics
        performance = metrics_collector.get_summary()
        assert performance.total_iterations == result.iterations
        assert performance.average_evaluation_time >= 0
        assert performance.average_optimization_time >= 0
        assert performance.average_chunking_time >= 0
    
    def test_optimization_history(
        self,
        chunking_context,
        mock_chunking_function,
        evaluate_use_case,
        optimize_use_case
    ):
        """Test that optimization history is properly recorded"""
        use_case = RunOptimizationLoopUseCase(
            evaluate_use_case=evaluate_use_case,
            optimize_use_case=optimize_use_case,
            chunking_function=mock_chunking_function
        )
        
        prompt = chunking_context.prompt
        config = OptimizationConfig(max_iterations=3, threshold=0.9)
        
        result = use_case.execute(prompt, chunking_context, config)
        
        # Verify history
        assert len(result.history) == result.iterations
        for i, entry in enumerate(result.history):
            assert entry["iteration"] == i
            assert "metrics" in entry
            assert "chunk_count" in entry
            assert entry["metrics"]["overall_score"] >= 0.0
