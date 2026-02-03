"""Integration tests for LangGraph optimization graph"""
import pytest
from unittest.mock import Mock, MagicMock
from langgraph.graph import StateGraph
from chunker_optimizer.infrastructure.langgraph.optimization_graph import (
    create_optimization_graph
)
from chunker_optimizer.infrastructure.langgraph.state import OptimizationState
from chunker_optimizer.domain.entities.prompt import Prompt
from chunker_optimizer.domain.entities.chunk import Chunk
from chunker_optimizer.domain.entities.chunking_context import ChunkingContext
from chunker_optimizer.domain.entities.document_context import (
    DocumentEnrichment,
    ParsedDocument
)
from chunker_optimizer.domain.entities.evaluation_metrics import EvaluationMetrics
from chunker_optimizer.application.use_cases.evaluate_chunks import EvaluateChunksUseCase
from chunker_optimizer.application.use_cases.optimize_prompt import OptimizePromptUseCase
from chunker_optimizer.application.use_cases.run_optimization_loop import OptimizationConfig
from chunker_optimizer.infrastructure.llm.llm_client import LLMClient


class MockLLMClient(LLMClient):
    """Mock LLM client for testing"""
    
    def optimize_prompt(self, current_prompt: str, optimization_instruction: str) -> str:
        return f"Optimized: {current_prompt}"


class TestLangGraphIntegration:
    """Test LangGraph integration"""
    
    @pytest.fixture
    def chunking_context(self):
        """Create chunking context"""
        parsed_doc = ParsedDocument(
            raw_content="test document content",
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
        return MagicMock(return_value=[
            Chunk(id="1", content="chunk", start_index=0, end_index=5)
        ])
    
    @pytest.fixture
    def evaluate_use_case(self):
        """Create evaluate use case"""
        return EvaluateChunksUseCase()
    
    @pytest.fixture
    def optimize_use_case(self):
        """Create optimize use case with mock LLM"""
        return OptimizePromptUseCase(llm_client=MockLLMClient())
    
    @pytest.fixture
    def config(self):
        """Create optimization config"""
        return OptimizationConfig(
            max_iterations=3,
            threshold=0.8
        )
    
    def test_create_graph(self):
        """Test that graph can be created"""
        graph = create_optimization_graph()
        assert graph is not None
        # Compiled graph is not StateGraph, it's CompiledStateGraph
        assert hasattr(graph, "invoke")
    
    def test_graph_execution_converges(
        self,
        chunking_context,
        mock_chunking_function,
        evaluate_use_case,
        optimize_use_case,
        config
    ):
        """Test graph execution when threshold is met"""
        # Mock evaluate to return high score (converged)
        evaluate_use_case.execute = MagicMock(
            return_value=EvaluationMetrics(
                boundary_clarity=0.9,
                chunk_stickiness=0.9,
                hope_score=0.9,
                intrinsic_properties={},
                extrinsic_properties={},
                coherence_score=0.9
            )
        )
        
        # Create initial state
        initial_state: OptimizationState = {
            "prompt": chunking_context.prompt,
            "chunking_context": chunking_context,
            "original_text": chunking_context.document_enrichment.parsed_document.raw_content,
            "chunks": [],
            "metrics": EvaluationMetrics(0.5, 0.5, 0.5, {}, {}, 0.5),
            "iteration": 0,
            "config": config,
            "history": [],
            "converged": False,
            "chunking_function": mock_chunking_function,
            "evaluate_use_case": evaluate_use_case,
            "optimize_use_case": optimize_use_case
        }
        
        # Create and run graph
        graph = create_optimization_graph()
        result = graph.invoke(initial_state)
        
        # Should converge quickly (after first evaluation)
        assert result["converged"] is True
        # Iteration might be 0 if it converges immediately, or >= 1 if it takes a loop
        assert result["iteration"] >= 0
        assert len(result["history"]) > 0
    
    def test_graph_execution_max_iterations(
        self,
        chunking_context,
        mock_chunking_function,
        evaluate_use_case,
        optimize_use_case,
        config
    ):
        """Test graph execution when max iterations reached"""
        # Mock evaluate to return low score (won't converge)
        evaluate_use_case.execute = MagicMock(
            return_value=EvaluationMetrics(
                boundary_clarity=0.5,
                chunk_stickiness=0.5,
                hope_score=0.5,
                intrinsic_properties={},
                extrinsic_properties={},
                coherence_score=0.5
            )
        )
        
        # Create initial state
        initial_state: OptimizationState = {
            "prompt": chunking_context.prompt,
            "chunking_context": chunking_context,
            "original_text": chunking_context.document_enrichment.parsed_document.raw_content,
            "chunks": [],
            "metrics": EvaluationMetrics(0.5, 0.5, 0.5, {}, {}, 0.5),
            "iteration": 0,
            "config": config,
            "history": [],
            "converged": False,
            "chunking_function": mock_chunking_function,
            "evaluate_use_case": evaluate_use_case,
            "optimize_use_case": optimize_use_case
        }
        
        # Create and run graph
        graph = create_optimization_graph()
        result = graph.invoke(initial_state)
        
        # Should reach max iterations
        assert result["converged"] is False
        # History includes all iterations (0 to max_iterations-1)
        # iteration is 0-indexed, so max_iterations=3 means iterations 0, 1, 2
        assert len(result["history"]) == config.max_iterations
        assert result["iteration"] == config.max_iterations - 1
    
    def test_nodes_execution(
        self,
        chunking_context,
        mock_chunking_function,
        evaluate_use_case,
        optimize_use_case
    ):
        """Test individual node execution"""
        from chunker_optimizer.infrastructure.langgraph.nodes import (
            generate_chunks_node,
            evaluate_node
        )
        
        state: OptimizationState = {
            "prompt": chunking_context.prompt,
            "chunking_context": chunking_context,
            "original_text": "test",
            "chunks": [],
            "metrics": EvaluationMetrics(0.5, 0.5, 0.5, {}, {}, 0.5),
            "iteration": 0,
            "config": OptimizationConfig(),
            "history": [],
            "converged": False,
            "chunking_function": mock_chunking_function,
            "evaluate_use_case": evaluate_use_case,
            "optimize_use_case": optimize_use_case
        }
        
        # Test generate_chunks_node
        state_after_generate = generate_chunks_node(state)
        assert len(state_after_generate["chunks"]) > 0
        
        # Test evaluate_node
        state_after_evaluate = evaluate_node(state_after_generate)
        assert state_after_evaluate["metrics"] is not None
        assert len(state_after_evaluate["history"]) == 1
