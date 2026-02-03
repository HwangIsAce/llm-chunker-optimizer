"""State definition for LangGraph optimization loop"""
from typing import TypedDict, List, Callable
from ...domain.entities.prompt import Prompt
from ...domain.entities.chunk import Chunk
from ...domain.entities.chunking_context import ChunkingContext
from ...domain.entities.evaluation_metrics import EvaluationMetrics
from ...application.use_cases.evaluate_chunks import EvaluateChunksUseCase
from ...application.use_cases.optimize_prompt import OptimizePromptUseCase
from ...application.use_cases.run_optimization_loop import OptimizationConfig


class OptimizationState(TypedDict):
    """State for LangGraph optimization loop"""
    
    prompt: Prompt
    chunking_context: ChunkingContext
    original_text: str
    chunks: List[Chunk]
    metrics: EvaluationMetrics
    iteration: int
    config: OptimizationConfig
    history: List[dict]
    converged: bool
    chunking_function: Callable[[ChunkingContext], List[Chunk]]
    evaluate_use_case: EvaluateChunksUseCase
    optimize_use_case: OptimizePromptUseCase
