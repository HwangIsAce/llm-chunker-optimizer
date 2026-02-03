"""LangGraph nodes for optimization loop"""
from typing import Annotated
from ...domain.entities.chunk import Chunk
from ...domain.entities.evaluation_metrics import EvaluationMetrics
from ...application.use_cases.evaluate_chunks import EvaluateChunksUseCase
from ...application.use_cases.optimize_prompt import OptimizePromptUseCase
from .state import OptimizationState


def generate_chunks_node(state: OptimizationState) -> OptimizationState:
    """
    Node to generate chunks using current prompt and context
    
    Args:
        state: Current optimization state
    
    Returns:
        Updated state with generated chunks
    """
    chunking_function = state["chunking_function"]
    chunking_context = state["chunking_context"]
    
    # Generate chunks
    chunks = chunking_function(chunking_context)
    
    return {
        **state,
        "chunks": chunks
    }


def evaluate_node(state: OptimizationState) -> OptimizationState:
    """
    Node to evaluate generated chunks
    
    Args:
        state: Current optimization state
    
    Returns:
        Updated state with evaluation metrics
    """
    evaluate_use_case = state["evaluate_use_case"]
    chunks = state["chunks"]
    original_text = state["original_text"]
    
    # Evaluate chunks
    metrics = evaluate_use_case.execute(chunks, original_text)
    
    # Update history
    history = state.get("history", [])
    history.append({
        "iteration": state["iteration"],
        "prompt_version": state["prompt"].version,
        "metrics": metrics.to_dict(),
        "chunk_count": len(chunks),
        "used_vlm": state["chunking_context"].document_enrichment.has_vlm_output(),
        "used_llm_summary": state["chunking_context"].document_enrichment.has_llm_summary()
    })
    
    return {
        **state,
        "metrics": metrics,
        "history": history
    }


def check_threshold_node(state: OptimizationState) -> OptimizationState:
    """
    Node to check if threshold is met
    
    Args:
        state: Current optimization state
    
    Returns:
        Updated state with convergence status
    """
    metrics = state["metrics"]
    config = state["config"]
    
    converged = metrics.meets_threshold(config.threshold)
    
    return {
        **state,
        "converged": converged
    }


def optimize_prompt_node(state: OptimizationState) -> OptimizationState:
    """
    Node to optimize prompt for next iteration
    
    Args:
        state: Current optimization state
    
    Returns:
        Updated state with optimized prompt
    """
    optimize_use_case = state["optimize_use_case"]
    current_prompt = state["prompt"]
    metrics = state["metrics"]
    chunks = state["chunks"]
    iteration = state["iteration"]
    
    # Optimize prompt
    new_prompt = optimize_use_case.execute(
        current_prompt,
        metrics,
        chunks,
        iteration
    )
    
    return {
        **state,
        "prompt": new_prompt,
        "iteration": state["iteration"] + 1
    }
