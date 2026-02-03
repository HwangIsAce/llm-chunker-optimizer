"""LangGraph state graph for optimization loop"""
from langgraph.graph import StateGraph, END
from .state import OptimizationState
from .nodes import (
    generate_chunks_node,
    evaluate_node,
    check_threshold_node,
    optimize_prompt_node
)


def should_continue_optimization(state: OptimizationState) -> str:
    """
    Determine if optimization should continue
    
    Args:
        state: Current optimization state
    
    Returns:
        "end" if should stop, "continue" if should continue
    """
    # Check if converged
    if state["converged"]:
        return "end"
    
    # Check if max iterations reached
    # Note: iteration is incremented in optimize_prompt_node, so we check before increment
    if state["iteration"] >= state["config"].max_iterations - 1:
        return "end"
    
    # Check early stopping
    history = state.get("history", [])
    if len(history) >= state["config"].early_stopping_patience + 1:
        # Check if there's been improvement in recent iterations
        recent_scores = [
            entry["metrics"]["overall_score"]
            for entry in history[-state["config"].early_stopping_patience:]
        ]
        
        # If no improvement, stop
        if len(recent_scores) >= 2:
            improvement = recent_scores[-1] - recent_scores[0]
            if improvement < state["config"].improvement_threshold:
                return "end"
    
    return "continue"


def create_optimization_graph() -> StateGraph:
    """
    Create LangGraph state graph for optimization loop
    
    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(OptimizationState)
    
    # Add nodes
    workflow.add_node("generate_chunks", generate_chunks_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("check_threshold", check_threshold_node)
    workflow.add_node("optimize_prompt", optimize_prompt_node)
    
    # Define edges
    workflow.set_entry_point("generate_chunks")
    workflow.add_edge("generate_chunks", "evaluate")
    workflow.add_edge("evaluate", "check_threshold")
    
    # Conditional edge from check_threshold
    workflow.add_conditional_edges(
        "check_threshold",
        should_continue_optimization,
        {
            "continue": "optimize_prompt",
            "end": END
        }
    )
    
    # Edge from optimize_prompt back to generate_chunks
    workflow.add_edge("optimize_prompt", "generate_chunks")
    
    return workflow.compile()
