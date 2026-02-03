"""Use case for running optimization loop"""
from typing import List, Optional, Callable
from dataclasses import dataclass
from ...domain.entities.chunk import Chunk
from ...domain.entities.prompt import Prompt
from ...domain.entities.chunking_context import ChunkingContext
from ...domain.entities.evaluation_metrics import EvaluationMetrics
from .evaluate_chunks import EvaluateChunksUseCase
from .optimize_prompt import OptimizePromptUseCase


@dataclass
class OptimizationConfig:
    """Configuration for optimization loop"""
    
    max_iterations: int = 10
    threshold: float = 0.8
    improvement_threshold: float = 0.01
    early_stopping_patience: int = 3
    
    def __post_init__(self):
        """Validate configuration"""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.improvement_threshold <= 1.0:
            raise ValueError("improvement_threshold must be between 0.0 and 1.0")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1")


@dataclass
class OptimizationResult:
    """Result of optimization loop"""
    
    final_prompt: Prompt
    final_metrics: EvaluationMetrics
    iterations: int
    history: List[dict]
    converged: bool
    
    def __post_init__(self):
        """Validate result"""
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if len(self.history) != self.iterations:
            raise ValueError("history length must match iterations")


class RunOptimizationLoopUseCase:
    """Main use case for running optimization loop"""
    
    def __init__(
        self,
        evaluate_use_case: EvaluateChunksUseCase,
        optimize_use_case: OptimizePromptUseCase,
        chunking_function: Callable[[ChunkingContext], List[Chunk]],
    ):
        """
        Initialize use case
        
        Args:
            evaluate_use_case: Use case for evaluating chunks
            optimize_use_case: Use case for optimizing prompts
            chunking_function: Function that takes ChunkingContext and returns chunks
        """
        self.evaluate_use_case = evaluate_use_case
        self.optimize_use_case = optimize_use_case
        self.chunking_function = chunking_function
    
    def execute(
        self,
        initial_prompt: Prompt,
        chunking_context: ChunkingContext,
        config: OptimizationConfig
    ) -> OptimizationResult:
        """
        Run optimization loop until threshold is met or max iterations
        
        Args:
            initial_prompt: Initial prompt to start optimization
            chunking_context: Context containing document enrichment and chunk unit
            config: Optimization configuration
        
        Returns:
            OptimizationResult with final prompt, metrics, and history
        """
        current_prompt = initial_prompt
        history = []
        best_score = 0.0
        no_improvement_count = 0
        
        # Get original text from document enrichment
        original_text = chunking_context.document_enrichment.parsed_document.raw_content
        
        for iteration in range(config.max_iterations):
            # Update context with current prompt
            current_context = ChunkingContext(
                document_enrichment=chunking_context.document_enrichment,
                prompt=current_prompt,
                chunk_unit=chunking_context.chunk_unit,
                additional_context=chunking_context.additional_context
            )
            
            # Generate chunks with current prompt and context
            chunks = self.chunking_function(current_context)
            
            # Evaluate chunks
            metrics = self.evaluate_use_case.execute(chunks, original_text)
            
            # Record history
            history.append({
                "iteration": iteration,
                "prompt_version": current_prompt.version,
                "metrics": metrics.to_dict(),
                "chunk_count": len(chunks),
                "used_vlm": chunking_context.document_enrichment.has_vlm_output(),
                "used_llm_summary": chunking_context.document_enrichment.has_llm_summary()
            })
            
            # Check if threshold is met
            if metrics.meets_threshold(config.threshold):
                return OptimizationResult(
                    final_prompt=current_prompt,
                    final_metrics=metrics,
                    iterations=iteration + 1,
                    history=history,
                    converged=True
                )
            
            # Check for improvement
            if metrics.overall_score > best_score + config.improvement_threshold:
                best_score = metrics.overall_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Early stopping
            if no_improvement_count >= config.early_stopping_patience:
                return OptimizationResult(
                    final_prompt=current_prompt,
                    final_metrics=metrics,
                    iterations=iteration + 1,
                    history=history,
                    converged=False
                )
            
            # Optimize prompt for next iteration
            if iteration < config.max_iterations - 1:
                current_prompt = self.optimize_use_case.execute(
                    current_prompt,
                    metrics,
                    chunks,
                    iteration
                )
        
        # Max iterations reached
        return OptimizationResult(
            final_prompt=current_prompt,
            final_metrics=metrics,
            iterations=config.max_iterations,
            history=history,
            converged=False
        )
