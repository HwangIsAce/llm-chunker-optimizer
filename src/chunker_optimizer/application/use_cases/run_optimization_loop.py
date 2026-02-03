"""Use case for running optimization loop"""
from typing import List, Optional, Callable
from dataclasses import dataclass
from ...domain.entities.chunk import Chunk
from ...domain.entities.prompt import Prompt
from ...domain.entities.chunking_context import ChunkingContext
from ...domain.entities.evaluation_metrics import EvaluationMetrics
from ...infrastructure.performance.metrics_collector import PerformanceMetricsCollector
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
        metrics_collector: Optional[PerformanceMetricsCollector] = None,
    ):
        """
        Initialize use case
        
        Args:
            evaluate_use_case: Use case for evaluating chunks
            optimize_use_case: Use case for optimizing prompts
            chunking_function: Function that takes ChunkingContext and returns chunks
            metrics_collector: Optional performance metrics collector
        """
        self.evaluate_use_case = evaluate_use_case
        self.optimize_use_case = optimize_use_case
        self.chunking_function = chunking_function
        self.metrics_collector = metrics_collector
    
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
        
        # Reset metrics collector if provided
        if self.metrics_collector:
            self.metrics_collector.reset()
        
        for iteration in range(config.max_iterations):
            # Update context with current prompt
            current_context = ChunkingContext(
                document_enrichment=chunking_context.document_enrichment,
                prompt=current_prompt,
                chunk_unit=chunking_context.chunk_unit,
                additional_context=chunking_context.additional_context
            )
            
            # Generate chunks with current prompt and context
            chunking_time = 0.0
            if self.metrics_collector:
                with self.metrics_collector.time_operation("chunking"):
                    chunks = self.chunking_function(current_context)
                chunking_time = self.metrics_collector.timers.get("chunking", 0.0)
            else:
                chunks = self.chunking_function(current_context)
            
            # Evaluate chunks
            evaluation_time = 0.0
            if self.metrics_collector:
                with self.metrics_collector.time_operation("evaluation"):
                    metrics = self.evaluate_use_case.execute(chunks, original_text)
                evaluation_time = self.metrics_collector.timers.get("evaluation", 0.0)
            else:
                metrics = self.evaluate_use_case.execute(chunks, original_text)
            
            # Record history with chunk samples
            chunk_samples = []
            for i, chunk in enumerate(chunks[:5]):  # Store first 5 chunks as samples
                chunk_samples.append({
                    "id": chunk.id,
                    "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    "length": len(chunk.content),
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index
                })
            
            history.append({
                "iteration": iteration,
                "prompt_version": current_prompt.version,
                "prompt_content": current_prompt.content,
                "metrics": metrics.to_dict(),
                "chunk_count": len(chunks),
                "chunk_samples": chunk_samples,
                "total_chunks": len(chunks),
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
            optimization_time = 0.0
            llm_call_time = 0.0
            token_count = 0
            
            if iteration < config.max_iterations - 1:
                if self.metrics_collector:
                    with self.metrics_collector.time_operation("optimization"):
                        current_prompt = self.optimize_use_case.execute(
                            current_prompt,
                            metrics,
                            chunks,
                            iteration
                        )
                    optimization_time = self.metrics_collector.timers.get("optimization", 0.0)
                    # LLM call time is part of optimization
                    llm_call_time = optimization_time  # Simplified - could be more granular
                else:
                    current_prompt = self.optimize_use_case.execute(
                        current_prompt,
                        metrics,
                        chunks,
                        iteration
                    )
            
            # Record performance metrics
            if self.metrics_collector:
                self.metrics_collector.record_metrics(
                    iteration=iteration,
                    evaluation_time=evaluation_time,
                    optimization_time=optimization_time,
                    chunking_time=chunking_time,
                    llm_call_time=llm_call_time,
                    token_count=token_count
                )
        
        # Max iterations reached
        return OptimizationResult(
            final_prompt=current_prompt,
            final_metrics=metrics,
            iterations=config.max_iterations,
            history=history,
            converged=False
        )
