"""Use case for optimizing chunking prompt based on evaluation"""
from typing import Optional
from ...domain.entities.prompt import Prompt
from ...domain.entities.evaluation_metrics import EvaluationMetrics
from ...infrastructure.llm.llm_client import LLMClient


class OptimizePromptUseCase:
    """Use case for optimizing chunking prompt based on evaluation feedback"""
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize use case with LLM client
        
        Args:
            llm_client: LLM client for prompt optimization
        """
        self.llm_client = llm_client
    
    def execute(
        self,
        current_prompt: Prompt,
        evaluation_metrics: EvaluationMetrics,
        chunks: list,
        iteration: int
    ) -> Prompt:
        """
        Generate improved prompt based on evaluation feedback
        
        Args:
            current_prompt: Current prompt to optimize
            evaluation_metrics: Evaluation results from current iteration
            chunks: Generated chunks from current prompt
            iteration: Current iteration number
        
        Returns:
            Optimized prompt with incremented version
        """
        # Build optimization instruction
        optimization_instruction = self._build_optimization_instruction(
            current_prompt,
            evaluation_metrics,
            iteration
        )
        
        # Call LLM to optimize prompt
        improved_content = self.llm_client.optimize_prompt(
            current_prompt.content,
            optimization_instruction
        )
        
        # Create new prompt with incremented version
        return Prompt(
            id=current_prompt.id,
            content=improved_content,
            version=current_prompt.version + 1,
            metadata={
                **current_prompt.metadata,
                "iteration": iteration,
                "previous_metrics": {
                    "overall_score": evaluation_metrics.overall_score,
                    "boundary_clarity": evaluation_metrics.boundary_clarity,
                    "chunk_stickiness": evaluation_metrics.chunk_stickiness,
                    "hope_score": evaluation_metrics.hope_score
                }
            }
        )
    
    def _build_optimization_instruction(
        self,
        prompt: Prompt,
        metrics: EvaluationMetrics,
        iteration: int
    ) -> str:
        """
        Build instruction for LLM to optimize prompt
        
        Args:
            prompt: Current prompt
            metrics: Evaluation metrics
            iteration: Current iteration number
        
        Returns:
            Optimization instruction string
        """
        return f"""
Optimize the following chunking prompt based on evaluation results.

Current Prompt:
{prompt.content}

Evaluation Results (Iteration {iteration}):
- Overall Score: {metrics.overall_score:.3f}
- Boundary Clarity: {metrics.boundary_clarity:.3f}
- Chunk Stickiness: {metrics.chunk_stickiness:.3f}
- HOPE Score: {metrics.hope_score:.3f}
- Extrinsic (Semantic Independence): {metrics.extrinsic_properties.get('score', 0):.3f}

Focus on improving areas with low scores. The goal is to create chunks that:
1. Have clear boundaries (high boundary clarity)
2. Have internally coherent content (high chunk stickiness)
3. Are semantically independent from each other (high extrinsic score)
4. Maintain coherence with the original document

Generate an improved version of the prompt that addresses the weaknesses identified in the evaluation.
"""
