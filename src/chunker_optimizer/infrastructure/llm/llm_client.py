"""LLM client interface"""
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract interface for LLM clients"""
    
    @abstractmethod
    def optimize_prompt(
        self,
        current_prompt: str,
        optimization_instruction: str
    ) -> str:
        """
        Optimize prompt using LLM
        
        Args:
            current_prompt: Current prompt to optimize
            optimization_instruction: Instruction for optimization
        
        Returns:
            Optimized prompt string
        """
        pass
