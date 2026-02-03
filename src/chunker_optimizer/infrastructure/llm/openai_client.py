"""OpenAI LLM client implementation"""
import os
from typing import Optional
from openai import OpenAI
from .llm_client import LLMClient


class OpenAIClient(LLMClient):
    """OpenAI implementation of LLM client"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Initialize OpenAI client
        
        Args:
            api_key: OpenAI API key (default: from OPENAI_API_KEY env var)
            model: Model to use (default: gpt-4)
            temperature: Temperature for generation (default: 0.7)
            max_tokens: Maximum tokens to generate (default: 1000)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def optimize_prompt(
        self,
        current_prompt: str,
        optimization_instruction: str
    ) -> str:
        """
        Optimize prompt using OpenAI
        
        Args:
            current_prompt: Current prompt to optimize
            optimization_instruction: Instruction for optimization
        
        Returns:
            Optimized prompt string
        """
        system_prompt = """You are an expert at optimizing text chunking prompts for RAG (Retrieval-Augmented Generation) systems.
Your goal is to improve prompts based on evaluation metrics to create better text chunks.
Return only the optimized prompt, without any additional explanation or commentary."""
        
        user_prompt = f"""{optimization_instruction}

Please provide the optimized prompt:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            optimized_prompt = response.choices[0].message.content.strip()
            return optimized_prompt
        
        except Exception as e:
            raise RuntimeError(f"Failed to optimize prompt with OpenAI: {e}") from e
    
    def get_token_count(self, text: str) -> int:
        """
        Get approximate token count for text
        
        Args:
            text: Text to count tokens for
        
        Returns:
            Approximate token count
        """
        try:
            from tiktoken import encoding_for_model
            encoding = encoding_for_model(self.model)
            return len(encoding.encode(text))
        except Exception:
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4
