"""Chunking context entity containing all information needed for chunking"""
from dataclasses import dataclass
from typing import Dict, Optional
from .document_context import DocumentEnrichment
from .prompt import Prompt


@dataclass
class ChunkingContext:
    """Chunking context containing all information needed for chunking"""
    
    document_enrichment: DocumentEnrichment
    prompt: Prompt
    chunk_unit: str
    additional_context: Optional[Dict] = None
    
    def __post_init__(self):
        """Validate and initialize chunking context"""
        if self.additional_context is None:
            self.additional_context = {}
        
        # Validation
        if self.document_enrichment is None:
            raise ValueError("document_enrichment cannot be None")
        if self.prompt is None:
            raise ValueError("prompt cannot be None")
        if not self.chunk_unit:
            raise ValueError("chunk_unit cannot be empty")
        
        # Validate chunk_unit
        valid_units = ["page", "element", "lifelog", "default"]
        if self.chunk_unit not in valid_units:
            raise ValueError(
                f"chunk_unit must be one of {valid_units}, got {self.chunk_unit}"
            )
    
    def get_enrichment_for_chunking(self) -> Dict:
        """
        Extract enrichment information needed for chunking
        
        Returns:
            Dictionary containing enrichment data for chunking
        """
        context = {
            "parsed_document": self.document_enrichment.parsed_document,
            "chunk_unit": self.chunk_unit
        }
        
        # Add VLM outputs if available
        if self.document_enrichment.has_vlm_output():
            context["vlm_outputs"] = self.document_enrichment.vlm_outputs
        
        # Add LLM summary if available
        if self.document_enrichment.has_llm_summary():
            context["llm_summary"] = self.document_enrichment.llm_summary
        
        # Add additional context
        context.update(self.additional_context)
        
        return context
