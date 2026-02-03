"""Enrichment pipeline interface"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from ...domain.entities.document_context import DocumentEnrichment


class EnrichmentPipeline(ABC):
    """Abstract interface for enrichment pipeline"""
    
    @abstractmethod
    def process(
        self,
        document: str,
        chunk_unit: str
    ) -> DocumentEnrichment:
        """
        Process document through enrichment pipeline
        
        Args:
            document: Document content or path
            chunk_unit: Chunk unit type ("page", "element", "lifelog", "default")
        
        Returns:
            DocumentEnrichment with parsed document, VLM outputs, and LLM summary
        """
        pass
    
    @abstractmethod
    def get_enrichment_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the enrichment pipeline
        
        Returns:
            Dictionary with pipeline metadata
        """
        pass
