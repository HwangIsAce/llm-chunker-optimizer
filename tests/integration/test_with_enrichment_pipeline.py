"""Integration tests with enrichment pipeline"""
import pytest
from chunker_optimizer.infrastructure.enrichment.enrichment_pipeline import EnrichmentPipeline
from chunker_optimizer.domain.entities.document_context import DocumentEnrichment
from chunker_optimizer.domain.entities.chunking_context import ChunkingContext
from chunker_optimizer.domain.entities.prompt import Prompt


class MockEnrichmentPipeline(EnrichmentPipeline):
    """Mock enrichment pipeline for testing"""
    
    def process(self, document: str, chunk_unit: str) -> DocumentEnrichment:
        """Mock enrichment processing"""
        from chunker_optimizer.domain.entities.document_context import (
            ParsedDocument,
            VLMOutput,
            LLMSummary
        )
        
        parsed_doc = ParsedDocument(
            raw_content=document,
            parsed_elements=[{"type": "text", "content": document}],
            metadata={"source": "mock_pipeline"}
        )
        
        vlm_outputs = []
        if chunk_unit == "page":
            vlm_outputs = [
                VLMOutput(
                    page_id="page_1",
                    script="Mock VLM script",
                    visual_elements=[],
                    metadata={"mock": True}
                )
            ]
        
        llm_summary = LLMSummary(
            summary="Mock LLM summary",
            key_points=["Point 1"],
            topics=["Topic A"],
            structure_info={},
            metadata={"mock": True}
        )
        
        return DocumentEnrichment(
            parsed_document=parsed_doc,
            vlm_outputs=vlm_outputs,
            llm_summary=llm_summary
        )
    
    def get_enrichment_metadata(self) -> dict:
        """Get pipeline metadata"""
        return {"type": "mock", "version": "1.0"}


class TestWithEnrichmentPipeline:
    """Integration tests with enrichment pipeline"""
    
    @pytest.fixture
    def mock_pipeline(self):
        """Create mock enrichment pipeline"""
        return MockEnrichmentPipeline()
    
    def test_pipeline_processing(self, mock_pipeline):
        """Test pipeline processing"""
        document = "Test document content"
        enrichment = mock_pipeline.process(document, chunk_unit="page")
        
        assert enrichment.parsed_document.raw_content == document
        assert enrichment.has_vlm_output()
        assert enrichment.has_llm_summary()
    
    def test_pipeline_with_different_chunk_units(self, mock_pipeline):
        """Test pipeline with different chunk units"""
        document = "Test document"
        
        # Test page unit
        page_enrichment = mock_pipeline.process(document, chunk_unit="page")
        assert page_enrichment.has_vlm_output()
        
        # Test element unit
        element_enrichment = mock_pipeline.process(document, chunk_unit="element")
        # Element might not have VLM output
        assert element_enrichment.parsed_document is not None
    
    def test_chunking_context_from_pipeline(self, mock_pipeline):
        """Test creating chunking context from pipeline output"""
        document = "Test document for chunking"
        enrichment = mock_pipeline.process(document, chunk_unit="page")
        
        context = ChunkingContext(
            document_enrichment=enrichment,
            prompt=Prompt(id="test", content="Test prompt", version=1),
            chunk_unit="page"
        )
        
        # Verify context
        assert context.document_enrichment == enrichment
        assert context.chunk_unit == "page"
        
        # Test enrichment extraction
        enrichment_info = context.get_enrichment_for_chunking()
        assert "parsed_document" in enrichment_info
        assert "vlm_outputs" in enrichment_info
        assert "llm_summary" in enrichment_info
    
    def test_pipeline_metadata(self, mock_pipeline):
        """Test pipeline metadata"""
        metadata = mock_pipeline.get_enrichment_metadata()
        assert isinstance(metadata, dict)
        assert "type" in metadata
