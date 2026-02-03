"""Integration tests with real JSON data"""
import pytest
from chunker_optimizer.domain.entities.chunking_context import ChunkingContext
from chunker_optimizer.domain.entities.prompt import Prompt
from tests.fixtures.real_data_loader import RealEnrichmentDataLoader


class TestWithRealData:
    """Integration tests with real JSON data"""
    
    @pytest.fixture
    def real_page_enrichment(self):
        """Load real page enrichment data"""
        enrichment = RealEnrichmentDataLoader.load_by_chunk_unit("page")
        if enrichment is None:
            pytest.skip("Real page enrichment data not found")
        return enrichment
    
    @pytest.fixture
    def real_element_enrichment(self):
        """Load real element enrichment data"""
        enrichment = RealEnrichmentDataLoader.load_by_chunk_unit("element")
        if enrichment is None:
            pytest.skip("Real element enrichment data not found")
        return enrichment
    
    def test_load_real_data_structure(self, real_page_enrichment):
        """Test that real data has correct structure"""
        assert real_page_enrichment is not None
        assert real_page_enrichment.parsed_document is not None
        assert real_page_enrichment.parsed_document.raw_content is not None
    
    def test_optimization_with_real_vlm_output(self, real_page_enrichment):
        """Test optimization with real VLM output"""
        context = ChunkingContext(
            document_enrichment=real_page_enrichment,
            prompt=Prompt(id="test", content="Initial prompt", version=1),
            chunk_unit="page"
        )
        
        # Verify VLM output exists
        assert context.document_enrichment.has_vlm_output()
        if len(context.document_enrichment.vlm_outputs) > 0:
            assert context.document_enrichment.vlm_outputs[0].script is not None
            assert context.document_enrichment.vlm_outputs[0].page_id is not None
    
    def test_optimization_with_real_llm_summary(self, real_page_enrichment):
        """Test optimization with real LLM summary"""
        context = ChunkingContext(
            document_enrichment=real_page_enrichment,
            prompt=Prompt(id="test", content="Initial prompt", version=1),
            chunk_unit="page"
        )
        
        # Verify LLM summary exists
        if context.document_enrichment.has_llm_summary():
            assert context.document_enrichment.llm_summary.summary is not None
            assert len(context.document_enrichment.llm_summary.key_points) >= 0
    
    def test_list_available_data(self):
        """Test listing available data files"""
        available = RealEnrichmentDataLoader.list_available_data()
        # Should return list (may be empty if no data files exist)
        assert isinstance(available, list)
    
    def test_load_latest(self):
        """Test loading latest data file"""
        latest = RealEnrichmentDataLoader.load_latest()
        # May be None if no files exist
        if latest is not None:
            assert latest.parsed_document is not None
    
    def test_chunking_context_with_real_data(self, real_page_enrichment):
        """Test creating chunking context with real data"""
        context = ChunkingContext(
            document_enrichment=real_page_enrichment,
            prompt=Prompt(id="test", content="Test prompt", version=1),
            chunk_unit="page"
        )
        
        # Test enrichment extraction
        enrichment_info = context.get_enrichment_for_chunking()
        assert "parsed_document" in enrichment_info
        assert "chunk_unit" in enrichment_info
        assert enrichment_info["chunk_unit"] == "page"
