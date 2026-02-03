"""Integration tests for peter-parser adapter"""
import pytest
from pathlib import Path
import sys

# Add peter-parser to path if available
PETER_PARSER_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "pipelines" / "parsing-pipeline" / "peter-parser"
if PETER_PARSER_PATH.exists() and str(PETER_PARSER_PATH) not in sys.path:
    sys.path.insert(0, str(PETER_PARSER_PATH))

try:
    from chunker_optimizer.infrastructure.adapters.peter_parser_adapter import (
        PeterParserAdapter,
        create_peter_parser_chunking_function,
        convert_peter_parsed_doc_to_optimizer,
        convert_peter_chunk_to_optimizer,
    )
    PETER_PARSER_AVAILABLE = True
except ImportError:
    PETER_PARSER_AVAILABLE = False


@pytest.mark.skipif(not PETER_PARSER_AVAILABLE, reason="peter-parser not available")
class TestPeterParserAdapter:
    """Tests for peter-parser adapter (requires peter-parser)"""
    
    def test_adapter_initialization(self):
        """Test adapter can be initialized"""
        adapter = PeterParserAdapter()
        assert adapter is not None
    
    def test_create_chunking_function(self):
        """Test creating chunking function"""
        chunking_function = create_peter_parser_chunking_function(
            chunk_unit="page",
            document_type=None
        )
        assert callable(chunking_function)
    
    def test_chunking_function_with_context(self):
        """Test chunking function with ChunkingContext"""
        from chunker_optimizer.domain.entities.chunking_context import ChunkingContext
        from chunker_optimizer.domain.entities.document_context import (
            DocumentEnrichment,
            ParsedDocument,
            LLMSummary
        )
        from chunker_optimizer.domain.entities.prompt import Prompt
        
        # Create test context
        parsed_doc = ParsedDocument(
            raw_content="Page 1 content.\n\nPage 2 content.",
            parsed_elements=[
                {"element_id": 1, "page_number": 1, "category": "text", "text": "Page 1 content."},
                {"element_id": 2, "page_number": 2, "category": "text", "text": "Page 2 content."}
            ],
            metadata={"total_pages": 2}
        )
        
        llm_summary = LLMSummary(
            summary="Test document summary",
            key_points=["Point 1"],
            topics=["Topic 1"],
            structure_info={}
        )
        
        enrichment = DocumentEnrichment(
            parsed_document=parsed_doc,
            vlm_outputs=[],
            llm_summary=llm_summary
        )
        
        prompt = Prompt(id="test", content="Test prompt", version=1)
        context = ChunkingContext(
            document_enrichment=enrichment,
            prompt=prompt,
            chunk_unit="page"
        )
        
        # Create chunking function
        chunking_function = create_peter_parser_chunking_function(chunk_unit="page")
        
        # Execute chunking
        chunks = chunking_function(context)
        
        assert len(chunks) > 0
        assert all(chunk.content for chunk in chunks)


class TestPeterParserAdapterFallback:
    """Tests for fallback behavior when peter-parser is not available"""
    
    def test_adapter_import_error(self):
        """Test that ImportError is raised when peter-parser is not available"""
        if not PETER_PARSER_AVAILABLE:
            # Import here to avoid NameError when peter-parser is available
            from chunker_optimizer.infrastructure.adapters.peter_parser_adapter import (
                PeterParserAdapter
            )
            with pytest.raises(ImportError):
                PeterParserAdapter()
        else:
            # Skip test when peter-parser is available
            pytest.skip("peter-parser is available, skipping fallback test")
    
    def test_chunking_function_import_error(self):
        """Test that ImportError is raised when creating chunking function"""
        if not PETER_PARSER_AVAILABLE:
            # Import here to avoid NameError when peter-parser is available
            from chunker_optimizer.infrastructure.adapters.peter_parser_adapter import (
                create_peter_parser_chunking_function
            )
            with pytest.raises(ImportError):
                create_peter_parser_chunking_function(chunk_unit="page")
        else:
            # Skip test when peter-parser is available
            pytest.skip("peter-parser is available, skipping fallback test")


class TestDataConversion:
    """Tests for data conversion functions"""
    
    @pytest.mark.skipif(not PETER_PARSER_AVAILABLE, reason="peter-parser not available")
    def test_convert_peter_parsed_doc(self):
        """Test converting peter-parser ParsedDocument"""
        from peter_parser_core.common.types import ParsedDocument as PeterParsedDocument, Page, Element, ContentModel
        
        # Create peter-parser document
        peter_doc = PeterParsedDocument(
            pages=[
                Page(page_number=1, text="Page 1", tables=[], images=[]),
                Page(page_number=2, text="Page 2", tables=[], images=[])
            ],
            elements=[
                Element(element_id=1, page_number=1, category="text", text="Page 1"),
                Element(element_id=2, page_number=2, category="text", text="Page 2")
            ],
            content=ContentModel(text="Page 1\n\nPage 2"),
            metadata={"total_pages": 2}
        )
        
        # Convert
        optimizer_doc = convert_peter_parsed_doc_to_optimizer(peter_doc)
        
        assert optimizer_doc.raw_content == "Page 1\n\nPage 2"
        assert len(optimizer_doc.parsed_elements) == 2
        assert optimizer_doc.metadata["total_pages"] == 2
    
    @pytest.mark.skipif(not PETER_PARSER_AVAILABLE, reason="peter-parser not available")
    def test_convert_peter_chunk(self):
        """Test converting peter-parser Chunk"""
        from peter_parser_core.common.types import Chunk as PeterChunk, ChunkMetadata
        
        # Create peter-parser chunk
        peter_chunk = PeterChunk(
            uuid="test-uuid",
            doc_title="Test Doc",
            chunk="Test chunk content",
            chunk_order=0,
            metadata=ChunkMetadata(
                page_number=1,
                chunk_size=20,
                start_index=0,
                end_index=20
            )
        )
        
        # Convert
        optimizer_chunk = convert_peter_chunk_to_optimizer(peter_chunk)
        
        assert optimizer_chunk.id == "test-uuid"
        assert optimizer_chunk.content == "Test chunk content"
        assert optimizer_chunk.metadata["chunk_order"] == 0
