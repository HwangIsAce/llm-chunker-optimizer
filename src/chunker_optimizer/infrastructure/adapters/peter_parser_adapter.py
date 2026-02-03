"""Adapter for integrating peter-parser with llm-chunker-optimizer"""
import sys
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional

# Add peter-parser to path if needed
PETER_PARSER_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "pipelines" / "parsing-pipeline" / "peter-parser"
if str(PETER_PARSER_PATH) not in sys.path:
    sys.path.insert(0, str(PETER_PARSER_PATH))

try:
    from peter_parser_core.common.types import ParsedDocument as PeterParsedDocument, Chunk as PeterChunk
    from peter_parser.graph.flow import PipelineFlow
    from peter_parser.graph.states import PipelineState, DocumentType
    from peter_parser.impl.chunker.vlm import VLMChunker
    from peter_parser.impl.chunker.lumber import LumberChunker
    from peter_parser.impl.chunker.lifelog import LifelogChunker
    from peter_parser.impl.db.lifelog_store import LifelogStore
    PETER_PARSER_AVAILABLE = True
except ImportError:
    PETER_PARSER_AVAILABLE = False
    # Create dummy types for type hints
    PeterParsedDocument = Any
    PeterChunk = Any
    DocumentType = Optional[str]  # Type alias for when peter-parser is not available

from chunker_optimizer.domain.entities.chunking_context import ChunkingContext
from chunker_optimizer.domain.entities.chunk import Chunk
from chunker_optimizer.domain.entities.document_context import ParsedDocument as OptimizerParsedDocument


def convert_peter_parsed_doc_to_optimizer(
    peter_doc: PeterParsedDocument
) -> OptimizerParsedDocument:
    """
    Convert peter-parser ParsedDocument to llm-chunker-optimizer ParsedDocument
    
    Args:
        peter_doc: peter-parser ParsedDocument
        
    Returns:
        llm-chunker-optimizer ParsedDocument
    """
    # Extract raw content
    raw_content = peter_doc.content_text
    
    # Convert elements to parsed_elements format
    parsed_elements = []
    for element in peter_doc.elements:
        parsed_elements.append({
            "element_id": element.element_id,
            "page_number": element.page_number,
            "category": element.category,
            "text": element.text,
            "coordinates": element.coordinates or [],
            "enrichment_metadata": element.enrichment_metadata or {},
            "chunk_uuid": element.chunk_uuid
        })
    
    # Extract metadata
    metadata = peter_doc.metadata.copy() if peter_doc.metadata else {}
    metadata["total_pages"] = len(peter_doc.pages)
    metadata["total_elements"] = len(peter_doc.elements)
    
    return OptimizerParsedDocument(
        raw_content=raw_content,
        parsed_elements=parsed_elements,
        metadata=metadata
    )


def convert_peter_chunk_to_optimizer(
    peter_chunk: PeterChunk,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None
) -> Chunk:
    """
    Convert peter-parser Chunk to llm-chunker-optimizer Chunk
    
    Args:
        peter_chunk: peter-parser Chunk
        start_index: Optional start index (if not in metadata)
        end_index: Optional end index (if not in metadata)
        
    Returns:
        llm-chunker-optimizer Chunk
    """
    # Extract metadata
    chunk_metadata = peter_chunk.metadata
    if hasattr(chunk_metadata, "model_dump"):
        metadata_dict = chunk_metadata.model_dump()
    elif hasattr(chunk_metadata, "dict"):
        metadata_dict = chunk_metadata.dict()
    else:
        metadata_dict = {}
    
    # Get start/end indices
    start = start_index or metadata_dict.get("start_index", 0)
    end = end_index or metadata_dict.get("end_index", len(peter_chunk.chunk))
    
    # Build metadata
    metadata = {
        "uuid": peter_chunk.uuid,
        "chunk_order": peter_chunk.chunk_order,
        "doc_title": peter_chunk.doc_title,
        **metadata_dict
    }
    
    return Chunk(
        id=peter_chunk.uuid,
        content=peter_chunk.chunk,
        start_index=start,
        end_index=end,
        metadata=metadata
    )


def create_peter_parser_chunking_function(
    chunk_unit: str = "page",
    document_type: Optional[str] = None,  # DocumentType when peter-parser is available
    prompt_content: Optional[str] = None
) -> Callable[[ChunkingContext], List[Chunk]]:
    """
    Create a chunking function that uses peter-parser's actual chunking logic
    
    Args:
        chunk_unit: Chunk unit ("page", "element", "lifelog")
        document_type: Optional document type for 4-case routing
        prompt_content: Optional prompt content (currently not used by peter-parser chunkers)
        
    Returns:
        Function that takes ChunkingContext and returns List[Chunk]
        
    Raises:
        ImportError: If peter-parser is not available
    """
    if not PETER_PARSER_AVAILABLE:
        raise ImportError(
            "peter-parser is not available. Please ensure it's installed and accessible."
        )
    
    # Determine which chunker to use
    use_slide = document_type == "slide" or chunk_unit == "page"
    use_lifelog = document_type == "lifelog" or chunk_unit == "lifelog"
    
    # Initialize chunker
    if use_lifelog:
        try:
            lifelog_store = LifelogStore()
        except Exception:
            lifelog_store = None
        chunker = LifelogChunker(lifelog_store=lifelog_store)
    elif use_slide:
        chunker = VLMChunker()
    else:
        chunker = LumberChunker()
    
    def chunking_function(context: ChunkingContext) -> List[Chunk]:
        """
        Chunking function that uses peter-parser's chunker
        
        Args:
            context: ChunkingContext with document enrichment and prompt
            
        Returns:
            List of Chunk objects
        """
        # Convert optimizer ParsedDocument to peter-parser format
        # For now, we'll need to reconstruct a minimal peter-parser ParsedDocument
        # This is a simplified conversion - in production, you might want a more complete conversion
        
        # Get enrichment data
        enrichment = context.document_enrichment
        optimizer_parsed_doc = enrichment.parsed_document
        
        # Create a minimal peter-parser ParsedDocument
        # Note: This is a simplified approach. In production, you might want to store
        # the original peter-parser ParsedDocument or have a more complete conversion.
        from peter_parser_core.common.types import Page, Element, ContentModel
        
        # Reconstruct pages from raw_content (simplified)
        pages = []
        page_texts = optimizer_parsed_doc.raw_content.split("\n\n")
        for i, page_text in enumerate(page_texts):
            pages.append(Page(
                page_number=i + 1,
                text=page_text,
                tables=[],
                images=[]
            ))
        
        # Reconstruct elements
        elements = []
        for elem_data in optimizer_parsed_doc.parsed_elements:
            from peter_parser_core.common.types import Element as PeterElement
            elements.append(PeterElement(
                element_id=elem_data.get("element_id", len(elements)),
                page_number=elem_data.get("page_number", 1),
                category=elem_data.get("category", "text"),
                text=elem_data.get("text", ""),
                coordinates=elem_data.get("coordinates"),
                enrichment_metadata=elem_data.get("enrichment_metadata"),
                chunk_uuid=elem_data.get("chunk_uuid")
            ))
        
        # Create peter-parser ParsedDocument
        peter_parsed_doc = PeterParsedDocument(
            pages=pages,
            elements=elements,
            content=ContentModel(text=optimizer_parsed_doc.raw_content),
            metadata=optimizer_parsed_doc.metadata or {}
        )
        
        # Get document_summary and item_metadata from enrichment
        document_summary = None
        if enrichment.has_llm_summary():
            document_summary = enrichment.llm_summary.summary
        
        item_metadata = {}
        if enrichment.has_vlm_output():
            # Convert VLM outputs to item_metadata format
            # VLM outputs are keyed by page_id, but item_metadata is keyed by element_id
            # For now, we'll create a simple mapping
            for vlm_output in enrichment.vlm_outputs:
                # Find elements on this page
                page_num = int(vlm_output.page_id.split("_")[-1]) if "_" in vlm_output.page_id else 1
                for elem_data in optimizer_parsed_doc.parsed_elements:
                    if elem_data.get("page_number") == page_num:
                        element_id = elem_data.get("element_id", len(item_metadata))
                        item_metadata[element_id] = {
                            "vlm_script": vlm_output.script,
                            "visual_elements": vlm_output.visual_elements,
                            **vlm_output.metadata
                        }
        
        # Detect boundaries
        if use_slide:
            boundaries = chunker.detect_boundaries(
                parsed_document=peter_parsed_doc,
                document_summary=document_summary or "",
                item_metadata=item_metadata
            )
        else:
            boundaries = chunker.detect_boundaries(
                parsed_document=peter_parsed_doc
            )
        
        # Create chunks
        doc_title = optimizer_parsed_doc.metadata.get("title", "Document") if optimizer_parsed_doc.metadata else "Document"
        peter_chunks, _ = chunker.chunk(
            parsed_document=peter_parsed_doc,
            chunk_boundaries=boundaries,
            doc_title=doc_title
        )
        
        # Convert to optimizer chunks
        optimizer_chunks = []
        current_index = 0
        for peter_chunk in peter_chunks:
            # Calculate start/end indices based on content position
            start_index = current_index
            end_index = start_index + len(peter_chunk.chunk)
            current_index = end_index + 1  # +1 for separator
            
            optimizer_chunk = convert_peter_chunk_to_optimizer(
                peter_chunk,
                start_index=start_index,
                end_index=end_index
            )
            optimizer_chunks.append(optimizer_chunk)
        
        return optimizer_chunks
    
    return chunking_function


class PeterParserAdapter:
    """Adapter class for peter-parser integration"""
    
    def __init__(self):
        """Initialize adapter"""
        if not PETER_PARSER_AVAILABLE:
            raise ImportError(
                "peter-parser is not available. Please ensure it's installed and accessible."
            )
    
    @staticmethod
    def create_chunking_function(
        chunk_unit: str = "page",
        document_type: Optional[str] = None,  # DocumentType when peter-parser is available
        prompt_content: Optional[str] = None
    ) -> Callable[[ChunkingContext], List[Chunk]]:
        """
        Create a chunking function using peter-parser
        
        Args:
            chunk_unit: Chunk unit
            document_type: Optional document type
            prompt_content: Optional prompt content
            
        Returns:
            Chunking function
        """
        return create_peter_parser_chunking_function(
            chunk_unit=chunk_unit,
            document_type=document_type,
            prompt_content=prompt_content
        )
    
    @staticmethod
    def run_pipeline(
        document: bytes,
        chunk_unit: Optional[str] = None,
        document_type: Optional[str] = None  # DocumentType when peter-parser is available
    ) -> Any:  # PipelineState when peter-parser is available
        """
        Run peter-parser pipeline
        
        Args:
            document: Document bytes
            chunk_unit: Optional chunk unit
            document_type: Optional document type
            
        Returns:
            PipelineState
        """
        if not PETER_PARSER_AVAILABLE:
            raise ImportError("peter-parser is not available")
        
        pipeline = PipelineFlow()
        return pipeline.invoke(
            document=document,
            chunk_unit=chunk_unit,
            document_type=document_type
        )
