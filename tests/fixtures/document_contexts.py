"""Document context fixtures for testing"""
from chunker_optimizer.domain.entities.document_context import (
    DocumentEnrichment,
    ParsedDocument,
    VLMOutput,
    LLMSummary
)
from chunker_optimizer.domain.entities.chunking_context import ChunkingContext
from chunker_optimizer.domain.entities.prompt import Prompt


class DocumentContextFactory:
    """Factory for creating document contexts for testing"""
    
    @staticmethod
    def create_parsed_document(
        content: str = None,
        elements: list = None
    ) -> ParsedDocument:
        """Create parsed document"""
        if content is None:
            content = "This is a sample document with multiple paragraphs."
        
        if elements is None:
            elements = [
                {"type": "paragraph", "content": content, "index": 0}
            ]
        
        return ParsedDocument(
            raw_content=content,
            parsed_elements=elements,
            metadata={"source": "test"}
        )
    
    @staticmethod
    def create_vlm_output(
        page_id: str = "page_1",
        script: str = None
    ) -> VLMOutput:
        """Create VLM output"""
        if script is None:
            script = """
            This page contains a title at the top, followed by a paragraph.
            There is an image on the right side with a caption below it.
            The layout is two-column with text on the left.
            """
        
        return VLMOutput(
            page_id=page_id,
            script=script.strip(),
            visual_elements=[
                {"type": "title", "position": "top", "text": "Sample Title"},
                {"type": "paragraph", "position": "left"},
                {"type": "image", "position": "right", "caption": "Sample Image"}
            ],
            metadata={"model": "test_vlm", "confidence": 0.9}
        )
    
    @staticmethod
    def create_llm_summary(
        summary: str = None,
        key_points: list = None
    ) -> LLMSummary:
        """Create LLM summary"""
        if summary is None:
            summary = "This document discusses the main topic with three key sections."
        
        if key_points is None:
            key_points = [
                "Introduction to the topic",
                "Main discussion points",
                "Conclusion and future work"
            ]
        
        return LLMSummary(
            summary=summary,
            key_points=key_points,
            topics=["Topic A", "Topic B"],
            structure_info={
                "sections": 3,
                "paragraphs": 5,
                "hierarchical": True
            },
            metadata={"model": "test_llm", "temperature": 0.7}
        )
    
    @staticmethod
    def create_document_enrichment(
        chunk_unit: str = "page",
        include_vlm: bool = True,
        include_llm_summary: bool = True
    ) -> DocumentEnrichment:
        """Create document enrichment"""
        parsed_doc = DocumentContextFactory.create_parsed_document()
        
        vlm_outputs = []
        if include_vlm and chunk_unit == "page":
            vlm_outputs = [
                DocumentContextFactory.create_vlm_output("page_1"),
                DocumentContextFactory.create_vlm_output("page_2")
            ]
        
        llm_summary = None
        if include_llm_summary:
            llm_summary = DocumentContextFactory.create_llm_summary()
        
        return DocumentEnrichment(
            parsed_document=parsed_doc,
            vlm_outputs=vlm_outputs,
            llm_summary=llm_summary,
            enrichment_metadata={"chunk_unit": chunk_unit}
        )
    
    @staticmethod
    def create_chunking_context(
        chunk_unit: str = "page",
        prompt: Prompt = None,
        include_vlm: bool = True,
        include_llm_summary: bool = True
    ) -> ChunkingContext:
        """Create chunking context"""
        if prompt is None:
            prompt = Prompt(
                id="test_prompt",
                content="Split the document into meaningful chunks.",
                version=1
            )
        
        enrichment = DocumentContextFactory.create_document_enrichment(
            chunk_unit=chunk_unit,
            include_vlm=include_vlm,
            include_llm_summary=include_llm_summary
        )
        
        return ChunkingContext(
            document_enrichment=enrichment,
            prompt=prompt,
            chunk_unit=chunk_unit,
            additional_context={}
        )


# Convenience fixtures for common scenarios
def create_page_chunking_context() -> ChunkingContext:
    """Create page chunking context"""
    return DocumentContextFactory.create_chunking_context(
        chunk_unit="page",
        include_vlm=True,
        include_llm_summary=True
    )


def create_element_chunking_context() -> ChunkingContext:
    """Create element chunking context"""
    return DocumentContextFactory.create_chunking_context(
        chunk_unit="element",
        include_vlm=False,
        include_llm_summary=True
    )


def create_lifelog_chunking_context() -> ChunkingContext:
    """Create lifelog chunking context"""
    return DocumentContextFactory.create_chunking_context(
        chunk_unit="lifelog",
        include_vlm=False,
        include_llm_summary=False
    )
