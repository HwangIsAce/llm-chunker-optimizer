"""Document context entities for enrichment data"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class ParsedDocument:
    """Parsed document information"""
    
    raw_content: str
    parsed_elements: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and initialize parsed document"""
        if self.metadata is None:
            self.metadata = {}
        if self.parsed_elements is None:
            self.parsed_elements = []
        
        if not self.raw_content:
            raise ValueError("raw_content cannot be empty")


@dataclass
class VLMOutput:
    """VLM (Vision Language Model) output result"""
    
    page_id: str
    script: str
    visual_elements: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and initialize VLM output"""
        if self.metadata is None:
            self.metadata = {}
        if self.visual_elements is None:
            self.visual_elements = []
        
        if not self.page_id:
            raise ValueError("page_id cannot be empty")
        if not self.script:
            raise ValueError("script cannot be empty")


@dataclass
class LLMSummary:
    """LLM summary information"""
    
    summary: str
    key_points: List[str]
    topics: List[str]
    structure_info: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and initialize LLM summary"""
        if self.metadata is None:
            self.metadata = {}
        if self.key_points is None:
            self.key_points = []
        if self.topics is None:
            self.topics = []
        if self.structure_info is None:
            self.structure_info = {}
        
        if not self.summary:
            raise ValueError("summary cannot be empty")


@dataclass
class DocumentEnrichment:
    """Document enrichment result"""
    
    parsed_document: ParsedDocument
    vlm_outputs: Optional[List[VLMOutput]] = None
    llm_summary: Optional[LLMSummary] = None
    enrichment_metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and initialize document enrichment"""
        if self.vlm_outputs is None:
            self.vlm_outputs = []
        if self.enrichment_metadata is None:
            self.enrichment_metadata = {}
        
        if self.parsed_document is None:
            raise ValueError("parsed_document cannot be None")
    
    def has_vlm_output(self) -> bool:
        """Check if VLM output exists"""
        return len(self.vlm_outputs) > 0
    
    def has_llm_summary(self) -> bool:
        """Check if LLM summary exists"""
        return self.llm_summary is not None
