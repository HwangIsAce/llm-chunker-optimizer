"""Real enrichment data JSON loader"""
from pathlib import Path
import json
from typing import Optional, List
from chunker_optimizer.domain.entities.document_context import (
    DocumentEnrichment,
    ParsedDocument,
    VLMOutput,
    LLMSummary
)


class RealEnrichmentDataLoader:
    """Real enrichment data loader from JSON files"""
    
    REAL_DATA_DIR = Path(__file__).parent / "real_data"
    
    @classmethod
    def load_from_json(cls, filename: str) -> DocumentEnrichment:
        """
        Load enrichment data from JSON file
        
        Args:
            filename: JSON filename (e.g., "page_enrichment_sample.json")
        
        Returns:
            DocumentEnrichment object
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON format is invalid
        """
        filepath = cls.REAL_DATA_DIR / filename
        
        if not filepath.exists():
            available = list(cls.REAL_DATA_DIR.glob("*.json"))
            raise FileNotFoundError(
                f"Enrichment data not found: {filepath}\n"
                f"Available files: {[f.name for f in available]}"
            )
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls._dict_to_enrichment(data)
    
    @classmethod
    def load_latest(cls) -> Optional[DocumentEnrichment]:
        """
        Load most recently modified JSON file
        
        Returns:
            DocumentEnrichment or None if no files exist
        """
        json_files = list(cls.REAL_DATA_DIR.glob("*.json"))
        if not json_files:
            return None
        
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        return cls.load_from_json(latest_file.name)
    
    @classmethod
    def load_by_chunk_unit(cls, chunk_unit: str) -> Optional[DocumentEnrichment]:
        """
        Load data by chunk unit
        
        Args:
            chunk_unit: "page", "element", "lifelog", "default"
        
        Returns:
            DocumentEnrichment or None if not found
        """
        # File pattern: {chunk_unit}_enrichment*.json
        pattern = f"{chunk_unit}_enrichment*.json"
        matching_files = list(cls.REAL_DATA_DIR.glob(pattern))
        
        if not matching_files:
            return None
        
        # Use most recent file
        latest_file = max(matching_files, key=lambda p: p.stat().st_mtime)
        return cls.load_from_json(latest_file.name)
    
    @classmethod
    def list_available_data(cls) -> List[str]:
        """List available data files"""
        return [f.name for f in cls.REAL_DATA_DIR.glob("*.json")]
    
    @staticmethod
    def _dict_to_enrichment(data: dict) -> DocumentEnrichment:
        """Convert dictionary to DocumentEnrichment"""
        # ParsedDocument
        parsed_doc_data = data["parsed_document"]
        parsed_document = ParsedDocument(
            raw_content=parsed_doc_data["raw_content"],
            parsed_elements=parsed_doc_data.get("parsed_elements", []),
            metadata=parsed_doc_data.get("metadata", {})
        )
        
        # VLMOutputs
        vlm_outputs = []
        for vlm_data in data.get("vlm_outputs", []):
            vlm_outputs.append(VLMOutput(
                page_id=vlm_data["page_id"],
                script=vlm_data["script"],
                visual_elements=vlm_data.get("visual_elements", []),
                metadata=vlm_data.get("metadata", {})
            ))
        
        # LLMSummary
        llm_summary = None
        if data.get("llm_summary"):
            llm_data = data["llm_summary"]
            llm_summary = LLMSummary(
                summary=llm_data["summary"],
                key_points=llm_data.get("key_points", []),
                topics=llm_data.get("topics", []),
                structure_info=llm_data.get("structure_info", {}),
                metadata=llm_data.get("metadata", {})
            )
        
        return DocumentEnrichment(
            parsed_document=parsed_document,
            vlm_outputs=vlm_outputs,
            llm_summary=llm_summary,
            enrichment_metadata=data.get("enrichment_metadata", {})
        )
