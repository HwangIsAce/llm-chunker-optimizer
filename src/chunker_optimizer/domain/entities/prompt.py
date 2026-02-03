"""Prompt entity for chunking instructions"""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Prompt:
    """Prompt entity for chunking instructions"""
    
    id: str
    content: str
    version: int
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        """Validate and initialize prompt"""
        if self.metadata is None:
            self.metadata = {}
        
        # Validation
        if not self.id:
            raise ValueError("id cannot be empty")
        if not self.content:
            raise ValueError("content cannot be empty")
        if self.version < 1:
            raise ValueError("version must be at least 1")
    
    def increment_version(self) -> "Prompt":
        """Create a new prompt with incremented version"""
        return Prompt(
            id=self.id,
            content=self.content,
            version=self.version + 1,
            metadata=self.metadata.copy()
        )
