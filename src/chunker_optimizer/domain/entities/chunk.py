"""Chunk entity representing a text segment"""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Chunk:
    """Chunk entity representing a text segment"""
    
    id: str
    content: str
    start_index: int
    end_index: int
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        """Validate and initialize chunk"""
        if self.metadata is None:
            self.metadata = {}
        
        # Validation
        if self.start_index < 0:
            raise ValueError("start_index must be non-negative")
        if self.end_index <= self.start_index:
            raise ValueError("end_index must be greater than start_index")
        if not self.content:
            raise ValueError("content cannot be empty")
        if not self.id:
            raise ValueError("id cannot be empty")
    
    @property
    def length(self) -> int:
        """Get chunk length"""
        return self.end_index - self.start_index
    
    def __len__(self) -> int:
        """Get chunk content length"""
        return len(self.content)
