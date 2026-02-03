"""Repository interface for chunk storage"""
from abc import ABC, abstractmethod
from typing import List, Optional
from ..entities.chunk import Chunk


class ChunkRepository(ABC):
    """Repository interface for chunk storage and retrieval"""
    
    @abstractmethod
    def save(self, chunks: List[Chunk]) -> None:
        """
        Save chunks to repository
        
        Args:
            chunks: List of chunks to save
        """
        pass
    
    @abstractmethod
    def find_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """
        Find chunk by ID
        
        Args:
            chunk_id: Chunk identifier
        
        Returns:
            Chunk if found, None otherwise
        """
        pass
    
    @abstractmethod
    def find_all(self) -> List[Chunk]:
        """
        Find all chunks
        
        Returns:
            List of all chunks
        """
        pass
    
    @abstractmethod
    def delete(self, chunk_id: str) -> bool:
        """
        Delete chunk by ID
        
        Args:
            chunk_id: Chunk identifier
        
        Returns:
            True if deleted, False if not found
        """
        pass
