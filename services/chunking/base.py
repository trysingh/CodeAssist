from abc import ABC, abstractmethod
from typing import List
from models.schemas import ChunkDocument

class BaseChunkingStrategy(ABC):
    """Abstract base class for pluggable file-type chunking strategies."""
    
    @abstractmethod
    def chunk_file(
        self,
        repo_id: str,
        file_path: str,
        file_hash: str,
        content: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[ChunkDocument]:
        pass
