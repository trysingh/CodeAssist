from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

class JobStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class ChunkMetadata(BaseModel):
    chunk_id: str
    repo_id: str
    file_path: str
    file_hash: str
    language: str
    strategy_used: str
    chunk_index: int
    total_chunks_in_file: int
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    sibling_group_id: str
    structural_context: str = ""
    token_count: int = 0

class ChunkDocument(BaseModel):
    chunk_id: str
    content: str
    metadata: ChunkMetadata

class RepoInfo(BaseModel):
    repo_id: str
    title: str
    root_path: str
    status: JobStatus
    total_files: int
    processed_files: int
    total_chunks: int
    created_at: datetime

class ChatRequest(BaseModel):
    repo_id: str
    thread_id: str
    query: str
