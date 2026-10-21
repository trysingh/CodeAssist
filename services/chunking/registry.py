from pathlib import Path
from typing import Dict, Type, List, Tuple
from concurrent.futures import ProcessPoolExecutor
from services.chunking.base import BaseChunkingStrategy
from services.chunking.strategies import (
    ASTLanguageChunkStrategy,
    StylesheetAndConfigStrategy,
    JsonAndJsonlStrategy,
    HtmlStrategy,
)
from models.schemas import ChunkDocument

class ChunkingRegistry:
    """Pluggable Strategy Registry allowing dynamic registration of new file extensions."""
    
    _strategies: Dict[str, BaseChunkingStrategy] = {
        ".py": ASTLanguageChunkStrategy(".py"),
        ".js": ASTLanguageChunkStrategy(".js"),
        ".jsx": ASTLanguageChunkStrategy(".js"),
        ".tsx": ASTLanguageChunkStrategy(".tsx"),
        ".ts": ASTLanguageChunkStrategy(".ts"),
        ".java": ASTLanguageChunkStrategy(".java"),
        ".css": StylesheetAndConfigStrategy(".css"),
        ".yaml": StylesheetAndConfigStrategy(".yaml"),
        ".yml": StylesheetAndConfigStrategy(".yml"),
        ".md": StylesheetAndConfigStrategy(".md"),
        ".json": JsonAndJsonlStrategy(".json"),
        ".jsonl": JsonAndJsonlStrategy(".jsonl"),
        ".html": HtmlStrategy(),
        ".htm": HtmlStrategy(),
    }

    @classmethod
    def register_strategy(cls, extension: str, strategy: BaseChunkingStrategy) -> None:
        cls._strategies[extension.lower()] = strategy

    @classmethod
    def supported_extensions(cls) -> List[str]:
        return list(cls._strategies.keys())

    @classmethod
    def get_strategy(cls, extension: str) -> BaseChunkingStrategy:
        ext = extension.lower()
        if ext in cls._strategies:
            return cls._strategies[ext]
        return StylesheetAndConfigStrategy(ext)

def _worker_chunk_single_file(args: Tuple[str, str, str, str, int, int]) -> Tuple[str, str, List[ChunkDocument], str]:
    """Top-level multiprocessing worker function for parallel execution across CPU cores."""
    repo_id, full_path, rel_path, file_hash, chunk_size, chunk_overlap = args
    try:
        ext = Path(rel_path).suffix.lower()
        content = Path(full_path).read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            return (rel_path, file_hash, [], "")
        strategy = ChunkingRegistry.get_strategy(ext)
        chunks = strategy.chunk_file(repo_id, rel_path, file_hash, content, chunk_size, chunk_overlap)
        return (rel_path, file_hash, chunks, "")
    except Exception as exc:
        return (rel_path, file_hash, [], str(exc))

def parallel_chunk_files(
    tasks: List[Tuple[str, str, str, str, int, int]],
    max_workers: int = 4
) -> List[Tuple[str, str, List[ChunkDocument], str]]:
    """Executes file reading and Rust-accelerated chunking across multiple processes."""
    if not tasks:
        return []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_worker_chunk_single_file, tasks))
    return results
