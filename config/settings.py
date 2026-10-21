from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List

class EngineSettings(BaseSettings):
    app_name: str = "CodeRAG Chunking Engine"
    storage_dir: Path = Field(default=Path("./.engine_storage"))
    sqlite_db_name: str = "metadata_and_jobs.db"
    checkpoint_db_name: str = "langgraph_checkpoints.db"
    chroma_dir_name: str = "chroma_vectors"

     # ---- web UI ----
    host: str = "0.0.0.0"          # local tool, no auth: keep it on loopback unless you add auth
    port: int = 2026 
    
    # Ollama Configuration
    ollama_base_url: str = Field(default="http://localhost:11434")
    llm_model: str = Field(default="llama3.1:latest")
    embedding_model: str = Field(default="nomic-embed-text:latest")
    
    # Attachments (pasted images / files / large text)
    vision_model: str = Field(default="")             # multimodal model used for image OCR; empty = llm_model
    max_attachment_mb: int = Field(default=15)
    max_attachments_per_message: int = Field(default=5)
    attachment_inline_chars: int = Field(default=24000)  # thread attachments below this total go in full; above -> retrieval
    attachment_top_k: int = Field(default=6)

    # Chunking & Parallelism Configuration
    chunk_size: int = Field(default=1200)
    chunk_overlap: int = Field(default=200)
    max_workers: int = Field(default=4)
    batch_size: int = Field(default=32)

    # Exclusions Configuration
    exclude_dirs: List[str] = Field(default=[
        ".git", "node_modules", "venv", ".venv", "__pycache__", 
        "dist", "build", ".engine_storage", ".idea", ".vscode", "target", "data", "workspaces"
    ])
    exclude_patterns: List[str] = Field(default=[
        "* copy.*", "*.pyc", "*.log", ".DS_Store", "*.lock", "package-lock.json", "*.db", "*.sqlite", "*.sqlite3", "*.tar.gz", "*.zip", "*.egg-info", "*.egg", "*.whl"
    ])
    
    model_config = SettingsConfigDict(env_prefix="ENGINE_", env_file=".env", extra="ignore")

    def ensure_dirs(self):
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        (self.storage_dir / self.chroma_dir_name).mkdir(parents=True, exist_ok=True)

    @property
    def sqlite_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.storage_dir / self.sqlite_db_name}"

    @property
    def checkpoint_path(self) -> str:
        return str(self.storage_dir / self.checkpoint_db_name)

    @property
    def chroma_path(self) -> str:
        return str(self.storage_dir / self.chroma_dir_name)

settings = EngineSettings()
settings.ensure_dirs()