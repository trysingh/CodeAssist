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
    ollama_base_url_for_sys1_model: str = Field(default="http://localhost:11434/v1/systemone")

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

    # Code change agents (Phase 1). Models run on the same Ollama endpoint as the others.
    router_model: str = Field(default="tev1:0.8b")              # small decision model: ask vs change. empty = keyword rules only
    router_min_confidence: float = Field(default=0.6)
    patcher_model: str = Field(default="")             # edit-block author; empty = llm_model
    change_max_attempts: int = Field(default=3)        # patch -> apply -> check retries
    change_max_files: int = Field(default=6)           # files shown to the patcher
    change_context_chars: int = Field(default=60000)   # raw-file budget per prompt
    change_num_ctx: int = Field(default=32768)
    change_max_output_tokens: int = Field(default=8192)

    # Raw (verbatim) file store: what may be persisted
    raw_store_max_file_kb: int = Field(default=512)
    raw_store_exclude_patterns: List[str] = Field(default=[
        "*.env", ".env*", "*.pem", "*.key", "*.pfx", "*.p12", "*.crt", "id_rsa*", "id_ed25519*",
        "*cookie*", "*secret*", "*credential*", "*token*", "*.kdbx"
    ])

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