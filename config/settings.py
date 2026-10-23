from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AliasChoices, Field
from typing import List

DEFAULT_MODEL = "gemma4:31b-cloud"

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
    ollama_base_url_for_sys1_model: str = Field(default="http://localhost:11434/v1/systemone")   # full URL, /v1, or base: all accepted
    decision_timeout_s: float = Field(default=10.0)
    decision_cooldown_s: int = Field(default=60)       # skip a failing decision endpoint for this long
    decision_max_state_chars: int = Field(default=2000)

    # ---- web search / scraping (optional context for chat) ----
    web_enabled: bool = True                           # master switch; False = never touch the internet
    searxng_url: str = Field(default="http://localhost:8080/search",
                             validation_alias=AliasChoices("ENGINE_SEARXNG_URL", "SEARXNG_URL"))
    web_max_results: int = 5                           # results requested from SearXNG
    web_max_pages: int = 3                             # pages actually scraped per question
    web_page_chars: int = 5000                         # per-page text budget sent to the LLM
    web_search_timeout_s: float = 10.0
    web_timeout_s: float = 12.0                        # whole search + scrape budget per question
    web_auto_threshold: float = 0.75                   # P(needs web) required in "auto" mode inside a repo
    web_allow_playwright: bool = False                 # browser fallback: +4-12s per page, off for chat
    web_min_words: int = 120                           # scraper early-exit threshold
    scraper_timeout: int = 15
    playwright_headless: bool = True
    scrape_fast_deadline: float = 2.5
    scrape_max_concurrent_heavy: int = 3

    llm_model: str = Field(default=DEFAULT_MODEL)
    embedding_model: str = Field(default="nomic-embed-text:latest")
    
    # Attachments (pasted images / files / large text)
    vision_model: str = Field(default="")             # multimodal model used for image OCR; empty = llm_model
    max_attachment_mb: int = Field(default=15)
    max_attachments_per_message: int = Field(default=5)
    attachment_inline_chars: int = Field(default=24000)  # thread attachments below this total go in full; above -> retrieval
    attachment_top_k: int = Field(default=6)
    # A request written inside an uploaded image (error, pasted chat, task) is acted on when the typed message is empty or vague
    attachment_request_enabled: bool = Field(default=True)
    attachment_request_kinds: List[str] = Field(default=["image"])      # attachment kinds read for a request
    attachment_request_timeout_s: float = Field(default=25.0)           # chat-model budget for pulling the request out

    # Chunking & Parallelism Configuration
    chunk_size: int = Field(default=1200)
    chunk_overlap: int = Field(default=200)
    max_workers: int = Field(default=4)
    batch_size: int = Field(default=32)

    # Code change agents (Phase 1). Models run on the same Ollama endpoint as the others.
    router_model: str = Field(default="tev1:0.8b")     # DECISION model (/v1/systemone): ask vs change, failure triage. empty = rules only
    router_min_confidence: float = Field(default=0.75)
    patcher_model: str = Field(default="")             # edit-block author; empty = llm_model
    change_max_attempts: int = Field(default=3)        # patch -> apply -> check retries
    change_max_files: int = Field(default=6)           # files shown to the patcher
    change_context_chars: int = Field(default=60000)   # raw-file budget per prompt
    change_num_ctx: int = Field(default=32768)
    change_max_output_tokens: int = Field(default=8192)
    change_plan_first: bool = Field(default=True)      # /change writes a short plan first (one extra model call); /plan always works
    change_call_check: bool = Field(default=True)      # fail a patch that calls methods no repo class defines
    change_author_tests: bool = Field(default=True)    # after the edits pass the static checks, a model writes pytest tests for the changed code;
                                                       # they run in the sandbox and must fail on the old code and pass on the new (one extra model call)
    change_author_max_files: int = Field(default=3)    # changed source files shown to the test author
    change_require_fail_to_pass: bool = Field(default=True)  # a generated test that also passes on the OLD code gets one rewrite request

    # Sandbox (Phase 2): isolated test runs for proposed changes
    sandbox_enabled: bool = Field(default=True)
    sandbox_backend: str = Field(default="docker")     # docker | local  (local = NO isolation: dev/tests only)
    sandbox_base_image: str = Field(default="python:3.12-slim")
    sandbox_image_prefix: str = Field(default="coderag-sbx")
    sandbox_memory: str = Field(default="1g")
    sandbox_cpus: str = Field(default="2")
    sandbox_pids: int = Field(default=256)
    sandbox_test_timeout_s: int = Field(default=180)
    sandbox_build_timeout_s: int = Field(default=1500)
    sandbox_max_workspace_mb: int = Field(default=64)
    sandbox_requirements_files: List[str] = Field(default=["requirements-test.txt", "requirements-dev.txt", "requirements.txt"])

     # Sandbox pre-warm: build the dependency image in the background instead of on the first change request
    sandbox_prewarm: bool = Field(default=True)          # warm at startup, after ingest, and when a requirements file changes
    sandbox_prewarm_max_repos: int = Field(default=5)    # at startup warm this many repos, most recently chatted-with first
    sandbox_prune_images: bool = Field(default=False)    # after the startup warm-up remove coderag-sbx images no repo needs
    sandbox_available_ttl_s: int = Field(default=60)     # cache the "is Docker up" check (a "down" answer for at most 10 s); 0 = off
    sandbox_build_retry_s: int = Field(default=600)
    sandbox_logs_keep: int = Field(default=20)            # raw console logs of sandbox runs kept per repo under <storage_dir>/sandbox_logs
    sandbox_build_run: List[str] = Field(default=[])      # extra `RUN <cmd>` lines in the image build (the only step with network),
                                                          # e.g. ["python -m nltk.downloader punkt"]; tiktoken's vocabulary is fetched automatically

    # Notifiations Configuration
    notify_at_os_level: bool = Field(default=False)

    # Telemetry & logs: every notification is stored in SQLite and replayed to the Telemetry tab on connect
    telemetry_persist: bool = Field(default=True)         # False = live stream only (nothing is stored or replayed)
    telemetry_history_default: int = Field(default=100)   # last N events replayed when the UI does not send its own N
    telemetry_history_max: int = Field(default=1000)      # hard ceiling on the N a client may ask for
    telemetry_max_rows: int = Field(default=20000)        # table is pruned to the newest rows (all repos together)
    telemetry_prune_every: int = Field(default=200)       # prune check after this many inserts

    # Raw (verbatim) file store: what may be persisted
    raw_store_max_file_kb: int = Field(default=512)
    raw_store_exclude_patterns: List[str] = Field(default=[
        "*.env", ".env*", "*.pem", "*.key", "*.pfx", "*.p12", "*.crt", "id_rsa*", "id_ed25519*",
        "*cookie*", "*secret*", "*credential*", "*token*", "*.kdbx", "*.patch", "*.diff"
    ])

    # Exclusions Configuration
    exclude_dirs: List[str] = Field(default=[
        ".git", "node_modules", "venv", ".venv", "__pycache__", 
        "dist", "build", ".engine_storage", ".idea", ".vscode", "target", "data", "workspaces"
    ])
    exclude_patterns: List[str] = Field(default=[
        "* copy.*", 
        "*.pyc", 
        "*.log", 
        ".DS_Store", 
        "*.lock", 
        "package-lock.json", 
        "*.db", "*.sqlite", 
        "*.sqlite3", 
        "*.tar.gz", 
        "*.zip", 
        "*.egg-info", 
        "*.egg", 
        "*.whl",
        "*.patch", 
        "*.diff", 
        "*.orig", 
        "*.rej"
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