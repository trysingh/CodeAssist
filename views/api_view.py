import uuid
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from controllers.engine_controller import EngineController
from models.schemas import ChatRequest
from services.notifier import notifier

app = FastAPI(title="CodeRAG Pluggable Chunking & Multi-Thread RAG API")
controller = EngineController()

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

class CreateThreadRequest(BaseModel):
    repo_id: str
    title: str

@app.on_event("startup")
async def startup_event():
    await controller.db.init_db()
    await controller.restore_all_watchers()

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serves the responsive dark-themed CodeRAG web dashboard."""
    return templates.TemplateResponse(request, "index.html")

@app.get("/static/{file_type}/{file_name}")
async def serve_static(file_type: str, file_name: str):
    """Custom static file handler that safely bypasses Windows path parenthesis bugs."""
    file_path = BASE_DIR / "static" / file_type / file_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Static asset not found")
    media_type = "text/css" if file_type == "css" else "application/javascript"
    return FileResponse(path=str(file_path), media_type=media_type)

@app.post("/api/v1/ingest")
async def trigger_ingest(repo_root: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(controller.ingest_repository, repo_root)
    return {"message": "Ingestion job queued in background", "repo_root": repo_root}

@app.get("/api/v1/repos")
async def list_repos():
    return await controller.list_ingested_repos()

@app.get("/api/v1/repos/{repo_id}/threads")
async def list_repo_threads(repo_id: str):
    """Returns all chat threads created for a specific repository."""
    await controller.db.init_db()
    return await controller.db.get_threads_for_repo(repo_id)

@app.post("/api/v1/repos/threads")
async def create_repo_thread(req: CreateThreadRequest):
    """Creates a new conversation thread for a repository."""
    await controller.db.init_db()
    thread_id = f"th_{uuid.uuid4().hex[:10]}"
    await controller.db.create_thread(thread_id, req.repo_id, req.title)
    return {"thread_id": thread_id, "repo_id": req.repo_id, "title": req.title}

@app.get("/api/v1/threads/{thread_id}/messages")
async def get_thread_history(thread_id: str):
    """Returns the full message history for a specific thread."""
    await controller.db.init_db()
    return await controller.db.get_thread_messages(thread_id)

@app.post("/api/v1/chat")
async def chat_endpoint(req: ChatRequest):
    """Processes a user query inside a specific repository thread and persists messages."""
    try:
        await controller.db.init_db()
        # Ensure thread record exists even if created ad-hoc
        await controller.db.create_thread(req.thread_id, req.repo_id, req.thread_id)
        # Save user message
        await controller.db.save_chat_message(req.thread_id, "user", req.query)
        
        # Run RAG workflow
        response = await controller.chat_with_repo(req.repo_id, req.thread_id, req.query)
        
        # Save assistant response
        await controller.db.save_chat_message(req.thread_id, "assistant", response)
        return {"repo_id": req.repo_id, "thread_id": req.thread_id, "answer": response}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/v1/notifications/stream")
async def notification_stream():
    return EventSourceResponse(notifier.subscribe())