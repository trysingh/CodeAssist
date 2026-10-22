import uuid
from pathlib import Path
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, File, Form, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse, Response, JSONResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from controllers.engine_controller import EngineController
from models.schemas import ChatRequest
from config.settings import settings
from services.notifier import notifier

app = FastAPI(title="CodeRAG Pluggable Chunking & Multi-Thread RAG API")
controller = EngineController()

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

class CreateThreadRequest(BaseModel):
    repo_id: str
    title: str

class ChangeRequest(BaseModel):
    request: str
    base_rev: int | None = None

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

@app.post("/api/v1/chat/attach")
async def chat_with_attachments(
    repo_id: str = Form(...),
    thread_id: str = Form(...),
    query: str = Form(""),
    files: List[UploadFile] = File(...),
):
    """Chat turn with pasted/uploaded attachments (image, pdf, docx, text). Extracted text is saved on the thread."""
    if len(files) > settings.max_attachments_per_message:
        raise HTTPException(status_code=413, detail=f"Max {settings.max_attachments_per_message} attachments per message.")
    limit = settings.max_attachment_mb * 1024 * 1024
    raw = []
    for f in files:
        data = await f.read()
        if len(data) > limit:
            raise HTTPException(status_code=413, detail=f"{f.filename} exceeds {settings.max_attachment_mb} MB.")
        raw.append((f.filename or "attachment", f.content_type or "", data))

    await controller.db.init_db()
    if not await controller.db.get_repo(repo_id):
        raise HTTPException(status_code=404, detail=f"Repository ID '{repo_id}' not found.")
    try:
        prepared = await controller.prepare_attachments(raw)   # OCR / text extraction
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))

    query = query.strip()
    await controller.db.create_thread(thread_id, repo_id, thread_id)
    msg_id = await controller.db.save_chat_message(thread_id, "user", query)
    saved = await controller.store_attachments(thread_id, msg_id, prepared)

    response = await controller.chat_with_repo(repo_id, thread_id, query or "Please review the attached content.")
    await controller.db.save_chat_message(thread_id, "assistant", response)
    return {"repo_id": repo_id, "thread_id": thread_id, "answer": response, "attachments": saved}

@app.post("/api/v1/repos/{repo_id}/changes")
async def propose_change(repo_id: str, req: ChangeRequest):
    """Skips the ask/change router: always runs the change agents. Returns a proposal; writes nothing to disk."""
    try:
        res = await controller.propose_change(repo_id, req.request, req.base_rev)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {**res.__dict__, "markdown": res.to_markdown()}

@app.get("/api/v1/repos/{repo_id}/revisions")
async def list_revisions(repo_id: str):
    return await controller.list_revisions(repo_id)

@app.get("/api/v1/repos/{repo_id}/revisions/{rev}/diff", response_class=PlainTextResponse)
async def revision_diff(repo_id: str, rev: int):
    diff = await controller.revision_diff(repo_id, rev)
    if diff is None:
        raise HTTPException(status_code=404, detail="Revision not found")
    return diff

@app.get("/api/v1/repos/{repo_id}/revisions/{rev}/download")
async def revision_download(repo_id: str, rev: int):
    data = await controller.revision_zip(repo_id, rev)
    if data is None:
        raise HTTPException(status_code=404, detail="Revision not found")
    return Response(content=data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{repo_id}_rev{rev}.zip"'})

@app.post("/api/v1/repos/{repo_id}/revisions/{rev}/apply")
async def apply_revision(repo_id: str, rev: int, force: bool = False):
    """Writes the revision into the repo folder. 409 + conflicts if files changed on disk since ingest."""
    try:
        res = await controller.apply_revision(repo_id, rev, force)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(res, status_code=200 if res["ok"] else 409)

@app.post("/api/v1/repos/{repo_id}/revisions/revert")
async def revert_revision(repo_id: str, to_rev: int = 0):
    """Appends a revision restoring the repo view to `to_rev` (0 = what is on disk)."""
    try:
        return {"revision": await controller.revert_revision(repo_id, to_rev)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/v1/notifications/stream")
async def notification_stream():
    return EventSourceResponse(notifier.subscribe())