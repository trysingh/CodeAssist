import uuid
import secrets
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, File, Form, UploadFile, Header
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse, Response, JSONResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse
from services.notifier import notifier, repo_scope
from controllers.engine_controller import EngineController
from models.schemas import ChatRequest
from config.settings import settings
from services.notifier import notifier
from services.model_calls import list_chat_models



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
    model: str = ""

@app.on_event("startup")
async def startup_event():
    await controller.db.init_db()
    notifier.bind(controller.db)        # persist notifications + thread-safe SSE fan-out (needs the running loop)
    # Sandbox images are built in the background (docker build runs in a worker thread); start() returns at
    # once. Started BEFORE restore_all_watchers(): that call waits up to 5 s for console input and then
    # re-syncs offline changes, and the build can run meanwhile.
    controller.warmer.start()
    await controller.restore_all_watchers()

@app.on_event("shutdown")
async def shutdown_event():
    await controller.warmer.stop()      # also kills a running `docker build` so closing the app is never held up
    await notifier.close()              # flush queued telemetry rows

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serves the responsive dark-themed CodeRAG web dashboard."""
    return templates.TemplateResponse(request, "index.html")

@app.get("/api/v1/models")
async def list_models():
    names = await list_chat_models()
    if settings.llm_model not in names:
        names.insert(0, settings.llm_model)
    return {"default": settings.llm_model, "models": names}

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
        response = await controller.chat_with_repo(req.repo_id, req.thread_id, req.query, web=req.web, model=req.model)
        
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
    web: str = Form("auto"),
    model: str = Form(""),
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

    # empty/vague text + an image that holds a request -> that request is acted on (see services/attachment_intent.py)
    response = await controller.chat_with_repo(repo_id, thread_id, query, web=web, attachments=prepared, model=model)
    await controller.db.save_chat_message(thread_id, "assistant", response)
    return {"repo_id": repo_id, "thread_id": thread_id, "answer": response, "attachments": saved}

@app.post("/api/v1/repos/{repo_id}/changes")
async def propose_change(repo_id: str, req: ChangeRequest):
    """Skips the ask/change router: always runs the change agents. Returns a proposal; writes nothing to disk."""
    try:
        res = await controller.propose_change(repo_id, req.request, req.base_rev, model=req.model)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {**res.__dict__, "markdown": res.to_markdown()}

@app.get("/api/v1/repos/{repo_id}/revisions")
async def list_revisions(repo_id: str):
    return await controller.list_revisions(repo_id)

@app.get("/api/v1/repos/{repo_id}/revisions/{rev}/diff", response_class=PlainTextResponse)
async def revision_diff(repo_id: str, rev: int, from_disk: bool = False):
    diff = await controller.revision_diff(repo_id, rev, from_disk)
    if diff is None:
        raise HTTPException(status_code=404, detail="Revision not found")
    return diff

@app.get("/api/v1/repos/{repo_id}/revisions/{rev}/patch")
async def revision_patch(repo_id: str, rev: int, from_disk: bool = True):
    """The patch as a file download (exact bytes, so nothing re-encodes or trims the final newline). By default it is
    cumulative against the repo as the engine last saw it, i.e. what a working tree matches, so it applies without the earlier
    revisions; from_disk=false gives only what this revision changed on top of the revision before it.
    Check with `git apply --check <file>`. Files that changed on disk since are named in `#` lines at the top and in the
    X-Patch-Conflicts header."""
    res = await controller.revision_patch(repo_id, rev, from_disk)
    if res is None:
        raise HTTPException(status_code=404, detail="Revision not found")
    diff, conflicts = res
    headers = {"Content-Disposition": f'attachment; filename="{repo_id}_rev{rev}.patch"'}
    if conflicts:
        headers["X-Patch-Conflicts"] = str(len(conflicts))
    return Response(content=diff.encode("utf-8"), media_type="text/x-diff; charset=utf-8", headers=headers)

@app.get("/api/v1/repos/{repo_id}/revisions/{rev}/download")
async def revision_download(repo_id: str, rev: int):
    data = await controller.revision_zip(repo_id, rev)
    if data is None:
        raise HTTPException(status_code=404, detail="Revision not found")
    return Response(content=data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{repo_id}_rev{rev}.zip"'})

@app.post("/api/v1/repos/{repo_id}/revisions/{rev}/apply")
async def apply_revision(repo_id: str, rev: int, force: bool = False,
                         x_confirm_phrase: str = Header(default="")):
    """Writes the revision into the repo folder. 409 + conflicts if files changed on disk since ingest."""
    phrase = settings.apply_confirmation_phrase
    if phrase:
        if not x_confirm_phrase:
            raise HTTPException(status_code=428, detail="Confirmation phrase required")
        if not secrets.compare_digest(x_confirm_phrase.encode(), phrase.encode()):
            raise HTTPException(status_code=403, detail="Wrong confirmation phrase")
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


@app.delete("/api/v1/threads/{thread_id}")
async def delete_thread(thread_id: str):
    await controller.db.init_db()
    if not await controller.db.delete_thread(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"status": "deleted"}


@app.get("/api/v1/notifications/stream")
async def notification_stream(repo_id: Optional[str] = None, limit: Optional[int] = None):
    """Telemetry & logs feed (SSE). First an event 'history' with the last `limit` stored events, then live
    'notification' events. General events reach everyone; repo events only when `repo_id` is that repo.
    `limit` defaults to settings.telemetry_history_default and is capped by settings.telemetry_history_max."""
    return EventSourceResponse(notifier.subscribe(repo_id=repo_id, limit=limit))


@app.post("/api/v1/repos/{repo_id}/reset-index")
async def api_reset_repo_index(repo_id: str):
    """
    Resets the index for a specific repository by clearing 
    Chroma collections and SQLite chunks.
    """
    try:
        # The controller method is async and handles the deletion logic
        await controller.reset_repo_index(repo_id)
        notifier.notify("Repository Index Reset", "Indexes reset successfull", level="INFO", repo_id=repo_id)
        return {
            "status": "success", 
            "repo_id": repo_id, 
            "message": "Index successfully reset"
        }
    except ValueError as e:
        # EngineController raises ValueError if repo_id is not found
        notifier.notify("Repository Index Reset", f"Indexes reset NOT successfull: {e}", level="ERROR", repo_id=repo_id)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Log the error and return a 500
        notifier.notify("Repository Index Reset", f"Indexes reset NOT successfull: {e}", level="ERROR", repo_id=repo_id)
        raise HTTPException(status_code=500, detail="Internal server error during index reset")