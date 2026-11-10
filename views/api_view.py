import platform
import asyncio
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
from services.runners import RunnerConfigError, check_runner_config
from services.runner_info import check_chat_choice, models_payload, runners_payload
from services.llm_errors import ModelAuthError, ModelQuotaError, auth_error_response, quota_error_response
from services import action_log



app = FastAPI(title="CodeRAG Pluggable Chunking & Multi-Thread RAG API")
app.add_exception_handler(ModelQuotaError, quota_error_response)    # cloud model out of credits (HTTP 402 from Ollama): readable 402 instead of a 500 + traceback
app.add_exception_handler(ModelAuthError, auth_error_response)      # bad / missing provider credentials (HTTP 401/403): readable 401
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
    runner: str = ""
    token_mode: str = ""

_sink_registered = False

def register_action_sink() -> None:
    """Forward every ActionEvent to the notifier (SSE + telemetry table). Once per process, gated by settings.action_log_enabled."""
    global _sink_registered
    if settings.action_log_enabled and not _sink_registered:
        action_log.add_sink(action_log.make_notifier_sink(notifier.notify))
        _sink_registered = True

@app.on_event("startup")
async def startup_event():
    try:
        check_runner_config()           # fail fast: e.g. LLM_RUNNER=cloudflare without CF_ACCOUNT_ID / CF_API_TOKEN
    except RunnerConfigError as exc:
        print(f"[startup] {exc}", flush=True)
        raise
    await controller.db.init_db()
    notifier.bind(controller.db)        # persist notifications + thread-safe SSE fan-out (needs the running loop)
    register_action_sink()
    try:
        await controller.reconcile_jobs()   # no vectorize job survives a restart: close stale rows, fix vector_status
    except Exception as exc:                # noqa: BLE001 - never block startup on bookkeeping
        print(f"[startup] job reconcile failed: {type(exc).__name__}: {exc}", flush=True)
    # Sandbox images are built in the background (docker build runs in a worker thread); start() returns at
    # once. Started BEFORE restore_all_watchers(): that call waits up to 5 s for console input and then
    # re-syncs offline changes, and the build can run meanwhile.
    controller.warmer.start()
    await controller.restore_all_watchers()
    print(" = " * 30)
    print(f"       Ollama base is looked at from {settings.ollama_base_url}", flush=True)
    print(" = " * 30)

@app.on_event("shutdown")
async def shutdown_event():
    await controller.warmer.stop()      # also kills a running `docker build` so closing the app is never held up
    await notifier.close()              # flush queued telemetry rows

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serves the responsive dark-themed CodeRAG web dashboard."""
    return templates.TemplateResponse(request, "index.html")

@app.get("/api/v1/runners")
async def list_runners():
    """Every registered runner for the UI switch: {"default", "runners": [{name, label, remote, configured, error}]}."""
    return runners_payload()

@app.get("/api/v1/models")
async def list_models(runner: Optional[str] = None):
    """Chat models of `runner` (default: the configured runner) and the default model that runner resolves to."""
    try:
        return await models_payload(runner)
    except RunnerConfigError as e:
        raise HTTPException(status_code=400, detail=str(e))

def _chat_choice(runner: str, token_mode: str):
    """Per-request runner / token_mode from the UI -> validated (runner, token_mode); a clear 400 when unusable."""
    try:
        return check_chat_choice(runner, token_mode)
    except (RunnerConfigError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/static/{file_type}/{file_name}")
async def serve_static(file_type: str, file_name: str):
    """Custom static file handler that safely bypasses Windows path parenthesis bugs."""
    file_path = BASE_DIR / "static" / file_type / file_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Static asset not found")
    media_type = "text/css" if file_type == "css" else "application/javascript"
    return FileResponse(path=str(file_path), media_type=media_type)

async def _run_ingest(repo_root: str) -> None:
    try:
        await controller.ingest_repository(repo_root)
    except Exception as exc:     # noqa: BLE001 - a background task has nobody to raise to: tell the user
        notifier.notify("Ingestion failed", f"{repo_root}: {type(exc).__name__}: {exc}", level="ERROR")

@app.post("/api/v1/ingest")
async def trigger_ingest(repo_root: str, background_tasks: BackgroundTasks):
    """Fast agentic ingest in the background (searchable in seconds); watch the notification stream for 'Repo ready'."""
    repo_id = controller.compute_repo_id(Path(repo_root))     # the same id the controller derives
    background_tasks.add_task(_run_ingest, repo_root)
    return {"message": "Ingestion job queued in background", "repo_root": repo_root, "repo_id": repo_id}

@app.get("/api/v1/repos")
async def list_repos():
    """RepoInfo fields plus vector_status (none|queued|running|ready|failed) and ingest_mode (agentic|vector)."""
    return await controller.list_repos_enriched()

class VectorizeRequest(BaseModel):
    reason: str = ""

@app.post("/api/v1/repos/{repo_id}/vectorize")
async def vectorize_repo(repo_id: str, req: Optional[VectorizeRequest] = None):
    """Enqueue (deduplicated) the background vector-index job for a repo. Returns the Job."""
    try:
        job = await controller.start_vectorize(repo_id, (req.reason if req else "") or "requested by user")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return job.to_dict()

@app.get("/api/v1/repos/{repo_id}/search-mode")
async def repo_search_mode(repo_id: str):
    try:
        return await controller.search_mode(repo_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/v1/jobs")
async def list_jobs(repo_id: Optional[str] = None, limit: int = 50):
    await controller.db.init_db()
    jobs = await controller.db.list_jobs(repo_id, max(1, min(int(limit), 500)))
    out = []
    for j in jobs:                                   # a running job's live progress is fresher than its throttled DB row
        live = controller.jobs.get_live(j["job_id"])
        out.append(live.to_dict() if live is not None else j)
    return {"jobs": out}

@app.get("/api/v1/jobs/{job_id}")
async def get_job(job_id: str):
    await controller.db.init_db()
    live = controller.jobs.get_live(job_id)
    job = live.to_dict() if live is not None else await controller.db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/api/v1/actions")
async def list_actions(request_id: Optional[str] = None, repo_id: Optional[str] = None, limit: int = 500):
    """Activity events (ActionEvent dicts, oldest first). In-memory ring buffer first; after a restart, the stored telemetry rows."""
    limit = max(1, min(int(limit), 5000))
    events = action_log.recent(request_id=request_id, repo_id=repo_id, limit=limit)
    if not events and request_id:
        await controller.db.init_db()
        rows = await controller.db.get_telemetry_by_request(request_id, limit)
        events = action_log.events_from_telemetry_rows(rows, repo_id)
    return {"events": events}

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
    """Processes a user query inside a specific repository thread and persists messages. Supports cancellation (499)."""
    runner, token_mode = _chat_choice(req.runner, req.token_mode)
    try:
        await controller.db.init_db()
        await controller.db.create_thread(req.thread_id, req.repo_id, req.thread_id)
        await controller.db.save_chat_message(req.thread_id, "user", req.query)
        request_id = uuid.uuid4().hex[:12]      # groups this turn's Activity events; returned so the UI can open "Show steps"
        try:
            response = await controller.chat_with_repo(req.repo_id, req.thread_id, req.query, web=req.web, model=req.model,
                                                       request_id=request_id, runner=runner, token_mode=token_mode)
        except asyncio.CancelledError:
            return JSONResponse(
                status_code=499,
                content={
                    "request_id": request_id,
                    "repo_id": req.repo_id,
                    "thread_id": req.thread_id,
                    "answer": "[Request stopped by user]",
                    "detail": "cancelled"
                }
            )
        await controller.db.save_chat_message(req.thread_id, "assistant", response)
        return {"request_id": request_id, "repo_id": req.repo_id, "thread_id": req.thread_id,
                "base_system": platform.system(), "answer": response}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/v1/chat/attach")
async def chat_with_attachments(
    repo_id: str = Form(...),
    thread_id: str = Form(...),
    query: str = Form(""),
    web: str = Form("auto"),
    model: str = Form(""),
    runner: str = Form(""),
    token_mode: str = Form(""),
    files: List[UploadFile] = File(...),
):
    """Chat turn with pasted/uploaded attachments (image, pdf, docx, text). Supports cancellation."""
    runner, token_mode = _chat_choice(runner, token_mode)
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
        prepared = await controller.prepare_attachments(raw)
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))

    query = query.strip()
    await controller.db.create_thread(thread_id, repo_id, thread_id)
    msg_id = await controller.db.save_chat_message(thread_id, "user", query)
    saved = await controller.store_attachments(thread_id, msg_id, prepared)

    request_id = uuid.uuid4().hex[:12]
    try:
        response = await controller.chat_with_repo(repo_id, thread_id, query, web=web, attachments=prepared, model=model,
                                                   request_id=request_id, runner=runner, token_mode=token_mode)
    except asyncio.CancelledError:
        return JSONResponse(
            status_code=499,
            content={
                "request_id": request_id,
                "repo_id": repo_id,
                "thread_id": thread_id,
                "answer": "[Request stopped by user]",
                "detail": "cancelled",
                "attachments": saved
            }
        )
    await controller.db.save_chat_message(thread_id, "assistant", response)
    return {"request_id": request_id, "repo_id": repo_id, "thread_id": thread_id, "answer": response, "attachments": saved}

@app.post("/api/v1/repos/{repo_id}/changes")
async def propose_change(repo_id: str, req: ChangeRequest):
    """Skips the ask/change router: always runs the change agents. Returns a proposal; writes nothing to disk."""
    runner, token_mode = _chat_choice(req.runner, req.token_mode)
    request_id = uuid.uuid4().hex[:12]
    try:
        with action_log.request_scope(request_id, None, repo_id=repo_id):
            res = await controller.propose_change(repo_id, req.request, req.base_rev, model=req.model,
                                                  runner=runner, token_mode=token_mode)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {**res.__dict__, "markdown": res.to_markdown(), "request_id": request_id}

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

@app.post("/api/v1/chat/stop")
async def stop_chat_endpoint(thread_id: str, repo_id: Optional[str] = None):
    """Terminate an ongoing chat session for a specific thread. Validates ownership if repo_id given."""
    try:
        await controller.db.init_db()
        if repo_id:
            try:
                import aiosqlite
                async with aiosqlite.connect(controller.db.db_path) as db:
                    async with db.execute("SELECT repo_id FROM chat_threads WHERE thread_id=?", (thread_id,)) as cur:
                        row = await cur.fetchone()
                        if row and row[0] != repo_id:
                            raise HTTPException(status_code=403, detail="Thread does not belong to this repo")
            except HTTPException:
                raise
            except Exception:
                pass
        success = await controller.stop_chat(thread_id)
        if not success:
            return JSONResponse(status_code=404, content={"detail": f"No active chat session found for thread_id: {thread_id}"})
        return JSONResponse(status_code=200, content={"detail": "Chat session terminated successfully", "thread_id": thread_id})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete("/api/v1/repos/{repo_id}")
async def api_remove_repo(repo_id: str):
    """
    Permanently removes a repository and its associated index.
    """
    try:
        await controller.remove_repo(repo_id)
        return {"status": "success", "message": f"Repository {repo_id} removed"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

