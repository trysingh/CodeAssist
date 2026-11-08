import asyncio
import uuid
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from controllers.engine_controller import EngineController
from services import action_log

console = Console()

class CLIView:
    """CLI View layer for Phase 1 command interactions."""
    
    def __init__(self):
        self.controller = EngineController()

    def run_ingest(self, repo_root: str, vectorize: bool = False):
        console.print(Panel(f"[bold cyan]Starting agentic ingest[/bold cyan]\nTarget Repo: {repo_root}"
                            + ("\nVector index: building now (--vectorize); this can take a while" if vectorize else
                               "\nVector index: not built (searched by grep/glob/symbols); add --vectorize or run it later")))
        try:
            repo_info = asyncio.run(self.controller.ingest_repository(repo_root, vectorize=vectorize, watch=False))
            vs = asyncio.run(self.controller._vector_status(repo_info.repo_id))
            console.print(
                f"[bold green]✔ Ingestion Complete![/bold green]\n"
                f"  • [bold]Repo ID:[/bold] {repo_info.repo_id}\n"
                f"  • [bold]Title:[/bold] {repo_info.title}\n"
                f"  • [bold]Searchable files:[/bold] {repo_info.total_files}\n"
                f"  • [bold]Vector index:[/bold] {vs} ({repo_info.total_chunks} chunks)"
            )
        except Exception as exc:
            console.print(f"[bold red]✖ Ingestion Interrupted/Failed:[/bold red] {exc}")
            console.print("[yellow]Tip: Re-run the exact same command to resume from the last checkpointed file.[/yellow]")

    def run_jobs(self, repo_id: str = None):
        async def load():
            await self.controller.db.init_db()
            return await self.controller.db.list_jobs(repo_id, 50)
        jobs = asyncio.run(load())
        if not jobs:
            console.print("[yellow]No vectorize jobs recorded.[/yellow]")
            return
        table = Table(title="Vectorize jobs")
        for col in ("Job", "Repo", "State", "Progress", "Updated", "Reason / error"):
            table.add_column(col)
        for j in jobs:
            pct = f"{j['progress']}/{j['total']}" if j["total"] else "-"
            table.add_row(j["job_id"], j["repo_id"], j["state"], pct, str(j["updated_at"])[:19],
                          (j["error"] or j["reason"] or "")[:80])
        console.print(table)

    def run_list(self):
        repos = asyncio.run(self.controller.list_repos_enriched())
        if not repos:
            console.print("[yellow]No repositories have been ingested yet.[/yellow]")
            return

        table = Table(title="Ingested Repositories")
        table.add_column("Repo ID", style="bold cyan")
        table.add_column("Title", style="bold white")
        table.add_column("Status", style="green")
        table.add_column("Search", style="magenta")
        table.add_column("Files", justify="right")
        table.add_column("Chunks", justify="right")
        table.add_column("Root Path", style="dim")

        for r in repos:
            table.add_row(
                r["repo_id"],
                r["title"],
                r["status"],
                "agentic" if r["vector_status"] != "ready" else "hybrid (vectors ready)",
                f"{r['processed_files']}/{r['total_files']}",
                str(r["total_chunks"]),
                r["root_path"]
            )
        console.print(table)

    def run_chat(self, repo_id: str, session_id: str = "", trace: bool = False):
        thread_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        console.print(Panel(
            f"[bold green]Interactive Grounded Chat Session[/bold green]\n"
            f"Repo ID: [cyan]{repo_id}[/cyan] | Conversation Thread ID: [magenta]{thread_id}[/magenta]\n"
            f"Type [bold]'exit'[/bold] or [bold]'quit'[/bold] to end the conversation."
        ))
        
        while True:
            try:
                user_query = console.input("\n[bold blue]You > [/bold blue]").strip()
                if not user_query:
                    continue
                if user_query.lower() in ("exit", "quit"):
                    console.print(f"[dim]Conversation state persisted under thread_id='{thread_id}'. Goodbye![/dim]")
                    break
                
                request_id = uuid.uuid4().hex[:12]
                answer = asyncio.run(self.controller.chat_with_repo(repo_id, thread_id, user_query, request_id=request_id))
                # the answer already ends with the delivery notes (assumptions / limitations / what was verified)
                console.print(f"\n[bold emerald]Engine ({repo_id}) >[/bold emerald]\n{answer}")
                if trace:
                    console.print(f"\n[dim]Steps (request {request_id}):[/dim]")
                    console.print(action_log.format_trace(action_log.recent(request_id=request_id)), markup=False, highlight=False)
            except KeyboardInterrupt:
                console.print("\n[dim]Session exited.[/dim]")
                break
            except Exception as exc:
                console.print(f"[bold red]Error:[/bold red] {exc}")
