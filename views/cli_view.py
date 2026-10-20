import asyncio
import uuid
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from controllers.engine_controller import EngineController

console = Console()

class CLIView:
    """CLI View layer for Phase 1 command interactions."""
    
    def __init__(self):
        self.controller = EngineController()

    def run_ingest(self, repo_root: str):
        console.print(Panel(f"[bold cyan]Starting Asynchronous & Parallel Ingestion[/bold cyan]\nTarget Repo: {repo_root}"))
        try:
            repo_info = asyncio.run(self.controller.ingest_repository(repo_root))
            console.print(
                f"[bold green]✔ Ingestion Complete![/bold green]\n"
                f"  • [bold]Repo ID:[/bold] {repo_info.repo_id}\n"
                f"  • [bold]Title:[/bold] {repo_info.title}\n"
                f"  • [bold]Files Processed:[/bold] {repo_info.processed_files}/{repo_info.total_files}\n"
                f"  • [bold]Total Linked Chunks:[/bold] {repo_info.total_chunks}"
            )
        except Exception as exc:
            console.print(f"[bold red]✖ Ingestion Interrupted/Failed:[/bold red] {exc}")
            console.print("[yellow]Tip: Re-run the exact same command to resume from the last checkpointed file.[/yellow]")

    def run_list(self):
        repos = asyncio.run(self.controller.list_ingested_repos())
        if not repos:
            console.print("[yellow]No repositories have been ingested yet.[/yellow]")
            return

        table = Table(title="Ingested Repositories")
        table.add_column("Repo ID", style="bold cyan")
        table.add_column("Title", style="bold white")
        table.add_column("Status", style="green")
        table.add_column("Files", justify="right")
        table.add_column("Chunks", justify="right")
        table.add_column("Root Path", style="dim")

        for r in repos:
            table.add_row(
                r.repo_id,
                r.title,
                r.status.value,
                f"{r.processed_files}/{r.total_files}",
                str(r.total_chunks),
                r.root_path
            )
        console.print(table)

    def run_chat(self, repo_id: str, session_id: str = ""):
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
                
                answer = asyncio.run(self.controller.chat_with_repo(repo_id, thread_id, user_query))
                console.print(f"\n[bold emerald]Engine ({repo_id}) >[/bold emerald]\n{answer}")
            except KeyboardInterrupt:
                console.print("\n[dim]Session exited.[/dim]")
                break
            except Exception as exc:
                console.print(f"[bold red]Error:[/bold red] {exc}")
