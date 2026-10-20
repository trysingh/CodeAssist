#!/usr/bin/env python3
"""
Phase 1 CLI & FastAPI Entrypoint.
Supports:
  > python engine.py ingest --repo <repo-root>   (also supports --rep0 alias)
  > python engine.py ingest list
  > python engine.py ingest chat <repo-id> [--thread <thread-id>]
  > python engine.py serve --port 8000
"""
import argparse
import sys
import uvicorn
from views.cli_view import CLIView
from config.settings import settings

def main():
    parser = argparse.ArgumentParser(description="Scalable Multi-Strategy Code Chunking & RAG Engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'ingest' command group
    ingest_parser = subparsers.add_parser("ingest", help="Ingest repositories, list repos, or chat with an ingested repo")
    ingest_parser.add_argument("action", nargs="?", default="run", help="Optional sub-action: 'list' or 'chat'")
    ingest_parser.add_argument("target_id", nargs="?", default=None, help="Repo ID when using 'ingest chat <repo-id>'")
    ingest_parser.add_argument("--repo", "--rep0", dest="repo_root", default=None, help="Root path of the repository to ingest")
    ingest_parser.add_argument("--thread", dest="thread_id", default="default_thread", help="Conversation thread ID for state persistence")

    # 'serve' command group for Web/FastAPI mode
    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI web server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port")

    args = parser.parse_args()
    cli = CLIView()

    if args.command == "serve":
        uvicorn.run("views.api_view:app", host=settings.host, port=settings.port, reload=False)
        return

    if args.command == "ingest":
        if args.action == "list":
            cli.run_list()
        elif args.action == "chat":
            if not args.target_id:
                print("Error: Please provide <repo-id>. Usage: python engine.py ingest chat <repo-id>")
                sys.exit(1)
            cli.run_chat(args.target_id, session_id=args.thread_id)
        elif args.repo_root:
            cli.run_ingest(args.repo_root)
        else:
            ingest_parser.print_help()

if __name__ == "__main__":
    main()
