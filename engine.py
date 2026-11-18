#!/usr/bin/env python3
"""
Phase 1 CLI & FastAPI Entrypoint.
Supports:
  > python engine.py ingest --repo <repo-root> [--vectorize]   (also supports --rep0 alias)
  > python engine.py ingest jobs [repo-id]
  > python engine.py ingest list
  > python engine.py ingest chat <repo-id> [--thread <thread-id>]
  > python engine.py serve --port 8000
"""
import argparse
import os
import sys
import uvicorn
from views.cli_view import CLIView
from config.settings import settings

def _tls_options():
    """uvicorn TLS kwargs, or {} for plain http. Never fails the launch: a bad/missing cert logs a warning and falls back to http.
    Order: SSL_CERTFILE + SSL_KEYFILE (env / .env) -> first *.pem pair in <repo>/certs/ (mkcert names: X.pem + X-key.pem)."""
    from pathlib import Path
    cert, key = (settings.ssl_certfile or "").strip(), (settings.ssl_keyfile or "").strip()
    if not (cert and key) and settings.ssl_auto:
        certs = Path(__file__).resolve().parent / "certs"
        for c in sorted(certs.glob("*.pem")) if certs.is_dir() else []:
            k = c.with_name(c.stem + "-key.pem")
            if not c.stem.endswith("-key") and k.is_file():
                cert, key = str(c), str(k)
                break
    if not (cert and key):
        return {}
    if not (os.path.isfile(cert) and os.path.isfile(key)):
        print(f"       WARNING: TLS cert/key not found ({cert}, {key}); serving plain http", flush=True)
        return {}
    try:
        import ssl
        ssl.create_default_context(ssl.Purpose.CLIENT_AUTH).load_cert_chain(cert, key)
    except Exception as exc:  # noqa: BLE001
        print(f"       WARNING: TLS cert unusable ({exc}); serving plain http", flush=True)
        return {}
    return {"ssl_certfile": cert, "ssl_keyfile": key}


def main():
    parser = argparse.ArgumentParser(description="Scalable Multi-Strategy Code Chunking & RAG Engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'ingest' command group
    ingest_parser = subparsers.add_parser("ingest", help="Ingest repositories, list repos, or chat with an ingested repo")
    ingest_parser.add_argument("action", nargs="?", default="run", help="Optional sub-action: 'list', 'chat' or 'jobs'")
    ingest_parser.add_argument("target_id", nargs="?", default=None, help="Repo ID when using 'ingest chat <repo-id>' or 'ingest jobs [repo-id]'")
    ingest_parser.add_argument("--repo", "--rep0", dest="repo_root", default=None, help="Root path of the repository to ingest")
    ingest_parser.add_argument("--vectorize", action="store_true", help="Also build the vector index now and wait for it (default: agentic search only, vectors on demand)")
    ingest_parser.add_argument("--trace", action="store_true", help="With 'ingest chat': print the step trace (intake, retrieval, tools, checks) after each answer")
    ingest_parser.add_argument("--thread", dest="thread_id", default="default_thread", help="Conversation thread ID for state persistence")

    # 'serve' command group for Web/FastAPI mode
    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI web server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    serve_parser.add_argument("--port", type=int, default=1951, help="Bind port")

    args = parser.parse_args()
    cli = CLIView()

    if args.command == "serve":
        print(" = " * 30)
        print(f"       Trying to connect to Ollama at {settings.ollama_base_url}", flush=True)
        print(" = " * 30)
        tls = _tls_options()
        print(f"       Serving on {'https' if tls else 'http'}://{settings.host}:{settings.port}"
              + (f" (cert {tls['ssl_certfile']})" if tls else ""), flush=True)
        uvicorn.run("views.api_view:app", host=settings.host, port=settings.port, reload=False, **tls)
        return

    if args.command == "ingest":
        if args.action == "list":
            cli.run_list()
        elif args.action == "chat":
            if not args.target_id:
                print("Error: Please provide <repo-id>. Usage: python engine.py ingest chat <repo-id>")
                sys.exit(1)
            cli.run_chat(args.target_id, session_id=args.thread_id, trace=args.trace)
        elif args.action == "jobs":
            cli.run_jobs(args.target_id)
        elif args.repo_root:
            cli.run_ingest(args.repo_root, vectorize=args.vectorize)
        else:
            ingest_parser.print_help()

if __name__ == "__main__":
    main()
