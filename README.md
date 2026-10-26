
## Virtual Env to be used
- Use aCodeAssist


## Core Commands
```
python engine.py serve

python engine.py ingest --rep0 "C:\Users\IN010M80866\Apps\Code_Assistant"      # Windows
python engine.py ingest --rep0 ~/apps/Code_Assistant                                # macOS / Linux

python engine.py ingest list

python engine.py ingest chat <repo-id>
    python engine.py ingest chat 91cfd341a2ea

python engine.py serve --host 127.0.0.1 --port 8000

```

## macOS / Linux notes
- Create the venv with `python3 -m venv .venv && source .venv/bin/activate`, then `pip install -r requirements.txt`.
- Sandbox needs Docker (Docker Desktop on macOS; on Linux add your user to the `docker` group: `sudo usermod -aG docker $USER`).
- Linux file watching uses inotify; for big repos raise the limit: `sudo sysctl fs.inotify.max_user_watches=524288`.
- Run the app from a terminal. With no terminal attached (service, container) the startup watcher prompt is skipped and ALL repos are watched.
