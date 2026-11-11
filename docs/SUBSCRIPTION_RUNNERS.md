# CodeAssist: Subscription runners (Claude Code CLI, Codex CLI)

Two runners let CodeAssist answer through the plan you already pay for, with no API key:

| Runner | Label in the UI | Runs |
|---|---|---|
| `claude_code` | Claude (subscription via Claude Code CLI) | the official `claude` CLI (Claude Pro / Max) |
| `chatgpt_codex` | ChatGPT (subscription via Codex CLI) | the official `codex` CLI (Sign in with ChatGPT) |

CodeAssist only starts the vendor's binary, once per model call, with the prompt on stdin and a temporary directory as working directory. You log in to the CLI yourself.

Items marked **[unverified]** could not be checked against a real login from the environment this was written in.

## Why CLIs (terms of use)

This is deliberate. CodeAssist **never reads, copies, stores or forwards** the CLIs' OAuth tokens or credential files (`~/.claude`, `~/.codex/auth.json`). Anthropic's terms forbid third-party apps from using a Claude subscription's OAuth directly; running your own Claude Code CLI is the supported path. Use this for personal, single-user work only: do not put it behind a shared service. Check the current terms of your plan before relying on it.

## 1. Install and log in

```
# Claude Code: install per Anthropic's instructions, then
claude            # run once, type /login, choose your Claude subscription
# Codex
npm i -g @openai/codex   # or the vendor's installer
codex login       # Sign in with ChatGPT
```

## 2. Configure `.env`

```
LLM_RUNNER=claude_code          # or chatgpt_codex
EMBEDDING_RUNNER=ollama         # keep: these runners have no embeddings
# optional
CLAUDE_CODE_MODEL=sonnet        # sonnet | opus | haiku | claude-...; empty = the CLI's default
CLAUDE_CODE_BIN=claude          # full path if not on PATH
CLAUDE_CODE_TIMEOUT_S=300
CLAUDE_CODE_MAX_INPUT_TOKENS=150000
CODEX_MODEL=                    # e.g. gpt-5.5; empty = the CLI's default (safest: always allowed for your plan)
CODEX_WEB_MODE=live              # native search mode: live (fresh pages) | cached (OpenAI index: faster, fewer tokens, can lag)
CODEX_MODELS=                   # optional comma list for the UI picker; see `codex` then /model for what your plan allows
CODEX_BIN=codex
CODEX_TIMEOUT_S=300
CODEX_MAX_INPUT_TOKENS=100000
```

Both runners can also be picked per request in the UI runner select. Without the binary on PATH a runner shows "(not set up)". The check only looks for the binary; it does not test the login.

## What the CLIs are asked to do

- Claude: `claude -p --output-format json --tools "" --no-session-persistence [--model M] [--system-prompt S]`, prompt on stdin. All tools are off: CodeAssist runs its own tool loop and parses text tool calls. `--max-turns` is not passed (not listed by `claude --help`; with no tools there is one turn).
- Codex: `codex exec --skip-git-repo-check --sandbox read-only --json --output-last-message <tmpfile> [-m M] -`, prompt on stdin; the reply is read from the file, token usage from the JSONL events. **[unverified: flags written from the documented `codex exec` options, codex was not installed]**. The argv is built in `build_codex_argv` (`services/runners/cli_core.py`); adjust it there if a release renames a flag.
- No vision (images are dropped with a note), no embeddings, no temperature / token-limit flags (JSON mode and the output cap are sent as instructions in the prompt; JSON replies are stripped of ``` fences).

## Web search

Both CLIs can search the web themselves. Who searches is the setting `WEB_PROVIDER` (`ENGINE_WEB_PROVIDER`):

- `app`: always CodeAssist's own SearXNG + scraper path (the behaviour before this option).
- `native`: the runner's own search when it has one (`claude_code`, `chatgpt_codex`), otherwise the app path.
- `auto` (default): native only when the chat is NOT repo-scoped and the runner can search. A repo-scoped chat keeps the app path, so code, file and client names are never written into search queries by the model. An unknown value is treated as `auto`.

The "does this message need the web" decision (`decide_web`, the Web chip, `/web`) is unchanged and runs on the local model, so it costs no quota. Only when it says yes and the native path is chosen, the CLI is started with its web tools on and the answer footer says "searched by <runner> (native search)". The citations come from the CLI itself (the prompt asks it to cite the URLs it used); there is no `[n]` source list.

- Claude: `--tools "WebSearch,WebFetch" --allowedTools "WebSearch,WebFetch"` instead of `--tools ""`. Only those two tools; never Bash, Edit or Read.
- Codex: Codex's web search is on by default (cached mode) even with `--sandbox read-only`, so the app always sets it: `-c web_search="disabled"` normally, `-c web_search="live"` for a native web turn.
- Cost: a web turn takes several model turns and live page fetches, so it is slower and uses more of the subscription window than a plain reply. The timeouts (`CLAUDE_CODE_TIMEOUT_S`, `CODEX_TIMEOUT_S`) were not changed; raise them if web answers time out.
- The runner status list reports `native_web` per runner (information for the UI only).

## Limits and quota

Subscription plans have usage windows (for example a 5-hour window and weekly caps), not per-token billing. When the CLI reports a usage or rate limit, CodeAssist raises the same quota error as the other runners (HTTP 402 with a readable message), so `LLM_FALLBACK_MODEL` and `LLM_QUOTA_COOLDOWN_S` apply. Calls are never retried (`max_retries` is 0) so a limit is not burnt faster. Remote-runner token saving (`TOKEN_MODE=auto` -> lean) is on. Each call starts a CLI process, so expect a few seconds of overhead.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "(not set up)" in the runner select | binary not on PATH: install it or set `CLAUDE_CODE_BIN` / `CODEX_BIN` to the full path |
| `The 'gpt-5' model is not supported when using Codex with a ChatGPT account` | OpenAI retires models for ChatGPT sign-in. The runner now retries once with the CLI's default model; pick **Default** in the UI or set `CODEX_MODEL` / `CODEX_MODELS` to models listed by `/model` in `codex` |
| Windows: `Cannot start 'codex'` / WinError 2 | npm installs `codex.cmd` / `claude.cmd`; the runner now runs these shims as `node <entry.js>`. If it still fails, run `where codex` and `where node` in the same terminal that starts CodeAssist (restart it after installing: PATH is read at start), or set `CODEX_BIN` to the full path |
| HTTP 401 "CLI is not logged in" | run `claude` and `/login`, or `codex login`, in a terminal as the same OS user that runs CodeAssist |
| HTTP 402 usage / rate limit | wait for the reset the CLI prints, pick another runner, or set `LLM_FALLBACK_MODEL` |
| HTTP 504 timeout | raise `CLAUDE_CODE_TIMEOUT_S` / `CODEX_TIMEOUT_S`, use a smaller model |
| "has no embeddings" / startup error | set `EMBEDDING_RUNNER=ollama` |
| Other CLI failure | the message ends with the last 500 characters of the CLI's output (never the prompt); run the same CLI by hand to see the full error |
