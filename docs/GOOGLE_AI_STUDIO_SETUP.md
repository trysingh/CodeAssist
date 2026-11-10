# CodeAssist: Google AI Studio (Gemini API) Setup

This guide sets CodeAssist's chat (and optionally embedding) runner to Google AI Studio, through the Gemini API's OpenAI-compatible endpoint. The runner is called `aistudio` and shows as "Google AI Studio" in the UI.

Items marked **[unverified]** could not be confirmed against Google's documentation from the environment this was written in and need a test run.

## Prerequisites

- A Google account and an API key from AI Studio.
- `pip install langchain-openai` (imported lazily; Ollama-only installs do not need it).
- Free tier and billing: the Gemini API has a free tier, and paid usage needs billing enabled on the key's Google Cloud project. Limits (requests and tokens per minute, requests per day) vary per model and plan and are shown in AI Studio; check them there rather than relying on numbers in this guide **[unverified: current free-tier limits and whether Gemma 4 models are on the free tier]**.

## 1. Get an API key

1. Open https://aistudio.google.com/apikey and sign in.
2. Create an API key (pick or create a project when asked) and copy it.
3. Keep it in `.env` as `GEMINI_API_KEY`. Never commit it; confirm `.env` is in `.gitignore`. `GOOGLE_API_KEY` is accepted as an alias.

## 2. Smoke test with curl

```
curl -sS "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions" \
  -H "Authorization: Bearer $GEMINI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-31b-it", "messages": [{"role": "user", "content": "Say hi in five words."}]}'
```

A JSON reply with `choices[0].message.content` means the key and the model work. Model list: `curl -sS -H "Authorization: Bearer $GEMINI_API_KEY" https://generativelanguage.googleapis.com/v1beta/openai/models` (ids may come back as `models/gemma-4-31b-it`; CodeAssist strips the prefix). **[unverified: that `gemma-4-31b-it` is served by this endpoint; use an id from the model list if it answers 404]**

## 3. Configure CodeAssist

```
GEMINI_API_KEY=your-key
# make it the default (optional: you can also pick "Google AI Studio" in the UI chat settings, per request)
LLM_RUNNER=aistudio
# optional, shown with their defaults
AISTUDIO_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
AISTUDIO_LLM_MODEL=gemma-4-31b-it
AISTUDIO_VISION_MODEL=gemma-4-31b-it
AISTUDIO_EMBEDDING_MODEL=gemini-embedding-001
AISTUDIO_MODELS=gemma-4-31b-it,gemma-4-26b-a4b-it
AISTUDIO_LIST_ALL_MODELS=true
AISTUDIO_FOLD_SYSTEM=auto
AISTUDIO_JSON_MODE=auto
AISTUDIO_TIMEOUT_S=180
AISTUDIO_MAX_RETRIES=1
AISTUDIO_REASONING_EFFORT=
```

Restart the server. In the UI open the chat settings chip and choose Runner "Google AI Studio" (a cloud mark means remote / billed). Without `GEMINI_API_KEY` the entry shows "(not set up)" with the reason as tooltip, and a chat request that picks it is rejected with HTTP 400. With `LLM_RUNNER=aistudio` the app refuses to start without the key; with another default runner it starts normally.

What changes when the runner is remote: `TOKEN_MODE=auto` runs lean (fewer resent tokens, input cap `AISTUDIO_MAX_INPUT_TOKENS`=60000); telemetry rows and the token footer carry `aistudio`.

Model names and "Default": the app's global default `LLM_MODEL` is an Ollama tag (`gemma4:31b-cloud`), and a request carries it when the UI model picker is on "Default". On `aistudio` that name, or an empty one, means **`AISTUDIO_LLM_MODEL`** (image calls: `AISTUDIO_VISION_MODEL`), so with `AISTUDIO_LLM_MODEL=gemini-2.5-flash` the picker shows "Default (gemini-2.5-flash)", the call goes to Flash and telemetry says so. Only another explicit Ollama-style name (for example `LLM_FALLBACK_MODEL=gemma4:26b`) goes through the small `MODEL_MAP` table in `services/runners/aistudio_runner.py`: `gemma4:31b*` to `gemma-4-31b-it`, `gemma4:26b*` to `gemma-4-26b-a4b-it`, any other `name:tag` to `AISTUDIO_LLM_MODEL`. Add rows to `MODEL_MAP` to extend it. Ids without a colon (`gemini-2.5-flash`) are sent as they are. Role models (`PATCHER_MODEL`, `DIAGNOSER_MODEL`, ...) written as Ollama tags are ignored on this runner and use the default. Each request logs `model resolved: runner=aistudio picked=Default -> <id>` in the Activity timeline and the telemetry list.

**Speed: use a Flash model for interactive work.** Gemma 4 31B on AI Studio can be much slower than a Gemini Flash model (one reported call: 128 s for 5.3k tokens in / 165 out). Set `AISTUDIO_LLM_MODEL` to a Flash id from your model list (for example `gemini-2.5-flash`), or pick one in the model picker; use Lean token mode for long agent runs. Gemini thinking models may also think before answering: `AISTUDIO_REASONING_EFFORT=low` (or `none`, `minimal`) limits it; it is empty (not sent) by default, `auto` follows `ASK_REASONING` (`off` becomes `none`) **[unverified: which values each Gemini / Gemma model accepts on this endpoint; Gemma may answer 400]**.

The model picker lists `AISTUDIO_MODELS` first, then (with `AISTUDIO_LIST_ALL_MODELS=true`) the chat models from `GET {base}/models`: ids starting with `gemini` or `gemma`, minus embedding, imagen, veo, tts, audio, live, aqa and image-generation models. The catalogue is cached for `COMPAT_MODELS_CACHE_S` (600 s); on any error the picker shows only `AISTUDIO_MODELS`.

## System prompt folding

Older Gemma models on the Gemini API reject system instructions. With `AISTUDIO_FOLD_SYSTEM=auto` (default) the runner merges the system message into the first user message (system text, blank line, user text) for model ids starting with `gemma`, and leaves `gemini*` models alone. `on` forces folding for every model, `off` never folds. If a Gemma model answers 400 about system instructions, use `on`; if a newer Gemma accepts them and you prefer the native behaviour, use `off` **[unverified: which Gemma versions accept system instructions]**.

## JSON mode

For `gemini*` models CodeAssist sends `response_format={"type": "json_object"}` when a step asks for JSON. For `gemma*` models it does not (auto), and the existing tolerant JSON parser extracts the object from the reply. `AISTUDIO_JSON_MODE=on` sends it for every model, `off` never **[unverified: that Gemma accepts `json_object` on this endpoint]**.

## Embeddings

`EMBEDDING_RUNNER=aistudio` uses `AISTUDIO_EMBEDDING_MODEL` (`gemini-embedding-001`) **[unverified: embeddings through the compatibility endpoint]**. Vectors of different models are not comparable: after changing `EMBEDDING_RUNNER` or the model, re-index every repo (`python reset_repo_index.py`). Picking a runner in the UI never changes embeddings.

## Troubleshooting

- **HTTP 400 "API key not valid" / API_KEY_INVALID**: wrong or truncated key, stray quotes or spaces in `.env`. CodeAssist reports it as an authentication error naming `GEMINI_API_KEY`. A 400 about system instructions or `response_format` is a model limitation: see the two sections above.
- **HTTP 403 PERMISSION_DENIED**: the key's project may not have the Generative Language API enabled, or the model or region is not allowed for it. Create a new key at https://aistudio.google.com/apikey.
- **HTTP 429 RESOURCE_EXHAUSTED**: you hit a per-minute or per-day limit (or the free-tier quota). CodeAssist shows a quota message with links; wait and retry, pick another model, check the limits in AI Studio (https://ai.google.dev/gemini-api/docs/rate-limits), or enable billing. Set `ENGINE_LLM_FALLBACK_MODEL` to an AI Studio model id to switch automatically.
- **HTTP 504 "Google AI Studio did not answer within 180s"**: the call hung (timeout on every attempt). A hang costs `AISTUDIO_TIMEOUT_S x (AISTUDIO_MAX_RETRIES + 1)` seconds (default 180 x 2). Try a faster model (a Gemini Flash id instead of Gemma 4 31B), Lean token mode, or raise `AISTUDIO_TIMEOUT_S`; check that the "model resolved" line shows the model you expect. The message names the setting and the real model id. HTTP 503 means the endpoint could not be reached at all (network, proxy, `AISTUDIO_BASE_URL`).
- **The task timer turns amber or red**: the bar shows the current model call against `AISTUDIO_TIMEOUT_S`. Past the limit the call is probably retrying; a multi-step task may run longer than one call's limit in total.
- **HTTP 404 model not found**: the id is not served to your key. Open the model picker (or the `/models` request above) and use an id from that list; fix `AISTUDIO_LLM_MODEL` / `AISTUDIO_MODELS`.
- **"needs the langchain-openai package"**: `pip install langchain-openai`.
