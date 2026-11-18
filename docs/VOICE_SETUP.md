# CodeAssist: Voice input (dictation) and read-aloud

Click the mic next to the paperclip (or press **Ctrl/Cmd+Shift+M**), speak, pause. After **2.5 s of silence** the audio is transcribed, tidied into a
**clean English prompt** and **appended** to the prompt box for you to review and edit. Plain end-of-speech never sends anything.
Say a trigger phrase at the very end ("... **go ahead now**") and the prompt is submitted through the normal send path.

You can speak English, Hindi, Punjabi or a mix (Hinglish). Code identifiers, file names, paths, CLI commands and library names are kept verbatim;
fillers ("umm", "matlab", "haan") are removed; nothing is added and the request is not answered. Clean English comes back with only light punctuation fixes.

## How it works

```
browser mic -> VAD (2.5 s pause) -> 16 kHz wav -> POST /api/v1/voice/transcribe
     server: STT provider chain -> trigger phrase check (raw text) -> chat runner tidies to English -> trigger check (English) -> JSON
browser: appends text to #chat-input-query; submits only when a trigger phrase ended the utterance and text is left
```

* The speech-to-text (STT) provider is chosen **independently of the chat runner** (like the embedding model). `STT_PROVIDER=auto` follows the runner.
* No usable server provider (or all of them fail) = **browser mode**: the browser's Web Speech API listens, and the text is sent to
  `POST /api/v1/voice/refine` for translation and trigger detection. If that is unavailable too, the mic hides or says "Voice unavailable - please type your prompt".
* The chat-model tidy-up never blocks you: on any error or after `VOICE_REFINE_TIMEOUT_S` (30 s) the raw transcript is used as is.
* Esc cancels and discards. Clicking the mic again while listening stops and processes. If no speech starts within 8 s the recording is cancelled quietly.

## Which STT provider is used (`STT_PROVIDER=auto`)

| Chat runner | First STT provider | Needs |
|---|---|---|
| `aistudio` | Gemini (`AISTUDIO_STT_MODEL`, default `gemini-2.5-flash`): transcript **and** English in one call | `GEMINI_API_KEY` / `GOOGLE_API_KEY` |
| `cloudflare` | Workers AI Whisper (`CF_STT_MODEL`, default `@cf/openai/whisper-large-v3-turbo`) | `CF_ACCOUNT_ID`, `CF_API_TOKEN` |
| `openai` | `OPENAI_COMPAT_BASE_URL/audio/transcriptions` (`OPENAI_COMPAT_STT_MODEL`, default `whisper-large-v3-turbo`, e.g. Groq) | `OPENAI_COMPAT_BASE_URL` (+ `_API_KEY`) |
| `chatgpt_codex` | OpenAI (`OPENAI_STT_MODEL`, default `gpt-4o-transcribe`) | `OPENAI_API_KEY` (the Codex login cannot be used for this) |
| `ollama`, `claude_code`, unknown | none | see below |

After that first provider the chain continues with the self-hosted server (`STT_BASE_URL`, provider `local`) and then every other **configured** cloud
provider; a provider that fails is skipped and the next one is tried. Nothing configured = browser mode.
`STT_PROVIDER` can force a first choice: `aistudio` | `cloudflare` | `openai_compat` | `openai` | `local` | `browser` (= never use the server).
**Privacy:** with the default fallback an Ollama user who also has, say, a Gemini key configured would have audio go to Gemini. Set `VOICE_CLOUD_FALLBACK=false`
to send audio only to the provider you named (plus `STT_BASE_URL`).

## Settings

| Variable | Default | |
|---|---|---|
| `VOICE_ENABLED` | `true` | master switch |
| `VOICE_PAUSE_MS` | `2500` | silence that ends an utterance |
| `VOICE_MAX_SECONDS` / `VOICE_MAX_BYTES` | `120` / `10485760` | one recording; upload above the byte limit gets HTTP 413 |
| `VOICE_TRIGGER_PHRASES` | `go ahead now,process now,go ahead,submit now,send it now,ab bhejo,process karo,bhej do,अब भेजो,प्रोसेस करो` | comma list |
| `VOICE_TRANSLATE_TO_ENGLISH` | `true` | tidy / translate non-English or Hinglish transcripts |
| `VOICE_REFINE_TIMEOUT_S` | `30` | bound for the chat-model tidy-up |
| `VOICE_CLOUD_FALLBACK` | `true` | see Privacy above |
| `STT_PROVIDER` | `auto` | see above |
| `STT_BASE_URL`, `STT_API_KEY`, `STT_MODEL` | `""`, `""`, `Systran/faster-whisper-large-v3-turbo` | the `local` provider: any OpenAI-compatible `/v1/audio/transcriptions` server |
| `STT_TIMEOUT_S` | `60` | per provider call |
| `OPENAI_API_KEY`, `OPENAI_STT_BASE_URL`, `OPENAI_STT_MODEL` | `""`, `https://api.openai.com/v1`, `gpt-4o-transcribe` | provider `openai` |
| `OPENAI_COMPAT_STT_MODEL` | `whisper-large-v3-turbo` | provider `openai_compat` (reuses `OPENAI_COMPAT_BASE_URL` / `_API_KEY`) |
| `CF_STT_MODEL`, `AISTUDIO_STT_MODEL` | see table | |
| `TTS_ENABLED`, `TTS_PROVIDER`, `TTS_MODEL`, `TTS_BASE_URL`, `TTS_VOICE` | `false`, `ollama`, `phonon`, `""` (= `OLLAMA_BASE_URL`), `""` | read-aloud, see below |

Trigger matching ignores case, punctuation and spacing ("Go ahead, now." works), only counts at the **end** ("go ahead and refactor it" does not send),
tolerates a trailing "please" / "thanks" / "yaar", and removes the phrase from the prompt. If nothing is left after removing it, nothing is submitted.
Because "go ahead" is also ordinary speech, remove the short phrases from `VOICE_TRIGGER_PHRASES` if they fire too often.
Per browser (Chat settings): **Voice language** (used by browser-mode recognition; Auto = English India) and **Read replies aloud**.

## HTTPS: required for the mic from another device

Browsers only expose the microphone on a secure origin: `https://...` or `http://localhost` / `127.0.0.1`.
Opening CodeAssist as `http://192.168.x.x:1951` or `http://nuc:1951` hides the mic completely, and no site
permission can change that. The UI now says so instead of "Microphone blocked".

Pick one:

1. **Same machine:** open `http://localhost:1951`.
2. **Quick test (Chrome desktop and Android):** open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`,
   add `http://192.168.x.x:1951`, enable it and relaunch Chrome. This is per device and for testing only.
3. **Proper fix, self-signed TLS with mkcert (no launch-command changes):**
   ```bash
   mkcert -install                                   # local CA on the server machine
   mkdir certs && cd certs
   mkcert 192.168.x.x localhost 127.0.0.1            # -> 192.168.x.x+2.pem / 192.168.x.x+2-key.pem
   ```
   Launch exactly as before (`python run.py`). It picks up the first `X.pem` + `X-key.pem` pair in `certs/`
   and serves https on the usual port. To use certs from elsewhere, set `SSL_CERTFILE` / `SSL_KEYFILE` (env or `.env`).
   To force plain http, set `SSL_AUTO=false`. A missing or broken cert only logs a warning and serves http.
   `certs/` is git-ignored because it holds a private key.
   Other PCs: import mkcert's `rootCA.pem` (`mkcert -CAROOT` shows where it is).
   Android: Settings > Security > Encryption & credentials > Install a certificate > CA certificate > pick `rootCA.pem`.
4. **Tailscale (no certificates to manage):** `tailscale serve --bg 1951` gives
   `https://<machine>.<tailnet>.ts.net` with a real certificate on every device in your tailnet.

## Self-hosted STT on an Intel NUC (`speaches`)

[`speaches`](https://github.com/speaches-ai/speaches) is an OpenAI-compatible server around faster-whisper. CPU int8 with large-v3-turbo is workable on a NUC
(expect a few seconds for a short utterance); use a smaller model (`Systran/faster-whisper-small`) if it is too slow.

```bash
# on the NUC (check the speaches README for the current image tag)
docker run -d --name speaches --restart unless-stopped -p 8000:8000 \
  -v hf-hub-cache:/home/ubuntu/.cache/huggingface/hub \
  ghcr.io/speaches-ai/speaches:latest-cpu
# download the model once
curl -X POST http://localhost:8000/v1/models/Systran/faster-whisper-large-v3-turbo
```

Then in CodeAssist's environment:

```
STT_PROVIDER=local
STT_BASE_URL=http://<nuc-ip>:8000          # /v1 is added for you
STT_MODEL=Systran/faster-whisper-large-v3-turbo
# audio stays on your network:
VOICE_CLOUD_FALLBACK=false
```

Whisper can transcribe Hindi / Punjabi / Hinglish; the chat runner then turns the text into the English prompt. Check with
`curl -F file=@sample.wav -F model=Systran/faster-whisper-large-v3-turbo http://<nuc-ip>:8000/v1/audio/transcriptions`.

## Provider notes

* **Gemini (AI Studio):** one call returns the original transcript, a clean English prompt and whether a send command was spoken. Audio goes to Google. Free-tier
  quotas apply; a 429 moves on to the next provider.
* **Cloudflare:** Workers AI Whisper turbo through `/ai/run/{model}` with `CF_API_TOKEN`; the token needs Workers AI permission. Output is the spoken language
  (Hindi may come back in Devanagari); your chat runner translates it.
* **OpenAI:** `gpt-4o-transcribe` needs `OPENAI_API_KEY` (billed API, separate from a ChatGPT / Codex subscription).
* **Browser fallback:** Chrome and Edge (and Safari to a degree) offer the Web Speech API, which sends audio to the browser vendor's service. The microphone
  needs **HTTPS or `http://localhost`**; on plain `http://<lan-ip>` the browser hides `getUserMedia`, so only the fallback (or nothing) is available.
  Firefox has no Web Speech API: the mic is hidden unless a server provider works.

## Read replies aloud (optional)

Off by default. Tick **Read replies aloud** in Chat settings. With `TTS_ENABLED=true` and `TTS_PROVIDER=ollama` the server calls Ollama's
`/v1/audio/speech` with `TTS_MODEL` (`phonon`) and plays the wav; on any error (or when TTS is off) the browser's `speechSynthesis` reads the reply instead.
Code blocks are skipped. `ollama pull phonon` first. **Phonon is a text-to-speech model, not speech recognition**, so it plays no part in dictation. It
currently runs only on Apple Silicon and supports English plus a few European languages. Playback stops when you start recording or send a message.

## Troubleshooting

| Symptom | Check |
|---|---|
| No mic button | `GET /api/v1/voice/config` -> `enabled`; browser lacks `getUserMedia` (plain http) and Web Speech; `VOICE_ENABLED=false` |
| "Microphone blocked" | allow the mic for the site (lock icon); on Windows/macOS check OS microphone privacy settings |
| Toast "Server transcription failed" | the message in the server log (`voice stt failed: ...`): bad key, quota, wrong `STT_BASE_URL`, model not downloaded |
| Always browser mode | no provider is configured for the picked runner (`providers` is empty in the config response), see the table |
| Text is not English | the chat runner failed or timed out (logged as `voice refine failed`): the raw transcript is kept; raise `VOICE_REFINE_TIMEOUT_S` for slow CLI runners |
| Cuts off mid-sentence | raise `VOICE_PAUSE_MS`; very noisy rooms keep the recorder "speaking" and never end: use Esc or click the mic |
| HTTP 413 | recording larger than `VOICE_MAX_BYTES` |

API: `GET /api/v1/voice/config`, `POST /api/v1/voice/transcribe` (multipart `audio`, `runner`, `lang`), `POST /api/v1/voice/refine`, `POST /api/v1/voice/speak`.
Code: `services/speech/`, `views/voice_routes.py`, `static/js/voice.js`. Tests: `tests/test_voice.py`, `node tests/js/test_voice_trigger.mjs`.
