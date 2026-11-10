# CodeAssist: Cloudflare Workers AI Setup

This guide sets CodeAssist's LLM (and optionally embedding) runner to Cloudflare Workers AI.

Items marked **[unverified]** could not be confirmed from developers.cloudflare.com and need a test run.

## Prerequisites

- A Cloudflare account.
- Workers Paid plan recommended. The free tier includes 10,000 neurons per day. Beyond the allotment, Workers Paid costs $0.011 per 1,000 neurons.
- Text generation rate limit of 300 requests/min, per project notes **[unverified: the pricing page links to a separate Limits page that I could not fetch]**.
- Python environment with CodeAssist's dependencies installed.

## 1. Get your Account ID

1. Log in to the Cloudflare dashboard.
2. Open the **Workers AI** page.
3. Select **Use REST API**.
4. Copy the value labelled **Account ID** under **Get Account ID**.

Set it as `CF_ACCOUNT_ID`. It identifies the account and is not a secret, but keep it in `.env` rather than in source code.

## 2. Create an API token

Use the template (simplest):

1. On the Workers AI page (**Use REST API**), select **Create a Workers AI API Token**.
2. Review the prefilled details and select **Create API Token**.
3. Select **Copy API Token** and save it immediately.

Token shown on your profile is only a reminder; copy it at creation time.

Least-privilege alternative: create a custom token with exactly two permissions, **Workers AI - Read** and **Workers AI - Edit**. Scope the token to the single account if the form offers that option **[unverified: exact resource-scope label in the UI]**.

Store the token as `CF_API_TOKEN` in `.env`. Never commit it. Confirm `.env` is listed in `.gitignore`.

## 3. (Optional) AI Gateway

AI Gateway adds caching, logs and rate limits in front of Workers AI. Docs state a `default` gateway is created automatically. Create or check gateways under **AI Gateway** in the dashboard.

Recommended for CodeAssist (keeps plain `@cf/...` model ids): the Workers AI provider path on the gateway, which exposes the same OpenAI-compatible API:

`CF_AI_GATEWAY_URL=https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/workers-ai/v1`

Run the chat smoke test from section 4 against this base URL before switching the app over. Other forms you may see in the docs:

- **Unified compat endpoint** (SDK base URL):
  `https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/compat`
  Chat path: `/chat/completions`.
- **Account REST path with gateway header**:
  `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1` plus header `cf-aig-gateway-id: {gateway_id}`
  **[conflicting sources: not resolved]**

Other gateway details from the docs:

- Compat examples prefix Workers AI models with `workers-ai/`, for example `workers-ai/@cf/...`. The direct endpoint uses the plain `@cf/...` id. **[unverified: whether the gateway accepts the plain id]**
- If the gateway requires authentication, send `cf-aig-authorization: Bearer {CF_API_TOKEN}`.

Set `CF_AI_GATEWAY_URL` to the base URL without a trailing slash once confirmed.

## 4. Smoke test with curl

```bash
export CF_ACCOUNT_ID=your_account_id
export CF_API_TOKEN=your_token
```

Chat completion:

```bash
curl -sS "https://api.cloudflare.com/client/v4/accounts/$CF_ACCOUNT_ID/ai/v1/chat/completions" \
  -H "Authorization: Bearer $CF_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"@cf/google/gemma-4-26b-a4b-it","messages":[{"role":"user","content":"Say hello in five words."}]}'
```

Embeddings:

```bash
curl -sS "https://api.cloudflare.com/client/v4/accounts/$CF_ACCOUNT_ID/ai/v1/embeddings" \
  -H "Authorization: Bearer $CF_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"@cf/baai/bge-base-en-v1.5","input":"hello world"}'
```

Expected: a JSON response with a chat `choices` array, and an embedding vector of 768 values for bge-base-en-v1.5.

**[unverified]** The model id `@cf/google/gemma-4-26b-a4b-it` was taken from project notes. Confirm it in the Workers AI model catalog before relying on it.

## 5. Configure CodeAssist

Add to `.env`:

```bash
LLM_RUNNER=cloudflare
EMBEDDING_RUNNER=ollama
CF_ACCOUNT_ID=your_account_id
CF_API_TOKEN=your_token
CF_LLM_MODEL=@cf/google/gemma-4-26b-a4b-it
CF_VISION_MODEL=@cf/google/gemma-4-26b-a4b-it
CF_EMBEDDING_MODEL=@cf/baai/bge-base-en-v1.5
CF_MODELS=@cf/google/gemma-4-26b-a4b-it
# CF_AI_GATEWAY_URL=https://gateway.ai.cloudflare.com/v1/your_account_id/default/workers-ai/v1
```

Install the OpenAI-compatible client and start the app:

```bash
pip install langchain-openai
python run.py
```

Notes:

- The model picker lists every Workers AI text-generation model on your account (`GET /accounts/{id}/ai/models/search`), with `CF_MODELS` pinned at the top. The list is cached for `CF_MODELS_CACHE_S` seconds (default 600). Set `CF_LIST_ALL_MODELS=false` to show only `CF_MODELS`. If the call fails (e.g. token without Workers AI Read), the picker falls back to `CF_MODELS`. Proxied third-party models are left out.
- Ollama-style names such as the default `gemma4:31b-cloud` map automatically to `CF_LLM_MODEL` when `LLM_RUNNER=cloudflare`.
- Workers AI does not host Gemma 4 31B. The closest option is `gemma-4-26b-a4b-it` (26B MoE, about 4B active, 256K context, per project notes). Output quality may differ.
- The decision/router model (tev1 via `/v1/systemone`) still needs Ollama running locally.

## 6. Switching back and mixing runners

- **Back to local:** set `LLM_RUNNER=ollama` (the default) and restart `python run.py`.
- **Mixing:** you can run the LLM on Cloudflare with embeddings on Ollama, or the reverse.
- **Embedding switch:** changing `EMBEDDING_RUNNER` changes the vector dimensions. Run `python reset_repo_index.py` and re-index. Do this in both directions, or search will fail or return bad results.

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| 401 Unauthorized | Token missing, mistyped, or revoked | Check `CF_API_TOKEN` for stray quotes or spaces; create a new token |
| 403 Forbidden | Token lacks Workers AI Read/Edit, or wrong account | Recreate from the template, or add both permissions; verify `CF_ACCOUNT_ID` |
| 429 Too Many Requests | Daily neuron quota used up (free tier: 10,000/day) or per-minute rate limit hit | Wait, reduce calls, or move to Workers Paid; check usage in the dashboard |
| Model not found | Wrong or misspelled `@cf/` id | Copy the exact id from the Workers AI model catalog |
| Vector dimension mismatch | Embedding runner changed after the index was built | `python reset_repo_index.py`, then re-index |
| Gateway 401 | Gateway requires auth and `cf-aig-authorization` is missing | Add the header, or check the gateway's auth setting |

Error codes follow standard HTTP semantics. The exact error body text was not verified against the docs.
