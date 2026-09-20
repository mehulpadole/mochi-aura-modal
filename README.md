# MoCHi Aura Modal deployment

Standalone Modal deployment for the Aura Gemma 4 31B Q5_K_M model used by MoCHi's Tomio Labs provider. The repository contains the model-serving application and deployment documentation only; model weights and credentials stay outside Git.

## What is implemented

- Downloads the pinned GGUF artifact into a persistent Modal Volume during preload.
- Verifies the expected file size and SHA-256 before serving it.
- Runs the official CUDA-enabled `llama.cpp` server image on an NVIDIA L40S with full GPU offload.
- Exposes llama.cpp's OpenAI-compatible `/v1/models` and `/v1/chat/completions` routes, including streaming chat completions.
- Enforces a server-side Bearer API key from a Modal Secret. The key is passed through a protected file rather than a command-line argument.

## Deployment configuration

| Setting | Value |
| --- | --- |
| Modal app | `mochi-aura` |
| Modal Volume | `aura-models` |
| Modal Secret | `aura-api` |
| Model source | `SevenOfNine/Aura-4o-Gemma-4-31B-GGUF` |
| Source revision | `5e8e4df42cb74f1241623feebb32e4ab13ce4d64` |
| Artifact | `Aura-Gemma-4-31B-Q5_K_M.gguf` |
| Artifact size | 21,845,565,056 bytes |
| Artifact SHA-256 | `3185d8cb9781fd80b724e0acecbbae007e813f518b17acfb2766096f3f708714` |
| Runtime | `llama.cpp` CUDA server |
| GPU | NVIDIA L40S |
| Context / maximum output | 131,072 / 8,192 tokens |
| Scaling | `min_containers=0`, `max_containers=1`, 15-minute scaledown window |

The CUDA server image is pinned by digest in `modal/app.py`. The model is stored in the named Volume and is never committed to Git.

## API

After deployment, use the HTTPS URL printed by Modal as `<modal-endpoint>`.

- `GET /health` reports llama.cpp readiness.
- `GET /v1/models` lists the served model.
- `POST /v1/chat/completions` accepts OpenAI-compatible chat completion requests.
- Set `"stream": true` for Server-Sent Events.
- Send `Authorization: Bearer <server-managed-key>` for authenticated model routes.

Example request shape:

```json
{
  "model": "aura-4o-gemma-4-31b-v1",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": true
}
```

## Request flow

```text
MoCHi client
  -> MoCHi server-side provider route
  -> authenticated Modal HTTPS endpoint
  -> llama.cpp CUDA server
  -> Aura GGUF in the Modal Volume
  -> OpenAI-compatible response or SSE stream
```

The API key belongs in Modal and MoCHi's server-side provider configuration. It must not be exposed through `NEXT_PUBLIC_*`, browser JavaScript, or client network requests.

## Run and deploy

Install and authenticate Modal, then create the named Secret through your local secret-management workflow. The variable expected by the server is `AURA_API_KEY`; do not commit its value.

```bash
python -m pip install --upgrade modal
modal setup
modal secret create aura-api AURA_API_KEY="<value supplied securely>"
modal run modal/app.py
modal deploy modal/app.py
```

`modal run` invokes the preload function and persists the verified artifact. `modal deploy` publishes the authenticated OpenAI-compatible server. Startup fails if the Volume does not contain the pinned GGUF or the Secret is missing.

## Performance notes

No reproducible Aura benchmark is stored in this repository. Cold-start time depends on Modal scheduling, image startup, and loading the 31B GGUF; warm TTFT and tokens per second depend on prompt length and sampling configuration. Record measurements against the deployed endpoint before publishing them.

## Security and release notes

- No API keys, Modal URLs, user data, or model weights are tracked.
- `.env.example` names the expected variable but contains no secret and is the only `.env*` file allowed by `.gitignore`.
- Keep private-beta authorization in MoCHi's server-side route rather than relying on the model picker.
- Review the upstream model and runtime licenses before making this repository or its service public.
