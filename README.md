# Aura on Modal

Standalone Modal deployment source for MoCHi's server-managed Aura model. The
deployment uses the pinned GGUF with the official llama.cpp CUDA image and one
NVIDIA L40S.

- Modal app: `mochi-aura`
- Volume: `aura-models`
- Modal Secret: `aura-api`, containing `AURA_API_KEY`
- Context: 128K; maximum output: 8K
- Scaling: `min_containers=0`, `max_containers=1`, 15-minute scale-down

Model weights are downloaded and checksum-verified into the Modal Volume by the
preload step. They are not stored in Git. Credentials and MoCHi/Cloudflare
secrets are also never stored here.

```bash
modal run modal/app.py
modal deploy modal/app.py
```

Configure the resulting endpoint and the matching server-side key in MoCHi's
Cloudflare Worker. Never expose the key in browser code and do not add a
recurring health poller.
