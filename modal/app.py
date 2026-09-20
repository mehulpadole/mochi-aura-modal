"""Aura V1 llama.cpp deployment for MoCHi.

Aura is an additional Tomio Labs server-managed model. The model is stored in
its own named Modal Volume and the serving container uses the official
ggml-org llama.cpp CUDA image. The endpoint is public at the Modal proxy layer
so MoCHi can reach it, while llama.cpp enforces the server-managed Bearer key.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.error
import urllib.request

import modal


APP_NAME = "mochi-aura"
VOLUME_NAME = "aura-models"
SECRET_NAME = "aura-api"
MODEL_MOUNT = "/models"
MODEL_FILENAME = "Aura-Gemma-4-31B-Q5_K_M.gguf"
MODEL_PATH = f"{MODEL_MOUNT}/{MODEL_FILENAME}"
MODEL_REPOSITORY = "SevenOfNine/Aura-4o-Gemma-4-31B-GGUF"
MODEL_REVISION = "5e8e4df42cb74f1241623feebb32e4ab13ce4d64"
MODEL_SIZE_BYTES = 21_845_565_056
MODEL_SHA256 = "3185d8cb9781fd80b724e0acecbbae007e813f518b17acfb2766096f3f708714"
MODEL_URL = (
    f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/"
    f"{MODEL_REVISION}/{MODEL_FILENAME}?download=true"
)

# Keep the serving image pinned so a future upstream tag change cannot silently
# alter the deployed runtime.
LLAMA_CPP_IMAGE = (
    "ghcr.io/ggml-org/llama.cpp:server-cuda@"
    "sha256:5ebb7a55e3d38e5fe185346fcd908033e69c035ad49b33df0066eb7cf4f7b06b"
)

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

download_image = modal.Image.debian_slim(python_version="3.12")
serve_image = modal.Image.from_registry(
    LLAMA_CPP_IMAGE,
    add_python="3.12",
    setup_dockerfile_commands=["ENTRYPOINT []"],
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_model_exists(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size != MODEL_SIZE_BYTES:
        return False
    return _sha256(path) == MODEL_SHA256


def _download_model() -> dict[str, object]:
    target = Path(MODEL_PATH)
    partial = target.with_suffix(target.suffix + ".part")
    metadata = target.with_suffix(target.suffix + ".json")

    if _verified_model_exists(target):
        return {
            "status": "already_present",
            "path": str(target),
            "size_bytes": target.stat().st_size,
            "sha256": MODEL_SHA256,
            "revision": MODEL_REVISION,
        }

    target.parent.mkdir(parents=True, exist_ok=True)
    resume_from = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "mochi-aura-modal-preload/1"}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"

    request = urllib.request.Request(MODEL_URL, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"Hugging Face Aura model download failed with HTTP {error.code}."
        ) from error

    append = resume_from > 0 and response.status == 206
    if not append:
        resume_from = 0

    mode = "ab" if append else "wb"
    with response, partial.open(mode) as output:
        while True:
            block = response.read(16 * 1024 * 1024)
            if not block:
                break
            output.write(block)

    if partial.stat().st_size != MODEL_SIZE_BYTES:
        raise RuntimeError(
            f"Downloaded Aura model size mismatch: expected {MODEL_SIZE_BYTES}, "
            f"got {partial.stat().st_size}."
        )

    checksum = _sha256(partial)
    if checksum != MODEL_SHA256:
        raise RuntimeError(
            "Downloaded Aura model checksum mismatch; refusing to publish it to the Volume."
        )

    os.replace(partial, target)
    metadata.write_text(
        json.dumps(
            {
                "repository": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
                "filename": MODEL_FILENAME,
                "size_bytes": MODEL_SIZE_BYTES,
                "sha256": MODEL_SHA256,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": "downloaded",
        "path": str(target),
        "size_bytes": MODEL_SIZE_BYTES,
        "sha256": checksum,
        "revision": MODEL_REVISION,
    }


@app.function(
    image=download_image,
    volumes={MODEL_MOUNT: model_volume},
    timeout=3 * 60 * 60,
    retries=0,
)
def preload_model() -> dict[str, object]:
    """Download and verify Aura V1 once, then persist it in the Volume."""

    result = _download_model()
    model_volume.commit()
    print(
        "Aura V1 model ready:",
        result["status"],
        result["size_bytes"],
        result["sha256"],
        result["revision"],
    )
    return result


def _verify_l40s_and_cuda() -> None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            "NVIDIA CUDA/GPU verification failed; refusing CPU fallback."
        ) from error

    gpu_info = result.stdout.strip()
    if "L40S" not in gpu_info:
        raise RuntimeError(
            f"Expected an NVIDIA L40S GPU, got: {gpu_info or 'no GPU reported'}"
        )
    print(f"Aura GPU verified: {gpu_info}")


def _print_gpu_memory() -> None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    print(f"Aura GPU memory after model load: {result.stdout.strip()}")


def _wait_for_model(
    server_process: subprocess.Popen[bytes], timeout_seconds: int = 1800
) -> None:
    deadline = time.monotonic() + timeout_seconds
    health_url = "http://127.0.0.1:8080/health"

    while time.monotonic() < deadline:
        if server_process.poll() is not None:
            raise RuntimeError(
                "llama-server exited during Aura startup with code "
                f"{server_process.returncode}."
            )

        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    print("Aura llama.cpp health check passed; model is ready.")
                    return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            pass

        time.sleep(2)

    server_process.terminate()
    raise TimeoutError("Timed out waiting for llama.cpp to load the Aura model.")


@app.function(
    image=serve_image,
    gpu="L40S",
    volumes={MODEL_MOUNT: model_volume},
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    min_containers=0,
    max_containers=1,
    scaledown_window=15 * 60,
    timeout=60 * 60,
)
@modal.web_server(8080, startup_timeout=30 * 60)
def llama_server() -> None:
    """Run the authenticated Aura llama.cpp OpenAI-compatible HTTP server."""

    api_key = os.environ.get("AURA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("AURA_API_KEY is missing from the Modal Secret.")

    model_path = Path(MODEL_PATH)
    if not _verified_model_exists(model_path):
        raise RuntimeError(
            "The verified Aura GGUF is missing from the Volume. Run the preload command first."
        )

    _verify_l40s_and_cuda()

    api_key_file = Path("/tmp/aura-api-key")
    api_key_file.write_text(api_key + "\n", encoding="utf-8")
    api_key_file.chmod(0o600)

    llama_binary = shutil.which("llama-server") or "/app/llama-server"
    command = [
        llama_binary,
        "--model",
        MODEL_PATH,
        "--alias",
        "aura-4o-gemma-4-31b-v1",
        "--ctx-size",
        "131072",
        "--n-predict",
        "8192",
        "--n-gpu-layers",
        "999",
        "--device",
        "CUDA0",
        "--parallel",
        "1",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        # Aura V1's model card recommends llama.cpp's native Jinja template.
        "--jinja",
        "--reasoning",
        "off",
        "--reasoning-format",
        "none",
        "--temp",
        "0.75",
        "--top-p",
        "0.8",
        "--top-k",
        "50",
        "--min-p",
        "0.05",
        "--repeat-penalty",
        "1.05",
        "--frequency-penalty",
        "0",
        "--presence-penalty",
        "0",
        "--api-key-file",
        str(api_key_file),
        "--no-webui",
        "--verbose",
    ]
    server_process = subprocess.Popen(command)
    _wait_for_model(server_process)
    _print_gpu_memory()


@app.local_entrypoint()
def main() -> None:
    """`modal run modal/aura/app.py` preloads the persistent Aura Volume."""

    print(preload_model.remote())
