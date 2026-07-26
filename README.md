# Companion — Self-Hosting Guide

This document is **self-contained**: every file you need is inlined below.
If you get stuck, paste this entire README into an LLM along with your error message and your GPU/OS — it has enough context for the LLM to help you.
HERE YOU HAVE THE ACTUAL SERVERS WITH CUSTOM ENDPOINTS: EARS.py and MOUTH.py READ THEM.

## What you're running

Three services, all bound to localhost only:

| Service | What | Port | API |
|---|---|---|---|
| **brain** | llama.cpp server running your GGUF model | `8080` | `POST /completion`, `POST /v1/chat/completions` |
| **ears** | faster-whisper speech-to-text | `8000` | `POST /v1/audio/transcriptions` (OpenAI-compatible) |
| **mouth** | Kokoro text-to-speech | `8001` | `POST /v1/audio/speech` (OpenAI-compatible) |

- **Linux / Windows (NVIDIA GPU):** everything runs in Docker. Brain is built from `Dockerfile.brain`; ears and mouth use prebuilt images.
- **macOS (Apple Silicon):** Docker on macOS has **no GPU access**, so the brain runs natively (Metal GPU via Homebrew llama.cpp), while ears + mouth run as CPU containers (they're small models — fine on M-chips).

---

## Section 1 — Linux / Windows with NVIDIA GPU

### Prerequisites

1. **Docker** with the compose plugin (`docker compose version` must work).
2. **NVIDIA driver** new enough for CUDA 12.9 (any 2024+ driver is fine). Check: `nvidia-smi`.
3. **Linux only:** [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) so Docker can see the GPU. Test: `docker run --rm --gpus all nvcr.io/nvidia/cuda:12.9.0-runtime-ubuntu22.04 nvidia-smi`.
4. **Windows only:** Docker Desktop with the WSL2 backend. GPU passthrough works out of the box on WSL2; run the commands below from PowerShell.
5. A **GGUF model file** in `./GGUF` (see "Getting a brain model" below).

### Files

Create a folder, put these 3 files in it, plus a `GGUF/` subfolder with your model.

**`Dockerfile.brain`** — builds llama.cpp with CUDA for **all** NVIDIA generations from Turing (RTX 20xx) through Blackwell (RTX 50xx). The driver picks the right code at runtime, so this one image works on any modern NVIDIA card. **Build takes 45–90 min the first time** (compiling for 7 architectures); it's cached after that.

```dockerfile
# Stage 1: Build - Pulling from NVIDIA's NGC registry (nvcr.io)
FROM nvcr.io/nvidia/cuda:12.9.0-devel-ubuntu22.04 AS build

RUN apt-get update && apt-get install -y build-essential cmake git libcurl4-openssl-dev

# Missing .so.1 symlink for the CUDA driver stub
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}"

RUN git clone https://github.com/ggml-org/llama.cpp.git /app
WORKDIR /app

# Multi-arch fat binary: sm_75 (Turing) → sm_120 (Blackwell). Do NOT use "native".
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;100;120" \
    -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build --config Release -j$(nproc)

# Stage 2: Run
FROM nvcr.io/nvidia/cuda:12.9.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y libcurl4-openssl-dev libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /app/build/bin/ /app/
ENV LD_LIBRARY_PATH="/app:${LD_LIBRARY_PATH}"
ENTRYPOINT ["/app/llama-server"]
```

**`docker-compose.yml`**

```yaml
name: companion

services:
  brain:
    build:
      context: .
      dockerfile: Dockerfile.brain
    container_name: companion-brain
    restart: unless-stopped
    ports:
      - "127.0.0.1:8080:8080"
    volumes:
      - ./GGUF:/models
    ulimits:
      memlock:
        soft: -1
        hard: -1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all        # or device_ids: ['0'] to pin one GPU
              capabilities: [gpu, compute, utility]
    command: [
      "-m", "/models/${BRAIN_MODEL}",
      "--host", "0.0.0.0",
      "--port", "8080",
      "-c", "${BRAIN_CTX}",
      "-b", "8192",
      "-ngl", "99",
      "--flash-attn", "on",
      "--cache-type-k", "q8_0",
      "--cache-type-v", "q8_0",
      "--chat-template", "gemma"
    ]

  ears:
    image: fedirz/faster-whisper-server:latest-cuda
    container_name: companion-ears
    restart: unless-stopped
    ports:
      - "127.0.0.1:8000:8000"
    environment:
      - WHISPER__MODEL=${WHISPER_MODEL}
      - WHISPER__INFERENCE_DEVICE=cuda
    volumes:
      - ears-models:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  mouth:
    image: ghcr.io/remsky/kokoro-fastapi-gpu:latest
    container_name: companion-mouth
    restart: unless-stopped
    ports:
      - "127.0.0.1:8001:8880"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  ears-models:
```

**`.env`** (copy and edit)

```
BRAIN_MODEL=gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf
BRAIN_CTX=32768
WHISPER_MODEL=Systran/faster-whisper-small
KOKORO_VOICE=af_bella
HF_TOKEN=
```

### Run it

```bash
mkdir GGUF   # put your .gguf in here first
docker compose up -d --build     # first run: builds the brain image (45-90 min!)
docker compose logs -f brain     # watch until you see the server listening
```

Later runs are just `docker compose up -d` (build is cached).

### Getting a brain model

Download a GGUF into `./GGUF`, e.g. from `huggingface.co/unsloth` (use `huggingface-cli download` or the website). Rough VRAM guide for the gemma-style quants:

| Model | Size | VRAM reality check |
|---|---|---|
| 12B Q4_K_M | ~7.5 GB | comfy on 12 GB |
| 12B Q8_K_XL | ~13 GB | 16 GB card |
| 26B Q4_K_XL | ~16 GB | 24 GB card |
| 26B Q6_K_XL | ~21.5 GB | 24 GB, tight with big context |
| 26B Q8_K_XL | ~28 GB | 32 GB+ |

Context (`BRAIN_CTX`) eats VRAM too (KV cache) — the q8_0 cache flags above help a lot. On 12–16 GB cards keep `BRAIN_CTX` at 32768 or lower.

---

## Section 2 — macOS (Apple Silicon)

Docker on macOS **cannot use the GPU at all**. So: brain runs **natively** (Metal GPU), ears + mouth run as **CPU containers**.

### 1. Brain (native, Metal)

```bash
brew install llama.cpp
# download a GGUF anywhere, then:
llama-server -m ~/models/gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf \
  --port 8080 -c 32768 -ngl 99 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --chat-template gemma
```

`-ngl 99` offloads all layers to the Metal GPU — this is automatic, no CUDA anything. Unified memory means a 32 GB Mac can run the 26B Q6 fine.

(Prefer MLX instead? `pip install mlx-lm && mlx_lm.server --model mlx-community/gemma-4-12B-it-OptiQ-4bit --port 8080` also works, but its API differs slightly from llama-server — the rest of this guide assumes llama-server on 8080.)

### 2. Ears + mouth (CPU containers)

**`docker-compose.macos.yml`**

```yaml
name: companion

services:
  ears:
    image: fedirz/faster-whisper-server:latest-cpu
    container_name: companion-ears
    restart: unless-stopped
    ports:
      - "127.0.0.1:8000:8000"
    environment:
      - WHISPER__MODEL=${WHISPER_MODEL:-Systran/faster-whisper-small}
      - WHISPER__INFERENCE_DEVICE=cpu
    volumes:
      - ears-models:/root/.cache/huggingface

  mouth:
    image: ghcr.io/remsky/kokoro-fastapi-cpu:latest
    container_name: companion-mouth
    restart: unless-stopped
    ports:
      - "127.0.0.1:8001:8880"

volumes:
  ears-models:
```

```bash
docker compose -f docker-compose.macos.yml up -d
```

---

## Section 3 — Verify everything works

```bash
# brain: lists the loaded model
curl http://127.0.0.1:8080/v1/models

# brain: actual completion
curl http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "<bos><start_of_turn>user\nSay hi.<end_of_turn>\n<start_of_turn>model\n", "n_predict": 32}'

# ears: transcribe an audio file (wav, webm, mp3 — ffmpeg is inside the container)
curl -F "file=@/path/to/some-audio.wav" http://127.0.0.1:8000/v1/audio/transcriptions

# mouth: synthesize speech to an mp3
curl -X POST http://127.0.0.1:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro", "input": "Hello, I am alive.", "voice": "af_bella", "response_format": "mp3"}' \
  --output hello.mp3
```

First-ever request to ears/mouth downloads the model from HuggingFace — expect a pause, subsequent runs are instant (ears caches into a docker volume).

---

## Section 4 — Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `could not select device driver "nvidia" with capabilities: [[gpu]]` | Linux: install **nvidia-container-toolkit**, restart docker. Windows: use Docker Desktop WSL2 backend, not Hyper-V. |
| `no kernel image is available for execution on the device` | Image built for wrong arch. This Dockerfile builds sm_75–sm_120 — if you edited the list, put your card's arch back. RTX 50xx = sm_120, 40xx = sm_89, 30xx = sm_86, 20xx = sm_75. |
| Brain OOM / `ggml_cuda_host_malloc failed` | Model + context too big for VRAM. Lower `BRAIN_CTX`, or pick a smaller quant (see VRAM table). |
| `CUDA error` on RTX 5090 with some python/torch container | Blackwell (sm_120) needs **CUDA 12.8+ and torch ≥2.8 with cu128 wheels**. Older images crash on 50-series cards. The compose above avoids this by using llama.cpp (no torch) + CUDA-12.9 base. |
| Port already in use | Something else is on 8080/8000/8001. Change the left side of the port mapping, e.g. `"127.0.0.1:18000:8000"`. |
| Ears/mouth first request hangs | It's downloading the model from HuggingFace. `docker compose logs -f ears` to watch. |
| Linux: permission denied on `/var/run/docker.sock` | Add yourself to the docker group: `sudo usermod -aG docker $USER` then re-login. |
| Everything works but slow brain | Check `docker compose logs brain` for `offloaded XX/XX layers to GPU`. If it says 0, GPU passthrough is broken (see first row). |

### Notes for LLMs reading this README

- The brain container speaks the **llama.cpp `/completion` API** and OpenAI-compatible `/v1/chat/completions` on port 8080.
- Ears speaks **OpenAI `/v1/audio/transcriptions`** on 8000 (multipart field name: `file`; extra form fields: `model`, `language`, `response_format`).
- Mouth speaks **OpenAI `/v1/audio/speech`** on 8001 (JSON body: `model`, `input`, `voice`, optional `response_format`: mp3/wav/pcm/opus/flac, supports `stream: true`).
- All services bind `127.0.0.1` only. To expose on LAN, change the left side of the port mapping to `"0.0.0.0:PORT:CONTAINERPORT"` — and put auth in front of it, these APIs have none.
- The brain image takes 45–90 min to build because it compiles llama.cpp CUDA kernels for 7 GPU architectures; this is intentional so one image works on any NVIDIA card from RTX 20xx to 50xx.
<img width="1067" height="322" alt="Screenshot From 2026-07-26 18-11-34" src="https://github.com/user-attachments/assets/546f5cf4-e87d-4700-b2a5-a4b0fe318f3e" />
