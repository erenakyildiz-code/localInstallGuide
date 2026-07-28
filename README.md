# Companion — Self-Hosting Guide

YOU NEED THE C# API: https://github.com/erenakyildiz-code/PublicBaratrumBackend.git

YOU ALSO NEED THIS Desktop App: https://github.com/erenakyildiz-code/PublicBaratrumDesktopApp.git

This file explains how you will run the MODELS; NOT THE C# API OR THE DESKTOP APP, THEY ARE WRITTEN IN THEIR OWN REPOSITORIES.

This document is **self-contained**: every file you need is inlined below.
If you get stuck, paste this entire README into an LLM along with your error message and your GPU/OS — it has enough context for the LLM to help you.

**The ears and mouth servers below (`ears.py`, `mouth.py`) are the ACTUAL servers with custom endpoints. They are NOT OpenAI-compatible. Read the endpoint contracts carefully — the client app expects exactly these routes.**

## What you're running

Three services:

| Service | What | Default port | API (custom — read this) |
|---|---|---|---|
| **brain** | llama.cpp server running your GGUF model | `8080` | `POST /completion` LLAMA.cpp TYPE ENDPOINT, you can use openai type if you go and change the c# api MANUALLY. IT IS RECOMMENDED YOU USE GOOGLE'S GEMMA 4 WITH LLAMA.CPP |
| **ears** | Qwen3-ASR-1.7B speech-to-text (`ears.py`) | `8000` | `POST /transcribe` — multipart form field `file` (webm/wav), returns parsed JSON |
| **mouth** | OmniVoice TTS, MLX / Apple Silicon (`mouth.py`) | `8080` | `WS /ws?session_id=...` + `POST /v1/audio/generate/{session_id}/{chunk_index}` + `POST /v1/audio/done/{session_id}` |

### Platform reality (important)

- **Brain**: NVIDIA (Docker, CUDA) or Apple Silicon (native, Metal). Any platform.
- **Ears**: pure PyTorch/transformers — runs anywhere Python + ffmpeg run. CPU works, GPU is faster.
- **Mouth**: **MLX — Apple Silicon Macs ONLY.** There is no Linux/Windows build of this server. If your brain/ears live on an NVIDIA box, run `mouth.py` on any Mac on the same LAN and point the client at it.
  - The author's own setup: brain on an RTX 5090 Linux box, mouth on an M-series Mac.
- **Port collision warning:** `mouth.py` binds **8080**, same as the brain's default. This is fine when mouth runs on a different machine (the intended setup). If you insist on running everything on one Mac, change the last line of `mouth.py` to another port (e.g. `8001`) and update the client accordingly.

---

## Section 1 — Brain (llama.cpp)

The brain speaks the standard llama.cpp server API, including OpenAI-compatible `/v1/chat/completions`. This part is platform-flexible.

### Linux / Windows with NVIDIA GPU — Docker

Prerequisites: Docker + compose plugin; NVIDIA driver (2024+); on Linux the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (test: `docker run --rm --gpus all nvcr.io/nvidia/cuda:12.9.0-runtime-ubuntu22.04 nvidia-smi`); on Windows, Docker Desktop with the WSL2 backend.

**`Dockerfile.brain`** — builds llama.cpp with CUDA for **all** NVIDIA generations from Turing (RTX 20xx) through Blackwell (RTX 50xx). **Build takes 45–90 min the first time** (compiling for 7 architectures); cached after that.

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

**`docker-compose.yml`** (brain only — ears/mouth are NOT docker images, see their sections)

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
```

**`.env`** (copy and edit)

```
BRAIN_MODEL=gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf
BRAIN_CTX=32768
HF_TOKEN=
```

Run:

```bash
mkdir GGUF   # put your .gguf in here first
docker compose up -d --build     # first run: builds the brain image (45-90 min!)
docker compose logs -f brain     # watch until you see the server listening
```

### macOS (Apple Silicon) — native, Metal

```bash
brew install llama.cpp
llama-server -m ~/models/gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf \
  --port 8080 -c 32768 -ngl 99 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --chat-template gemma
```

`-ngl 99` offloads all layers to the Metal GPU automatically. Unified memory means a 32 GB Mac runs the 26B Q6 fine.

### Getting a brain model

Download a GGUF into `./GGUF`, e.g. from `huggingface.co/unsloth`. Rough VRAM guide:

| Model | Size | VRAM reality check |
|---|---|---|
| 12B Q4_K_M | ~7.5 GB | comfy on 12 GB |
| 12B Q8_K_XL | ~13 GB | 16 GB card |
| 26B Q4_K_XL | ~16 GB | 24 GB card |
| 26B Q6_K_XL | ~21.5 GB | 32 GB, tight with big context |
| 26B Q8_K_XL | ~28 GB | 32 GB+ |

Context (`BRAIN_CTX`) eats VRAM too (KV cache) — the q8_0 cache flags help a lot. On 12–16 GB cards keep `BRAIN_CTX` at 32768 or lower.

---

## Section 2 — Ears (`ears.py`) — Qwen3-ASR

The actual ASR server. **One endpoint: `POST /transcribe`.** It accepts an uploaded audio file (the client sends `.webm`), converts it to 16 kHz mono WAV with **ffmpeg** (must be installed on the host), transcribes with Qwen3-ASR-1.7B, and returns the parsed result as JSON.

  YOU MAY WANT TO USE WHISPER, I HAVE A THICK TURKISH ACCENT SO I USED THIS HEAVY MODEL. IT EATS UP A BIT OF VRAM, WHISPER-TINY TYPE MODELS MIGHT BE GOOD ENOUGH FOR YOU. Tho if you are going to talk
  languages other than english I recommend qwen3-ASR-1.7B

**`ears.py`** (verbatim — this is the real server):

```python
import os
import torch
import subprocess
from fastapi import FastAPI, UploadFile, File
from transformers import AutoProcessor, AutoModelForMultimodalLM

app = FastAPI(title="Qwen3-ASR API")
MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"

print("Loading Qwen3-ASR-1.7B...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16
)
print("Model ready on CPU fallback.")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    temp_webm_path = f"/tmp/{file.filename}"
    temp_wav_path = f"/tmp/{file.filename}.wav"
    
    # 1. Save the incoming webm
    with open(temp_webm_path, "wb") as f:
        f.write(await file.read())
        
    try:
        # 2. Convert webm to wav using ffmpeg (force 16kHz, mono)
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_webm_path, 
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", 
            temp_wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 3. Pass the WAV file to the processor
        inputs = processor.apply_transcription_request(
            audio=temp_wav_path
        ).to(model.device, model.dtype)
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512)
            
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        parsed = processor.decode(generated_ids, return_format="parsed")[0]
        
        return parsed
    finally:
        # Cleanup both files
        if os.path.exists(temp_webm_path):
            os.remove(temp_webm_path)
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
```

### Install & run

```bash
# system dependency: ffmpeg
sudo apt install ffmpeg        # Debian/Ubuntu/Arch: pacman -S ffmpeg
# brew install ffmpeg          # macOS

pip install fastapi uvicorn transformers torch python-multipart
# python-multipart is REQUIRED — UploadFile breaks without it.

uvicorn ears:app --host 127.0.0.1 --port 8000
```

First run downloads ~3–4 GB of model weights from HuggingFace.

### Notes

- **GPU:** the file ships with `device_map="cpu"` because torch builds older than cu128 crash on Blackwell (RTX 50xx). On older NVIDIA cards or once you have torch ≥ 2.8/cu128, change it to `device_map="cuda"` (or `"auto"`) for much faster transcription. On Apple Silicon, `"mps"` works.
- **Endpoint contract:** `POST /transcribe`, multipart form field named exactly **`file`**. Any audio container ffmpeg can read works (webm, wav, mp3, ogg) — it's always converted to 16 kHz mono before inference. Response is the parsed transcription JSON (transcribed text plus detected language metadata).
- There is **no** `/v1/audio/transcriptions` route — if your client gets a 404 there, it's still pointing at the old OpenAI-style assumption.

---

## Section 3 — Mouth (`mouth.py`) — OmniVoice (MLX, Apple Silicon only)

The actual TTS server. **This is NOT a request/response API.** Audio never comes back in an HTTP response — it streams as **raw PCM bytes over a WebSocket**. The flow is:

1. **Open the WebSocket first:** `WS /ws?session_id=<your-session-id>` — the server registers the connection. If you call generate before connecting, you get HTTP 400.
2. **Request audio chunks:** `POST /v1/audio/generate/{session_id}/{chunk_index}` with JSON body `{"input": "<text to speak>", "lang_code": "<lang>"}`. The server runs OmniVoice and pushes each audio chunk to the WebSocket as a **binary frame of little-endian int16 PCM** (mono, at the model's sample rate — 24 kHz for OmniVoice; verify with `model.sample_rate` if unsure). Returns `{"status": "streamed"}` when the chunk is fully sent.
3. **Signal end of response:** `POST /v1/audio/done/{session_id}` — the server sends a **text frame `"EOS"`** over the WebSocket so the client knows playback for this turn is complete.

Other details that matter:

- `lang_code` is only read on the **first** generate call for a session; after that it's locked in `STREAM_LANG_MAP`.
- Voice cloning needs a **reference audio directory**: `VOICESDIR/<lang>/<lang>.wav` + `VOICESDIR/<lang>.txt` (the transcript of that wav). Edit `REFERENCE_AUDIO_PATH` in the file to point at yours. Example: `VOICESDIR/en/en.wav` + `VOICESDIR/en/en.txt`.
- Generation is serialized behind a GPU lock — concurrent requests queue, they don't parallelize. One Mac, one voice at a time.
- `chunk_index` in the path is currently bookkeeping only — ordering is the client's responsibility.
- you only need to care about the endpoints, if you can return what they return and accept what they accept, you can switch this out with any TTS model you want.
- 
**`mouth.py`** (verbatim — this is the real server; note it binds port **8080**, see the port-collision warning above):

```python
import uvicorn
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gc
import queue
import threading
import mlx.core as mx
from mlx_audio.tts.utils import load_model

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_ID = "mlx-community/OmniVoice-bf16"
# Paths as per your omni.py
REFERENCE_AUDIO_PATH = "/Users/devil/moss/VOICESDIR" #CHANGE THIS TO YOUR OWN LANG DIRECTORY (DIR SHOULD LOOK LIKE THIS langCode/langCode.wav langCode.txt)
STREAM_LANG_MAP = {}
app = FastAPI(title="Baratrum OmniVoice Streamer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

REF_TEXT_CACHE = {}
gpu_lock = asyncio.Lock()

model = None
active_connections: dict[str, WebSocket] = {}

class GenerateRequest(BaseModel):
    input: str
    lang_code: str

@app.on_event("startup")
async def load_tts_model():
    global model
    print(f"📦 Loading {MODEL_ID}...")
    model = load_model(MODEL_ID, trust_remote_code=False)
    print("✅ Model loaded.")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: str = Query(...)):
    await websocket.accept()
    active_connections[session_id] = websocket
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.pop(session_id, None)


def sync_generator_worker(text, audioPath, content, out_queue):
    """Runs the heavy MLX generation in a background thread."""
    try:
        for result in model.generate(text=text, ref_audio=audioPath, ref_text=content, normalize_text=False):
            # Convert float32 to int16 PCM
            audio_array = np.array(result.audio, dtype=np.float32)
            pcm16_array = (audio_array * 32767).astype(np.int16)
            
            # Put the bytes into the thread-safe queue
            out_queue.put(pcm16_array.tobytes())

            del audio_array
            del pcm16_array
            del result
            
        out_queue.put(None) # Send EOF signal
    except Exception as e:
        out_queue.put(e)
    finally:
        # Cleanup runs safely inside the locked thread
        gc.collect()
        mx.metal.clear_cache()

async def stream_chunk_to_client(session_id: str, text: str):
    if session_id not in active_connections: return
    websocket = active_connections[session_id]
    
    lang = STREAM_LANG_MAP[session_id]
    audioPath = f"{REFERENCE_AUDIO_PATH}/{lang}/{lang}.wav"
    transcriptPath = f"{REFERENCE_AUDIO_PATH}/{lang}/{lang}.txt"
    
    if lang not in REF_TEXT_CACHE:
        with open(transcriptPath, 'r', encoding='utf-8') as f:
            REF_TEXT_CACHE[lang] = f.read()
            
    content = REF_TEXT_CACHE[lang]

    # 2. Acquire the lock so no other request can hit the GPU concurrently
    async with gpu_lock:
        loop = asyncio.get_running_loop()
        out_queue = queue.Queue()
        
        # 3. Spin up the background worker
        thread = threading.Thread(
            target=sync_generator_worker, 
            args=(text, audioPath, content, out_queue)
        )
        thread.start()
        
        try:
            while True:
                # 4. Await queue items via executor to yield control back to FastAPI
                chunk = await loop.run_in_executor(None, out_queue.get)
                
                if chunk is None: 
                    break # EOF reached
                if isinstance(chunk, Exception):
                    print(f"❌ Generation error: {chunk}")
                    break
                    
                await websocket.send_bytes(chunk)
                
        except Exception as e:
            print(f"⚠️ Websocket error during streaming: {e}")

@app.post("/v1/audio/generate/{session_id}/{chunk_index}")
async def generate_chunk(session_id: str, chunk_index: int, req: GenerateRequest):
    # 1. Fail loudly if the websocket isn't ready
    if session_id not in active_connections:
        raise HTTPException(status_code=400, detail="WebSocket not connected for this session. Wait for connection before generating.")

    lang = STREAM_LANG_MAP.get(session_id)
    if lang is None:
        STREAM_LANG_MAP[session_id] = req.lang_code
        
    print(f"Session: {session_id}, language: {lang}")
    print(req.input)
    await stream_chunk_to_client(session_id, req.input)
    return {"status": "streamed"}

@app.post("/v1/audio/done/{session_id}")
async def mark_session_done(session_id: str):
    if session_id in active_connections:
        await active_connections[session_id].send_text("EOS")
    return {"status": "eos_sent"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080)
```

### Install & run (macOS, Apple Silicon)

```bash
pip install mlx mlx-audio fastapi "uvicorn[standard]" numpy

# 1. Edit REFERENCE_AUDIO_PATH in mouth.py to your voices dir:
#    VOICESDIR/en/en.wav + VOICESDIR/en/en.txt  (and tr/tr.wav + tr/tr.txt, etc.)
# 2. Run:
python mouth.py
```

First run downloads the OmniVoice weights (~a few GB) from HuggingFace.

---

## Section 4 — Verify everything works

```bash
# brain: lists the loaded model
curl http://127.0.0.1:8080/v1/models

# brain: actual completion
curl http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "<bos><start_of_turn>user\nSay hi.<end_of_turn>\n<start_of_turn>model\n", "n_predict": 32}'

# ears: transcribe an audio file (the field name MUST be "file")
curl -F "file=@/path/to/some-audio.webm" http://127.0.0.1:8000/transcribe

# mouth: full handshake test (needs websocat: brew install websocat / pacman -S websocat)
# terminal 1 — connect the websocket, watch binary frames + EOS arrive:
websocat "ws://127.0.0.1:8080/ws?session_id=test1"

# terminal 2 — request speech, then signal done:
curl -X POST http://127.0.0.1:8080/v1/audio/generate/test1/0 \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, I am alive.", "lang_code": "en"}'
curl -X POST http://127.0.0.1:8080/v1/audio/done/test1
```

If the mouth test 400s, you called generate before the websocket was registered — that's the #1 mistake.

---

## Section 5 — Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `could not select device driver "nvidia" with capabilities: [[gpu]]` | Linux: install **nvidia-container-toolkit**, restart docker. Windows: Docker Desktop WSL2 backend, not Hyper-V. |
| `no kernel image is available for execution on the device` | Brain image built for wrong arch. RTX 50xx = sm_120, 40xx = sm_89, 30xx = sm_86, 20xx = sm_75. |
| Brain OOM / `ggml_cuda_host_malloc failed` | Model + context too big for VRAM. Lower `BRAIN_CTX`, or pick a smaller quant (see VRAM table). |
| Ears: `python-multipart` / `Form data requires "python-multipart"` error | You skipped a dependency: `pip install python-multipart`. |
| Ears: ffmpeg errors / silent failure on transcribe | ffmpeg isn't installed on the host, or `/tmp` isn't writable. Install ffmpeg; the server shells out to it for webm→wav conversion. |
| Ears slow | It's on CPU (`device_map="cpu"` in the file). Switch to `"cuda"`/`"mps"` if your torch build supports your GPU. RTX 50xx needs torch ≥ 2.8 with cu128 wheels. |
| Mouth: `pip install mlx` fails | You're not on an Apple Silicon Mac. MLX does not exist on Linux/Windows — run mouth on a Mac on your LAN instead. |
| Mouth: HTTP 400 "WebSocket not connected" | You called `/v1/audio/generate/...` before opening `WS /ws?session_id=...`. The websocket must connect FIRST, with the same session_id. |
| Mouth: FileNotFoundError on `<lang>.wav` / `<lang>.txt` | `REFERENCE_AUDIO_PATH` doesn't match the required layout: `VOICESDIR/<lang>/<lang>.wav` + `VOICESDIR/<lang>/<lang>.txt`. |
| Mouth: client hears garbage/static | The stream is **raw int16 PCM, not a wav/mp3 file** — no header. Feed it to your audio pipeline as PCM16 mono at the model's sample rate (24 kHz for OmniVoice). |
| Mouth and brain both on 8080 | Intended only when they're on different machines. Same machine → change the port in the last line of `mouth.py`. |
| Port already in use | Something else is bound. For the python servers change the uvicorn port; for the brain change the left side of the port mapping, e.g. `"127.0.0.1:18080:8080"`. |
| Linux: permission denied on `/var/run/docker.sock` | `sudo usermod -aG docker $USER` then re-login. |
| Everything works but slow brain | Check `docker compose logs brain` for `offloaded XX/XX layers to GPU`. If it says 0, GPU passthrough is broken. |

### Notes for LLMs reading this README (endpoint contracts — do not "fix" these)

- **Brain** (llama.cpp, port 8080): standard llama.cpp server. `POST /completion` (JSON `prompt`, `n_predict`) and OpenAI-compatible `POST /v1/chat/completions`, `GET /v1/models`.
- **Ears** (Qwen3-ASR, port 8000): exactly one route — `POST /transcribe`. Multipart form field name: **`file`**. Accepts any ffmpeg-readable audio; converts to 16 kHz mono WAV internally; returns parsed transcription JSON. There is NO `/v1/audio/transcriptions`, NO `model`/`language` form fields, and it is NOT OpenAI-compatible.
- **Mouth** (OmniVoice MLX, port 8080, Apple Silicon only): three cooperating routes —
  1. `WS /ws?session_id=<id>` — must be connected before generating; audio arrives here as binary frames.
  2. `POST /v1/audio/generate/{session_id}/{chunk_index}` — JSON body `{"input": str, "lang_code": str}` (field is `input`, NOT `text`); returns `{"status": "streamed"}`; audio is pushed over the websocket as raw little-endian **int16 PCM** chunks.
  3. `POST /v1/audio/done/{session_id}` — server sends text frame `"EOS"` over the websocket.
  There is NO `/v1/audio/speech` route and audio is NEVER in an HTTP response body.
- Mouth voice cloning requires `VOICESDIR/<lang>/<lang>.wav` + `.txt`; `lang_code` is locked per session on first generate call.
- All services are designed for localhost / trusted LAN. None of these APIs have auth — do not expose them to the internet without putting auth in front.
- The brain image takes 45–90 min to build because it compiles llama.cpp CUDA kernels for 7 GPU architectures; this is intentional so one image works on any NVIDIA card from RTX 20xx to 50xx.
- THE STUFF YOU SEE HERE ARE INTERCHANGABLE, THE ONLY IMPORTANT THINGS ARE THE ENDPOINTS. THE OTHER PROGRAMS THE USER IS GOING TO RUN ARE GOING TO CONNECT TO THE ENDPOINTS, THINK OF THESE AS BLACK BOXES THAT WILL ACCEPT DATA AND RETURN DATA DON'T THINK ABOUT THEM LIKE THE MIDDLE PART IS IMPORTANT. SO YOU CAN RUN HIGGS AUDIO, KOKORO, WHATEVER THE FUCK YOU WANT TO RUN, AS LONG AS THE ENDPOINTS ARE CORRECTLY MADE EVERYTHING WILL WORK PROPERLY. JUST DON'T FUCK WITH THE ENDPOINTS. DO NOT CHANGE ENDPOINTS AND THEIR INPUT OUTPUT BEHAVIOUR.
