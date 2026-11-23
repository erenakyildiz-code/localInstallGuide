# Companion — Self-Hosting Guide

it is recommended you let an llm guide you through the installation process.


YOU NEED THE C# API: https://github.com/erenakyildiz-code/PublicBaratrumBackend.git

YOU ALSO NEED THIS Desktop App: https://github.com/erenakyildiz-code/PublicBaratrumDesktopApp.git

YOU WANT TO RUN server.py ON AN MLX CAPABLE COMPUTER (Apple Silicon Mac), OR EDIT IT TO BE USED ON NON-MAC SILICON.

This file explains how you will run the MODELS; NOT THE C# API OR THE DESKTOP APP, THEY ARE WRITTEN IN THEIR OWN REPOSITORIES.

This document is **self-contained**: every file you need is inlined below.
If you get stuck, paste this entire README into an LLM along with your error message and your GPU/OS — it has enough context for the LLM to help you.

**The audio server below (`server.py`) is the ACTUAL server with custom endpoints. It is NOT OpenAI-compatible. Read the endpoint contracts carefully — the client app expects exactly these routes.**

## What you're running

Two services:

| Service | What | Default port | API (custom — read this) |
|---|---|---|---|
| **brain** | llama.cpp server running your GGUF model | `8080` | `POST /completion` LLAMA.cpp TYPE ENDPOINT, you can use openai type if you go and change the c# api MANUALLY. IT IS RECOMMENDED YOU USE GOOGLE'S GEMMA 4 WITH LLAMA.CPP |
| **ears+mouth** | Qwen3-ASR-1.7B speech-to-text + Qwen3-TTS-1.7B-VoiceDesign text-to-speech, one server (`server.py`), MLX / Apple Silicon | `8080` | `WS /ws?session_id=...` + `POST /v1/audio/generate/{session_id}/{chunk_index}` + `POST /v1/audio/done/{session_id}` + `POST /transcribe` + `WS /ws/stt/{session_id}` |

### Platform reality (important)

- **Brain**: NVIDIA (Docker, CUDA) or Apple Silicon (native, Metal). Any platform.
- **Ears+mouth**: **MLX — Apple Silicon Macs ONLY.** Both the ASR and the TTS are MLX models in the same file; there is no Linux/Windows build of this server. If your brain lives on an NVIDIA box, run `server.py` on any Mac on the same LAN and point the client at it.
  - The author's own setup: brain on an RTX 5090 Linux box, ears+mouth on an M-series Mac.
- **Port collision warning:** `server.py` binds **8080**, same as the brain's default. This is fine when the audio server runs on a different machine (the intended setup). If you insist on running everything on one Mac, change the last line of `server.py` to another port (e.g. `8001`) and update the client accordingly.

---

## Section 1 — Brain (llama.cpp)

The brain speaks the standard llama.cpp server API

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

**`docker-compose.yml`** (brain only — ears+mouth is NOT a docker image, see its section)

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

## Section 2 — Ears + Mouth (`server.py`) — Qwen3-ASR + Qwen3-TTS (MLX, Apple Silicon only)

The actual audio server. **One file does BOTH hearing and speaking.** It loads two MLX models at startup:

- **ASR:** `mlx-community/Qwen3-ASR-1.7B-8bit` — I HAVE A THICK TURKISH ACCENT SO I USED THIS HEAVY MODEL. If you are going to talk languages other than english I recommend qwen3-ASR.
- **TTS:** `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` — the voice is **designed with a text description**, not cloned from a reference wav. Edit the `INSTRUCT` constant at the top of the file to change her voice. VoiceDesign voices drift a little between generations — that is the design of this model, not a bug.

**None of this is a request/response audio API.** TTS audio never comes back in an HTTP response — it streams as **raw PCM bytes over a WebSocket**. The five routes:

### TTS flow (mouth)

1. **Open the WebSocket first:** `WS /ws?session_id=<your-session-id>` — the server registers the connection. If you call generate before connecting, you get HTTP 400.
2. **Request audio chunks:** `POST /v1/audio/generate/{session_id}/{chunk_index}` with JSON body `{"input": "<text to speak>", "lang_code": "<lang>"}`. The server runs the TTS and pushes audio to the WebSocket. **Each audio chunk arrives as TWO frames: first a text frame with JSON metadata `{"type": "audio_chunk", "chunk_index": N, "seq": M}`, then a binary frame of little-endian int16 PCM** (mono, 24 kHz for Qwen3-TTS). Your client must tolerate text frames interleaved with the binary audio. Returns `{"status": "streamed"}` when the chunk is fully sent.
3. **Signal end of response:** `POST /v1/audio/done/{session_id}` — the server sends a **text frame `"EOS"`** over the WebSocket so the client knows playback for this turn is complete.

`lang_code` is only read on the **first** generate call for a session; after that it's locked in `session_langs`. NOTE: as shipped, the TTS generate call still passes `lang_code="en"` to the model — if you need her to actually speak other languages, change that one line to use the stored session language.

### Batch transcription (ears, simple path)

`POST /transcribe` — multipart form field **`file`** containing a WAV (16 kHz mono PCM16 recommended — the server does NOT run ffmpeg or convert containers anymore, it feeds the file to the model as-is). Optional query params: `language` (forces a language; omit for auto-detect) and `session_id` (if given and that session has an open `/ws`, the transcript is also pushed there as `{"type": "transcript", "text": ...}`). Returns `{"transcription": "...", "text": "..."}`.

### Streaming transcription (ears, realtime path)

`WS /ws/stt/{session_id}` — streaming ASR for live mic input:

- **You send:** binary frames of **raw, headerless little-endian int16 PCM, 16 kHz mono** (NOT a wav file — strip any header, no webm).
- **Server sends:** every 0.5 s of audio, a text frame `{"type": "hypothesis", "stable": "<best transcript so far>", "unstable": ""}`.
- **You send:** a **text** frame exactly `commit` when the user stops speaking.
- **Server sends:** `{"type": "final", "text": "<final transcript>"}` and resets its buffer for the next utterance.
- **Barge-in:** the moment this socket connects, the server sets a kill flag that aborts any in-flight TTS generation at the next streaming interval — she stops talking when you start listening. The next generate call clears the flag.

### Other details that matter

- Both models serialize on one uvicorn event loop; `asyncio.sleep(0)` yields between streamed TTS chunks so handshakes and ears-connects aren't starved. One Mac, one voice at a time.
- `chunk_index` in the path is currently bookkeeping only — ordering is the client's responsibility.
- you only need to care about the endpoints, if you can return what they return and accept what they accept, you can switch the models out with anything you want.

**`server.py`** (verbatim — this is the real server; note it binds port **8080**, see the port-collision warning above):

```python
import uvicorn
import numpy as np
from fastapi import File, UploadFile, FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import gc
import os
import json
import tempfile
import mlx.core as mx
from mlx_audio.tts.utils import load_model
from mlx_audio.stt import load as load_stt_model

MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
ASR_MODEL_ID = "mlx-community/Qwen3-ASR-1.7B-8bit"
INSTRUCT = "A young woman speaking with a mid-pitched, distinctly feminine voice"

TTS_STREAMING_INTERVAL_S = 0.5
STT_SAMPLE_RATE = 16000
PARTIAL_INTERVAL_S = 0.5
MAX_TAIL_S = 4.0
KEEP_S = 2.0
OVERLAP_S = 1.5

app = FastAPI(title="Baratrum OmniVoice Streamer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = None
asr_model = None
active_connections: dict[str, WebSocket] = {}
session_langs: dict[str, str] = {}
kill_tts = False

class GenerateRequest(BaseModel):
    input: str
    lang_code: str

@app.on_event("startup")
async def load_models():
    global model, asr_model
    print(f"📦 Loading TTS {MODEL_ID}...")
    model = load_model(MODEL_ID, trust_remote_code=False)
    print("✅ TTS Model loaded.")
    print(f"📦 Loading ASR {ASR_MODEL_ID}...")
    asr_model = load_stt_model(ASR_MODEL_ID)
    print("✅ ASR Model loaded.")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: str = Query(...)):
    await websocket.accept()
    active_connections[session_id] = websocket
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.pop(session_id, None)

@app.post("/v1/audio/generate/{session_id}/{chunk_index}")
async def generate_chunk(session_id: str, chunk_index: int, req: GenerateRequest):
    if session_id not in active_connections:
        raise HTTPException(status_code=400, detail="WebSocket not connected for this session.")
    websocket = active_connections[session_id]

    lang = session_langs.setdefault(session_id, req.lang_code)
    print(f"Session: {session_id}, language: {lang}")
    print(req.input)

    global kill_tts
    kill_tts = False
    try:
        seq = 0
        for result in model.generate(
            text=req.input,
            voice="af_heart",
            speed=1.0,
            lang_code="en",
            instruct=INSTRUCT,
            stream=True,
            streaming_interval=TTS_STREAMING_INTERVAL_S,
        ):
            await asyncio.sleep(0)
            if kill_tts:
                print(f"🛑 TTS chunk {chunk_index} killed mid-generation (ears connected)")
                break
            audio = np.asarray(result.audio, dtype=np.float32).flatten()
            if audio.size == 0:
                continue
            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            await websocket.send_text(json.dumps({
                "type": "audio_chunk",
                "chunk_index": chunk_index,
                "seq": seq,
            }))
            await websocket.send_bytes(pcm16.tobytes())
            seq += 1
    except Exception as e:
        print(f"❌ Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        gc.collect()
        mx.clear_cache()

    return {"status": "streamed"}

@app.post("/v1/audio/done/{session_id}")
async def mark_session_done(session_id: str):
    if session_id in active_connections:
        await active_connections[session_id].send_text("EOS")
    return {"status": "eos_sent"}

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    session_id: str = Query(None),
    language: str = Query(None),
):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio body.")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        tmp.write(audio_bytes)
        tmp.close()
        if language:
            result = asr_model.generate(tmp.name, language=language)
        else:
            result = asr_model.generate(tmp.name)
        text = result.text
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        gc.collect()
        mx.clear_cache()
        os.unlink(tmp.name)

    print(f"🎙️ {text}")

    if session_id and session_id in active_connections:
        await active_connections[session_id].send_text(
            json.dumps({"type": "transcript", "text": text})
        )

    return {"transcription": text, "text": text}

def _pcm16_to_f32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

def _decode(audio_np: np.ndarray) -> str:
    text = ""
    for result in asr_model.stream_transcribe(audio_np, temperature=0.0):
        if result.is_final:
            continue
        text += result.text
    return text.strip()

def _stitch(frozen_text: str, window_text: str) -> str:
    fwords = frozen_text.split()
    wwords = window_text.split()
    norm = lambda w: w.strip(".,!?;:\"'").lower()
    for k in range(min(len(fwords), len(wwords), 12), 0, -1):
        if [norm(w) for w in fwords[-k:]] == [norm(w) for w in wwords[:k]]:
            return " ".join(fwords[:-k] + wwords)
    return (frozen_text + " " + window_text).strip()

@app.websocket("/ws/stt/{session_id}")
async def stt_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    global kill_tts
    kill_tts = True
    print(f"🔌 STT client connected: {session_id}")
    audio_chunks: list[np.ndarray] = []
    frozen_text = ""
    frozen_upto = 0
    samples_since_partial = 0
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("bytes") is not None:
                samples = _pcm16_to_f32(msg["bytes"])
                audio_chunks.append(samples)
                samples_since_partial += samples.size

                if samples_since_partial >= int(STT_SAMPLE_RATE * PARTIAL_INTERVAL_S):
                    samples_since_partial = 0
                    full = np.concatenate(audio_chunks)
                    start = max(0, frozen_upto - int(OVERLAP_S * STT_SAMPLE_RATE))
                    window = full[start:]
                    window_text = _decode(window)
                    partial = _stitch(frozen_text, window_text) if frozen_text else window_text

                    tail_s = (full.size - frozen_upto) / STT_SAMPLE_RATE
                    if tail_s > MAX_TAIL_S:
                        new_frozen_upto = full.size - int(KEEP_S * STT_SAMPLE_RATE)
                        into_window_s = (new_frozen_upto - start) / STT_SAMPLE_RATE
                        window_dur_s = window.size / STT_SAMPLE_RATE
                        fwords = partial.split()
                        wwords = window_text.split()
                        cut = int(len(wwords) * into_window_s / window_dur_s)
                        frozen_text = " ".join(fwords[:len(fwords) - len(wwords) + cut])
                        frozen_upto = new_frozen_upto

                    print(f"🎙️ [partial] {partial}")
                    await websocket.send_text(json.dumps({
                        "type": "hypothesis",
                        "stable": partial,
                        "unstable": "",
                    }))
                    gc.collect()
                    mx.clear_cache()
            elif msg.get("text") == "commit":
                if not audio_chunks:
                    print("🏁 COMMIT received but no audio buffered — skipping")
                    continue
                full = np.concatenate(audio_chunks)
                print(f"🏁 COMMIT received — finalizing tail ({(full.size - frozen_upto) / STT_SAMPLE_RATE:.1f}s unfrozen of {full.size / STT_SAMPLE_RATE:.1f}s total)")
                audio_chunks = []

                start = max(0, frozen_upto - int(OVERLAP_S * STT_SAMPLE_RATE))
                window_text = _decode(full[start:])
                text = _stitch(frozen_text, window_text) if frozen_text else window_text

                frozen_text = ""
                frozen_upto = 0
                samples_since_partial = 0

                print(f"🎙️ {text}")
                await websocket.send_text(json.dumps({"type": "final", "text": text}))
                gc.collect()
                mx.clear_cache()
            elif msg.get("text") is not None:
                print(f"📨 Text msg (ignored): {msg['text']!r}")
    except (WebSocketDisconnect, RuntimeError):
        print(f"🔌 STT client disconnected: {session_id}")
    finally:
        gc.collect()
        mx.clear_cache()

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080)
```

### Install & run (macOS, Apple Silicon)

```bash
pip install mlx mlx-audio fastapi "uvicorn[standard]" numpy python-multipart
# python-multipart is REQUIRED — UploadFile (/transcribe) breaks without it.

# 1. Optional: edit the INSTRUCT constant at the top of server.py to change the voice.
# 2. Run:
python server.py
```

First run downloads both model weights (~a few GB total) from HuggingFace.
YOU COULD USE KOKORO OR WHISPER OR ANYTHING ELSE FOR THIS AS LONG AS THE ENDPOINTS DO THE SAME THING THEY DO HERE. THE MODEL IS NOT IMPORTANT THE LOGIC IS.

---

## Section 3 — Verify everything works

```bash
# brain: lists the loaded model
curl http://127.0.0.1:8080/v1/models

# brain: actual completion
curl http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "<bos><start_of_turn>user\nSay hi.<end_of_turn>\n<start_of_turn>model\n", "n_predict": 32}'

# ears (batch): transcribe a wav file (the field name MUST be "file"; send a real wav, 16kHz mono preferred)
curl -F "file=@/path/to/some-audio.wav" http://127.0.0.1:8080/transcribe

# mouth: full handshake test (needs websocat: brew install websocat / pacman -S websocat)
# terminal 1 — connect the websocket, watch {"type":"audio_chunk",...} text frames + binary frames + EOS arrive:
websocat "ws://127.0.0.1:8080/ws?session_id=test1"

# terminal 2 — request speech, then signal done:
curl -X POST http://127.0.0.1:8080/v1/audio/generate/test1/0 \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, I am alive.", "lang_code": "en"}'
curl -X POST http://127.0.0.1:8080/v1/audio/done/test1
```

If the mouth test 400s, you called generate before the websocket was registered — that's the #1 mistake.

Streaming ears (`/ws/stt`) needs a real websocket client because you must mix binary audio frames with a text `commit` frame — websocat can't do both on one socket. Minimal python test:

```python
import asyncio, websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8080/ws/stt/test1") as ws:
        pcm = open("audio-16k-mono.pcm", "rb").read()  # raw headerless PCM16, 16kHz mono
        await ws.send(pcm)        # binary frame
        await ws.send("commit")   # text frame — must be TEXT, not binary
        async for msg in ws:
            print(msg)

asyncio.run(main())
```

---

## Section 4 — Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `could not select device driver "nvidia" with capabilities: [[gpu]]` | Linux: install **nvidia-container-toolkit**, restart docker. Windows: Docker Desktop WSL2 backend, not Hyper-V. |
| `no kernel image is available for execution on the device` | Brain image built for wrong arch. RTX 50xx = sm_120, 40xx = sm_89, 30xx = sm_86, 20xx = sm_75. |
| Brain OOM / `ggml_cuda_host_malloc failed` | Model + context too big for VRAM. Lower `BRAIN_CTX`, or pick a smaller quant (see VRAM table). |
| `Form data requires "python-multipart"` on /transcribe | You skipped a dependency: `pip install python-multipart`. |
| `pip install mlx` / `mlx-audio` fails | You're not on an Apple Silicon Mac. MLX does not exist on Linux/Windows — run `server.py` on a Mac on your LAN instead. |
| HTTP 400 "WebSocket not connected" | You called `/v1/audio/generate/...` before opening `WS /ws?session_id=...`. The websocket must connect FIRST, with the same session_id. |
| Client hears garbage/static from TTS | The stream is **raw int16 PCM, not a wav/mp3 file** — no header. Feed it to your audio pipeline as PCM16 mono at the model's sample rate (24 kHz for Qwen3-TTS). Also remember every binary frame is preceded by a `{"type":"audio_chunk",...}` TEXT frame — if you feed those text frames into the audio buffer you get noise. |
| She speaks English even though you sent another `lang_code` | Known: the generate call passes `lang_code="en"` literally. Edit that one line in `server.py` to use the stored session language. |
| Her voice changes slightly between runs | VoiceDesign drifts by design — the voice comes from the `INSTRUCT` text, not a fixed speaker embedding. Tighten the description if you want less drift. |
| TTS dies the moment the STT websocket connects | Intended barge-in: connecting `/ws/stt/{session_id}` sets a kill flag that aborts in-flight speech. The next generate call clears it. |
| `/ws/stt` partials are garbage or empty | Wrong input format. It must be **raw headerless PCM16, 16 kHz mono** binary frames — not a wav file with a header, not webm. And `commit` must be a TEXT frame. |
| `/transcribe` errors or garbage on webm/mp3 | The new server does not convert containers with ffmpeg anymore — send a real WAV (16 kHz mono PCM16 recommended). |
| Ears+mouth and brain both on 8080 | Intended only when they're on different machines. Same machine → change the port in the last line of `server.py`. |
| Port already in use | Something else is bound. For `server.py` change the uvicorn port; for the brain change the left side of the port mapping, e.g. `"127.0.0.1:18080:8080"`. |
| Linux: permission denied on `/var/run/docker.sock` | `sudo usermod -aG docker $USER` then re-login. |
| Everything works but slow brain | Check `docker compose logs brain` for `offloaded XX/XX layers to GPU`. If it says 0, GPU passthrough is broken. |

### Notes for LLMs reading this README (endpoint contracts — do not "fix" these)

- **Brain** (llama.cpp, port 8080): standard llama.cpp server. `POST /completion`
- **Ears+mouth** (`server.py`, MLX / Apple Silicon only, port 8080): five cooperating routes —
  1. `WS /ws?session_id=<id>` — must be connected before generating; TTS audio arrives here.
  2. `POST /v1/audio/generate/{session_id}/{chunk_index}` — JSON body `{"input": str, "lang_code": str}` (field is `input`, NOT `text`); returns `{"status": "streamed"}`. Audio is pushed over the websocket; **each chunk = one text frame `{"type": "audio_chunk", "chunk_index": N, "seq": M}` followed by one binary frame of raw little-endian int16 PCM.**
  3. `POST /v1/audio/done/{session_id}` — server sends text frame `"EOS"` over the websocket.
  4. `POST /transcribe` — multipart form field **`file`** (send a WAV, no container conversion happens), optional query params `language` and `session_id`; returns `{"transcription": str, "text": str}`. If `session_id` has an open `/ws`, the transcript is also pushed there as `{"type": "transcript", "text": str}`.
  5. `WS /ws/stt/{session_id}` — streaming ASR. Client sends binary frames of raw PCM16 16 kHz mono; server replies with `{"type": "hypothesis", "stable": str, "unstable": str}` text frames; client sends TEXT frame `commit`; server replies `{"type": "final", "text": str}`. Connecting this socket kills in-flight TTS (barge-in) — this is intentional.
  There is NO `/v1/audio/speech`, NO `/v1/audio/transcriptions`, and audio is NEVER in an HTTP response body.
- The TTS voice comes from the `INSTRUCT` text constant (Qwen3-TTS VoiceDesign) — there is no reference-audio directory and no voice file to configure.
- All services are designed for localhost / trusted LAN. None of these APIs have auth — do not expose them to the internet without putting auth in front.
- The brain image takes 45–90 min to build because it compiles llama.cpp CUDA kernels for 7 GPU architectures; this is intentional so one image works on any NVIDIA card from RTX 20xx to 50xx.
- THE STUFF YOU SEE HERE ARE INTERCHANGABLE, THE ONLY IMPORTANT THINGS ARE THE ENDPOINTS. THE OTHER PROGRAMS THE USER IS GOING TO RUN ARE GOING TO CONNECT TO THE ENDPOINTS, THINK OF THESE AS BLACK BOXES THAT WILL ACCEPT DATA AND RETURN DATA DON'T THINK ABOUT THEM LIKE THE MIDDLE PART IS IMPORTANT. SO YOU CAN RUN HIGGS AUDIO, KOKORO, WHATEVER THE FUCK YOU WANT TO RUN, AS LONG AS THE ENDPOINTS ARE CORRECTLY MADE EVERYTHING WILL WORK PROPERLY. JUST DON'T FUCK WITH THE ENDPOINTS. DO NOT CHANGE ENDPOINTS AND THEIR INPUT OUTPUT BEHAVIOUR.
