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
