
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
REFERENCE_AUDIO_PATH = "/Users/devil/moss/VOICESDIR"
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
