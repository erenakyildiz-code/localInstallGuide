import uvicorn
import numpy as np
from fastapi import File, UploadFile, FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gc
import os
import re
import json
import tempfile
import mlx.core as mx
from mlx_audio.tts.utils import load_model
from mlx_audio.stt import load as load_stt_model
from mlx_lm import load as load_llm, generate as llm_generate

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-5bit"
ASR_MODEL_ID = "mlx-community/Qwen3-ASR-0.6B-8bit"
EMOTION_MODEL_ID = "mlx-community/Qwen3-0.6B-4bit"
INSTRUCT = "A young woman speaking with a mid-pitched, distinctly feminine voice"

app = FastAPI(title="Baratrum OmniVoice Streamer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = None
asr_model = None
emotion_model = None
emotion_tokenizer = None
active_connections: dict[str, WebSocket] = {}
session_langs: dict[str, str] = {}

class GenerateRequest(BaseModel):
    input: str
    lang_code: str

@app.on_event("startup")
async def load_models():
    global model, asr_model, emotion_model, emotion_tokenizer
    print(f"📦 Loading TTS {MODEL_ID}...")
    model = load_model(MODEL_ID, trust_remote_code=False)
    print("✅ TTS Model loaded.")
    print(f"📦 Loading ASR {ASR_MODEL_ID}...")
    asr_model = load_stt_model(ASR_MODEL_ID)
    print("✅ ASR Model loaded.")
    print(f"📦 Loading Emotion {EMOTION_MODEL_ID}...")
    emotion_model, emotion_tokenizer = load_llm(EMOTION_MODEL_ID)
    print("✅ Emotion Model loaded.")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: str = Query(...)):
    await websocket.accept()
    active_connections[session_id] = websocket
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.pop(session_id, None)

# ---------------------------------------------------------
# TTS: analyze emotion, send it, then generate and stream PCM16
# ---------------------------------------------------------
@app.post("/v1/audio/generate/{session_id}/{chunk_index}")
async def generate_chunk(session_id: str, chunk_index: int, req: GenerateRequest):
    if session_id not in active_connections:
        raise HTTPException(status_code=400, detail="WebSocket not connected for this session.")
    websocket = active_connections[session_id]

    lang = session_langs.setdefault(session_id, req.lang_code)
    print(f"Session: {session_id}, language: {lang}")
    print(req.input)

    # Ask the small model which animation fits this line of text
    prompt = emotion_tokenizer.apply_chat_template(
        [{"role": "user", "content": (
            "You control the animations of a virtual character. "
            "The character is about to speak a line. Pick the single best matching animation "
            "from this exact list: "
            "excited, happy, sad, angry, thinking, sleepy, relaxed, lookAround, sway, dance, "
            "greeting, goodbye, agree, hardAgree, disagree, annoyed, doubtful, sarcastic, "
            "relieved, cocky, ack, dismiss, lookAway, happyGesture, angryGesture, clap, "
            "peace, kiss, shy, recoil, point, surprised, neutral. "
            "Also pick an intensity between 0.1 and 0.35. "
            'Answer with ONLY a JSON object, e.g. {"emotion": "happy", "intensity": 0.35}\n'
            f'Line: "{req.input}"'
        )}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    raw = llm_generate(emotion_model, emotion_tokenizer, prompt=prompt, max_tokens=32, verbose=False)

    # Pull the {...} out of the reply, clamp intensity to 0.35 max, fall back to neutral
    try:
        data = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
        emotion = str(data.get("emotion", "neutral"))
        intensity = max(0.0, min(0.35, float(data.get("intensity", 0.35))))
    except Exception:
        emotion = "neutral"
        intensity = 0.0

    print(f"🎭 {emotion} @ {intensity}")
    await websocket.send_text(json.dumps({"type": "emotion", "emotion": emotion, "intensity": intensity}))

    # Generate the audio for this line and stream it as PCM16
    try:
        for result in model.generate(
            text=req.input,
            voice="af_heart",
            speed=1.0,
            lang_code="en",
            instruct=INSTRUCT,
        ):
            audio = np.asarray(result.audio, dtype=np.float32).flatten()
            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            await websocket.send_bytes(pcm16.tobytes())
    except Exception as e:
        print(f"❌ Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        gc.collect()
        mx.clear_cache()

    return {"status": "streamed", "emotion": emotion, "intensity": intensity}

@app.post("/v1/audio/done/{session_id}")
async def mark_session_done(session_id: str):
    if session_id in active_connections:
        await active_connections[session_id].send_text("EOS")
    return {"status": "eos_sent"}

# ---------------------------------------------------------
# STT
# ---------------------------------------------------------
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

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080)
