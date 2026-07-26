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
    # Hardcoding to CPU for tonight so your RTX 5090 doesn't crash on cu121
    device_map="cpu", 
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
