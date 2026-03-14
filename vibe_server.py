from fastapi import FastAPI, UploadFile, Form
import torch
from transformers import ClapModel, ClapProcessor
import numpy as np
import io
import uvicorn
import librosa

app = FastAPI()

print("⏳ Loading CLAP Model... (This happens only once)")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(device)
processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
print(f"✅ Model Loaded on {device}! Server Ready.")

@app.post("/check")
async def check_vibe(file: UploadFile, prompt: str = Form(...)):
    # 1. Read Audio Bytes
    contents = await file.read()
    # Convert bytes back to numpy array (assuming float32 raw buffer)
    audio_array = np.frombuffer(contents, dtype=np.float32)
    
    # 2. Resample if necessary (CLAP expects 48000)
    # Note: We assume the client might send 44100 or 48000
    # For now, let's assume client sends 48000 as per the plan
    
    # 3. Run Inference
    inputs = processor(text=[prompt], audios=[audio_array], sampling_rate=48000, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Similarity is dots product of normalized embeddings, scaled by 100
        similarity = outputs.logits_per_audio.item() / 100.0

    print(f"[SERVER] Analyzed: '{prompt}' | Similarity: {similarity:.4f}")
    return {"score": similarity}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
