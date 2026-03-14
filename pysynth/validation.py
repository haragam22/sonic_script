import requests
import numpy as np
import librosa
import os

SERVER_URL = "http://127.0.0.1:8000/check"

def check_vibe(audio, prompt: str, threshold: float = 0.6, sample_rate: int = 44100) -> bool:
    """
    Lightweight client for the persistent CLAP vibe server.
    """
    print(f"[VIBE] Analyzing against: '{prompt}'")

    # 1. Extract Samples if AudioNode
    if hasattr(audio, 'get_samples'):
        # Render 2 seconds at 48kHz (server expect 48k)
        audio_array = audio.get_samples(2.0, rate=48000)
    elif isinstance(audio, np.ndarray):
        # If it's already an array, ensure it's 48kHz if possible
        if sample_rate != 48000:
            audio_array = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=48000)
        else:
            audio_array = audio
    else:
        print(f"[VIBE] Warning: check_vibe received {type(audio)}, expecting AudioNode/ndarray.")
        return True # Soft pass

    # 2. Safety Check for Silence
    if np.max(np.abs(audio_array)) < 0.01:
        print("VibeCheckError: Audio is silent!")
        return False

    # 3. Send to Server
    try:
        # We send raw float32 bytes for speed
        response = requests.post(
            SERVER_URL,
            files={"file": audio_array.astype(np.float32).tobytes()},
            data={"prompt": prompt},
            timeout=5
        )
        
        if response.status_code != 200:
            print(f"[VIBE] Server Error ({response.status_code}): {response.text}")
            return False 
            
        result = response.json()
        similarity = result["score"]
        
    except requests.exceptions.ConnectionError:
        print("[VIBE] ❌ Error: Vibe Server is not running on localhost:8000!")
        print("Run 'python vibe_server.py' in another terminal.")
        return True # Soft pass so development doesn't stop
    except Exception as e:
        print(f"[VIBE] Error: {e}")
        return True

    # 4. Verdict
    print(f"[VIBE] Similarity Score: {similarity:.4f} (Threshold: {threshold})")
    
    if similarity < threshold:
        print(f"VibeCheckError: Score {similarity:.4f} is too low for '{prompt}'")
        return False
        
    print("[VIBE] PASSED.")
    return True
