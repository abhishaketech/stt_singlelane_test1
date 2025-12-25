import json
import base64
import numpy as np
import soundfile as sf
from pathlib import Path

# --- CONFIGURATION ---
# Replace this with the name of the JSON file you want to convert
target_filename = r"transcription_results\recording_bbe6aa94-4b92-444c-9e43-087252568fe5.json"
# ---------------------

def convert_to_wav():
    path = Path(target_filename)
    
    # 1. Check if file exists
    if not path.exists():
        print(f"❌ Error: Could not find file at: {path}")
        print("   Make sure you put the correct filename in the script!")
        return

    # 2. Load JSON
    print(f"📂 Reading {path}...")
    with open(path, 'r') as f:
        data = json.load(f)

    # 3. Extract Base64 String
    b64_string = data.get("voice_fingerprint_sample_b64")
    if not b64_string:
        print("❌ Error: No 'voice_fingerprint_sample_b64' found in this JSON.")
        return

    # 4. Decode: Base64 -> Raw Bytes -> Numpy Array
    try:
        raw_bytes = base64.b64decode(b64_string)
        audio_data = np.frombuffer(raw_bytes, dtype=np.float32)
        
        print(f"   Audio Shape: {audio_data.shape}")
        print(f"   Duration:    {len(audio_data)/16000:.2f} seconds")

        # 5. Save as WAV
        output_wav = path.stem + ".wav"  # e.g., recording_123.wav
        
        # We use 16000 Hz because that is what your server recorded at
        sf.write(output_wav, audio_data, 16000)
        
        print(f"✅ Success! Saved audio to: {output_wav}")
        print("   You can now play this file with any media player.")

    except Exception as e:
        print(f"❌ Failed to convert: {e}")

if __name__ == "__main__":
    convert_to_wav()