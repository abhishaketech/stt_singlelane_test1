import base64
import numpy as np
import soundfile as sf
import json
from pathlib import Path

def save_base64_to_wav(json_result_path):
    """
    Reads a result JSON file, extracts the Base64 fingerprint, 
    and saves it as a playable WAV file.
    """
    path = Path(json_result_path)
    if not path.exists():
        print("File not found!")
        return

    with open(path, 'r') as f:
        data = json.load(f)

    b64_string = data.get("voice_fingerprint_sample_b64")
    
    if not b64_string:
        print("No fingerprint found in this file.")
        return

    # 1. Decode Base64 to Bytes
    raw_bytes = base64.b64decode(b64_string)
    
    # 2. Convert Bytes to Numpy Array (Float32)
    audio_array = np.frombuffer(raw_bytes, dtype=np.float32)
    
    # 3. Save as WAV
    output_filename = path.stem + "_fingerprint.wav"
    sf.write(output_filename, audio_array, 16000)
    
    print(f"✅ Success! Saved audio to: {output_filename}")

# --- Usage ---
# Look in your 'transcription_results' folder for the newest .json file
# Example: save_base64_to_wav("transcription_results/some_file.json")