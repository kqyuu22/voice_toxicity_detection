import sounddevice as sd
import numpy as np

"""
This script tests the audio thresholding logic by continuously monitoring the microphone input and printing whether it's currently "RECORDING" or "SILENT" based on the defined volume threshold. 
Adjust the THRESHOLD value as needed to find the right balance for your environment.
"""

# --- PARAMETERS ---
FS = 44100          # Sample rate
BLOCK_SIZE = 1024   # How many samples per block
THRESHOLD = 0.04    # Minimum volume to be "audible" (Adjust this based on the environment noise)
SILENCE_CHUNKS = 30 # How many consecutive silent blocks before we stop

def audio_callback(indata, frames, time, status):
    """This function is called for every block of audio captured."""
    volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
    
    if volume_norm > THRESHOLD:
        print(f"Status: RECORDING | Volume: {volume_norm:.4f}", end="\r")
    else:
        print(f"Status: SILENT    | Volume: {volume_norm:.4f}", end="\r")

print(f"Listening... (Press Ctrl+C to stop)")
try:
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=FS, blocksize=BLOCK_SIZE):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\nStopped by user.")