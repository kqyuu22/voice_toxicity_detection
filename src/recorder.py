import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import noisereduce as nr
from scipy.signal import butter, lfilter
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AUDIO_OUTPUT, ensure_output_dirs


"""
Note: 
- The threshold has been tuned, and here is the conclusion
    + In quiet environments, you can set it around 0.04
    + In environments with fan noise, you might want to set it around 0.06
    + You can set the threshold higher, but you must speak louder and it may miss softer speech.
- The silence limit is set to 2 seconds. If the sound drops below the threshold for 2 seconds, it will finish recording.
"""

class SingleShotRecorder:
    """This class encapsulates the logic for a single recording session triggered by voice activity.
    It listens to the microphone, detects when the user starts and stops speaking, and saves the final cleaned audio to a file."""
    def __init__(self, threshold=0.05, silence_limit=2.0, fs=44100):
        self.fs = fs
        self.threshold = threshold
        self.silence_limit = silence_limit
        self.chunk_size = 1024
        
        self.recording_buffer = []
        self.silent_chunks_count = 0
        self.is_recording_active = False
        self.has_finished_recording = False # The "Exit" trigger

    def apply_cleaning(self, audio):
        """Final cleanup station."""
        # High-pass filter
        nyq = 0.5 * self.fs
        b, a = butter(5, 100/nyq, btype='high')
        filtered = lfilter(b, a, audio)
        
        # Noise Reduce
        reduced = nr.reduce_noise(y=filtered, sr=self.fs, stationary=True, prop_decrease=1.0)
        
        # Normalize
        peak = np.max(np.abs(reduced))
        if peak > 0:
            reduced = reduced / peak
        return reduced

    def stream_callback(self, indata, frames, time, status):
        volume = np.linalg.norm(indata) / np.sqrt(len(indata))
        
        if volume > self.threshold:
            if not self.is_recording_active:
                print("\n[VAD] Voice detected. Recording...")
            self.is_recording_active = True
            self.silent_chunks_count = 0
            self.recording_buffer.append(indata.copy())
        else:
            if self.is_recording_active:
                self.recording_buffer.append(indata.copy())
                self.silent_chunks_count += 1
                
                # If silence exceeds limit, stop and mark as finished
                if self.silent_chunks_count > (self.silence_limit * self.fs / self.chunk_size):
                    print("[VAD] Silence threshold met. Finalizing...")
                    self.is_recording_active = False
                    self.has_finished_recording = True

def main():
    ensure_output_dirs()
    recorder = SingleShotRecorder(threshold=0.06, silence_limit=2.0)
    
    print("Pipeline Ready. Waiting for audio...")
    
    with sd.InputStream(samplerate=recorder.fs, channels=1, 
                        blocksize=recorder.chunk_size, callback=recorder.stream_callback):
        # Wait for the recorder to flag completion
        while not recorder.has_finished_recording:
            sd.sleep(100)
    
    # Execution moves here once the loop breaks
    if recorder.recording_buffer:
        raw_audio = np.concatenate(recorder.recording_buffer, axis=0).flatten()
        clean_audio = recorder.apply_cleaning(raw_audio)
        
        wav.write(str(AUDIO_OUTPUT), recorder.fs, (clean_audio * 32767).astype(np.int16))
        print(f"\n--- SUCCESS ---")
        print(f"File saved: {AUDIO_OUTPUT}")
        print("Pipeline closing automatically.")
    else:
        print("\nNo audio was captured.")

if __name__ == "__main__":
    main()