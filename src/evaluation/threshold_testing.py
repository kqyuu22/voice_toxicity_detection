import argparse
import sounddevice as sd
import numpy as np
import sys
from pathlib import Path
from typing import List

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DEFAULT_VAD_THRESHOLD, VAD_THRESHOLD_FILE

"""
This script tests the audio thresholding logic by continuously monitoring the microphone input and printing whether it's currently "RECORDING" or "SILENT" based on the defined volume threshold. 
Adjust the THRESHOLD value as needed to find the right balance for your environment.
"""

# --- PARAMETERS ---
FS = 44100         # Sample rate
BLOCK_SIZE = 1024  # How many samples per block

def parse_args():
    parser = argparse.ArgumentParser(
        description="Monitor the microphone and estimate a suitable VAD threshold."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Seconds to sample audio before computing a threshold.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=VAD_THRESHOLD_FILE,
        help="Path to write the computed threshold.",
    )
    return parser.parse_args()


def audio_callback(indata: np.ndarray, frames: int, time, status, samples: List[float]):
    """This function is called for every block of audio captured."""
    # RMS volume calculation 
    # to have a single, positive value representing the "energy" or "loudness" of that specific chunk of time
    volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
    samples.append(volume_norm)
    print(f"Listening... Volume: {volume_norm:.4f}", end="\r")


def compute_threshold(samples):
    if not samples:
        return DEFAULT_VAD_THRESHOLD

    # Look at the 90th percentile of all recorded volume chunks.
    noise_floor = np.percentile(samples, 90)

    # Multiply by 2.5 (rule of thumb)
    # to set the threshold above the noise floor, 
    # ensuring that it captures actual speech while minimizing false positives from background noise
    return max(DEFAULT_VAD_THRESHOLD, noise_floor * 2.5)


def main():
    args = parse_args()
    samples = []
    print(f"Listening for {args.duration:.1f}s... (Ctrl+C to stop)")
    try:
        with sd.InputStream(
            callback=lambda indata, frames, time, status: audio_callback(
                indata, frames, time, status, samples
            ),
            channels=1,
            samplerate=FS,
            blocksize=BLOCK_SIZE,
        ):
            sd.sleep(int(args.duration * 1000))
    except KeyboardInterrupt:
        print("\nStopped by user.")

    threshold = compute_threshold(samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(f"{threshold:.6f}", encoding="utf-8")
    print(f"\nRecommended threshold: {threshold:.6f}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()