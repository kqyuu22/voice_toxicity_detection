import whisper
import torch
import os
import librosa
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT_DIRS, ensure_output_dirs, AUDIO_OUTPUT

class SpeechTranscriber:
    def __init__(self, model_size="medium"):
        """
        Initialize transcriber with optimized parameters.
        
        Model size recommendations (tested and optimized):
        - "tiny": ~39M parameters, fastest, lowest accuracy (10x faster than base)
        - "base": ~74M parameters, good balance, 1x baseline speed
        - "small": ~244M parameters, better accuracy, ~2x slower than base
        - "medium": ~769M parameters, high accuracy, ~4x slower - BEST PERFORMANCE (RECOMMENDED)
        - "large": ~1550M parameters, best accuracy, ~8x slower
        
        Default: "medium" - Best balance between accuracy and reasonable speed
        """
        print(f"Loading Whisper model '{model_size}'...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size, device=self.device)
        print(f"Model loaded on {self.device}.")
        self.model_size = model_size

    def transcribe(self, audio_path, language="en", temperature=0.0, beam_size=5):
        """
        Transcribe audio with optimized parameters.
        
        Parameters:
        - audio_path: Path to audio file
        - language: Language code (e.g., "en" for English)
        - temperature: 0.0 for deterministic output, higher for more variation
        - beam_size: Higher values = more accurate but slower (typical: 5)
        """
        if not os.path.exists(audio_path):
            return f"Error: {audio_path} not found."

        print(f"Loading audio via Librosa: {audio_path}")
        
        # Librosa loads and resamples to 16000Hz automatically
        audio_array, _ = librosa.load(audio_path, sr=16000)
        
        print(f"Transcribing with {self.model_size} model...")
        
        # Transcribe with optimized parameters
        result = self.model.transcribe(
            audio_array,
            language=language,
            temperature=temperature,
            beam_size=beam_size,
            fp16=(self.device == "cuda"),
            verbose=False
        )
        
        return result['text'].strip()

if __name__ == "__main__":
    # Test with your existing file - using optimized "medium" model
    ensure_output_dirs()
    test_file = str(AUDIO_OUTPUT)
    if not os.path.exists(test_file):
        print(f"Warning: {test_file} not found.")
        print("Run recorder.py first to generate audio.")
    else:
        ts = SpeechTranscriber(model_size="medium")
        text = ts.transcribe(test_file, language="en", temperature=0.0, beam_size=5)
        
        # Get filename without extension
        filename = Path(test_file).stem
        output_file = OUTPUT_DIRS["audio"] / f"{filename}_transcription.txt"
        
        # Save to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        
        # Print to console
        print("-" * 40)
        print("TRANSCRIPTION:", text)
        print("-" * 40)
        print(f"Saved to: {output_file}")
