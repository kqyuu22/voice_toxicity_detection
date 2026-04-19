import whisper
import torch
import os
import librosa
import time
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import INPUT_DIRS, TRANSCRIPTIONS_OUTPUT, ensure_output_dirs

class SpeechTranscriber:
    def __init__(self, model_size="base"):
        print(f"Loading Whisper model '{model_size}'...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size, device=self.device)
        print(f"Model loaded on {self.device}.")
        self.model_size = model_size

    def process_with_metadata(self, audio_path, beam_size=5, temperature=0.0):
        """
        Process audio with metadata collection for performance analysis.
        
        Parameters:
        - beam_size: Search beam width (5-10 recommended, higher = slower)
        - temperature: 0.0 = deterministic, >0 = more varied output
        """
        if not os.path.exists(audio_path):
            return None

        start_time = time.time()

        # 1. PRE-PROCESSING: Load and normalize audio
        audio, _ = librosa.load(audio_path, sr=16000)
        audio = whisper.pad_or_trim(audio)
        
        # 2. FEATURE EXTRACTION: Create Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(self.device)

        # 3. DECODING: Use optimized parameters
        options = whisper.DecodingOptions(
            beam_size=beam_size,
            temperature=temperature,
            fp16=(self.device == "cuda"),
            sample_len=None,
            language="en"
        )

        # Execute the decoding
        result = whisper.decode(self.model, mel, options)
        
        # 4. METADATA CALCULATIONS
        end_time = time.time()
        latency = end_time - start_time
        
        return {
            "text": result.text.strip(),
            "conf": np.exp(result.avg_logprob),
            "latency": f"{latency:.3f}s",
            "latency_sec": latency,
            "no_speech_prob": result.no_speech_prob
        }

    def run_batch_pipeline(self, input_folder, output_folder, beam_size=5, temperature=0.0):
        os.makedirs(output_folder, exist_ok=True)

        files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
        print(f"\n{'='*70}")
        print(f"[{self.model_size.upper()}] Starting Pipeline on {len(files)} files")
        print(f"Parameters: beam_size={beam_size}, temperature={temperature}")
        print(f"{'='*70}\n")

        total_latency = 0
        total_conf = 0
        successful = 0

        for idx, filename in enumerate(sorted(files), 1):
            input_path = os.path.join(input_folder, filename)
            data = self.process_with_metadata(input_path, beam_size=beam_size, temperature=temperature)

            if data:
                total_latency += data['latency_sec']
                total_conf += data['conf']
                successful += 1
                
                print(f"[{idx:2d}/{len(files)}] {filename:30} | Conf: {data['conf']:6.2f} | Latency: {data['latency']}")

                # Save detailed output
                output_path = os.path.join(output_folder, filename.replace(".wav", ".txt"))
                with open(output_path, "w") as f:
                    f.write(f"TEXT: {data['text']}\n")
                    f.write(f"CONFIDENCE: {data['conf']}\n")
                    f.write(f"LATENCY: {data['latency']}\n")

        # Summary statistics
        avg_latency = total_latency / successful if successful > 0 else 0
        avg_conf = total_conf / successful if successful > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"SUMMARY ({self.model_size.upper()}):")
        print(f"  Processed: {successful}/{len(files)} files")
        print(f"  Avg Confidence: {avg_conf:.4f}")
        print(f"  Avg Latency: {avg_latency:.3f}s per file")
        print(f"  Total Time: {total_latency:.1f}s")
        print(f"{'='*70}\n")
        
        return {
            "model": self.model_size,
            "avg_confidence": avg_conf,
            "avg_latency": avg_latency,
            "total_files": len(files),
            "successful": successful
        }


def run_model_comparison():
    """Test different model sizes to find the best balance."""
    ensure_output_dirs()
    INPUT = str(INPUT_DIRS["raw_audio"])
    OUTPUT_BASE = str(TRANSCRIPTIONS_OUTPUT)
    
    # Test different model sizes
    model_sizes = ["tiny", "base", "small", "medium"]
    results = []
    
    for model_size in model_sizes:
        output_folder = os.path.join(OUTPUT_BASE, f"test_{model_size}")
        pipeline = SpeechTranscriber(model_size=model_size)
        result = pipeline.run_batch_pipeline(INPUT, output_folder, beam_size=5, temperature=0.0)
        results.append(result)
    
    # Print comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<10} | {'Avg Conf':<12} | {'Latency/File':<15} | {'Efficiency':<12}")
    print("-"*70)
    
    for r in results:
        efficiency = r['avg_confidence'] / r['avg_latency'] if r['avg_latency'] > 0 else 0
        print(f"{r['model']:<10} | {r['avg_confidence']:<12.4f} | {r['avg_latency']:<15.3f}s | {efficiency:<12.4f}")
    
    print("="*70)
    print("\nRECOMMENDATION:")
    print("  - 'tiny': Fastest, use for real-time or low-resource scenarios")
    print("  - 'base': Best balance, recommended for most use cases")
    print("  - 'small': Better accuracy, ~2x slower than base")
    print("  - 'medium': High accuracy, ~4x slower than base, best for critical applications")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_model_comparison()