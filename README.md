# Audio Recording & Transcription Experiment

A project to experiment with recording audio with voice activity detection (VAD) and transcribing using OpenAI Whisper.

## Pipeline

```
1. python src/recorder.py     -> Speak into microphone
                              -> Saves output/audio/final_output.wav

2. python src/transcriber.py  -> Transcribes the wav file
                              -> Saves output/audio/final_output_transcription.txt
```

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
test/
├── README.md
├── requirements.txt
├── config.py
│
├── src/
│   ├── recorder.py              # Record audio with VAD
│   ├── transcriber.py           # Transcribe audio
│   ├── transcriber_test.py      # Test different models
│   ├── threshold_testing.py     # Test VAD threshold
│   ├── results_analysis.py      # Analyze WER
│   └── result_visualization.py  # Generate charts
│
├── data/
│   ├── raw/10dB/                # Dataset (noizeus)
│   ├── reference/               # Ground truth labels
│   └── training/                # Training datasets
│
└── outputs/
    ├── audio/
    ├── transcriptions/
    ├── visualizations/
    └── analysis/
```

## Quick Start

Run from project root: `python src/<script.py>`

### Recording

1. Test VAD threshold (optional):
   ```bash
   python src/threshold_testing.py
   ```
   Adjust `THRESHOLD` in the script based on your environment.

2. Record audio:
   ```bash
   python src/recorder.py
   ```
   Speaks until silence is detected. Saves to `outputs/audio/final_output.wav`.

### Transcription

Transcribe the recorded file:
```bash
python src/transcriber.py
```
Uses Whisper medium model. Results printed to console.

### Model Testing

Test and compare models (tiny, base, small, medium):
```bash
python src/transcriber_test.py
```
Results saved to `outputs/transcriptions/test_{model}/`.

### Analysis

Calculate WER and accuracy:
```bash
python src/results_analysis.py
```

Visualize performance:
```bash
python src/result_visualization.py
```

## Details

### Recording

- **threshold_testing.py**: Test and adjust VAD threshold value for your environment
- **recorder.py**: Records audio until silent (silence_limit=2 seconds). Applies high-pass filter, noise reduction, and normalization. Saves to `outputs/audio/final_output.wav`.

### Transcription

- **Model**: OpenAI Whisper
- **Tested Models**: tiny, base, small, medium
- **Dataset**: Speech noise dataset from https://ecs.utdallas.edu/loizou/speech/noizeus/
- **Selected Model**: medium (~769M parameters) - provides best accuracy
- **transcriber.py**: Transcribes single audio file with Whisper medium model
- **transcriber_test.py**: Batch process and compare all models


