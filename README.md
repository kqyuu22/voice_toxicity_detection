# Audio Recording & Transcription Experiment

A project to experiment with recording audio with voice activity detection (VAD) and transcribing using OpenAI Whisper.

## Pipeline

```
python src/run_pipeline.py    -> Runs threshold check, recording, transcription,
                              -> and toxicity prediction in order

Individual steps:
1. python src/recorder.py          -> Saves outputs/audio/final_output.wav
2. python src/transcriber.py       -> Saves outputs/audio/final_output_transcription.txt
3. python src/predict_toxicity.py  -> Predicts toxicity without retraining
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
│   ├── predict_toxicity.py      # Predict toxicity from latest transcript
│   ├── run_pipeline.py          # Run pipeline steps with one command
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

Run these commands from `recorder_and_transcriber`.

Train the toxicity classifier once before using `predict`:
```bash
python ..\Text_classification.py train
```

Then run the full audio-to-toxicity pipeline with one command:
```bash
python src\run_pipeline.py
```

By default, this runs `threshold_testing.py` for 5 seconds, then `recorder.py`, `transcriber.py`, and `predict_toxicity.py`. To skip the threshold check:
```bash
python src\run_pipeline.py --skip-threshold
```

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

Transcribe the recorded audio before running toxicity prediction:
```bash
python src/transcriber.py
```
Uses Whisper medium model and always writes the latest transcript to `outputs/audio/final_output_transcription.txt`.

### Toxic Text Classification

After `src/transcriber.py` has created `outputs/audio/final_output_transcription.txt`, classify that transcript without passing any file path:
```bash
python src\predict_toxicity.py
```

`predict` only loads the saved classifier from `..\toxic_classifier_model`; it does not train. If the default transcript file is missing or empty, it exits with a clear message asking you to run `python src\transcriber.py` again.

You can still classify a custom text file when needed:
```bash
python src\predict_toxicity.py --file outputs\audio\final_output_transcription.txt
```

The classifier splits the transcript by commas and sentence punctuation. A segment is counted as toxic when any toxicity label is at least `0.5`; the final toxicity percentage is `toxic_segments / total_segments * 100`.

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


