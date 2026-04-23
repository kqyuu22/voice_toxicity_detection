# Voice Toxicity Detection

This project provides a tool for recording audio locally and analyzing it for toxic content. It aims to automate the transition from spoken word to a toxicity classification label.

## Pipeline

```
python run_pipeline.py        -> Runs threshold check, recording, transcription,
                              -> and toxicity prediction in order

Individual steps:
1. python src/pipeline/record.py          -> Saves outputs/pipeline_results/recording_output.wav
2. python src/pipeline/transcribe.py       -> Saves outputs/pipeline_results/recording_output_transcription.txt
3. python src/pipeline/predict_toxicity.py  -> Predicts toxicity without retraining
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
│   ├── pipeline/
│   │   ├── record.py              # Record audio with VAD
│   │   ├── transcribe.py           # Transcribe audio
│   │   └── predict_toxicity.py      # Predict toxicity from latest transcript
│   └── evaluation/
│       ├── transcriber_test.py      # Test different speech-to-text models
│       ├── threshold_testing.py     # Test VAD threshold
│       ├── results_analysis.py      # Analyze WER
│       └── result_visualization.py  # Generate charts
│
├── models/
│   ├── toxic_model/                 # Saved PyTorch classification model files
│   └── text_classification.py       # Classification model architecture and training script
│
├── run_pipeline.py            # Run pipeline steps with one command
│
├── data/
│   ├── raw/10dB/                # Dataset (noizeus), for speech-to-text models
│   └── reference/               # Ground truth labels
│
│
└── outputs/
    ├── pipeline_results/
    └── evaluation/
        ├── transcriptions/
        ├── visualizations/
        └── analysis/
```

## Quick Start

Run these commands from the project root directory.

### 1. Setup toxic classifier


**Option A:**
Train the toxicity classifier once before using `predict`:
```bash
python models/text_classification.py train
```

**Option B:** 

1. Manually download the folder `toxic_model/` from this [Google Drive link](https://drive.google.com/drive/folders/1swarH4R_sV-BlQPPcavakTNJbV7EBXHL?usp=drive_link)

2. Alternative: Google Drive Desktop (Synced)

- Install Google Drive for Desktop so your drive appears as a local disk.

- Place the model folder at: G:/My Drive/toxic_model/

### 2. Run the Pipeline

Then run the full audio-to-toxicity pipeline with one command:
```bash
python run_pipeline.py
```

By default, this runs `threshold_testing.py` for 5 seconds, then `record.py`, `transcribe.py`, and `predict_toxicity.py`. The recorder step times out after 120 seconds unless `--record-timeout-seconds` is changed. To skip the threshold check:
```bash
python run_pipeline.py --skip-threshold
```

### Recording

1. Test VAD threshold (optional):
   ```bash
   python src/evaluation/threshold_testing.py
   ```
   Adjust `THRESHOLD` in the script based on your environment.

2. Record audio:
   ```bash
   python src/pipeline/record.py
   ```
   Speaks until silence is detected. Saves to `outputs/pipeline_results/recording_output.wav`.

### Transcription

Transcribe the recorded audio before running toxicity prediction:
```bash
python src/pipeline/transcribe.py
```
Uses Whisper medium model and always writes the latest transcript to `outputs/pipeline_results/recording_output_transcription.txt`.

### Toxic Text Classification

After `src/pipeline/transcribe.py` has created `outputs/pipeline_results/recording_output_transcription.txt`, classify that transcript without passing any file path:
```bash
python src\pipeline\predict_toxicity.py
```

`src\pipeline\predict_toxicity.py` is a wrapper that calls `models\text_classification.py predict` from the project root. `predict` only loads a saved classifier from `models\toxic_model` by default, with legacy fallback path `toxic_classifier_model`; it does not train. If the default transcript file is missing or empty, it exits with a clear message asking you to run `python src\pipeline\transcribe.py` again.

You can still classify a custom text file when needed:
```bash
python src\pipeline\predict_toxicity.py --file outputs\pipeline_results\recording_output_transcription.txt
```

The classifier splits the transcript by commas and sentence punctuation. A segment is counted as toxic when any toxicity label is at least `0.5`; the final toxicity percentage is `toxic_segments / total_segments * 100`.


## Dataset

**For Speech-to-text Model**: [NOIZEUS](https://ecs.utdallas.edu/loizou/speech/noizeus/) (developed by Dr. Philip Loizou, UT Dallas). 

**For Toxic Classification Model**: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset (mirrored by [thesofakillers](https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge)).