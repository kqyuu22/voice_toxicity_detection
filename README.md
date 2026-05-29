# Voice Toxicity Detection

Local audio-to-text-to-toxicity pipeline with Whisper transcription and a multi-label toxic comment classifier. The workflow records speech, transcribes it, and estimates toxicity by segment.

## Pipeline

Run the full pipeline:
```
python run_pipeline.py
```

Pipeline steps (individual scripts):
1. python src/pipeline/record.py           &rarr; outputs/pipeline_results/recording_output.wav
2. python src/pipeline/transcribe.py       &rarr; outputs/pipeline_results/recording_output_transcription.txt
3. python src/pipeline/predict_toxicity.py &rarr; toxicity report for latest transcript

If the recorder times out without detecting speech, the pipeline falls back to a dataset transcript
and skips live transcription. A flag file is written to outputs/pipeline_results/used_dataset_fallback.flag
so run_pipeline.py can report the source and continue safely.

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Project Structure

```
README.md
requirements.txt
config.py
run_pipeline.py
├── src/
│   ├── pipeline/
│   │   ├── record.py                # Microphone recording with VAD + fallback
│   │   ├── transcribe.py            # Whisper transcription for the latest recording
│   │   └── predict_toxicity.py      # Wrapper for model prediction
│   └── evaluation/
│       ├── threshold_testing.py     # Live VAD threshold monitor
│       ├── transcriber_test.py      # Batch STT model comparison
│       ├── results_analysis.py      # WER analysis for transcriptions
│       ├── result_visualization.py  # Confidence/latency charts
│       └── toxicity_evaluation.py   # Classifier metrics + confusion matrices
├── models/
│   ├── text_classification.py       # Training + prediction CLI
│   └── toxic_model/                 # Saved model and tokenizer (if present)
├── data/
│   ├── raw/10dB/                     # Audio dataset (NOIZEUS)
│   └── reference/                    # Ground truth transcripts
└── outputs/
    ├── pipeline_results/
    └── evaluation/
        ├── transcriptions/
        ├── visualizations/
        └── analysis/
```

## Quick Start

### 1. Prepare the toxicity classifier

Option A (train locally):
```bash
python models/text_classification.py train
```

Option B (use a pretrained folder):
1. Download toxic_model/ from Google Drive: https://drive.google.com/drive/folders/1swarH4R_sV-BlQPPcavakTNJbV7EBXHL?usp=drive_link
2. Place it at models/toxic_model/

The training script also checks these fallback paths:
- G:/My Drive/DADN/toxic_model
- toxic_classifier_model

### 2. Run the pipeline

```bash
python run_pipeline.py
```

Useful flags:
```
python run_pipeline.py --skip-threshold
python run_pipeline.py --threshold-seconds 5
python run_pipeline.py --record-timeout-seconds 120
python run_pipeline.py --skip-predict
python run_pipeline.py --predict-threshold 0.5
python run_pipeline.py --json
```

### Manual steps

Record:
```bash
python src/evaluation/threshold_testing.py
python src/pipeline/record.py
```

Transcribe:
```bash
python src/pipeline/transcribe.py
```

Predict toxicity:
```bash
python src/pipeline/predict_toxicity.py
python src/pipeline/predict_toxicity.py --file outputs/pipeline_results/recording_output_transcription.txt
```

Prediction notes:
- The transcript is split by commas and sentence punctuation.
- A segment is toxic if any label probability is >= the threshold (default 0.5).
- The final score is toxic_segments / total_segments * 100.

## Evaluation

Transcriber comparison (tiny/base/small/medium):
```bash
python src/evaluation/transcriber_test.py
```

WER analysis for transcription outputs:
```bash
python src/evaluation/results_analysis.py
```

Visualization of confidence/latency:
```bash
python src/evaluation/result_visualization.py
```

Classifier metrics + confusion matrices:
```bash
python src/evaluation/toxicity_evaluation.py
```

## Datasets

- Speech-to-text: NOIZEUS (https://ecs.utdallas.edu/loizou/speech/noizeus/)
- Toxic classification: Jigsaw Toxic Comment Classification Challenge (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- HF mirror: https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge