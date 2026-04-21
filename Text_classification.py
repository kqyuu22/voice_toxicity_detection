"""
Text Classification - Toxic Comment Detection
===============================================
Train an XLM-RoBERTa model to classify toxic comments (multi-label).
After training, the model is saved to disk for later reuse without retraining.

Usage:
    python Text_classification.py train
    python Text_classification.py train --test
    python Text_classification.py predict --text "hello, you loser."
    python Text_classification.py predict --file recorder_and_transcriber/outputs/audio/final_output_transcription.txt
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    classification_report,
    multilabel_confusion_matrix,
)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent
TRANSCRIBER_PROJECT_ROOT = PROJECT_ROOT / "recorder_and_transcriber"
DEFAULT_TRANSCRIPTION_FILE = (
    TRANSCRIBER_PROJECT_ROOT / "outputs" / "audio" / "final_output_transcription.txt"
)

CONFIG = {
    # Model
    "model_name": "xlm-roberta-base",
    "max_length": 256,

    # Training
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,

    # Data
    "test_size": 0.1,

    # Output
    "output_dir": str(PROJECT_ROOT / "toxic_classifier_output"),
    "model_save_path": str(PROJECT_ROOT / "toxic_classifier_model"),
}

TOXICITY_LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


class PredictionInputError(RuntimeError):
    """Raised when predict cannot find usable input text."""


# ──────────────────────────────────────────────
# Text Preprocessing
# ──────────────────────────────────────────────

def clean_text(text):
    """Remove URLs, IPs, and normalize whitespace."""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)        # Remove URLs
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '', text)  # Remove IPs
    text = re.sub(r'[\n\t\r]+', ' ', text)                    # Normalize newlines/tabs
    text = re.sub(r'\s{2,}', ' ', text)                       # Collapse multiple spaces
    return text.strip()


def split_speech_segments(text):
    """Split transcript into spoken segments using comma/sentence punctuation."""
    segments = []
    for segment in re.split(r"[,.;:!?]+", text):
        cleaned = clean_text(segment)
        if cleaned:
            segments.append(cleaned)
    return segments


# ──────────────────────────────────────────────
# Data Loading & Preparation
# ──────────────────────────────────────────────

def load_data():
    """Load the Jigsaw toxic comment dataset from HuggingFace."""
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset(
        "thesofakillers/jigsaw-toxic-comment-classification-challenge",
        split="train",
    )
    df = dataset.to_pandas()
    print(f"Loaded {len(df)} samples")
    return df


def prepare_datasets(df, tokenizer):
    """Clean text, tokenize, split into train/val, and format for PyTorch."""
    # Clean text
    df['cleaned_text'] = df['comment_text'].apply(clean_text)
    print(df[['comment_text', 'cleaned_text']].head(5))

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples['cleaned_text'],
            padding='max_length',
            truncation=True,
            max_length=CONFIG['max_length'],
        )

    cleaned_dataset = Dataset.from_pandas(df[['cleaned_text'] + TOXICITY_LABELS])

    print("Tokenizing dataset...")
    tokenized = cleaned_dataset.map(tokenize_fn, batched=True)

    # Train/Val split
    split = tokenized.train_test_split(test_size=CONFIG['test_size'], seed=SEED)
    print(f"\nDataset structure: {split}")

    # Convert label columns into a single 'labels' vector
    def add_labels(examples):
        labels = []
        for i in range(len(examples[TOXICITY_LABELS[0]])):
            label_vec = [float(examples[col][i]) for col in TOXICITY_LABELS]
            labels.append(label_vec)
        examples['labels'] = labels
        return examples

    split = split.map(add_labels, batched=True)
    split.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    train_dataset = split['train']
    val_dataset = split['test']

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    return train_dataset, val_dataset


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_metrics(eval_pred):
    """Compute F1 and ROC-AUC for multi-label classification."""
    logits, labels = eval_pred
    labels = labels.astype(int)
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    predictions = (probs > 0.5).astype(float)
    return {
        'f1_micro': f1_score(labels, predictions, average='micro'),
        'f1_macro': f1_score(labels, predictions, average='macro'),
        'roc_auc': roc_auc_score(labels, probs, average='micro'),
    }


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_model(train_dataset, val_dataset, tokenizer):
    """Initialize and train the classification model. Returns (trainer, model)."""
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=len(TOXICITY_LABELS),
        problem_type="multi_label_classification",
    )

    training_args = TrainingArguments(
        output_dir=CONFIG['output_dir'],
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=CONFIG['learning_rate'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        num_train_epochs=CONFIG['num_epochs'],
        weight_decay=CONFIG['weight_decay'],
        warmup_ratio=CONFIG['warmup_ratio'],
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        fp16=torch.cuda.is_available(),
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)
    trainer.train()

    return trainer, model


# ──────────────────────────────────────────────
# Evaluation & Visualization
# ──────────────────────────────────────────────

def evaluate_model(trainer, val_dataset):
    """Evaluate model and display classification report + confusion matrices."""
    # Validation metrics
    metrics = trainer.evaluate()
    print("\nValidation Metrics:", metrics)

    # Predictions on validation set
    predictions = trainer.predict(val_dataset)
    probs = 1 / (1 + np.exp(-predictions.predictions))
    y_pred = (probs > 0.5).astype(int)
    y_true = predictions.label_ids

    # Classification report
    print("\n--- Classification Report per Label ---")
    print(classification_report(y_true, y_pred, target_names=TOXICITY_LABELS))

    # Confusion matrices
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for i, (matrix, label) in enumerate(zip(mcm, TOXICITY_LABELS)):
        sns.heatmap(
            matrix, annot=True, fmt='d', ax=axes[i], cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
        )
        axes[i].set_title(f'Confusion Matrix: {label}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# Save & Load Model
# ──────────────────────────────────────────────

def save_model(model, tokenizer):
    """Save trained model and tokenizer to disk for later reuse."""
    save_path = CONFIG['model_save_path']
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n✓ Model and tokenizer saved to: {os.path.abspath(save_path)}")


def load_saved_model():
    """Load a previously saved model and tokenizer from disk."""
    save_path = CONFIG['model_save_path']
    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"No saved model found at '{save_path}'. Run training first."
        )
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    model = AutoModelForSequenceClassification.from_pretrained(save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Using device: {device}")
    print(f"✓ Model loaded from: {os.path.abspath(save_path)}")
    return model, tokenizer


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────

def predict_comment(text, model, tokenizer):
    """Predict toxicity probabilities for a single comment."""
    cleaned = clean_text(text)
    inputs = tokenizer(
        cleaned, return_tensors="pt",
        truncation=True, max_length=CONFIG['max_length'],
    ).to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits).cpu().numpy()[0]

    return {label: float(prob) for label, prob in zip(TOXICITY_LABELS, probs)}


def format_prediction(prediction):
    """Format prediction probabilities for console/table output."""
    return {label: f"{prob:.2%}" for label, prob in prediction.items()}


def classify_transcript_text(text, model, tokenizer, threshold=0.5):
    """Classify each spoken segment and compute overall toxic percentage."""
    segments = split_speech_segments(text)
    rows = []

    for index, segment in enumerate(segments, start=1):
        prediction = predict_comment(segment, model, tokenizer)
        max_label = max(prediction, key=prediction.get)
        max_probability = prediction[max_label]
        is_toxic = any(prob >= threshold for prob in prediction.values())
        rows.append({
            "index": index,
            "text": segment,
            "is_toxic": is_toxic,
            "max_label": max_label,
            "max_probability": max_probability,
            "probabilities": prediction,
        })

    toxic_count = sum(1 for row in rows if row["is_toxic"])
    total_count = len(rows)
    toxicity_ratio = toxic_count / total_count if total_count else 0.0

    return {
        "total_segments": total_count,
        "toxic_segments": toxic_count,
        "toxicity_ratio": toxicity_ratio,
        "toxicity_percent": toxicity_ratio * 100,
        "threshold": threshold,
        "segments": rows,
    }


def print_classification_result(result):
    """Print a compact classification report for a transcript."""
    print("\n--- Toxicity Classification ---")
    print(f"Segments: {result['total_segments']}")
    print(f"Toxic segments: {result['toxic_segments']}")
    print(f"Toxicity ratio: {result['toxicity_percent']:.2f}%")
    print(f"Threshold: {result['threshold']:.2f}")

    if not result["segments"]:
        print("No text segments found.")
        return

    print("\n--- Segment Details ---")
    for row in result["segments"]:
        status = "TOXIC" if row["is_toxic"] else "NON_TOXIC"
        print(
            f"{row['index']:02d}. [{status}] "
            f"{row['max_label']}={row['max_probability']:.2%} | {row['text']}"
        )


def resolve_input_file(file_arg):
    """Resolve an input text file from cwd, repo root, or transcriber project root."""
    raw_path = Path(file_arg)
    if raw_path.is_absolute():
        return raw_path

    candidates = [
        Path.cwd() / raw_path,
        PROJECT_ROOT / raw_path,
        TRANSCRIBER_PROJECT_ROOT / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def read_prediction_text(args):
    """Read predict input and fail before model loading if text is unavailable."""
    if args.text and args.file:
        raise PredictionInputError("Use either --text or --file, not both.")

    if args.text is not None:
        text = args.text.strip()
        if not text:
            raise PredictionInputError("Input text from --text is empty.")
        return text, "CLI text"

    if args.file:
        file_path = resolve_input_file(args.file)
        source = f"custom file: {file_path}"
    else:
        file_path = DEFAULT_TRANSCRIPTION_FILE
        source = f"default transcriber output: {file_path}"

    if not file_path.exists():
        raise PredictionInputError(
            "Transcript file not found.\n"
            f"Expected: {file_path}\n"
            "Run the transcriber first, for example:\n"
            "  cd recorder_and_transcriber\n"
            "  python src\\transcriber.py\n"
            "Then run predict again:\n"
            "  python ..\\Text_classification.py predict"
        )

    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        raise PredictionInputError(
            "Transcript file is empty.\n"
            f"File: {file_path}\n"
            "Run the transcriber again and make sure it prints a non-empty transcription."
        )

    return text, source


def run_test_samples(model, tokenizer):
    """Run prediction on a set of multilingual test samples."""
    test_samples = [
        # --- Vietnamese ---
        "Khóa học này thực sự rất bổ ích, cảm ơn tác giả!",
        "Bạn giải thích vấn đề này rất dễ hiểu.",
        "Đồ ngu, có thế mà cũng không biết làm, cút đi!",  # Toxic
        "Tôi không đồng ý với quan điểm của bạn nhưng tôn trọng nó.",

        # --- German ---
        "Guten Morgen, wie geht es dir heute?",
        "Das ist eine sehr interessante Perspektive, danke fürs Teilen.",
        "Du bist so ein Idiot, verschwinde von hier!",  # Toxic
        "Ich liebe die deutsche Kultur và das Essen.",

        # --- Japanese ---
        "こんにちは、お元気ですか？ (Chào bạn, bạn khỏe không?)",
        "この本はとても勉強になりました。 (Cuốn sách này rất có ích)",
        "お前は本当に最低だな、死ね！ (Mày thật tồi tệ, chết đi!)",  # Toxic
        "富士山はとても綺麗ですね。 (Núi Phú Sĩ đẹp quá nhỉ)",

        # --- English ---
        "I really appreciate your hard work on this project.",
        "Could you please help me with this task?",
        "Shut up! Your opinion doesn't matter here, you loser.",  # Toxic
        "The library is a great place to focus and study.",
        "Science is the key to understanding our universe.",
        "I hope you have a wonderful day ahead!",
        "You are a complete failure and everyone hates you.",  # Toxic
        "It is important to stay positive in difficult times.",
    ]

    results_list = []
    for text in test_samples:
        pred = format_prediction(predict_comment(text, model, tokenizer))
        results_list.append({"Comment": text, **pred})

    results_df = pd.DataFrame(results_list)
    print("\n--- Test Sample Predictions ---")
    print(results_df.to_string(index=False))
    return results_df


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def train_and_save(run_samples=False):
    """Train the model once, evaluate it, and save it for future prediction."""
    df = load_data()
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    train_dataset, val_dataset = prepare_datasets(df, tokenizer)
    trainer, model = train_model(train_dataset, val_dataset, tokenizer)
    evaluate_model(trainer, val_dataset)
    save_model(model, tokenizer)

    if run_samples:
        run_test_samples(model, tokenizer)


def predict_from_args(args):
    """Load the saved model and classify text from CLI args."""
    text, source = read_prediction_text(args)

    model, tokenizer = load_saved_model()
    result = classify_transcript_text(
        text=text,
        model=model,
        tokenizer=tokenizer,
        threshold=args.threshold,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Prediction source: {source}")
        print_classification_result(result)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or reuse a saved toxic text classifier."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Legacy alias: train, save, then run test samples.",
    )

    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train and save the model.")
    train_parser.add_argument(
        "--test",
        action="store_true",
        help="Run built-in prediction samples after training.",
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="Load the saved model and classify text without retraining.",
    )
    predict_parser.add_argument("--text", help="Text/transcript to classify.")
    predict_parser.add_argument(
        "--file",
        help=(
            "Text file to classify. Defaults to "
            "recorder_and_transcriber/outputs/audio/final_output_transcription.txt"
        ),
    )
    predict_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="A segment is toxic if any toxicity label is >= this value.",
    )
    predict_parser.add_argument(
        "--json",
        action="store_true",
        help="Print full result as JSON.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    command = args.command or "train"

    try:
        if command == "train":
            run_samples = getattr(args, "test", False)
            train_and_save(run_samples=run_samples)
        elif command == "predict":
            predict_from_args(args)
        else:
            raise ValueError(f"Unknown command: {command}")
    except PredictionInputError as exc:
        print(f"Prediction input error:\n{exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
