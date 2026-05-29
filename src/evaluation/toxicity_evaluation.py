import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from transformers import Trainer, TrainingArguments

# Add parent directory to path to import config and models
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import ANALYSIS_OUTPUT, ensure_output_dirs
from models.text_classification import (
    CONFIG,
    TOXICITY_LABELS,
    compute_metrics,
    load_data,
    load_saved_model,
    prepare_datasets,
)


def evaluate_classifier(output_path, show_plot, metrics_json_path):
    ensure_output_dirs()

    model, tokenizer = load_saved_model()
    df = load_data()
    _, test_dataset = prepare_datasets(df, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(ANALYSIS_OUTPUT),
        per_device_eval_batch_size=CONFIG["batch_size"],
        report_to=[],
        do_train=False,
        do_eval=True,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    print("Evaluation Metrics:", metrics)

    predictions = trainer.predict(test_dataset)
    probs = 1 / (1 + np.exp(-predictions.predictions))
    y_pred = (probs > 0.5).astype(int)
    y_true = predictions.label_ids

    print("\n--- Classification Report per Label ---")
    report_text = classification_report(y_true, y_pred, target_names=TOXICITY_LABELS)
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=TOXICITY_LABELS,
        output_dict=True,
    )
    print(report_text)

    mcm = multilabel_confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for i, (matrix, label) in enumerate(zip(mcm, TOXICITY_LABELS)):
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            ax=axes[i],
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
        )
        axes[i].set_title(f"Confusion Matrix: {label}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    plt.tight_layout()
    fig.savefig(output_path)
    print(f"Confusion matrices saved to: {output_path}")

    metrics_payload = {
        "metrics": metrics,
        "classification_report": report_dict,
        "labels": TOXICITY_LABELS,
        "threshold": 0.5,
        "split": "test",
        "num_samples": len(y_true),
    }

    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Metrics JSON saved to: {metrics_json_path}")

    if show_plot:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the toxicity classifier and save confusion matrices."
    )
    parser.add_argument(
        "--output",
        default=str(ANALYSIS_OUTPUT / "toxicity_confusion_matrix.png"),
        help="Output path for the confusion matrix image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the confusion matrix plot window.",
    )
    parser.add_argument(
        "--metrics-json",
        default=str(ANALYSIS_OUTPUT / "toxicity_metrics.json"),
        help="Output path for evaluation metrics JSON.",
    )

    args = parser.parse_args()
    evaluate_classifier(
        Path(args.output),
        args.show,
        Path(args.metrics_json),
    )


if __name__ == "__main__":
    main()
