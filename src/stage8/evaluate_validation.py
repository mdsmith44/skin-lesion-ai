"""Evaluate the adapted Stage 8 VLM on the HAM10000 validation split.

Evaluation uses free generation from image + instruction only. The target
response is never provided to the model.

This script intentionally evaluates the validation split only. The held-out
test split remains isolated until the Stage 8 model and evaluation procedure
are locked.
"""

from collections import Counter
from pathlib import Path
import csv
import time

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.stage8.inference import (
    generate_prediction,
    load_adapted_model,
    load_processor,
)
from src.stage8.dataset import (
    CATEGORIES,
    HAM10000MultimodalDataset,
)


PROJECT_ROOT = Path("/workspace/skin-lesion-ai")

CHECKPOINT = Path(
    "/workspace/storage/skin-lesion-ai/outputs/stage8/"
    "lora-b64-fixed/LOWEST_VAL"
)

OUTPUT_DIR = Path(
    "/workspace/storage/skin-lesion-ai/outputs/stage8/"
    "validation-evaluation"
)

PREDICTIONS_CSV = OUTPUT_DIR / "predictions.csv"
METRICS_TXT = OUTPUT_DIR / "metrics.txt"


def normalize_prediction(text: str) -> str | None:
    """Accept only an exact seven-class response after basic normalization."""
    normalized = text.strip().lower()

    if normalized in CATEGORIES:
        return normalized

    return None


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = HAM10000MultimodalDataset(
        project_root=PROJECT_ROOT,
        split="val",
    )

    print(f"Checkpoint: {CHECKPOINT.resolve()}")
    print(f"Validation images: {len(dataset)}")
    print()

    processor = load_processor()
    model = load_adapted_model(CHECKPOINT)

    # Warm up the inference path before timing.
    _ = generate_prediction(
        model,
        processor,
        dataset[0],
    )

    records = []
    true_labels = []
    valid_true_labels = []
    valid_predictions = []

    start = time.perf_counter()

    for i in range(len(dataset)):
        sample = dataset[i]

        raw_prediction = generate_prediction(
            model,
            processor,
            sample,
        )

        prediction = normalize_prediction(raw_prediction)
        valid = prediction is not None

        true_labels.append(sample.class_abbreviation)

        if valid:
            valid_true_labels.append(
                sample.class_abbreviation
            )
            valid_predictions.append(prediction)

        records.append(
            {
                "image_id": sample.image_id,
                "lesion_id": sample.lesion_id,
                "true_label": sample.class_abbreviation,
                "raw_prediction": raw_prediction,
                "prediction": prediction or "",
                "valid_prediction": valid,
                "correct": (
                    valid
                    and prediction
                    == sample.class_abbreviation
                ),
            }
        )

        completed = i + 1

        if completed % 100 == 0 or completed == len(dataset):
            elapsed = time.perf_counter() - start
            rate = completed / elapsed

            print(
                f"{completed:4d}/{len(dataset)}  "
                f"{rate:.2f} images/s"
            )

    elapsed = time.perf_counter() - start

    with PREDICTIONS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=records[0].keys(),
        )
        writer.writeheader()
        writer.writerows(records)

    invalid_count = sum(
        not record["valid_prediction"]
        for record in records
    )

    invalid_rate = invalid_count / len(records)

    # Invalid generations count as incorrect for overall accuracy.
    correct_count = sum(
        record["correct"]
        for record in records
    )
    overall_accuracy = correct_count / len(records)

    prediction_counts = Counter(
        record["prediction"]
        if record["valid_prediction"]
        else "<invalid>"
        for record in records
    )

    lines = []

    lines.append("Stage 8 validation evaluation")
    lines.append("=" * 72)
    lines.append(f"Checkpoint: {CHECKPOINT.resolve()}")
    lines.append(f"Validation images: {len(dataset)}")
    lines.append(f"Elapsed seconds: {elapsed:.2f}")
    lines.append(
        f"Images/second: {len(dataset) / elapsed:.2f}"
    )
    lines.append("")

    lines.append("Generation validity")
    lines.append("-" * 72)
    lines.append(f"Invalid outputs: {invalid_count}")
    lines.append(f"Invalid rate: {invalid_rate:.4f}")
    lines.append("")

    lines.append("Overall classification")
    lines.append("-" * 72)
    lines.append(
        f"Accuracy: {overall_accuracy:.4f}"
    )

    # The remaining sklearn metrics require valid class predictions.
    # If there are invalid generations, these metrics are reported over
    # valid predictions only and clearly identified as such.
    if valid_predictions:
        valid_accuracy = accuracy_score(
            valid_true_labels,
            valid_predictions,
        )
        balanced_accuracy = balanced_accuracy_score(
            valid_true_labels,
            valid_predictions,
        )
        macro_precision = precision_score(
            valid_true_labels,
            valid_predictions,
            labels=list(CATEGORIES),
            average="macro",
            zero_division=0,
        )
        macro_recall = recall_score(
            valid_true_labels,
            valid_predictions,
            labels=list(CATEGORIES),
            average="macro",
            zero_division=0,
        )
        macro_f1 = f1_score(
            valid_true_labels,
            valid_predictions,
            labels=list(CATEGORIES),
            average="macro",
            zero_division=0,
        )

        lines.append(
            f"Valid-only accuracy: {valid_accuracy:.4f}"
        )
        lines.append(
            f"Balanced accuracy (valid only): "
            f"{balanced_accuracy:.4f}"
        )
        lines.append(
            f"Macro precision (valid only): "
            f"{macro_precision:.4f}"
        )
        lines.append(
            f"Macro recall (valid only): "
            f"{macro_recall:.4f}"
        )
        lines.append(
            f"Macro F1 (valid only): {macro_f1:.4f}"
        )

        lines.append("")
        lines.append("Prediction counts")
        lines.append("-" * 72)

        for label in CATEGORIES:
            lines.append(
                f"{label:5s}: {prediction_counts[label]}"
            )

        if invalid_count:
            lines.append(
                f"{'<invalid>':9s}: "
                f"{prediction_counts['<invalid>']}"
            )

        lines.append("")
        lines.append("Per-class report")
        lines.append("-" * 72)

        report = classification_report(
            valid_true_labels,
            valid_predictions,
            labels=list(CATEGORIES),
            target_names=list(CATEGORIES),
            digits=4,
            zero_division=0,
        )

        lines.append(report)

        lines.append("Confusion matrix")
        lines.append(
            "Rows = true class; columns = predicted class"
        )
        lines.append(
            "Class order: " + ", ".join(CATEGORIES)
        )
        lines.append("-" * 72)

        matrix = confusion_matrix(
            valid_true_labels,
            valid_predictions,
            labels=list(CATEGORIES),
        )

        header = "true\\pred " + " ".join(
            f"{label:>6s}"
            for label in CATEGORIES
        )
        lines.append(header)

        for label, row in zip(CATEGORIES, matrix):
            values = " ".join(
                f"{value:6d}"
                for value in row
            )
            lines.append(
                f"{label:>9s} {values}"
            )

    else:
        lines.append(
            "No valid predictions were generated; "
            "classification metrics unavailable."
        )

    metrics_text = "\n".join(lines)

    METRICS_TXT.write_text(
        metrics_text,
        encoding="utf-8",
    )

    print()
    print(metrics_text)
    print()
    print(f"Predictions saved to: {PREDICTIONS_CSV}")
    print(f"Metrics saved to:     {METRICS_TXT}")


if __name__ == "__main__":
    main()
