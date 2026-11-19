import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def load_classifications(csv_path):
    rows = []
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")
        if not reader.fieldnames:
            raise ValueError("CSV is empty or has no header")

        required_fields = {"filename", "real", "predicted"}
        missing_fields = required_fields - set(reader.fieldnames)
        if missing_fields:
            raise ValueError(
                "CSV is missing required columns: " + ", ".join(sorted(missing_fields))
            )
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("CSV header contains duplicate columns")

        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"line {line_number} has more fields than the header")
            real = (row.get("real") or "").strip()
            predicted = (row.get("predicted") or "").strip()
            if not real or not predicted:
                raise ValueError(
                    f"line {line_number} must have non-empty real and predicted labels"
                )
            rows.append((real, predicted))

    if not rows:
        raise ValueError("CSV has a header but contains no classification rows")
    return rows


def build_confusion_matrix(rows):
    if not rows:
        raise ValueError("No classification rows to analyze")

    classes = sorted({label for real, predicted in rows for label in (real, predicted)})
    matrix = confusion_matrix(
        [real for real, _ in rows],
        [predicted for _, predicted in rows],
        labels=classes,
    )
    return classes, matrix


def calculate_metrics(matrix, classes=None):
    """Calculate per-class, micro and macro metrics; undefined ratios are zero."""
    total = int(matrix.sum())
    if not total:
        raise ValueError("No classifications to analyze")

    if classes is None:
        classes = list(range(len(matrix)))
    if len(classes) != len(matrix):
        raise ValueError("The number of class labels must match the matrix dimensions")

    counts = []
    for index in range(len(matrix)):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum()) - tp
        fn = int(matrix[index, :].sum()) - tp
        tn = total - tp - fp - fn
        counts.append((tp, fp, fn, tn))

    def metrics(tp, fp, fn, tn):
        def ratio(numerator, denominator):
            return numerator / denominator if denominator else 0.0

        return {
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
            "accuracy": ratio(tp + tn, tp + fp + fn + tn),
            "specificity": ratio(tn, tn + fp),
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
        }

    per_class = {
        label: metrics(*count) for label, count in zip(classes, counts)
    }
    micro = metrics(*(sum(values) for values in zip(*counts)))
    correct = sum(matrix[index, index] for index in range(len(matrix)))
    micro["accuracy"] = correct / total
    macro = {name: sum(item[name] for item in per_class.values()) / len(per_class)
             for name in micro}
    return {"per_class": per_class, "micro": micro, "macro": macro}


def display_metrics(metrics):
    metric_names = ("precision", "recall", "f1", "accuracy", "specificity")
    print("Per-label (one-vs-rest):")
    print(f"{'Label':<20} " + " ".join(f"{name:>12}" for name in metric_names))
    for label, values in metrics["per_class"].items():
        print(
            f"{str(label):<20} "
            + " ".join(f"{values[name]:>12.4f}" for name in metric_names)
        )

    for average in ("micro", "macro"):
        print(f"{average.capitalize()} (one-vs-rest):")
        for name in metric_names:
            print(f"  {name}: {metrics[average][name]:.4f}")


def display_confusion_matrix(classes, matrix):
    if not classes:
        raise ValueError("No confusion matrix to display")

    figure, axis = plt.subplots(figsize=(max(5, len(classes)), max(5, len(classes))))
    axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=range(len(classes)),
        yticks=range(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted",
        ylabel="Actual",
        title="Confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2
    for actual in range(len(classes)):
        for predicted in range(len(classes)):
            value = matrix[actual, predicted]
            axis.text(
                predicted,
                actual,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    figure.tight_layout()
    plt.show()
    return figure, axis


def main():
    parser = argparse.ArgumentParser(
        description="Display a confusion matrix from a classification CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=Path("calssification.csv"),
        help="semicolon-delimited input CSV (default: calssification.csv)",
    )
    args = parser.parse_args()

    try:
        rows = load_classifications(args.csv_path)
    except (OSError, csv.Error, ValueError) as error:
        parser.error(str(error))

    classes, matrix = build_confusion_matrix(rows)
    display_metrics(calculate_metrics(matrix, classes))
    display_confusion_matrix(classes, matrix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())