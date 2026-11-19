import argparse
import csv
import math
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


def build_confusion_matrices(rows):
    if not rows:
        raise ValueError("No classification rows to analyze")

    classes = sorted({label for real, predicted in rows for label in (real, predicted)})
    return {
        label: confusion_matrix(
            [real == label for real, _ in rows],
            [predicted == label for _, predicted in rows],
            labels=[False, True],
        )
        for label in classes
    }


def display_confusion_matrices(matrices):
    if not matrices:
        raise ValueError("No confusion matrices to display")

    columns = min(3, len(matrices))
    rows = math.ceil(len(matrices) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)

    for axis, (label, matrix) in zip(axes.flat, matrices.items()):
        axis.imshow(matrix, cmap="Blues")
        axis.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            xlabel="Predicted",
            ylabel="Actual",
            title=label,
        )
        threshold = matrix.max() / 2
        for actual in range(2):
            for predicted in range(2):
                value = matrix[actual, predicted]
                axis.text(
                    predicted,
                    actual,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                )

    for axis in axes.flat[len(matrices):]:
        axis.set_visible(False)

    figure.tight_layout()
    plt.show()
    return figure, axes


def main():
    parser = argparse.ArgumentParser(
        description="Display one-vs-rest confusion matrices from a classification CSV."
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

    display_confusion_matrices(build_confusion_matrices(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())