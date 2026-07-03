"""
Evaluate time-derivative contrastive embeddings by nearest class center.

The script computes one embedding center from the training set for each broad
clinical group (healthy, neuro, ortho). It then measures every test embedding's
distance to those centers, predicts the nearest center, and saves a confusion
matrix plus per-sample distances.
"""

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import normalize


DEFAULT_EMBEDDINGS_DIR = Path("contrastive_time_derivative_encoder_outputs")
DEFAULT_OUTPUT_DIR = Path("contrastive_time_derivative_embedding_center_eval")
DEFAULT_CLASSES = ["healthy", "neuro", "ortho"]


def embedding_columns(df):
    columns = [column for column in df.columns if column.startswith("embedding_")]
    if not columns:
        raise ValueError("No embedding columns found.")
    return columns


def load_embedding_table(csv_path):
    df = pd.read_csv(csv_path)
    if "category" not in df.columns:
        raise ValueError(f"{csv_path} must contain a category column.")
    columns = embedding_columns(df)
    return df, columns


def prepare_embeddings(df, columns, normalize_embeddings):
    embeddings = df[columns].to_numpy(dtype=np.float32)
    if normalize_embeddings:
        embeddings = normalize(embeddings, norm="l2")
    return embeddings


def compute_centers(train_df, train_embeddings, classes):
    centers = {}
    for class_name in classes:
        mask = train_df["category"] == class_name
        if not mask.any():
            raise ValueError(f"No training samples found for class '{class_name}'.")
        centers[class_name] = train_embeddings[mask.to_numpy()].mean(axis=0)
    return centers


def distance_to_centers(embeddings, centers, metric):
    class_names = list(centers.keys())
    center_matrix = np.stack([centers[class_name] for class_name in class_names], axis=0)

    if metric == "euclidean":
        distances = np.linalg.norm(embeddings[:, None, :] - center_matrix[None, :, :], axis=2)
    elif metric == "cosine":
        embeddings_norm = normalize(embeddings, norm="l2")
        centers_norm = normalize(center_matrix, norm="l2")
        distances = 1.0 - np.matmul(embeddings_norm, centers_norm.T)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

    return class_names, distances


def save_centers(output_path, centers):
    rows = []
    for class_name, center in centers.items():
        row = {"category": class_name}
        for dim_idx, value in enumerate(center):
            row[f"center_{dim_idx:03d}"] = value
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def plot_confusion_matrix(matrix, classes, output_path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title(title)

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            color = "white" if value > threshold else "black"
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate derivative contrastive embeddings with nearest class centers."
    )
    parser.add_argument(
        "--embeddings-dir",
        default=str(DEFAULT_EMBEDDINGS_DIR),
        help="Directory containing train_embeddings.csv and test_embeddings.csv",
    )
    parser.add_argument("--train-csv", default=None, help="Optional train embeddings CSV path")
    parser.add_argument("--test-csv", default=None, help="Optional test embeddings CSV path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for evaluation outputs")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Class names to evaluate and order in the confusion matrix",
    )
    parser.add_argument(
        "--metric",
        choices=["euclidean", "cosine"],
        default="euclidean",
        help="Distance metric from test embeddings to training centers",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not L2-normalize embeddings before computing centers.",
    )
    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    train_csv = Path(args.train_csv) if args.train_csv else embeddings_dir / "train_embeddings.csv"
    test_csv = Path(args.test_csv) if args.test_csv else embeddings_dir / "test_embeddings.csv"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, train_columns = load_embedding_table(train_csv)
    test_df, test_columns = load_embedding_table(test_csv)
    if train_columns != test_columns:
        raise ValueError("Train and test embedding columns do not match.")

    normalize_embeddings = not args.no_normalize
    train_embeddings = prepare_embeddings(train_df, train_columns, normalize_embeddings)
    test_embeddings = prepare_embeddings(test_df, test_columns, normalize_embeddings)

    centers = compute_centers(train_df, train_embeddings, args.classes)
    class_names, distances = distance_to_centers(test_embeddings, centers, args.metric)
    predicted_indices = distances.argmin(axis=1)
    predictions = [class_names[index] for index in predicted_indices]

    distance_columns = {
        f"distance_to_{class_name}": distances[:, class_idx]
        for class_idx, class_name in enumerate(class_names)
    }
    prediction_df = test_df[
        [
            column
            for column in [
                "split",
                "file_path",
                "category",
                "disease_group",
                "subgroup",
                "subject_name",
                "trial_name",
            ]
            if column in test_df.columns
        ]
    ].copy()
    for column, values in distance_columns.items():
        prediction_df[column] = values
    prediction_df["predicted_category"] = predictions
    prediction_df["correct"] = prediction_df["category"] == prediction_df["predicted_category"]

    matrix = confusion_matrix(
        prediction_df["category"],
        prediction_df["predicted_category"],
        labels=args.classes,
    )
    accuracy = accuracy_score(prediction_df["category"], prediction_df["predicted_category"])
    report = classification_report(
        prediction_df["category"],
        prediction_df["predicted_category"],
        labels=args.classes,
        zero_division=0,
    )

    save_centers(output_dir / "train_embedding_centers.csv", centers)
    prediction_df.to_csv(output_dir / "test_center_distances_predictions.csv", index=False)
    pd.DataFrame(matrix, index=args.classes, columns=args.classes).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    plot_confusion_matrix(
        matrix=matrix,
        classes=args.classes,
        output_path=output_dir / "confusion_matrix.png",
        title=f"Nearest-center embedding classification ({args.metric})",
    )
    (output_dir / "classification_report.txt").write_text(
        f"Accuracy: {accuracy:.4f}\n\n{report}",
        encoding="utf-8",
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(report)
    print(f"Saved class centers: {output_dir / 'train_embedding_centers.csv'}")
    print(f"Saved test distances and predictions: {output_dir / 'test_center_distances_predictions.csv'}")
    print(f"Saved confusion matrix: {output_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
