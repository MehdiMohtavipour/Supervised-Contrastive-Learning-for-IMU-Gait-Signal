"""
Compare a time-derivative embedding biomarker with handcrafted gait features.

The derivative contrastive encoder produces a vector embedding for each trial.
This script computes training centroids in embedding space for healthy, neuro,
and ortho, converts each trial to one scalar biomarker using:
distance_to_healthy + distance_to_neuro - distance_to_ortho,
then plots that biomarker beside handcrafted feature distributions. Plots are
grouped by broad disease group.
"""

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_EMBEDDINGS_CSV = Path(
    "contrastive_time_derivative_encoder_outputs/all_embeddings.csv"
)
DEFAULT_FEATURES_DIR = Path("final_extracted_features")
DEFAULT_OUTPUT_DIR = Path("time_derivative_embedding_handcrafted_comparison_plots")
DEFAULT_GROUP_ORDER = ["healthy", "neuro", "ortho"]
DEFAULT_CENTROID_CLASSES = ["healthy", "neuro", "ortho"]
DEFAULT_METRICS = [
    "avg_stride_time",
    "max_stride_time",
    "std_stride_time",
    "u_turn_time",
    "gait_cadence",
    #"avg_acceleration_peak",
    #"max_acceleration_peak",
    #"std_acceleration_peak",
    #"avg_angular_velocity_peak",
    #"max_angular_velocity_peak",
    #"std_angular_velocity_peak",
    #"avg_jerk",
    #"max_jerk",
    #"std_jerk",
    #"avg_angular_acceleration",
    #"max_angular_acceleration",
    #"std_angular_acceleration",
]


def embedding_columns(df):
    columns = [column for column in df.columns if column.startswith("embedding_")]
    if not columns:
        raise ValueError("No embedding columns found in embeddings CSV.")
    return columns


def l2_normalize(values):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def compute_distances_to_centroid(embeddings, centroid, metric):
    if metric == "euclidean":
        return np.linalg.norm(embeddings - centroid.reshape(1, -1), axis=1)

    if metric == "cosine":
        embeddings_norm = l2_normalize(embeddings)
        centroid_norm = centroid / max(np.linalg.norm(centroid), 1e-12)
        return 1.0 - np.matmul(embeddings_norm, centroid_norm)

    raise ValueError(f"Unknown metric: {metric}")


def compute_train_centroids(df, embeddings, classes):
    centroids = {}
    if "split" in df.columns:
        train_mask = df["split"].astype(str) == "train"
    else:
        train_mask = pd.Series(True, index=df.index)

    for class_name in classes:
        class_mask = train_mask & (df["category"].astype(str) == class_name)
        if not class_mask.any():
            raise ValueError(
                f"No train samples found for centroid class '{class_name}'."
            )
        centroids[class_name] = embeddings[class_mask.to_numpy()].mean(axis=0)
    return centroids


def load_embedding_centroid_distance_biomarker(
    embeddings_csv,
    centroid_classes=None,
    metric="euclidean",
    normalize_before_centroid=True,
):
    if centroid_classes is None:
        centroid_classes = DEFAULT_CENTROID_CLASSES

    df = pd.read_csv(embeddings_csv)
    columns = embedding_columns(df)
    embeddings = df[columns].to_numpy(dtype=np.float32)
    centroid_embeddings = l2_normalize(embeddings) if normalize_before_centroid else embeddings
    centroids = compute_train_centroids(df, centroid_embeddings, centroid_classes)
    distance_embeddings = l2_normalize(embeddings) if normalize_before_centroid else embeddings

    distance_columns = []
    for class_name, centroid in centroids.items():
        column = f"distance_to_{class_name}_centroid"
        df[column] = compute_distances_to_centroid(
            distance_embeddings,
            centroid,
            metric=metric,
        )
        distance_columns.append(column)
    required_columns = [
        "distance_to_healthy_centroid",
        "distance_to_neuro_centroid",
        "distance_to_ortho_centroid",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "The healthy+neuro-ortho biomarker requires centroid classes "
            f"healthy, neuro, and ortho. Missing: {', '.join(missing_columns)}"
        )
    df["distance_healthy_plus_neuro_minus_ortho"] = (
        df["distance_to_healthy_centroid"]
        + df["distance_to_neuro_centroid"]
        - df["distance_to_ortho_centroid"]
    )

    df["Disease_Group"] = df["disease_group"].astype(str)
    df["Clinical_Category"] = np.where(
        df["Disease_Group"] == "healthy",
        "healthy",
        df["Disease_Group"] + "_" + df["subgroup"].astype(str),
    )
    return df, centroids, distance_columns


def feature_key_from_path(path):
    name = path.stem
    if name.startswith("features_"):
        return name[len("features_") :]
    return name


def feature_paths(features_dir, pattern):
    paths = sorted(Path(features_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No handcrafted feature CSVs matched {features_dir / pattern}")
    return paths


def values_by_group(df, value_column, group_column, group_order):
    values = []
    labels = []
    for group in group_order:
        if group not in set(df[group_column].dropna().astype(str)):
            continue
        group_values = pd.to_numeric(
            df.loc[df[group_column].astype(str) == group, value_column],
            errors="coerce",
        ).dropna()
        if len(group_values) == 0:
            continue
        values.append(group_values.to_numpy(dtype=float))
        labels.append(group)
    return labels, values


DISPLAY_METRIC_LABELS = {
    "avg_stride_time": "Avg stride time",
    "max_stride_time": "Max stride time",
    "std_stride_time": "Std stride time",
    "u_turn_time": "U-turn time",
    "gait_cadence": "Gait cadence",
    "avg_acceleration_peak": "Avg peak",
    "max_acceleration_peak": "Max peak",
    "std_acceleration_peak": "Std peak",
    "avg_angular_velocity_peak": "Avg peak",
    "max_angular_velocity_peak": "Max peak",
    "std_angular_velocity_peak": "Std peak",
    "avg_jerk": "Avg change",
    "max_jerk": "Max change",
    "std_jerk": "Std change",
    "avg_angular_acceleration": "Avg rate change",
    "max_angular_acceleration": "Max rate change",
    "std_angular_acceleration": "Std rate change",
}


def display_metric_label(metric):
    return DISPLAY_METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def display_metric_unit(metric):
    if metric == "gait_cadence":
        return "Value (1/sec)"
    return "Value (sec)"


def draw_box_with_points(ax, df, value_column, group_column, group_order, title, ylabel="Value"):
    labels, values = values_by_group(df, value_column, group_column, group_order)
    if not values:
        ax.set_visible(False)
        return

    positions = np.arange(1, len(values) + 1)
    ax.boxplot(
        values,
        positions=positions,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "#d9e8f5", "edgecolor": "#2f4f6f"},
        medianprops={"color": "#a23b3b", "linewidth": 1.4},
        whiskerprops={"color": "#2f4f6f"},
        capprops={"color": "#2f4f6f"},
    )

    rng = np.random.default_rng(42)
    for position, group_values in zip(positions, values):
        if len(group_values) > 500:
            plot_values = rng.choice(group_values, size=500, replace=False)
        else:
            plot_values = group_values
        jitter = rng.uniform(-0.16, 0.16, size=len(plot_values))
        ax.scatter(
            np.full(len(plot_values), position) + jitter,
            plot_values,
            s=12,
            alpha=0.35,
            color="#1f2937",
            linewidth=0,
        )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(group_column)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.28)
    ax.tick_params(axis="x", rotation=25)


def plot_embedding_only(embedding_df, output_path, group_column, group_order):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    draw_box_with_points(
        ax=ax,
        df=embedding_df,
        value_column="distance_healthy_plus_neuro_minus_ortho",
        group_column=group_column,
        group_order=group_order,
        title="This work",
        ylabel="Biomarker Magnitude",
    )
    fig.suptitle("Time-derivative healthy+neuro-ortho biomarker by group", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_feature_comparison(
    embedding_df,
    feature_df,
    feature_key,
    metrics,
    output_path,
    group_column,
    group_order,
):
    available_metrics = [
        metric for metric in metrics if f"{feature_key}_{metric}" in feature_df.columns
    ]
    if not available_metrics:
        print(f"Skipping {feature_key}: none of the requested feature metrics were found.")
        return False

    plot_specs = [
        (
            "distance_healthy_plus_neuro_minus_ortho",
            embedding_df,
            "This work",
            "Biomarker Magnitude",
        )
    ]
    for metric in available_metrics:
        column = f"{feature_key}_{metric}"
        plot_specs.append((column, feature_df, display_metric_label(metric), display_metric_unit(metric)))

    ncols = 3
    nrows = int(np.ceil(len(plot_specs) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 4.2, nrows * 3.5),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, (column, df, title, ylabel) in zip(axes_flat, plot_specs):
        draw_box_with_points(
            ax=ax,
            df=df,
            value_column=column,
            group_column=group_column,
            group_order=group_order,
            title=title,
            ylabel=ylabel,
        )

    for ax in axes_flat[len(plot_specs) :]:
        ax.set_visible(False)

    fig.suptitle(
        "Multi-centroid embedding biomarker vs handcrafted features",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Compare derivative healthy+neuro-ortho embedding biomarker with handcrafted feature boxplots."
    )
    parser.add_argument(
        "--embeddings-csv",
        default=str(DEFAULT_EMBEDDINGS_CSV),
        help="Path to derivative all_embeddings.csv",
    )
    parser.add_argument(
        "--features-dir",
        default=str(DEFAULT_FEATURES_DIR),
        help="Directory containing handcrafted features_*.csv files",
    )
    parser.add_argument(
        "--feature-pattern",
        default="features_*_Magnitude.csv",
        help="Glob pattern for handcrafted feature CSVs",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for output plots")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Handcrafted metric suffixes to plot",
    )
    parser.add_argument(
        "--centroid-classes",
        nargs="+",
        default=DEFAULT_CENTROID_CLASSES,
        help="Category labels used to compute train centroids for the healthy+neuro-ortho biomarker",
    )
    parser.add_argument(
        "--metric",
        choices=["euclidean", "cosine"],
        default="euclidean",
        help="Distance metric from each embedding to each centroid",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not L2-normalize embeddings before computing centroid distances.",
    )
    parser.add_argument(
        "--group-column",
        default="Disease_Group",
        choices=["Disease_Group", "Clinical_Category"],
        help="Group column for boxplots",
    )
    parser.add_argument(
        "--group-order",
        nargs="+",
        default=DEFAULT_GROUP_ORDER,
        help="Group order for broad disease-group plots",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_df, centroids, distance_columns = load_embedding_centroid_distance_biomarker(
        embeddings_csv=Path(args.embeddings_csv),
        centroid_classes=args.centroid_classes,
        metric=args.metric,
        normalize_before_centroid=not args.no_normalize,
    )
    embedding_scalar_path = output_dir / "distance_healthy_plus_neuro_minus_ortho_by_trial.csv"
    metadata_columns = [
        column
        for column in [
            "split",
            "file_path",
            "category",
            "disease_group",
            "subgroup",
            "subject_name",
            "trial_name",
            "Disease_Group",
            "Clinical_Category",
            *distance_columns,
            "distance_healthy_plus_neuro_minus_ortho",
        ]
        if column in embedding_df.columns
    ]
    embedding_df[metadata_columns].to_csv(embedding_scalar_path, index=False)
    centroid_rows = []
    for class_name, centroid in centroids.items():
        centroid_rows.append(
            {
                "centroid_class": class_name,
                "metric": args.metric,
                "normalized_before_centroid": not args.no_normalize,
                **{
                    f"centroid_{idx:03d}": value
                    for idx, value in enumerate(centroid)
                },
            }
        )
    pd.DataFrame(centroid_rows).to_csv(output_dir / "embedding_centroids.csv", index=False)

    group_order = args.group_order
    if args.group_column == "Clinical_Category" and args.group_order == DEFAULT_GROUP_ORDER:
        group_order = sorted(embedding_df["Clinical_Category"].dropna().unique())

    plot_embedding_only(
        embedding_df=embedding_df,
        output_path=output_dir / "distance_healthy_plus_neuro_minus_ortho_boxplot.png",
        group_column=args.group_column,
        group_order=group_order,
    )

    saved_count = 0
    for path in feature_paths(args.features_dir, args.feature_pattern):
        feature_df = pd.read_csv(path)
        feature_key = feature_key_from_path(path)
        output_path = output_dir / f"{feature_key}_embedding_vs_handcrafted_boxplots.png"
        saved = plot_feature_comparison(
            embedding_df=embedding_df,
            feature_df=feature_df,
            feature_key=feature_key,
            metrics=args.metrics,
            output_path=output_path,
            group_column=args.group_column,
            group_order=group_order,
        )
        if saved:
            saved_count += 1
            print(f"Saved comparison plot: {output_path}")

    print(f"Saved embedding biomarker table: {embedding_scalar_path}")
    print(f"Saved embedding centroids: {output_dir / 'embedding_centroids.csv'}")
    print(f"Saved embedding-only boxplot: {output_dir / 'distance_healthy_plus_neuro_minus_ortho_boxplot.png'}")
    print(f"Saved {saved_count} embedding-vs-handcrafted comparison plots.")


if __name__ == "__main__":
    main()
