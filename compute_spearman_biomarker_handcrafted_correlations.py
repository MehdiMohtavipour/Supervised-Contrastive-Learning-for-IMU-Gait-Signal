"""
Compute pairwise Spearman correlations for test subjects.

Outputs are separated into:
  1. A timing/cadence heatmap:
     Proposed Biomarker, Avg/Max/Std stride time, U-turn time, Gait cadence.
  2. Four sensor-specific signal-feature heatmaps:
     RF, LF, LB, and HE. Each includes the Proposed Biomarker plus acceleration,
     jerk, angular velocity, angular acceleration, and simple mean/std features.
"""

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from compare_time_derivative_embedding_with_handcrafted_features import (
    DEFAULT_CENTROID_CLASSES,
    DEFAULT_EMBEDDINGS_CSV,
    DEFAULT_FEATURES_DIR,
    feature_key_from_path,
    feature_paths,
    load_embedding_centroid_distance_biomarker,
)


DEFAULT_OUTPUT_DIR = Path("time_derivative_embedding_handcrafted_spearman")
BIOMARKER_COLUMN = "distance_healthy_plus_neuro_minus_ortho"
BIOMARKER_LABEL = "Proposed Biomarker"
DEFAULT_SENSORS = ["RF", "LF", "LB", "HE"]

TIMING_METRICS = [
    "avg_stride_time",
    "max_stride_time",
    "std_stride_time",
    "u_turn_time",
    "gait_cadence",
]
FREEACC_SIGNAL_METRICS = [
    "avg_jerk",
    "max_jerk",
    "std_jerk",
]
GYR_SIGNAL_METRICS = [
    "avg_angular_acceleration",
    "std_angular_acceleration",
]

FEATURE_LABELS = {
    "avg_stride_time": "Avg stride time",
    "max_stride_time": "Max stride time",
    "std_stride_time": "Std stride time",
    "u_turn_time": "U-turn time",
    "gait_cadence": "Gait cadence",
    "avg_acceleration_peak": "Avg acceleration peak",
    "std_acceleration_peak": "Std acceleration peak",
    "avg_jerk": "Avg jerk",
    "max_jerk": "Max jerk",
    "std_jerk": "Std jerk",
    "avg_angular_velocity_peak": "Avg angular velocity peak",
    "std_angular_velocity_peak": "Std angular velocity peak",
    "avg_angular_acceleration": "Avg angular acceleration",
    "std_angular_acceleration": "Std angular acceleration",
}
SIMPLE_SIGNAL_FEATURES = [
    "Mean acceleration",
    "Std acceleration",
    "Mean angular velocity",
    "Std angular velocity",
]


def feature_label(metric):
    return FEATURE_LABELS.get(metric, metric.replace("_", " ").title())


def embedding_match_table(embedding_df):
    required = ["subject_name", "trial_name", BIOMARKER_COLUMN]
    missing = [column for column in required if column not in embedding_df.columns]
    if missing:
        raise ValueError(f"Missing required embedding columns: {', '.join(missing)}")

    columns = [
        column
        for column in [
            "subject_name",
            "trial_name",
            "file_path",
            "split",
            "Disease_Group",
            "Clinical_Category",
            BIOMARKER_COLUMN,
        ]
        if column in embedding_df.columns
    ]
    table = embedding_df[columns].copy()
    table["Subject_Name"] = table["subject_name"].astype(str)
    table["Trial_Name"] = table["trial_name"].astype(str)
    table[BIOMARKER_COLUMN] = pd.to_numeric(table[BIOMARKER_COLUMN], errors="coerce")
    table = table.rename(columns={BIOMARKER_COLUMN: BIOMARKER_LABEL})
    return table.dropna(subset=[BIOMARKER_LABEL])


def load_feature_tables(features_dir, feature_pattern):
    tables = {}
    for path in feature_paths(features_dir, feature_pattern):
        tables[feature_key_from_path(path)] = pd.read_csv(path)
    return tables


def merge_feature_columns(wide_df, feature_df, column_map):
    if not column_map:
        return wide_df, []

    required = ["Subject_Name", "Trial_Name", *column_map.keys()]
    missing = [column for column in required if column not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing required handcrafted feature columns: {', '.join(missing)}")

    feature_table = feature_df[required].copy()
    feature_table["Subject_Name"] = feature_table["Subject_Name"].astype(str)
    feature_table["Trial_Name"] = feature_table["Trial_Name"].astype(str)
    for column in column_map:
        feature_table[column] = pd.to_numeric(feature_table[column], errors="coerce")
    feature_table = feature_table.rename(columns=column_map)

    merged = wide_df.merge(feature_table, on=["Subject_Name", "Trial_Name"], how="inner")
    return merged, list(column_map.values())


def magnitude_from_axes(df, sensor, component):
    axis_columns = [f"{sensor}_{component}_{axis}" for axis in ["X", "Y", "Z"]]
    if any(column not in df.columns for column in axis_columns):
        return None

    axis_values = [
        pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        for column in axis_columns
    ]
    stacked = np.vstack(axis_values)
    return np.sqrt(np.nansum(stacked ** 2, axis=0))


def compute_simple_signal_features(file_path, sensor):
    try:
        signal_df = pd.read_csv(file_path, sep="\t")
    except Exception:
        return {feature: np.nan for feature in SIMPLE_SIGNAL_FEATURES}

    acceleration = magnitude_from_axes(signal_df, sensor, "FreeAcc")
    angular_velocity = magnitude_from_axes(signal_df, sensor, "Gyr")

    features = {}
    if acceleration is None:
        features["Mean acceleration"] = np.nan
        features["Std acceleration"] = np.nan
    else:
        acceleration = acceleration[np.isfinite(acceleration)]
        features["Mean acceleration"] = float(np.mean(acceleration)) if acceleration.size else np.nan
        features["Std acceleration"] = float(np.std(acceleration)) if acceleration.size else np.nan

    if angular_velocity is None:
        features["Mean angular velocity"] = np.nan
        features["Std angular velocity"] = np.nan
    else:
        angular_velocity = angular_velocity[np.isfinite(angular_velocity)]
        features["Mean angular velocity"] = float(np.mean(angular_velocity)) if angular_velocity.size else np.nan
        features["Std angular velocity"] = float(np.std(angular_velocity)) if angular_velocity.size else np.nan

    return features


def add_simple_signal_features(wide_df, sensor):
    if "file_path" not in wide_df.columns:
        return wide_df, []

    simple_rows = [
        compute_simple_signal_features(file_path, sensor)
        for file_path in wide_df["file_path"].astype(str)
    ]
    simple_df = pd.DataFrame(simple_rows, index=wide_df.index)
    return pd.concat([wide_df, simple_df], axis=1), SIMPLE_SIGNAL_FEATURES.copy()


def build_timing_table(embedding_table, feature_tables):
    feature_key = "RF_FreeAcc_Magnitude"
    if feature_key not in feature_tables:
        raise FileNotFoundError(f"Missing representative timing feature table: {feature_key}")

    column_map = {
        f"{feature_key}_{metric}": feature_label(metric)
        for metric in TIMING_METRICS
    }
    wide_df, feature_columns = merge_feature_columns(
        wide_df=embedding_table.copy(),
        feature_df=feature_tables[feature_key],
        column_map=column_map,
    )
    value_columns = [BIOMARKER_LABEL, *feature_columns]
    return wide_df.dropna(subset=value_columns), value_columns


def build_sensor_signal_table(embedding_table, feature_tables, sensor):
    wide_df = embedding_table.copy()
    value_columns = [BIOMARKER_LABEL]

    freeacc_key = f"{sensor}_FreeAcc_Magnitude"
    if freeacc_key in feature_tables:
        freeacc_map = {
            f"{freeacc_key}_{metric}": feature_label(metric)
            for metric in FREEACC_SIGNAL_METRICS
        }
        wide_df, columns = merge_feature_columns(wide_df, feature_tables[freeacc_key], freeacc_map)
        value_columns.extend(columns)

    gyr_key = f"{sensor}_Gyr_Magnitude"
    if gyr_key in feature_tables:
        gyr_map = {
            f"{gyr_key}_{metric}": feature_label(metric)
            for metric in GYR_SIGNAL_METRICS
        }
        wide_df, columns = merge_feature_columns(wide_df, feature_tables[gyr_key], gyr_map)
        value_columns.extend(columns)

    wide_df, simple_columns = add_simple_signal_features(wide_df, sensor)
    value_columns.extend(simple_columns)
    value_columns = list(dict.fromkeys(value_columns))
    return wide_df.dropna(subset=value_columns), value_columns


def collect_pairwise_correlations(group_name, wide_df, value_columns, splits):
    rows = []
    matrices = {}
    for split_name in splits:
        if split_name == "all":
            split_df = wide_df
        else:
            if "split" not in wide_df.columns:
                continue
            split_df = wide_df.loc[wide_df["split"].astype(str) == split_name]

        split_values = split_df[value_columns].apply(pd.to_numeric, errors="coerce").dropna()
        if len(split_values) < 3:
            continue

        matrices[(group_name, split_name)] = split_values.corr(method="spearman")
        for left_idx, left_column in enumerate(value_columns):
            for right_column in value_columns[left_idx + 1:]:
                pair_df = split_values[[left_column, right_column]].dropna()
                if len(pair_df) < 3:
                    continue
                rho, p_value = stats.spearmanr(pair_df[left_column], pair_df[right_column])
                rows.append(
                    {
                        "plot_group": group_name,
                        "split": split_name,
                        "feature_1": left_column,
                        "feature_2": right_column,
                        "n": int(len(pair_df)),
                        "spearman_rho": float(rho),
                        "p_value": float(p_value),
                    }
                )
    return rows, matrices


def format_value(value):
    if pd.isna(value):
        return ""
    if isinstance(value, str):
        return value
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) < 0.001:
        return f"{value:.4e}"
    return f"{value:.4f}"


def write_markdown_summary(results_df, output_path):
    columns = ["plot_group", "split", "feature_1", "feature_2", "n", "spearman_rho", "p_value"]
    display = results_df[columns].copy()
    display = display.sort_values(["plot_group", "split", "p_value"], na_position="last")
    display.columns = ["Plot group", "Split", "Feature 1", "Feature 2", "N", "Spearman rho", "p-value"]
    for column in display.columns:
        display[column] = display[column].map(format_value)

    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.to_numpy(dtype=str)]

    lines = [
        "# Pairwise Spearman Correlations",
        "",
        "Rows are matched by `Subject_Name` and `Trial_Name`.",
        "The timing plot uses sensor-independent timing/cadence features.",
        "The RF, LF, LB, and HE plots use that sensor's own FreeAcc/Gyr signal features.",
        "",
        header,
        separator,
        *rows,
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_file_label(label):
    return label.lower().replace(" ", "_").replace("-", "_")


def plot_title(group_name, split_name):
    if group_name in DEFAULT_SENSORS:
        return f"{group_name} sensor ({split_name})"
    if group_name == "Timing":
        return f"Timing features ({split_name})"
    return f"{group_name} ({split_name})"


def plot_heatmaps(matrices, output_dir):
    plot_paths = []
    for (group_name, split_name), matrix_df in matrices.items():
        if matrix_df.empty:
            continue

        fig_width = max(7.5, len(matrix_df.columns) * 1.05)
        fig_height = max(6.5, len(matrix_df.index) * 0.95)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        image = ax.imshow(matrix_df.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")

        ax.set_xticks(np.arange(len(matrix_df.columns)))
        ax.set_xticklabels(matrix_df.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(matrix_df.index)))
        ax.set_yticklabels(matrix_df.index)
        ax.set_title(plot_title(group_name, split_name))

        for row_idx, row_name in enumerate(matrix_df.index):
            for col_idx, column_name in enumerate(matrix_df.columns):
                value = matrix_df.loc[row_name, column_name]
                text_color = "white" if abs(value) > 0.45 else "#1f2937"
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)

        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("Spearman rho")
        fig.tight_layout()

        file_label = safe_file_label(group_name)
        output_path = output_dir / f"spearman_{file_label}_{split_name}_heatmap.png"
        fig.savefig(output_path, dpi=190)
        plt.close(fig)
        plot_paths.append(output_path)
    return plot_paths


def save_correlation_matrices(matrices, output_dir):
    matrix_paths = []
    for (group_name, split_name), matrix_df in matrices.items():
        file_label = safe_file_label(group_name)
        output_path = output_dir / f"spearman_{file_label}_{split_name}_matrix.csv"
        matrix_df.to_csv(output_path)
        matrix_paths.append(output_path)
    return matrix_paths


def main():
    parser = argparse.ArgumentParser(
        description="Compute grouped pairwise Spearman correlations for the proposed biomarker and handcrafted features."
    )
    parser.add_argument("--embeddings-csv", default=str(DEFAULT_EMBEDDINGS_CSV))
    parser.add_argument("--features-dir", default=str(DEFAULT_FEATURES_DIR))
    parser.add_argument("--feature-pattern", default="features_*_Magnitude.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--sensors", nargs="+", default=DEFAULT_SENSORS, choices=DEFAULT_SENSORS)
    parser.add_argument("--centroid-classes", nargs="+", default=DEFAULT_CENTROID_CLASSES)
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_df, _, _ = load_embedding_centroid_distance_biomarker(
        embeddings_csv=Path(args.embeddings_csv),
        centroid_classes=args.centroid_classes,
        metric=args.metric,
        normalize_before_centroid=not args.no_normalize,
    )
    if "all" not in args.splits and "split" in embedding_df.columns:
        embedding_df = embedding_df.loc[embedding_df["split"].astype(str).isin(args.splits)].copy()

    embedding_table = embedding_match_table(embedding_df)
    feature_tables = load_feature_tables(args.features_dir, args.feature_pattern)

    all_rows = []
    all_matrices = {}

    timing_df, timing_columns = build_timing_table(embedding_table, feature_tables)
    rows, matrices = collect_pairwise_correlations("Timing", timing_df, timing_columns, args.splits)
    all_rows.extend(rows)
    all_matrices.update(matrices)

    for sensor in args.sensors:
        sensor_df, sensor_columns = build_sensor_signal_table(embedding_table, feature_tables, sensor)
        rows, matrices = collect_pairwise_correlations(sensor, sensor_df, sensor_columns, args.splits)
        all_rows.extend(rows)
        all_matrices.update(matrices)

    results_df = pd.DataFrame(all_rows)
    if results_df.empty:
        raise ValueError("No Spearman correlation rows were computed.")

    results_df = results_df.sort_values(["plot_group", "split", "p_value"], na_position="last")
    results_path = output_dir / "spearman_grouped_pairwise_correlations.csv"
    summary_path = output_dir / "spearman_grouped_pairwise_correlations.md"
    results_df.to_csv(results_path, index=False)
    matrix_paths = save_correlation_matrices(all_matrices, output_dir)
    write_markdown_summary(results_df, summary_path)
    plot_paths = plot_heatmaps(all_matrices, output_dir)

    print(f"Saved Spearman correlation results: {results_path}")
    print(f"Saved Spearman correlation summary: {summary_path}")
    for matrix_path in matrix_paths:
        print(f"Saved Spearman correlation matrix: {matrix_path}")
    for plot_path in plot_paths:
        print(f"Saved Spearman correlation plot: {plot_path}")
    print(results_df[["plot_group", "split", "feature_1", "feature_2", "n", "spearman_rho", "p_value"]].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
