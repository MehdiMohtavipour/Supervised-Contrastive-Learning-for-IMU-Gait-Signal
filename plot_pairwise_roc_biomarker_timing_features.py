"""
Plot pairwise ROC curves for the embedding biomarker and timing features.

The script creates one ROC figure per pair:
  1. healthy vs neuro
  2. healthy vs ortho
  3. neuro vs ortho

It also writes per-pair AUC values and one macro-AUC summary per feature.
Rows are matched by subject and trial between the saved embedding biomarker
table and the representative RF FreeAcc Magnitude timing feature table.
"""

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_BIOMARKER_CSV = Path(
    "time_derivative_embedding_handcrafted_comparison_plots/"
    "distance_healthy_plus_neuro_minus_ortho_by_trial.csv"
)
DEFAULT_FEATURES_CSV = Path("final_extracted_features/features_RF_FreeAcc_Magnitude.csv")
DEFAULT_OUTPUT_DIR = Path("pairwise_roc_biomarker_timing_features")
BIOMARKER_COLUMN = "distance_healthy_plus_neuro_minus_ortho"
BIOMARKER_LABEL = "EDGB (This Work)"
DEFAULT_GROUP_PAIRS = [
    ("healthy", "neuro"),
    ("healthy", "ortho"),
    ("neuro", "ortho"),
]
TIMING_METRICS = [
    "avg_stride_time",
    "max_stride_time",
    "std_stride_time",
    "u_turn_time",
    "gait_cadence",
]
FEATURE_LABELS = {
    BIOMARKER_COLUMN: BIOMARKER_LABEL,
    "RF_FreeAcc_Magnitude_avg_stride_time": "Avg stride time",
    "RF_FreeAcc_Magnitude_max_stride_time": "Max stride time",
    "RF_FreeAcc_Magnitude_std_stride_time": "Std stride time",
    "RF_FreeAcc_Magnitude_u_turn_time": "U-turn time",
    "RF_FreeAcc_Magnitude_gait_cadence": "Gait cadence",
}
PLOT_COLORS = {
    BIOMARKER_COLUMN: "#2f6f9f",
    "RF_FreeAcc_Magnitude_avg_stride_time": "#7a9f35",
    "RF_FreeAcc_Magnitude_max_stride_time": "#a45f2b",
    "RF_FreeAcc_Magnitude_std_stride_time": "#7b5ea7",
    "RF_FreeAcc_Magnitude_u_turn_time": "#b14b67",
    "RF_FreeAcc_Magnitude_gait_cadence": "#2f8f83",
}


def standardize_biomarker_table(df):
    df = df.copy()
    rename_map = {}
    if "subject_name" in df.columns:
        rename_map["subject_name"] = "Subject_Name"
    if "trial_name" in df.columns:
        rename_map["trial_name"] = "Trial_Name"
    if "disease_group" in df.columns and "Disease_Group" not in df.columns:
        rename_map["disease_group"] = "Disease_Group"
    df = df.rename(columns=rename_map)

    required = ["Subject_Name", "Trial_Name", "Disease_Group", BIOMARKER_COLUMN]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required biomarker columns: {', '.join(missing)}")

    keep_columns = ["Subject_Name", "Trial_Name", "Disease_Group", BIOMARKER_COLUMN]
    if "split" in df.columns:
        keep_columns.insert(0, "split")

    table = df[keep_columns].copy()
    table["Subject_Name"] = table["Subject_Name"].astype(str)
    table["Trial_Name"] = table["Trial_Name"].astype(str)
    table["Disease_Group"] = table["Disease_Group"].astype(str)
    table[BIOMARKER_COLUMN] = pd.to_numeric(table[BIOMARKER_COLUMN], errors="coerce")
    return table.dropna(subset=[BIOMARKER_COLUMN])


def timing_columns(feature_df, metrics, prefix):
    columns = [f"{prefix}_{metric}" for metric in metrics]
    missing = [column for column in columns if column not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing timing feature columns: {', '.join(missing)}")
    return columns


def load_matched_table(biomarker_csv, features_csv, metrics, timing_prefix):
    biomarker_df = standardize_biomarker_table(pd.read_csv(biomarker_csv))
    feature_df = pd.read_csv(features_csv)

    timing_value_columns = timing_columns(feature_df, metrics, timing_prefix)
    feature_columns = ["Subject_Name", "Trial_Name", *timing_value_columns]
    missing = [column for column in feature_columns if column not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")

    feature_table = feature_df[feature_columns].copy()
    feature_table["Subject_Name"] = feature_table["Subject_Name"].astype(str)
    feature_table["Trial_Name"] = feature_table["Trial_Name"].astype(str)
    for column in timing_value_columns:
        feature_table[column] = pd.to_numeric(feature_table[column], errors="coerce")

    matched = biomarker_df.merge(
        feature_table,
        on=["Subject_Name", "Trial_Name"],
        how="inner",
    )
    return matched, [BIOMARKER_COLUMN, *timing_value_columns]


def roc_curve_and_auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]

    if labels.size == 0 or np.unique(labels).size != 2:
        return None, None, np.nan

    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    scores = scores[order]

    positive_count = np.sum(labels == 1)
    negative_count = np.sum(labels == 0)
    distinct_threshold_indices = np.where(np.diff(scores))[0]
    threshold_indices = np.r_[distinct_threshold_indices, labels.size - 1]

    true_positives = np.cumsum(labels == 1)[threshold_indices]
    false_positives = np.cumsum(labels == 0)[threshold_indices]

    tpr = np.r_[0.0, true_positives / positive_count, 1.0]
    fpr = np.r_[0.0, false_positives / negative_count, 1.0]
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def pairwise_roc(df, feature_column, group_a, group_b):
    pair_df = df.loc[df["Disease_Group"].isin([group_a, group_b])].copy()
    pair_df[feature_column] = pd.to_numeric(pair_df[feature_column], errors="coerce")
    pair_df = pair_df.dropna(subset=[feature_column])

    labels = (pair_df["Disease_Group"].astype(str) == group_b).astype(int).to_numpy()
    scores = pair_df[feature_column].to_numpy(dtype=float)
    fpr, tpr, raw_auc = roc_curve_and_auc(labels, scores)
    if np.isnan(raw_auc):
        return None

    flipped = raw_auc < 0.5
    if flipped:
        fpr, tpr, oriented_auc = roc_curve_and_auc(labels, -scores)
        higher_score_group = group_a
    else:
        oriented_auc = raw_auc
        higher_score_group = group_b

    return {
        "pair": f"{group_a}_vs_{group_b}",
        "group_a": group_a,
        "group_b": group_b,
        "positive_class_raw": group_b,
        "feature": feature_column,
        "feature_label": FEATURE_LABELS.get(feature_column, feature_column),
        "n": int(len(pair_df)),
        "n_group_a": int(np.sum(pair_df["Disease_Group"].astype(str) == group_a)),
        "n_group_b": int(np.sum(pair_df["Disease_Group"].astype(str) == group_b)),
        "raw_auc": float(raw_auc),
        "auc": float(oriented_auc),
        "score_flipped_for_auc": bool(flipped),
        "higher_score_group": higher_score_group,
        "fpr": fpr,
        "tpr": tpr,
    }


def safe_filename(value):
    return value.lower().replace(" ", "_").replace("-", "_")


def plot_pair_curves(pair_results, output_path, title):
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    for result in pair_results:
        feature = result["feature"]
        label = result["feature_label"]
        ax.plot(
            result["fpr"],
            result["tpr"],
            linewidth=2.0,
            color=PLOT_COLORS.get(feature),
            label=label,
        )

    ax.plot([0, 1], [0, 1], color="#6b7280", linestyle="--", linewidth=1.0, label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def format_markdown_value(value):
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def dataframe_to_markdown(df):
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [format_markdown_value(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown_summary(pair_auc_df, macro_auc_df, output_path):
    lines = [
        "# Pairwise ROC AUC Summary",
        "",
        "AUC values are oriented so values below 0.5 are flipped for discriminative performance.",
        "`raw_auc` keeps the unflipped direction where the second group in the pair is positive.",
        "",
        "## Macro AUC",
        "",
        dataframe_to_markdown(macro_auc_df),
        "",
        "## Pairwise AUC",
        "",
        dataframe_to_markdown(pair_auc_df),
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot ROC curves for digital biomarker and timing handcrafted features."
    )
    parser.add_argument("--biomarker-csv", default=str(DEFAULT_BIOMARKER_CSV))
    parser.add_argument("--features-csv", default=str(DEFAULT_FEATURES_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--timing-prefix", default="RF_FreeAcc_Magnitude")
    parser.add_argument("--metrics", nargs="+", default=TIMING_METRICS)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        help="Splits to include. Use 'all' to include every row.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matched_df, feature_columns = load_matched_table(
        biomarker_csv=Path(args.biomarker_csv),
        features_csv=Path(args.features_csv),
        metrics=args.metrics,
        timing_prefix=args.timing_prefix,
    )
    if "all" not in args.splits and "split" in matched_df.columns:
        matched_df = matched_df.loc[matched_df["split"].astype(str).isin(args.splits)].copy()

    if matched_df.empty:
        raise ValueError("No matched biomarker/feature rows remain after filtering.")

    all_results = []
    for group_a, group_b in DEFAULT_GROUP_PAIRS:
        pair_results = []
        for feature_column in feature_columns:
            result = pairwise_roc(matched_df, feature_column, group_a, group_b)
            if result is None:
                continue
            pair_results.append(result)
            all_results.append(result)

        if not pair_results:
            continue

        title = f"ROC curves: {group_a} vs {group_b}"
        output_path = output_dir / f"roc_{safe_filename(group_a)}_vs_{safe_filename(group_b)}.png"
        plot_pair_curves(pair_results, output_path, title)

    if not all_results:
        raise ValueError("No pairwise ROC results were computed.")

    pair_auc_df = pd.DataFrame(
        [
            {
                key: value
                for key, value in result.items()
                if key not in {"fpr", "tpr"}
            }
            for result in all_results
        ]
    )
    pair_auc_df = pair_auc_df.sort_values(["feature", "pair"])
    macro_auc_df = (
        pair_auc_df.groupby(["feature", "feature_label"], as_index=False)
        .agg(
            macro_auc=("auc", "mean"),
            macro_raw_auc=("raw_auc", "mean"),
            n_pairs=("pair", "nunique"),
            total_n=("n", "sum"),
        )
        .sort_values("macro_auc", ascending=False)
    )

    pair_auc_path = output_dir / "pairwise_auc.csv"
    macro_auc_path = output_dir / "macro_auc.csv"
    summary_path = output_dir / "roc_auc_summary.md"
    pair_auc_df.to_csv(pair_auc_path, index=False)
    macro_auc_df.to_csv(macro_auc_path, index=False)
    write_markdown_summary(pair_auc_df, macro_auc_df, summary_path)

    print(f"Matched rows used: {len(matched_df)}")
    print(f"Saved pairwise AUC table: {pair_auc_path}")
    print(f"Saved macro AUC table: {macro_auc_path}")
    print(f"Saved summary: {summary_path}")
    for group_a, group_b in DEFAULT_GROUP_PAIRS:
        print(f"Saved ROC plot: {output_dir / f'roc_{group_a}_vs_{group_b}.png'}")
    print("\nMacro AUC:")
    print(macro_auc_df.to_string(index=False))


if __name__ == "__main__":
    main()
