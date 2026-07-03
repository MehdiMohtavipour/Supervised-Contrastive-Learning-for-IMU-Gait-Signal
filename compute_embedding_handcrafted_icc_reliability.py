"""
Compute ICC reliability across repeated trials within subjects.

This script estimates test-retest reliability for:
  1. The embedding biomarker:
     distance_to_healthy + distance_to_neuro - distance_to_ortho
  2. The handcrafted features saved by feature_extraction_adapted.py

Subjects are the ICC targets and repeated trial numbers are the repeated
measurements. By default the script reports ICC(2,1): two-way random-effects,
single-measure, absolute-agreement ICC.
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
DEFAULT_FEATURES_DIR = Path("Conventional_extracted_features")
DEFAULT_OUTPUT_DIR = Path("icc_reliability_outputs")
DEFAULT_GROUPS = ["all", "healthy", "neuro", "ortho"]
DEFAULT_HANDCRAFTED_SUFFIXES = [
    "avg_stride_time",
    "max_stride_time",
    "std_stride_time",
    "u_turn_time",
    "gait_cadence",
    "avg_acceleration_peak",
    "max_acceleration_peak",
    "std_acceleration_peak",
    "avg_angular_velocity_peak",
    "max_angular_velocity_peak",
    "std_angular_velocity_peak",
    "avg_jerk",
    "max_jerk",
    "std_jerk",
    "avg_angular_acceleration",
    "max_angular_acceleration",
    "std_angular_acceleration",
]
METADATA_COLUMNS = {
    "split",
    "file_path",
    "category",
    "disease_group",
    "subgroup",
    "Disease_Group",
    "Subgroup",
    "Clinical_Category",
    "Subject_ID",
    "Subject_Name",
    "subject_name",
    "Trial_ID",
    "Trial_Name",
    "trial_name",
    "Num_Samples",
    "gender",
    "age",
    "BMI",
    "metadata_group",
    "group",
    "height",
}
ICC_PLOT_FEATURE_LABELS = {
    "distance_healthy_plus_neuro_minus_ortho": "EDGB (This work)",
    "RF_FreeAcc_Magnitude_avg_stride_time": "Average stride time",
    "RF_FreeAcc_Magnitude_gait_cadence": "Gait cadence",
    "RF_FreeAcc_Magnitude_std_stride_time": "Std stride time",
    "RF_FreeAcc_Magnitude_max_stride_time": "Max stride time",
    "RF_FreeAcc_Magnitude_u_turn_time": "U-turn time",
}
ICC_PLOT_FEATURE_ORDER = list(ICC_PLOT_FEATURE_LABELS)


def standardize_repeated_measure_columns(df):
    df = df.copy()
    if "subject_name" in df.columns and "Subject_Name" not in df.columns:
        df["Subject_Name"] = df["subject_name"]
    if "trial_name" in df.columns and "Trial_Name" not in df.columns:
        df["Trial_Name"] = df["trial_name"]
    if "disease_group" in df.columns and "Disease_Group" not in df.columns:
        df["Disease_Group"] = df["disease_group"]

    if "Trial_ID" not in df.columns:
        if "Trial_Name" not in df.columns:
            raise ValueError("Input table must contain Trial_ID or Trial_Name.")
        df["Trial_ID"] = (
            df["Trial_Name"]
            .astype(str)
            .str.extract(r"_(\d+)$", expand=False)
            .astype(float)
        )

    if "Subject_Name" not in df.columns:
        raise ValueError("Input table must contain Subject_Name or subject_name.")
    if "Disease_Group" not in df.columns:
        raise ValueError("Input table must contain Disease_Group or disease_group.")

    df["Subject_Name"] = df["Subject_Name"].astype(str)
    df["Disease_Group"] = df["Disease_Group"].astype(str)
    return df


def icc_2_1(values):
    """Compute ICC(2,1) for a complete subject x trial matrix."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values).all(axis=1)]
    n_subjects, n_trials = values.shape
    if n_subjects < 2 or n_trials < 2:
        return np.nan, {
            "n_subjects": n_subjects,
            "n_trials": n_trials,
            "ms_subject": np.nan,
            "ms_trial": np.nan,
            "ms_error": np.nan,
        }

    grand_mean = values.mean()
    subject_means = values.mean(axis=1, keepdims=True)
    trial_means = values.mean(axis=0, keepdims=True)

    ss_subject = n_trials * np.sum((subject_means - grand_mean) ** 2)
    ss_trial = n_subjects * np.sum((trial_means - grand_mean) ** 2)
    ss_total = np.sum((values - grand_mean) ** 2)
    ss_error = ss_total - ss_subject - ss_trial

    ms_subject = ss_subject / (n_subjects - 1)
    ms_trial = ss_trial / (n_trials - 1)
    ms_error = ss_error / ((n_subjects - 1) * (n_trials - 1))

    denominator = (
        ms_subject
        + (n_trials - 1) * ms_error
        + n_trials * (ms_trial - ms_error) / n_subjects
    )
    icc = (ms_subject - ms_error) / denominator if denominator != 0 else np.nan
    return float(icc), {
        "n_subjects": n_subjects,
        "n_trials": n_trials,
        "ms_subject": float(ms_subject),
        "ms_trial": float(ms_trial),
        "ms_error": float(ms_error),
    }


def complete_subject_trial_matrix(df, value_column, min_trials):
    pivot = df.pivot_table(
        index="Subject_Name",
        columns="Trial_ID",
        values=value_column,
        aggfunc="mean",
    )
    valid_trial_counts = pivot.notna().sum(axis=1)
    pivot = pivot.loc[valid_trial_counts >= min_trials]
    usable_columns = pivot.columns[pivot.notna().sum(axis=0) >= 2]
    pivot = pivot[usable_columns]

    if pivot.shape[1] > min_trials:
        best_columns = (
            pivot.notna()
            .sum(axis=0)
            .sort_values(ascending=False)
            .head(pivot.shape[1])
            .index
        )
        pivot = pivot[best_columns]

    pivot = pivot.dropna(axis=0, how="any")
    return pivot


def compute_icc_rows(df, value_columns, source, group_values, min_trials):
    rows = []
    df = standardize_repeated_measure_columns(df)

    for group in group_values:
        if group == "all":
            group_df = df
        else:
            group_df = df.loc[df["Disease_Group"] == group]

        if group_df.empty:
            continue

        for value_column in value_columns:
            if value_column not in group_df.columns:
                continue
            numeric_df = group_df.copy()
            numeric_df[value_column] = pd.to_numeric(
                numeric_df[value_column],
                errors="coerce",
            )
            numeric_df = numeric_df.dropna(subset=[value_column, "Trial_ID"])
            matrix = complete_subject_trial_matrix(numeric_df, value_column, min_trials)
            icc, details = icc_2_1(matrix.to_numpy(dtype=float))
            rows.append(
                {
                    "source": source,
                    "group": group,
                    "feature": value_column,
                    "icc_2_1": icc,
                    "n_subjects": details["n_subjects"],
                    "n_trials": details["n_trials"],
                    "trial_ids_used": ",".join(str(col) for col in matrix.columns),
                    "ms_subject": details["ms_subject"],
                    "ms_trial": details["ms_trial"],
                    "ms_error": details["ms_error"],
                }
            )
    return rows


def biomarker_columns(df, requested_columns):
    if requested_columns:
        return requested_columns

    default_column = "distance_healthy_plus_neuro_minus_ortho"
    if default_column in df.columns:
        return [default_column]

    return [
        column
        for column in df.columns
        if column.startswith("distance_") and pd.api.types.is_numeric_dtype(df[column])
    ]


def handcrafted_feature_columns(df, suffixes):
    columns = []
    for column in df.columns:
        if column in METADATA_COLUMNS:
            continue
        if any(column.endswith(f"_{suffix}") for suffix in suffixes):
            columns.append(column)
    return columns


def load_handcrafted_tables(features_dir, pattern, include_derived):
    paths = sorted(Path(features_dir).glob(pattern))
    if not include_derived:
        paths = [path for path in paths if "RF_minus_LF" not in path.name]
    if not paths:
        raise FileNotFoundError(f"No handcrafted feature CSVs found in {features_dir}")
    return paths


def plot_icc_summary(results_df, output_path):
    plot_df = results_df.loc[
        (results_df["group"] == "all")
        & results_df["icc_2_1"].notna()
    ].copy()
    if plot_df.empty:
        return

    plot_df = plot_df.loc[plot_df["feature"].isin(ICC_PLOT_FEATURE_ORDER)].copy()
    plot_df["plot_label"] = plot_df["feature"].map(ICC_PLOT_FEATURE_LABELS)
    plot_df["plot_order"] = plot_df["feature"].map(
        {feature: index for index, feature in enumerate(ICC_PLOT_FEATURE_ORDER)}
    )
    plot_df = plot_df.sort_values("plot_order")
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    y = np.arange(len(plot_df))
    colors = np.where(plot_df["source"] == "embedding_biomarker", "#356f9f", "#6a9f58")
    ax.barh(y, plot_df["icc_2_1"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["plot_label"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("ICC(2,1)")
    ax.set_title("Overall reliability estimates")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0.75, color="gray", linestyle=":", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def selected_plot_rows(results_df):
    plot_df = results_df.loc[
        (results_df["group"] == "all")
        & results_df["icc_2_1"].notna()
        & results_df["feature"].isin(ICC_PLOT_FEATURE_ORDER)
    ].copy()
    plot_df["plot_label"] = plot_df["feature"].map(ICC_PLOT_FEATURE_LABELS)
    plot_df["plot_order"] = plot_df["feature"].map(
        {feature: index for index, feature in enumerate(ICC_PLOT_FEATURE_ORDER)}
    )
    return plot_df.sort_values("plot_order")


def main():
    parser = argparse.ArgumentParser(
        description="Compute ICC reliability for embedding biomarker and handcrafted features."
    )
    parser.add_argument("--biomarker-csv", default=str(DEFAULT_BIOMARKER_CSV), help="Embedding biomarker CSV")
    parser.add_argument(
        "--biomarker-columns",
        nargs="+",
        default=None,
        help="Biomarker columns to evaluate. Defaults to distance_healthy_plus_neuro_minus_ortho.",
    )
    parser.add_argument("--features-dir", default=str(DEFAULT_FEATURES_DIR), help="Handcrafted feature CSV directory")
    parser.add_argument("--feature-pattern", default="features_*_Magnitude.csv", help="Handcrafted feature CSV glob")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for ICC outputs")
    parser.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS, help="Groups to compute: all healthy neuro ortho")
    parser.add_argument("--min-trials", type=int, default=2, help="Minimum complete repeated trials per subject")
    parser.add_argument("--include-derived", action="store_true", help="Include RF_minus_LF feature files if present")
    parser.add_argument(
        "--feature-suffixes",
        nargs="+",
        default=DEFAULT_HANDCRAFTED_SUFFIXES,
        help="Handcrafted feature suffixes to include",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    biomarker_df = pd.read_csv(args.biomarker_csv)
    biomarker_value_columns = biomarker_columns(biomarker_df, args.biomarker_columns)
    rows.extend(
        compute_icc_rows(
            df=biomarker_df,
            value_columns=biomarker_value_columns,
            source="embedding_biomarker",
            group_values=args.groups,
            min_trials=args.min_trials,
        )
    )

    for path in load_handcrafted_tables(args.features_dir, args.feature_pattern, args.include_derived):
        feature_df = pd.read_csv(path)
        feature_columns = handcrafted_feature_columns(feature_df, args.feature_suffixes)
        rows.extend(
            compute_icc_rows(
                df=feature_df,
                value_columns=feature_columns,
                source=path.stem.replace("features_", ""),
                group_values=args.groups,
                min_trials=args.min_trials,
            )
        )

    results_df = pd.DataFrame(rows)
    results_path = output_dir / "icc_reliability_results.csv"
    results_df.to_csv(results_path, index=False)

    biomarker_results = results_df.loc[results_df["source"] == "embedding_biomarker"].copy()
    biomarker_path = output_dir / "embedding_biomarker_icc.csv"
    biomarker_results.to_csv(biomarker_path, index=False)

    overall_path = output_dir / "icc_reliability_overall.csv"
    results_df.loc[results_df["group"] == "all"].to_csv(overall_path, index=False)
    selected_plot_rows(results_df).to_csv(output_dir / "icc_reliability_summary_rows.csv", index=False)
    plot_icc_summary(results_df, output_dir / "icc_reliability_summary.png")

    print(f"Saved ICC reliability results: {results_path}")
    print(f"Saved embedding biomarker ICC: {biomarker_path}")
    print(f"Saved overall ICC table: {overall_path}")
    print(f"Saved ICC summary rows: {output_dir / 'icc_reliability_summary_rows.csv'}")
    print(f"Saved ICC summary plot: {output_dir / 'icc_reliability_summary.png'}")
    if not biomarker_results.empty:
        print("\nEmbedding biomarker ICC:")
        print(biomarker_results.to_string(index=False))
    if not results_df.empty:
        print("\nTop ICC results:")
        print(results_df.sort_values("icc_2_1", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
