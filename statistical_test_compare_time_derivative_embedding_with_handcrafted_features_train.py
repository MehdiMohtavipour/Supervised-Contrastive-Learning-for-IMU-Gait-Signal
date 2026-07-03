"""
Run one-way ANOVA tests for the training split.

The script tests the proposed digital biomarker and the selected handcrafted
features across disease groups. It reports p-values, effect sizes, and a
bootstrap confidence interval for eta-squared.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from compare_time_derivative_embedding_with_handcrafted_features import (
    DEFAULT_CENTROID_CLASSES,
    DEFAULT_EMBEDDINGS_CSV,
    DEFAULT_FEATURES_DIR,
    DEFAULT_GROUP_ORDER,
    DEFAULT_METRICS,
    display_metric_label,
    feature_key_from_path,
    feature_paths,
    load_embedding_centroid_distance_biomarker,
)


DEFAULT_OUTPUT_DIR = Path("time_derivative_embedding_handcrafted_anova_train")
BIOMARKER_COLUMN = "distance_healthy_plus_neuro_minus_ortho"


def filter_embedding_split(embedding_df, split_name):
    if "split" not in embedding_df.columns:
        raise ValueError("Embeddings CSV must contain a split column.")

    split_df = embedding_df.loc[embedding_df["split"].astype(str) == split_name].copy()
    if split_df.empty:
        raise ValueError(f"No rows with split == '{split_name}' found in embeddings CSV.")
    return split_df


def split_trial_keys(split_embedding_df):
    required = ["subject_name", "trial_name"]
    missing = [column for column in required if column not in split_embedding_df.columns]
    if missing:
        raise ValueError(f"Missing required embedding columns: {', '.join(missing)}")

    return set(
        zip(
            split_embedding_df["subject_name"].astype(str),
            split_embedding_df["trial_name"].astype(str),
        )
    )


def filter_feature_df_to_trial_keys(feature_df, keys):
    required = ["Subject_Name", "Trial_Name"]
    missing = [column for column in required if column not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing required handcrafted feature columns: {', '.join(missing)}")

    row_keys = list(
        zip(
            feature_df["Subject_Name"].astype(str),
            feature_df["Trial_Name"].astype(str),
        )
    )
    mask = [key in keys for key in row_keys]
    return feature_df.loc[mask].copy()


def values_by_group(df, value_column, group_column, group_order):
    groups = []
    labels = []
    for group in group_order:
        if group not in set(df[group_column].dropna().astype(str)):
            continue
        values = pd.to_numeric(
            df.loc[df[group_column].astype(str) == group, value_column],
            errors="coerce",
        ).dropna()
        if len(values) == 0:
            continue
        labels.append(group)
        groups.append(values.to_numpy(dtype=float))
    return labels, groups


def anova_effect_sizes(groups):
    all_values = np.concatenate(groups)
    grand_mean = all_values.mean()
    ss_between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in groups)
    ss_within = sum(((group - group.mean()) ** 2).sum() for group in groups)
    ss_total = ss_between + ss_within
    df_between = len(groups) - 1
    df_within = len(all_values) - len(groups)
    ms_within = ss_within / df_within if df_within > 0 else np.nan

    eta_squared = ss_between / ss_total if ss_total > 0 else np.nan
    omega_denominator = ss_total + ms_within
    omega_squared = (
        (ss_between - df_between * ms_within) / omega_denominator
        if omega_denominator > 0
        else np.nan
    )
    return {
        "df_between": df_between,
        "df_within": df_within,
        "eta_squared": float(eta_squared),
        "omega_squared": float(max(0.0, omega_squared)) if np.isfinite(omega_squared) else np.nan,
    }


def eta_squared_ci(groups, n_bootstrap, ci, rng):
    if n_bootstrap <= 0:
        return np.nan, np.nan

    estimates = []
    for _ in range(n_bootstrap):
        sampled_groups = [
            rng.choice(group, size=len(group), replace=True)
            for group in groups
        ]
        eta_squared = anova_effect_sizes(sampled_groups)["eta_squared"]
        if np.isfinite(eta_squared):
            estimates.append(eta_squared)

    if not estimates:
        return np.nan, np.nan

    alpha = (100.0 - ci) / 2.0
    return tuple(np.percentile(estimates, [alpha, 100.0 - alpha]))


def anova_row(df, value_column, source, display_name, group_column, group_order, n_bootstrap, ci, rng):
    labels, groups = values_by_group(df, value_column, group_column, group_order)
    usable_groups = [group for group in groups if len(group) >= 2]
    usable_labels = [label for label, group in zip(labels, groups) if len(group) >= 2]

    if len(usable_groups) < 2:
        return None

    f_statistic, p_value = stats.f_oneway(*usable_groups)
    effects = anova_effect_sizes(usable_groups)
    ci_low, ci_high = eta_squared_ci(usable_groups, n_bootstrap, ci, rng)

    row = {
        "source": source,
        "feature": display_name,
        "value_column": value_column,
        "group_column": group_column,
        "groups": ",".join(usable_labels),
        "group_counts": ",".join(str(len(group)) for group in usable_groups),
        "f_statistic": float(f_statistic),
        "p_value": float(p_value),
        "eta_squared": effects["eta_squared"],
        "eta_squared_ci_low": float(ci_low),
        "eta_squared_ci_high": float(ci_high),
        "omega_squared": effects["omega_squared"],
        "df_between": effects["df_between"],
        "df_within": effects["df_within"],
    }
    for label, group in zip(usable_labels, usable_groups):
        row[f"{label}_mean"] = float(np.mean(group))
        row[f"{label}_std"] = float(np.std(group, ddof=1)) if len(group) > 1 else np.nan
    return row


def collect_rows(
    split_name,
    embedding_df,
    features_dir,
    feature_pattern,
    metrics,
    group_column,
    group_order,
    n_bootstrap,
    ci,
    rng,
):
    split_df = filter_embedding_split(embedding_df, split_name)
    keys = split_trial_keys(split_df)

    if group_column == "Clinical_Category" and group_order == DEFAULT_GROUP_ORDER:
        group_order = sorted(split_df["Clinical_Category"].dropna().unique())

    rows = []
    rows.append(
        anova_row(
            df=split_df,
            value_column=BIOMARKER_COLUMN,
            source="This work",
            display_name="This work",
            group_column=group_column,
            group_order=group_order,
            n_bootstrap=n_bootstrap,
            ci=ci,
            rng=rng,
        )
    )

    for path in feature_paths(features_dir, feature_pattern):
        feature_df = pd.read_csv(path)
        split_feature_df = filter_feature_df_to_trial_keys(feature_df, keys)
        if split_feature_df.empty:
            continue

        feature_key = feature_key_from_path(path)
        for metric in metrics:
            column = f"{feature_key}_{metric}"
            if column not in split_feature_df.columns:
                continue
            rows.append(
                anova_row(
                    df=split_feature_df,
                    value_column=column,
                    source=feature_key,
                    display_name=display_metric_label(metric),
                    group_column=group_column,
                    group_order=group_order,
                    n_bootstrap=n_bootstrap,
                    ci=ci,
                    rng=rng,
                )
            )

    return [row for row in rows if row is not None]


def main():
    parser = argparse.ArgumentParser(
        description="Run ANOVA tests for training biomarker and handcrafted features."
    )
    parser.add_argument("--embeddings-csv", default=str(DEFAULT_EMBEDDINGS_CSV))
    parser.add_argument("--features-dir", default=str(DEFAULT_FEATURES_DIR))
    parser.add_argument("--feature-pattern", default="features_*_Magnitude.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--centroid-classes", nargs="+", default=DEFAULT_CENTROID_CLASSES)
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--group-column", default="Disease_Group", choices=["Disease_Group", "Clinical_Category"])
    parser.add_argument("--group-order", nargs="+", default=DEFAULT_GROUP_ORDER)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--ci", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    embedding_df, _, _ = load_embedding_centroid_distance_biomarker(
        embeddings_csv=Path(args.embeddings_csv),
        centroid_classes=args.centroid_classes,
        metric=args.metric,
        normalize_before_centroid=not args.no_normalize,
    )
    rows = collect_rows(
        split_name="train",
        embedding_df=embedding_df,
        features_dir=args.features_dir,
        feature_pattern=args.feature_pattern,
        metrics=args.metrics,
        group_column=args.group_column,
        group_order=args.group_order,
        n_bootstrap=args.n_bootstrap,
        ci=args.ci,
        rng=rng,
    )

    results_df = pd.DataFrame(rows).sort_values("p_value", na_position="last")
    results_path = output_dir / "train_anova_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"Saved train ANOVA results: {results_path}")
    if not results_df.empty:
        print(results_df[["source", "feature", "p_value", "eta_squared", "eta_squared_ci_low", "eta_squared_ci_high"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
