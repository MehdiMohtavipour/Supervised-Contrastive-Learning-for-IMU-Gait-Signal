"""
Compare the time-derivative embedding biomarker with handcrafted features on test data only.

Centroids are computed from train embeddings, then the biomarker
distance_to_healthy + distance_to_neuro - distance_to_ortho is plotted only for
held-out test trials. Handcrafted feature CSVs are filtered to the same test
Subject_Name/Trial_Name pairs before plotting.
"""

from pathlib import Path
import argparse

import pandas as pd

from compare_time_derivative_embedding_with_handcrafted_features import (
    DEFAULT_CENTROID_CLASSES,
    DEFAULT_EMBEDDINGS_CSV,
    DEFAULT_FEATURES_DIR,
    DEFAULT_GROUP_ORDER,
    DEFAULT_METRICS,
    feature_key_from_path,
    feature_paths,
    load_embedding_centroid_distance_biomarker,
    plot_embedding_only,
    plot_feature_comparison,
)


DEFAULT_OUTPUT_DIR = Path("time_derivative_embedding_handcrafted_comparison_test_only_plots")


def filter_to_test_trials(embedding_df):
    if "split" not in embedding_df.columns:
        raise ValueError("Embeddings CSV must contain a split column for test-only plotting.")

    test_df = embedding_df.loc[embedding_df["split"].astype(str) == "test"].copy()
    if test_df.empty:
        raise ValueError("No rows with split == 'test' found in embeddings CSV.")
    return test_df


def test_trial_keys(test_embedding_df):
    required = ["subject_name", "trial_name"]
    missing = [column for column in required if column not in test_embedding_df.columns]
    if missing:
        raise ValueError(f"Missing required embedding columns: {', '.join(missing)}")

    return set(
        zip(
            test_embedding_df["subject_name"].astype(str),
            test_embedding_df["trial_name"].astype(str),
        )
    )


def filter_feature_df_to_test_trials(feature_df, keys):
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


def main():
    parser = argparse.ArgumentParser(
        description="Compare derivative embedding biomarker with handcrafted feature boxplots on test data only."
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
        help="Category labels used to compute train centroids for the biomarker",
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
    test_embedding_df = filter_to_test_trials(embedding_df)
    keys = test_trial_keys(test_embedding_df)

    group_order = args.group_order
    if args.group_column == "Clinical_Category" and args.group_order == DEFAULT_GROUP_ORDER:
        group_order = sorted(test_embedding_df["Clinical_Category"].dropna().unique())

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
        if column in test_embedding_df.columns
    ]
    test_biomarker_path = output_dir / "test_distance_healthy_plus_neuro_minus_ortho_by_trial.csv"
    test_embedding_df[metadata_columns].to_csv(test_biomarker_path, index=False)

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
    pd.DataFrame(centroid_rows).to_csv(output_dir / "train_embedding_centroids.csv", index=False)

    plot_embedding_only(
        embedding_df=test_embedding_df,
        output_path=output_dir / "test_distance_healthy_plus_neuro_minus_ortho_boxplot.png",
        group_column=args.group_column,
        group_order=group_order,
    )

    saved_count = 0
    for path in feature_paths(args.features_dir, args.feature_pattern):
        feature_df = pd.read_csv(path)
        test_feature_df = filter_feature_df_to_test_trials(feature_df, keys)
        if test_feature_df.empty:
            continue

        feature_key = feature_key_from_path(path)
        output_path = output_dir / f"{feature_key}_test_embedding_vs_handcrafted_boxplots.png"
        saved = plot_feature_comparison(
            embedding_df=test_embedding_df,
            feature_df=test_feature_df,
            feature_key=feature_key,
            metrics=args.metrics,
            output_path=output_path,
            group_column=args.group_column,
            group_order=group_order,
        )
        if saved:
            saved_count += 1
            print(f"Saved test-only comparison plot: {output_path}")

    print(f"Saved test biomarker table: {test_biomarker_path}")
    print(f"Saved train centroids: {output_dir / 'train_embedding_centroids.csv'}")
    print(f"Saved test-only embedding boxplot: {output_dir / 'test_distance_healthy_plus_neuro_minus_ortho_boxplot.png'}")
    print(f"Saved {saved_count} test-only embedding-vs-handcrafted comparison plots.")


if __name__ == "__main__":
    main()
