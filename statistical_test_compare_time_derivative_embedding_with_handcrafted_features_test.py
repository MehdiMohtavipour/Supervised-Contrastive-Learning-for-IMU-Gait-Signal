"""
Run one-way ANOVA tests for the held-out test split.

The implementation reuses the training statistical-test helpers and switches
only the split and default output path.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from compare_time_derivative_embedding_with_handcrafted_features import (
    DEFAULT_CENTROID_CLASSES,
    DEFAULT_EMBEDDINGS_CSV,
    DEFAULT_FEATURES_DIR,
    DEFAULT_GROUP_ORDER,
    DEFAULT_METRICS,
    load_embedding_centroid_distance_biomarker,
)
from statistical_test_compare_time_derivative_embedding_with_handcrafted_features_train import (
    collect_rows,
)


DEFAULT_OUTPUT_DIR = Path("time_derivative_embedding_handcrafted_anova_test")


def main():
    parser = argparse.ArgumentParser(
        description="Run ANOVA tests for test biomarker and handcrafted features."
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
        split_name="test",
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
    results_path = output_dir / "test_anova_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"Saved test ANOVA results: {results_path}")
    if not results_df.empty:
        print(results_df[["source", "feature", "p_value", "eta_squared", "eta_squared_ci_low", "eta_squared_ci_high"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
