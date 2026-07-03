"""
Plot contrastive time-domain derivative encoder embeddings in 2D.

This script reads embeddings saved by
contrastive_time_derivative_encoder_pretrain.py and saves category-colored PCA
or t-SNE scatter plots for clustering review.
"""

from pathlib import Path
import argparse

from plot_contrastive_embeddings import (
    embedding_csv_paths,
    load_embeddings,
    plot_embedding,
    reduce_embeddings,
)


DEFAULT_EMBEDDINGS_DIR = Path("contrastive_time_derivative_encoder_outputs")
DEFAULT_OUTPUT_DIR = Path("contrastive_time_derivative_embedding_plots")


def main():
    parser = argparse.ArgumentParser(
        description="Plot contrastive time-domain derivative encoder embeddings in 2D."
    )
    parser.add_argument(
        "--embeddings-csv",
        default=None,
        help="Optional single embeddings CSV path. If omitted, train/test/all CSVs are plotted.",
    )
    parser.add_argument(
        "--embeddings-dir",
        default=str(DEFAULT_EMBEDDINGS_DIR),
        help="Directory containing train_embeddings.csv, test_embeddings.csv, and all_embeddings.csv",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for plots")
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca", help="2D reduction method")
    parser.add_argument("--color-by", default="category", help="Metadata column used for point colors")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in embedding_csv_paths(args.embeddings_csv, args.embeddings_dir):
        metadata, embeddings = load_embeddings(csv_path)
        reduced = reduce_embeddings(
            embeddings=embeddings,
            method=args.method,
            seed=args.seed,
            perplexity=args.perplexity,
        )

        split_name = csv_path.stem.replace("_embeddings", "")
        output_path = output_dir / f"{args.method}_{args.color_by}_{split_name}_embeddings.png"
        plot_embedding(
            metadata=metadata,
            reduced=reduced,
            color_by=args.color_by,
            output_path=output_path,
            title=f"{args.method.upper()} of {split_name} contrastive time-derivative embeddings",
        )
        print(f"Saved embedding plot: {output_path}")


if __name__ == "__main__":
    main()
