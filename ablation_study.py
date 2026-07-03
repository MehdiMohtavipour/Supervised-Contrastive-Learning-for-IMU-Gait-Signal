"""
Retraining-based input-channel ablation for the time-derivative encoder.

For each input channel, this script removes that channel, trains a fresh CNN
encoder with the reduced input depth, computes train-set class centroids, and
reports pairwise plus macro ROC AUC on the held-out test set.
"""

from pathlib import Path
import argparse
import itertools
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from contrastive_time_derivative_encoder_pretrain import (
    BalancedCategoryBatchSampler,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TIME_SENSORS,
    GaitTimeDerivativeContrastiveDataset,
    TimeDerivativeContrastiveEncoder,
    class_counts_for_items,
    discover_trials,
    set_seed,
    split_trials_by_subject,
    train_epoch,
)


DEFAULT_ABLATION_OUTPUT_DIR = Path("contrastive_time_derivative_ablation_study")


class CachedSignalDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "signal": torch.from_numpy(self.signals[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }


def load_checkpoint(checkpoint_path, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_metadata(encoder_output_dir):
    metadata_path = Path(encoder_output_dir) / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def build_channel_names(sensors, acc_component):
    return [
        channel_name
        for sensor in sensors
        for component in [acc_component, "Gyr"]
        for channel_name in [
            f"{sensor}_{component}_Magnitude",
            f"{sensor}_{component}_Magnitude_Derivative",
        ]
    ]


def select_ablation_channels(channel_names, requested_channels, max_channels):
    if requested_channels:
        missing = [name for name in requested_channels if name not in channel_names]
        if missing:
            raise ValueError(f"Unknown ablation channel(s): {', '.join(missing)}")
        selected = requested_channels
    else:
        selected = list(channel_names)

    if max_channels is not None:
        if max_channels < 1:
            raise ValueError("--max-channels must be at least 1.")
        selected = selected[:max_channels]

    return [(channel_names.index(name), name) for name in selected]


def preload_dataset(dataset):
    signals = []
    labels = []
    rows = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        signals.append(sample["signal"].numpy())
        labels.append(int(sample["label"].item()))
        file_path, trial_info = dataset.items[idx]
        rows.append(
            {
                "file_path": str(file_path),
                "category": trial_info["category"],
                "disease_group": trial_info["disease_group"],
                "subgroup": trial_info["subgroup"],
                "subject_name": trial_info["subject_name"],
                "trial_name": trial_info["trial_name"],
            }
        )

    return np.stack(signals, axis=0), np.asarray(labels, dtype=int), pd.DataFrame(rows)


def extract_embeddings_from_cache(model, signals, batch_size, device):
    model.eval()
    embeddings = []

    with torch.no_grad():
        for start_idx in range(0, len(signals), batch_size):
            signal = torch.from_numpy(signals[start_idx : start_idx + batch_size]).to(device)
            batch_embeddings = model(signal, return_projection=False)
            batch_embeddings = F.normalize(batch_embeddings, dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)

    return np.vstack(embeddings)


def compute_class_centers(embeddings, labels, classes):
    centers = []
    for class_idx, class_name in enumerate(classes):
        mask = labels == class_idx
        if not np.any(mask):
            raise ValueError(f"No training samples found for class '{class_name}'.")
        centers.append(embeddings[mask].mean(axis=0))
    return np.vstack(centers)


def distances_to_centers(embeddings, centers, metric):
    if metric == "euclidean":
        return np.linalg.norm(embeddings[:, None, :] - centers[None, :, :], axis=2)
    if metric == "cosine":
        embeddings_norm = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
        centers_norm = centers / np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12, None)
        return 1.0 - np.matmul(embeddings_norm, centers_norm.T)
    raise ValueError(f"Unknown metric: {metric}")


def roc_curve_and_auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]

    if labels.size == 0 or np.unique(labels).size != 2:
        return np.nan

    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    scores = scores[order]

    positive_count = np.sum(labels == 1)
    negative_count = np.sum(labels == 0)
    threshold_indices = np.r_[np.where(np.diff(scores))[0], labels.size - 1]

    true_positives = np.cumsum(labels == 1)[threshold_indices]
    false_positives = np.cumsum(labels == 0)[threshold_indices]

    tpr = np.r_[0.0, true_positives / positive_count, 1.0]
    fpr = np.r_[0.0, false_positives / negative_count, 1.0]
    return float(np.trapezoid(tpr, fpr))


def pairwise_auc_rows(distances, labels, classes, ablation_name, channel_idx):
    rows = []
    for class_a, class_b in itertools.combinations(range(len(classes)), 2):
        pair_mask = np.isin(labels, [class_a, class_b])
        pair_labels = (labels[pair_mask] == class_b).astype(int)
        pair_distances = distances[pair_mask]

        scores = pair_distances[:, class_a] - pair_distances[:, class_b]
        raw_auc = roc_curve_and_auc(pair_labels, scores)
        flipped = bool(raw_auc < 0.5) if np.isfinite(raw_auc) else False
        auc = 1.0 - raw_auc if flipped else raw_auc

        rows.append(
            {
                "ablated_channel_index": channel_idx,
                "ablated_channel": ablation_name,
                "pair": f"{classes[class_a]}_vs_{classes[class_b]}",
                "group_a": classes[class_a],
                "group_b": classes[class_b],
                "positive_class_raw": classes[class_b],
                "n": int(pair_mask.sum()),
                "n_group_a": int(np.sum(labels[pair_mask] == class_a)),
                "n_group_b": int(np.sum(labels[pair_mask] == class_b)),
                "raw_auc": raw_auc,
                "auc": auc,
                "score_flipped_for_auc": flipped,
            }
        )
    return rows


def train_reduced_encoder(train_signals, train_labels, model_params, train_params, device, seed):
    set_seed(seed)
    model = TimeDerivativeContrastiveEncoder(
        in_channels=train_signals.shape[1],
        encoder_features=model_params["encoder_features"],
        embedding_dim=model_params["embedding_dim"],
        projection_dim=model_params["projection_dim"],
        temporal_bins=model_params["temporal_bins"],
    ).to(device)

    train_dataset = CachedSignalDataset(train_signals, train_labels)
    sampler = BalancedCategoryBatchSampler(
        labels=train_labels,
        categories_per_batch=train_params["categories_per_batch"],
        samples_per_category=train_params["samples_per_category"],
        seed=seed,
    )
    loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_params["learning_rate"],
        weight_decay=train_params["weight_decay"],
    )

    history = []
    for epoch in range(1, train_params["epochs"] + 1):
        loss = train_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            temperature=train_params["temperature"],
        )
        history.append({"epoch": epoch, "loss": loss})
        print(f"    epoch {epoch:03d}/{train_params['epochs']} contrastive_loss={loss:.4f}")

    return model, history


def evaluate_reduced_inputs(
    train_signals,
    train_labels,
    test_signals,
    test_labels,
    test_rows,
    keep_indices,
    classes,
    batch_size,
    device,
    metric,
    model_params,
    train_params,
    seed,
    ablation_name,
    ablated_channel_idx,
):
    reduced_train = train_signals[:, keep_indices, :]
    reduced_test = test_signals[:, keep_indices, :]
    model, history = train_reduced_encoder(
        train_signals=reduced_train,
        train_labels=train_labels,
        model_params=model_params,
        train_params=train_params,
        device=device,
        seed=seed,
    )

    train_embeddings = extract_embeddings_from_cache(model, reduced_train, batch_size, device)
    test_embeddings = extract_embeddings_from_cache(model, reduced_test, batch_size, device)
    centers = compute_class_centers(train_embeddings, train_labels, classes)
    distances = distances_to_centers(test_embeddings, centers, metric)
    pair_rows = pairwise_auc_rows(
        distances=distances,
        labels=test_labels,
        classes=classes,
        ablation_name=ablation_name,
        channel_idx=ablated_channel_idx,
    )

    prediction_df = test_rows.copy()
    for class_idx, class_name in enumerate(classes):
        prediction_df[f"distance_to_{class_name}"] = distances[:, class_idx]
    prediction_df["ablated_channel_index"] = ablated_channel_idx
    prediction_df["ablated_channel"] = ablation_name
    prediction_df["kept_channel_indices"] = ",".join(str(idx) for idx in keep_indices)

    return pair_rows, prediction_df, history


def checkpoint_defaults(checkpoint, metadata, args):
    checkpoint_args = checkpoint.get("args", {})
    checkpoint_metadata = checkpoint.get("metadata", metadata)

    model_params = {
        "encoder_features": int(checkpoint_metadata.get("encoder_features", checkpoint_args.get("encoder_features", 16))),
        "embedding_dim": int(checkpoint_metadata.get("embedding_dim", checkpoint_args.get("embedding_dim", 32))),
        "projection_dim": int(checkpoint_metadata.get("projection_dim", checkpoint_args.get("projection_dim", 32))),
        "temporal_bins": int(checkpoint_metadata.get("temporal_bins", checkpoint_args.get("temporal_bins", 16))),
    }
    train_params = {
        "epochs": int(args.epochs if args.epochs is not None else checkpoint_args.get("epochs", 30)),
        "categories_per_batch": int(args.categories_per_batch if args.categories_per_batch is not None else checkpoint_args.get("categories_per_batch", 3)),
        "samples_per_category": int(args.samples_per_category if args.samples_per_category is not None else checkpoint_args.get("samples_per_category", 5)),
        "learning_rate": float(args.learning_rate if args.learning_rate is not None else checkpoint_args.get("learning_rate", 1e-3)),
        "weight_decay": float(args.weight_decay if args.weight_decay is not None else checkpoint_args.get("weight_decay", 1e-3)),
        "temperature": float(args.temperature if args.temperature is not None else checkpoint_args.get("temperature", 0.1)),
    }
    return checkpoint_args, checkpoint_metadata, model_params, train_params


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrain one reduced-input encoder per removed input channel and compute macro AUC."
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Dataset data directory")
    parser.add_argument(
        "--encoder-output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory containing metadata.json and time_derivative_contrastive_encoder.pt",
    )
    parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path for hyperparameters")
    parser.add_argument("--output-dir", default=str(DEFAULT_ABLATION_OUTPUT_DIR))
    parser.add_argument("--sensors", nargs="+", default=None)
    parser.add_argument("--acc-component", default=None, choices=["FreeAcc", "Acc"])
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--training-seeds",
        nargs="+",
        type=int,
        default=None,
        help="One or more training seeds to repeat each ablation on the same data split.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=None, help="Epochs per reduced-input training run")
    parser.add_argument("--categories-per-batch", type=int, default=None)
    parser.add_argument("--samples-per-category", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Optional channel names to ablate. Defaults to every channel in metadata.",
    )
    parser.add_argument("--max-channels", type=int, default=None, help="Optional limit for quick runs.")
    return parser.parse_args()


def main():
    args = parse_args()
    encoder_output_dir = Path(args.encoder_output_dir)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else encoder_output_dir / "time_derivative_contrastive_encoder.pt"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(encoder_output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    checkpoint = load_checkpoint(checkpoint_path, device)
    checkpoint_args, checkpoint_metadata, model_params, train_params = checkpoint_defaults(
        checkpoint=checkpoint,
        metadata=metadata,
        args=args,
    )

    sensors = args.sensors or checkpoint_metadata.get("sensors") or checkpoint_args.get("sensors") or DEFAULT_TIME_SENSORS
    acc_component = args.acc_component or checkpoint_metadata.get("acc_component") or checkpoint_args.get("acc_component") or "FreeAcc"
    sequence_length = int(args.sequence_length or checkpoint_metadata.get("sequence_length") or checkpoint_args.get("sequence_length") or 1900)
    test_size = float(args.test_size if args.test_size is not None else checkpoint_metadata.get("test_size", checkpoint_args.get("test_size", 0.2)))
    seed = int(args.seed if args.seed is not None else checkpoint_args.get("seed", 42))
    training_seeds = args.training_seeds or [seed]

    trials = discover_trials(args.data_root)
    if not trials:
        raise FileNotFoundError(f"No *_processed_data.txt files found under {args.data_root}")

    classes = checkpoint_metadata.get("categories") or sorted({trial_info["category"] for _, trial_info in trials})
    class_to_index = {class_name: idx for idx, class_name in enumerate(classes)}
    train_items, test_items, _, _ = split_trials_by_subject(
        trials=trials,
        seed=seed,
        test_size=test_size,
    )

    dataset_kwargs = {
        "class_to_index": class_to_index,
        "sensors": sensors,
        "acc_component": acc_component,
        "sequence_length": sequence_length,
    }
    train_dataset = GaitTimeDerivativeContrastiveDataset(items=train_items, **dataset_kwargs)
    test_dataset = GaitTimeDerivativeContrastiveDataset(items=test_items, **dataset_kwargs)

    channel_names = checkpoint_metadata.get("channel_names") or build_channel_names(sensors, acc_component)
    time_channels = int(checkpoint_metadata.get("time_channels", len(channel_names)))
    if time_channels != len(channel_names):
        raise ValueError(f"Metadata time_channels={time_channels}, but {len(channel_names)} channel names were found.")

    selected_channels = select_ablation_channels(
        channel_names=channel_names,
        requested_channels=args.channels,
        max_channels=args.max_channels,
    )

    print(f"Preloading train tensors: {len(train_dataset)} trials")
    train_signals, train_labels, _ = preload_dataset(train_dataset)
    print(f"Preloading test tensors: {len(test_dataset)} trials")
    test_signals, test_labels, test_rows = preload_dataset(test_dataset)
    print(f"Epochs per ablation: {train_params['epochs']}")
    print(f"Training seeds: {', '.join(str(value) for value in training_seeds)}")

    all_pair_rows = []
    all_prediction_rows = []
    all_history_rows = []
    for channel_idx, channel_name in selected_channels:
        keep_indices = [idx for idx in range(time_channels) if idx != channel_idx]
        for training_seed in training_seeds:
            run_seed = training_seed + channel_idx + 1
            print(f"Training without channel {channel_idx:02d}: {channel_name} seed={run_seed}")
            pair_rows, prediction_df, history = evaluate_reduced_inputs(
                train_signals=train_signals,
                train_labels=train_labels,
                test_signals=test_signals,
                test_labels=test_labels,
                test_rows=test_rows,
                keep_indices=keep_indices,
                classes=classes,
                batch_size=args.batch_size,
                device=device,
                metric=args.metric,
                model_params=model_params,
                train_params=train_params,
                seed=run_seed,
                ablation_name=channel_name,
                ablated_channel_idx=channel_idx,
            )
            for row in pair_rows:
                row["training_seed"] = run_seed
            prediction_df["training_seed"] = run_seed
            all_pair_rows.extend(pair_rows)
            all_prediction_rows.append(prediction_df)
            for row in history:
                all_history_rows.append(
                    {
                        "ablated_channel_index": channel_idx,
                        "ablated_channel": channel_name,
                        "training_seed": run_seed,
                        **row,
                    }
                )

    pair_auc_df = pd.DataFrame(all_pair_rows)
    macro_auc_by_seed_df = (
        pair_auc_df.groupby(["ablated_channel_index", "ablated_channel", "training_seed"], as_index=False)
        .agg(
            macro_auc=("auc", "mean"),
            macro_raw_auc=("raw_auc", "mean"),
            n_pairs=("pair", "nunique"),
            total_pair_n=("n", "sum"),
        )
        .sort_values("macro_auc", ascending=True)
    )
    macro_auc_by_seed_df["macro_auc_percent"] = macro_auc_by_seed_df["macro_auc"] * 100.0
    macro_auc_df = (
        macro_auc_by_seed_df.groupby(["ablated_channel_index", "ablated_channel"], as_index=False)
        .agg(
            macro_auc_mean=("macro_auc", "mean"),
            macro_auc_std=("macro_auc", "std"),
            macro_auc_min=("macro_auc", "min"),
            macro_auc_max=("macro_auc", "max"),
            n_training_seeds=("training_seed", "nunique"),
        )
        .sort_values("macro_auc_mean", ascending=True)
    )
    macro_auc_df["macro_auc_mean_percent"] = macro_auc_df["macro_auc_mean"] * 100.0
    macro_auc_df["macro_auc_std_percent"] = macro_auc_df["macro_auc_std"].fillna(0.0) * 100.0

    pair_auc_path = output_dir / "pairwise_auc_by_ablation.csv"
    macro_auc_by_seed_path = output_dir / "macro_auc_by_ablation_seed.csv"
    macro_auc_path = output_dir / "macro_auc_by_ablation.csv"
    prediction_path = output_dir / "test_center_distances_by_ablation.csv"
    history_path = output_dir / "retraining_history_by_ablation.csv"
    summary_path = output_dir / "ablation_summary.txt"

    pair_auc_df.to_csv(pair_auc_path, index=False)
    macro_auc_by_seed_df.to_csv(macro_auc_by_seed_path, index=False)
    macro_auc_df.to_csv(macro_auc_path, index=False)
    pd.concat(all_prediction_rows, ignore_index=True).to_csv(prediction_path, index=False)
    pd.DataFrame(all_history_rows).to_csv(history_path, index=False)

    summary_lines = [
        "Retraining-based input-channel ablation macro AUC",
        f"checkpoint hyperparameters loaded from: {checkpoint_path}",
        f"metric: {args.metric}",
        f"epochs_per_ablation: {train_params['epochs']}",
        f"training_seeds: {training_seeds}",
        f"train_class_counts: {class_counts_for_items(train_items, classes)}",
        f"test_class_counts: {class_counts_for_items(test_items, classes)}",
        "",
        macro_auc_df.to_string(index=False),
        "",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\nSaved pairwise AUC table: {pair_auc_path}")
    print(f"Saved per-seed macro AUC table: {macro_auc_by_seed_path}")
    print(f"Saved macro AUC table: {macro_auc_path}")
    print(f"Saved test distances: {prediction_path}")
    print(f"Saved retraining history: {history_path}")
    print(f"Saved summary: {summary_path}")
    print("\nMacro AUC by removed input after retraining:")
    print(macro_auc_df.to_string(index=False))


if __name__ == "__main__":
    main()
