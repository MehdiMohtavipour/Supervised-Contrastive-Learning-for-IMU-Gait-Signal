"""
Pretrain a time-domain gait encoder with magnitude and derivative streams.

Positive pairs are trials from the same clinical category. Negative pairs are
trials from different clinical categories. Each trial uses the usual time-domain
accelerometer and gyroscope magnitude channels for HE, LB, LF, and RF, plus a
first-derivative channel for each magnitude channel.
"""

from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATA_ROOT = Path(
    r"C:\Users\Mehdi\Desktop\Gait_Pattern\Clinical_Dataset\dataset\data"
)
DEFAULT_OUTPUT_DIR = Path("contrastive_time_derivative_encoder_outputs")
DEFAULT_TIME_SENSORS = ["HE", "LB", "LF", "RF"]
#DEFAULT_TIME_SENSORS = ["LF", "RF"]
CLINICAL_CATEGORIES = ["healthy", "neuro", "ortho"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_trial_info(file_path, data_root):
    rel_parts = file_path.relative_to(data_root).parts
    disease_group = rel_parts[0] if len(rel_parts) > 0 else "unknown"
    subgroup = rel_parts[1] if len(rel_parts) > 1 else "unknown"
    subject_name = rel_parts[2] if len(rel_parts) > 2 else file_path.parent.parent.name
    trial_name = rel_parts[3] if len(rel_parts) > 3 else file_path.parent.name
    category = disease_group

    return {
        "category": category,
        "disease_group": disease_group,
        "subgroup": subgroup,
        "subject_name": subject_name,
        "trial_name": trial_name,
    }


def discover_trials(data_root):
    data_root = Path(data_root)
    trials = []
    for file_path in sorted(data_root.rglob("*_processed_data.txt")):
        trial_info = parse_trial_info(file_path, data_root)
        if trial_info["category"] not in CLINICAL_CATEGORIES:
            continue
        trials.append((file_path, trial_info))
    return trials


def sensor_magnitude(df, sensor, component):
    axis_cols = [f"{sensor}_{component}_{axis}" for axis in ["X", "Y", "Z"]]
    if any(col not in df.columns for col in axis_cols):
        return None

    axes = [
        pd.to_numeric(df[col], errors="coerce")
        .astype(float)
        .interpolate(limit_direction="both")
        for col in axis_cols
    ]
    return np.sqrt(sum(axis_values.to_numpy(dtype=float) ** 2 for axis_values in axes))


def resize_signal(signal, target_length):
    if signal is None or len(signal) == 0 or np.isnan(signal).all():
        return np.zeros(target_length, dtype=np.float32)

    signal = pd.Series(signal, dtype="float64").interpolate(limit_direction="both")
    values = signal.to_numpy(dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    if len(values) == target_length:
        return values.astype(np.float32)
    return resample(values, target_length).astype(np.float32)


def standardize_per_channel(stream):
    means = stream.mean(axis=1, keepdims=True)
    stds = stream.std(axis=1, keepdims=True)
    return ((stream - means) / (stds + 1e-6)).astype(np.float32)


def build_sensor_stream(df, sensor, component, sequence_length):
    channel = resize_signal(sensor_magnitude(df, sensor, component), sequence_length)
    return standardize_per_channel(channel.reshape(1, -1))


def build_derivative_stream(stream):
    derivatives = []
    for channel in stream:
        derivatives.append(np.gradient(channel).astype(np.float32))
    return standardize_per_channel(np.stack(derivatives, axis=0))


def subject_key(trial_info):
    return (
        trial_info["category"],
        trial_info["disease_group"],
        trial_info["subgroup"],
        trial_info["subject_name"],
    )


def flatten_subjects(subjects):
    return [item for subject in subjects for item in subject["items"]]


def split_trials_by_subject(trials, seed, test_size):
    subject_trials = {}
    for file_path, trial_info in trials:
        key = subject_key(trial_info)
        subject_trials.setdefault(key, []).append((file_path, trial_info))

    subjects = [
        {
            "key": key,
            "category": key[0],
            "items": items,
        }
        for key, items in sorted(subject_trials.items())
    ]
    subject_labels = [subject["category"] for subject in subjects]

    train_subjects, test_subjects = train_test_split(
        subjects,
        test_size=test_size,
        random_state=seed,
        stratify=subject_labels,
    )

    return (
        flatten_subjects(train_subjects),
        flatten_subjects(test_subjects),
        train_subjects,
        test_subjects,
    )


def save_subject_split_csv(output_path, train_subjects, test_subjects):
    rows = []
    for split_name, subjects in [("train", train_subjects), ("test", test_subjects)]:
        for subject in subjects:
            category, disease_group, subgroup, subject_name = subject["key"]
            rows.append(
                {
                    "Split": split_name,
                    "Clinical_Category": category,
                    "Disease_Group": disease_group,
                    "Subgroup": subgroup,
                    "Subject_Name": subject_name,
                    "Trial_Count": len(subject["items"]),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def class_counts_for_items(items, categories):
    labels = [trial_info["category"] for _, trial_info in items]
    return {category: labels.count(category) for category in categories}


class BalancedCategoryBatchSampler:
    def __init__(self, labels, categories_per_batch, samples_per_category, seed=42):
        self.labels = np.asarray(labels)
        self.categories_per_batch = categories_per_batch
        self.samples_per_category = samples_per_category
        self.seed = seed
        self.epoch = 0
        self.label_to_indices = {
            label: np.flatnonzero(self.labels == label).tolist()
            for label in sorted(set(self.labels.tolist()))
        }
        self.labels_with_enough_samples = [
            label
            for label, indices in self.label_to_indices.items()
            if len(indices) >= 2
        ]
        if len(self.labels_with_enough_samples) < 2:
            raise ValueError("Need at least two categories with at least two trials each.")

        self.batch_size = self.categories_per_batch * self.samples_per_category
        self.num_batches = max(1, len(self.labels) // self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        for _ in range(self.num_batches):
            labels = rng.sample(
                self.labels_with_enough_samples,
                min(self.categories_per_batch, len(self.labels_with_enough_samples)),
            )
            batch = []
            for label in labels:
                indices = self.label_to_indices[label]
                replace = len(indices) < self.samples_per_category
                if replace:
                    batch.extend(rng.choice(indices) for _ in range(self.samples_per_category))
                else:
                    batch.extend(rng.sample(indices, self.samples_per_category))
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


def supervised_contrastive_loss(projections, labels, temperature):
    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).float().to(projections.device)
    self_mask = torch.eye(labels.size(0), device=projections.device)
    positive_mask = positive_mask * (1.0 - self_mask)

    logits = torch.matmul(projections, projections.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits_mask = 1.0 - self_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positive_counts = positive_mask.sum(dim=1)
    valid_rows = positive_counts > 0
    if not valid_rows.any():
        return projections.new_tensor(0.0, requires_grad=True)

    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_counts.clamp_min(1.0)
    return -mean_log_prob_pos[valid_rows].mean()


class ConvBranch(nn.Module):
    def __init__(self, in_channels, output_features=16, temporal_bins=8):
        super().__init__()
        self.temporal_bins = temporal_bins
        first_channels = max(output_features // 4, 4)
        second_channels = max(output_features // 2, 8)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, first_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(first_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(first_channels, second_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(second_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(second_channels, output_features, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_features),
            nn.ReLU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(temporal_bins)
        self.max_pool = nn.AdaptiveMaxPool1d(temporal_bins)
        self.output_dim = output_features * temporal_bins * 2

    def forward(self, x):
        features = self.feature_extractor(x)
        pooled = torch.cat(
            [
                self.avg_pool(features),
                self.max_pool(features),
            ],
            dim=1,
        )
        return torch.flatten(pooled, start_dim=1)


def count_trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def add_split_column(df, split_name):
    df = df.copy()
    df.insert(0, "split", split_name)
    return df


def save_checkpoint(output_path, model, metadata, args):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
            "args": vars(args),
        },
        output_path,
    )


def make_time_derivative_sample(file_path, sensors, acc_component, sequence_length):
    df = pd.read_csv(file_path, delimiter="\t")
    channels = []
    channel_names = []
    for sensor in sensors:
        for component in [acc_component, "Gyr"]:
            magnitude_stream = build_sensor_stream(df, sensor, component, sequence_length)
            derivative_stream = build_derivative_stream(magnitude_stream)
            channels.extend([magnitude_stream, derivative_stream])
            channel_names.extend(
                [
                    f"{sensor}_{component}_Magnitude",
                    f"{sensor}_{component}_Magnitude_Derivative",
                ]
            )

    return {
        "signal": np.concatenate(channels, axis=0).astype(np.float32),
        "channel_names": channel_names,
    }


class GaitTimeDerivativeContrastiveDataset(Dataset):
    def __init__(
        self,
        items,
        class_to_index,
        sensors,
        acc_component,
        sequence_length,
    ):
        self.items = items
        self.class_to_index = class_to_index
        self.sensors = sensors
        self.acc_component = acc_component
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        file_path, trial_info = self.items[idx]
        sample = make_time_derivative_sample(
            file_path=file_path,
            sensors=self.sensors,
            acc_component=self.acc_component,
            sequence_length=self.sequence_length,
        )
        label = self.class_to_index[trial_info["category"]]
        return {
            "signal": torch.from_numpy(sample["signal"]),
            "label": torch.tensor(label, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }


class TimeDerivativeContrastiveEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        encoder_features=16,
        embedding_dim=32,
        projection_dim=32,
        temporal_bins=8,
    ):
        super().__init__()
        self.backbone = ConvBranch(in_channels, encoder_features, temporal_bins)
        self.embedding_head = nn.Sequential(
            nn.Linear(self.backbone.output_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim),
        )

    def forward(self, signal, return_projection=True):
        features = self.backbone(signal)
        embeddings = self.embedding_head(features)
        if not return_projection:
            return embeddings

        projections = self.projection_head(embeddings)
        return embeddings, F.normalize(projections, dim=1)


def train_epoch(model, loader, optimizer, device, temperature):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        signal = batch["signal"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        _, projections = model(signal)
        loss = supervised_contrastive_loss(projections, labels, temperature)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    return total_loss / max(total_samples, 1)


def extract_embeddings(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in loader:
            signal = batch["signal"].to(device)
            embeddings = model(signal, return_projection=False)
            embeddings = F.normalize(embeddings, dim=1).cpu().numpy()

            for row_idx, dataset_idx in enumerate(batch["index"].numpy().tolist()):
                file_path, trial_info = dataset.items[dataset_idx]
                row = {
                    "file_path": str(file_path),
                    "category": trial_info["category"],
                    "disease_group": trial_info["disease_group"],
                    "subgroup": trial_info["subgroup"],
                    "subject_name": trial_info["subject_name"],
                    "trial_name": trial_info["trial_name"],
                }
                for dim_idx, value in enumerate(embeddings[row_idx]):
                    row[f"embedding_{dim_idx:03d}"] = value
                rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain a time-domain CNN encoder with magnitude and derivative streams."
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Dataset data directory")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for checkpoints and embeddings")
    parser.add_argument(
        "--sensors",
        nargs="+",
        default=DEFAULT_TIME_SENSORS,
        help="Sensor prefixes to include in the time-domain input",
    )
    parser.add_argument("--acc-component", default="FreeAcc", choices=["FreeAcc", "Acc"], help="Accelerometer component")
    parser.add_argument("--sequence-length", type=int, default=1900, help="Resampled time-series length")
    parser.add_argument("--epochs", type=int, default=30, help="Pretraining epochs")
    parser.add_argument("--categories-per-batch", type=int, default=3, help="Clinical categories per contrastive batch")
    parser.add_argument("--samples-per-category", type=int, default=5, help="Trials per category in each contrastive batch")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Adam weight decay")
    parser.add_argument("--temperature", type=float, default=0.1, help="Contrastive loss temperature")
    parser.add_argument("--encoder-features", type=int, default=16, help="CNN backbone output features")
    parser.add_argument(
        "--temporal-bins",
        type=int,
        default=16,
        help="Temporal bins kept per CNN feature map before flattening",
    )
    parser.add_argument("--embedding-dim", type=int, default=32, help="Saved embedding dimension")
    parser.add_argument("--projection-dim", type=int, default=32, help="Contrastive projection dimension")
    parser.add_argument("--embedding-batch-size", type=int, default=64, help="Batch size for embedding extraction")
    parser.add_argument("--test-size", type=float, default=0.20, help="Held-out subject fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    if args.encoder_features < 4:
        raise ValueError("--encoder-features must be at least 4")
    if args.temporal_bins < 1:
        raise ValueError("--temporal-bins must be at least 1")
    if args.embedding_dim < 1:
        raise ValueError("--embedding-dim must be at least 1")
    if args.projection_dim < 1:
        raise ValueError("--projection-dim must be at least 1")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = discover_trials(args.data_root)
    if not trials:
        raise FileNotFoundError(f"No *_processed_data.txt files found under {args.data_root}")

    categories = sorted({trial_info["category"] for _, trial_info in trials})
    class_to_index = {category: idx for idx, category in enumerate(categories)}
    class_counts = class_counts_for_items(trials, categories)

    train_items, test_items, train_subjects, test_subjects = split_trials_by_subject(
        trials=trials,
        seed=args.seed,
        test_size=args.test_size,
    )
    train_labels = [class_to_index[trial_info["category"]] for _, trial_info in train_items]
    train_class_counts = class_counts_for_items(train_items, categories)
    test_class_counts = class_counts_for_items(test_items, categories)
    save_subject_split_csv(output_dir / "subject_split.csv", train_subjects, test_subjects)

    dataset_kwargs = {
        "class_to_index": class_to_index,
        "sensors": args.sensors,
        "acc_component": args.acc_component,
        "sequence_length": args.sequence_length,
    }
    train_dataset = GaitTimeDerivativeContrastiveDataset(items=train_items, **dataset_kwargs)
    test_dataset = GaitTimeDerivativeContrastiveDataset(items=test_items, **dataset_kwargs)
    sampler = BalancedCategoryBatchSampler(
        labels=train_labels,
        categories_per_batch=args.categories_per_batch,
        samples_per_category=args.samples_per_category,
        seed=args.seed,
    )
    loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0)

    base_channels = len(args.sensors) * 2
    derivative_channels = base_channels
    time_channels = base_channels + derivative_channels
    channel_names = [
        channel_name
        for sensor in args.sensors
        for component in [args.acc_component, "Gyr"]
        for channel_name in [
            f"{sensor}_{component}_Magnitude",
            f"{sensor}_{component}_Magnitude_Derivative",
        ]
    ]

    metadata = {
        "categories": categories,
        "class_to_index": class_to_index,
        "class_counts": class_counts,
        "train_class_counts": train_class_counts,
        "test_class_counts": test_class_counts,
        "sensors": args.sensors,
        "time_channels": time_channels,
        "base_magnitude_channels": base_channels,
        "derivative_channels": derivative_channels,
        "channel_names": channel_names,
        "encoder_features": args.encoder_features,
        "temporal_bins": args.temporal_bins,
        "backbone_output_features": args.encoder_features * args.temporal_bins * 2,
        "temporal_pooling": "adaptive_avg_and_max_pool_per_bin_then_flatten",
        "embedding_dim": args.embedding_dim,
        "projection_dim": args.projection_dim,
        "channels_per_sensor": [args.acc_component, "Gyr"],
        "extra_streams": ["first_derivative_of_each_magnitude_channel"],
        "acc_component": args.acc_component,
        "sequence_length": args.sequence_length,
        "split_grouping": "subject",
        "train_size": 1.0 - args.test_size,
        "test_size": args.test_size,
        "loss": "supervised_contrastive",
        "positive_pairs": "same clinical category",
        "negative_pairs": "different clinical category",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Found {len(trials)} trials across {len(categories)} categories.")
    for category in categories:
        print(f"  {category}: {class_counts[category]}")
    print(f"Split sizes: train={len(train_items)}, test={len(test_items)}")
    print(f"Time-domain input shape per trial: ({time_channels}, {args.sequence_length})")
    print(f"  magnitude channels: {base_channels}")
    print(f"  derivative channels: {derivative_channels}")
    print(f"Contrastive batch size: {sampler.batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeDerivativeContrastiveEncoder(
        in_channels=time_channels,
        encoder_features=args.encoder_features,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        temporal_bins=args.temporal_bins,
    ).to(device)
    print(f"Trainable model parameters: {count_trainable_parameters(model):,}")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    checkpoint_path = output_dir / "time_derivative_contrastive_encoder.pt"

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            temperature=args.temperature,
        )
        history.append({"epoch": epoch, "loss": loss})
        print(f"Epoch {epoch:03d}/{args.epochs} contrastive_loss={loss:.4f}")

        save_checkpoint(checkpoint_path, model, metadata, args)

    pd.DataFrame(history).to_csv(output_dir / "pretraining_history.csv", index=False)
    train_embeddings = add_split_column(
        extract_embeddings(
            model=model,
            dataset=train_dataset,
            batch_size=args.embedding_batch_size,
            device=device,
        ),
        "train",
    )
    test_embeddings = add_split_column(
        extract_embeddings(
            model=model,
            dataset=test_dataset,
            batch_size=args.embedding_batch_size,
            device=device,
        ),
        "test",
    )
    all_embeddings = pd.concat([train_embeddings, test_embeddings], ignore_index=True)

    train_embeddings.to_csv(output_dir / "train_embeddings.csv", index=False)
    test_embeddings.to_csv(output_dir / "test_embeddings.csv", index=False)
    all_embeddings.to_csv(output_dir / "all_embeddings.csv", index=False)

    # Backward-compatible alias for plotting scripts that expect this filename.
    all_embeddings.to_csv(output_dir / "trial_embeddings.csv", index=False)

    print(f"Saved encoder checkpoint: {checkpoint_path}")
    print(f"Saved train embeddings: {output_dir / 'train_embeddings.csv'}")
    print(f"Saved test embeddings: {output_dir / 'test_embeddings.csv'}")
    print(f"Saved all embeddings: {output_dir / 'all_embeddings.csv'}")


if __name__ == "__main__":
    main()
