"""
Retraining-based sensor ablation for the time-derivative encoder.

Each condition keeps only the requested sensors, trains a fresh CNN encoder with
that reduced input depth, computes train-set class centroids, and reports
pairwise plus macro ROC AUC on the held-out test set.
"""

from pathlib import Path
import argparse

import pandas as pd
import torch

from ablation_study import (
    evaluate_reduced_inputs,
    load_checkpoint,
    load_metadata,
    preload_dataset,
    checkpoint_defaults,
)
from contrastive_time_derivative_encoder_pretrain import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TIME_SENSORS,
    GaitTimeDerivativeContrastiveDataset,
    class_counts_for_items,
    discover_trials,
    split_trials_by_subject,
)


DEFAULT_OUTPUT_DIR_SENSOR = Path("contrastive_time_derivative_sensor_ablation_study")
DEFAULT_SENSOR_CONDITIONS = [
    ("train_RF", ["RF"], True),
    ("train_LF", ["LF"], True),
    ("train_HE", ["HE"], True),
    ("train_LB", ["LB"], True),
    ("train_HE_LB_LF_RF_no_derivative", ["HE", "LB", "LF", "RF"], False),
]


def channel_indices_for_kept_sensors(channel_names, sensors_to_keep, include_derivatives=True):
    selected = []
    sensors_to_keep = set(sensors_to_keep)
    for idx, channel_name in enumerate(channel_names):
        sensor_name = channel_name.split("_", 1)[0]
        if sensor_name in sensors_to_keep and (include_derivatives or "Derivative" not in channel_name):
            selected.append(idx)
    if not selected:
        raise ValueError(f"No channels matched sensors: {', '.join(sorted(sensors_to_keep))}")
    return selected


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrain one reduced-input encoder per kept sensor set and compute macro AUC."
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Dataset data directory")
    parser.add_argument(
        "--encoder-output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory containing metadata.json and time_derivative_contrastive_encoder.pt",
    )
    parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path for hyperparameters")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR_SENSOR))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=None, help="Epochs per reduced-input training run")
    parser.add_argument("--categories-per-batch", type=int, default=None)
    parser.add_argument("--samples-per-category", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--seed", type=int, default=45)
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

    sensors = checkpoint_metadata.get("sensors") or checkpoint_args.get("sensors") or DEFAULT_TIME_SENSORS
    acc_component = checkpoint_metadata.get("acc_component") or checkpoint_args.get("acc_component") or "FreeAcc"
    sequence_length = int(checkpoint_metadata.get("sequence_length") or checkpoint_args.get("sequence_length") or 1900)
    test_size = float(args.test_size if args.test_size is not None else checkpoint_metadata.get("test_size", checkpoint_args.get("test_size", 0.2)))
    seed = int(args.seed if args.seed is not None else checkpoint_args.get("seed", 42))

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

    channel_names = checkpoint_metadata["channel_names"]

    print(f"Preloading train tensors: {len(train_dataset)} trials")
    train_signals, train_labels, _ = preload_dataset(train_dataset)
    print(f"Preloading test tensors: {len(test_dataset)} trials")
    test_signals, test_labels, test_rows = preload_dataset(test_dataset)
    print(f"Epochs per condition: {train_params['epochs']}")
    print(f"Training seed: {seed}")

    all_pair_rows = []
    all_prediction_rows = []
    all_history_rows = []
    for condition_idx, (condition_name, kept_sensors, include_derivatives) in enumerate(DEFAULT_SENSOR_CONDITIONS):
        keep_indices = channel_indices_for_kept_sensors(
            channel_names=channel_names,
            sensors_to_keep=kept_sensors,
            include_derivatives=include_derivatives,
        )
        run_seed = seed + 100 + condition_idx
        print(
            f"Training condition {condition_name}: sensors={','.join(kept_sensors)} "
            f"include_derivatives={include_derivatives} "
            f"channels={','.join(str(idx) for idx in keep_indices)} seed={run_seed}"
        )
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
            ablation_name=condition_name,
            ablated_channel_idx=-1,
        )
        for row in pair_rows:
            row["kept_sensors"] = ",".join(kept_sensors)
            row["include_derivatives"] = include_derivatives
            row["kept_channel_indices"] = ",".join(str(idx) for idx in keep_indices)
            row["training_seed"] = run_seed
        prediction_df["condition"] = condition_name
        prediction_df["kept_sensors"] = ",".join(kept_sensors)
        prediction_df["include_derivatives"] = include_derivatives
        prediction_df["kept_channel_indices"] = ",".join(str(idx) for idx in keep_indices)
        prediction_df["training_seed"] = run_seed

        all_pair_rows.extend(pair_rows)
        all_prediction_rows.append(prediction_df)
        for row in history:
            all_history_rows.append(
                {
                    "condition": condition_name,
                    "kept_sensors": ",".join(kept_sensors),
                    "include_derivatives": include_derivatives,
                    "training_seed": run_seed,
                    **row,
                }
            )

    pair_auc_df = pd.DataFrame(all_pair_rows)
    macro_auc_df = (
        pair_auc_df.groupby(
            [
                "ablated_channel",
                "kept_sensors",
                "include_derivatives",
                "kept_channel_indices",
                "training_seed",
            ],
            as_index=False,
        )
        .agg(
            macro_auc=("auc", "mean"),
            macro_raw_auc=("raw_auc", "mean"),
            n_pairs=("pair", "nunique"),
            total_pair_n=("n", "sum"),
        )
        .rename(columns={"ablated_channel": "condition"})
        .sort_values("macro_auc", ascending=False)
    )
    macro_auc_df["macro_auc_percent"] = macro_auc_df["macro_auc"] * 100.0

    pair_auc_path = output_dir / "pairwise_auc_by_sensor_retraining.csv"
    macro_auc_path = output_dir / "macro_auc_by_sensor_retraining.csv"
    prediction_path = output_dir / "test_center_distances_by_sensor_retraining.csv"
    history_path = output_dir / "retraining_history_by_sensor_condition.csv"
    summary_path = output_dir / "sensor_retraining_ablation_summary.txt"

    pair_auc_df.to_csv(pair_auc_path, index=False)
    macro_auc_df.to_csv(macro_auc_path, index=False)
    pd.concat(all_prediction_rows, ignore_index=True).to_csv(prediction_path, index=False)
    pd.DataFrame(all_history_rows).to_csv(history_path, index=False)

    summary_lines = [
        "Retraining-based sensor ablation macro AUC",
        f"checkpoint hyperparameters loaded from: {checkpoint_path}",
        f"metric: {args.metric}",
        f"epochs_per_condition: {train_params['epochs']}",
        f"training_seed: {seed}",
        f"train_class_counts: {class_counts_for_items(train_items, classes)}",
        f"test_class_counts: {class_counts_for_items(test_items, classes)}",
        "",
        macro_auc_df.to_string(index=False),
        "",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\nSaved pairwise AUC table: {pair_auc_path}")
    print(f"Saved macro AUC table: {macro_auc_path}")
    print(f"Saved test distances: {prediction_path}")
    print(f"Saved retraining history: {history_path}")
    print(f"Saved summary: {summary_path}")
    print("\nMacro AUC by sensor training condition:")
    print(macro_auc_df.to_string(index=False))


if __name__ == "__main__":
    main()
