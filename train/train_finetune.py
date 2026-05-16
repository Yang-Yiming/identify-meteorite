import math
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from augmentations import apply_cutmix, apply_mixup, build_soft_targets, soft_target_cross_entropy
from calibration import (
    POSITIVE_LABEL,
    apply_bayes_prior_correction,
    build_class_weights_for_target_prior,
    build_target_priors_from_ratio,
    compute_binary_f1,
    compute_class_priors,
    search_best_threshold,
    summarize_priors,
)
from data import (
    MeteoriteDataset,
    build_image_index,
    build_mask_image_index,
    build_pseudo_labeled_dataframe,
    filter_dataframe_by_skip_ids,
    rebalance_binary_subset_to_ratio,
    stratified_split,
    stratified_subsplit,
)
from modeling import (
    ConvNeXtClassifier,
    build_transforms,
    create_optimizer,
    freeze_backbone_for_head_only,
    resolve_backbone_data_settings,
    unfreeze_backbone_all,
)
from utils import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    normalize_image_size,
    normalize_stats,
    parse_args,
    save_json,
    set_seed,
)
from wandb_utils import finish_wandb_run, init_wandb_run, update_wandb_summary


def compute_grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm = float(parameter.grad.detach().norm(2).item())
        total += grad_norm * grad_norm
    return math.sqrt(total)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_weights: torch.Tensor,
    optimizer: Optional[torch.optim.Optimizer] = None,
    cutmix_alpha: float = 0.0,
    cutmix_prob: float = 0.0,
    mixup_alpha: float = 0.0,
    mixup_prob: float = 0.0,
    label_smoothing: float = 0.0,
    wandb_run=None,
    stage: Optional[str] = None,
    epoch: Optional[int] = None,
    global_step: int = 0,
    log_interval: int = 0,
) -> Dict[str, object]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    cutmix_batches = 0
    mixup_batches = 0
    grad_norm_sum = 0.0
    grad_norm_steps = 0
    collected_labels: List[torch.Tensor] = []
    collected_probabilities: List[torch.Tensor] = []

    autocast_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=autocast_enabled) if is_train else None

    for pixel_values, labels, sample_weights in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        sample_weights = sample_weights.to(device, non_blocking=True)

        soft_targets = build_soft_targets(labels, num_classes=num_classes, smoothing=label_smoothing)
        if is_train:
            mixed_batch = apply_cutmix(
                pixel_values,
                labels,
                sample_weights,
                num_classes=num_classes,
                alpha=cutmix_alpha,
                probability=cutmix_prob,
                smoothing=label_smoothing,
            )
            pixel_values = mixed_batch.pixel_values
            soft_targets = mixed_batch.mixed_labels
            sample_weights = mixed_batch.sample_weights
            if mixed_batch.applied:
                cutmix_batches += 1

            mixed_batch2 = apply_mixup(
                pixel_values,
                labels,
                sample_weights,
                num_classes=num_classes,
                alpha=mixup_alpha,
                probability=mixup_prob,
                smoothing=label_smoothing,
            )
            pixel_values = mixed_batch2.pixel_values
            soft_targets = mixed_batch2.mixed_labels
            sample_weights = mixed_batch2.sample_weights
            if mixed_batch2.applied:
                mixup_batches += 1

            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                logits = model(pixel_values)
                loss = soft_target_cross_entropy(
                    logits,
                    soft_targets,
                    class_weights=class_weights,
                    sample_weights=sample_weights,
                )

            if is_train:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                batch_grad_norm = compute_grad_norm(model.parameters())
                grad_norm_sum += batch_grad_norm
                grad_norm_steps += 1
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
                if wandb_run is not None and log_interval > 0 and global_step % log_interval == 0:
                    wandb_run.log(
                        {
                            "epoch": int(epoch) if epoch is not None else 0,
                            "stage": stage,
                            "step": int(global_step),
                            "step/train/loss": float(loss.item()),
                            "step/train/grad_norm": float(batch_grad_norm),
                        },
                        step=global_step,
                    )

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

        if not is_train:
            collected_labels.append(labels.detach().cpu())
            collected_probabilities.append(torch.softmax(logits.detach(), dim=1)[:, POSITIVE_LABEL].cpu())

    metrics: Dict[str, object] = {
        "loss": total_loss / max(1, total_samples),
        "accuracy": total_correct / max(1, total_samples),
    }
    if is_train:
        metrics["cutmix_batches"] = float(cutmix_batches)
        metrics["mixup_batches"] = float(mixup_batches)
        metrics["grad_norm"] = grad_norm_sum / max(1, grad_norm_steps)
        metrics["global_step"] = global_step
    else:
        metrics["labels"] = torch.cat(collected_labels) if collected_labels else torch.empty(0, dtype=torch.long)
        metrics["prob_pos"] = (
            torch.cat(collected_probabilities) if collected_probabilities else torch.empty(0, dtype=torch.float32)
        )
    return metrics


def maybe_apply_bayes_correction(
    probabilities: torch.Tensor,
    train_priors: Dict[str, object],
    target_priors: Dict[str, object],
    enabled: bool,
) -> torch.Tensor:
    if not enabled:
        return probabilities
    return apply_bayes_prior_correction(
        probabilities,
        train_pos_prior=train_priors["priors"][str(POSITIVE_LABEL)],
        target_pos_prior=target_priors["priors"][str(POSITIVE_LABEL)],
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.head_only_epochs < 0:
        raise ValueError("--head-only-epochs must be >= 0.")
    if args.epochs < 0:
        raise ValueError("--epochs must be >= 0.")
    if not (0.0 < args.llrd_decay <= 1.0):
        raise ValueError("--llrd-decay must be in (0, 1].")
    if not (0.0 < args.train_sample_ratio <= 1.0):
        raise ValueError("--train-sample-ratio must be in (0, 1].")
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be > 0.")
    if args.early_stop is not None and args.early_stop <= 0:
        raise ValueError("--early-stop must be > 0 when provided.")
    if args.pseudo_weight <= 0.0:
        raise ValueError("--pseudo-weight must be > 0.")
    labels_csv = args.labels_csv.resolve()
    val_root = args.val_root.expanduser().resolve()

    val_labels_csv = args.val_labels_csv
    if val_labels_csv is None:
        val_labels_csv = val_root / "labels.csv"
    else:
        val_labels_csv = val_labels_csv.expanduser()
    val_labels_csv = val_labels_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.mask_dir.resolve()
    mask_train_dir = mask_dir / "train"
    val_mask_dir = mask_dir / args.val_mask_split

    df = pd.read_csv(labels_csv)
    required_columns = {"id", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"{labels_csv} must contain columns: {sorted(required_columns)}")
    raw_sample_count = len(df)

    cleaning_metadata: Dict[str, object] = {
        "enabled": False,
        "original_total_count": raw_sample_count,
        "skip_ids_txt": None,
        "requested_skip_count": 0,
        "matched_skip_count": 0,
        "unmatched_skip_count": 0,
        "remaining_total_count": raw_sample_count,
        "unmatched_examples": [],
    }
    if args.skip_image_ids_txt is not None:
        skip_ids_txt = args.skip_image_ids_txt.resolve()
        df, skip_stats = filter_dataframe_by_skip_ids(df, skip_ids_txt, id_column="id")
        cleaning_metadata = {
            "enabled": True,
            "original_total_count": raw_sample_count,
            **skip_stats,
        }
        print(
            "Data cleaning | "
            f"original_total={raw_sample_count} | "
            f"skipped={cleaning_metadata['matched_skip_count']} | "
            f"remaining_total={cleaning_metadata['remaining_total_count']}"
        )
        if cleaning_metadata["unmatched_skip_count"] > 0:
            print(
                "Data cleaning warning | "
                f"unmatched_skip_ids={cleaning_metadata['unmatched_skip_count']} | "
                f"examples={cleaning_metadata['unmatched_examples']}"
            )

    unique_labels = sorted(df["label"].unique().tolist())
    if len(unique_labels) != 2:
        raise ValueError("This training setup currently expects binary classification.")

    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    df["label_idx"] = df["label"].map(label_to_idx)
    if POSITIVE_LABEL not in idx_to_label:
        raise ValueError("This training setup assumes label index 1 is the meteorite class.")

    if args.train_sample_ratio < 1.0:
        train_df = df.sample(frac=args.train_sample_ratio, random_state=args.seed).reset_index(drop=True)
    else:
        train_df = df.copy().reset_index(drop=True)
    train_df["sample_weight"] = 1.0
    if not mask_train_dir.is_dir():
        raise FileNotFoundError(f"Mask train directory not found: {mask_train_dir}")
    mask_index, masked_ids, skipped_ids = build_mask_image_index(
        mask_train_dir, train_df["id"].astype(str).tolist()
    )
    if not masked_ids:
        raise RuntimeError("No mask images found for training set. Check mask/train/ directory.")
    train_df = train_df[train_df["id"].astype(str).isin(set(masked_ids))].reset_index(drop=True)
    train_image_index = mask_index
    print(
        f"Mask training | mask_dir={mask_train_dir} | "
        f"kept={len(masked_ids)} | skipped={len(skipped_ids)}"
    )
    if args.val_split_ratio > 0.0:
        if not (0.0 < args.val_split_ratio < 1.0):
            raise ValueError("--val-split-ratio must be in (0, 1).")
        train_df, val_df = stratified_split(
            train_df,
            label_column="label_idx",
            val_ratio=args.val_split_ratio,
            seed=args.seed + 3,
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        val_image_index = train_image_index
        print(
            f"Val from train split | val_ratio={args.val_split_ratio:.4f} | "
            f"train_count={len(train_df)} | val_count={len(val_df)}"
        )
    if args.pseudo_prob_csv is not None:
        pseudo_prob_csv = args.pseudo_prob_csv.resolve()
        pseudo_df = build_pseudo_labeled_dataframe(
            pseudo_prob_csv,
            confidence_threshold=args.pseudo_prop,
            positive_label=POSITIVE_LABEL,
        )
        if not pseudo_df.empty:
            pseudo_images_dir = args.pseudo_images_dir.resolve()
            pseudo_image_index = build_image_index(pseudo_images_dir)
            missing_pseudo_ids = [image_id for image_id in pseudo_df["id"].astype(str).tolist() if image_id not in pseudo_image_index]
            if missing_pseudo_ids:
                raise RuntimeError(
                    f"Missing pseudo-label images under {pseudo_images_dir}; examples: {missing_pseudo_ids[:5]}"
                )
            train_image_index.update(pseudo_image_index)
            pseudo_train_df = pseudo_df[["id", "label", "label_idx"]].copy()
            pseudo_train_df["sample_weight"] = float(args.pseudo_weight)
            train_df = pd.concat([train_df, pseudo_train_df], axis=0, ignore_index=True)
            print(
                "Pseudo-labeling | "
                f"file={pseudo_prob_csv} | threshold={args.pseudo_prop:.4f} | kept={len(pseudo_df)} | "
                f"pseudo_weight={args.pseudo_weight:.4f}"
            )

    if train_df["label_idx"].nunique() < 2:
        raise RuntimeError("Training set must contain both classes. Increase --train-sample-ratio or add pseudo labels.")

    if args.val_split_ratio <= 0.0:
        if not val_mask_dir.is_dir():
            raise FileNotFoundError(f"Validation mask directory not found: {val_mask_dir}")
        val_df = pd.read_csv(val_labels_csv)
        if not required_columns.issubset(val_df.columns):
            raise ValueError(f"{val_labels_csv} must contain columns: {sorted(required_columns)}")
        val_df["label_idx"] = val_df["label"].map(label_to_idx)
        if val_df["label_idx"].isna().any():
            missing_labels = sorted(val_df.loc[val_df["label_idx"].isna(), "label"].unique().tolist())
            raise ValueError(f"Validation labels contain unknown classes: {missing_labels}")
        val_df["label_idx"] = val_df["label_idx"].astype(int)
        val_mask_index, val_masked_ids, val_skipped_ids = build_mask_image_index(
            val_mask_dir, val_df["id"].astype(str).tolist()
        )
        if not val_masked_ids:
            raise RuntimeError(f"No mask images found for validation set. Check {val_mask_dir}.")
        val_df = val_df[val_df["id"].astype(str).isin(set(val_masked_ids))].reset_index(drop=True)
        val_image_index = val_mask_index
        print(
            f"Mask validation | mask_dir={val_mask_dir} | "
            f"kept={len(val_masked_ids)} | skipped={len(val_skipped_ids)}"
        )

    threshold_search_df, model_select_df = stratified_subsplit(
        val_df,
        label_column="label_idx",
        first_ratio=args.threshold_search_ratio,
        seed=args.seed + 7,
    )
    if not args.disable_bayes_correction:
        threshold_search_df = rebalance_binary_subset_to_ratio(
            threshold_search_df,
            label_column="label_idx",
            target_neg_pos_ratio=args.target_neg_pos_ratio,
            seed=args.seed + 17,
        )
        model_select_df = rebalance_binary_subset_to_ratio(
            model_select_df,
            label_column="label_idx",
            target_neg_pos_ratio=args.target_neg_pos_ratio,
            seed=args.seed + 29,
        )

    missing_train_ids = [image_id for image_id in train_df["id"].astype(str).tolist() if image_id not in train_image_index]
    if missing_train_ids:
        raise RuntimeError(f"Missing train images after mask/pseudo indexing; examples: {missing_train_ids[:5]}")

    missing_val_ids = [image_id for image_id in val_df["id"].astype(str).tolist() if image_id not in val_image_index]
    if missing_val_ids:
        val_source = str(mask_train_dir) if args.val_split_ratio > 0.0 else str(val_mask_dir)
        raise RuntimeError(f"Missing val images under {val_source}; examples: {missing_val_ids[:5]}")

    csv_priors = compute_class_priors(df, label_column="label_idx", positive_label=POSITIVE_LABEL)
    train_priors = compute_class_priors(train_df, label_column="label_idx", positive_label=POSITIVE_LABEL)
    val_priors = compute_class_priors(val_df, label_column="label_idx", positive_label=POSITIVE_LABEL)
    threshold_search_priors = compute_class_priors(
        threshold_search_df,
        label_column="label_idx",
        positive_label=POSITIVE_LABEL,
    )
    model_select_priors = compute_class_priors(
        model_select_df,
        label_column="label_idx",
        positive_label=POSITIVE_LABEL,
    )
    target_priors = build_target_priors_from_ratio(args.target_neg_pos_ratio)
    threshold_search_enabled = args.open_threshold_search
    print(
        "Split summary | "
        f"train_count={len(train_df)} train_neg_pos_ratio={train_priors['negative_to_positive_ratio']:.4f} | "
        f"val_count={len(val_df)} val_neg_pos_ratio={val_priors['negative_to_positive_ratio']:.4f} | "
        f"train_sample_ratio={args.train_sample_ratio:.4f}"
    )
    print(
        "Validation rebalance | "
        f"threshold_search_neg_pos_ratio={threshold_search_priors['negative_to_positive_ratio']:.4f} | "
        f"model_select_neg_pos_ratio={model_select_priors['negative_to_positive_ratio']:.4f} | "
        f"target_neg_pos_ratio={target_priors['negative_to_positive_ratio']:.4f} | "
        f"threshold_search_enabled={threshold_search_enabled}"
    )

    model = ConvNeXtClassifier(
        backbone_name=args.backbone,
        backbone_checkpoint=args.backbone_checkpoint,
        num_classes=len(unique_labels),
        dropout=args.dropout,
        pretrained_backbone=not args.no_pretrained,
        drop_path_rate=args.drop_path_rate,
    )
    head_only_epochs = args.head_only_epochs
    finetune_epochs = args.epochs
    total_epochs = head_only_epochs + finetune_epochs
    if total_epochs <= 0:
        raise ValueError("Total epochs must be > 0. Adjust --head-only-epochs and/or --epochs.")

    freeze_backbone_for_head_only(model.backbone)
    print(f"Stage head_only | trainable classifier only | epochs={head_only_epochs}")
    if finetune_epochs > 0:
        freeze_backbone_for_head_only(model.backbone)
        print(f"Stage finetune prepared | will unfreeze full backbone after epoch {head_only_epochs}")

    backbone_image_size, backbone_image_mean, backbone_image_std = resolve_backbone_data_settings(model.backbone)
    image_size = args.image_size or normalize_image_size(backbone_image_size)
    image_mean = normalize_stats(args.image_mean, normalize_stats(backbone_image_mean, DEFAULT_MEAN))
    image_std = normalize_stats(args.image_std, normalize_stats(backbone_image_std, DEFAULT_STD))
    train_transform, eval_transform = build_transforms(
        image_size,
        image_mean,
        image_std,
        hflip_prob=args.hflip_prob,
        rotate_degrees=args.rotate_degrees,
        randaugment_n=args.randaugment_n,
        randaugment_m=args.randaugment_m,
        color_jitter_prob=args.color_jitter_prob,
        color_jitter_brightness=args.color_jitter_brightness,
        color_jitter_contrast=args.color_jitter_contrast,
        color_jitter_saturation=args.color_jitter_saturation,
        color_jitter_hue=args.color_jitter_hue,
    )

    train_dataset = MeteoriteDataset(
        train_df, train_image_index, train_transform,
    )
    threshold_search_dataset = MeteoriteDataset(
        threshold_search_df, val_image_index, eval_transform,
    )
    model_select_dataset = MeteoriteDataset(
        model_select_df, val_image_index, eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    threshold_search_loader = DataLoader(
        threshold_search_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model_select_loader = DataLoader(
        model_select_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device)
    model = model.to(device)
    if args.disable_bayes_correction:
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    else:
        class_weights = build_class_weights_for_target_prior(train_priors, target_priors).to(device)
    optimizer = create_optimizer(
        model,
        args.head_lr,
        args.backbone_lr,
        args.weight_decay,
        llrd_decay=args.llrd_decay,
    )
    current_stage = "head_only"

    wandb_run, wandb_identity = init_wandb_run(
        args,
        default_job_type="finetune",
        config=vars(args),
        output_dir=output_dir,
    )

    save_json(vars(args), output_dir / "train_args.json")
    save_json(
        {
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "positive_label": POSITIVE_LABEL,
            "image_size": image_size,
            "image_mean": image_mean,
            "image_std": image_std,
            "full_dataset_priors": summarize_priors(csv_priors),
            "train_priors": summarize_priors(train_priors),
            "val_priors": summarize_priors(val_priors),
            "threshold_search_priors": summarize_priors(threshold_search_priors),
            "model_select_priors": summarize_priors(model_select_priors),
            "target_priors": summarize_priors(target_priors),
            "class_weights": class_weights.detach().cpu().tolist(),
            "target_negative_to_positive_ratio": args.target_neg_pos_ratio,
            "selection_subsets_rebalanced_to_target_ratio": True,
            "threshold_metric": args.threshold_metric,
            "threshold_search_ratio": args.threshold_search_ratio,
            "threshold_search_enabled": threshold_search_enabled,
            "threshold_search_default_threshold": 0.5,
            "bayes_correction_enabled": not args.disable_bayes_correction,
            "train_sample_ratio": args.train_sample_ratio,
            "validation_source": {
                "mode": "train_split" if args.val_split_ratio > 0.0 else "external_masked",
                "val_split_enabled": args.val_split_ratio > 0.0,
                "val_split_ratio": args.val_split_ratio,
                "val_root": str(val_root) if args.val_split_ratio <= 0.0 else None,
                "mask_dir": str(mask_dir),
                "train_mask_dir": str(mask_train_dir),
                "val_mask_split": args.val_mask_split if args.val_split_ratio <= 0.0 else "train",
                "val_images_dir": str(mask_train_dir) if args.val_split_ratio > 0.0 else str(val_mask_dir),
                "val_labels_csv": str(val_labels_csv) if args.val_split_ratio <= 0.0 else str(labels_csv),
            },
            "data_cleaning": cleaning_metadata,
            "augmentations": {
                "hflip_prob": args.hflip_prob,
                "rotate_degrees": args.rotate_degrees,
                "cutmix_alpha": args.cutmix_alpha,
                "cutmix_prob": args.cutmix_prob,
                "mixup_alpha": args.mixup_alpha,
                "mixup_prob": args.mixup_prob,
                "randaugment_n": args.randaugment_n,
                "randaugment_m": args.randaugment_m,
                "color_jitter_prob": args.color_jitter_prob,
                "color_jitter_brightness": args.color_jitter_brightness,
                "color_jitter_contrast": args.color_jitter_contrast,
                "color_jitter_saturation": args.color_jitter_saturation,
                "color_jitter_hue": args.color_jitter_hue,
            },
            "pseudo_labeling": {
                "enabled": args.pseudo_prob_csv is not None,
                "confidence_threshold": args.pseudo_prop,
                "pseudo_weight": args.pseudo_weight,
            },
            "training_stages": {
                "head_only_epochs": head_only_epochs,
                "finetune_epochs": finetune_epochs,
                "total_epochs": total_epochs,
                "finetune_scope": "full_backbone",
                "llrd_decay": args.llrd_decay,
                "early_stop_patience": args.early_stop,
            },
            "wandb": {
                **wandb_identity,
                "mode": args.wandb_mode,
                "entity": args.wandb_entity,
                "tags": args.wandb_tags,
                "log_interval": args.log_interval,
            },
        },
        output_dir / "metadata.json",
    )

    history: List[Dict[str, object]] = []
    best_val_score = float("-inf")
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(1, total_epochs + 1):
        if current_stage == "head_only" and epoch > head_only_epochs:
            unfreeze_backbone_all(model.backbone)
            optimizer = create_optimizer(
                model,
                args.head_lr,
                args.backbone_lr,
                args.weight_decay,
                llrd_decay=args.llrd_decay,
            )
            current_stage = "finetune"
            print(f"Stage finetune | epoch={epoch:02d} | unfreezing full backbone")

        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            num_classes=len(unique_labels),
            class_weights=class_weights,
            optimizer=optimizer,
            cutmix_alpha=args.cutmix_alpha,
            cutmix_prob=args.cutmix_prob,
            mixup_alpha=args.mixup_alpha,
            mixup_prob=args.mixup_prob,
            label_smoothing=args.label_smoothing,
            wandb_run=wandb_run,
            stage=current_stage,
            epoch=epoch,
            global_step=global_step,
            log_interval=args.log_interval,
        )
        global_step = int(train_metrics.get("global_step", global_step))
        threshold_metrics = run_epoch(
            model,
            threshold_search_loader,
            device,
            num_classes=len(unique_labels),
            class_weights=class_weights,
        )
        model_select_metrics = run_epoch(
            model,
            model_select_loader,
            device,
            num_classes=len(unique_labels),
            class_weights=class_weights,
        )

        threshold_raw_prob_pos = threshold_metrics["prob_pos"]
        threshold_corrected_prob_pos = maybe_apply_bayes_correction(
            threshold_raw_prob_pos,
            train_priors,
            target_priors,
            enabled=not args.disable_bayes_correction,
        )
        if threshold_search_enabled:
            threshold_result = search_best_threshold(
                threshold_corrected_prob_pos,
                threshold_metrics["labels"],
                metric=args.threshold_metric,
            )
        else:
            threshold_result = {
                "threshold": 0.5,
                "metric_value": compute_binary_f1(
                    threshold_corrected_prob_pos,
                    threshold_metrics["labels"],
                    threshold=0.5,
                ),
            }
        search_f1_at_best = threshold_result["metric_value"]

        model_select_raw_prob_pos = model_select_metrics["prob_pos"]
        model_select_corrected_prob_pos = maybe_apply_bayes_correction(
            model_select_raw_prob_pos,
            train_priors,
            target_priors,
            enabled=not args.disable_bayes_correction,
        )
        model_select_raw_f1_at_default = compute_binary_f1(
            model_select_raw_prob_pos,
            model_select_metrics["labels"],
            threshold=0.5,
        )
        model_select_f1_at_search_threshold = compute_binary_f1(
            model_select_corrected_prob_pos,
            model_select_metrics["labels"],
            threshold=threshold_result["threshold"],
        )
        val_f1 = float(model_select_f1_at_search_threshold)

        epoch_metrics = {
            "epoch": epoch,
            "stage": current_stage,
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "train_grad_norm": float(train_metrics.get("grad_norm", 0.0)),
            "train_cutmix_batches": float(train_metrics.get("cutmix_batches", 0.0)),
            "threshold_search_loss": float(threshold_metrics["loss"]),
            "threshold_search_accuracy": float(threshold_metrics["accuracy"]),
            "threshold_search_f1_corrected_best": float(search_f1_at_best),
            "model_select_loss": float(model_select_metrics["loss"]),
            "model_select_accuracy": float(model_select_metrics["accuracy"]),
            "model_select_f1_raw_threshold_0.5": float(model_select_raw_f1_at_default),
            "model_select_f1_corrected_search_threshold": float(model_select_f1_at_search_threshold),
            "val_f1": val_f1,
            "best_threshold": float(threshold_result["threshold"]),
            f"{current_stage}/epoch_train_loss": float(train_metrics["loss"]),
            f"{current_stage}/epoch_threshold_search_loss": float(threshold_metrics["loss"]),
            f"{current_stage}/epoch_model_select_loss": float(model_select_metrics["loss"]),
            f"{current_stage}/epoch_train_accuracy": float(train_metrics["accuracy"]),
            f"{current_stage}/epoch_train_grad_norm": float(train_metrics.get("grad_norm", 0.0)),
            f"{current_stage}/epoch_threshold_search_accuracy": float(threshold_metrics["accuracy"]),
            f"{current_stage}/epoch_model_select_accuracy": float(model_select_metrics["accuracy"]),
            f"{current_stage}/epoch_threshold_search_f1_corrected_best": float(search_f1_at_best),
            f"{current_stage}/epoch_model_select_f1_corrected_search_threshold": float(model_select_f1_at_search_threshold),
            f"{current_stage}/epoch_best_threshold": float(threshold_result["threshold"]),
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch:02d} | stage={current_stage} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"cutmix_batches={int(train_metrics.get('cutmix_batches', 0.0))} | "
            f"search_loss={threshold_metrics['loss']:.4f} search_acc={threshold_metrics['accuracy']:.4f} "
            f"search_f1={search_f1_at_best:.4f} best_thr={threshold_result['threshold']:.4f} | "
            f"model_select_loss={model_select_metrics['loss']:.4f} "
            f"model_select_acc={model_select_metrics['accuracy']:.4f} "
            f"model_select_f1@thr={model_select_f1_at_search_threshold:.4f}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "stage": current_stage,
            "metrics": epoch_metrics,
            "threshold": float(threshold_result["threshold"]),
            "threshold_metric": args.threshold_metric,
            "threshold_search_enabled": threshold_search_enabled,
            "train_priors": summarize_priors(train_priors),
            "val_priors": summarize_priors(val_priors),
            "threshold_search_priors": summarize_priors(threshold_search_priors),
            "model_select_priors": summarize_priors(model_select_priors),
            "target_priors": summarize_priors(target_priors),
            "class_weights": class_weights.detach().cpu().tolist(),
            "bayes_correction_enabled": not args.disable_bayes_correction,
            "augmentations": {
                "hflip_prob": args.hflip_prob,
                "rotate_degrees": args.rotate_degrees,
                "cutmix_alpha": args.cutmix_alpha,
                "cutmix_prob": args.cutmix_prob,
                "mixup_alpha": args.mixup_alpha,
                "mixup_prob": args.mixup_prob,
                "randaugment_n": args.randaugment_n,
                "randaugment_m": args.randaugment_m,
                "color_jitter_prob": args.color_jitter_prob,
                "color_jitter_brightness": args.color_jitter_brightness,
                "color_jitter_contrast": args.color_jitter_contrast,
                "color_jitter_saturation": args.color_jitter_saturation,
                "color_jitter_hue": args.color_jitter_hue,
            },
            "training_stages": {
                "head_only_epochs": head_only_epochs,
                "finetune_epochs": finetune_epochs,
                "total_epochs": total_epochs,
                "finetune_scope": "full_backbone",
                "llrd_decay": args.llrd_decay,
                "early_stop_patience": args.early_stop,
            },
        }
        if args.save_every_epoch:
            torch.save(checkpoint, output_dir / f"epoch_{epoch:02d}.pt")
        torch.save(checkpoint, output_dir / "last.pt")
        is_best_checkpoint = val_f1 > best_val_score
        if is_best_checkpoint:
            best_val_score = val_f1
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": int(epoch),
                    "step": int(global_step),
                    "stage": current_stage,
                    "step/val/loss": float(model_select_metrics["loss"]),
                    "step/val/accuracy": float(model_select_metrics["accuracy"]),
                    "step/val/f1": val_f1,
                    "step/threshold_search/loss": float(threshold_metrics["loss"]),
                    "step/threshold_search/accuracy": float(threshold_metrics["accuracy"]),
                    "step/threshold_search/f1_corrected_best": float(search_f1_at_best),
                    "step/best_threshold": float(threshold_result["threshold"]),
                    "epoch/train/loss": float(train_metrics["loss"]),
                    "epoch/train/accuracy": float(train_metrics["accuracy"]),
                    "epoch/train/grad_norm": float(train_metrics.get("grad_norm", 0.0)),
                    "epoch/val/loss": float(model_select_metrics["loss"]),
                    "epoch/val/accuracy": float(model_select_metrics["accuracy"]),
                    "epoch/val/f1": val_f1,
                    "epoch/val/best_f1": float(best_val_score),
                    "epoch/threshold_search/loss": float(threshold_metrics["loss"]),
                    "epoch/threshold_search/accuracy": float(threshold_metrics["accuracy"]),
                    "epoch/threshold_search/f1_corrected_best": float(search_f1_at_best),
                    "epoch/best_threshold": float(threshold_result["threshold"]),
                }
            )

        if args.early_stop is not None and epochs_without_improvement >= args.early_stop:
            print(
                "Early stop | "
                f"patience={args.early_stop} | last_epoch={epoch:02d} | best_val_f1={best_val_score:.4f}"
            )
            break

    save_json({"history": history}, output_dir / "history.json")
    best_epoch_record = max(history, key=lambda record: record["val_f1"])
    update_wandb_summary(
        wandb_run,
        {
            "best_epoch": int(best_epoch_record["epoch"]),
            "best_stage": best_epoch_record["stage"],
            "best_model_select_f1": float(best_epoch_record["val_f1"]),
            "best_val_f1": float(best_epoch_record["val_f1"]),
            "best_threshold": float(best_epoch_record["best_threshold"]),
            "last_epoch": int(history[-1]["epoch"]),
            "final_train_loss": float(history[-1]["train_loss"]),
            "final_search_f1": float(history[-1]["threshold_search_f1_corrected_best"]),
            "early_stop_patience": args.early_stop,
            "epochs_without_improvement": epochs_without_improvement,
            "output_dir": str(output_dir),
            "best_checkpoint_path": str(output_dir / "best.pt"),
            "last_checkpoint_path": str(output_dir / "last.pt"),
        },
    )
    finish_wandb_run(wandb_run)


if __name__ == "__main__":
    main()
