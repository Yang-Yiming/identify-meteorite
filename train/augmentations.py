from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CutMixBatch:
    pixel_values: torch.Tensor
    labels: torch.Tensor
    mixed_labels: torch.Tensor
    sample_weights: torch.Tensor
    lambda_value: float
    applied: bool


def build_soft_targets(labels: torch.Tensor, num_classes: int, smoothing: float = 0.0) -> torch.Tensor:
    one_hot = F.one_hot(labels, num_classes=num_classes).to(dtype=torch.float32)
    if smoothing > 0.0:
        return one_hot * (1.0 - smoothing) + smoothing / num_classes
    return one_hot


def soft_target_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=1)
    if class_weights is not None:
        log_probs = log_probs * class_weights.view(1, -1)
    loss = -(soft_targets * log_probs).sum(dim=1)
    if sample_weights is not None:
        loss = loss * sample_weights
    return loss.mean()


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=class_weights)
    target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    focal_weight = (1.0 - target_probs) ** gamma
    if alpha != 1.0:
        target_weights = torch.full_like(targets, 1.0 - alpha, dtype=torch.float32)
        target_weights = torch.where(targets == 1, alpha, target_weights).to(device=logits.device)
        focal_weight = focal_weight * target_weights
    loss = focal_weight * ce_loss
    if sample_weights is not None:
        loss = loss * sample_weights
    return loss.mean()


def apply_cutmix(
    pixel_values: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    num_classes: int,
    alpha: float,
    probability: float,
    smoothing: float = 0.0,
) -> CutMixBatch:
    if alpha <= 0.0 or probability <= 0.0 or pixel_values.size(0) < 2:
        soft_targets = build_soft_targets(labels, num_classes=num_classes, smoothing=smoothing)
        return CutMixBatch(pixel_values, labels, soft_targets, sample_weights, 1.0, False)

    if torch.rand(1, device=pixel_values.device).item() >= probability:
        soft_targets = build_soft_targets(labels, num_classes=num_classes, smoothing=smoothing)
        return CutMixBatch(pixel_values, labels, soft_targets, sample_weights, 1.0, False)

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    indices = torch.randperm(pixel_values.size(0), device=pixel_values.device)
    mixed_pixel_values = pixel_values.clone()

    height = pixel_values.size(2)
    width = pixel_values.size(3)
    cut_ratio = (1.0 - lam) ** 0.5
    cut_w = max(1, int(width * cut_ratio))
    cut_h = max(1, int(height * cut_ratio))

    center_x = int(torch.randint(width, (1,), device=pixel_values.device).item())
    center_y = int(torch.randint(height, (1,), device=pixel_values.device).item())

    x1 = max(center_x - cut_w // 2, 0)
    y1 = max(center_y - cut_h // 2, 0)
    x2 = min(center_x + cut_w // 2, width)
    y2 = min(center_y + cut_h // 2, height)
    if x1 == x2:
        x2 = min(width, x1 + 1)
    if y1 == y2:
        y2 = min(height, y1 + 1)

    mixed_pixel_values[:, :, y1:y2, x1:x2] = pixel_values[indices, :, y1:y2, x1:x2]

    patch_area = float((x2 - x1) * (y2 - y1))
    lambda_value = 1.0 - patch_area / float(width * height)
    soft_targets = (
        lambda_value * build_soft_targets(labels, num_classes=num_classes, smoothing=smoothing)
        + (1.0 - lambda_value) * build_soft_targets(labels[indices], num_classes=num_classes, smoothing=smoothing)
    )
    mixed_sample_weights = lambda_value * sample_weights + (1.0 - lambda_value) * sample_weights[indices]
    return CutMixBatch(mixed_pixel_values, labels, soft_targets, mixed_sample_weights, lambda_value, True)


@dataclass(frozen=True)
class MixUpBatch:
    pixel_values: torch.Tensor
    labels: torch.Tensor
    mixed_labels: torch.Tensor
    sample_weights: torch.Tensor
    lambda_value: float
    applied: bool


def apply_mixup(
    pixel_values: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    num_classes: int,
    alpha: float,
    probability: float,
    smoothing: float = 0.0,
) -> MixUpBatch:
    if alpha <= 0.0 or probability <= 0.0 or pixel_values.size(0) < 2:
        soft_targets = build_soft_targets(labels, num_classes=num_classes, smoothing=smoothing)
        return MixUpBatch(pixel_values, labels, soft_targets, sample_weights, 1.0, False)

    if torch.rand(1, device=pixel_values.device).item() >= probability:
        soft_targets = build_soft_targets(labels, num_classes=num_classes, smoothing=smoothing)
        return MixUpBatch(pixel_values, labels, soft_targets, sample_weights, 1.0, False)

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    indices = torch.randperm(pixel_values.size(0), device=pixel_values.device)

    mixed_pixel_values = lam * pixel_values + (1.0 - lam) * pixel_values[indices]
    soft_targets = (
        lam * build_soft_targets(labels, num_classes=num_classes, smoothing=smoothing)
        + (1.0 - lam) * build_soft_targets(labels[indices], num_classes=num_classes, smoothing=smoothing)
    )
    mixed_sample_weights = lam * sample_weights + (1.0 - lam) * sample_weights[indices]
    return MixUpBatch(mixed_pixel_values, labels, soft_targets, mixed_sample_weights, lam, True)
