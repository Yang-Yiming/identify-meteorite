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


def build_soft_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).to(dtype=torch.float32)


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


def apply_cutmix(
    pixel_values: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    num_classes: int,
    alpha: float,
    probability: float,
) -> CutMixBatch:
    if alpha <= 0.0 or probability <= 0.0 or pixel_values.size(0) < 2:
        soft_targets = build_soft_targets(labels, num_classes=num_classes)
        return CutMixBatch(pixel_values, labels, soft_targets, sample_weights, 1.0, False)

    if torch.rand(1, device=pixel_values.device).item() >= probability:
        soft_targets = build_soft_targets(labels, num_classes=num_classes)
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
        lambda_value * build_soft_targets(labels, num_classes=num_classes)
        + (1.0 - lambda_value) * build_soft_targets(labels[indices], num_classes=num_classes)
    )
    mixed_sample_weights = lambda_value * sample_weights + (1.0 - lambda_value) * sample_weights[indices]
    return CutMixBatch(mixed_pixel_values, labels, soft_targets, mixed_sample_weights, lambda_value, True)
