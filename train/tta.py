from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn


TTA_MODE_TO_VIEWS = {
    "4way": ["identity", "hflip", "vflip", "rot90"],
    "8way": ["identity", "rot90", "rot180", "rot270", "hflip", "vflip", "transpose", "transverse"],
}


def resolve_tta_views(mode: str) -> List[str]:
    try:
        return list(TTA_MODE_TO_VIEWS[mode])
    except KeyError as exc:
        supported = ", ".join(sorted(TTA_MODE_TO_VIEWS))
        raise ValueError(f"Unsupported TTA mode: {mode}. Expected one of: {supported}") from exc


def apply_tta_view(pixel_values: torch.Tensor, view: str) -> torch.Tensor:
    if view == "identity":
        return pixel_values
    if view == "hflip":
        return torch.flip(pixel_values, dims=(-1,))
    if view == "vflip":
        return torch.flip(pixel_values, dims=(-2,))
    if view == "rot90":
        return torch.rot90(pixel_values, k=1, dims=(-2, -1))
    if view == "rot180":
        return torch.rot90(pixel_values, k=2, dims=(-2, -1))
    if view == "rot270":
        return torch.rot90(pixel_values, k=3, dims=(-2, -1))
    if view == "transpose":
        return pixel_values.transpose(-2, -1)
    if view == "transverse":
        return torch.rot90(pixel_values.transpose(-2, -1), k=2, dims=(-2, -1))
    raise ValueError(f"Unsupported TTA view: {view}")


def predict_probabilities_with_tta(
    model: nn.Module,
    pixel_values: torch.Tensor,
    positive_label: int,
    views: Iterable[str],
    device_type: str,
    autocast_enabled: bool,
) -> torch.Tensor:
    view_probabilities = []
    for view in views:
        augmented_pixels = apply_tta_view(pixel_values, view)
        with torch.amp.autocast(device_type=device_type, enabled=autocast_enabled):
            logits = model(augmented_pixels)
        view_probabilities.append(torch.softmax(logits, dim=1)[:, positive_label])

    if not view_probabilities:
        raise ValueError("TTA views must not be empty")
    return torch.stack(view_probabilities, dim=0).mean(dim=0)
