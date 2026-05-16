from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import timm
from timm.data import resolve_model_data_config
from torchvision import transforms


def find_backbone_blocks(backbone: nn.Module):
    candidates = []
    for module_name, module in backbone.named_modules():
        for attr in ("stages", "features", "blocks", "layer", "layers"):
            if not hasattr(module, attr):
                continue
            blocks = getattr(module, attr)
            if not isinstance(blocks, (nn.ModuleList, nn.Sequential, list, tuple)):
                continue
            if len(blocks) == 0 or not all(isinstance(block, nn.Module) for block in blocks):
                continue
            full_name = f"{module_name}.{attr}" if module_name else attr
            candidates.append((full_name, blocks))

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: len(item[1]), reverse=True)
    return candidates[0]


def extract_backbone_state_dict(raw_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    candidate_roots = ("backbone.", "model.backbone.", "module.backbone.", "encoder.", "model.encoder.")
    for prefix in candidate_roots:
        filtered = {
            key[len(prefix):]: value
            for key, value in raw_state_dict.items()
            if key.startswith(prefix)
        }
        if filtered:
            return filtered
    return raw_state_dict


def load_backbone(
    backbone_name: str,
    backbone_checkpoint: Optional[Path] = None,
    pretrained: bool = True,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    backbone = timm.create_model(backbone_name, pretrained=pretrained and backbone_checkpoint is None, num_classes=0, drop_path_rate=drop_path_rate)

    if backbone_checkpoint is None:
        return backbone

    checkpoint = torch.load(backbone_checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            checkpoint = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {backbone_checkpoint}")

    checkpoint = extract_backbone_state_dict(checkpoint)
    missing, unexpected = backbone.load_state_dict(checkpoint, strict=False)
    print(
        "Loaded backbone checkpoint file",
        f"| missing_keys={len(missing)}",
        f"| unexpected_keys={len(unexpected)}",
    )
    return backbone


def freeze_backbone_for_head_only(backbone: nn.Module) -> None:
    for param in backbone.parameters():
        param.requires_grad = False


def unfreeze_backbone_all(backbone: nn.Module) -> None:
    for param in backbone.parameters():
        param.requires_grad = True


def build_backbone_llrd_param_groups(
    backbone: nn.Module,
    backbone_lr: float,
    llrd_decay: float,
) -> List[Dict[str, object]]:
    block_container_name, blocks = find_backbone_blocks(backbone)
    if blocks is None:
        return [{"params": [p for p in backbone.parameters() if p.requires_grad], "lr": backbone_lr}]

    layer_id_by_param_name: Dict[str, int] = {}
    for layer_id, block in enumerate(blocks, start=1):
        for name, _ in block.named_parameters(prefix="", recurse=True):
            full_name = f"{block_container_name}.{layer_id - 1}.{name}" if name else f"{block_container_name}.{layer_id - 1}"
            layer_id_by_param_name[full_name] = layer_id

    max_layer_id = len(blocks)
    grouped_params: Dict[int, List[nn.Parameter]] = {}
    for name, parameter in backbone.named_parameters():
        if not parameter.requires_grad:
            continue
        layer_id = layer_id_by_param_name.get(name, 0)
        grouped_params.setdefault(layer_id, []).append(parameter)

    param_groups: List[Dict[str, object]] = []
    for layer_id in sorted(grouped_params.keys()):
        lr_scale_power = max_layer_id - layer_id
        scaled_lr = backbone_lr * (llrd_decay ** lr_scale_power)
        param_groups.append(
            {
                "params": grouped_params[layer_id],
                "lr": scaled_lr,
                "group_type": "backbone",
                "layer_id": layer_id,
            }
        )
    return param_groups


def resolve_backbone_data_settings(backbone: nn.Module) -> Tuple[int, List[float], List[float]]:
    data_config = resolve_model_data_config(backbone)
    input_size = data_config.get("input_size", (3, 224, 224))
    image_size = int(input_size[-1])
    image_mean = [float(v) for v in data_config.get("mean", (0.485, 0.456, 0.406))]
    image_std = [float(v) for v in data_config.get("std", (0.229, 0.224, 0.225))]
    return image_size, image_mean, image_std


class ConvNeXtClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        backbone_checkpoint: Optional[Path],
        num_classes: int,
        dropout: float,
        pretrained_backbone: bool = True,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = load_backbone(
            backbone_name=backbone_name,
            backbone_checkpoint=backbone_checkpoint,
            pretrained=pretrained_backbone,
            drop_path_rate=drop_path_rate,
        )
        hidden_size = int(getattr(self.backbone, "num_features"))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.backbone(pixel_values)
        return self.classifier(features)


def build_transforms(
    image_size: int,
    image_mean: List[float],
    image_std: List[float],
    hflip_prob: float,
    rotate_degrees: float,
):
    train_ops = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=hflip_prob),
    ]
    if rotate_degrees > 0.0:
        train_ops.append(transforms.RandomRotation(degrees=(-rotate_degrees, rotate_degrees)))
    train_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ]
    )
    train_transform = transforms.Compose(train_ops)
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ]
    )
    return train_transform, eval_transform


def create_optimizer(
    model: ConvNeXtClassifier,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
    llrd_decay: float,
):
    param_groups = build_backbone_llrd_param_groups(
        model.backbone,
        backbone_lr=backbone_lr,
        llrd_decay=llrd_decay,
    )
    head_params = [param for param in model.classifier.parameters() if param.requires_grad]
    param_groups.append({"params": head_params, "lr": head_lr, "group_type": "head"})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
