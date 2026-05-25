#!/usr/bin/env python3
"""Masked Autoencoder (MAE) domain adaptation on all stone images.

Continues DINOv2 ViT pretraining on ~9k unlabeled stone images
(train+myval+test+mytest). After MAE, the backbone is saved for
supervised fine-tuning on the original 4780 train labels.

Based on: He et al. "Masked Autoencoders Are Scalable Vision Learners" (2022)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_image_paths(roots: List[Path]) -> List[str]:
    paths: List[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(str(path))
    return paths


class MAEImageDataset(Dataset):
    def __init__(self, paths: List[str], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        img = Image.open(self.paths[index]).convert("RGB")
        img = ImageOps.exif_transpose(img)
        return self.transform(img)


class PatchEmbed(nn.Module):
    """Patch embedding layer matching timm ViT conventions."""

    def __init__(self, img_size: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class MAEDecoder(nn.Module):
    def __init__(
        self,
        num_patches: int,
        encoder_dim: int = 768,
        decoder_dim: int = 384,
        decoder_depth: int = 8,
        decoder_heads: int = 8,
        patch_size: int = 14,
    ):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim))
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder_blocks = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_size * patch_size * 3)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, num_visible, encoder_dim)
        # ids_restore: (B, num_patches) — indices to restore original order
        x = self.enc_to_dec(x)
        B, N_vis, D = x.shape
        L = ids_restore.shape[1]  # total patches

        mask_tokens = self.mask_token.expand(B, L - N_vis, -1)
        x_full = torch.cat([x, mask_tokens], dim=1)  # (B, L, D)

        # Restore original order
        batch_idx = torch.arange(B, device=x.device).unsqueeze(-1).expand(-1, L)
        x_full = x_full[batch_idx, ids_restore]

        # Add cls-like token and pos embed
        cls_token = torch.zeros(B, 1, D, device=x.device)
        x_full = torch.cat([cls_token, x_full], dim=1)
        x_full = x_full + self.decoder_pos_embed

        x_full = self.decoder_blocks(x_full)
        x_full = self.decoder_norm(x_full)
        x_full = x_full[:, 1:, :]  # remove cls
        x_full = self.pred(x_full)  # (B, L, patch_size^2 * 3)
        return x_full


def random_masking(x: torch.Tensor, mask_ratio: float):
    """Randomly mask patches."""
    B, N, D = x.shape
    len_keep = int(N * (1.0 - mask_ratio))

    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

    mask = torch.ones(B, N, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


class MAEViT(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        img_size: int = 224,
        patch_size: int = 14,
        encoder_dim: int = 768,
        decoder_dim: int = 384,
        decoder_depth: int = 8,
        decoder_heads: int = 8,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.img_size = img_size
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbed(img_size, patch_size, encoder_dim)
        self.encoder = encoder
        self.decoder = MAEDecoder(
            num_patches=num_patches,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            decoder_heads=decoder_heads,
            patch_size=patch_size,
        )

        # Copy pos_embed from encoder if available
        encoder_pos_embed = getattr(encoder, "pos_embed", None)
        if encoder_pos_embed is not None:
            self.register_buffer("pos_embed", encoder_pos_embed.data.clone())
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, encoder_dim))

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        B, C, H, W = imgs.shape
        x = imgs.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, (H // p) * (W // p), p * p * C)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        h = w = self.img_size // p
        B = x.shape[0]
        x = x.reshape(B, h, w, p, p, 3)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, 3, self.img_size, self.img_size)
        return x

    def forward_encoder(self, imgs: torch.Tensor):
        # Patch embed
        x = self.patch_embed(imgs)

        # Add pos embed (excluding cls token)
        cls_pos = self.pos_embed[:, :1, :]
        patch_pos = self.pos_embed[:, 1:, :]

        # Interpolate if needed
        if patch_pos.shape[1] != x.shape[1]:
            patch_pos = nn.functional.interpolate(
                patch_pos.transpose(1, 2).reshape(1, -1, int(math.sqrt(patch_pos.shape[1])), int(math.sqrt(patch_pos.shape[1]))),
                size=(int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1]))),
                mode="bicubic",
            ).flatten(2).transpose(1, 2)

        x = x + patch_pos

        # Random masking
        x, mask, ids_restore = random_masking(x, self.mask_ratio)

        # Add cls token
        cls_tokens = cls_pos.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Encode through ViT blocks (skip the patch_embed already done)
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)

        return x, mask, ids_restore

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor):
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs: torch.Tensor):
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-name", type=str, default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--decoder-dim", type=int, default=384)
    parser.add_argument("--decoder-depth", type=int, default=8)
    parser.add_argument("--decoder-heads", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("train/outputs/mae_domain"))
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--train-roots", type=str, nargs="*", default=[
        "preprocess/bbox_crop/train",
        "preprocess/bbox_crop/myval",
        "preprocess/bbox_crop/test",
        "mytest",
    ])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    roots = [Path(r) for r in args.train_roots]
    paths = collect_image_paths(roots)
    print(f"Collected {len(paths)} unlabeled images from {[str(r) for r in roots]}")

    # Create model
    print(f"Loading encoder: {args.encoder_name}")
    encoder = timm.create_model(
        args.encoder_name,
        pretrained=True,
        num_classes=0,
        img_size=args.img_size,
    )
    data_config = resolve_model_data_config(encoder)

    patch_size = encoder.patch_embed.patch_size[0]
    encoder_dim = encoder.embed_dim
    print(f"Encoder: dim={encoder_dim}, patch_size={patch_size}, img_size={args.img_size}")

    model = MAEViT(
        encoder=encoder,
        img_size=args.img_size,
        patch_size=patch_size,
        encoder_dim=encoder_dim,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
        mask_ratio=args.mask_ratio,
    )
    device = torch.device(args.device)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model params: {total_params:.1f}M total, {trainable_params:.1f}M trainable")

    # Data
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[float(v) for v in data_config.get("mean", (0.5, 0.5, 0.5))],
            std=[float(v) for v in data_config.get("std", (0.5, 0.5, 0.5))],
        ),
    ])
    dataset = MAEImageDataset(paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data: {len(dataset)} images, {len(loader)} batches/epoch")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min,
    )

    # Training
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    history: list = []
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_idx, imgs in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = model(imgs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if batch_idx % args.log_interval == 0:
                print(
                    f"  ep {epoch:03d} [{batch_idx:04d}/{len(loader):04d}] "
                    f"loss={loss.item():.6f} lr={scheduler.get_last_lr()[0]:.2e}"
                )

        avg_loss = epoch_loss / len(loader)
        scheduler.step()
        history.append({"epoch": epoch, "loss": avg_loss, "lr": float(scheduler.get_last_lr()[0])})
        print(f"Epoch {epoch:03d} | avg_loss={avg_loss:.6f} | lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model": encoder.state_dict(), "epoch": epoch, "loss": avg_loss}, args.output_dir / "best.pt")

        if epoch % args.save_every == 0:
            torch.save(
                {"model": encoder.state_dict(), "epoch": epoch, "loss": avg_loss},
                args.output_dir / f"epoch_{epoch:03d}.pt",
            )

    # Save final and history
    torch.save({"model": encoder.state_dict(), "epoch": args.epochs, "loss": avg_loss}, args.output_dir / "last.pt")
    with (args.output_dir / "history.json").open("w") as f:
        json.dump({"history": history, "args": vars(args)}, f, indent=2)

    print(f"Done. Best loss: {best_loss:.6f}. Encoder saved to {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
