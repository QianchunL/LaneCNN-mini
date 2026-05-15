#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lane_binary.dataset import LaneRoiDataset, build_val_transform
from lane_binary.model import create_model
from lane_binary.utils import BinaryMetricMeter, build_loss, compute_class_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LaneCNN checkpoint.")
    parser.add_argument("--csv", type=Path, default=Path("data/val.csv"))
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--model", default=None, help="Override model name from checkpoint.")
    parser.add_argument("--image-size", type=int, default=None, help="Override image size from checkpoint.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default="weighted_ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def checkpoint_args(checkpoint: dict) -> dict:
    raw_args = checkpoint.get("args", {})
    return raw_args if isinstance(raw_args, dict) else {}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, object]:
    model.eval()
    meter = BinaryMetricMeter()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        meter.update(logits, labels)

    return total_loss / max(total_samples, 1), meter.compute()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint_args(checkpoint)

    model_name = args.model or ckpt_args.get("model", "lane_cnn")
    image_size = args.image_size or int(ckpt_args.get("image_size", 224))
    dropout = float(ckpt_args.get("dropout", 0.2))

    dataset = LaneRoiDataset(
        args.csv,
        image_root=args.image_root,
        transform=build_val_transform(image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_model(model_name, num_classes=2, dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state"])

    eval_class_counts = dataset.class_counts()
    loss_class_counts = checkpoint.get("class_counts", eval_class_counts)
    class_weights = compute_class_weights(loss_class_counts).to(device) if args.loss in {"weighted_ce", "focal"} else None
    criterion = build_loss(args.loss, class_weights=class_weights, focal_gamma=args.focal_gamma).to(device)

    loss, metrics = evaluate(model, loader, criterion, device)
    result = {
        "loss": loss,
        "accuracy": metrics.accuracy,
        "lane_out_precision": metrics.lane_out_precision,
        "lane_out_recall": metrics.lane_out_recall,
        "lane_out_f1": metrics.lane_out_f1,
        "lane_in_precision": metrics.lane_in_precision,
        "lane_in_recall": metrics.lane_in_recall,
        "lane_in_f1": metrics.lane_in_f1,
        "confusion_matrix": metrics.confusion_matrix,
        "class_counts": eval_class_counts,
        "loss_class_counts": loss_class_counts,
    }

    print(json.dumps(result, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
