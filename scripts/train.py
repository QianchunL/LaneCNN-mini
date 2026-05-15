#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lane_binary.dataset import LaneRoiDataset, build_train_transform, build_val_transform
from lane_binary.model import create_model
from lane_binary.utils import BinaryMetricMeter, build_loss, compute_class_weights, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LaneCNN for lane_in/lane_out ROI classification.")
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/val.csv"))
    parser.add_argument("--image-root", type=Path, required=True, help="nuScenes dataroot for relative image paths.")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/lane_cnn"))
    parser.add_argument("--model", default="lane_cnn")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default="weighted_ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--save-every", type=int, default=0, help="Save periodic checkpoints every N epochs. 0 disables it.")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def make_loader(
    dataset: LaneRoiDataset,
    batch_size: int,
    num_workers: int,
    train: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=train,
    )


def move_loss_to_device(criterion: nn.Module, device: torch.device) -> nn.Module:
    return criterion.to(device)


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, object]:
    model.train()
    meter = BinaryMetricMeter()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        meter.update(logits, labels)

    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, meter.compute()


@torch.no_grad()
def run_eval_epoch(
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

    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, meter.compute()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_score: float,
    args: argparse.Namespace,
    class_counts: dict[int, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "args": vars(args),
            "class_counts": class_counts,
        },
        path,
    )


def log_metrics(writer: SummaryWriter, prefix: str, loss: float, metrics, epoch: int) -> None:
    writer.add_scalar(f"{prefix}/loss", loss, epoch)
    writer.add_scalar(f"{prefix}/accuracy", metrics.accuracy, epoch)
    writer.add_scalar(f"{prefix}/lane_out_precision", metrics.lane_out_precision, epoch)
    writer.add_scalar(f"{prefix}/lane_out_recall", metrics.lane_out_recall, epoch)
    writer.add_scalar(f"{prefix}/lane_out_f1", metrics.lane_out_f1, epoch)
    writer.add_scalar(f"{prefix}/lane_in_precision", metrics.lane_in_precision, epoch)
    writer.add_scalar(f"{prefix}/lane_in_recall", metrics.lane_in_recall, epoch)
    writer.add_scalar(f"{prefix}/lane_in_f1", metrics.lane_in_f1, epoch)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    train_dataset = LaneRoiDataset(
        args.train_csv,
        image_root=args.image_root,
        transform=build_train_transform(args.image_size),
    )
    val_dataset = LaneRoiDataset(
        args.val_csv,
        image_root=args.image_root,
        transform=build_val_transform(args.image_size),
    )
    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, train=True, device=device)
    val_loader = make_loader(val_dataset, args.batch_size, args.num_workers, train=False, device=device)

    class_counts = train_dataset.class_counts()
    class_weights = compute_class_weights(class_counts).to(device) if args.loss in {"weighted_ce", "focal"} else None
    criterion = move_loss_to_device(
        build_loss(args.loss, class_weights=class_weights, focal_gamma=args.focal_gamma),
        device,
    )

    model = create_model(args.model, num_classes=2, dropout=args.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.run_dir))
    (args.output_dir / "train_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n"
    )

    best_score = -1.0
    best_path = args.output_dir / "best.pt"
    last_path = args.output_dir / "last.pt"

    print(f"device={device}")
    print(f"class_counts={class_counts}")
    if class_weights is not None:
        print(f"class_weights={[round(v, 4) for v in class_weights.detach().cpu().tolist()]}")

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_metrics = run_train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_metrics = run_eval_epoch(model, val_loader, criterion, device)
            scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/lr", current_lr, epoch)
            log_metrics(writer, "train", train_loss, train_metrics, epoch)
            log_metrics(writer, "val", val_loss, val_metrics, epoch)

            # Safety-oriented validation score: prioritize lane_out recall, then overall F1.
            score = 0.7 * val_metrics.lane_out_recall + 0.3 * val_metrics.lane_out_f1
            is_best = score > best_score
            if is_best:
                best_score = score
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_score,
                    args,
                    class_counts,
                )

            save_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_score,
                args,
                class_counts,
            )
            if args.save_every > 0 and epoch % args.save_every == 0:
                save_checkpoint(
                    args.output_dir / f"epoch_{epoch:03d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_score,
                    args,
                    class_counts,
                )

            print(
                "epoch={:03d} lr={:.6f} train_loss={:.4f} val_loss={:.4f} "
                "val_acc={:.4f} val_lane_out_recall={:.4f} val_lane_out_f1={:.4f}{}"
                .format(
                    epoch,
                    current_lr,
                    train_loss,
                    val_loss,
                    val_metrics.accuracy,
                    val_metrics.lane_out_recall,
                    val_metrics.lane_out_f1,
                    " best" if is_best else "",
                )
            )
    finally:
        writer.close()

    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
