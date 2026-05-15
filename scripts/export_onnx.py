#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LaneCNN checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("onnx/lane_cnn.onnx"))
    parser.add_argument("--model", default=None, help="Override model name from checkpoint.")
    parser.add_argument("--image-size", type=int, default=None, help="Override image size from checkpoint.")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--no-dynamic-batch", action="store_true")
    return parser.parse_args()


def checkpoint_args(checkpoint: dict) -> dict:
    raw_args = checkpoint.get("args", {})
    return raw_args if isinstance(raw_args, dict) else {}


def main() -> None:
    args = parse_args()
    import torch

    from lane_binary.model import create_model

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint_args(checkpoint)

    model_name = args.model or ckpt_args.get("model", "lane_cnn")
    image_size = args.image_size or int(ckpt_args.get("image_size", 224))
    dropout = float(ckpt_args.get("dropout", 0.2))

    model = create_model(model_name, num_classes=2, dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if not args.no_dynamic_batch:
        dynamic_axes = {
            "image": {0: "batch"},
            "logits": {0: "batch"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )

    print(f"Exported ONNX model to {args.output}")
    print(f"input=image shape=[batch, 3, {image_size}, {image_size}]")
    print("output=logits shape=[batch, 2]")


if __name__ == "__main__":
    main()
