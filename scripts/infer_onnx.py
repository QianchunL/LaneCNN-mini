#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ONNX Runtime inference for one ROI image.")
    parser.add_argument("--onnx", type=Path, default=Path("onnx/lane_cnn.onnx"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"), required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--providers", default=None, help="Comma-separated ONNX Runtime providers.")
    return parser.parse_args()


def make_session(onnx_path: Path, providers: str | None):
    import onnxruntime as ort

    if providers:
        provider_list = [provider.strip() for provider in providers.split(",") if provider.strip()]
    else:
        provider_list = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), providers=provider_list)


def main() -> None:
    args = parse_args()
    from lane_binary.utils import CLASS_NAMES, load_roi_image, preprocess_pil_to_numpy, softmax

    roi = tuple(args.roi)
    roi_image = load_roi_image(args.image, roi)
    input_array = preprocess_pil_to_numpy(roi_image, args.image_size)

    session = make_session(args.onnx, args.providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logits = session.run([output_name], {input_name: input_array})[0]
    probs = softmax(logits)
    pred = int(probs.argmax(axis=1)[0])

    result = {
        "class_id": pred,
        "class_name": CLASS_NAMES[pred],
        "confidence": float(probs[0, pred]),
        "probabilities": {
            CLASS_NAMES[class_id]: float(probs[0, class_id])
            for class_id in range(len(CLASS_NAMES))
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
