from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from lane_binary.dataset.transforms import IMAGENET_MEAN, IMAGENET_STD


CLASS_NAMES = ["lane_out", "lane_in"]


def load_roi_image(
    image_path: str | Path,
    roi: tuple[int, int, int, int],
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    x1, y1, x2, y2 = roi
    width, height = image.size
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 >= x2 or y1 >= y2:
        raise ValueError(f"ROI {roi} outside image bounds {(width, height)} for {image_path}")
    return image.crop((x1, y1, x2, y2))


def preprocess_pil_to_numpy(image: Image.Image, image_size: int = 224) -> np.ndarray:
    resized = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    array = (array - mean) / std
    array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)

