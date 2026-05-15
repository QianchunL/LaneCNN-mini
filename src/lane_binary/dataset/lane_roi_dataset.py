from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch.utils.data import Dataset


LABEL_OUT = 0
LABEL_IN = 1
REQUIRED_COLUMNS = {"image_path", "roi_x1", "roi_y1", "roi_x2", "roi_y2", "label"}


@dataclass(frozen=True)
class LaneRoiSample:
    image_path: Path
    roi: tuple[int, int, int, int]
    label: int
    metadata: dict[str, str]


class LaneRoiDataset(Dataset):
    """Dataset for lane_in/lane_out ROI classification.

    The CSV is expected to contain at least:
    image_path, roi_x1, roi_y1, roi_x2, roi_y2, label.
    Image paths can be absolute or relative to image_root.
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path | None = None,
        transform: Callable[[Image.Image], Any] | None = None,
        return_metadata: bool = False,
        validate_roi: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root) if image_root is not None else None
        self.transform = transform
        self.return_metadata = return_metadata
        self.validate_roi = validate_roi
        self.samples = self._read_csv(self.csv_path)

        if not self.samples:
            raise ValueError(f"No usable samples found in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = self._load_image(sample.image_path)
        roi_image = self._crop_roi(image, sample.roi, sample.image_path)

        if self.transform is not None:
            roi_image = self.transform(roi_image)

        if self.return_metadata:
            return {
                "image": roi_image,
                "label": sample.label,
                "metadata": {
                    **sample.metadata,
                    "image_path": str(sample.image_path),
                    "roi": sample.roi,
                },
            }

        return roi_image, sample.label

    def class_counts(self) -> dict[int, int]:
        counts = {LABEL_OUT: 0, LABEL_IN: 0}
        for sample in self.samples:
            counts[sample.label] += 1
        return counts

    def _read_csv(self, csv_path: Path) -> list[LaneRoiSample]:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        samples: list[LaneRoiSample] = []
        with csv_path.open("r", newline="") as fp:
            reader = csv.DictReader(fp)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {csv_path}")

            missing = REQUIRED_COLUMNS - set(reader.fieldnames)
            if missing:
                raise ValueError(f"CSV {csv_path} missing required columns: {sorted(missing)}")

            for row_idx, row in enumerate(reader, start=2):
                label = self._parse_label(row["label"], row_idx)
                image_path = self._resolve_image_path(row["image_path"])
                roi = (
                    self._parse_int(row["roi_x1"], "roi_x1", row_idx),
                    self._parse_int(row["roi_y1"], "roi_y1", row_idx),
                    self._parse_int(row["roi_x2"], "roi_x2", row_idx),
                    self._parse_int(row["roi_y2"], "roi_y2", row_idx),
                )
                if roi[0] >= roi[2] or roi[1] >= roi[3]:
                    raise ValueError(f"Invalid ROI at {csv_path}:{row_idx}: {roi}")

                metadata = {
                    key: value
                    for key, value in row.items()
                    if key not in REQUIRED_COLUMNS and value is not None
                }
                samples.append(
                    LaneRoiSample(
                        image_path=image_path,
                        roi=roi,
                        label=label,
                        metadata=metadata,
                    )
                )

        return samples

    def _resolve_image_path(self, raw_path: str) -> Path:
        image_path = Path(raw_path)
        if image_path.is_absolute():
            return image_path
        if self.image_root is None:
            return image_path
        return self.image_root / image_path

    @staticmethod
    def _parse_int(raw_value: str, field_name: str, row_idx: int) -> int:
        try:
            return int(float(raw_value))
        except ValueError as exc:
            raise ValueError(f"Invalid {field_name} at CSV row {row_idx}: {raw_value}") from exc

    @staticmethod
    def _parse_label(raw_value: str, row_idx: int) -> int:
        label = LaneRoiDataset._parse_int(raw_value, "label", row_idx)
        if label not in (LABEL_OUT, LABEL_IN):
            raise ValueError(f"Invalid label at CSV row {row_idx}: {label}")
        return label

    @staticmethod
    def _load_image(image_path: Path) -> Image.Image:
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        return Image.open(image_path).convert("RGB")

    def _crop_roi(
        self,
        image: Image.Image,
        roi: tuple[int, int, int, int],
        image_path: Path,
    ) -> Image.Image:
        x1, y1, x2, y2 = roi
        width, height = image.size
        if self.validate_roi and (x1 < 0 or y1 < 0 or x2 > width or y2 > height):
            raise ValueError(
                f"ROI {roi} outside image bounds {(width, height)} for {image_path}"
            )
        return image.crop((x1, y1, x2, y2))

