from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from lane_binary.dataset.lane_roi_dataset import LaneRoiDataset
from lane_binary.dataset.transforms import build_train_transform, build_val_transform


def make_lane_dataloader(
    csv_path: str | Path,
    image_root: str | Path | None,
    batch_size: int,
    train: bool,
    image_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool | None = None,
) -> DataLoader:
    transform = build_train_transform(image_size) if train else build_val_transform(image_size)
    dataset = LaneRoiDataset(
        csv_path=csv_path,
        image_root=image_root,
        transform=transform,
        return_metadata=False,
    )
    if shuffle is None:
        shuffle = train

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=train,
    )

