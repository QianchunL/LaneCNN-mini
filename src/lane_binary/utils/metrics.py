from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    lane_out_precision: float
    lane_out_recall: float
    lane_out_f1: float
    lane_in_precision: float
    lane_in_recall: float
    lane_in_f1: float
    confusion_matrix: list[list[int]]


@dataclass(frozen=True)
class BinaryConfusion:
    tn: int
    fp: int
    fn: int
    tp: int


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def compute_binary_confusion(logits: torch.Tensor, labels: torch.Tensor) -> BinaryConfusion:
    preds = logits.argmax(dim=1)
    labels = labels.to(preds.device)

    tn = int(((preds == 0) & (labels == 0)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())
    tp = int(((preds == 1) & (labels == 1)).sum().item())
    return BinaryConfusion(tn=tn, fp=fp, fn=fn, tp=tp)


def metrics_from_confusion(confusion: BinaryConfusion) -> BinaryMetrics:
    """Compute metrics for labels 0=lane_out, 1=lane_in."""

    tn = confusion.tn
    fp = confusion.fp
    fn = confusion.fn
    tp = confusion.tp

    total = tn + fp + fn + tp
    accuracy = _safe_divide(tn + tp, total)

    lane_out_precision = _safe_divide(tn, tn + fn)
    lane_out_recall = _safe_divide(tn, tn + fp)
    lane_out_f1 = _safe_divide(
        2 * lane_out_precision * lane_out_recall,
        lane_out_precision + lane_out_recall,
    )

    lane_in_precision = _safe_divide(tp, tp + fp)
    lane_in_recall = _safe_divide(tp, tp + fn)
    lane_in_f1 = _safe_divide(
        2 * lane_in_precision * lane_in_recall,
        lane_in_precision + lane_in_recall,
    )

    return BinaryMetrics(
        accuracy=accuracy,
        lane_out_precision=lane_out_precision,
        lane_out_recall=lane_out_recall,
        lane_out_f1=lane_out_f1,
        lane_in_precision=lane_in_precision,
        lane_in_recall=lane_in_recall,
        lane_in_f1=lane_in_f1,
        confusion_matrix=[[tn, fp], [fn, tp]],
    )


def compute_binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> BinaryMetrics:
    return metrics_from_confusion(compute_binary_confusion(logits, labels))


class BinaryMetricMeter:
    def __init__(self) -> None:
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.tp = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        confusion = compute_binary_confusion(logits.detach(), labels.detach())
        self.tn += confusion.tn
        self.fp += confusion.fp
        self.fn += confusion.fn
        self.tp += confusion.tp

    def compute(self) -> BinaryMetrics:
        return metrics_from_confusion(
            BinaryConfusion(tn=self.tn, fp=self.fp, fn=self.fn, tp=self.tp)
        )
