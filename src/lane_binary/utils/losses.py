from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def compute_class_weights(class_counts: dict[int, int], num_classes: int = 2) -> torch.Tensor:
    """Return inverse-frequency class weights in class-id order."""

    total = sum(class_counts.get(class_id, 0) for class_id in range(num_classes))
    if total <= 0:
        raise ValueError("Cannot compute class weights from an empty dataset")

    weights = []
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 0)
        if count <= 0:
            raise ValueError(f"Class {class_id} has zero samples")
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Multi-class focal loss for hard or imbalanced samples."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(
    name: str,
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
) -> nn.Module:
    normalized_name = name.lower()
    if normalized_name == "ce":
        return nn.CrossEntropyLoss()
    if normalized_name == "weighted_ce":
        if class_weights is None:
            raise ValueError("weighted_ce requires class_weights")
        return nn.CrossEntropyLoss(weight=class_weights)
    if normalized_name == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=class_weights)
    raise ValueError(f"Unsupported loss: {name}")

