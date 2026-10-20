"""Loss functions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss (https://arxiv.org/abs/1708.02002) for binary and multi-class classification.

    Adapted from https://github.com/itakurah/Focal-loss-PyTorch.

    Parameters
    ----------
    gamma : float
        Focusing parameter: strength of the modulating factor (1 - p_t)^gamma.
    alpha : float, torch.Tensor or None
        Class balancing: weight of the positive class (binary) or per-class weights (multi-class).
    reduction : {"mean", "sum", "none"}
    task_type : {"binary", "multi-class"}
        Binary: ``inputs`` are logits of the positive class, shape (batch,). Multi-class: logits of shape
        (batch, num_classes).
    num_classes : int, optional
        Number of classes (multi-class only).
    """

    def __init__(self, gamma=2, alpha=None, reduction="mean", task_type="binary", num_classes=None):
        super().__init__()
        if task_type not in ("binary", "multi-class"):
            raise ValueError(f"Unsupported task_type {task_type!r}; use 'binary' or 'multi-class'.")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        if self.task_type == "binary":
            loss = self._binary(inputs, targets)
        else:
            loss = self._multi_class(inputs, targets)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def _binary(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss
        return focal_weight * bce_loss

    def _multi_class(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        ce_loss = -targets_one_hot * torch.log(probs)
        p_t = torch.sum(probs * targets_one_hot, dim=1)
        focal_weight = (1 - p_t) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device).gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss
        return focal_weight.unsqueeze(1) * ce_loss
