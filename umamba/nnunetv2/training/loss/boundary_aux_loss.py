"""
Boundary auxiliary loss and deep supervision wrapper for BAGD.

Provides:
- compute_boundary_gt: extracts boundary GT from segmentation GT via morphological ops
- BoundaryLoss: Dice + BCE loss on boundary maps
- DeepSupervisionWrapperBAGD: wraps seg loss + boundary loss for multi-scale outputs
"""

import torch
from torch import nn
from torch.nn import functional as F
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.utilities.helpers import softmax_helper_dim1


def compute_boundary_gt(seg_gt: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Extract boundary from binary segmentation ground truth.

    Uses morphological dilation - erosion to get boundary region.

    Args:
        seg_gt: (B, 1, H, W) segmentation GT, values in {0, 1} or long
        kernel_size: size of morphological kernel (default 3)

    Returns:
        (B, 1, H, W) boundary map, float, values in {0, 1}
    """
    seg_float = seg_gt.float()
    pad = kernel_size // 2
    dilated = F.max_pool2d(seg_float, kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-seg_float, kernel_size, stride=1, padding=pad)
    boundary = (dilated - eroded).clamp(0, 1)
    return boundary


class BoundaryLoss(nn.Module):
    """Binary boundary loss: Dice + BCE on boundary maps."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def _dice_loss(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss on boundary maps.

        Uses per-sample Dice (sum over spatial dims only, then mean over batch)
        to avoid fp16 overflow when summing over the entire batch.
        """
        pred = torch.sigmoid(pred_logits)
        axes = tuple(range(2, pred.ndim))  # spatial dims
        # Per-sample computation: (B, 1) after spatial sum
        intersection = (pred * target).sum(dim=axes)
        sum_pred = pred.sum(dim=axes)
        sum_gt = target.sum(dim=axes)
        dice = (2 * intersection + self.smooth) / (sum_pred + sum_gt + self.smooth)
        return 1 - dice.mean()

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits: (B, 1, H, W) raw logits from BoundaryHead
            target: (B, 1, H, W) boundary GT, float {0, 1}
        """
        # Force fp32 to prevent fp16 overflow in loss computation
        pred_logits = pred_logits.float()
        target = target.float()
        bce_loss = self.bce(pred_logits, target)
        dice_loss = self._dice_loss(pred_logits, target)
        return bce_loss + dice_loss


class DeepSupervisionWrapperBAGD(nn.Module):
    """Deep supervision wrapper that handles both segmentation and boundary outputs.

    Expects forward(seg_outputs, boundary_outputs, targets) during training.
    """

    def __init__(self, seg_loss, weight_factors, lambda_boundary=0.3,
                 boundary_kernel=3):
        super().__init__()
        assert any([x != 0 for x in weight_factors])
        self.weight_factors = tuple(weight_factors)
        self.seg_loss = seg_loss
        self.lambda_boundary = lambda_boundary
        self.boundary_kernel = boundary_kernel
        self.boundary_loss = BoundaryLoss()

    def forward(self, seg_outputs, boundary_outputs, targets):
        """
        Args:
            seg_outputs: list of (B, C, H_s, W_s) segmentation predictions at each scale
            boundary_outputs: list of (B, 1, H_s, W_s) boundary logits at each scale
            targets: list of (B, 1, H_s, W_s) segmentation targets at each scale
        """
        weights = self.weight_factors

        # 1. Segmentation loss (same as original DeepSupervisionWrapper)
        total_seg = sum(
            weights[i] * self.seg_loss(seg_outputs[i], targets[i])
            for i in range(len(seg_outputs))
            if weights[i] != 0.0
        )

        # 2. Boundary loss at each scale
        bnd_losses = []
        for i in range(len(boundary_outputs)):
            if weights[i] == 0.0:
                continue
            bnd_gt = compute_boundary_gt(targets[i], self.boundary_kernel)
            # Skip boundary loss if no boundary pixels at this scale
            if bnd_gt.sum() < 1:
                continue
            bnd_losses.append(weights[i] * self.boundary_loss(
                boundary_outputs[i], bnd_gt
            ))

        if bnd_losses:
            return total_seg + self.lambda_boundary * sum(bnd_losses)
        return total_seg
