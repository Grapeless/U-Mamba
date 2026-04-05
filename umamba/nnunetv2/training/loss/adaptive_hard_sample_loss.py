import torch
from torch import nn
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


class AdaptiveHardSampleLoss(nn.Module):
    """
    Adaptive Hard Sample Loss (AHSL).

    Wraps Dice+CE loss with sample-level adaptive weighting:
    samples with lower online Dice scores receive higher loss weights.

    Weight formula: w_i = (1 - dice_i) ^ gamma, normalized to mean=1.

    Progressive activation schedule:
      - epoch < warmup_end:       gamma = 0 (standard Dice+CE)
      - warmup_end <= epoch < rampup_end: gamma linearly increases to gamma_max
      - epoch >= rampup_end:      gamma = gamma_max
    """

    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1,
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss,
                 gamma_max=2.0, warmup_end=150, rampup_end=400):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(reduction='none', **ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

        self.gamma_max = gamma_max
        self.warmup_end = warmup_end
        self.rampup_end = rampup_end
        self._current_gamma = 0.0

    def set_epoch(self, epoch):
        if epoch < self.warmup_end:
            self._current_gamma = 0.0
        elif epoch < self.rampup_end:
            self._current_gamma = self.gamma_max * (epoch - self.warmup_end) / (self.rampup_end - self.warmup_end)
        else:
            self._current_gamma = self.gamma_max

    @property
    def current_gamma(self):
        return self._current_gamma

    def _per_sample_dice(self, net_output, target):
        """Compute per-sample foreground Dice score (detached, no grad)."""
        with torch.no_grad():
            x = softmax_helper_dim1(net_output)
            if x.ndim != target.ndim:
                target = target.view((target.shape[0], 1, *target.shape[1:]))
            if x.shape == target.shape:
                y_onehot = target
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, target.long(), 1)

            # foreground only (skip bg channel 0)
            x_fg = x[:, 1:]
            y_fg = y_onehot[:, 1:].float()

            axes = tuple(range(2, x.ndim))
            intersect = (x_fg * y_fg).sum(dim=axes).sum(dim=1)  # (B,)
            sum_pred = x_fg.sum(dim=axes).sum(dim=1)
            sum_gt = y_fg.sum(dim=axes).sum(dim=1)

            dice = (2 * intersect + 1e-5) / (sum_pred + sum_gt + 1e-5)  # (B,)
        return dice

    def _per_sample_ce(self, net_output, target):
        """Compute per-sample mean CE loss."""
        # target shape: (B, 1, ...) -> (B, ...)
        if target.ndim == net_output.ndim:
            t = target[:, 0]
        else:
            t = target
        ce_map = self.ce(net_output, t.long())  # (B, ...)
        axes = tuple(range(1, ce_map.ndim))
        return ce_map.mean(dim=axes)  # (B,)

    def _per_sample_dice_loss(self, net_output, target):
        """Compute per-sample Dice loss."""
        x = softmax_helper_dim1(net_output)
        if x.ndim != target.ndim:
            target = target.view((target.shape[0], 1, *target.shape[1:]))
        if x.shape == target.shape:
            y_onehot = target
        else:
            with torch.no_grad():
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, target.long(), 1)

        # foreground only
        x_fg = x[:, 1:]
        y_fg = y_onehot[:, 1:].float()

        axes = tuple(range(2, x.ndim))
        intersect = (x_fg * y_fg).sum(dim=axes)  # (B, C-1)

        with torch.no_grad():
            sum_gt = y_fg.sum(dim=axes)
        sum_pred = x_fg.sum(dim=axes)

        dc = (2 * intersect + 1e-5) / (torch.clamp(sum_gt + sum_pred + 1e-5, min=1e-8))  # (B, C-1)
        dc = dc.mean(dim=1)  # (B,)
        return -dc

    def forward(self, net_output, target):
        if self.ignore_label is not None:
            assert target.shape[1] == 1
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
        else:
            target_dice = target

        gamma = self._current_gamma

        if gamma < 1e-6:
            # Standard Dice+CE (no adaptive weighting)
            dc_loss = self.dc(net_output, target_dice) if self.weight_dice != 0 else 0
            ce_loss_map = self.ce(net_output, target_dice[:, 0].long() if target_dice.ndim == net_output.ndim else target_dice.long())
            ce_loss = ce_loss_map.mean() if self.weight_ce != 0 else 0
            return self.weight_ce * ce_loss + self.weight_dice * dc_loss

        # --- Adaptive weighting ---
        # Per-sample Dice score for weighting (detached)
        sample_dice = self._per_sample_dice(net_output, target_dice)  # (B,)

        # Compute weights: harder samples (lower dice) get higher weight
        weights = (1.0 - sample_dice) ** gamma  # (B,)
        weights = weights / (weights.mean() + 1e-8)  # normalize to mean=1

        # Per-sample losses
        dc_per_sample = self._per_sample_dice_loss(net_output, target_dice) if self.weight_dice != 0 else 0  # (B,)
        ce_per_sample = self._per_sample_ce(net_output, target_dice) if self.weight_ce != 0 else 0  # (B,)

        loss_per_sample = self.weight_dice * dc_per_sample + self.weight_ce * ce_per_sample  # (B,)
        loss = (weights * loss_per_sample).mean()
        return loss
