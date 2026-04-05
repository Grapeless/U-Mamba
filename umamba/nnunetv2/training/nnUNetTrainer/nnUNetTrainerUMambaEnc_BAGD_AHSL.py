"""
UMambaEnc trainer with BAGD decoder + AHSL loss (E3 experiment).

Combines:
- Module 1 (BAGD): Attention-gated skip connections + boundary supervision
- Module 2 (AHSL): Adaptive hard sample loss with progressive activation
"""

import numpy as np
import torch
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaEnc_BAGD import nnUNetTrainerUMambaEnc_BAGD
from nnunetv2.training.loss.adaptive_hard_sample_loss import AdaptiveHardSampleLoss
from nnunetv2.training.loss.boundary_aux_loss import DeepSupervisionWrapperBAGD
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


class nnUNetTrainerUMambaEnc_BAGD_AHSL(nnUNetTrainerUMambaEnc_BAGD):
    """UMambaEnc with BAGD decoder and AHSL loss.

    Inherits BAGD's network architecture and train/validation step handling.
    Overrides _build_loss to use AHSL as the seg loss within DeepSupervisionWrapperBAGD.
    """

    def _build_loss(self):
        seg_loss = AdaptiveHardSampleLoss(
            soft_dice_kwargs={'batch_dice': self.configuration_manager.batch_dice,
                              'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            ce_kwargs={},
            weight_ce=1, weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
            gamma_max=2.0, warmup_end=150, rampup_end=400,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapperBAGD(
                seg_loss, weights,
                lambda_boundary=0.3,
                boundary_kernel=3
            )
        else:
            loss = seg_loss
        return loss

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        # Update AHSL gamma schedule
        # seg_loss is inside DeepSupervisionWrapperBAGD
        if hasattr(self.loss, 'seg_loss') and hasattr(self.loss.seg_loss, 'set_epoch'):
            self.loss.seg_loss.set_epoch(self.current_epoch)
            self.print_to_log_file(f"AHSL gamma: {self.loss.seg_loss.current_gamma:.4f}")
        elif hasattr(self.loss, 'set_epoch'):
            self.loss.set_epoch(self.current_epoch)
            self.print_to_log_file(f"AHSL gamma: {self.loss.current_gamma:.4f}")
