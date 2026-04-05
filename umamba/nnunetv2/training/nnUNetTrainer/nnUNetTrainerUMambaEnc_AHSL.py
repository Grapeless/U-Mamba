import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaEnc import nnUNetTrainerUMambaEnc
from nnunetv2.training.loss.adaptive_hard_sample_loss import AdaptiveHardSampleLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


class nnUNetTrainerUMambaEnc_AHSL(nnUNetTrainerUMambaEnc):
    """UMambaEnc trainer with Adaptive Hard Sample Loss (AHSL).

    Uses the original UMambaEnc architecture (no structural changes),
    only replaces the loss function with AHSL.
    """

    def _build_loss(self):
        loss = AdaptiveHardSampleLoss(
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
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        # Update gamma schedule for AHSL
        if hasattr(self.loss, 'loss'):
            # Wrapped in DeepSupervisionWrapper
            self.loss.loss.set_epoch(self.current_epoch)
            self.print_to_log_file(f"AHSL gamma: {self.loss.loss.current_gamma:.4f}")
        elif hasattr(self.loss, 'set_epoch'):
            self.loss.set_epoch(self.current_epoch)
            self.print_to_log_file(f"AHSL gamma: {self.loss.current_gamma:.4f}")
