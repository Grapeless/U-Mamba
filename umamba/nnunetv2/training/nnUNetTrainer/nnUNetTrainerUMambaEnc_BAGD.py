"""
UMambaEnc trainer with Boundary-Aware Gated Decoder (BAGD).

Uses BAGD decoder (attention-gated skip connections + boundary supervision)
with the original UMambaEnc encoder (no structural changes to encoder).
"""

import numpy as np
import torch
from torch import autocast
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaEnc import nnUNetTrainerUMambaEnc
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.boundary_aux_loss import DeepSupervisionWrapperBAGD
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from nnunetv2.nets.UMambaEnc_2d_BAGD import get_umamba_enc_bagd_2d_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainerUMambaEnc_BAGD(nnUNetTrainerUMambaEnc):
    """UMambaEnc trainer with Boundary-Aware Gated Decoder.

    Overrides:
    - build_network_architecture: uses BAGD decoder
    - _build_loss: uses DeepSupervisionWrapperBAGD with boundary loss
    - train_step / validation_step: handles (seg, boundary) tuple output
    """

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 2:
            model = get_umamba_enc_bagd_2d_from_plans(
                plans_manager, dataset_json, configuration_manager,
                num_input_channels, deep_supervision=enable_deep_supervision
            )
        else:
            raise NotImplementedError("BAGD is currently only implemented for 2D")

        print("UMambaEnc_BAGD: {}".format(model))
        return model

    def _build_loss(self):
        seg_loss = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {}, weight_ce=1, weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
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

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            if isinstance(output, tuple):
                seg_output, boundary_output = output
                l = self.loss(seg_output, boundary_output, target)
            else:
                l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            if isinstance(output, tuple):
                seg_output, boundary_output = output
                l = self.loss(seg_output, boundary_output, target)
                # For pseudo dice computation, use highest resolution seg output
                output = seg_output[0]
                target = target[0]
            else:
                del data
                l = self.loss(output, target)
                if self.enable_deep_supervision:
                    output = output[0]
                    target = target[0]

        # Online evaluation: pseudo dice (same as parent)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard,
                'fp_hard': fp_hard, 'fn_hard': fn_hard}
