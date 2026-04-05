"""
UMambaEnc 2D with Boundary-Aware Gated Decoder (BAGD).

Modifications over the original UMambaEnc_2d:
1. AttentionGate at each skip connection to filter noisy encoder features
2. BoundaryHead at each decoder scale for auxiliary boundary supervision

The encoder (ResidualMambaEncoder) is unchanged.
"""

import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from dynamic_network_architectures.building_blocks.residual import BasicBlockD

# Reuse encoder and building blocks from the original file
from nnunetv2.nets.UMambaEnc_2d import (
    MambaLayer, BasicResBlock, ResidualMambaEncoder, UpsampleLayer
)


class AttentionGate(nn.Module):
    """Attention gate for skip connections.

    Uses decoder (gating) features to produce spatial attention weights
    that filter encoder skip features, suppressing irrelevant activations.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int,
                 norm_op=nn.InstanceNorm2d, nonlin=nn.LeakyReLU):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, bias=False)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, bias=False)
        self.norm_g = norm_op(F_int, affine=True)
        self.norm_x = norm_op(F_int, affine=True)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.relu = nonlin(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: gating signal from decoder (upsampled features), (B, F_g, H, W)
            x: encoder skip connection features, (B, F_l, H, W)
        Returns:
            Attention-weighted skip features, (B, F_l, H, W)
        """
        att = self.relu(self.norm_g(self.W_g(g)) + self.norm_x(self.W_x(x)))
        att = self.psi(att)  # (B, 1, H, W)
        return x * att


class BoundaryHead(nn.Module):
    """Lightweight boundary prediction head (1x1 conv to 1 channel)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # (B, 1, H, W), raw logits


class UNetResDecoderBAGD(nn.Module):
    """U-Net residual decoder with Attention Gates and Boundary Heads."""

    def __init__(self, encoder, num_classes,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1

        stages = []
        upsample_layers = []
        seg_layers = []
        attention_gates = []
        boundary_heads = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]

            upsample_layers.append(UpsampleLayer(
                conv_op=encoder.conv_op,
                input_channels=input_features_below,
                output_channels=input_features_skip,
                pool_op_kernel_size=stride_for_upsampling,
                mode='nearest'
            ))

            # Attention gate for skip connection (not needed for last stage which has no skip)
            if s < n_stages_encoder - 1:
                F_int = max(input_features_skip // 2, 16)
                attention_gates.append(AttentionGate(
                    F_g=input_features_skip,   # decoder upsampled channels
                    F_l=input_features_skip,   # encoder skip channels
                    F_int=F_int,
                    norm_op=encoder.norm_op,
                    nonlin=encoder.nonlin
                ))
            else:
                attention_gates.append(None)  # placeholder, won't be used

            stages.append(nn.Sequential(
                BasicResBlock(
                    conv_op=encoder.conv_op,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                    input_channels=2 * input_features_skip if s < n_stages_encoder - 1 else input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)],
                    padding=encoder.conv_pad_sizes[-(s + 1)],
                    stride=1,
                    use_1x1conv=True
                ),
                *[
                    BasicBlockD(
                        conv_op=encoder.conv_op,
                        input_channels=input_features_skip,
                        output_channels=input_features_skip,
                        kernel_size=encoder.kernel_sizes[-(s + 1)],
                        stride=1,
                        conv_bias=encoder.conv_bias,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        nonlin=encoder.nonlin,
                        nonlin_kwargs=encoder.nonlin_kwargs,
                    ) for _ in range(n_conv_per_stage[s - 1] - 1)
                ]
            ))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))
            boundary_heads.append(BoundaryHead(input_features_skip))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)
        # Filter out None placeholders for ModuleList
        self.attention_gates = nn.ModuleList(
            [ag for ag in attention_gates if ag is not None]
        )
        self.boundary_heads = nn.ModuleList(boundary_heads)
        # Store which decoder stages have attention gates (all except the last)
        self._ag_stage_indices = [i for i, ag in enumerate(attention_gates) if ag is not None]

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        boundary_outputs = []
        ag_idx = 0

        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            if s < (len(self.stages) - 1):
                skip = skips[-(s + 2)]
                # Apply attention gate
                if s in self._ag_stage_indices:
                    skip = self.attention_gates[ag_idx](x, skip)
                    ag_idx += 1
                x = torch.cat((x, skip), 1)
            x = self.stages[s](x)

            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
                boundary_outputs.append(self.boundary_heads[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]
        boundary_outputs = boundary_outputs[::-1]

        if not self.deep_supervision:
            return seg_outputs[0]  # inference: single tensor
        else:
            return seg_outputs, boundary_outputs  # training: tuple of lists

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
                # boundary head adds 1 channel per scale
                output += np.prod([1, *skip_sizes[-(s + 1)]], dtype=np.int64)
        return output


class UMambaEncBAGD(nn.Module):
    """UMambaEnc with Boundary-Aware Gated Decoder."""

    def __init__(self,
                 input_size: Tuple[int, ...],
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 stem_channels: int = None
                 ):
        super().__init__()
        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks_per_stage[s] = 1

        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_per_stage_decoder[s] = 1

        assert len(n_blocks_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == (n_stages - 1)

        # Encoder is identical to original UMambaEnc
        self.encoder = ResidualMambaEncoder(
            input_size,
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            stem_channels=stem_channels
        )

        # Decoder with Attention Gates and Boundary Heads
        self.decoder = UNetResDecoderBAGD(
            self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        return self.encoder.compute_conv_feature_map_size(input_size) + \
               self.decoder.compute_conv_feature_map_size(input_size)


def get_umamba_enc_bagd_2d_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True
    ):
    num_stages = len(configuration_manager.conv_kernel_sizes)
    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)
    label_manager = plans_manager.get_label_manager(dataset_json)

    kwargs = {
        'input_size': configuration_manager.patch_size,
        'conv_bias': True,
        'norm_op': get_matching_instancenorm(conv_op),
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'dropout_op': None, 'dropout_op_kwargs': None,
        'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
    }

    model = UMambaEncBAGD(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        n_conv_per_stage=configuration_manager.n_conv_per_stage_encoder,
        n_conv_per_stage_decoder=configuration_manager.n_conv_per_stage_decoder,
        deep_supervision=deep_supervision,
        **kwargs
    )
    model.apply(InitWeights_He(1e-2))

    return model
