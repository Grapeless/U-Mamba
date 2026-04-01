#!/bin/bash
# Usage:
#   bash scripts/train_TN3K.sh <nnUNetTrainerUMambaEnc|nnUNetTrainerUMambaBot> [--c]
# Example:
#   bash scripts/train_TN3K.sh nnUNetTrainerUMambaEnc        # 从头训练
#   bash scripts/train_TN3K.sh nnUNetTrainerUMambaEnc --c    # 从检查点继续训练

MAMBA_MODEL=$1
RESUME_FLAG=$2
GPU_ID="0"

echo "Training ${MAMBA_MODEL} on Dataset705_TN3K..."
CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_train 705 2d all -tr ${MAMBA_MODEL} ${RESUME_FLAG}
