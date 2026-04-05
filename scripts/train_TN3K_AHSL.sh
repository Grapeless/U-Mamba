#!/bin/bash
# Usage:
#   bash scripts/train_TN3K_AHSL.sh         # train from scratch
#   bash scripts/train_TN3K_AHSL.sh --c     # resume from checkpoint

export OMP_NUM_THREADS=8

export nnUNet_raw="data/nnUNet_raw"
export nnUNet_preprocessed="data/nnUNet_preprocessed"
export nnUNet_results="data/nnUNet_results"

cd "$(dirname "$0")/.." || exit 1
UMAMBA_ENV="/root/miniconda3/envs/umamba"
CUDA_VISIBLE_DEVICES=0 "$UMAMBA_ENV/bin/nnUNetv2_train" 705 2d all -tr nnUNetTrainerUMambaEnc_AHSL $1
