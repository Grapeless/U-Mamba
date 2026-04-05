#!/bin/bash
# Usage:
#   bash scripts/train_TN3K_BAGD_AHSL.sh         # 从头训练
#   bash scripts/train_TN3K_BAGD_AHSL.sh --c     # 从检查点继续训练

# Fix OMP_NUM_THREADS (autodl sets it to 0, causing libgomp errors)
export OMP_NUM_THREADS=8

# nnUNet env
export nnUNet_raw="data/nnUNet_raw"
export nnUNet_preprocessed="data/nnUNet_preprocessed"
export nnUNet_results="data/nnUNet_results"

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 705 2d all -tr nnUNetTrainerUMambaEnc_BAGD_AHSL $1
