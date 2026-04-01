#!/bin/bash
# Usage: bash scripts/eval_TN3K.sh <nnUNetTrainerUMambaEnc|nnUNetTrainerUMambaBot> [checkpoint_name]
# Example:
#   bash scripts/eval_TN3K.sh nnUNetTrainerUMambaEnc
#   bash scripts/eval_TN3K.sh nnUNetTrainerUMambaEnc checkpoint_best.pth
#   bash scripts/eval_TN3K.sh nnUNetTrainerUMambaBot checkpoint_final.pth

MAMBA_MODEL=$1
CHECKPOINT=${2:-"checkpoint_best.pth"}
PRED_OUTPUT_PATH="data/nnUNet_results/Dataset705_TN3K/${MAMBA_MODEL}__nnUNetPlans__2d/pred_results"
EVAL_METRIC_PATH="data/nnUNet_results/Dataset705_TN3K/${MAMBA_MODEL}__nnUNetPlans__2d"
GPU_ID="0"

echo "Predicting with ${MAMBA_MODEL} (${CHECKPOINT})..." &&
CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_predict \
    -i "data/nnUNet_raw/Dataset705_TN3K/imagesTs" \
    -o "${PRED_OUTPUT_PATH}" \
    -d 705 \
    -c 2d \
    -tr "${MAMBA_MODEL}" \
    --disable_tta \
    -f all \
    -chk "${CHECKPOINT}" &&

echo "Evaluating (DSC, IoU, Precision, Recall, HD95)..."
python evaluation/eval_2d_common.py \
    --gt_path "data/nnUNet_raw/Dataset705_TN3K/labelsTs" \
    --seg_path "${PRED_OUTPUT_PATH}" \
    --save_path "${EVAL_METRIC_PATH}/metric_all.csv" &&

echo "Done. Results saved to ${EVAL_METRIC_PATH}/metric_all.csv"
