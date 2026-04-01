#!/bin/bash
# Usage:
#   bash scripts/eval_TN3K.sh                        # 默认用 checkpoint_best.pth
#   bash scripts/eval_TN3K.sh checkpoint_final.pth   # 指定 checkpoint

CHECKPOINT=${1:-"checkpoint_best.pth"}
RESULT_DIR="data/nnUNet_results/Dataset705_TN3K/nnUNetTrainerUMambaEnc__nnUNetPlans__2d"

echo "Predicting (${CHECKPOINT})..." &&
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
    -i "data/nnUNet_raw/Dataset705_TN3K/imagesTs" \
    -o "${RESULT_DIR}/pred_results" \
    -d 705 -c 2d -f all \
    -tr nnUNetTrainerUMambaEnc \
    --disable_tta \
    -chk "${CHECKPOINT}" &&

echo "Evaluating (DSC, IoU, Precision, Recall, HD95)..."
python evaluation/eval_2d_common.py \
    --gt_path "data/nnUNet_raw/Dataset705_TN3K/labelsTs" \
    --seg_path "${RESULT_DIR}/pred_results" \
    --save_path "${RESULT_DIR}/metric_all.csv" &&

echo "Done. Results saved to ${RESULT_DIR}/metric_all.csv"
