#!/bin/bash
# Usage:
#   bash scripts/eval_TN3K_AHSL.sh                        # default checkpoint_best.pth
#   bash scripts/eval_TN3K_AHSL.sh checkpoint_final.pth   # specify checkpoint

export OMP_NUM_THREADS=8

export nnUNet_raw="data/nnUNet_raw"
export nnUNet_preprocessed="data/nnUNet_preprocessed"
export nnUNet_results="data/nnUNet_results"

CHECKPOINT=${1:-"checkpoint_best.pth"}
RESULT_DIR="data/nnUNet_results/Dataset705_TN3K/nnUNetTrainerUMambaEnc_AHSL__nnUNetPlans__2d"

cd "$(dirname "$0")/.." || exit 1

echo "Predicting (${CHECKPOINT})..." &&
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/umamba/bin/nnUNetv2_predict \
    -i "data/nnUNet_raw/Dataset705_TN3K/imagesTs" \
    -o "${RESULT_DIR}/pred_results" \
    -d 705 -c 2d -f all \
    -tr nnUNetTrainerUMambaEnc_AHSL \
    --disable_tta \
    -chk "${CHECKPOINT}" &&

echo "Evaluating (DSC, IoU, Precision, Recall, HD95)..." &&
/root/miniconda3/envs/umamba/bin/python evaluation/eval_2d_common.py \
    --gt_path "data/nnUNet_raw/Dataset705_TN3K/labelsTs" \
    --seg_path "${RESULT_DIR}/pred_results" \
    --save_path "${RESULT_DIR}/metric_all.csv" &&

echo "Done. Results saved to ${RESULT_DIR}/metric_all.csv"
