MAMBA_MODEL=$1
PRED_OUTPUT_PATH="data/nnUNet_results/Dataset705_TN3K/${MAMBA_MODEL}__nnUNetPlans__2d/pred_results"
EVAL_METRIC_PATH="data/nnUNet_results/Dataset705_TN3K/${MAMBA_MODEL}__nnUNetPlans__2d"
GPU_ID="0"

# CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_train 705 2d all -tr ${MAMBA_MODEL} &&

echo "Predicting..." &&
CUDA_VISIBLE_DEVICES=${GPU_ID} nnUNetv2_predict \
    -i "data/nnUNet_raw/Dataset705_TN3K/imagesTs" \
    -o "${PRED_OUTPUT_PATH}" \
    -d 705 \
    -c 2d \
    -tr "${MAMBA_MODEL}" \
    --disable_tta \
    -f all \
    -chk "checkpoint_best.pth" &&

echo "Computing dice..."
python evaluation/eval_2d_DSC.py \
    --gt_path "data/nnUNet_raw/Dataset705_TN3K/labelsTs" \
    --seg_path "${PRED_OUTPUT_PATH}" \
    --save_path "${EVAL_METRIC_PATH}/metric_DSC.csv" &&

echo "Computing NSD..."
python evaluation/eval_2d_NSD.py \
    --gt_path "data/nnUNet_raw/Dataset705_TN3K/labelsTs" \
    --seg_path "${PRED_OUTPUT_PATH}" \
    --save_path "${EVAL_METRIC_PATH}/metric_NSD.csv" &&

echo "Done."
