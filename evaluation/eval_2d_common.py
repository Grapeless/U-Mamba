# -*- coding: utf-8 -*-
"""
Unified 2D segmentation evaluation script.
Computes: DSC, IoU, Precision, Recall, HD95
"""

import numpy as np
import cv2
import os
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_robust_hausdorff, compute_dice_coefficient
from tqdm import tqdm

join = os.path.join
basename = os.path.basename

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gt_path', type=str, required=True)
parser.add_argument('--seg_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)
args = parser.parse_args()

gt_path = args.gt_path
seg_path = args.seg_path
save_path = args.save_path

filenames = sorted([x for x in os.listdir(seg_path) if x.endswith('.png')])

seg_metrics = OrderedDict(
    Name=list(),
    DSC=list(),
    IoU=list(),
    Precision=list(),
    Recall=list(),
    HD95=list(),
)

for name in tqdm(filenames):
    seg_metrics['Name'].append(name)

    gt_raw = cv2.imread(join(gt_path, name), cv2.IMREAD_GRAYSCALE)
    seg_raw = cv2.imread(join(seg_path, name), cv2.IMREAD_GRAYSCALE)

    if gt_raw is None:
        print(f"WARNING: GT file not found for {name}, skipping metrics (all 0).")
        seg_metrics['DSC'].append(0.0)
        seg_metrics['IoU'].append(0.0)
        seg_metrics['Precision'].append(0.0)
        seg_metrics['Recall'].append(0.0)
        seg_metrics['HD95'].append(np.inf)
        continue

    gt_data = np.uint8(gt_raw)
    seg_data = np.uint8(seg_raw)

    gt_labels = np.unique(gt_data)[1:]
    seg_labels = np.unique(seg_data)[1:]
    labels = np.union1d(gt_labels, seg_labels)

    if len(labels) == 0:
        seg_metrics['DSC'].append(1.0)
        seg_metrics['IoU'].append(1.0)
        seg_metrics['Precision'].append(1.0)
        seg_metrics['Recall'].append(1.0)
        seg_metrics['HD95'].append(0.0)
        continue

    dsc_arr, iou_arr, prec_arr, rec_arr, hd95_arr = [], [], [], [], []

    for i in labels:
        gt_i = (gt_data == i)
        seg_i = (seg_data == i)
        gt_sum = np.sum(gt_i)
        seg_sum = np.sum(seg_i)

        if gt_sum == 0 and seg_sum == 0:
            dsc_arr.append(1.0)
            iou_arr.append(1.0)
            prec_arr.append(1.0)
            rec_arr.append(1.0)
            hd95_arr.append(0.0)
        elif gt_sum == 0 or seg_sum == 0:
            dsc_arr.append(0.0)
            iou_arr.append(0.0)
            prec_arr.append(0.0)
            rec_arr.append(0.0)
            hd95_arr.append(np.inf)
        else:
            tp = np.sum(gt_i & seg_i)
            fp = np.sum(~gt_i & seg_i)
            fn = np.sum(gt_i & ~seg_i)

            dsc_arr.append(compute_dice_coefficient(gt_i, seg_i))
            iou_arr.append(tp / (tp + fp + fn))
            prec_arr.append(tp / (tp + fp))
            rec_arr.append(tp / (tp + fn))

            surface_distances = compute_surface_distances(gt_i[..., None], seg_i[..., None], [1, 1, 1])
            hd95_arr.append(compute_robust_hausdorff(surface_distances, 95))

    seg_metrics['DSC'].append(round(np.mean(dsc_arr), 4))
    seg_metrics['IoU'].append(round(np.mean(iou_arr), 4))
    seg_metrics['Precision'].append(round(np.mean(prec_arr), 4))
    seg_metrics['Recall'].append(round(np.mean(rec_arr), 4))
    hd95_finite = [x for x in hd95_arr if np.isfinite(x)]
    hd95_mean = np.mean(hd95_finite) if hd95_finite else np.inf
    seg_metrics['HD95'].append(round(hd95_mean, 4) if np.isfinite(hd95_mean) else np.inf)

dataframe = pd.DataFrame(seg_metrics)

# Summary
total = len(dataframe)
hd95_col = dataframe['HD95'].replace(np.inf, np.nan)
failed = dataframe['DSC'].eq(0.0).sum()

summary_text = (f"Results for {basename(seg_path)} ({total} samples, {failed} failed): "
                f"DSC: {dataframe['DSC'].mean():.4f}  "
                f"IoU: {dataframe['IoU'].mean():.4f}  "
                f"Precision: {dataframe['Precision'].mean():.4f}  "
                f"Recall: {dataframe['Recall'].mean():.4f}  "
                f"HD95: {hd95_col.mean():.4f} ({hd95_col.notna().sum()}/{total} finite)")

# Append summary as last row
summary_row = pd.DataFrame([{
    'Name': summary_text,
    'DSC': round(dataframe['DSC'].mean(), 4),
    'IoU': round(dataframe['IoU'].mean(), 4),
    'Precision': round(dataframe['Precision'].mean(), 4),
    'Recall': round(dataframe['Recall'].mean(), 4),
    'HD95': round(hd95_col.mean(), 4),
}])
dataframe = pd.concat([dataframe, summary_row], ignore_index=True)
dataframe.to_csv(save_path, index=False)

print(20 * '>')
print(summary_text)
print(20 * '<')
