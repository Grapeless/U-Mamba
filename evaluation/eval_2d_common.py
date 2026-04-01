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

    gt_data = np.uint8(cv2.imread(join(gt_path, name), cv2.IMREAD_UNCHANGED))
    seg_data = np.uint8(cv2.imread(join(seg_path, name), cv2.IMREAD_UNCHANGED))

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
        elif gt_sum == 0 and seg_sum > 0:
            dsc_arr.append(0.0)
            iou_arr.append(0.0)
            prec_arr.append(0.0)
            rec_arr.append(0.0)
            hd95_arr.append(np.inf)
        elif gt_sum > 0 and seg_sum == 0:
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
    hd95_mean = np.mean([x for x in hd95_arr if x != np.inf])
    seg_metrics['HD95'].append(round(hd95_mean, 4) if not np.isnan(hd95_mean) else np.inf)

dataframe = pd.DataFrame(seg_metrics)
dataframe.to_csv(save_path, index=False)

print(20 * '>')
avg = dataframe.mean(axis=0, numeric_only=True)
print(f"Results for {basename(seg_path)}:")
print(f"  DSC:       {avg['DSC']:.4f}")
print(f"  IoU:       {avg['IoU']:.4f}")
print(f"  Precision: {avg['Precision']:.4f}")
print(f"  Recall:    {avg['Recall']:.4f}")
hd95_valid = dataframe['HD95'].replace(np.inf, np.nan)
print(f"  HD95:      {hd95_valid.mean():.4f}")
print(20 * '<')
