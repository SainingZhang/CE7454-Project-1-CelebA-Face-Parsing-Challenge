import os
import sys

import numpy as np
from PIL import Image
from sklearn.metrics import f1_score


def read_masks(path):
    mask = Image.open(path)
    mask = np.array(mask)

    return mask


# replace submit_dir to your result path here
submit_dir = ''

# replace truth_dir to ground-truth path here
truth_dir = ''

# replace output_dir to the desired output path, and you will find 'scores.txt' containing the calcuated mIoU
output_dir = 'score'

if not os.path.isdir(submit_dir):
    print("%s doesn't exist" % submit_dir)

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    submit_dir_list = os.listdir(submit_dir)
    if len(submit_dir_list) == 1:
        submit_dir = os.path.join(submit_dir, "%s" % submit_dir_list[0])
        assert os.path.isdir(submit_dir)

    area_intersect_all = np.zeros(19)
    area_union_all = np.zeros(19)
    f1_scores = []
    num_classes = 19
    for idx in range(1000):
        pred_mask = read_masks(os.path.join(submit_dir, "%s.png" % idx))
        gt_mask = read_masks(os.path.join(truth_dir, "%s.png" % idx))
        for cls_idx in range(19):
            area_intersect = np.sum(
                (pred_mask == gt_mask) * (pred_mask == cls_idx))

            area_pred_label = np.sum(pred_mask == cls_idx)
            area_gt_label = np.sum(gt_mask == cls_idx)
            area_union = area_pred_label + area_gt_label - area_intersect

            area_intersect_all[cls_idx] += area_intersect
            area_union_all[cls_idx] += area_union
        pred_mask_flat = pred_mask.flatten()
        gt_mask_flat = gt_mask.flatten()

        f1 = f1_score(gt_mask_flat, pred_mask_flat, average=None, labels=range(num_classes), zero_division=1)
        f1_scores.append(f1)

    iou_all = area_intersect_all / area_union_all * 100.0
    miou = iou_all.mean()
    mean_f1 = np.mean(f1_scores)

    # Create the evaluation score path
    output_filename = os.path.join(output_dir, 'unetscores.txt')

    with open(output_filename, 'w') as f3:
        f3.write('mIOU: {}'.format(miou))
        f3.write('F1 Score: {:.4f}\n\n'.format(mean_f1))
