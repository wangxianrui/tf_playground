import numpy as np


def areas(gt_bboxes):
    xmin, ymin, xmax, ymax = np.split(gt_bboxes, 4, axis=1)
    return (xmax - xmin) * (ymax - ymin)


def intersection(gt_bboxes, pri_bboxes):
    # num_anchors * 1
    gt_xmin, gt_ymin, gt_xmax, gt_ymax = np.split(gt_bboxes, 4, axis=1)
    # 1 * num_anchors
    pri_xmin, pri_ymin, pri_xmax, pri_ymax = [np.transpose(a, [1, 0]) for a in np.split(pri_bboxes, 4, axis=1)]
    int_xmin = np.maximum(gt_xmin, pri_xmin)
    int_ymin = np.maximum(gt_ymin, pri_ymin)
    int_xmax = np.minimum(gt_xmax, pri_xmax)
    int_ymax = np.minimum(gt_ymax, pri_ymax)
    w = np.maximum(int_xmax - int_xmin, 0)
    h = np.maximum(int_ymax - int_ymin, 0)
    return w * h


print(intersection(np.random.random([10, 4]) * 20, np.random.random([10, 4]) * 20))
