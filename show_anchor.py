'''
@Author: rayenwang
@LastEditTime: 2019-09-04 09:13:33
@Description: 
'''
import numpy as np
import math
import cv2

img_shape = (300, 300)
num_classes = 21
feat_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
anchor_size_bounds = [0.15, 0.90]
anchor_sizes = [(21., 45.), (45., 99.), (99., 153.), (153., 207.), (207., 261.), (261., 315.)]
anchor_ratios = [[2, .5, 3, 1. / 3], [2, .5, 3, 1. / 3], [2, .5, 3, 1. / 3], [2, .5, 3, 1. / 3], [2, .5, 3, 1. / 3],
                 [2, .5, 3, 1. / 3]]
anchor_steps = [8, 16, 32, 64, 100, 300]
anchor_offset = 0.5
normalizations = [20, -1, -1, -1, -1, -1]
prior_scaling = [0.1, 0.1, 0.2, 0.2]


def center_2_corner(bbox):
    x1 = np.maximum(0, bbox[:, 0] - bbox[:, 2] / 2).reshape(-1, 1)
    y1 = np.maximum(0, bbox[:, 1] - bbox[:, 3] / 2).reshape(-1, 1)
    x2 = np.minimum(1, bbox[:, 0] + bbox[:, 2] / 2).reshape(-1, 1)
    y2 = np.minimum(1, bbox[:, 1] + bbox[:, 3] / 2).reshape(-1, 1)
    return np.concatenate([x1, y1, x2, y2], axis=1)


def ssd_anchor_one_layer(img_shape, feat_shape, sizes, ratios, step, offset=0.5, dtype=np.float32):
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i + di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i + di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return x, y, w, h


def iou(gt, bbox):
    xmin = np.maximum(gt[:, 0], bbox[:, 0])
    ymin = np.maximum(gt[:, 1], bbox[:, 1])
    xmax = np.minimum(gt[:, 2], bbox[:, 2])
    ymax = np.minimum(gt[:, 3], bbox[:, 3])
    inter = np.maximum(0, xmax - xmin) * np.maximum(0, (ymax - ymin))
    union = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]) + (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] -
                                                                                         bbox[:, 1]) - inter
    return inter / union


bbox = []
for shape, size, ratio, step in zip(feat_shapes[::-1], anchor_sizes[::-1], anchor_ratios[::-1], anchor_steps[::-1]):
    x, y, w, h = ssd_anchor_one_layer(img_shape, shape, size, ratio, step)
    x = np.repeat(np.expand_dims(x, 2), w.shape[0], 2)
    y = np.repeat(np.expand_dims(y, 2), w.shape[0], 2)
    w = np.expand_dims(
        np.repeat(np.expand_dims(np.repeat(np.expand_dims(w, 0), x.shape[0], axis=0), 0), x.shape[0], axis=0), -1)
    h = np.expand_dims(
        np.repeat(np.expand_dims(np.repeat(np.expand_dims(h, 0), x.shape[0], axis=0), 0), x.shape[0], axis=0), -1)
    layer = np.reshape(np.concatenate([x, y, w, h], axis=-1), [-1, 4])
    bbox.append(center_2_corner(layer))
# bbox = np.concatenate(bbox, axis=0)
# img = np.zeros([300, 300, 3]).astype(np.uint8)
# for b in bbox:
#     x1, y1, x2, y2 = int(b[0] * 300), int(b[1] * 300), int(b[2] * 300), int(b[3] * 300)
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
# cv2.imwrite('test.jpg', img)
gt = np.asarray([0.3, 0.3, 0.8, 0.8]).reshape(1, 4)
for b in bbox:
    iou_score = iou(gt, b)
    print(sorted(iou_score, reverse=True)[:10])
