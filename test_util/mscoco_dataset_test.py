import sys
sys.path.append('./')

import tensorflow as tf
from datasets.mscoco_dataset import MSCOCODataset
import cv2
import numpy as np

sess = tf.Session()
dataset = MSCOCODataset(is_train=False, img_size=304)
dataiter = dataset.build(1)
img_info, image, groundtruth = dataiter.get_next()

for i in range(10):
    gt = sess.run([img_info, image, groundtruth])
    img = gt[1][0].astype(np.uint8)
    height, width, channel = img.shape
    for line in gt[2][0]:
        if line[0]:
            cv2.rectangle(img, (int(height * line[2]), int(width * line[1])),
                          (int(height * line[4]), int(width * line[3])),
                          (0, 255, 0), 1)
    cv2.imshow('win', img)
    cv2.waitKey()