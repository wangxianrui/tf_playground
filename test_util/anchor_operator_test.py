from networks.detection.anchor_operator import anchors_all_layers
import cv2
import tensorflow as tf
import numpy as np


sess = tf.Session()
all_anchors = sess.run(anchors_all_layers())
print(all_anchors)
image = np.zeros([320, 320], dtype=np.uint8) + 255
for anchor in all_anchors:
    cv2.rectangle(image, (int(320*(anchor[1] - anchor[3]/2)), int(320*(anchor[0] - anchor[2]/2))),
                  (int(320*(anchor[1] + anchor[3]/2)), int(320*(anchor[0] + anchor[2]/2))), (0, 255, 0), 1)
cv2.namedWindow('win', cv2.WINDOW_NORMAL)
cv2.imshow('win', image)
cv2.waitKey()
