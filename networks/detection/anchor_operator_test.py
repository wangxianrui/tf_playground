import sys

sys.path.append('.')
from anchor_operator import *
import tensorflow as tf
sess = tf.Session()
all_anchors = get_all_anchors()
print(sess.run(all_anchors))
