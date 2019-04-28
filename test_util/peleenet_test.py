import sys

sys.path.append('.')

import tensorflow as tf
from networks.classify.peleenet import PeleeNetClassify
from networks.detection.peleenet_ssd import PeleeNetSSD


def test_peleenetclassify():
    sess = tf.Session()
    x = tf.random_normal([1, 224, 224, 3])
    y = PeleeNetClassify().forward(x, True)
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(y).shape)


def test_peleenetssd():
    sess = tf.Session()
    x = tf.random_normal([1, 304, 404, 3])
    y = PeleeNetSSD().forward(x, True)
    init = tf.global_variables_initializer()
    sess.run(init)
    loc, cla = sess.run(y)
    for l in loc:
        print(l.shape)
    for c in cla:
        print(c.shape)


test_peleenetclassify()
test_peleenetssd()
