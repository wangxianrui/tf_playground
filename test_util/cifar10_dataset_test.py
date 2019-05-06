import tensorflow as tf

from cifar10_dataset import Cifar10Dataset

sess = tf.Session()
dataset = Cifar10Dataset(False).build(32)
image, label = dataset.get_next()

for i in range(10):
    for a in sess.run([image, label]):
        print(a)
