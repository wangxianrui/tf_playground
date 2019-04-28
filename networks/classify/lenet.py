import tensorflow as tf

from parameters import ClassifyParam


class LeNet:
    def __init__(self):
        pass

    def forward(self, inputs, is_train):
        with tf.variable_scope('layer1'):
            inputs = tf.layers.Conv2D(32, 5, 1)(inputs)
            inputs = tf.nn.relu(inputs)
            inputs = tf.layers.MaxPooling2D(2, 2)(inputs)
        with tf.variable_scope('layer2'):
            inputs = tf.layers.Conv2D(64, 5, 1)(inputs)
            inputs = tf.nn.relu(inputs)
            inputs = tf.layers.MaxPooling2D(2, 2)(inputs)
        with tf.variable_scope('layer3'):
            # inputs_size = inputs.get_shape().as_list()[1:-1]
            # inputs = tf.image.resize_images(inputs, [inputs_size[0] * 2, inputs_size[1] * 2])
            inputs = tf.layers.Flatten()(inputs)
            inputs = tf.layers.Dense(256)(inputs)
            inputs = tf.nn.relu(inputs)
        with tf.variable_scope('layer4'):
            inputs = tf.layers.Dense(ClassifyParam.num_classes)(inputs)
        return inputs

    def calc_loss(self, logits, labels, trainabel_vars):
        loss = tf.losses.softmax_cross_entropy(labels, logits)
        loss_filter = lambda var: 'batch_norm' not in var.name
        loss += ClassifyParam.weight_decay * tf.add_n(
            [tf.nn.l2_loss(var) for var in trainabel_vars if loss_filter(var)])
        return loss

    def calc_accuracy(self, logits, labels):
        target = tf.argmax(labels, 1)
        acc_top1 = tf.reduce_mean(tf.to_float(tf.nn.in_top_k(tf.nn.softmax(logits), target, 1)))
        acc_top5 = tf.reduce_mean(tf.to_float(tf.nn.in_top_k(tf.nn.softmax(logits), target, 5)))
        return acc_top1, acc_top5
