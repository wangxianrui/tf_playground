import tensorflow as tf


class PeleeNet:
    def __init__(self):
        self.inner_channels = [16, 32, 64, 64]
        self.extra_channel = 16

    def conv_bn_relu(self, inputs, is_train, output_channel, kernel_size, stride, padding='same', use_relu=True):
        conv = tf.layers.Conv2D(output_channel, kernel_size, stride, padding, use_bias=False)(inputs)
        conv_bn = tf.layers.BatchNormalization()(conv, training=is_train)
        if use_relu:
            return tf.nn.relu(conv_bn)
        else:
            return conv_bn

    def stem_block(self, inputs, is_train):
        stem1 = self.conv_bn_relu(inputs, is_train, 32, 3, 2)
        stem2 = self.conv_bn_relu(stem1, is_train, 16, 1, 1)
        stem3 = self.conv_bn_relu(stem2, is_train, 32, 3, 2)
        stem1_pool = tf.layers.MaxPooling2D(2, 2)(stem1)
        stem_cat = tf.concat([stem3, stem1_pool], axis=-1)
        stem_final = self.conv_bn_relu(stem_cat, is_train, 32, 1, 1)
        return stem_final

    def dense_block(self, inputs, is_train, inner_channel, extra_channel):
        dense_a1 = self.conv_bn_relu(inputs, is_train, inner_channel, 1, 1)
        dense_a2 = self.conv_bn_relu(dense_a1, is_train, extra_channel, 3, 1)
        dense_b1 = self.conv_bn_relu(inputs, is_train, inner_channel, 1, 1)
        dense_b2 = self.conv_bn_relu(dense_b1, is_train, extra_channel, 3, 1)
        dense_b3 = self.conv_bn_relu(dense_b2, is_train, extra_channel, 3, 1)
        dense_final = tf.concat([inputs, dense_a2, dense_b3], axis=-1)
        return dense_final

    def transition_block(self, inputs, is_train, is_pooling):
        input_channel = inputs.get_shape().as_list()[3]
        if is_pooling:
            trans_0 = self.conv_bn_relu(inputs, is_train, input_channel, 1, 1)
            trans_pool = tf.layers.MaxPooling2D(2, 2)(trans_0)
            return trans_pool
        else:
            trans_0 = self.conv_bn_relu(inputs, is_train, input_channel, 1, 1)
            return trans_0

    def __call__(self, inputs, is_train):

        with tf.variable_scope('stage_0'):
            with tf.variable_scope('stem_block'):
                stage_0 = self.stem_block(inputs, is_train)

        with tf.variable_scope('stage_1'):
            stage_1_input = stage_0
            scope_name = 'dense_block_%d'
            for i in range(3):
                with tf.variable_scope(scope_name % i):
                    stage_1_input = self.dense_block(stage_1_input, is_train, self.inner_channels[0],
                                                     self.extra_channel)
            with tf.variable_scope('transition_block'):
                stage_1 = self.transition_block(stage_1_input, is_train, is_pooling=True)

        with tf.variable_scope('stage_2'):
            stage_2_input = stage_1
            scope_name = 'dense_block_%d'
            for i in range(4):
                with tf.variable_scope(scope_name % i):
                    stage_2_input = self.dense_block(stage_2_input, is_train, self.inner_channels[1],
                                                     self.extra_channel)
            with tf.variable_scope('transition_block'):
                stage_2 = self.transition_block(stage_2_input, is_train, is_pooling=True)

        with tf.variable_scope('stage_3'):
            stage_3_input = stage_2
            scope_name = 'dense_block_%d'
            for i in range(8):
                with tf.variable_scope(scope_name % i):
                    stage_3_input = self.dense_block(stage_3_input, is_train, self.inner_channels[2],
                                                     self.extra_channel)
            with tf.variable_scope('transition_block'):
                stage_3 = self.transition_block(stage_3_input, is_train, is_pooling=True)

        with tf.variable_scope('stage_4'):
            stage_4_input = stage_3
            scope_name = 'dense_block_%d'
            for i in range(6):
                with tf.variable_scope(scope_name % i):
                    stage_4_input = self.dense_block(stage_4_input, is_train, self.inner_channels[3],
                                                     self.extra_channel)
            with tf.variable_scope('transition_block'):
                stage_4 = self.transition_block(stage_4_input, is_train, is_pooling=False)

        return stage_3_input, stage_4


class PeleeNetClassify:
    def __init__(self, num_classes):
        self.backbone = PeleeNet()
        self.num_classes = num_classes

    def __call__(self, inputs, is_train):
        _, feature = self.backbone(inputs, is_train)
        feature_size = feature.get_shape().as_list()[1:3]
        pooling = tf.layers.AveragePooling2D(feature_size, 1)(feature)
        flatten = tf.layers.Flatten()(pooling)
        dense = tf.layers.Dense(self.num_classes)(flatten)
        return dense
